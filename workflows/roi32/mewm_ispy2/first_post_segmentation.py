from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import argparse
import csv
import fcntl
import gc
import importlib.metadata
import itertools
import json
import os
import re
import signal
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import yaml
from scipy import ndimage

from .first_post_data import timestamp, write_json

SCHEMA = "ispy2_first_post_mamamia_segmentation_v1"
REPO = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def configuration(path: Path) -> dict[str, Any]:
    config = _release_yaml(path.read_text())
    if config["schema"] != SCHEMA or config["orientation"] != "LPS":
        raise ValueError("Unexpected segmentation contract")
    if config["folds"] != list(range(5)) or config["mirror"] is not True:
        raise ValueError("This run requires all five folds and mirror inference")
    if config["tile_step_size"] != 0.5 or config["cpu_threads"] < 1:
        raise ValueError("Unexpected inference settings")
    return config


def execution_configuration(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["output_dir"]) / "execution_profile.json"
    if not path.exists():
        return config
    profile = read_json(path)
    gpus = profile.get("gpus")
    if (
        profile["schema"] != "first_post_mamamia_execution_v1"
        or gpus not in ([1], [1, 0])
        or profile["cpu_fallback"] is not False
    ):
        raise ValueError("Unexpected dedicated GPU execution profile")
    execution = {
        "gpus": gpus,
        "cpu_fallback": False,
        "max_workers": profile.get("max_workers", 1),
        "cases_per_worker": profile.get("cases_per_worker", 1),
        "poll_seconds": profile.get(
            "poll_seconds", config["queue"].get("poll_seconds", 10)
        ),
        "resource_poll_seconds": profile.get("resource_poll_seconds", 10),
        "admission_stagger_seconds": profile.get("admission_stagger_seconds", 15),
    }
    if (
        type(execution["max_workers"]) is not int
        or not 1 <= execution["max_workers"] <= len(gpus)
        or type(execution["cases_per_worker"]) is not int
        or not 1 <= execution["cases_per_worker"] <= 128
        or not 0 < execution["poll_seconds"] <= execution["resource_poll_seconds"]
        or execution["resource_poll_seconds"] < 10
        or execution["admission_stagger_seconds"] < 10
    ):
        raise ValueError("Invalid parallel worker limits")
    inference = profile.get("inference", {})
    if inference and (
        inference.get("tile_batch_size") not in (1, 2, 4, 8, 16)
        or inference.get("mirror_batch_size") not in (1, 2, 4, 8)
        or inference.get("gpu_accumulation") is not True
        or inference.get("prefetch_cases", 2) not in (1, 2)
    ):
        raise ValueError("Invalid GPU batching settings")
    return {
        **config,
        "queue": {**config["queue"], **execution},
        "inference_execution": inference,
    }


def native_image(array: np.ndarray, record: dict[str, Any]) -> sitk.Image:
    if list(array.shape) != record["shape_zyx"] or not np.isfinite(array).all():
        raise ValueError("Source array shape or intensity is invalid")
    if np.ptp(array) <= 0:
        raise ValueError("Source image has no intensity variation")
    iop = np.asarray(record["image_orientation_patient"], dtype=np.float64)
    origin = np.asarray(record["image_position_patient_first"], dtype=np.float64)
    spacing = np.asarray(
        [
            record["pixel_spacing_yx_mm"][1],
            record["pixel_spacing_yx_mm"][0],
            record["slice_spacing_mm"],
        ],
        dtype=np.float64,
    )
    if iop.shape != (6,) or origin.shape != (3,) or spacing.shape != (3,):
        raise ValueError("Incomplete image geometry")
    if not np.isfinite(np.r_[iop, origin, spacing]).all() or min(spacing) <= 0:
        raise ValueError("Invalid physical geometry")
    direction = np.column_stack((iop[:3], iop[3:], np.cross(iop[:3], iop[3:])))
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError("Non-orthonormal source direction")
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    image.SetSpacing(tuple(spacing))
    image.SetOrigin(tuple(origin))
    image.SetDirection(tuple(direction.ravel()))
    return image


def geometry(image: sitk.Image) -> dict[str, Any]:
    return {
        "shape_zyx": list(reversed(image.GetSize())),
        "spacing_xyz_mm": list(image.GetSpacing()),
        "origin_lps_mm": list(image.GetOrigin()),
        "direction_lps": list(image.GetDirection()),
        "orientation": sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
            image.GetDirection()
        ),
    }


def same_grid(left: sitk.Image, right: sitk.Image) -> bool:
    return left.GetSize() == right.GetSize() and all(
        np.allclose(a, b, atol=1e-4, rtol=1e-6)
        for a, b in (
            (left.GetSpacing(), right.GetSpacing()),
            (left.GetOrigin(), right.GetOrigin()),
            (left.GetDirection(), right.GetDirection()),
        )
    )


def write_image(path: Path, image: sitk.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name.removesuffix(".nii.gz") + ".tmp.nii.gz")
    sitk.WriteImage(image, str(temporary), True)
    reread = sitk.ReadImage(str(temporary))
    if not same_grid(image, reread) or not np.array_equal(
        sitk.GetArrayViewFromImage(image), sitk.GetArrayViewFromImage(reread)
    ):
        raise ValueError("NIfTI round-trip changed image values or geometry")
    temporary.replace(path)


def standard_input(record: dict[str, Any], output: Path) -> sitk.Image:
    source = Path(record["source_image"])
    stat = source.stat()
    if source.is_symlink() or (stat.st_size, stat.st_mtime_ns) != (
        record["size_bytes"],
        record["local_mtime_ns"],
    ):
        raise ValueError("Source changed since preparation")
    nii = nib.load(source)
    if not np.allclose(nii.affine, np.eye(4)):
        raise ValueError("Source no longer has the expected converted ZYX layout")
    native = native_image(np.asarray(nii.dataobj, dtype=np.float32), record)
    standard = sitk.DICOMOrient(native, "LPS")
    # Axis flips/permutations must preserve every native voxel and physical point.
    restored = sitk.DICOMOrient(standard, geometry(native)["orientation"])
    if not same_grid(native, restored) or not np.array_equal(
        sitk.GetArrayViewFromImage(native), sitk.GetArrayViewFromImage(restored)
    ):
        raise ValueError("Orientation round-trip failed")
    write_image(output, standard)
    return standard


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    contract = {"schema": SCHEMA, "configuration": config}
    contract_path = root / "run_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("Existing segmentation run has a different contract")
    manifest = read_json(Path(config["source_manifest"]))
    verification = read_json(Path(config["source_verification"]))
    for document in (manifest, verification):
        if document["registered"] is not False or document["phase_index"] != 1:
            raise ValueError("Segmentation requires unregistered first-post images")
    if verification["stage"] != "complete" or verification["content_differences"] != 0:
        raise ValueError("Source transfer has not been verified")
    source_root = Path(manifest["destination_root"]).resolve()
    if source_root != Path(verification["destination_root"]).resolve():
        raise ValueError("Verification refers to a different source")
    split = read_json(Path(config["baseline_bundle"]) / "split.json")
    if set(split["train"]) & set(split["val"]):
        raise ValueError("Baseline train and validation patients overlap")
    with (Path(config["baseline_bundle"]) / "visits.csv").open(newline="") as handle:
        folds = {
            (r["patient_id"], r["visit"]): r["fold"] for r in csv.DictReader(handle)
        }
    records = []
    for original in manifest["records"]:
        record = dict(original)
        case = record["patient_id"] + "_" + record["visit"] + "_aqc1"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", case) or record["visit"] not in (
            "T0",
            "T1",
            "T2",
            "T3",
        ):
            raise ValueError("Invalid case identifier")
        relative = Path(record["relative_path"])
        source = source_root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or source.is_symlink()
            or not source.resolve().is_relative_to(source_root)
            or not source.name.endswith("_dce_aqc_1.nii.gz")
        ):
            raise ValueError("Unexpected source image path")
        stat = source.stat()
        if stat.st_size != record["size_bytes"]:
            raise ValueError("Source size differs from transfer manifest")
        # Validate geometry without decoding a full volume during inventory.
        dummy_record = dict(record, shape_zyx=[2, 2, 2])
        dummy = native_image(
            np.arange(8, dtype=np.float32).reshape(2, 2, 2), dummy_record
        )
        record.update(
            case_id=case,
            source_image=str(source),
            local_mtime_ns=stat.st_mtime_ns,
            native_orientation=geometry(dummy)["orientation"],
            cohort_fold=folds.get(
                (record["patient_id"], record["visit"]), "outside_vqgan_cohort"
            ),
        )
        if record["native_orientation"] not in ("LPS", "RAS"):
            raise ValueError("New anatomical axis convention requires review")
        records.append(record)
    if len(records) != config["expected_visits"] or len(
        {r["case_id"] for r in records}
    ) != len(records):
        raise ValueError("Source case count differs or contains duplicates")
    groups = defaultdict(list)
    for record in records:
        if record["cohort_fold"] == "train":
            if record["patient_id"] not in split["train"]:
                raise ValueError("Visit and patient splits disagree")
            groups[(record["visit"], record["native_orientation"])].append(record)
    rng = np.random.default_rng(config["seed"])
    pilot = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r["case_id"])
        rng.shuffle(group)
        pilot.extend(group[: config["pilot_per_visit_orientation"]])
    # Interleave visits/orientations so the initial results cover each stratum.
    pilot.sort(key=lambda r: r["case_id"])
    pilot_ids = {r["case_id"] for r in pilot}
    buckets = defaultdict(list)
    for record in pilot:
        buckets[(record["visit"], record["native_orientation"])].append(record)
    pilot_order = [
        r
        for row in itertools.zip_longest(*[buckets[k] for k in sorted(buckets)])
        for r in row
        if r
    ]
    remaining = sorted(
        (r for r in records if r["case_id"] not in pilot_ids),
        key=lambda r: ({"train": 0, "val": 1}.get(r["cohort_fold"], 2), r["case_id"]),
    )
    ordered = [
        dict(r, pilot=r["case_id"] in pilot_ids) for r in pilot_order + remaining
    ]
    inventory = {
        "schema": SCHEMA,
        "records": ordered,
        "count": len(ordered),
        "pilot_count": len(pilot_ids),
        "source_root": str(source_root),
        "patient_count": len({r["patient_id"] for r in records}),
        "native_orientation_counts": dict(
            Counter(r["native_orientation"] for r in records)
        ),
        "cohort_counts": dict(Counter(r["cohort_fold"] for r in records)),
        "pretraining_overlap": "not_resolved_no_independent_evaluation_claim",
    }
    inventory_path = root / "inventory.json"
    if inventory_path.exists() and read_json(inventory_path) != inventory:
        raise ValueError("Source inventory changed; do not mix segmentation runs")
    write_json(contract_path, contract)
    write_json(inventory_path, inventory)
    for directory in ("inputs", "masks", "case_reports", "overlays", "logs"):
        (root / directory).mkdir(exist_ok=True)
    write_json(
        root / "preparation.json",
        {
            "status": "prepared",
            "updated_at_utc": timestamp(),
            **{k: v for k, v in inventory.items() if k != "records"},
            "normalization": "packaged_DefaultPreprocessor_single_image_ZScoreNormalization",
            "segmentation_spacing_xyz_mm": [1.0, 1.0, 1.0],
            "input_storage": "retain_pilot_inputs_stream_remaining_inputs",
            "registered": False,
        },
    )
    return inventory


def predictor(config: dict[str, Any], device: str):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    torch.set_num_threads(config["cpu_threads"])
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(config["cpu_threads"])
    model = Path(config["model_dir"])
    plans = read_json(model / "plans.json")
    dataset = read_json(model / "dataset.json")
    cfg = plans["configurations"]["3d_fullres"]
    if (
        plans["dataset_name"] != "Dataset105_full_image"
        or plans["image_reader_writer"] != "SimpleITKIO"
        or dataset["channel_names"] != {"0": "T1"}
        or dataset["labels"] != {"background": 0, "tumor": 1}
        or dataset["file_ending"] != ".nii.gz"
        or cfg["spacing"] != [1.0, 1.0, 1.0]
        or cfg["normalization_schemes"] != ["ZScoreNormalization"]
        or cfg["use_mask_for_norm"] != [False]
        or cfg["patch_size"] != [128, 128, 128]
    ):
        raise ValueError("Packaged model input contract differs")
    manager = PlansManager(plans)
    cm = manager.get_configuration("3d_fullres")
    network = get_network_from_plans(
        cm.network_arch_class_name,
        {**cm.network_arch_init_kwargs, "deep_supervision": False},
        cm.network_arch_init_kwargs_req_import,
        1,
        2,
        allow_init=False,
    )
    parameters, fold_info = [], []
    for fold in config["folds"]:
        path = model / f"fold_{fold}" / "checkpoint_final.pth"
        # These are the user's official downloaded checkpoints, including metadata.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint["trainer_name"] != "nnUNetTrainer_4000epochs"
            or checkpoint["init_args"]["configuration"] != "3d_fullres"
            or tuple(checkpoint["inference_allowed_mirroring_axes"]) != (0, 1, 2)
        ):
            raise ValueError("Unsupported checkpoint trainer/configuration")
        weights = checkpoint["network_weights"]
        network.load_state_dict(weights, strict=True)
        if not all(torch.isfinite(value).all() for value in weights.values()):
            raise ValueError("Checkpoint contains non-finite weights")
        parameters.append(weights)
        fold_info.append(
            {
                "fold": fold,
                "network_entries": len(weights),
                "epoch": checkpoint["current_epoch"],
                "size_bytes": path.stat().st_size,
            }
        )
        del checkpoint
    acceleration = config.get("inference_execution", {}) if device != "cpu" else {}
    predictor_class, extra = nnUNetPredictor, {}
    if acceleration:
        from .first_post_segmentation_acceleration import BatchedNNUNetPredictor

        predictor_class = BatchedNNUNetPredictor
        extra = {
            "tile_batch_size": acceleration["tile_batch_size"],
            "mirror_batch_size": acceleration["mirror_batch_size"],
        }
    instance = predictor_class(
        tile_step_size=config["tile_step_size"],
        use_gaussian=True,
        use_mirroring=config["mirror"],
        perform_everything_on_device=acceleration.get("gpu_accumulation", False),
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
        **extra,
    )
    instance.manual_initialization(
        network, manager, cm, parameters, dataset, "nnUNetTrainer_4000epochs", (0, 1, 2)
    )
    info = {
        "folds": fold_info,
        "trainer": instance.trainer_name,
        "source": "https://github.com/LidiaGarrucho/nnUNet",
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "nnunetv2",
                "dynamic-network-architectures",
                "numpy",
                "SimpleITK",
            )
        },
        "mirror_axes": [0, 1, 2],
        "tile_step_size": config["tile_step_size"],
        "orientation_policy": "LPS_per_visit_axis_flips_no_spatial_interpolation",
    }
    return instance, info


def audit_model(config: dict[str, Any]) -> None:
    instance, info = predictor(config, "cpu")
    instance.network.eval()
    torch.manual_seed(config["seed"])
    with torch.inference_mode():
        result = instance.network(torch.randn(1, 1, 64, 64, 64))
    if result.shape != (1, 2, 64, 64, 64) or not torch.isfinite(result).all():
        raise ValueError("Model forward check failed")
    write_json(
        Path(config["output_dir"]) / "model_validation.json",
        {
            "status": "passed",
            "updated_at_utc": timestamp(),
            **info,
            "cpu_forward_shape": list(result.shape),
            "all_folds_strictly_loaded": True,
            "scope": "runtime_check_not_segmentation_accuracy",
        },
    )


def mask_qc(mask: sitk.Image, reference: sitk.Image) -> dict[str, Any]:
    if not same_grid(mask, reference):
        raise ValueError("Mask and input occupy different physical grids")
    array = sitk.GetArrayFromImage(mask)
    if not np.isin(array, [0, 1]).all():
        raise ValueError("Prediction has unexpected labels")
    count = int(np.count_nonzero(array))
    voxel_mm3 = float(np.prod(mask.GetSpacing()))
    flags = []
    qc: dict[str, Any] = {
        "labels": [int(x) for x in np.unique(array)],
        "tumor_voxels": count,
        "tumor_volume_mm3": count * voxel_mm3,
        "geometry": geometry(mask),
        "review_status": "pending_visual_review",
        "flags": flags,
        "bbox_zyx_exclusive": None,
        "bbox_extent_xyz_mm": None,
        "bbox_center_lps_mm": None,
    }
    if not count:
        flags.append("empty_mask_not_proof_of_complete_response")
        return qc
    positions = np.argwhere(array != 0)
    low, high = positions.min(axis=0), positions.max(axis=0) + 1
    qc["bbox_zyx_exclusive"] = np.column_stack((low, high)).tolist()
    qc["bbox_extent_xyz_mm"] = (
        (high - low)[::-1] * np.asarray(mask.GetSpacing())
    ).tolist()
    center_xyz = ((low + high - 1) / 2)[::-1]
    qc["bbox_center_lps_mm"] = list(
        mask.TransformContinuousIndexToPhysicalPoint(tuple(center_xyz))
    )
    _, components = ndimage.label(array, structure=np.ones((3, 3, 3)))
    qc["connected_components_26"] = components
    if components > 1:
        flags.append("multiple_components_review_required")
    if np.any(low == 0) or np.any(high == array.shape):
        flags.append("foreground_touches_image_border")
    if count / array.size > 0.2:
        flags.append("large_foreground_fraction")
    return qc


def overlay(
    path: Path, image: sitk.Image, mask: sitk.Image, qc: dict[str, Any]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = sitk.GetArrayFromImage(image)
    labels = sitk.GetArrayFromImage(mask)
    if np.any(labels):
        center = [
            int(
                np.argmax(
                    np.count_nonzero(
                        labels, axis=tuple(i for i in range(3) if i != axis)
                    )
                )
            )
            for axis in range(3)
        ]
    else:
        center = [n // 2 for n in data.shape]
    planes = (
        (
            data[center[0]],
            labels[center[0]],
            image.GetSpacing()[1] / image.GetSpacing()[0],
            "Axial",
        ),
        (
            data[:, center[1]],
            labels[:, center[1]],
            image.GetSpacing()[2] / image.GetSpacing()[0],
            "Coronal",
        ),
        (
            data[:, :, center[2]],
            labels[:, :, center[2]],
            image.GetSpacing()[2] / image.GetSpacing()[1],
            "Sagittal",
        ),
    )
    nonzero = data[data != 0]
    low, high = np.percentile(nonzero, [1, 99.7])
    fig, axes = plt.subplots(2, 3, figsize=(10, 7), layout="constrained")
    for column, (plane, seg, aspect, title) in enumerate(planes):
        for row in range(2):
            axes[row, column].imshow(
                plane, cmap="gray", vmin=low, vmax=high, aspect=aspect, origin="lower"
            )
            axes[row, column].axis("off")
        axes[0, column].set_title(title)
        if np.any(seg):
            axes[1, column].contour(
                seg, levels=[0.5], colors=["#ff4b64"], linewidths=0.7
            )
    fig.suptitle(
        path.stem
        + " | predicted mask | "
        + (", ".join(qc["flags"]) or "visual review pending"),
        fontsize=9,
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)


def finished(root: Path, record: dict[str, Any]) -> bool:
    report_path = root / "case_reports" / (record["case_id"] + ".json")
    if not report_path.exists():
        return False
    report = read_json(report_path)
    if (
        report.get("status") != "completed"
        or report.get("source_mtime_ns") != record["local_mtime_ns"]
    ):
        raise ValueError("Existing case report has incompatible provenance")
    mask = root / "masks" / (record["case_id"] + ".nii.gz")
    if not mask.exists() or mask.stat().st_size != report["mask_size_bytes"]:
        raise ValueError("Completed segmentation is missing or changed")
    if not (root / "overlays" / (record["case_id"] + ".png")).is_file():
        raise ValueError("Completed segmentation overlay is missing")
    return True


@torch.inference_mode()
def infer_logits(instance: Any, data: np.ndarray) -> torch.Tensor:
    # The author's older predictor updates inference tensors while ensembling folds.
    return instance.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()


def prepare_case_input(config, record, instance, stage=None):
    root, case = Path(config["output_dir"]), record["case_id"]
    started = time.monotonic()
    if stage:
        stage("preparing_input")
    input_path = root / "inputs" / (case + "_0000.nii.gz")
    image = standard_input(record, input_path)
    prepared_at = time.monotonic()
    if stage:
        stage("nnunet_preprocessing")
    preprocessor = instance.configuration_manager.preprocessor_class(verbose=False)
    data, _, properties = preprocessor.run_case(
        [str(input_path)],
        None,
        instance.plans_manager,
        instance.configuration_manager,
        instance.dataset_json,
    )
    if not np.isfinite(data).all():
        raise ValueError("Preprocessing generated non-finite voxels")
    return {
        "image": image,
        "input_path": input_path,
        "data": data,
        "properties": properties,
        "preprocessed_shape": list(data.shape),
        "started": started,
        "stage_seconds": {
            "preparing_input": prepared_at - started,
            "nnunet_preprocessing": time.monotonic() - prepared_at,
        },
    }


def export_case(
    config,
    record,
    device,
    instance,
    prepared,
    logits,
    *,
    reused_model=True,
    inference_stats=None,
    stage=None,
):
    from nnunetv2.inference.export_prediction import (
        convert_predicted_logits_to_segmentation_with_correct_shape,
    )

    root, case = Path(config["output_dir"]), record["case_id"]
    image, timings = prepared["image"], prepared["stage_seconds"]
    exporting_at = time.monotonic()
    if not torch.isfinite(logits).all():
        raise ValueError("Prediction contains non-finite logits")
    if stage:
        stage("exporting_mask")
    labels = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits,
        instance.plans_manager,
        instance.configuration_manager,
        instance.label_manager,
        prepared["properties"],
        num_threads_torch=config["cpu_threads"],
    )
    mask = sitk.GetImageFromArray(np.asarray(labels, dtype=np.uint8))
    mask.CopyInformation(image)
    qc = mask_qc(mask, image)
    mask_path = root / "masks" / (case + ".nii.gz")
    write_image(mask_path, mask)
    reread = sitk.ReadImage(str(mask_path))
    mask_qc(reread, image)
    rendering_at = time.monotonic()
    timings["exporting_mask"] = rendering_at - exporting_at
    if stage:
        stage("rendering_overlay")
    overlay(root / "overlays" / (case + ".png"), image, mask, qc)
    timings["rendering_overlay"] = time.monotonic() - rendering_at
    if stage:
        stage("committing_output")
    write_json(
        root / "case_reports" / (case + ".json"),
        {
            "schema": SCHEMA,
            "status": "completed",
            "case_id": case,
            "updated_at_utc": timestamp(),
            "elapsed_seconds": time.monotonic() - prepared["started"],
            "stage_seconds": timings,
            "reused_model": reused_model,
            "inference_execution": config.get("inference_execution", {}),
            "inference_stats": inference_stats or {},
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")
            if device != "cpu"
            else None,
            "source_image": record["source_image"],
            "source_mtime_ns": record["local_mtime_ns"],
            "mask_size_bytes": mask_path.stat().st_size,
            "device": device,
            "cohort_fold": record["cohort_fold"],
            "visit": record["visit"],
            "pilot": record["pilot"],
            "folds": config["folds"],
            "mirror": config["mirror"],
            "preprocessed_shape_czyx": prepared["preprocessed_shape"],
            "qc": qc,
            "input_retained": record["pilot"],
            "registered": False,
        },
    )
    if not record["pilot"]:
        prepared["input_path"].unlink()
    if stage:
        stage("completed")
    print(
        json.dumps({"case_id": case, "status": "completed", "qc_flags": qc["flags"]}),
        flush=True,
    )
    gc.collect()


def run_case(
    config: dict[str, Any],
    record: dict[str, Any],
    device: str,
    *,
    instance: Any = None,
    status_path: Path | None = None,
) -> None:
    root, case = Path(config["output_dir"]), record["case_id"]
    if finished(root, record):
        return
    started = time.monotonic()
    reused_model = instance is not None

    def stage(name, **extra):
        write_json(
            status_path or root / "worker_status.json",
            {
                "status": name,
                "case_id": case,
                "device": device,
                "pid": os.getpid(),
                "updated_at_utc": timestamp(),
                **extra,
            },
        )

    if instance is None:
        stage("loading_model")
        instance, _ = predictor(config, device)
    loading_seconds = time.monotonic() - started
    prepared = prepare_case_input(config, record, instance, stage)
    prepared["started"] = started
    if not reused_model:
        prepared["stage_seconds"]["loading_model"] = loading_seconds
    stage(
        "segmenting",
        preprocessed_shape=prepared["preprocessed_shape"],
        folds=config["folds"],
    )
    inference_at = time.monotonic()
    logits = infer_logits(instance, prepared.pop("data"))
    prepared["stage_seconds"]["segmenting"] = time.monotonic() - inference_at
    export_case(
        config,
        record,
        device,
        instance,
        prepared,
        logits,
        reused_model=reused_model,
        stage=stage,
    )


def select_gpu(config: dict[str, Any]):
    from .first_post_vqgan import gpu_idle, gpu_reserved

    lock_root = Path(config["queue"]["lock_root"])
    lock_root.mkdir(parents=True, exist_ok=True)
    for gpu in config["queue"]["gpus"]:
        if gpu_reserved(gpu, config):
            continue
        handle = (lock_root / f"gpu{gpu}.lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if gpu_idle(gpu) and not gpu_reserved(gpu, config):
                return gpu, handle
        except BlockingIOError:
            pass
        handle.close()
    return None, None


def next_case(pending: list[dict[str, Any]], cpu: bool) -> dict[str, Any]:
    if not cpu:
        return pending[0]

    def priority(record: dict[str, Any]):
        stage = (
            0
            if record["pilot"]
            else {"train": 1, "val": 2}.get(record["cohort_fold"], 3)
        )
        physical_volume = (
            np.prod(record["shape_zyx"])
            * np.prod(record["pixel_spacing_yx_mm"])
            * record["slice_spacing_mm"]
        )
        return stage, physical_volume, record["case_id"]

    return min(pending, key=priority)


def sustained_runtime_reasons(
    reasons: list[str], pressure_samples: int
) -> tuple[list[str], int]:
    pressure_samples = pressure_samples + 1 if "host_memory_pressure" in reasons else 0
    immediate = [reason for reason in reasons if reason != "host_memory_pressure"]
    if pressure_samples >= 6:
        immediate.append("sustained_host_memory_pressure")
    return immediate, pressure_samples


def run_queue(config: dict[str, Any], config_path: Path) -> None:
    execution = execution_configuration(config)
    if (
        execution["queue"].get("max_workers", 1) > 1
        or execution["queue"].get("cases_per_worker", 1) > 1
    ):
        from .first_post_segmentation_parallel import run_parallel_queue

        run_parallel_queue(config, execution, config_path)
        return
    from .first_post_vqgan import resource_reasons, resources

    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    stop = False
    child = None
    gpu_lock = None

    def request_stop(*_: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = prepare(config)
        records = inventory["records"]
        execution = execution_configuration(config)

        def status(name: str, **extra: Any) -> None:
            write_json(
                root / "queue_status.json",
                {
                    "status": name,
                    "pid": os.getpid(),
                    "updated_at_utc": timestamp(),
                    "total_cases": len(records),
                    **extra,
                },
            )

        try:
            while not stop:
                pending = [r for r in records if not finished(root, r)]
                complete = len(records) - len(pending)
                if not pending:
                    status("completed", completed_cases=complete)
                    return
                snapshot = resources(root)
                reasons = resource_reasons(snapshot, config, runtime=False)
                if reasons:
                    status(
                        "waiting_for_resources",
                        completed_cases=complete,
                        reasons=reasons,
                        resources=snapshot,
                    )
                    time.sleep(config["queue"]["poll_seconds"])
                    continue
                gpu, gpu_lock = select_gpu(execution)
                if gpu is None and not execution["queue"]["cpu_fallback"]:
                    status("waiting_for_gpu", completed_cases=complete)
                    time.sleep(config["queue"]["poll_seconds"])
                    continue
                record = next_case(pending, cpu=gpu is None)
                device = "cpu" if gpu is None else "cuda:0"
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
                with (root / "logs" / (record["case_id"] + ".log")).open("a") as log:
                    child = subprocess.Popen(
                        [
                            config["python"],
                            "-u",
                            "-m",
                            "mewm_ispy2.first_post_segmentation",
                            "worker",
                            "--config",
                            str(config_path),
                            "--case",
                            record["case_id"],
                            "--device",
                            device,
                        ],
                        cwd=REPO,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    interrupted = False
                    pressure_samples = 0
                    while child.poll() is None:
                        snapshot = resources(root)
                        reasons, pressure_samples = sustained_runtime_reasons(
                            resource_reasons(snapshot, config, runtime=True),
                            pressure_samples,
                        )
                        status(
                            "segmenting",
                            completed_cases=complete,
                            case_id=record["case_id"],
                            worker_pid=child.pid,
                            device=device,
                            physical_gpu=gpu,
                            resources=snapshot,
                            pilot=record["pilot"],
                            pressure_samples=pressure_samples,
                        )
                        if stop or reasons:
                            interrupted = True
                            child.terminate()
                            try:
                                child.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                            status(
                                "paused" if stop else "waiting_for_resources",
                                completed_cases=complete,
                                reasons=reasons or ["requested_stop"],
                            )
                            with (root / "resource_events.jsonl").open("a") as events:
                                events.write(
                                    json.dumps(
                                        {
                                            "updated_at_utc": timestamp(),
                                            "case_id": record["case_id"],
                                            "reasons": reasons or ["requested_stop"],
                                            "resources": snapshot,
                                        }
                                    )
                                    + "\n"
                                )
                            break
                        time.sleep(config["queue"]["poll_seconds"])
                if gpu_lock is not None:
                    gpu_lock.close()
                    gpu_lock = None
                if not interrupted and (
                    child.returncode != 0 or not finished(root, record)
                ):
                    raise RuntimeError(
                        f"Segmentation worker failed for {record['case_id']}; inspect case log"
                    )
                child = None
            status("paused", reasons=["requested_stop"])
        except BaseException as error:
            status("failed", error=str(error))
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if gpu_lock is not None:
                gpu_lock.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "validate-model", "worker", "worker-batch", "queue"),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--status-file", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = configuration(config_path)
    if args.stage == "prepare":
        result = prepare(config)
        print(json.dumps({k: v for k, v in result.items() if k != "records"}))
    elif args.stage == "validate-model":
        audit_model(config)
    elif args.stage == "queue":
        run_queue(config, config_path)
    elif args.stage == "worker-batch":
        from .first_post_segmentation_parallel import run_batch

        if args.batch is None or args.status_file is None:
            parser.error("worker-batch requires --batch and --status-file")
        run_batch(config, args.batch, args.device, args.status_file)
    else:
        inventory = read_json(Path(config["output_dir"]) / "inventory.json")
        record = next(r for r in inventory["records"] if r["case_id"] == args.case)
        run_case(config, record, args.device)


if __name__ == "__main__":
    main()
