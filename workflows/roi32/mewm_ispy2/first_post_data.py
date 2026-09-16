from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import csv
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import yaml

from .preprocessing import _crop_or_pad

NUMERIC_CONTRACT = "ispy2_first_post_unregistered_train_zscore_v1"
ALL_VISITS_POLICY = "all_visits_preserve_patient_split_v1"


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_config(path: str | Path) -> dict[str, Any]:
    config = _release_yaml(Path(path).read_text())
    if config["schema"] != "ispy2_first_post_unregistered_training_v1":
        raise ValueError("Unsupported first-post training configuration")
    data = config["data"]
    if data["registered"] is not False or data["phase_index"] != 1:
        raise ValueError("This workflow requires unregistered aqc_1 images")
    if data["normalization"] != "train_resampled_nonzero_global_zscore":
        raise ValueError("Unsupported normalization")
    if data["orientation"] != "LPS":
        raise ValueError("The image orientation must be LPS")
    crops = (data["training_crop"], data["validation_crop"])
    if crops not in (
        ("random_spatial_patch", "center"),
        ("tumor_union_adaptive", "tumor_union_adaptive"),
    ):
        raise ValueError("Unsupported crop policy")
    if crops[0] == "tumor_union_adaptive":
        crop = data["tumor_crop"]
        if not np.isfinite(crop["margin_mm"]) or crop["margin_mm"] < 0:
            raise ValueError("Invalid tumor crop margin")
        if (
            crop["empty_mask"] not in ("full_image_fov", "t0_mask")
            or crop["components"] != "keep_all"
        ):
            raise ValueError("Unsupported empty-mask or multifocal crop policy")
        if crop["empty_mask"] == "t0_mask":
            if crop.get("reference_mapping") not in (
                "physical_lps",
                "transform_manifest",
            ):
                raise ValueError(
                    "T0 fallback requires an explicit physical mapping policy"
                )
            if crop["reference_mapping"] == "transform_manifest" and not crop.get(
                "reference_transform_manifest"
            ):
                raise ValueError("T0 fallback requires a transform manifest")
    if len(data["output_shape_zyx"]) != 3 or any(
        n < 32 or n % 4 for n in data["output_shape_zyx"]
    ):
        raise ValueError("Patch dimensions must be at least 32 and divisible by four")
    spacing = data["target_spacing_xyz"]
    if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Invalid target spacing")
    return config


def select_records(config: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    manifest = json.loads(Path(config["source_manifest"]).read_text())
    verified = json.loads(Path(config["source_verification"]).read_text())
    for document in (manifest, verified):
        if document["registered"] is not False or document["phase_index"] != 1:
            raise ValueError("Source is not the requested unregistered phase")
    if verified["stage"] != "complete" or verified["content_differences"] != 0:
        raise ValueError("Source transfer has not passed verification")
    root = Path(manifest["destination_root"]).resolve()
    if root != Path(verified["destination_root"]).resolve():
        raise ValueError("Source verification refers to a different directory")
    lookup = {(r["patient_id"], r["visit"]): r for r in manifest["records"]}
    if len(lookup) != len(manifest["records"]):
        raise ValueError("Duplicate source visits")
    split = patient_split(config, manifest["records"])
    if config.get("cohort"):
        fold_by_patient = {
            patient: fold for fold, patients in split.items() for patient in patients
        }
        visits = [
            {**record, "fold": fold_by_patient[record["patient_id"]]}
            for record in manifest["records"]
        ]
    else:
        with (Path(config["baseline_bundle"]) / "visits.csv").open(
            newline=""
        ) as handle:
            visits = list(csv.DictReader(handle))
    selected = []
    for visit in visits:
        fold = visit["fold"]
        if fold not in ("train", "val") or visit["patient_id"] not in split[fold]:
            raise ValueError("Visit and patient splits disagree")
        record = dict(lookup[(visit["patient_id"], visit["visit"])])
        relative = Path(record["relative_path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.name.endswith("_dce_aqc_1.nii.gz")
        ):
            raise ValueError("Unexpected first-post image path")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Image path leaves the verified dataset")
        stat = path.stat()
        if stat.st_size != record["size_bytes"]:
            raise ValueError(f"Image size changed: {relative}")
        record.update(fold=fold, local_mtime_ns=stat.st_mtime_ns)
        if config.get("cohort", {}).get("require_phase_metadata"):
            record["phase_evidence"] = verify_phase_metadata(record)
        selected.append(record)
    counts = Counter(r["fold"] for r in selected)
    for fold in ("train", "val"):
        expected = config["data"].get(f"expected_{fold}_visits")
        if expected is not None and counts[fold] != expected:
            raise ValueError(f"Unexpected {fold} cohort size: {counts[fold]}")
    if len({(r["patient_id"], r["visit"]) for r in selected}) != len(selected):
        raise ValueError("Duplicate baseline visits")
    return root, sorted(
        selected, key=lambda r: (r["fold"], r["patient_id"], r["visit"])
    )


def patient_split(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, list[str]]:
    previous = json.loads((Path(config["baseline_bundle"]) / "split.json").read_text())
    train, val = set(previous["train"]), set(previous["val"])
    if train & val:
        raise ValueError("Patient leakage between training and validation")
    if len(train) != len(previous["train"]) or len(val) != len(previous["val"]):
        raise ValueError("Duplicate patients in the previous split")
    cohort = config.get("cohort")
    if cohort:
        if cohort["policy"] != ALL_VISITS_POLICY:
            raise ValueError("Unsupported first-post cohort policy")
        patients = {r["patient_id"] for r in records}
        if (
            len(records) != cohort["expected_visits"]
            or len(patients) != cohort["expected_patients"]
        ):
            raise ValueError("Full first-post cohort size changed")
        if not (train | val) <= patients:
            raise ValueError("Previous split patients are missing from the source")
        fraction = float(cohort["validation_fraction"])
        if not 0 < fraction < 1:
            raise ValueError("Validation fraction must be between zero and one")
        validation_count = round(len(patients) * fraction)
        candidates = sorted(patients - train - val)
        additional = validation_count - len(val)
        if not 0 < validation_count < len(patients) or not 0 <= additional <= len(
            candidates
        ):
            raise ValueError("Requested ratio conflicts with preserved patient splits")
        # Keep patients seen by the DCE0 initializer out of the new validation set.
        np.random.default_rng(config["seed"]).shuffle(candidates)
        val.update(candidates[:additional])
        train = patients - val
    return {"train": sorted(train), "val": sorted(val)}


def verify_phase_metadata(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["metadata_source"])
    metadata = json.loads(path.read_text())
    phases = metadata["dce_phase_ids"]
    paths = metadata["dce_paths"]
    if (
        (metadata["patient_id"], metadata["visit"])
        != (record["patient_id"], record["visit"])
        or len(phases) < 2
        or phases != list(range(len(phases)))
        or len(phases) != len(paths)
        or len(phases) != metadata["n_times"]
        or not Path(paths[0]).name.endswith("_dce_aqc_0.nii.gz")
        or Path(paths[1]).name != Path(record["relative_path"]).name
        or [metadata["n_slices"], metadata["rows"], metadata["cols"]]
        != record["shape_zyx"]
    ):
        raise ValueError("First-post phase metadata does not match the source image")
    stat = path.stat()
    return {
        "phase_index": 1,
        "phase_count": len(phases),
        "evidence": "ordered_zero_based_dce_conversion_metadata",
        "metadata_size_bytes": stat.st_size,
        "metadata_mtime_ns": stat.st_mtime_ns,
        "conversion_qc_status": metadata.get("qc_status", "unspecified"),
    }


def cohort_selection_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        **{
            key: config[key]
            for key in ("source_manifest", "source_verification", "baseline_bundle")
        },
        "cohort": config.get("cohort"),
        "seed": config.get("seed") if config.get("cohort") else None,
    }


def write_cohort(
    config: dict[str, Any], root: Path, records: list[dict[str, Any]]
) -> dict[str, Any]:
    output = Path(config["output_dir"]) / "data"
    split = {
        fold: sorted({r["patient_id"] for r in records if r["fold"] == fold})
        for fold in ("train", "val")
    }
    summary = {
        "schema": "ispy2_first_post_cohort_v1",
        "stage": "cohort_selected",
        "registered": False,
        "phase_index": 1,
        "selection_config": cohort_selection_config(config),
        "visit_counts": dict(Counter(r["fold"] for r in records)),
        "patient_counts": {fold: len(patients) for fold, patients in split.items()},
        "total_visits": len(records),
        "patient_overlap": len(set(split["train"]) & set(split["val"])),
        "split_unit": "patient_all_visits_together",
        "split_file": str(output / "split.json"),
        "cohort_file": str(output / "cohort.json"),
    }
    if config.get("cohort", {}).get("require_phase_metadata"):
        summary["phase_verification"] = {
            "verified_visits": len(records),
            "phase_index": 1,
            "phase_count_distribution": dict(
                Counter(str(r["phase_evidence"]["phase_count"]) for r in records)
            ),
            "conversion_qc_counts": dict(
                Counter(r["phase_evidence"]["conversion_qc_status"] for r in records)
            ),
            "evidence": "ordered_zero_based_dce_conversion_metadata",
        }
    documents = {
        "cohort.json": {
            "schema": summary["schema"],
            "selection_config": summary["selection_config"],
            "source_root": str(root),
            "records": records,
        },
        "split.json": split,
        "cohort_summary.json": summary,
    }
    for name, document in documents.items():
        path = output / name
        if path.exists() and json.loads(path.read_text()) != document:
            raise ValueError(
                "Saved first-post cohort changed; use a new output directory"
            )
    for name, document in documents.items():
        if not (output / name).exists():
            write_json(output / name, document)
    return summary


def prepare_cohort(config: dict[str, Any]) -> dict[str, Any]:
    root, records = select_records(config)
    return write_cohort(config, root, records)


def resample_native(
    array: np.ndarray, record: dict[str, Any], spacing_xyz: list[float]
) -> tuple[np.ndarray, dict[str, Any]]:
    direction = np.asarray(record["image_orientation_patient"], dtype=np.float64)
    origin = np.asarray(record["image_position_patient_first"], dtype=np.float64)
    spacing = np.asarray(
        [
            record["pixel_spacing_yx_mm"][1],
            record["pixel_spacing_yx_mm"][0],
            record["slice_spacing_mm"],
        ]
    )
    if (
        direction.shape != (6,)
        or origin.shape != (3,)
        or not np.isfinite(np.r_[direction, origin, spacing]).all()
        or min(spacing) <= 0
    ):
        raise ValueError("Incomplete native physical geometry")
    matrix = np.column_stack(
        (direction[:3], direction[3:], np.cross(direction[:3], direction[3:]))
    )
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4):
        raise ValueError("Non-orthonormal DICOM orientation")
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(tuple(float(v) for v in spacing))
    image.SetDirection(tuple(matrix.ravel()))
    image.SetOrigin(tuple(origin))
    # DICOMOrient permutes/flips axes while retaining each visit's physical frame.
    image = sitk.DICOMOrient(image, "LPS")
    size = [
        max(1, round((n - 1) * s / t) + 1)
        for n, s, t in zip(
            image.GetSize(), image.GetSpacing(), spacing_xyz, strict=True
        )
    ]
    result = sitk.Resample(
        image,
        size,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        image.GetOrigin(),
        spacing_xyz,
        image.GetDirection(),
        0.0,
        sitk.sitkFloat32,
    )
    return sitk.GetArrayFromImage(result), {
        "shape_zyx": list(reversed(result.GetSize())),
        "spacing_xyz": list(result.GetSpacing()),
        "direction_lps": list(result.GetDirection()),
        "origin_lps": list(result.GetOrigin()),
        "coordinate_frame": "per_visit_native_no_intervisit_registration",
    }


def load_volume(
    root: Path, record: dict[str, Any], config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    path = root / record["relative_path"]
    stat = path.stat()
    if (
        path.is_symlink()
        or stat.st_size != record["size_bytes"]
        or stat.st_mtime_ns != record["local_mtime_ns"]
    ):
        raise ValueError("Source image changed after cohort selection")
    image = nib.load(path)
    if not np.allclose(image.affine, np.eye(4)):
        raise ValueError("Unexpected affine in the converted ZYX source")
    array = np.asarray(image.dataobj, dtype=np.float32)
    if list(array.shape) != record["shape_zyx"] or not np.isfinite(array).all():
        raise ValueError("Image shape mismatch or non-finite source voxels")
    if not np.any(array != 0):
        raise ValueError("Empty source image")
    return resample_native(array, record, config["data"]["target_spacing_xyz"])


def nonzero_moments(array: np.ndarray) -> dict[str, float | int]:
    if not np.isfinite(array).all():
        raise ValueError("Non-finite resampled image")
    values = array[array != 0].astype(np.float64)
    if not len(values) or float(values.max()) <= float(values.min()):
        raise ValueError("Image has no usable intensity range")
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "m2": float(values.var() * len(values)),
    }


def combine_moments(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    count, mean, m2 = 0, 0.0, 0.0
    for row in rows:
        n, value, variance_sum = row["count"], row["mean"], row["m2"]
        delta = value - mean
        total = count + n
        mean += delta * n / total
        m2 += variance_sum + delta * delta * count * n / total
        count = total
    if not count or m2 <= 0:
        raise ValueError("Training intensities have no variance")
    return {"count": count, "mean": mean, "std": float(np.sqrt(m2 / count))}


def image_patch(
    array: np.ndarray, shape: list[int], *, random_crop: bool
) -> np.ndarray:
    starts = []
    for available, requested in zip(array.shape, shape, strict=True):
        spare = available - requested
        starts.append(
            int(torch.randint(spare + 1, ()).item())
            if random_crop and spare > 0
            else spare // 2
        )
    patch = _crop_or_pad(array, starts, shape, fill_value=0)
    if not np.any(patch != 0):
        raise ValueError("Selected image patch is empty")
    return patch


def preparation_signature(config: dict[str, Any]) -> dict[str, Any]:
    signature = {
        k: config[k]
        for k in (
            "schema",
            "source_manifest",
            "source_verification",
            "baseline_bundle",
            "data",
        )
    }
    if config.get("cohort"):
        signature.update(cohort=config["cohort"], seed=config["seed"])
    return signature


def source_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v for k, v in record.items() if k not in ("resampled_geometry", "moments")
    }


def scan_record(
    root: Path, record: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    array, geometry = load_volume(root, record, config)
    moments = nonzero_moments(array)
    image_patch(array, config["data"]["output_shape_zyx"], random_crop=False)
    return {**record, "resampled_geometry": geometry, "moments": moments}


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    if config["data"].get("training_crop") == "tumor_union_adaptive":
        from .first_post_tumor_crops import prepare_tumor_crops

        return prepare_tumor_crops(config)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    root, records = select_records(config)
    if config.get("cohort"):
        write_cohort(config, root, records)
    output = Path(config["output_dir"]) / "data"
    output.mkdir(parents=True, exist_ok=True)
    signature = preparation_signature(config)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        original = [source_record(r) for r in existing["records"]]
        if existing["preparation_config"] != signature or original != records:
            raise ValueError("Existing prepared data belongs to different inputs")
        if (output / "normalization.json").exists() and (
            output / "summary.json"
        ).exists():
            return json.loads((output / "summary.json").read_text())
    completed = 0
    scan_path = output / "scan_state.json"
    if scan_path.exists():
        previous = json.loads(scan_path.read_text())
        completed = len(previous["records"])
        if (
            previous["preparation_config"] != signature
            or [source_record(r) for r in previous["records"]] != records[:completed]
        ):
            raise ValueError("Preparation recovery inputs changed")
        records[:completed] = previous["records"]
    workers = int(config["data"].get("preparation_workers", 1))
    if not 1 <= workers <= 4:
        raise ValueError("Preparation supports one to four workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scans = pool.map(
            lambda record: scan_record(root, record, config), records[completed:]
        )
        for index, record in enumerate(scans, start=completed):
            records[index] = record
            if index % 25 == 0 or index + 1 == len(records):
                progress = {
                    "stage": "preparing",
                    "pid": os.getpid(),
                    "completed_visits": index + 1,
                    "total_visits": len(records),
                    "workers": workers,
                    "updated_at_utc": timestamp(),
                }
                write_json(
                    scan_path,
                    {"preparation_config": signature, "records": records[: index + 1]},
                )
                write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    training = [r for r in records if r["fold"] == "train"]
    normalization = {
        "schema": NUMERIC_CONTRACT,
        **combine_moments([r["moments"] for r in training]),
        "fit_fold": "train",
        "fit_visits": len(training),
        "scope": "whole_resampled_finite_nonzero_volumes",
        "background_and_padding": 0.0,
        "image_clipping": False,
    }
    if config.get("cohort"):
        normalization.update(
            preparation_config=signature,
            fit_patients=len({r["patient_id"] for r in training}),
        )
    summary = {
        "stage": "prepared",
        "completed_at_utc": timestamp(),
        "registered": False,
        "phase_index": 1,
        "visit_counts": dict(Counter(r["fold"] for r in records)),
        "patient_counts": {
            fold: len({r["patient_id"] for r in records if r["fold"] == fold})
            for fold in ("train", "val")
        },
        "fully_decoded_visits": len(records),
        "source_root": str(root),
        "normalization": normalization,
        "preparation_config": signature,
    }
    write_json(
        manifest_path,
        {"preparation_config": signature, "source_root": str(root), "records": records},
    )
    write_json(output / "normalization.json", normalization)
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", summary)
    return summary


class FirstPostDataset(torch.utils.data.Dataset):
    def __init__(
        self, config: dict[str, Any], fold: str, *, shape: list[int] | None = None
    ) -> None:
        self.tumor_dataset = None
        if config["data"].get("training_crop") == "tumor_union_adaptive":
            from .first_post_tumor_crops import TumorCropDataset

            self.tumor_dataset = TumorCropDataset(config, fold, shape=shape)
            self.records = self.tumor_dataset.records
            self.normalization = self.tumor_dataset.normalization
            return
        prepared = Path(config["output_dir"]) / "data"
        manifest = json.loads((prepared / "manifest.json").read_text())
        self.normalization = json.loads((prepared / "normalization.json").read_text())
        if manifest["preparation_config"] != preparation_signature(config):
            raise ValueError("Training preprocessing differs from the prepared data")
        if (
            self.normalization["schema"] != NUMERIC_CONTRACT
            or self.normalization["fit_fold"] != "train"
        ):
            raise ValueError("Invalid first-post normalization contract")
        if config.get("cohort"):
            root, selected = select_records(config)
            cohort = json.loads((prepared / "cohort.json").read_text())
            if (
                cohort["selection_config"] != cohort_selection_config(config)
                or cohort["source_root"] != str(root)
                or cohort["records"] != selected
                or [source_record(r) for r in manifest["records"]] != selected
                or self.normalization.get("preparation_config")
                != preparation_signature(config)
                or self.normalization["fit_visits"]
                != sum(r["fold"] == "train" for r in selected)
            ):
                raise ValueError(
                    "Prepared cohort or normalization differs from the split"
                )
        self.root = Path(manifest["source_root"])
        self.records = [r for r in manifest["records"] if r["fold"] == fold]
        self.config, self.fold = config, fold
        self.shape = shape or config["data"]["output_shape_zyx"]

    def __len__(self) -> int:
        if self.tumor_dataset is not None:
            return len(self.tumor_dataset)
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.tumor_dataset is not None:
            return self.tumor_dataset[index]
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
        array, _ = load_volume(self.root, self.records[index], self.config)
        patch = image_patch(array, self.shape, random_crop=self.fold == "train")
        foreground = patch != 0
        patch[foreground] = (
            patch[foreground] - self.normalization["mean"]
        ) / self.normalization["std"]
        if not np.isfinite(patch).all():
            raise ValueError("Non-finite normalized first-post patch")
        return {"image": torch.from_numpy(patch[None])}
