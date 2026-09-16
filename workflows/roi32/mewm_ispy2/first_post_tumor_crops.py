from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
import shutil
import stat
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

from .first_post_data import (
    NUMERIC_CONTRACT,
    combine_moments,
    preparation_signature,
    read_config,
    select_records,
    source_record,
    timestamp,
    write_cohort,
    write_json,
)
from .first_post_segmentation import geometry, native_image, overlay, same_grid

CROP_POLICY = "tumor_union_adaptive"
GIB = 1024**3
RENDER_LOCK = threading.Lock()


class CropExclusion(ValueError):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def file_identity(path: Path) -> dict[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular crop dependency")
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def case_id(record: dict[str, Any]) -> str:
    return f"{record['patient_id']}_{record['visit']}_aqc1"


def advise_verified_cache(path, identity, maximum):
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or {"size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns} != identity
        ):
            raise ValueError("Crop cache changed before memory advice")
        length = min(info.st_size, maximum)
        os.posix_fadvise(descriptor, 0, length, os.POSIX_FADV_DONTNEED)
        return length
    finally:
        os.close(descriptor)


def release_verified_crop_cache(config):
    from .first_post_vqgan import resources

    root = Path(config["output_dir"])
    manifest = json.loads((root / "data/manifest.json").read_text())
    if manifest["preparation_config"] != preparation_signature(config):
        raise ValueError("Crop cache release configuration differs")
    before = snapshot = resources(root)
    target = (config["queue"]["admission_memory_gib"] + 4) * GIB
    maximum, advised, files = 32 * GIB, 0, 0
    cache_root = (root / "data/crops").resolve()
    for record in reversed(manifest["records"]):
        if (
            snapshot["memory_headroom_bytes"] is None
            or snapshot["memory_headroom_bytes"] >= target
            or advised >= maximum
        ):
            break
        path = cache_root / f"{case_id(record)}.npy"
        if path.is_symlink() or not path.resolve().is_relative_to(cache_root):
            raise ValueError("Crop cache leaves the prepared dataset")
        advised += advise_verified_cache(
            path, record["cache_identity"], maximum - advised
        )
        files += 1
        snapshot = resources(root)
    return {
        "event": "verified_crop_cache_release",
        "updated_at_utc": timestamp(),
        "files_advised": files,
        "bytes_advised": advised,
        "max_advice_bytes": maximum,
        "target_headroom_bytes": target,
        "before": before,
        "after": snapshot,
    }


def crop_reference(
    image: sitk.Image,
    mask: sitk.Image,
    data: dict[str, Any],
    *,
    t0_mask: sitk.Image | None = None,
    t0_to_visit: sitk.Transform | None = None,
) -> tuple[sitk.Image, dict[str, Any]]:
    if not same_grid(image, mask):
        raise ValueError("MRI and segmentation grids differ")
    labels = sitk.GetArrayViewFromImage(mask)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Segmentation is not binary")
    count = int(np.count_nonzero(labels))
    if not count and data["tumor_crop"].get("empty_mask") == "t0_mask":
        return t0_crop_reference(image, data, t0_mask, t0_to_visit)
    if count:
        positions = np.where(labels)
        low = np.array([int(p.min()) for p in positions])[::-1]
        high = np.array([int(p.max()) + 1 for p in positions])[::-1]
        margin = float(data["tumor_crop"]["margin_mm"])
        localization = "all_predicted_components_union"
    else:
        low, high = np.zeros(3), np.array(image.GetSize())
        margin = 0.0
        localization = "empty_mask_full_image_fallback"
    native_spacing = np.array(image.GetSpacing())
    size = np.array(data["output_shape_zyx"][::-1])
    base_spacing = np.array(data["target_spacing_xyz"])
    extent = (high - low) * native_spacing
    # Include outer voxel faces and the margin inside the output voxel centers.
    scale = max(1.0, float(np.max((extent + 2 * margin) / ((size - 1) * base_spacing))))
    spacing = base_spacing * scale
    center = np.array(
        image.TransformContinuousIndexToPhysicalPoint(tuple((low + high - 1) / 2))
    )
    direction = np.array(image.GetDirection()).reshape(3, 3)
    origin = center - direction @ ((size - 1) * spacing / 2)
    reference = sitk.Image([int(n) for n in size], sitk.sitkFloat32)
    reference.SetOrigin(tuple(origin))
    reference.SetDirection(image.GetDirection())
    reference.SetSpacing(tuple(spacing))
    padding_mm = ((size - 1) * spacing - extent) / 2
    if np.any(padding_mm < margin - 1e-5):
        raise ValueError("Crop excludes part of the localization or its margin")
    return reference, {
        "localization": localization,
        "source_mask_voxels": count,
        "bbox_xyz_exclusive": np.column_stack((low, high)).tolist(),
        "bbox_extent_xyz_mm": extent.tolist(),
        "margin_mm": margin,
        "minimum_margin_xyz_mm": padding_mm.tolist(),
        "spacing_scale": scale,
        "all_source_mask_voxel_faces_in_fov": True,
        "geometry": geometry(reference),
    }


def t0_crop_reference(image, data, t0_mask, t0_to_visit):
    if t0_mask is None or t0_to_visit is None:
        raise ValueError(
            "Empty follow-up requires a nonempty T0 mask and explicit T0-to-visit mapping"
        )
    if t0_to_visit.GetDimension() != 3 or not t0_to_visit.IsLinear():
        raise ValueError(
            "T0 bounding-box transfer requires a linear physical transform"
        )
    offset = np.asarray(t0_to_visit.TransformPoint((0.0, 0.0, 0.0)))
    matrix = np.column_stack(
        [
            np.asarray(t0_to_visit.TransformPoint(tuple(axis))) - offset
            for axis in np.eye(3)
        ]
    )
    if not np.isfinite(matrix).all() or np.linalg.det(matrix) <= 0:
        raise ValueError("T0 mapping must preserve a nondegenerate physical volume")
    values = sitk.GetArrayViewFromImage(t0_mask)
    if not np.isin(values, (0, 1)).all() or not np.any(values):
        raise ValueError("T0 localization mask must be nonempty and binary")
    indices = np.where(values)
    low = np.array([p.min() for p in indices])[::-1] - 0.5
    high = np.array([p.max() for p in indices])[::-1] + 0.5
    # Transfer voxel faces in physical coordinates, never reuse T0 array indices.
    points = np.asarray(
        [
            t0_to_visit.TransformPoint(
                t0_mask.TransformContinuousIndexToPhysicalPoint(
                    tuple(float(v) for v in point)
                )
            )
            for point in itertools.product(*zip(low, high, strict=True))
        ]
    )
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    projected = np.linalg.solve(direction, points.T).T
    minimum, maximum = projected.min(axis=0), projected.max(axis=0)
    extent = maximum - minimum
    if not np.isfinite(points).all() or np.any(extent <= 0):
        raise ValueError("Invalid transformed T0 mask extent")
    size = np.asarray(data["output_shape_zyx"][::-1])
    base_spacing = np.asarray(data["target_spacing_xyz"])
    margin = float(data["tumor_crop"]["margin_mm"])
    scale = max(1.0, float(np.max((extent + 2 * margin) / ((size - 1) * base_spacing))))
    spacing = base_spacing * scale
    center = direction @ ((minimum + maximum) / 2)
    origin = center - direction @ ((size - 1) * spacing / 2)
    reference = sitk.Image([int(n) for n in size], sitk.sitkFloat32)
    reference.SetOrigin(tuple(origin))
    reference.SetDirection(image.GetDirection())
    reference.SetSpacing(tuple(spacing))
    padding = ((size - 1) * spacing - extent) / 2
    if np.any(padding < margin - 1e-5):
        raise ValueError("Crop excludes transformed T0 mask faces or required margin")
    acquired = np.asarray(
        [
            image.TransformPhysicalPointToContinuousIndex(tuple(point))
            for point in points
        ]
    )
    acquired_center = np.asarray(
        image.TransformPhysicalPointToContinuousIndex(tuple(center))
    )
    acquisition_limits = np.asarray(image.GetSize()) - 0.5
    return reference, {
        "localization": "empty_mask_t0_mask_fallback",
        "source_mask_voxels": 0,
        "localization_mask_voxels": int(np.count_nonzero(values)),
        "localization_reference_visit": "T0",
        "reference_mask_faces_in_visit_lps_mm": points.tolist(),
        "bbox_xyz_exclusive": None,
        "bbox_extent_xyz_mm": extent.tolist(),
        "margin_mm": margin,
        "minimum_margin_xyz_mm": padding.tolist(),
        "spacing_scale": scale,
        "all_source_mask_voxel_faces_in_fov": True,
        "all_localization_mask_voxel_faces_in_fov": True,
        "localization_center_in_acquisition": bool(
            np.all((acquired_center >= -0.5) & (acquired_center <= acquisition_limits))
        ),
        "all_localization_bbox_corners_in_acquisition": bool(
            np.all((acquired >= -0.5) & (acquired <= acquisition_limits))
        ),
        "geometry": geometry(reference),
    }


def normalize_cache(array: np.ndarray, normalization: dict[str, Any]) -> np.ndarray:
    foreground = array != 0
    if not foreground.any() or not np.isfinite(array).all():
        raise ValueError("Crop has empty or non-finite MRI values")
    array[foreground] = (array[foreground] - normalization["mean"]) / normalization[
        "std"
    ]
    cached = array.astype(np.float16)
    if not np.isfinite(cached).all():
        raise ValueError("Non-finite normalized float16 crop")
    if np.any(np.abs(cached.astype(np.float32) - array) > np.abs(array) / 2048 + 1e-6):
        raise ValueError("Unexpected float16 cache rounding error")
    return cached


def verified_base(config: dict[str, Any]):
    root, selected = select_records(config)
    base = Path(config["data"]["tumor_crop"]["prepared_data"])
    manifest = json.loads((base / "manifest.json").read_text())
    normalization = json.loads((base / "normalization.json").read_text())
    if (
        manifest["source_root"] != str(root)
        or [source_record(r) for r in manifest["records"]] != selected
    ):
        raise ValueError("Base preparation does not match the complete patient split")
    original = manifest["preparation_config"]
    for key in (
        "source_manifest",
        "source_verification",
        "baseline_bundle",
        "cohort",
        "seed",
    ):
        if original.get(key) != config.get(key):
            raise ValueError("Base preparation selection changed")
    for key in (
        "registered",
        "phase_index",
        "orientation",
        "normalization",
        "target_spacing_xyz",
    ):
        if original["data"][key] != config["data"][key]:
            raise ValueError("Base normalization geometry changed")
    training = [r for r in manifest["records"] if r["fold"] == "train"]
    moments = combine_moments([r["moments"] for r in training])
    if (
        normalization.get("preparation_config", original) != original
        or normalization["schema"] != NUMERIC_CONTRACT
        or normalization["fit_fold"] != "train"
        or normalization["fit_visits"] != len(training)
    ):
        raise ValueError("Base normalization is not training-only")
    if any(
        not np.isclose(normalization[k], moments[k], rtol=1e-12, atol=1e-10)
        for k in moments
    ):
        raise ValueError("Base normalization moments do not reproduce")
    normalization = {
        **normalization,
        "preparation_config": preparation_signature(config),
        "reused_whole_volume_statistics_from": str(base / "normalization.json"),
        "cache_storage_dtype": "float16_round_to_nearest",
    }
    return root, selected, normalization


def dependencies(segmentation: Path, root: Path, record: dict[str, Any]):
    case = case_id(record)
    mask_path = segmentation / "masks" / f"{case}.nii.gz"
    report_path = segmentation / "case_reports" / f"{case}.json"
    report = json.loads(report_path.read_text())
    source = root / record["relative_path"]
    if file_identity(source) != {
        "size_bytes": record["size_bytes"],
        "mtime_ns": record["local_mtime_ns"],
    }:
        raise ValueError("Source changed after cohort selection")
    if (
        report["status"] != "completed"
        or report["case_id"] != case
        or report["source_image"] != str(source)
        or report["source_mtime_ns"] != record["local_mtime_ns"]
        or report["mask_size_bytes"] != mask_path.stat().st_size
        or report["registered"] is not False
        or report["folds"] != [0, 1, 2, 3, 4]
        or report["mirror"] is not True
        or not (segmentation / "overlays" / f"{case}.png").is_file()
    ):
        raise ValueError("Segmentation provenance or completeness check failed")
    return (
        mask_path,
        report,
        {
            "mask": file_identity(mask_path),
            "report": file_identity(report_path),
            "source": file_identity(source),
        },
    )


def t0_reference_dependencies(segmentation, root, record, config, records):
    if record["visit"] not in ("T1", "T2", "T3"):
        raise CropExclusion(
            "empty_t0_mask",
            "Empty T0 requires manual localization; a future mask is not permitted",
        )
    baseline = records.get((record["patient_id"], "T0"))
    if baseline is None:
        raise CropExclusion(
            "missing_t0_reference",
            "T0 mask is unavailable; manual localization is required",
        )
    if baseline["patient_id"] != record["patient_id"] or baseline["visit"] != "T0":
        raise ValueError("T0 reference belongs to a different patient or visit")
    if baseline["fold"] != record["fold"]:
        raise ValueError("T0 and follow-up cross patient splits")
    try:
        path, report, identities = dependencies(segmentation, root, baseline)
    except FileNotFoundError as error:
        raise CropExclusion(
            "missing_t0_reference", "T0 reference files are unavailable"
        ) from error
    if report["qc"]["tumor_voxels"] <= 0:
        raise CropExclusion(
            "empty_t0_reference", "T0 mask is empty; manual localization is required"
        )
    policy = config["data"]["tumor_crop"]
    mode = policy["reference_mapping"]
    reference = {
        "case_id": case_id(baseline),
        "mapping": mode,
        "tumor_voxels": report["qc"]["tumor_voxels"],
        **identities,
    }
    if mode == "physical_lps":
        transform = sitk.Transform(3, sitk.sitkIdentity)
    elif mode == "transform_manifest":
        manifest_path = Path(policy["reference_transform_manifest"])
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["records"].get(case_id(record))
        if manifest.get("schema") != "t0_mask_physical_transforms_v1" or entry is None:
            raise ValueError("A validated T0 transform is unavailable")
        if (
            entry.get("status") != "passed"
            or entry.get("case_id") != case_id(record)
            or entry.get("reference_case_id") != case_id(baseline)
            or entry.get("mapping_direction") != "T0_LPS_to_visit_LPS"
            or entry.get("reference_image_identity") != identities["source"]
            or entry.get("visit_image_identity")
            != file_identity(root / record["relative_path"])
        ):
            raise ValueError("T0 transform source, direction or image identity differs")
        transform_path = manifest_path.parent / entry["transform_path"]
        transform_identity = file_identity(transform_path)
        if transform_identity != entry["transform_identity"]:
            raise ValueError("T0 transform changed")
        transform = sitk.ReadTransform(str(transform_path))
        reference.update(
            transform=transform_identity,
            transform_manifest=file_identity(manifest_path),
        )
    else:
        raise ValueError("T0 fallback requires an explicit physical mapping policy")
    if transform.GetDimension() != 3 or not transform.IsLinear():
        raise ValueError(
            "T0 bounding-box transfer requires a linear physical transform"
        )
    return path, transform, reference


def admit_crop_records(root, records, config):
    """Exclude individual unlocalizable visits while preserving the patient split."""
    if config["data"]["tumor_crop"]["empty_mask"] != "t0_mask":
        return records, []
    segmentation = Path(config["data"]["tumor_crop"]["segmentation_root"])
    references = {(r["patient_id"], r["visit"]): r for r in records}
    admitted, exclusions = [], []
    for record in records:
        path, report, identities = dependencies(segmentation, root, record)
        try:
            if not report["qc"]["tumor_voxels"]:
                baseline_path, transform, reference = t0_reference_dependencies(
                    segmentation, root, record, config, references
                )
                identities["t0_reference"] = reference
                mask = sitk.ReadImage(str(path), sitk.sitkUInt8)
                if np.any(sitk.GetArrayViewFromImage(mask)):
                    raise ValueError("Decoded mask differs from segmentation report")
                baseline = sitk.ReadImage(str(baseline_path), sitk.sitkUInt8)
                _, crop = crop_reference(
                    mask, mask, config["data"], t0_mask=baseline, t0_to_visit=transform
                )
                if crop["localization_mask_voxels"] != reference["tumor_voxels"]:
                    raise ValueError(
                        "Decoded T0 mask differs from its segmentation report"
                    )
                if not crop["localization_center_in_acquisition"]:
                    raise CropExclusion(
                        "t0_center_outside_acquisition",
                        "T0 tumor-box center is outside the current acquisition",
                    )
        except CropExclusion as error:
            exclusions.append(
                {
                    "case_id": case_id(record),
                    "patient_id": record["patient_id"],
                    "visit": record["visit"],
                    "fold": record["fold"],
                    "reason": error.reason,
                    "dependencies": identities,
                }
            )
            continue
        admitted.append(record)
    return admitted, exclusions


def prepare_one(
    root, record, config, normalization, review_cases, reference_records=None
):
    output = Path(config["output_dir"]) / "data"
    segmentation = Path(config["data"]["tumor_crop"]["segmentation_root"])
    case = case_id(record)
    report_path = output / "crop_reports" / f"{case}.json"
    cache_path = output / "crops" / f"{case}.npy"
    mask_path, segmentation_report, identity = dependencies(segmentation, root, record)
    baseline_path, baseline_transform = None, None
    if (
        config["data"]["tumor_crop"]["empty_mask"] == "t0_mask"
        and not segmentation_report["qc"]["tumor_voxels"]
    ):
        if reference_records is None:
            raise ValueError(
                "T0 reference inventory is required for empty-mask preparation"
            )
        baseline_path, baseline_transform, reference_identity = (
            t0_reference_dependencies(
                segmentation, root, record, config, reference_records
            )
        )
        identity["t0_reference"] = reference_identity
    contract = {"data": config["data"], "normalization": normalization}
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if (
            existing["source_record"] != record
            or existing["dependencies"] != identity
            or existing["contract"] != contract
            or existing["cache_identity"] != file_identity(cache_path)
        ):
            raise ValueError("Existing crop or dependency changed")
        if (
            case in review_cases
            and not (output / "crop_review" / f"{case}.png").is_file()
        ):
            raise ValueError("Crop review image is missing")
        return existing
    if shutil.disk_usage(output).free < 12 * GIB:
        raise RuntimeError("Insufficient disk headroom for crop preparation")
    source = root / record["relative_path"]
    nii = nib.load(source)
    if not np.allclose(nii.affine, np.eye(4)):
        raise ValueError("Native source is not the expected converted ZYX image")
    image = sitk.DICOMOrient(
        native_image(np.asarray(nii.dataobj, dtype=np.float32), record), "LPS"
    )
    mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    baseline_mask = (
        sitk.ReadImage(str(baseline_path), sitk.sitkUInt8)
        if baseline_path is not None
        else None
    )
    reference, crop = crop_reference(
        image,
        mask,
        config["data"],
        t0_mask=baseline_mask,
        t0_to_visit=baseline_transform,
    )
    if baseline_path is not None:
        if not crop["localization_center_in_acquisition"]:
            raise CropExclusion(
                "t0_center_outside_acquisition",
                "T0 tumor-box center is outside the current acquisition; exclude this visit",
            )
        if crop["localization_mask_voxels"] != identity["t0_reference"]["tumor_voxels"]:
            raise ValueError("Decoded T0 mask differs from its segmentation report")
        crop["t0_reference_case_id"] = identity["t0_reference"]["case_id"]
        crop["reference_mapping"] = identity["t0_reference"]["mapping"]
    if crop["source_mask_voxels"] != segmentation_report["qc"]["tumor_voxels"]:
        raise ValueError("Decoded mask differs from segmentation report")
    cropped_image = sitk.Resample(
        image, reference, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear, 0.0
    )
    cropped_mask = sitk.Resample(
        mask,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    crop["resampled_mask_voxels"] = int(
        np.count_nonzero(sitk.GetArrayViewFromImage(cropped_mask))
    )
    crop["segmentation_flags"] = segmentation_report["qc"]["flags"]
    if case in review_cases:
        with RENDER_LOCK:
            overlay(
                output / "crop_review" / f"{case}.png",
                cropped_image,
                cropped_mask,
                {"flags": [crop["localization"]]},
            )
    cache = normalize_cache(sitk.GetArrayFromImage(cropped_image), normalization)
    temporary = cache_path.with_suffix(".tmp.npy")
    np.save(temporary, cache, allow_pickle=False)
    reread = np.load(temporary, mmap_mode="r", allow_pickle=False)
    if not np.array_equal(cache, reread):
        raise ValueError("Crop cache readback differs")
    del reread
    temporary.replace(cache_path)
    result = {
        "source_record": record,
        "dependencies": identity,
        "contract": contract,
        "crop": crop,
        "cache_identity": file_identity(cache_path),
    }
    if (
        file_identity(source) != identity["source"]
        or file_identity(mask_path) != identity["mask"]
    ):
        raise ValueError("Crop dependency changed during resampling")
    if baseline_path is not None:
        _, _, current_reference = t0_reference_dependencies(
            segmentation, root, record, config, reference_records
        )
        if current_reference != identity["t0_reference"]:
            raise ValueError("T0 localization dependency changed during resampling")
    write_json(report_path, result)
    # Native volumes are no longer needed once their immutable crops are committed.
    with source.open("rb") as handle:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return result


def select_review_cases(records, segmentation):
    chosen = set()
    training = [r for r in records if r["fold"] == "train"]
    for visit in sorted({r["visit"] for r in training}):
        candidates = [r for r in training if r["visit"] == visit]
        for record in (candidates[0], candidates[len(candidates) // 2], candidates[-1]):
            chosen.add(case_id(record))
    reports = [
        json.loads((segmentation / "case_reports" / f"{case_id(r)}.json").read_text())
        for r in training
    ]
    for flag in (
        "empty_mask_not_proof_of_complete_response",
        "multiple_components_review_required",
        "foreground_touches_image_border",
    ):
        candidates = [r for r in reports if flag in r["qc"]["flags"]]
        chosen.update(r["case_id"] for r in candidates[:2])
    nonempty = [r for r in reports if r["qc"]["tumor_voxels"]]
    for axis in range(3):
        chosen.add(
            max(nonempty, key=lambda r: r["qc"]["bbox_extent_xyz_mm"][axis])["case_id"]
        )
    return sorted(chosen)


def prepare_tumor_crops(config):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    output = Path(config["output_dir"]) / "data"
    for name in ("crops", "crop_reports", "crop_review"):
        (output / name).mkdir(parents=True, exist_ok=True)
    with (output / "crop_preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root, records, normalization = verified_base(config)
        reference_records = {(r["patient_id"], r["visit"]): r for r in records}
        write_cohort(config, root, records)
        segmentation = Path(config["data"]["tumor_crop"]["segmentation_root"])
        status = json.loads((segmentation / "queue_status.json").read_text())
        if (
            status["status"] != "completed"
            or status["completed_cases"] != len(records)
            or status["total_cases"] != len(records)
        ):
            raise ValueError("Complete all segmentations before crop preparation")
        selected_count = len(records)
        records, exclusions = admit_crop_records(root, records, config)
        write_json(
            output / "crop_exclusions.json",
            {"selected_visits": selected_count, "exclusions": exclusions},
        )
        review_cases = select_review_cases(records, segmentation)
        signature = preparation_signature(config)
        contract = {
            "preparation_config": signature,
            "normalization": normalization,
            "review_cases": review_cases,
        }
        contract_path = output / "crop_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("Crop preparation contract changed")
        write_json(contract_path, contract)
        results = []
        workers = config["data"].get("preparation_workers", 2)
        if not 1 <= workers <= 4:
            raise ValueError("Crop preparation supports one to four workers")
        # Futures return only small metadata; full images stay bounded by workers.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, row in enumerate(
                pool.map(
                    lambda r: prepare_one(
                        root, r, config, normalization, review_cases, reference_records
                    ),
                    records,
                ),
                1,
            ):
                results.append(row)
                if index % 50 == 0 or index == len(records):
                    progress = {
                        "stage": "preparing_tumor_crops",
                        "pid": os.getpid(),
                        "completed_visits": index,
                        "total_visits": len(records),
                        "updated_at_utc": timestamp(),
                    }
                    write_json(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        summary = {
            "stage": "prepared",
            "completed_at_utc": timestamp(),
            "preparation_config": signature,
            "total_visits": len(records),
            "selected_visits": selected_count,
            "excluded_visits": len(exclusions),
            "exclusion_counts": dict(Counter(r["reason"] for r in exclusions)),
            "visit_counts": dict(Counter(r["fold"] for r in records)),
            "patient_counts": {
                fold: len({r["patient_id"] for r in records if r["fold"] == fold})
                for fold in ("train", "val")
            },
            "localization_counts": dict(
                Counter(r["crop"]["localization"] for r in results)
            ),
            "expanded_fov_counts": dict(
                Counter(
                    r["source_record"]["fold"]
                    for r in results
                    if r["crop"]["spacing_scale"] > 1
                )
            ),
            "all_source_mask_voxel_faces_in_fov": all(
                r["crop"]["all_source_mask_voxel_faces_in_fov"] for r in results
            ),
            "vanished_after_mask_resampling": sum(
                r["crop"]["source_mask_voxels"] > 0
                and r["crop"]["resampled_mask_voxels"] == 0
                for r in results
            ),
            "spacing_scale_percentiles": np.percentile(
                [
                    r["crop"]["spacing_scale"]
                    for r in results
                    if r["source_record"]["fold"] == "train"
                ],
                [0, 50, 90, 95, 99, 100],
            ).tolist(),
            "cache_dtype": "float16",
            "normalization": normalization,
            "review_cases": review_cases,
            "review_scope": "automated_geometry_and_coverage_with_sample_visual_review_pending_not_clinical_adjudication",
        }
        write_json(
            output / "manifest.json",
            {
                "preparation_config": signature,
                "source_root": str(root),
                "exclusions": exclusions,
                "records": [
                    {
                        **r["source_record"],
                        "crop": r["crop"],
                        "cache_identity": r["cache_identity"],
                        "dependencies": r["dependencies"],
                    }
                    for r in results
                ],
            },
        )
        write_json(output / "normalization.json", normalization)
        write_json(output / "summary.json", summary)
        write_json(output / "progress.json", summary)
        return summary


class TumorCropDataset(torch.utils.data.Dataset):
    def __init__(self, config, fold, *, shape=None):
        self.output = Path(config["output_dir"]) / "data"
        root, selected, normalization = verified_base(config)
        selected, exclusions = admit_crop_records(root, selected, config)
        manifest = json.loads((self.output / "manifest.json").read_text())
        if manifest["preparation_config"] != preparation_signature(config) or manifest[
            "source_root"
        ] != str(root):
            raise ValueError("Tumor crop manifest configuration differs")
        if manifest.get("exclusions", []) != exclusions:
            raise ValueError("Tumor crop exclusion decisions changed")
        if [
            {
                k: v
                for k, v in r.items()
                if k not in ("crop", "cache_identity", "dependencies")
            }
            for r in manifest["records"]
        ] != selected:
            raise ValueError("Tumor crop membership differs from the patient split")
        self.normalization = json.loads(
            (self.output / "normalization.json").read_text()
        )
        if self.normalization != normalization:
            raise ValueError(
                "Tumor crop normalization differs from training-only moments"
            )
        if fold not in ("train", "val"):
            raise ValueError("Unknown dataset fold")
        self.records = [r for r in manifest["records"] if r["fold"] == fold]
        self.config, self.fold, self.shape = config, fold, shape

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.output / "crops" / f"{case_id(record)}.npy"
        if file_identity(path) != record["cache_identity"]:
            raise ValueError("Tumor crop cache changed")
        image = np.load(path, allow_pickle=False, mmap_mode="r")
        if (
            list(image.shape) != self.config["data"]["output_shape_zyx"]
            or image.dtype != np.float16
        ):
            raise ValueError("Tumor crop cache shape or dtype changed")
        if self.shape is not None:
            start = [(n - m) // 2 for n, m in zip(image.shape, self.shape, strict=True)]
            image = image[
                tuple(slice(s, s + n) for s, n in zip(start, self.shape, strict=True))
            ]
        image = np.array(image, dtype=np.float32)
        if not np.isfinite(image).all():
            raise ValueError("Non-finite tumor crop input")
        from .first_post_vqgan import resources

        headroom = resources(self.output)["memory_headroom_bytes"]
        if headroom is not None and headroom < 16 * GIB:
            # The tensor owns its CPU copy; disk data survive clean page eviction.
            advise_verified_cache(
                path, record["cache_identity"], record["cache_identity"]["size_bytes"]
            )
        return {"image": torch.from_numpy(image[None])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_tumor_crops(read_config(args.config))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("review_cases", "normalization", "preparation_config")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
