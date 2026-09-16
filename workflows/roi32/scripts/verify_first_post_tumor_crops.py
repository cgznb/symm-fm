from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from mewm_ispy2.first_post_data import (
    preparation_signature,
    read_config,
    timestamp,
    write_json,
)
from mewm_ispy2.first_post_tumor_crops import (
    admit_crop_records,
    case_id,
    dependencies,
    file_identity,
    t0_reference_dependencies,
    verified_base,
)


def verify_one(record, config, source_root, reference_records=None):
    output = Path(config["output_dir"]) / "data"
    path = output / "crops" / f"{case_id(record)}.npy"
    if file_identity(path) != record["cache_identity"]:
        raise ValueError("Crop cache identity differs")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        list(array.shape) != config["data"]["output_shape_zyx"]
        or array.dtype != np.float16
        or not np.isfinite(array).all()
        or not np.any(array)
    ):
        raise ValueError("Invalid crop tensor")
    mask_path, report, identities = dependencies(
        Path(config["data"]["tumor_crop"]["segmentation_root"]), source_root, record
    )
    reference_mask, reference_transform = None, None
    if (
        config["data"]["tumor_crop"]["empty_mask"] == "t0_mask"
        and not report["qc"]["tumor_voxels"]
    ):
        if reference_records is None:
            reference_records = {
                (r["patient_id"], r["visit"]): r for r in verified_base(config)[1]
            }
        path, reference_transform, reference_identity = t0_reference_dependencies(
            Path(config["data"]["tumor_crop"]["segmentation_root"]),
            source_root,
            record,
            config,
            reference_records,
        )
        identities["t0_reference"] = reference_identity
        reference_mask = sitk.ReadImage(str(path))
    if identities != record["dependencies"]:
        raise ValueError("Segmentation dependency changed")
    mask = sitk.ReadImage(str(mask_path))
    labels = sitk.GetArrayViewFromImage(mask)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Nonbinary source segmentation")
    count = np.count_nonzero(labels)
    if (
        count != record["crop"]["source_mask_voxels"]
        or count != report["qc"]["tumor_voxels"]
    ):
        raise ValueError("Source segmentation voxel count differs")
    if reference_mask is not None:
        values = sitk.GetArrayViewFromImage(reference_mask)
        if not np.isin(values, (0, 1)).all() or not np.any(values):
            raise ValueError("Invalid T0 localization mask")
        if record["crop"]["localization"] != "empty_mask_t0_mask_fallback" or record[
            "crop"
        ]["localization_mask_voxels"] != np.count_nonzero(values):
            raise ValueError("T0 localization provenance differs")
        indices = np.where(values)
        low = np.array([p.min() for p in indices])[::-1] - 0.5
        high = np.array([p.max() for p in indices])[::-1] + 0.5
        mask = reference_mask
    elif count:
        indices = np.where(labels)
        low = np.array([p.min() for p in indices])[::-1] - 0.5
        high = np.array([p.max() for p in indices])[::-1] + 0.5
    else:
        low, high = np.full(3, -0.5), np.array(mask.GetSize()) - 0.5
    physical = np.array(
        [
            mask.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in point))
            for point in itertools.product(*zip(low, high, strict=True))
        ]
    )
    if reference_transform is not None:
        physical = np.asarray(
            [reference_transform.TransformPoint(tuple(point)) for point in physical]
        )
        current = sitk.ReadImage(str(mask_path))
        center = np.asarray(
            current.TransformPhysicalPointToContinuousIndex(
                tuple(physical.mean(axis=0))
            )
        )
        if np.any(center < -0.5) or np.any(
            center > np.asarray(current.GetSize()) - 0.5
        ):
            raise ValueError("T0 center is outside the current acquisition")
        if record["crop"]["resampled_mask_voxels"] != 0:
            raise ValueError("An empty follow-up mask acquired baseline labels")
    geometry = record["crop"]["geometry"]
    direction = np.array(geometry["direction_lps"]).reshape(3, 3)
    spacing = np.array(geometry["spacing_xyz_mm"])
    projected_mm = np.linalg.solve(
        direction, (physical - geometry["origin_lps_mm"]).T
    ).T
    limits = (np.array(geometry["shape_zyx"][::-1]) - 1) * spacing
    margin = float(record["crop"]["margin_mm"])
    if np.any(projected_mm < margin - 1e-4) or np.any(
        projected_mm > limits - margin + 1e-4
    ):
        raise ValueError("A source mask face or required margin is outside the crop")
    return {
        "fold": record["fold"],
        "empty": not bool(count),
        "expanded": record["crop"]["spacing_scale"] > 1,
        "max_abs_normalized_intensity": float(np.max(np.abs(array))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--visual-review-passed", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    output = Path(config["output_dir"]) / "data"
    root, selected, normalization = verified_base(config)
    reference_records = {(r["patient_id"], r["visit"]): r for r in selected}
    selected, exclusions = admit_crop_records(root, selected, config)
    manifest = json.loads((output / "manifest.json").read_text())
    if (
        manifest["preparation_config"] != preparation_signature(config)
        or json.loads((output / "normalization.json").read_text()) != normalization
    ):
        raise ValueError("Prepared crop configuration or normalization differs")
    if manifest.get("exclusions", []) != exclusions:
        raise ValueError("Prepared crop exclusions differ")
    records = manifest["records"]
    if [
        {
            k: v
            for k, v in r.items()
            if k not in ("crop", "cache_identity", "dependencies")
        }
        for r in records
    ] != selected:
        raise ValueError("Crop cohort differs from all selected first-post images")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda r: verify_one(r, config, root, reference_records), records)
        )
    contract = json.loads((output / "crop_contract.json").read_text())
    review_cases = contract["review_cases"]
    for case in review_cases:
        file_identity(output / "crop_review" / f"{case}.png")
    train = {r["patient_id"] for r in records if r["fold"] == "train"}
    val = {r["patient_id"] for r in records if r["fold"] == "val"}
    if train & val:
        raise ValueError("Patient leakage in crop cohort")
    result = {
        "status": "passed",
        "updated_at_utc": timestamp(),
        "preparation_config": preparation_signature(config),
        "verified_cache_and_mask_pairs": len(results),
        "excluded_visits": len(exclusions),
        "visit_counts": dict(Counter(r["fold"] for r in results)),
        "patient_counts": {"train": len(train), "val": len(val)},
        "patient_overlap": 0,
        "all_cache_values_finite": True,
        "all_mask_voxel_faces_and_margins_inside_crop": True,
        "empty_mask_fallback_counts": dict(
            Counter(r["fold"] for r in results if r["empty"])
        ),
        "expanded_fov_counts": dict(
            Counter(r["fold"] for r in results if r["expanded"])
        ),
        "maximum_abs_normalized_intensity": max(
            r["max_abs_normalized_intensity"] for r in results
        ),
        "training_only_normalization_reproduced": True,
        "sample_review_images": len(review_cases),
        "review_scope": "technical_geometry_and_crop_coverage_not_clinical_segmentation_accuracy",
    }
    write_json(output / "verification.json", result)
    if args.visual_review_passed:
        write_json(
            output / "crop_review.json",
            {
                **result,
                "reviewed_cases": review_cases,
                "visual_review": "sample_overlays_inspected_for_spatial_alignment_and_field_coverage",
                "limitations": [
                    "empty_masks_do_not_establish_complete_response",
                    "all_predicted_components_retained_including_possible_false_positives",
                    "border_predictions_and_native_scan_truncation_remain_flagged",
                    "full_cohort_has_not_been_clinically_adjudicated",
                ],
            },
        )
    print(
        json.dumps({k: v for k, v in result.items() if k != "preparation_config"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
