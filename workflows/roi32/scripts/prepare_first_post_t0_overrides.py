"""Prepare verified empty-mask replacements and examination-level exclusions."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    prepare_one,
    verified_base,
)
from scripts.verify_first_post_tumor_crops import verify_one


def prepare_overrides(config):
    policy = config["data"]["tumor_crop"]
    if (
        policy["empty_mask"] != "t0_mask"
        or policy["reference_mapping"] != "physical_lps"
    ):
        raise ValueError("This correction requires direct T0 physical-LPS localization")
    output = Path(config["output_dir"]) / "data"
    for name in ("crops", "crop_reports", "crop_review"):
        (output / name).mkdir(parents=True, exist_ok=True)
    with (output / "crop_preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
        root, selected, normalization = verified_base(config)
        references = {(r["patient_id"], r["visit"]): r for r in selected}
        admitted, exclusions = admit_crop_records(root, selected, config)
        segmentation = Path(policy["segmentation_root"])
        candidates, unchanged = [], []
        for record in admitted:
            _, report, identities = dependencies(segmentation, root, record)
            if report["qc"]["tumor_voxels"]:
                unchanged.append(
                    {
                        "case_id": case_id(record),
                        "patient_id": record["patient_id"],
                        "visit": record["visit"],
                        "fold": record["fold"],
                        "dependencies": identities,
                    }
                )
            else:
                candidates.append(record)
        admission = {
            "selected_visits": len(selected),
            "retained_visits": len(admitted),
            "replacements": len(candidates),
            "unchanged_nonempty_visits": len(unchanged),
            "excluded_visits": len(exclusions),
            "exclusion_counts": dict(Counter(r["reason"] for r in exclusions)),
        }
        write_json(
            output / "crop_admission.json", {**admission, "exclusions": exclusions}
        )
        print(json.dumps(admission), flush=True)
        review = {
            case_id(r)
            for stage in ("T1", "T2", "T3")
            for r in [v for v in candidates if v["visit"] == stage][:2]
        }
        rows = []
        with ThreadPoolExecutor(
            max_workers=config["data"]["preparation_workers"]
        ) as pool:
            for index, result in enumerate(
                pool.map(
                    lambda r: prepare_one(
                        root, r, config, normalization, review, references
                    ),
                    candidates,
                ),
                1,
            ):
                record = {
                    **result["source_record"],
                    "crop": result["crop"],
                    "dependencies": result["dependencies"],
                    "cache_identity": result["cache_identity"],
                }
                verify_one(record, config, root, references)
                rows.append(
                    {
                        "case_id": case_id(record),
                        "patient_id": record["patient_id"],
                        "visit": record["visit"],
                        "fold": record["fold"],
                        "crop": record["crop"],
                        "dependencies": record["dependencies"],
                        "cache_path": str(output / "crops" / f"{case_id(record)}.npy"),
                        "cache_identity": record["cache_identity"],
                    }
                )
                if index % 10 == 0 or index == len(candidates):
                    progress = {
                        "stage": "preparing_t0_overrides",
                        "completed": index,
                        "total": len(candidates),
                        "updated_utc": timestamp(),
                    }
                    write_json(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        # Verify exclusion decisions again before publishing a consumer manifest.
        if admit_crop_records(root, selected, config) != (admitted, exclusions):
            raise ValueError("Crop admission changed during preparation")
        result = {
            "schema": "first_post_t0_crop_overrides_v1",
            "verified": True,
            "preparation_config": preparation_signature(config),
            "source_root": str(root),
            "source_manifest_identity": file_identity(Path(config["source_manifest"])),
            "image_normalization": {k: normalization[k] for k in ("mean", "std")},
            "policy": {
                "empty_mask": "t0_mask",
                "reference_mapping": "physical_lps",
                "outside_center": "exclude_visit",
                "unavailable_t0": "exclude_visit",
            },
            "summary": admission,
            "records": rows,
            "unchanged": unchanged,
            "exclusions": exclusions,
            "review_cases": sorted(review),
        }
        manifest = output / "t0_crop_overrides.json"
        if manifest.exists() and json.loads(manifest.read_text()) != result:
            raise ValueError(
                "Published T0 correction differs; use a new output directory"
            )
        if not manifest.exists():
            write_json(manifest, result)
        write_json(
            output / "progress.json",
            {"stage": "complete", "updated_utc": timestamp(), **admission},
        )
        return admission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_overrides(read_config(args.config))), flush=True)


if __name__ == "__main__":
    main()
