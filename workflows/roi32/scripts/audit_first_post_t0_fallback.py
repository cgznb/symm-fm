"""Audit empty-mask T0 references and direct physical-coordinate coverage."""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mewm_ispy2.first_post_data import timestamp, write_json


def audit(segmentation):
    inventory = json.loads((segmentation / "inventory.json").read_text())
    records = inventory["records"]
    index = {(r["patient_id"], r["visit"]): r for r in records}
    reports = {
        r["case_id"]: json.loads(
            (segmentation / "case_reports" / f"{r['case_id']}.json").read_text()
        )
        for r in records
    }
    cases = []
    for record in records:
        current = reports[record["case_id"]]["qc"]
        if current["tumor_voxels"]:
            continue
        baseline = index.get((record["patient_id"], "T0"))
        row = {
            "case_id": record["case_id"],
            "visit": record["visit"],
            "reference_visit": "T0",
        }
        if baseline is None:
            row["reference_status"] = "t0_missing"
        elif not reports[baseline["case_id"]]["qc"]["tumor_voxels"]:
            row["reference_status"] = "t0_empty"
        else:
            row["reference_status"] = "available"
            row["reference_case_id"] = baseline["case_id"]
            baseline_qc = reports[baseline["case_id"]]["qc"]
            g0, g = baseline_qc["geometry"], current["geometry"]
            bounds = np.asarray(baseline_qc["bbox_zyx_exclusive"])[::-1]
            points = np.asarray(
                list(itertools.product(*[(a - 0.5, b - 0.5) for a, b in bounds]))
            )
            physical = (
                np.asarray(g0["direction_lps"]).reshape(3, 3)
                @ (points * g0["spacing_xyz_mm"]).T
            ).T + g0["origin_lps_mm"]
            native = (
                np.linalg.solve(
                    np.asarray(g["direction_lps"]).reshape(3, 3),
                    (physical - g["origin_lps_mm"]).T,
                ).T
                / g["spacing_xyz_mm"]
            )
            limits = np.asarray(g["shape_zyx"])[::-1] - 0.5
            row["direct_T0_bbox_center_inside_current_acquisition"] = bool(
                np.all((native.mean(axis=0) >= -0.5) & (native.mean(axis=0) <= limits))
            )
            row["direct_T0_bbox_all_corners_inside_current_acquisition"] = bool(
                np.all((native >= -0.5) & (native <= limits))
            )
        cases.append(row)
    available = [r for r in cases if r["reference_status"] == "available"]
    summary = {
        "total_visits": len(records),
        "empty_visits": len(cases),
        "empty_by_visit": dict(Counter(r["visit"] for r in cases)),
        "reference_status": dict(Counter(r["reference_status"] for r in cases)),
        "direct_T0_bbox_center_outside_current_acquisition": sum(
            not r["direct_T0_bbox_center_inside_current_acquisition"] for r in available
        ),
        "direct_T0_bbox_all_corners_inside_current_acquisition": sum(
            r["direct_T0_bbox_all_corners_inside_current_acquisition"]
            for r in available
        ),
    }
    return {
        "updated_utc": timestamp(),
        "segmentation_root": str(segmentation),
        "summary": summary,
        "scope": "metadata-only acquisition containment; not anatomical registration or segmentation accuracy",
        "model_inference": False,
        "crop_cache_replaced": False,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segmentation-root",
        type=Path,
        default=Path(
            _release_path('@data/ispy2_first_post_mamamia_segmentation_v1')
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.segmentation_root)
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
