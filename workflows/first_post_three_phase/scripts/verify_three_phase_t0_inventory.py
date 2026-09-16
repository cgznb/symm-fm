"""Independently check the corrected cohort and all source/target view grids."""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mewm_ispy2.first_post_world_data import read_json, write_json
from mewm_ispy2.three_phase_all_pairs_data import load_config


def verify(cfg):
    root = Path(cfg["output_root"])
    new = read_json(root / "inventory.json")
    old = read_json(Path(cfg["reuse_preparation_from"]) / "inventory.json")
    corrections = read_json(cfg["crop_overrides"])
    excluded_cases = {r["case_id"] for r in corrections["exclusions"]}
    excluded = {
        r["visit_id"]
        for r in old["visits"]
        if f"{r['patient_id']}_{r['visit']}_aqc1" in excluded_cases
    }
    expected = {
        r["pair_id"]
        for r in old["pairs"]
        if r["earlier_visit_id"] not in excluded and r["later_visit_id"] not in excluded
    }
    if {r["pair_id"] for r in new["pairs"]} != expected:
        raise ValueError(
            "Corrected pairs do not exactly match examination-level exclusions"
        )
    old_visits = {r["visit_id"]: r for r in old["visits"]}
    visits = {r["visit_id"]: r for r in new["visits"]}
    replacements = {r["case_id"]: r for r in corrections["records"]}
    fallback_views = 0
    for view in new["views"]:
        source, target = visits[view["source_visit_id"]], visits[view["visit_id"]]
        grid = view["geometry"]
        case = f"{target['patient_id']}_{target['visit']}_aqc1"
        if (
            source["split"] != target["split"]
            or source["split"] != old_visits[source["visit_id"]]["split"]
        ):
            raise ValueError("Patient split changed")
        if case in replacements:
            crop = replacements[case]["crop"]
            if grid != crop["geometry"]:
                raise ValueError("A training view changes the T0-derived crop")
            points = np.asarray(crop["reference_mask_faces_in_visit_lps_mm"])
            projected = np.linalg.solve(
                np.asarray(grid["direction_lps"]).reshape(3, 3),
                (points - grid["origin_lps_mm"]).T,
            ).T
            limits = (np.asarray(grid["shape_zyx"])[::-1] - 1) * grid["spacing_xyz_mm"]
            if np.any(projected < 20 - 1e-4) or np.any(projected > limits - 20 + 1e-4):
                raise ValueError("T0 bounds or margin are truncated by a training view")
            fallback_views += 1
        else:
            for field in ("shape_zyx", "spacing_xyz_mm", "direction_lps"):
                if grid[field] != source["crop_geometry"][field]:
                    raise ValueError("Nonempty target source extent changed")

            def center(g):
                return np.asarray(g["origin_lps_mm"]) + np.asarray(
                    g["direction_lps"]
                ).reshape(3, 3) @ (
                    (np.asarray(g["shape_zyx"])[::-1] - 1) * g["spacing_xyz_mm"] / 2
                )

            np.testing.assert_allclose(
                center(grid), center(target["crop_geometry"]), atol=1e-6
            )
    counts = {
        s: {
            "patients": len({p["patient_id"] for p in new["pairs"] if p["split"] == s}),
            "visits": sum(v["split"] == s for v in new["visits"]),
            "pairs": sum(p["split"] == s for p in new["pairs"]),
        }
        for s in ("train", "val")
    }
    train, val = [
        {p["patient_id"] for p in new["pairs"] if p["split"] == s}
        for s in ("train", "val")
    ]
    if train & val:
        raise ValueError("Patient split overlap")
    if set(new["pair_counts"]["train"]) != {
        f"T{a}->T{b}" for a, b in itertools.combinations(range(4), 2)
    }:
        raise ValueError("A forward transition was lost")
    result = {
        "passed": True,
        "exclusion_counts": dict(Counter(r["reason"] for r in new["exclusions"])),
        "removed_visits_from_old_phase_inventory": len(excluded),
        "removed_pairs": len(old["pairs"]) - len(new["pairs"]),
        "retained": counts,
        "verified_views": len(new["views"]),
        "t0_fallback_views": fallback_views,
        "t0_fallback_visits": sum(
            v["crop_localization"] == "empty_mask_t0_mask_fallback"
            for v in new["visits"]
        ),
        "geometry_admission_pending": True,
    }
    write_json(root / "t0_inventory_verification.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    verify(load_config(parser.parse_args().config))
