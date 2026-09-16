from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mewm_ispy2.first_post_segmentation import (
    configuration,
    read_json,
    run_case,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = configuration(args.config.resolve())
    root = Path(config["output_dir"])
    records = read_json(root / "inventory.json")["records"]
    candidates = [r for r in records if r["pilot"] and r["cohort_fold"] == "train"]
    record = min(
        candidates,
        key=lambda r: (
            np.prod(r["shape_zyx"])
            * np.prod(r["pixel_spacing_yx_mm"])
            * r["slice_spacing_mm"],
            r["case_id"],
        ),
    )
    smoke_root = root / "cpu_smoke_single_fold"
    for directory in ("inputs", "masks", "case_reports", "overlays"):
        (smoke_root / directory).mkdir(parents=True, exist_ok=True)
    smoke = dict(config, output_dir=str(smoke_root), folds=[0], mirror=False)
    run_case(smoke, record, "cpu")
    print(
        json.dumps(
            {
                "status": "passed",
                "scope": "full_image_single_fold_runtime_smoke_only",
                "case_id": record["case_id"],
                "output_dir": str(smoke_root),
            }
        )
    )


if __name__ == "__main__":
    main()
