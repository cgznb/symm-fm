from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mewm_ispy2.first_post_segmentation import (
    configuration,
    finished,
    read_json,
    run_case,
    same_grid,
    select_gpu,
    write_json,
)
from mewm_ispy2.first_post_segmentation_parallel import run_batch


def main() -> None:
    import numpy as np
    import SimpleITK as sitk

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-reference", type=Path)
    parser.add_argument("--accelerated", action="store_true")
    parser.add_argument("--tile-batch-size", type=int, default=1)
    args = parser.parse_args()
    config = configuration(args.config.resolve())
    root = Path(config["output_dir"])
    candidates = [
        r
        for r in read_json(root / "inventory.json")["records"]
        if r["cohort_fold"] == "train" and finished(root, r)
    ]
    candidates.sort(
        key=lambda r: np.prod(
            read_json(root / "case_reports" / f"{r['case_id']}.json")[
                "preprocessed_shape_czyx"
            ]
        )
    )
    selected = [candidates[len(candidates) // 4], candidates[3 * len(candidates) // 4]]
    if args.fresh_reference:
        selected = read_json(args.fresh_reference / "inventory.json")["records"]
    smoke_root = args.output.resolve()
    if smoke_root.exists():
        raise ValueError("Use a new benchmark output directory")
    execution = {**config, "queue": {**config["queue"], "gpus": [args.gpu]}}
    gpu, lock = select_gpu(execution)
    if gpu is None:
        raise RuntimeError("Benchmark GPU is occupied or reserved")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for directory in ("inputs", "masks", "case_reports", "overlays"):
            (smoke_root / directory).mkdir(parents=True, exist_ok=True)
        write_json(smoke_root / "inventory.json", {"records": selected})
        batch = smoke_root / "batch.json"
        write_json(batch, {"case_ids": [r["case_id"] for r in selected]})
        started = time.monotonic()
        smoke_config = {**config, "output_dir": str(smoke_root)}
        if args.accelerated:
            smoke_config["inference_execution"] = {
                "tile_batch_size": args.tile_batch_size,
                "mirror_batch_size": 8,
                "gpu_accumulation": True,
                "prefetch_cases": 2,
            }
        if args.fresh_reference and not args.accelerated:
            for record in selected:
                run_case(smoke_config, record, "cuda:0")
        else:
            run_batch(
                smoke_config,
                batch,
                "cuda:0",
                smoke_root / "worker_status.json",
            )
        comparisons = []
        for record in selected:
            name = record["case_id"] + ".nii.gz"
            previous = sitk.ReadImage(
                str((args.fresh_reference or root) / "masks" / name)
            )
            current = sitk.ReadImage(str(smoke_root / "masks" / name))
            grid_matches = same_grid(previous, current)
            old_values = sitk.GetArrayFromImage(previous)
            new_values = sitk.GetArrayFromImage(current)
            different = int(np.count_nonzero(old_values != new_values))
            foreground_sum = int(
                np.count_nonzero(old_values) + np.count_nonzero(new_values)
            )
            dice = (
                1.0
                if foreground_sum == 0
                else 2
                * int(np.count_nonzero((old_values != 0) & (new_values != 0)))
                / foreground_sum
            )
            comparisons.append(
                {
                    "same_grid": grid_matches,
                    "different_voxels": different,
                    "mask_dice": dice,
                }
            )
        passed = all(
            r["same_grid"]
            and (
                r["mask_dice"] >= 0.999
                if args.accelerated
                else r["different_voxels"] == 0
            )
            for r in comparisons
        )
        result = {
            "status": "passed" if passed else "failed",
            "scope": "same_gpu_fp16_batched_vs_original"
            if args.accelerated
            else "same_gpu_fresh_vs_reused"
            if args.fresh_reference
            else "cross_gpu_reused_vs_original",
            "physical_gpu": gpu,
            "elapsed_seconds": time.monotonic() - started,
            "comparisons": comparisons,
            "inference_execution": smoke_config.get("inference_execution", {}),
            "acceptance": "same physical grid; mask Dice >= 0.999"
            if args.accelerated
            else "same physical grid; identical mask voxels",
        }
        write_json(smoke_root / "verification.json", result)
        print(result, flush=True)
        if not passed:
            raise RuntimeError("Reused-model masks differ from committed outputs")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
