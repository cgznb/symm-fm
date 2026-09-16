from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mewm_ispy2.first_post_segmentation import (
    configuration,
    predictor,
    select_gpu,
    write_json,
)


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = configuration(args.config.resolve())
    execution = {**config, "queue": {**config["queue"], "gpus": [args.gpu]}}
    gpu, lock = select_gpu(execution)
    if gpu is None:
        raise RuntimeError("Benchmark GPU is occupied or reserved")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        instance, _ = predictor(config, "cuda:0")
        network = instance.network.eval().to("cuda:0")
        total = torch.cuda.get_device_properties(0).total_memory
        trials = []
        torch.manual_seed(config["seed"])
        for batch in (1, 2, 4, 8, 16, 32, 64):
            data = output = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                data = torch.randn(batch, 1, 128, 128, 128, device="cuda:0")
                seconds = []
                with torch.inference_mode(), torch.autocast("cuda"):
                    for repetition in range(4):
                        torch.cuda.synchronize()
                        started = time.monotonic()
                        output = network(data)
                        torch.cuda.synchronize()
                        elapsed = time.monotonic() - started
                        if not torch.isfinite(output).all():
                            raise RuntimeError("Non-finite benchmark output")
                        del output
                        output = None
                        if repetition:
                            seconds.append(elapsed)
                trial = {
                    "batch_size": batch,
                    "status": "passed",
                    "median_forward_seconds": statistics.median(seconds),
                    "patches_per_second": batch / statistics.median(seconds),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            except torch.cuda.OutOfMemoryError:
                trial = {"batch_size": batch, "status": "out_of_memory"}
            trials.append(trial)
            print(trial, flush=True)
            del data, output
            gc.collect()
            torch.cuda.empty_cache()
            usable = [
                r
                for r in trials
                if r["status"] == "passed" and r["peak_reserved_bytes"] < total * 0.9
            ]
            best = max(usable, key=lambda r: r["patches_per_second"])
            selected = best["batch_size"]
            write_json(
                args.output,
                {
                    "scope": "real_128_cubed_network_forward_batch_throughput",
                    "physical_gpu": gpu,
                    "gpu": torch.cuda.get_device_name(),
                    "total_memory_bytes": total,
                    "trials": trials,
                    "selected_forward_batch": selected,
                    "inference": {
                        "tile_batch_size": max(1, selected // 8),
                        "mirror_batch_size": min(8, selected),
                        "gpu_accumulation": True,
                        "prefetch_cases": 2,
                    },
                },
            )
            if trial["status"] != "passed":
                break
    finally:
        lock.close()


if __name__ == "__main__":
    main()
