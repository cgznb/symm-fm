"""Run an isolated three-phase geometry, codec, training and pCR pilot."""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/three_phase_symmflow_pilot_v2.yaml")
    parser.add_argument("--stage", choices=("inventory", "geometry-audit", "codec-check", "prepare",
                                           "train", "sample", "pcr-interface", "all"), default="codec-check")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Pause after this many new optimizer updates")
    args = parser.parse_args()
    from mewm_ispy2.three_phase_pilot_data import (
        admit_inventory, codec_report, geometry_audit, load_config, prepare_images, prepare_inventory, require_geometry_audit,
    )
    from mewm_ispy2.first_post_world_data import read_json, write_json
    cfg = load_config(args.config)
    output = Path(cfg["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    gpu = str(cfg["runtime"]["physical_gpu"])
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be positive")
    if args.stage not in ("inventory", "geometry-audit") and os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
        raise SystemExit(result.returncode)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = prepare_inventory(cfg)
        if args.stage == "inventory":
            print(json.dumps({"stage": "inventory_ready", "visits": len(inventory["visits"]), "pairs": len(inventory["pairs"])}))
            return
        if args.stage in ("geometry-audit", "prepare", "all", "codec-check"):
            report = geometry_audit(cfg, inventory)
            print(json.dumps({key: report[key] for key in ("spatial_policy", "visits", "minimum_coverage", "passed")}), flush=True)
            if args.stage == "geometry-audit":
                return
        if args.stage != "codec-check":
            inventory, admission = admit_inventory(cfg, inventory, read_json(output / "geometry_audit.json"))
            print(json.dumps({"stage": "data_admission", **admission}), flush=True)
        if args.stage in ("prepare", "train", "all"):
            require_geometry_audit(cfg)
        gpu_lock_root = Path(_release_path('@workspace/.codex/source_bridge_recovery_20260912/gpu_locks'))
        gpu_lock_root.mkdir(parents=True, exist_ok=True)
        with (gpu_lock_root / f"gpu{gpu}.lock").open("a") as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            import torch
            free, total = torch.cuda.mem_get_info()
            if total - free > 2 * 2**30:
                raise RuntimeError("Pilot GPU is occupied")
            torch.cuda.set_per_process_memory_fraction(cfg["runtime"]["memory_fraction"])
            def progress(value):
                write_json(output / "progress.json", value)
                print(json.dumps({key: item for key, item in value.items() if key != "records"}), flush=True)
            torch.set_num_threads(cfg["runtime"]["cpu_threads"])
            if args.stage in ("codec-check", "prepare", "all"):
                limit = cfg["selection"]["codec_check_train_patients"] if args.stage == "codec-check" else None
                prepare_images(cfg, inventory, case_limit=limit, progress=progress)
            if args.stage == "codec-check":
                result = codec_report(cfg, inventory)
                progress({key: result[key] for key in ("stage", "training_patients", "visits", "phase_summary")})
            elif args.stage == "prepare":
                progress({"stage": "preparation_complete", "visits": sum(v["pilot_split"] != "excluded_geometry"
                                                                                      for v in inventory["visits"])})
            elif args.stage == "all":
                result = codec_report(cfg, inventory)
                progress({key: result[key] for key in ("stage", "training_patients", "visits", "phase_summary")})
            if args.stage in ("train", "all"):
                from mewm_ispy2.three_phase_pilot_training import train
                completed = train(cfg, inventory, resume=args.resume, progress=progress, stop_after=args.stop_after)
                if not completed:
                    return
            if args.stage in ("sample", "all"):
                from mewm_ispy2.three_phase_pilot_training import sample_validation
                progress(sample_validation(cfg, inventory))
            if args.stage in ("pcr-interface", "all"):
                from mewm_ispy2.three_phase_pilot_training import verify_pcr_interface
                progress(verify_pcr_interface(cfg, inventory))


if __name__ == "__main__":
    main()
