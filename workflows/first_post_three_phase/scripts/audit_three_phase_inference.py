"""Run real inference with future image/cache reads explicitly rejected."""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from mewm_ispy2.three_phase_pilot_data import load_config
    from mewm_ispy2.first_post_world_data import read_json, write_json
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/three_phase_symmflow_pilot_v2.yaml")
    parser.add_argument("--stage", choices=("sample", "pcr-interface"), required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    gpu = str(cfg["runtime"]["physical_gpu"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
        raise SystemExit(result.returncode)
    root = Path(cfg["output_root"])
    inventory = read_json(root / "admitted_inventory.json")
    denied, allowed, opened = set(), set(), set()
    staging_prefix = str((root / "staging").resolve()) + os.sep
    for visit in inventory["visits"]:
        cache = str((root / visit["cache_file"]).resolve())
        if visit["pilot_split"] == "val" and visit["visit"] == "T0":
            allowed.add(cache)
        else:
            denied.add(cache)
        if visit["visit"] != "T0":
            for field in ("native_first_post", "image_path", "mask_path"):
                denied.add(str(Path(visit[field]).resolve()))
    def guard(event, arguments):
        if event != "open" or not isinstance(arguments[0], (str, bytes)):
            return
        path = str(Path(os.fsdecode(arguments[0])).resolve())
        if path in denied or path.startswith(staging_prefix):
            raise RuntimeError("Inference attempted to read a forbidden future or training image")
        if path in allowed:
            opened.add(path)
    sys.addaudithook(guard)
    runner = ROOT / "scripts/run_three_phase_symmflow_pilot.py"
    sys.argv = [str(runner), "--config", str(args.config.resolve()), "--stage", args.stage]
    runpy.run_path(str(runner), run_name="__main__")
    if opened != allowed:
        raise ValueError("Inference did not read exactly the declared validation T0 caches")
    result = {"passed": True, "stage": args.stage, "future_image_reads": 0,
              "guard": "Python file-open audit on current image-loading paths",
              "native_phase_staging_reads_forbidden": True,
              "forbidden_asset_paths": len(denied), "allowed_source_caches": len(allowed),
              "opened_source_caches": sorted(str(Path(p).relative_to(root)) for p in opened)}
    write_json(root / f"{args.stage}_access_audit.json", result)
    print({key: value for key, value in result.items() if key != "opened_source_caches"}, flush=True)


if __name__ == "__main__":
    main()
