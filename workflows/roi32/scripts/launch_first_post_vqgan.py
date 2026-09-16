from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _release_yaml(config_path.read_text())
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print((root / "queue_status.json").read_text())
            return
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="",
    )
    with (root / "launcher.log").open("a") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "mewm_ispy2.first_post_vqgan",
                "queue",
                "--config",
                str(config_path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(
        json.dumps(
            {
                "queue_pid": child.pid,
                "output_dir": str(root),
                "status_file": str(root / "queue_status.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
