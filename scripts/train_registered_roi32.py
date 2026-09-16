"""Run the registered ROI32 adapter against the shared selected VQ and caches."""

from research_release import path as _release_path, load_yaml as _release_yaml

import argparse
import subprocess
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "registered_roi32_5090.yaml"))
    parser.add_argument("--stage", choices=("smoke-fm", "fm", "evaluate"))
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    config = _release_yaml(Path(args.config).read_text())
    if config["schema"] != "registered_roi32_external_runner_v1":
        raise ValueError("Unexpected registered ROI32 runner configuration")
    command = [config["python"], "-B", config["runner"], "--config", config["shared_config"],
               "--stage", args.stage or config["default_stage"]]
    if args.detach:
        command.append("--detach")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
