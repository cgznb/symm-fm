from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from mewm_ispy2.ispy2_biflow_latent_contract import (
    prepare_ispy2_biflow_latent_cache,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prepare-ispy2-biflow-latents")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    identity = prepare_ispy2_biflow_latent_cache(
        Path(args.source_root),
        Path(args.target_root),
    )
    print(identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
