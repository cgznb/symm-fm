#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mewm_ispy2.ispy2_biflow_cohort_evaluation import export_adjacent_metric_subset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export adjacent-transition metrics from a full BiFlow cohort"
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export_adjacent_metric_subset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
