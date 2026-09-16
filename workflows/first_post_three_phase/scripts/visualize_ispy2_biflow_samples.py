#!/usr/bin/env python3
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mewm_ispy2.ispy2_biflow_visualization import main


if __name__ == "__main__":
    raise SystemExit(main())
