"""Use the original ITK/elastix runtime to replay retained selected transforms."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration-repo", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.registration_repo) / "src"))
    import itk
    from ispy2_registration.fallback_visualization import load_parameter_object
    from ispy2_registration.models import Geometry
    from ispy2_registration.transforms import displacement_and_jacobian
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(2)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            meta = json.loads(Path(request["meta_path"]).read_text())
            geometries = []
            for name in ("source_geometry", "target_geometry"):
                value = meta[name]
                geometries.append(Geometry(tuple(value["shape_zyx"]), tuple(value["spacing_xyz"]), tuple(value["direction"])))
            moving, fixed = geometries
            field, _ = displacement_and_jacobian(moving, fixed, load_parameter_object(request["pair_dir"]))
            output = Path(request["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, field, allow_pickle=False)
            print(json.dumps({"status": "passed", "output_path": str(output)}), flush=True)
        except Exception as error:
            print(json.dumps({"status": "error", "reason": str(error)}), flush=True)


if __name__ == "__main__":
    main()
