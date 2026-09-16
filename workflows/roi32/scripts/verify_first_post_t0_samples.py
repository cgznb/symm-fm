"""Replay sampled crop intensities from native MRI with an independent sampler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mewm_ispy2.first_post_data import read_config, write_json
from mewm_ispy2.first_post_tumor_crops import case_id, verified_base


def verify(config):
    output = Path(config["output_dir"]) / "data"
    manifest = json.loads((output / "t0_crop_overrides.json").read_text())
    root, records, norm = verified_base(config)
    sources = {case_id(r): r for r in records}
    reports, thumbnails = [], []
    for row in manifest["records"]:
        if row["case_id"] not in manifest["review_cases"]:
            continue
        source = sources[row["case_id"]]
        raw = np.asarray(
            nib.load(root / source["relative_path"]).dataobj, dtype=np.float32
        )
        cached = np.load(row["cache_path"], mmap_mode="r", allow_pickle=False)
        g = row["crop"]["geometry"]
        axes = [np.linspace(0, size - 1, 13).astype(int) for size in cached.shape]
        zyx = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        xyz = zyx[:, ::-1]
        physical = (
            np.asarray(g["direction_lps"]).reshape(3, 3) @ (xyz * g["spacing_xyz_mm"]).T
        ).T + g["origin_lps_mm"]
        iop = np.asarray(source["image_orientation_patient"])
        direction = np.column_stack((iop[:3], iop[3:], np.cross(iop[:3], iop[3:])))
        spacing = [*source["pixel_spacing_yx_mm"][::-1], source["slice_spacing_mm"]]
        indices = (
            np.linalg.solve(
                direction, (physical - source["image_position_patient_first"]).T
            ).T
            / spacing
        )
        limits = np.asarray(raw.shape)[::-1] - 1
        inside = np.all((indices >= 0) & (indices <= limits), axis=1)
        outside = np.any((indices < -0.5) | (indices > limits + 0.5), axis=1)
        expected = map_coordinates(
            raw, indices[inside, ::-1].T, order=1, mode="nearest"
        )
        nonzero = expected != 0
        expected[nonzero] = (expected[nonzero] - norm["mean"]) / norm["std"]
        expected = expected.astype(np.float16).astype(np.float32)
        actual = np.asarray(cached[tuple(zyx[inside].T)], np.float32)
        error = np.abs(actual - expected)
        if not inside.any() or not np.all(error <= np.abs(expected) / 1024 + 1e-5):
            raise ValueError("Independent native-to-crop intensity replay differs")
        if np.any(cached[tuple(zyx[outside].T)]):
            raise ValueError("Out-of-acquisition crop pixels are not zero")
        reports.append(
            {
                "case_id": row["case_id"],
                "inside_samples": int(inside.sum()),
                "outside_samples": int(outside.sum()),
                "max_error": float(error.max()),
            }
        )
        with Image.open(output / "crop_review" / f"{row['case_id']}.png") as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((900, 600))
            thumbnails.append(thumbnail)
    if len(reports) != len(manifest["review_cases"]) or not reports:
        raise ValueError("Incomplete T0 sample review")
    sheet = Image.new("RGB", (1800, 600 * ((len(thumbnails) + 1) // 2)), "white")
    for index, thumbnail in enumerate(thumbnails):
        sheet.paste(thumbnail, ((index % 2) * 900, (index // 2) * 600))
    sheet.save(output / "crop_review_contact.png")
    result = {
        "passed": True,
        "method": "scipy_native_physical_coordinate_intensity_replay",
        "reviewed_crops": len(reports),
        "records": reports,
    }
    write_json(output / "independent_sample_verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    verify(read_config(parser.parse_args().config))
