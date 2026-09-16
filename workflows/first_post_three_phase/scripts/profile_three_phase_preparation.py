"""Compare phase transport and exact-grid preparation against completed cases."""

from __future__ import annotations

import argparse
import copy
import filecmp
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mewm_ispy2 import three_phase_all_pairs_data as original
from mewm_ispy2.first_post_world_data import identity, read_json, write_json
from mewm_ispy2.three_phase_pilot_data import normalize_images, support_coverage
from mewm_ispy2.three_phase_preparation import PhaseTransfer, grid_key, prepare_arrays


def run(cfg, output, count):
    output.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["output_root"])
    inventory = read_json(root / "inventory.json")
    visits = {v["visit_id"]: v for v in inventory["visits"]}
    reports = [read_json(p) for p in sorted((root / "prepared").glob("*.json"))]
    selected = []
    for report in reports:
        if report["excluded_training_patient"]:
            continue
        views = [v for v in inventory["views"] if v["patient_id"] == report["patient_id"]]
        if len({grid_key(v) for v in views}) < len(views):
            selected.append((report, views))
        if len(selected) == count:
            break
    data, settings = original.pcr_helpers(cfg)
    old_settings, new_settings = [copy.deepcopy(settings) for _ in range(2)]
    old_settings["output_dir"] = str(output / "legacy")
    new_settings["output_dir"] = str(output / "multiplexed")
    for value in (old_settings, new_settings):
        Path(value["output_dir"]).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    results = []
    with PhaseTransfer(data, new_settings) as transfer:
        for index, (report, views) in enumerate(selected):
            patient_visits = [visits[k] for k in sorted({v["visit_id"] for v in views})]
            timings = {}
            for mode in (("legacy", "multiplexed") if index % 2 == 0 else ("multiplexed", "legacy")):
                started = time.perf_counter()
                if mode == "legacy":
                    data.stage_phases(old_settings, patient_visits)
                else:
                    transfer.stage(patient_visits)
                timings[mode + "_transfer_seconds"] = time.perf_counter() - started
            try:
                for visit in patient_visits:
                    for remote in visit["remote_phases"].values():
                        relative = Path(remote).relative_to(settings["remote_root"])
                        if not filecmp.cmp(output / "legacy/staging" / relative,
                                           output / "multiplexed/staging" / relative, shallow=False):
                            raise ValueError("Transport changed native phase bytes")
                fast, coverage, profile = prepare_arrays(
                    data, new_settings, views, visits, original._resample, normalize_images,
                    inventory["image_normalization"], support_coverage)
                timings.update(profile)
                images = {}
                for visit in patient_visits:
                    paths = [output / "legacy/staging" / Path(visit["remote_phases"]["pre"]).relative_to(settings["remote_root"]),
                             Path(visit["native_first_post"]),
                             output / "legacy/staging" / Path(visit["remote_phases"]["late"]).relative_to(settings["remote_root"])]
                    images[visit["visit_id"]] = [data.native_image(path, visit) for path in paths]
                started = time.perf_counter()
                source_support = {key: original._resample(images[key][1], visits[key]["crop_geometry"], support=True).astype(bool)
                                  for key in {v["source_visit_id"] for v in views}}
                for view, row in zip(views, coverage, strict=True):
                    native = images[view["visit_id"]]
                    raw = np.stack([original._resample(image, view["geometry"]) for image in native])
                    support = np.stack([original._resample(image, view["geometry"], support=True).astype(bool) for image in native])
                    normalized, foreground = normalize_images(raw, **inventory["image_normalization"])
                    if not all(np.array_equal(a, b) for a, b in zip((normalized, foreground, support), fast[grid_key(view)], strict=True)):
                        raise ValueError("Deduplication changed MRI values or masks")
                    values = [support_coverage(source_support[view["source_visit_id"]], mask) for mask in support]
                    if values != row["values"]:
                        raise ValueError("Deduplication changed source-dependent coverage")
                timings["legacy_resample_seconds"] = time.perf_counter() - started
                original_latents, compared = {}, 0
                for view in views:
                    key = grid_key(view)
                    latent = np.load(root / view["latent_file"], allow_pickle=False)
                    if key in original_latents:
                        if not np.array_equal(latent, original_latents[key]):
                            raise ValueError("Identical-grid legacy codec outputs differ")
                        compared += 1
                    else:
                        original_latents[key] = latent
                results.append({"case_index": index, "exact_arrays": True, "exact_transport": True,
                                "exact_duplicate_latents": compared, **timings})
                write_json(output / "profile.json", {"inventory": identity(root / "inventory.json"), "records": results})
                print(results[-1], flush=True)
                del images, fast, original_latents
            finally:
                data.release_phases(old_settings, patient_visits)
                data.release_phases(new_settings, patient_visits)
    write_json(output / "COMPLETE.json", {"passed": True, "cases": len(results), "gpu_inference": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=3)
    args = parser.parse_args()
    run(original.load_config(args.config), args.output, args.cases)
