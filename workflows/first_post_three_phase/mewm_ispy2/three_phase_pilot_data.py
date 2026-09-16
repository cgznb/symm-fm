"""Development-only three-phase MRI with explicit longitudinal ROI framing."""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from .first_post_world_data import identity, load_codec, read_json, write_json
from .three_phase_symmflow import IMAGE_SHAPE, PHASES, SharedThreePhaseCodec


SCHEMA = "three_phase_symmflow_pilot_v1"
FIXED_GRID = "source_t0_grid_no_longitudinal_registration"
CENTERED_GRID = "visit_center_with_t0_extent"
PAIR_FIELDS = (
    "pair_id", "patient_id", "earlier_visit_id", "later_visit_id", "earlier_stage", "later_stage",
    "delta_days", "interval_missing", "interval_source", "baseline_clinical", "treatment",
)


def load_config(path):
    cfg = _release_yaml(Path(path).read_text())
    if cfg["schema"] != SCHEMA or tuple(cfg["phase_order"]) != PHASES:
        raise ValueError("Unsupported three-phase experiment contract")
    if cfg["source_policy"] != "direct_real_t0" or cfg["spatial_policy"] not in (FIXED_GRID, CENTERED_GRID):
        raise ValueError("Unsupported source or spatial policy")
    if not cfg["selection"]["development_only"] or not cfg["selection"]["require_complete_four_visits"]:
        raise ValueError("This pilot requires complete development patients")
    return cfg


def training_geometry(visit, policy):
    """Future localization constructs supervised targets; it is never a model input."""
    result = copy.deepcopy(visit["source_geometry"])
    if policy == FIXED_GRID or visit["visit"] == "T0":
        return result
    if policy != CENTERED_GRID:
        raise ValueError("Unsupported spatial policy")
    native = visit["crop_geometry"]
    def center_offset(geometry):
        size_xyz = np.asarray(geometry["shape_zyx"])[::-1]
        direction = np.asarray(geometry["direction_lps"]).reshape(3, 3)
        return direction @ ((size_xyz - 1) * np.asarray(geometry["spacing_xyz_mm"]) / 2)
    center = np.asarray(native["origin_lps_mm"]) + center_offset(native)
    result["origin_lps_mm"] = (center - center_offset(result)).tolist()
    return result


def pcr_helpers(cfg):
    root = str(Path(cfg["pillar_repo"]).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from src import first_post_pcr_data as data
    settings = data.load_config(cfg["pcr_input_config"])
    settings.update(output_dir=cfg["output_root"], reserve_gib=cfg["runtime"]["reserve_gib"],
                    transfer_workers=cfg["runtime"]["transfer_workers"])
    data.world_imports(settings)
    return data, settings


def select_patients(cohort, world, cfg):
    visits = {}
    for row in cohort["visits"]:
        if row["split"] == "train":
            visits.setdefault(row["canonical_patient_id"], {})[row["visit"]] = row
    eligible = sorted(pid for pid, rows in visits.items() if set(rows) == {"T0", "T1", "T2", "T3"})
    count = cfg["selection"]["train_patients"] + cfg["selection"]["validation_patients"]
    if len(eligible) < count:
        raise ValueError("Insufficient complete development patients for the pilot")
    rng = np.random.default_rng(cfg["seed"])
    chosen = [eligible[int(i)] for i in rng.permutation(len(eligible))[:count]]
    train_count = cfg["selection"]["train_patients"]
    assignments = {pid: "train" if i < train_count else "val" for i, pid in enumerate(chosen)}
    if set(chosen) & set(cohort["split"]["val"]):
        raise ValueError("The original validation cohort must not enter this pilot")
    pair_index = {(p["patient_id"], p["earlier_stage"], p["later_stage"]): p for p in world["pairs"]}
    selected_visits, pairs = [], []
    for case_index, pid in enumerate(chosen):
        source = visits[pid]["T0"]
        for stage in ("T0", "T1", "T2", "T3"):
            row = copy.deepcopy(visits[pid][stage])
            if row["late_index"] <= 1 or not row["remote_phases"]["pre"].endswith("_aqc_0.nii.gz"):
                raise ValueError("Three-phase metadata does not identify pre and late")
            row.update(pilot_split=assignments[pid], pilot_case_index=case_index,
                       grid_visit_id=source["visit_id"], source_geometry=copy.deepcopy(source["crop_geometry"]),
                       cache_file=f"prepared/case_{case_index:03d}_{stage}.npz")
            selected_visits.append(row)
            if stage != "T0":
                original = pair_index[pid, "T0", stage]
                pair = {key: copy.deepcopy(original.get(key)) for key in PAIR_FIELDS}
                pair.update(split=assignments[pid], grid_visit_id=source["visit_id"])
                pairs.append(pair)
    return selected_visits, pairs


def prepare_inventory(cfg):
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    binding = {key: identity(cfg[key]) for key in ("source_cohort", "source_world_bundle", "codec_checkpoint")}
    path = root / "inventory.json"
    if path.exists():
        inventory = read_json(path)
        if (inventory["source_files"] != binding or inventory["selection"] != cfg["selection"]
                or inventory["seed"] != cfg["seed"] or inventory["spatial_policy"] != cfg["spatial_policy"]
                or inventory["source_policy"] != cfg["source_policy"]):
            raise ValueError("Existing pilot inventory differs from the requested data")
        return inventory
    cohort, world = read_json(cfg["source_cohort"]), read_json(cfg["source_world_bundle"])
    visits, pairs = select_patients(cohort, world, cfg)
    inventory = {"schema": SCHEMA, "phase_order": list(PHASES), "source_files": binding,
                 "selection": cfg["selection"], "seed": cfg["seed"], "visits": visits, "pairs": pairs,
                 "spatial_policy": cfg["spatial_policy"], "source_policy": cfg["source_policy"],
                 "original_validation_included": False,
                 "image_normalization": {"mean": world["normalization"]["image_mean"],
                                         "std": world["normalization"]["image_std"],
                                         "scope": "shared_frozen_first_post_training_statistics"}}
    write_json(path, inventory)
    return inventory


def normalize_images(images, mean, std):
    images = np.asarray(images, dtype=np.float32)
    if images.shape != (len(PHASES), *IMAGE_SHAPE) or not np.isfinite(images).all():
        raise ValueError("Invalid three-phase MRI shape or values")
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid shared intensity statistics")
    foreground = images != 0
    result = np.zeros_like(images)
    result[foreground] = (images[foreground] - mean) / std
    return result, foreground


def atomic_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cached_arrays(cfg, visit):
    path = Path(cfg["output_root"]) / visit["cache_file"]
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.array(source[name]) for name in source.files}
    validate_cache(arrays)
    return arrays


def validate_cache(arrays):
    expected = {"images": (3, *IMAGE_SHAPE), "reconstruction": (3, *IMAGE_SHAPE),
                "foreground": (3, *IMAGE_SHAPE), "latent": (24, 24, 64, 64),
                "source_support": IMAGE_SHAPE, "reference_support": (3, *IMAGE_SHAPE)}
    for name, shape in expected.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid cached {name} shape or values")
    if not arrays["source_support"].any():
        raise ValueError("Source ROI has no acquisition support")


def support_coverage(source_support, reference_support):
    source = np.asarray(source_support, dtype=bool)
    reference = np.asarray(reference_support, dtype=bool)
    if source.shape != IMAGE_SHAPE or reference.shape != IMAGE_SHAPE or not source.any():
        raise ValueError("Invalid source or reference acquisition support")
    return float(reference[source].mean())


def geometry_audit(cfg, inventory):
    """Inspect local first-post fields before downloading other phases or training."""
    import SimpleITK as sitk
    data, _ = pcr_helpers(cfg)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(cfg["runtime"]["cpu_threads"])
    sources = {}
    rows = []
    for visit in inventory["visits"]:
        if data.identity(visit["native_first_post"]) != visit["native_first_post_identity"]:
            raise ValueError("Native first-post image changed")
        target = {**visit, "crop_geometry": training_geometry(visit, cfg["spatial_policy"])}
        support = data.resample_native_roi(visit["native_first_post"], target, support=True).astype(bool)
        if visit["visit"] == "T0":
            sources[visit["grid_visit_id"]] = support
        coverage = support_coverage(sources[visit["grid_visit_id"]], support)
        rows.append({"case_index": visit["pilot_case_index"], "stage": visit["visit"],
                     "source_support_coverage": coverage, "roi_acquisition_fraction": float(support.mean())})
    threshold = cfg.get("quality", {}).get("minimum_source_support_coverage", 0.5)
    result = {"spatial_policy": cfg["spatial_policy"], "visits": len(rows), "records": rows,
              "minimum_coverage": min(row["source_support_coverage"] for row in rows),
              "minimum_required_coverage": threshold,
              "passed": all(row["source_support_coverage"] >= threshold for row in rows),
              "reference_localization_uses_target_annotations": cfg["spatial_policy"] == CENTERED_GRID,
              "longitudinal_deformable_registration": False, "original_validation_loaded": False}
    write_json(Path(cfg["output_root"]) / "geometry_audit.json", result)
    return result


def require_geometry_audit(cfg):
    result = read_json(Path(cfg["output_root"]) / "data_admission.json")
    if result["spatial_policy"] != cfg["spatial_policy"] or not result["passed"]:
        raise ValueError("Acquisition coverage failed; inspect geometry_audit.json before training")


def admit_inventory(cfg, inventory, audit):
    """Record whole-patient training exclusions; never select on validation quality."""
    if audit["spatial_policy"] != cfg["spatial_policy"]:
        raise ValueError("Geometry audit spatial policy changed")
    threshold = cfg.get("quality", {}).get("minimum_source_support_coverage", 0.5)
    failed_cases = {r["case_index"] for r in audit["records"] if r["source_support_coverage"] < threshold}
    validation = {v["pilot_case_index"] for v in inventory["visits"] if v["pilot_split"] == "val"}
    if failed_cases & validation:
        raise ValueError("Validation acquisition coverage failed; validation patients cannot be filtered")
    if failed_cases and not cfg.get("quality", {}).get("exclude_failing_training_patients", False):
        raise ValueError("Acquisition coverage failed; inspect geometry_audit.json before training")
    result = copy.deepcopy(inventory)
    excluded_ids = set()
    for visit in result["visits"]:
        if visit["pilot_case_index"] in failed_cases:
            visit["pilot_split"] = "excluded_geometry"
            excluded_ids.add(visit["canonical_patient_id"])
    for pair in result["pairs"]:
        if pair["patient_id"] in excluded_ids:
            pair["split"] = "excluded_geometry"
    counts = {split: len({v["canonical_patient_id"] for v in result["visits"] if v["pilot_split"] == split})
              for split in ("train", "val", "excluded_geometry")}
    if not counts["train"] or not counts["val"]:
        raise ValueError("Geometry admission leaves an empty training or validation split")
    report = {"passed": True, "spatial_policy": cfg["spatial_policy"], "patients": counts,
              "excluded_case_indices": sorted(failed_cases), "minimum_required_coverage": threshold,
              "reason": "Whole training patients excluded when any visit fails acquisition coverage",
              "validation_filtered": False, "original_validation_included": False}
    root = Path(cfg["output_root"])
    for filename, value in (("admitted_inventory.json", result), ("data_admission.json", report)):
        path = root / filename
        if path.exists():
            if read_json(path) != value:
                raise ValueError("Existing data admission changed; use a new experiment directory")
        else:
            write_json(path, value)
    return result, report


def prepare_images(cfg, inventory, *, case_limit=None, progress=None):
    import SimpleITK as sitk
    data, settings = pcr_helpers(cfg)
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(cfg["runtime"]["cpu_threads"])
    root = Path(cfg["output_root"])
    codec = SharedThreePhaseCodec(load_codec(cfg["codec_checkpoint"], "cuda:0"))
    selected = [v for v in inventory["visits"] if v["pilot_split"] != "excluded_geometry"
                and (case_limit is None or v["pilot_case_index"] < case_limit)]
    by_id = {v["visit_id"]: v for v in inventory["visits"]}
    for index, visit in enumerate(selected):
        path = root / visit["cache_file"]
        if data.identity(visit["native_first_post"]) != visit["native_first_post_identity"]:
            raise ValueError("Native first-post image changed")
        if path.exists():
            cached_arrays(cfg, visit)
            continue
        data.disk_gate(settings, required=2 * 2**30)
        data.stage_phases(settings, [visit])
        try:
            target = {**visit, "crop_geometry": training_geometry(visit, cfg["spatial_policy"])}
            phase_paths = [root / "staging" / Path(visit["remote_phases"]["pre"]).relative_to(settings["remote_root"]),
                           Path(visit["native_first_post"]),
                           root / "staging" / Path(visit["remote_phases"]["late"]).relative_to(settings["remote_root"])]
            if data.identity(visit["native_first_post"]) != visit["native_first_post_identity"]:
                raise ValueError("Native first-post image changed")
            images = np.stack([data.resample_native_roi(p, target) for p in phase_paths])
            source = by_id[visit["grid_visit_id"]]
            source_support = data.resample_native_roi(source["native_first_post"], source, support=True).astype(bool)
            target_support = np.stack([data.resample_native_roi(p, target, support=True).astype(bool) for p in phase_paths])
            stats = inventory["image_normalization"]
            normalized, foreground = normalize_images(images, stats["mean"], stats["std"])
            tensor = torch.from_numpy(normalized)[None].to("cuda:0")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent = codec.encode(tensor).float()
                reconstructed = codec.decode(latent).float()
            arrays = dict(images=normalized, latent=latent[0].cpu().numpy(),
                          reconstruction=reconstructed[0].cpu().numpy(), foreground=foreground,
                          source_support=source_support, reference_support=target_support)
            validate_cache(arrays)
            atomic_arrays(path, **arrays)
            del tensor, latent, reconstructed
        finally:
            data.release_phases(settings, [visit])
        if progress:
            progress({"stage": "preparing_three_phase_images", "completed": index + 1, "total": len(selected)})
    del codec
    torch.cuda.empty_cache()


def phase_errors(predicted, target, foreground):
    predicted, target = np.asarray(predicted, np.float32), np.asarray(target, np.float32)
    foreground = np.asarray(foreground, dtype=bool)
    if (predicted.shape != (3, *IMAGE_SHAPE) or target.shape != predicted.shape
            or foreground.shape != predicted.shape or not np.isfinite(predicted).all()
            or not np.isfinite(target).all()):
        raise ValueError("Invalid phase metric shapes or nonfinite images")
    result = {}
    for index, phase in enumerate(PHASES):
        mask = np.asarray(foreground[index], dtype=bool)
        if not mask.any():
            raise ValueError("A phase has no foreground on the selected ROI grid")
        error = np.asarray(predicted[index], np.float32)[mask] - np.asarray(target[index], np.float32)[mask]
        result[phase] = {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean()))}
    return result


def build_generated_pillar_input(cfg, prediction, source_geometry, source_support, normalization):
    """Build all three pCR channels without opening any target-visit assets."""
    data, _ = pcr_helpers(cfg)
    prediction = np.asarray(prediction, dtype=np.float32)
    support = np.asarray(source_support, dtype=bool)
    if prediction.shape != (3, *IMAGE_SHAPE) or support.shape != IMAGE_SHAPE or not np.isfinite(prediction).all():
        raise ValueError("Invalid generated three-phase MRI or source-defined support")
    mean, std = normalization["mean"], normalization["std"]
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid generated-image intensity mapping")
    raw = prediction * std + mean
    raw[:, ~support] = 0
    spacing = source_geometry["spacing_xyz_mm"][::-1]
    volume = torch.stack([data.pillar_channel(raw[index], spacing) for index in range(3)])
    if tuple(volume.shape) != data.PILLAR_SHAPE or not torch.isfinite(volume).all():
        raise ValueError("Invalid generated-only Pillar input")
    return volume


def render_comparison(path, columns, *, z, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    real = next(iter(columns.values()))
    observed = np.asarray(real)[np.asarray(real) != 0]
    low, high = np.percentile(observed, [1, 99])
    fig, axes = plt.subplots(3, len(columns), figsize=(3.2 * len(columns), 9.2), squeeze=False)
    for row, phase in enumerate(PHASES):
        for col, (label, values) in enumerate(columns.items()):
            axes[row, col].imshow(values[row, z], cmap="gray", vmin=low, vmax=high)
            axes[row, col].set_title(f"{phase}: {label}", fontsize=9)
            axes[row, col].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def codec_report(cfg, inventory):
    rows = []
    limit = cfg["selection"]["codec_check_train_patients"]
    visits = [v for v in inventory["visits"] if v["pilot_case_index"] < limit]
    for visit in visits:
        arrays = cached_arrays(cfg, visit)
        errors = phase_errors(arrays["reconstruction"], arrays["images"], arrays["foreground"])
        coverage = {phase: support_coverage(arrays["source_support"], arrays["reference_support"][i])
                    for i, phase in enumerate(PHASES)}
        rows.append({"case_index": visit["pilot_case_index"], "stage": visit["visit"],
                     "phase_errors": errors, "source_grid_acquisition_coverage": coverage})
        if visit["visit"] in ("T0", "T3"):
            source = next(v for v in visits if v["visit_id"] == visit["grid_visit_id"])
            source_arrays = cached_arrays(cfg, source)
            z = int(np.argmax(source_arrays["foreground"][1].sum(axis=(1, 2))))
            render_comparison(Path(cfg["output_root"]) / "codec_check" / f"case_{visit['pilot_case_index']:03d}_{visit['visit']}.png",
                              {"Real": arrays["images"], "VQ reconstruction": arrays["reconstruction"]},
                              z=z, title=f"Development codec check, case {visit['pilot_case_index'] + 1}, {visit['visit']}")
    summary = {phase: {metric: float(np.mean([row["phase_errors"][phase][metric] for row in rows]))
                       for metric in ("mae", "rmse")} for phase in PHASES}
    result = {"schema": SCHEMA, "stage": "codec_check_complete", "training_patients": limit,
              "visits": len(rows), "phase_order": list(PHASES), "phase_summary": summary, "records": rows,
              "normalization": inventory["image_normalization"], "original_validation_loaded": False,
              "spatial_policy": cfg["spatial_policy"],
              "interpretation": "Small development reconstruction audit; no generator trained and no pCR performance evaluated"}
    write_json(Path(cfg["output_root"]) / "codec_check/report.json", result)
    return result
