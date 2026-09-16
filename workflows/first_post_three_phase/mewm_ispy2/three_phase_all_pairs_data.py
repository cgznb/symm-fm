"""Source-relative three-phase views for the original first-post patient split."""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import copy
import os
import signal
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .first_post_world_data import (
    disk_gate,
    identity,
    load_codec,
    public,
    read_json,
    reference_image,
    write_json,
)
from .three_phase_pilot_data import (
    CENTERED_GRID,
    PAIR_FIELDS,
    atomic_arrays,
    normalize_images,
    pcr_helpers,
    support_coverage,
    training_geometry,
)
from .three_phase_preparation import (
    PhasePrefetch,
    PhaseTransfer,
    grid_key,
    load_local_phase_sources,
    prepare_arrays,
)
from .three_phase_symmflow import (
    IMAGE_SHAPE,
    JOINT_LATENT_SHAPE,
    PHASES,
    SharedThreePhaseCodec,
)

SCHEMA = "three_phase_symmflow_all_pairs_v1"
T0_GRID_POLICY = "target_center_source_extent_preserve_t0_fallback"
GIB = 2**30


def load_config(path):
    cfg = _release_yaml(Path(path).read_text())
    if cfg["schema"] != SCHEMA or tuple(cfg["phase_order"]) != PHASES:
        raise ValueError("Not a formal all-pairs three-phase configuration")
    if cfg["source_policy"] != "all_forward_real_source" or cfg["initialization"] != "fresh_generator_frozen_codec":
        raise ValueError("Unsupported source or initialization policy")
    if cfg["spatial_policy"] not in ("target_center_with_source_extent", T0_GRID_POLICY):
        raise ValueError("Unsupported source-relative geometry")
    if bool(cfg.get("crop_overrides")) != (cfg["spatial_policy"] == T0_GRID_POLICY):
        raise ValueError("T0 crop overrides require the T0-preserving spatial policy")
    settings = cfg["training"]
    if settings["effective_batch_size"] % settings["microbatch"] or settings["max_optimizer_steps"] <= 0:
        raise ValueError("Invalid training batch or update count")
    if settings["full_validation_interval"] % settings["light_validation_interval"]:
        raise ValueError("Full validation must coincide with light validation")
    if settings["solver_steps"] != 20:
        raise ValueError("Formal evaluation is fixed to Euler20")
    return cfg


def validate_geometry(geometry):
    if tuple(geometry["shape_zyx"]) != IMAGE_SHAPE:
        raise ValueError("Unexpected ROI shape")
    spacing = np.asarray(geometry["spacing_xyz_mm"])
    direction = np.asarray(geometry["direction_lps"]).reshape(3, 3)
    if (not np.isfinite(spacing).all() or (spacing <= 0).any()
            or not np.isfinite(geometry["origin_lps_mm"]).all()
            or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4)):
        raise ValueError("Invalid physical ROI geometry")


def view_geometry(source, target, policy="target_center_with_source_extent"):
    validate_geometry(source["crop_geometry"])
    validate_geometry(target["crop_geometry"])
    if target.get("crop_localization") == "empty_mask_t0_mask_fallback":
        if policy != T0_GRID_POLICY:
            raise ValueError("T0 fallback extent must be preserved in training views")
        return copy.deepcopy(target["crop_geometry"])
    return training_geometry({"visit": target["visit"], "source_geometry": source["crop_geometry"],
                              "crop_geometry": target["crop_geometry"]}, CENTERED_GRID)


def apply_crop_overrides(world, overrides):
    if overrides.get("schema") != "first_post_t0_crop_overrides_v1" or overrides.get("verified") is not True:
        raise ValueError("T0 crop corrections have not been verified")
    if overrides["policy"] != {"empty_mask": "t0_mask", "reference_mapping": "physical_lps",
                                "outside_center": "exclude_visit", "unavailable_t0": "exclude_visit"}:
        raise ValueError("Unexpected T0 crop exclusion policy")
    if overrides["image_normalization"] != {"mean": world["normalization"]["image_mean"],
                                             "std": world["normalization"]["image_std"]}:
        raise ValueError("T0 crop intensity normalization differs from frozen codec inputs")
    rows = {}
    for category in ("records", "unchanged", "exclusions"):
        for row in overrides[category]:
            key = f"{row['patient_id']}_{row['visit']}_aqc1"
            if row["case_id"] != key or key in rows:
                raise ValueError("Duplicate or inconsistent T0 crop correction")
            rows[key] = category, row
    result = copy.deepcopy(world)
    result["visits"], result["crop_exclusions"] = [], []
    for original in world["visits"]:
        case = f"{original['patient_id']}_{original['visit']}_aqc1"
        if case not in rows:
            raise ValueError("T0 admission does not cover a generator visit")
        category, row = rows[case]
        if row["fold"] != original["split"]:
            raise ValueError("T0 correction changes the original patient split")
        if category == "exclusions":
            result["crop_exclusions"].append({"visit_id": original["visit_id"], "reason": row["reason"]})
            continue
        visit = copy.deepcopy(original)
        if category == "records":
            crop = row["crop"]
            if (original["crop_localization"] != "empty_mask_full_image_fallback"
                    or crop["localization"] != "empty_mask_t0_mask_fallback"
                    or crop["source_mask_voxels"] != 0 or crop["resampled_mask_voxels"] != 0
                    or crop["reference_mapping"] != "physical_lps"
                    or not crop["localization_center_in_acquisition"]):
                raise ValueError("Invalid T0 fallback crop replacement")
            visit.update(crop_geometry=copy.deepcopy(crop["geometry"]), crop_localization=crop["localization"],
                         crop_reference=copy.deepcopy(crop), image_path=row["cache_path"], image_identity=row["cache_identity"])
        elif original["crop_localization"] != "all_predicted_components_union":
            raise ValueError("An empty generator visit lacks a T0 replacement")
        visit["crop_dependencies"] = copy.deepcopy(row["dependencies"])
        result["visits"].append(visit)
    return result


def verify_crop_dependencies(overrides, native):
    root = Path(overrides["source_root"])
    segmentation = Path(overrides["preparation_config"]["data"]["tumor_crop"]["segmentation_root"])
    sources = {f"{r['patient_id']}_{r['visit']}_aqc1": root / r["relative_path"] for r in native["records"]}
    checked = set()
    for category in ("records", "unchanged", "exclusions"):
        for row in overrides[category]:
            pending = [(row["case_id"], row["dependencies"])]
            if "t0_reference" in row["dependencies"]:
                baseline = row["dependencies"]["t0_reference"]
                pending.append((baseline["case_id"], baseline))
            for case, recorded in pending:
                for key, path in (("source", sources[case]), ("mask", segmentation / "masks" / f"{case}.nii.gz"),
                                  ("report", segmentation / "case_reports" / f"{case}.json")):
                    token = (str(path), recorded[key]["size_bytes"], recorded[key]["mtime_ns"])
                    if token not in checked and identity(path) != recorded[key]:
                        raise ValueError("T0 crop dependency changed")
                    checked.add(token)


def pair_counts(pairs):
    return {split: dict(sorted(Counter(f"{p['earlier_stage']}->{p['later_stage']}"
                                      for p in pairs if p["split"] == split).items())) for split in ("train", "val")}


def build_inventory(world, native, metadata, aliases, spatial_policy="target_center_with_source_extent"):
    """Join phase metadata without requiring pCR labels or a complete T0 prefix."""
    rows = {}
    for _, row in metadata.iterrows():
        pid = aliases.get(str(row.pid), str(row.pid))
        if pid in rows:
            raise ValueError("Duplicate canonical phase metadata")
        rows[pid] = row
    native_index = {(r["patient_id"], r["visit"]): r for r in native["records"]}
    visits, exclusions, assignments = {}, copy.deepcopy(world.get("crop_exclusions", [])), {}
    for original in world["visits"]:
        pid, stage = original["canonical_patient_id"], original["visit"]
        if original["split"] not in ("train", "val") or assignments.setdefault(pid, original["split"]) != original["split"]:
            raise ValueError("Patient crosses generator splits")
        row = rows.get(pid)
        late = float(row.get(f"post_late_{stage}", np.nan)) if row is not None else np.nan
        pre = float(row.get(f"pre_{stage}", np.nan)) if row is not None else np.nan
        if not np.isfinite(late) or not np.isfinite(pre):
            exclusions.append({"visit_id": original["visit_id"], "reason": "missing_explicit_phase_metadata"})
            continue
        if pre != 0 or late <= 1 or not late.is_integer():
            raise ValueError("Invalid explicit pre/late phase selection")
        record = native_index[original["patient_id"], stage]
        remote = str(record["source_path"])
        if not remote.endswith("_aqc_1.nii.gz"):
            raise ValueError("Native source is not first-post aqc1")
        visit = copy.deepcopy(original)
        visit.update(late_index=int(late),
                     native_first_post=str(Path(native["destination_root"]) / record["relative_path"]),
                     remote_phases={"pre": remote.removesuffix("_aqc_1.nii.gz") + "_aqc_0.nii.gz",
                                    "late": remote.removesuffix("_aqc_1.nii.gz") + f"_aqc_{int(late)}.nii.gz"})
        validate_geometry(visit["crop_geometry"])
        visits[visit["visit_id"]] = visit
    pairs, views = [], {}
    indices = {key: index for index, key in enumerate(sorted(visits))}
    for original in world["pairs"]:
        source_id, target_id = original["earlier_visit_id"], original["later_visit_id"]
        if source_id not in visits or target_id not in visits:
            continue
        source, target = visits[source_id], visits[target_id]
        if (source["canonical_patient_id"] != target["canonical_patient_id"]
                or source["split"] != target["split"] or source["split"] != original["split"]
                or original["patient_id"] != source["canonical_patient_id"]
                or source["visit"] != original["earlier_stage"] or target["visit"] != original["later_stage"]
                or int(source["visit"][1:]) >= int(target["visit"][1:])):
            raise ValueError("Invalid forward longitudinal pair")
        pair = {key: copy.deepcopy(original.get(key)) for key in PAIR_FIELDS}
        pair.update(split=source["split"], pair_index=len(pairs))
        for name, visit in (("source", source), ("target", target)):
            key = f"v{indices[source_id]:04d}_v{indices[visit['visit_id']]:04d}"
            pair[f"{name}_view"] = key
            views.setdefault(key, {"view_id": key, "source_visit_id": source_id, "visit_id": visit["visit_id"],
                                   "patient_id": source["canonical_patient_id"], "split": source["split"],
                                   "geometry": view_geometry(source, visit, spatial_policy), "latent_file": f"latents/{key}.npy",
                                   "reference_file": f"references/{key}.npz" if source["split"] == "val" else None})
        pairs.append(pair)
    used = {v["visit_id"] for v in views.values()}
    if any(not any(p["split"] == split for p in pairs) for split in ("train", "val")):
        raise ValueError("Empty training or validation split")
    return {"schema": SCHEMA, "phase_order": list(PHASES), "visits": [visits[k] for k in sorted(used)],
            "views": list(views.values()), "pairs": pairs, "exclusions": exclusions,
            "candidate_pair_counts": pair_counts(world["pairs"]), "pair_counts": pair_counts(pairs),
            "image_normalization": {"mean": world["normalization"]["image_mean"],
                                    "std": world["normalization"]["image_std"]}}


def prepare_inventory(cfg):
    import pandas as pd
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    data, settings = pcr_helpers(cfg)
    metadata_path = data.repo_path(settings["metadata_csv"])
    bindings = {"world": identity(cfg["source_world_bundle"]), "codec": identity(cfg["codec_checkpoint"]),
                "native": identity(settings["native_manifest"]), "phases": identity(metadata_path),
                "aliases": identity(settings["identity_mapping"])}
    overrides = None
    if cfg.get("crop_overrides"):
        bindings["crop_overrides"] = identity(cfg["crop_overrides"])
        overrides = read_json(cfg["crop_overrides"])
        if overrides["source_manifest_identity"] != bindings["native"]:
            raise ValueError("T0 correction uses a different native inventory")
        verify_crop_dependencies(overrides, read_json(settings["native_manifest"]))
        for record in overrides["records"]:
            if identity(record["cache_path"]) != record["cache_identity"]:
                raise ValueError("Corrected T0 crop cache changed")
    path = root / "inventory.json"
    if path.exists():
        result = read_json(path)
        if result["schema"] != SCHEMA or result["source_files"] != bindings or result["config"] != public(cfg):
            raise ValueError("Existing inventory differs; use a new experiment directory")
        return result
    mapping = pd.read_excel(settings["identity_mapping"])
    aliases = {str(r["TCIA PATIENT ID"]): f"ISPY2-{int(r['I-SPY 2 Research ID'])}"
               for _, r in mapping.iterrows() if pd.notna(r["I-SPY 2 Research ID"])}
    world, native = read_json(cfg["source_world_bundle"]), read_json(settings["native_manifest"])
    if (world.get("registered") is not False or world.get("phase_index") != 1
            or not world.get("verified_for_world_training") or native.get("registered") is not False
            or native.get("phase_index") != 1):
        raise ValueError("Formal inputs must be verified native first-post data")
    if overrides is not None:
        world = apply_crop_overrides(world, overrides)
    result = build_inventory(world, native, pd.read_csv(metadata_path), aliases, cfg["spatial_policy"])
    unavailable = set()
    for visit in result["visits"]:
        meta = read_json(visit["metadata_path"])
        if not {0, 1, visit["late_index"]}.issubset(set(meta["dce_phase_ids"])):
            unavailable.add(visit["visit_id"])
            result["exclusions"].append({"visit_id": visit["visit_id"], "reason": "phase_absent_in_native_inventory"})
        visit["native_first_post_identity"] = identity(visit["native_first_post"])
        if overrides is not None and (visit["native_first_post_identity"] != visit["crop_dependencies"]["source"]
                                      or identity(visit["mask_path"]) != visit["crop_dependencies"]["mask"]):
            raise ValueError("T0 correction source MRI or segmentation changed")
    result["pairs"] = [p for p in result["pairs"] if not {p["earlier_visit_id"], p["later_visit_id"]} & unavailable]
    prune_inventory(result)
    sizes = data.remote_phase_sizes(settings, result["visits"])
    for visit, values in zip(result["visits"], sizes, strict=True):
        visit["phase_bytes"] = values
    result.update(source_files=bindings, config=public(cfg), pair_counts=pair_counts(result["pairs"]))
    write_json(path, result)
    return result


def prune_inventory(inventory):
    used_views = {p[key] for p in inventory["pairs"] for key in ("source_view", "target_view")}
    inventory["views"] = [v for v in inventory["views"] if v["view_id"] in used_views]
    used_visits = {v["visit_id"] for v in inventory["views"]}
    inventory["visits"] = [v for v in inventory["visits"] if v["visit_id"] in used_visits]


def pack_latent(value):
    value = np.asarray(value, np.float32)
    if value.shape != JOINT_LATENT_SHAPE or not np.isfinite(value).all():
        raise ValueError("Invalid three-phase latent")
    compact = value.astype(np.float16)
    if not np.array_equal(value, compact.astype(np.float32)):
        raise ValueError("BF16 latent cannot round-trip exactly through FP16 storage")
    return compact


def atomic_latent(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, pack_latent(value), allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_latent(root, view):
    value = np.load(Path(root) / view["latent_file"], allow_pickle=False)
    if value.shape != JOINT_LATENT_SHAPE or value.dtype != np.float16 or not np.isfinite(value).all():
        raise ValueError("Invalid formal latent cache")
    return value.astype(np.float32)


def write_reference(path, images, foreground, support):
    atomic_arrays(path, images=images, foreground=np.packbits(foreground.reshape(-1)),
                  support=np.packbits(support.reshape(-1)))


def read_reference(root, view):
    with np.load(Path(root) / view["reference_file"], allow_pickle=False) as archive:
        images = archive["images"]
        count = int(np.prod((3, *IMAGE_SHAPE)))
        foreground = np.unpackbits(archive["foreground"], count=count).reshape(3, *IMAGE_SHAPE).astype(bool)
        support = np.unpackbits(archive["support"], count=count).reshape(3, *IMAGE_SHAPE).astype(bool)
    if images.shape != (3, *IMAGE_SHAPE) or images.dtype != np.float32 or not np.isfinite(images).all():
        raise ValueError("Invalid validation reference")
    if not all(mask.any() for mask in foreground) or not all(mask.any() for mask in support):
        raise ValueError("Empty phase in validation reference")
    return {"images": images, "foreground": foreground, "support": support}


def storage_requirement(inventory):
    voxels = int(np.prod(IMAGE_SHAPE))
    latent = len(inventory["views"]) * int(np.prod(JOINT_LATENT_SHAPE)) * 2
    references = sum(v["split"] == "val" for v in inventory["views"]) * (3 * voxels * 4 + 2 * ((3 * voxels + 7) // 8))
    patients = defaultdict(int)
    for visit in inventory["visits"]:
        patients[visit["canonical_patient_id"]] += sum(visit["phase_bytes"].values())
    return {"latent_bytes": latent, "reference_bytes_upper_bound": references,
            "staging_bytes": max(patients.values(), default=0), "checkpoint_and_preview_bytes": 6 * GIB}


def admit_inventory(inventory, reports):
    excluded = {r["patient_id"] for r in reports if r["excluded_training_patient"]}
    result = copy.deepcopy(inventory)
    result["pairs"] = [p for p in result["pairs"] if p["patient_id"] not in excluded]
    prune_inventory(result)
    counts = pair_counts(result["pairs"])
    if not all(counts.values()):
        raise ValueError("Geometry admission leaves an empty split")
    result.update(pair_counts=counts, geometry_excluded_training_patients=sorted(excluded))
    return result


def _resample(image, geometry, *, support=False):
    import SimpleITK as sitk
    if support:
        source = sitk.Image(image.GetSize(), sitk.sitkUInt8) + 1
        source.CopyInformation(image)
    else:
        source = image
    result = sitk.Resample(source, reference_image(geometry), sitk.Transform(3, sitk.sitkIdentity),
                           sitk.sitkNearestNeighbor if support else sitk.sitkLinear, 0.0)
    return sitk.GetArrayFromImage(result)


def reuse_prepared_patients(cfg, inventory):
    """Reuse complete patient caches only when every retained view is identical."""
    if not cfg.get("reuse_preparation_from"):
        return
    root, previous = Path(cfg["output_root"]), Path(cfg["reuse_preparation_from"])
    if root.resolve() == previous.resolve():
        raise ValueError("Corrected preparation must use a new output directory")
    old = read_json(previous / "inventory.json")
    if (old["schema"] != inventory["schema"] or old["image_normalization"] != inventory["image_normalization"]
            or any(inventory["source_files"].get(k) != v for k, v in old["source_files"].items())
            or old["config"]["quality"] != cfg["quality"]):
        raise ValueError("Previous preparation has different sources, codec or normalization")
    old_views = {(v["source_visit_id"], v["visit_id"]): v for v in old["views"]}
    old_visits = {v["visit_id"]: v for v in old["visits"]}
    old_reports = {r["patient_id"]: r for p in (previous / "prepared").glob("*.json") for r in [read_json(p)]}
    groups = defaultdict(list)
    for view in inventory["views"]:
        groups[view["patient_id"]].append(view)
    new_visits = {v["visit_id"]: v for v in inventory["visits"]}
    reused = []
    for index, (pid, views) in enumerate(sorted(groups.items())):
        marker = root / "prepared" / f"patient_{index:04d}.json"
        report = old_reports.get(pid)
        if marker.exists() or report is None or report["excluded_training_patient"]:
            continue
        matches = [old_views.get((v["source_visit_id"], v["visit_id"])) for v in views]
        if any(o is None or o["geometry"] != v["geometry"] or o["split"] != v["split"]
               for o, v in zip(matches, views, strict=True)):
            continue
        visit_ids = {v["visit_id"] for v in views}
        keys = ("native_first_post_identity", "remote_phases", "metadata_identity", "crop_geometry")
        if any(any(old_visits[k].get(field) != new_visits[k].get(field) for field in keys) for k in visit_ids):
            continue
        if report["inventory"] != identity(previous / "inventory.json"):
            raise ValueError("Previous preparation inventory changed")
        files, coverage = {}, []
        old_coverage = {r["view_id"]: r for r in report["views"]}
        for view, old_view in zip(views, matches, strict=True):
            for field in ("latent_file", "reference_file"):
                if not view[field]:
                    continue
                name = old_view[field]
                expected = report["files"][name]
                if identity(previous / name) != expected:
                    raise ValueError("Previous preparation cache changed")
                destination = root / view[field]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    os.link(previous / name, destination)
                if identity(destination) != expected:
                    raise ValueError("Reused preparation cache differs")
                files[view[field]] = expected
            coverage.append({**old_coverage[old_view["view_id"]], "view_id": view["view_id"]})
        write_json(marker, {"patient_id": pid, "split": views[0]["split"], "excluded_training_patient": False,
                            "minimum_coverage": min(v["minimum"] for v in coverage), "views": coverage, "files": files,
                            "inventory": identity(root / "inventory.json"), "reused_from": str(previous)})
        reused.append({"patient_id": pid, "files": len(files), "bytes": sum(v["size_bytes"] for v in files.values())})
    if reused:
        write_json(root / "reused_preparation.json", {"source": str(previous), "patients": len(reused),
                                                    "bytes": sum(r["bytes"] for r in reused), "records": reused})


def _prepare_patient(cfg, inventory, views, visits, codec, data, settings, stopped):
    root = Path(cfg["output_root"])
    arrays, rows, timing = prepare_arrays(data, settings, views, visits, _resample, normalize_images,
                                          inventory["image_normalization"], support_coverage)
    coverage = [{"view_id": row["view_id"], "coverage": dict(zip(PHASES, row["values"], strict=True)),
                 "minimum": row["minimum"]} for row in rows]
    minimum = min(row["minimum"] for row in coverage)
    excluded = views[0]["split"] == "train" and minimum < cfg["quality"]["minimum_source_support_coverage"]
    files, latents = {}, {}
    timing.update(encode_seconds=0.0, write_seconds=0.0, encoded_visit_grids=0)
    if not excluded:
        for view in views:
            if stopped.is_set():
                raise InterruptedError("Preparation pause requested")
            key = grid_key(view)
            normalized, foreground, support = arrays[key]
            disk_gate(root, required=GIB, reserve_gib=cfg["runtime"]["reserve_gib"])
            if key not in latents:
                began = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    latents[key] = codec.encode(torch.from_numpy(normalized)[None].to("cuda:0")).float()[0].cpu().numpy()
                timing["encode_seconds"] += time.perf_counter() - began
            began = time.perf_counter()
            atomic_latent(root / view["latent_file"], latents[key])
            files[view["latent_file"]] = identity(root / view["latent_file"])
            if view["reference_file"]:
                write_reference(root / view["reference_file"], normalized, foreground, support)
                files[view["reference_file"]] = identity(root / view["reference_file"])
            timing["write_seconds"] += time.perf_counter() - began
    if stopped.is_set():
        raise InterruptedError("Preparation pause requested")
    timing["encoded_visit_grids"] = len(latents)
    return {"patient_id": views[0]["patient_id"], "split": views[0]["split"], "excluded_training_patient": excluded,
            "minimum_coverage": minimum, "views": coverage, "files": files, "timing": timing,
            "inventory": identity(root / "inventory.json")}


def prepare(cfg, inventory, progress):
    import SimpleITK as sitk
    root = Path(cfg["output_root"])
    reserve = cfg["runtime"]["reserve_gib"]
    reuse_prepared_patients(cfg, inventory)
    visits = {v["visit_id"]: v for v in inventory["visits"]}
    groups = defaultdict(list)
    for view in inventory["views"]:
        groups[view["patient_id"]].append(view)
    reports, jobs = {}, []
    inventory_identity = identity(root / "inventory.json")
    for index, (pid, views) in enumerate(sorted(groups.items())):
        marker = root / "prepared" / f"patient_{index:04d}.json"
        patient_visits = [visits[k] for k in sorted({v["visit_id"] for v in views})]
        for visit in patient_visits:
            if identity(visit["native_first_post"]) != visit["native_first_post_identity"]:
                raise ValueError("Native first-post input changed")
        if marker.exists():
            report = read_json(marker)
            if report["inventory"] != inventory_identity or report["patient_id"] != pid:
                raise ValueError("Prepared patient contract changed")
            for name, recorded in report["files"].items():
                if identity(root / name) != recorded:
                    raise ValueError("Prepared cache changed")
            reports[pid] = report
            continue
        jobs.append((index, pid, views, patient_visits))
    data, settings = pcr_helpers(cfg)
    settings["local_phase_sources"] = load_local_phase_sources(settings, inventory)
    unfinished = {"views": [v for _, _, views, _ in jobs for v in views],
                  "visits": [v for _, _, _, patient_visits in jobs for v in patient_visits]}
    estimate = storage_requirement(unfinished)
    staging = [sum(visit["phase_bytes"][phase] for visit in patient_visits
                   for phase, remote in visit["remote_phases"].items()
                   if str(Path(remote).relative_to(settings["remote_root"])) not in settings["local_phase_sources"])
               for _, _, _, patient_visits in jobs]
    estimate["staging_bytes"] = sum(sorted(staging, reverse=True)[:2])
    disk_gate(root, required=sum(estimate.values()), reserve_gib=reserve)
    write_json(root / "storage_admission.json", {**estimate, "reserve_gib": reserve, "passed": True,
               "scope": "unfinished_patients", "completed_patients": len(reports), "remaining_patients": len(jobs),
               "maximum_staged_patients": min(2, len(jobs))})
    reused = len(reports)
    progress({"stage": "preparing", "completed_patients": reused, "total_patients": len(groups),
              "reused_prepared_patients": reused, "prepared_this_session": 0})
    stopped, codec = threading.Event(), None
    handlers = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if jobs:
            torch.set_num_threads(cfg["runtime"]["cpu_threads"])
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(cfg["runtime"]["cpu_threads"])
            codec = SharedThreePhaseCodec(load_codec(cfg["codec_checkpoint"], "cuda:0"))
            started = time.perf_counter()
            with PhaseTransfer(data, settings) as transfer, PhasePrefetch(transfer, jobs, stopped) as prefetch:
                for position, (index, pid, views, patient_visits) in enumerate(jobs):
                    began = time.perf_counter()
                    timing = prefetch.take(position)
                    try:
                        report = _prepare_patient(cfg, inventory, views, visits, codec, data, settings, stopped)
                        report["timing"].update(timing, total_seconds=time.perf_counter() - began)
                        write_json(root / "prepared" / f"patient_{index:04d}.json", report)
                        reports[pid] = report
                    finally:
                        data.release_phases(settings, patient_visits)
                    progress({"stage": "preparing", "completed_patients": len(reports), "total_patients": len(groups),
                              "reused_prepared_patients": reused, "prepared_this_session": position + 1,
                              "last_patient_timing": report["timing"],
                              "eta_seconds": (time.perf_counter() - started) / (position + 1) * (len(jobs) - position - 1)})
        if stopped.is_set():
            raise InterruptedError("Preparation pause requested")
    except InterruptedError:
        if not stopped.is_set():
            raise
        progress({"stage": "paused", "reason": "preparation_signal", "completed_patients": len(reports),
                  "total_patients": len(groups), "optimizer_updates": 0})
        return None
    finally:
        del codec
        torch.cuda.empty_cache()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    reports = [reports[pid] for pid in sorted(groups)]
    admitted = admit_inventory(inventory, reports)
    admitted_path = root / "admitted_inventory.json"
    if admitted_path.exists() and read_json(admitted_path) != public(admitted):
        raise ValueError("Existing data admission differs")
    if not admitted_path.exists():
        write_json(admitted_path, admitted)
    files = {name: value for report in reports for name, value in report["files"].items()}
    write_json(root / "preparation.json", {"passed": True, "schema": SCHEMA, "pair_counts": admitted["pair_counts"],
               "patients": {s: len({p["patient_id"] for p in admitted["pairs"] if p["split"] == s}) for s in ("train", "val")},
               "files": files, "reports": reports, "validation_filtered_by_coverage": False})
    progress({"stage": "preparation_complete", "views": len(admitted["views"]), "pair_counts": admitted["pair_counts"]})
    return admitted


def fit_statistics(inventory, reader):
    total, square, count = np.zeros(24, np.float64), np.zeros(24, np.float64), 0
    seen, patients = set(), set()
    for view in inventory["views"]:
        if view["split"] != "train" or view["view_id"] in seen:
            continue
        value = np.asarray(reader(view), np.float64).reshape(24, -1)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite training latent")
        total += value.sum(axis=1)
        square += np.square(value).sum(axis=1)
        count += value.shape[1]
        seen.add(view["view_id"])
        patients.add(view["patient_id"])
    if not count:
        raise ValueError("No training latent views")
    mean = total / count
    std = np.sqrt(np.maximum(0, square / count - mean**2))
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Degenerate latent normalization")
    return {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "train", "unique_views": len(seen),
            "patients": len(patients), "phase_order": list(PHASES)}


class PairDataset(Dataset):
    def __init__(self, root, inventory, statistics, split):
        self.root = Path(root)
        self.records = [p for p in inventory["pairs"] if p["split"] == split]
        self.views = {v["view_id"]: v for v in inventory["views"]}
        self.mean = torch.tensor(statistics["mean"], dtype=torch.float32).view(24, 1, 1, 1)
        self.std = torch.tensor(statistics["std"], dtype=torch.float32).view(24, 1, 1, 1)

    def source(self, record):
        return (torch.from_numpy(read_latent(self.root, self.views[record["source_view"]])) - self.mean) / self.std

    def __getitem__(self, index):
        record = self.records[index]
        target = torch.from_numpy(read_latent(self.root, self.views[record["target_view"]]))
        return {"source": self.source(record), "target": (target - self.mean) / self.std, "record": record}

    def __len__(self):
        return len(self.records)

    def denormalize(self, value):
        return value * self.std.to(value.device)[None] + self.mean.to(value.device)[None]
