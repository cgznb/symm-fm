"""Registered DCE0 crops localized by native first-post model T0 masks."""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import argparse
import csv
import json
import math
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import yaml
from nibabel.spaces import vox2out_vox
from scipy.ndimage import affine_transform, label, map_coordinates
from torch.utils.data import Dataset

SCHEMA = "registered_dce0_roi32_firstpostmask_v1"
SHAPE = (32, 128, 128)
SPACING = np.array([2.0, 0.7032, 0.7032])
REVERSE = np.eye(4)[[2, 1, 0, 3]]
LPS_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def read_json(path):
    return json.loads(Path(path).read_text())


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def verify_identity(identity):
    if file_identity(identity["path"]) != identity:
        raise ValueError(f"Source file changed: {identity['path']}")


def read_config(path):
    config = _release_yaml(Path(path).read_text())
    if config.get("schema") != SCHEMA:
        raise ValueError("Unsupported ROI32 experiment")
    data = config["data"]
    if tuple(data["shape_zyx"]) != SHAPE or not np.array_equal(data["spacing_zyx_mm"], SPACING):
        raise ValueError("This experiment requires fixed ZYX 32x128x128 and original spacing")
    if data["crop_policy"] != "largest_native_26_component_t0_bbox_center":
        raise ValueError("Unexpected T0 localization policy")
    if config["runtime"]["precision"] != "bf16":
        raise ValueError("This experiment requires BF16 with FP32 quantization")
    return config


def check_disk(config, extra_bytes=0):
    root = Path(config["output_dir"])
    while not root.exists():
        root = root.parent
    reserve = config["runtime"]["disk_reserve_gib"] * 1024**3
    if shutil.disk_usage(root).free < reserve + extra_bytes:
        raise RuntimeError("Insufficient free disk above the configured reserve")


def visit_filename(visit_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+:T[0-3]", visit_id):
        raise ValueError("Invalid visit identifier")
    return visit_id.replace(":", "_") + ".npz"


def acquisition_affine(meta):
    """Use saved acquisition geometry, never the MRI file's placeholder affine."""
    orientation = np.asarray(meta["image_orientation_patient"], dtype=float)
    direction = np.column_stack((orientation[:3], orientation[3:], np.cross(orientation[:3], orientation[3:])))
    spacing = np.array([meta["pixel_spacing"][1], meta["pixel_spacing"][0], meta["spacing_between_slices"]], dtype=float)
    origin = np.asarray(meta["image_position_patient_first"], dtype=float)
    if not np.isfinite(direction).all() or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError("Invalid saved acquisition orientation")
    if not np.isfinite(spacing).all() or np.any(spacing <= 0) or not np.isfinite(origin).all():
        raise ValueError("Invalid saved acquisition spacing/origin")
    result = np.eye(4)
    result[:3, :3] = direction @ np.diag(spacing)
    result[:3, 3] = origin
    return result


def centered_affine(geometry):
    direction = np.asarray(geometry["direction"]).reshape(3, 3)
    spacing = np.asarray(geometry["spacing_xyz"])
    result = np.eye(4)
    result[:3, :3] = direction @ np.diag(spacing)
    result[:3, 3] = -direction @ ((np.asarray(geometry["shape_zyx"][::-1]) - 1) * spacing / 2)
    return result


def linear_affine(transform):
    if not transform.IsLinear():
        raise ValueError("The T0 phase transform must be linear")
    result = np.eye(4)
    result[:3, 3] = transform.TransformPoint((0.0, 0.0, 0.0))
    for axis in range(3):
        result[:3, axis] = np.asarray(transform.TransformPoint(tuple(np.eye(3)[axis]))) - result[:3, 3]
    return result


def image_affine(image):
    result = np.eye(4)
    result[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    result[:3, 3] = image.GetOrigin()
    return result


def largest_region(data, mask_to_strict_xyz):
    coordinates = np.argwhere(data > 0)
    if not len(coordinates):
        return None
    lo, hi = coordinates.min(axis=0), coordinates.max(axis=0) + 1
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
    components, count = label(data[slices] > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
    ids = components[tuple((coordinates - lo).T)]
    sizes = np.bincount(ids)
    sizes[0] = 0
    chosen = int(np.argmax(sizes))
    major = ids == chosen
    points = (coordinates[:, ::-1] @ mask_to_strict_xyz[:3, :3].T + mask_to_strict_xyz[:3, 3])[:, ::-1]
    half = np.abs(mask_to_strict_xyz[:3, :3]).sum(axis=1)[::-1] / 2
    low, high = points[major].min(axis=0) - half, points[major].max(axis=0) + half
    center = (low + high) / 2
    extent = (high - low) * SPACING
    all_low, all_high = points.min(axis=0) - half, points.max(axis=0) + half
    all_extent = 2 * np.maximum(np.abs(all_low - center), np.abs(all_high - center)) * SPACING
    inside = np.all(np.abs(points - center) <= np.asarray(SHAPE) / 2 + 1e-7, axis=1)
    major_data = np.zeros(data.shape, dtype=np.uint8)
    major_data[slices] = components == chosen
    return {"center": center, "largest": major_data, "components": int(count),
            "largest_label": chosen, "source_voxels": len(coordinates), "largest_voxels": int(major.sum()),
            "largest_extent_mm": extent.tolist(), "all_required_extent_mm": all_extent.tolist(),
            "largest_clipped": bool(np.any(extent > np.asarray(SHAPE) * SPACING + 1e-6)),
            "all_clipped": bool(np.any(all_extent > np.asarray(SHAPE) * SPACING + 1e-6)),
            "largest_source_center_retention": float(inside[major].mean()),
            "all_source_center_retention": float(inside.mean())}


def resample_array(data, output_to_input_zyx, order):
    return affine_transform(data, output_to_input_zyx[:3, :3], output_to_input_zyx[:3, 3],
                            output_shape=SHAPE, order=order, mode="constant", cval=0, prefilter=False)


def load_mri_payload(path, meta):
    image = nib.load(str(path))
    shape = (int(meta["n_slices"]), int(meta["rows"]), int(meta["cols"]))
    if tuple(image.shape) != shape:
        raise ValueError(f"Stored ZYX payload disagrees with saved metadata: {path}")
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite source MRI: {path}")
    return data


def baseline_conditions(row):
    def category(value):
        if value in (None, "", "nan", "NaN"):
            return None
        if value in ("0.0", "1.0"):
            return str(int(float(value)))
        return str(value)
    raw_age = row.get("Age_at_Screening")
    age = float(raw_age) if raw_age not in (None, "", "nan", "NaN") else None
    if age is not None and not math.isfinite(age):
        age = None
    return {"hr_status": category(row.get("HR")), "her2_status": category(row.get("HER2")),
            "mammaprint": category(row.get("MP")), "menopausal_status": category(row.get("menopausal_status")),
            "age": age}


def connected_pairs(rows, visits):
    by_visit = {row["visit_id"]: row for row in visits}
    edges = defaultdict(dict)
    for row in rows:
        if row["source_visit_id"] not in by_visit or row["target_visit_id"] not in by_visit:
            continue
        earlier, later = by_visit[row["source_visit_id"]], by_visit[row["target_visit_id"]]
        index = int(row["source_visit"][1:])
        if int(row["target_visit"][1:]) != index + 1 or int(row["delta_days"]) <= 0:
            raise ValueError("Invalid original adjacent edge")
        if earlier["patient_id"] != later["patient_id"] or earlier["fold"] != later["fold"]:
            raise ValueError("Pair crosses patient or split")
        if index in edges[row["patient_id"]]:
            raise ValueError("Duplicate adjacent edge")
        edges[row["patient_id"]][index] = row
    results = []
    for patient, chain in sorted(edges.items()):
        for start in range(3):
            for stop in range(start + 1, 4):
                part = [chain.get(index) for index in range(start, stop)]
                if not all(part):
                    continue
                earlier = by_visit[part[0]["source_visit_id"]]
                later = by_visit[part[-1]["target_visit_id"]]
                for edge in part:
                    for key in ("source_visit_id", "target_visit_id"):
                        visit = by_visit[edge[key]]
                        if (baseline_conditions(visit) != baseline_conditions(earlier)
                                or visit["trial_arm"] != earlier["trial_arm"]):
                            raise ValueError("Baseline clinical conditions change within a patient")
                results.append({"pair_id": f"{patient}:T{start}->T{stop}", "patient_id": patient,
                                "split": earlier["fold"], "earlier_stage": f"T{start}", "later_stage": f"T{stop}",
                                "earlier_visit_id": earlier["visit_id"], "later_visit_id": later["visit_id"],
                                "delta_days": sum(int(edge["delta_days"]) for edge in part),
                                "interval_missing": False, "interval_source": f"source_bundle:{earlier['visit_date_source']}",
                                "baseline_clinical": baseline_conditions(earlier),
                                "treatment": {"treatment_arm": earlier["trial_arm"]},
                                "edge_ids": [edge["transition_id"] for edge in part]})
    return results


def build_inventory(config):
    data = config["data"]
    bundle, audit, segment = (Path(data[key]) for key in ("bundle_dir", "geometry_audit_dir", "segmentation_dir"))
    split = read_json(bundle / "split.json")
    if set(split) != {"train", "val"} or set(split["train"]) & set(split["val"]):
        raise ValueError("Expected disjoint original train/validation patients")
    mappings = {r["registered_patient_id"]: r for r in read_json(audit / "registered_new_mask_mapping_inventory.json")["patients"]}
    masks = {(r["patient_id"], r["visit"]): r for r in read_json(segment / "inventory.json")["records"]}
    geometry = read_json(audit / "registered_new_mask_patient_geometry.json")
    excluded = {r["patient_id"] for r in geometry["excluded"]}
    if len(excluded) != 1 or not excluded <= set(split["train"]):
        raise ValueError("Expected exactly one originally empty training T0")
    with (bundle / "visits.csv").open() as handle:
        raw_visits = list(csv.DictReader(handle))
    with (bundle / "transitions.csv").open() as handle:
        raw_edges = list(csv.DictReader(handle))
    patients = []
    source_geometry = {r["patient_id"]: r for r in geometry["new_prediction"]}
    with (audit / "largest_component_crop_sizes.csv").open() as handle:
        extents = {r["patient_id"]: r for r in csv.DictReader(handle)}
    for fold, ids in split.items():
        for patient in sorted(ids):
            mapping = mappings[patient]
            record = masks[mapping["native_patient_id"], "T0"]
            mask_path = segment / "masks" / (record["case_id"] + ".nii.gz")
            if patient in excluded:
                mask_image = sitk.ReadImage(str(mask_path))
                if np.count_nonzero(sitk.GetArrayViewFromImage(mask_image)):
                    raise ValueError("Previously empty T0 mask is no longer empty")
                continue
            phase = Path(source_geometry[patient]["transform_path"])
            if not phase.is_file():
                phase = Path(mapping["phase_transform_path"])
            patients.append({"patient_id": patient, "native_patient_id": mapping["native_patient_id"], "fold": fold,
                             "t0_metadata": file_identity(mapping["meta_path"]), "native_metadata": file_identity(record["metadata_source"]),
                             "t0_mask": file_identity(mask_path), "phase_transform": file_identity(phase),
                             "expected_mask_to_strict_xyz": source_geometry[patient]["mask_to_strict_index_xyz"],
                             "expected_center_zyx": json.loads(extents[patient]["center_strict_zyx"]),
                             "expected_mask_voxels": source_geometry[patient]["new_mask_native_voxels"]})
    visits = []
    for row in raw_visits:
        if row["patient_id"] in excluded:
            continue
        if row["patient_id"] not in split[row["fold"]] or row["quality_pass"] != "True":
            raise ValueError("Original visit QC/split mismatch")
        mapping = mappings[row["patient_id"]]
        record = masks.get((mapping["native_patient_id"], row["visit"]))
        visits.append({**{key: row[key] for key in ("patient_id", "visit_id", "visit", "fold", "trial_arm", "visit_date_source", "visit_date",
                                                     "HR", "HER2", "MP", "Age_at_Screening", "menopausal_status")},
                       "image_source": file_identity(row["dce0_path"]), "metadata_source": file_identity(row["meta_path"]),
                       "model_mask_path": str(segment / "masks" / (record["case_id"] + ".nii.gz")) if record else None,
                       "mask_native_metadata_path": record["metadata_source"] if record else None})
    if len({v["visit_id"] for v in visits}) != len(visits):
        raise ValueError("Duplicate unique visits")
    pairs = connected_pairs(raw_edges, visits)
    for name, items, key in (("patients", patients, "fold"), ("visits", visits, "fold"), ("pairs", pairs, "split")):
        if dict(Counter(r[key] for r in items)) != data[f"expected_{name}"]:
            raise ValueError(f"Unexpected {name} counts")
    sources = [bundle / name for name in ("split.json", "visits.csv", "transitions.csv")]
    sources += [segment / "inventory.json", audit / "registered_new_mask_patient_geometry.json",
                audit / "largest_component_crop_sizes.csv", audit / "registered_new_mask_mapping_inventory.json"]
    return {"schema": SCHEMA, "data_configuration": data, "sources": [file_identity(p) for p in sources],
            "patients": patients, "visits": visits, "pairs": pairs,
            "exclusions": [{"patient_id": p, "reason": "originally_empty_model_T0"} for p in sorted(excluded)],
            "image_phase": "registered_dce0", "mask_source": "first_post_model_prediction", "test_split": None}


def prepare_patient(task):
    config, patient, visits = task
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    root = Path(config["output_dir"]) / "data"
    marker = root / "patients" / (patient["patient_id"] + ".json")
    identities = [patient[k] for k in ("t0_metadata", "native_metadata", "t0_mask", "phase_transform")]
    identities += [v[k] for v in visits for k in ("image_source", "metadata_source")]
    for identity in identities:
        verify_identity(identity)
    if marker.exists():
        result = read_json(marker)
        if result["source_identities"] != identities or result["patient_contract"] != patient:
            raise ValueError("Prepared patient sources changed")
        for identity in result["cache_identities"]:
            verify_identity(identity)
        return result
    check_disk(config, 32 * 1024**2)
    t0_meta = read_json(patient["t0_metadata"]["path"])
    original_meta = read_json(patient["native_metadata"]["path"])
    if t0_meta["registration"]["status"] != "fixed_reference":
        raise ValueError("T0 must be the saved fixed reference")
    native_ras = LPS_RAS @ acquisition_affine(t0_meta)
    shape = tuple(t0_meta["target_geometry"]["shape_zyx"])
    _, strict_ras = vox2out_vox((shape[::-1], native_ras), voxel_sizes=SPACING[::-1])
    phase = sitk.ReadTransform(patient["phase_transform"]["path"])
    strict_to_mask_lps = (acquisition_affine(original_meta) @ np.linalg.inv(centered_affine(t0_meta["source_geometry"]))
                          @ linear_affine(phase) @ centered_affine(t0_meta["target_geometry"]) @ np.linalg.inv(native_ras) @ strict_ras)
    mask_image = sitk.ReadImage(patient["t0_mask"]["path"])
    mask = sitk.GetArrayFromImage(mask_image).astype(np.uint8)
    # Independently recreate the native LPS mask grid from original metadata.
    reference = sitk.GetImageFromArray(np.zeros((int(original_meta["n_slices"]), int(original_meta["rows"]), int(original_meta["cols"])), dtype=np.uint8))
    original_affine = acquisition_affine(original_meta)
    original_spacing = np.linalg.norm(original_affine[:3, :3], axis=0)
    reference.SetSpacing(tuple(original_spacing))
    reference.SetDirection(tuple((original_affine[:3, :3] / original_spacing).ravel()))
    reference.SetOrigin(tuple(original_affine[:3, 3]))
    reference = sitk.DICOMOrient(reference, "LPS")
    if mask_image.GetSize() != reference.GetSize() or not np.allclose(image_affine(mask_image), image_affine(reference), atol=2e-5):
        raise ValueError("Model mask does not match saved native acquisition geometry")
    to_strict = np.linalg.inv(strict_to_mask_lps) @ image_affine(mask_image)
    if not np.allclose(to_strict, patient["expected_mask_to_strict_xyz"], atol=1e-6):
        raise ValueError("Recomputed T0 mapping differs from the verified geometry audit")
    major = largest_region(mask, to_strict)
    if major is None or major["source_voxels"] != patient["expected_mask_voxels"]:
        raise ValueError("T0 mask contents changed")
    center = major["center"]
    if not np.allclose(center, patient["expected_center_zyx"], atol=1e-6):
        raise ValueError("Largest-component crop center changed")
    start = center - (np.asarray(SHAPE) - 1) / 2
    translation = np.eye(4)
    translation[:3, 3] = start[::-1]
    crop_ras = strict_ras @ translation
    to_mask = REVERSE @ np.linalg.inv(to_strict) @ translation @ REVERSE
    cropped_mask = resample_array(mask, to_mask, 0)
    cropped_largest = resample_array(major["largest"], to_mask, 0)
    if not cropped_largest.any():
        raise ValueError(f"Nonempty T0 largest component becomes empty: {patient['patient_id']}")
    mask_file = root / "patients" / (patient["patient_id"] + ".npz")
    save_npz(mask_file, mask=cropped_mask > 0, largest=cropped_largest > 0)
    result = {"patient_contract": patient, "source_identities": identities,
              "patient_id": patient["patient_id"], "fold": patient["fold"],
              "center_strict_zyx": center.tolist(), "start_strict_zyx": start.tolist(),
              "crop_affine_ras": crop_ras.tolist(), "crop_affine_lps": (LPS_RAS @ crop_ras).tolist(),
              "shape_zyx": list(SHAPE), "spacing_zyx_mm": SPACING.tolist(),
              "mask_to_strict_xyz": to_strict.tolist(), "actual_mask_voxels": int(cropped_mask.sum()),
              "actual_largest_voxels": int(cropped_largest.sum()),
              **{k: v for k, v in major.items() if k not in ("center", "largest")},
              "visits": [], "cache_identities": [file_identity(mask_file)]}
    for visit in visits:
        meta = read_json(visit["metadata_source"]["path"])
        if meta["target_geometry"] != t0_meta["target_geometry"] or not np.allclose(acquisition_affine(meta), acquisition_affine(t0_meta), atol=1e-6):
            raise ValueError("Registered visits do not share the saved T0 target grid")
        array = load_mri_payload(visit["image_source"]["path"], meta)
        transform = REVERSE @ np.linalg.inv(LPS_RAS @ acquisition_affine(meta)) @ crop_ras @ REVERSE
        cropped = resample_array(array, transform, 1)
        coverage = resample_array(np.ones(array.shape, dtype=np.uint8), transform, 0).astype(bool)
        valid = coverage & (cropped != 0)
        if not valid.any() or not np.isfinite(cropped).all():
            raise ValueError(f"Empty/non-finite MRI crop: {visit['visit_id']}")
        # A separate coordinate sampler checks array order and interpolation.
        probes = np.array([[0, 0, 0], [15, 63, 63], [31, 127, 127], [9, 22, 117]], dtype=float)
        locations = probes @ transform[:3, :3].T + transform[:3, 3]
        expected = map_coordinates(array, locations.T, order=1, mode="constant", cval=0, prefilter=False)
        actual = cropped[tuple(probes.astype(int).T)]
        if not np.allclose(expected, actual, atol=1e-3, rtol=2e-6):
            raise ValueError("Independent MRI interpolation check failed")
        file = root / "images" / visit_filename(visit["visit_id"])
        save_npz(file, image=cropped, valid=valid, coverage=coverage)
        values = cropped[valid].astype(np.float64)
        result["visits"].append({"visit_id": visit["visit_id"], "fold": visit["fold"], "visit": visit["visit"],
                                 "foreground_voxels": int(valid.sum()), "padding_fraction": float(1 - coverage.mean()),
                                 "sum": float(values.sum()), "square_sum": float(np.square(values).sum()),
                                 "interpolation_max_error": float(np.max(np.abs(expected - actual)))})
        result["cache_identities"].append(file_identity(file))
    write_json(marker, result)
    return result


def prepare(config, limit=None):
    root = Path(config["output_dir"]) / "data"
    root.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(config)
    inventory_path = root / "inventory.json"
    if inventory_path.exists() and read_json(inventory_path) != inventory:
        raise ValueError("Preparation configuration/cohort/sources changed; use a new output")
    write_json(inventory_path, inventory)
    by_patient = defaultdict(list)
    for visit in inventory["visits"]:
        by_patient[visit["patient_id"]].append(visit)
    patients = inventory["patients"][:limit]
    tasks = [(config, p, by_patient[p["patient_id"]]) for p in patients]
    started = time.monotonic()
    reports = []
    with ProcessPoolExecutor(max_workers=config["data"]["workers"]) as executor:
        for report in executor.map(prepare_patient, tasks):
            reports.append(report)
            progress = {"stage": "prepare", "patients_completed": len(reports), "patients_total": len(tasks),
                        "seconds": time.monotonic() - started, "updated_at": timestamp()}
            write_json(root / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    if limit is not None:
        return {"stage": "partial_preparation", "patients": len(reports)}
    stats = [v for p in reports for v in p["visits"] if v["fold"] == "train"]
    count = sum(v["foreground_voxels"] for v in stats)
    mean = sum(v["sum"] for v in stats) / count
    std = math.sqrt(max(0, sum(v["square_sum"] for v in stats) / count - mean**2))
    if std <= 1e-8:
        raise ValueError("Degenerate training intensity statistics")
    normalization = {"schema": SCHEMA, "fit_split": "train", "scope": "unique_train_crop_nonzero_foreground",
                     "mean": mean, "std": std, "count": count, "fit_visit_ids": sorted(v["visit_id"] for v in stats),
                     "background_and_padding": 0}
    write_json(root / "normalization.json", normalization)
    summary = {"status": "passed", "updated_at": timestamp(), "patients": config["data"]["expected_patients"],
               "visits": config["data"]["expected_visits"], "pairs": config["data"]["expected_pairs"],
               "largest_clipped": sum(p["largest_clipped"] for p in reports),
               "all_components_clipped": sum(p["all_clipped"] for p in reports),
               "empty_t0_crops": sum(p["actual_largest_voxels"] == 0 for p in reports),
               "originally_empty_t0_excluded": 1,
               "max_padding_fraction": max(v["padding_fraction"] for p in reports for v in p["visits"]),
               "followup_model_masks": "separate_mapping_audit_required", "test_split": None,
               "configuration": config["data"], "inventory_identity": file_identity(inventory_path),
               "normalization_identity": file_identity(root / "normalization.json")}
    if summary["largest_clipped"] != 71 or summary["all_components_clipped"] != 235:
        raise ValueError("Crop support statistics disagree with the original audit")
    write_json(root / "COMPLETE.json", summary)
    return summary


class CropDataset(Dataset):
    def __init__(self, config, fold=None, records=None):
        self.root = Path(config["output_dir"]) / "data"
        complete = read_json(self.root / "COMPLETE.json")
        if complete["status"] != "passed" or complete["configuration"] != config["data"]:
            raise ValueError("Data preparation is incomplete or incompatible")
        verify_identity(complete["inventory_identity"])
        verify_identity(complete["normalization_identity"])
        self.inventory = read_json(self.root / "inventory.json")
        self.records = list(records) if records is not None else [r for r in self.inventory["visits"] if fold is None or r["fold"] == fold]
        self.normalization = read_json(self.root / "normalization.json")
        expected = sorted(v["visit_id"] for v in self.inventory["visits"] if v["fold"] == "train")
        if self.normalization["fit_split"] != "train" or self.normalization["fit_visit_ids"] != expected:
            raise ValueError("Intensity statistics are not fitted on unique training visits")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(self.root / "images" / visit_filename(record["visit_id"]), allow_pickle=False) as archive:
            raw = archive["image"].astype(np.float32)
            valid = archive["valid"].astype(bool)
            coverage = archive["coverage"].astype(bool)
        with np.load(self.root / "patients" / (record["patient_id"] + ".npz"), allow_pickle=False) as archive:
            mask = archive["mask"].astype(bool)
        image = np.zeros(SHAPE, dtype=np.float32)
        image[valid] = (raw[valid] - self.normalization["mean"]) / self.normalization["std"]
        if raw.shape != SHAPE or not np.isfinite(image).all():
            raise ValueError("Invalid prepared image crop")
        return {"image": torch.from_numpy(image[None]), "mask": torch.from_numpy(mask[None]),
                "valid": torch.from_numpy(valid[None]), "coverage": torch.from_numpy(coverage[None]),
                "visit_id": record["visit_id"], "patient_id": record["patient_id"], "visit": record["visit"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(read_config(args.config), limit=args.limit)), flush=True)
