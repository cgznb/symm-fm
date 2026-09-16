"""Native first-post, three-phase tumor ROI inputs for a frozen Pillar evaluator."""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml, ssh_host

import ast
import copy
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import DRUG_COLS
from src.pillar import normalize_volume, pad_or_crop, resample_volume

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "first_post_unregistered_tumor_roi_pcr_v1"
IMAGE_SHAPE = (96, 256, 256)
PILLAR_SHAPE = (3, 384, 384, 192)


def now():
    return datetime.now(timezone.utc).isoformat()


def repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    return json.loads(Path(path).read_text())


def public(value):
    if isinstance(value, dict):
        return {str(k): public(v) for k, v in value.items()
                if not any(x in str(k).lower() for x in ("sha256", "checksum", "fingerprint", "hash"))}
    if isinstance(value, (list, tuple)):
        return [public(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return public(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, str):
        return re.sub(r"(?<![a-zA-Z0-9])[a-fA-F0-9]{40,64}(?![a-zA-Z0-9])", "<legacy-digest>", value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(public(value), indent=2, allow_nan=False) + "\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def identity(path):
    stat = Path(path).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_config(path):
    cfg = _release_yaml(repo_path(path).read_text())
    if cfg.get("schema") != SCHEMA:
        raise ValueError("Unexpected first-post pCR schema")
    cfg["output_dir"] = str(repo_path(cfg["output_dir"]))
    return cfg


def world_imports(cfg):
    path = cfg["world_repo"]
    if path not in sys.path:
        sys.path.insert(0, path)


def disk_gate(cfg, required=0):
    free = shutil.disk_usage(cfg["output_dir"]).free
    if free - required < float(cfg["reserve_gib"]) * 2**30:
        raise RuntimeError(f"Insufficient disk: {free / 2**30:.2f} GiB free, "
                           f"{required / 2**30:.2f} GiB required plus reserve")


def ssh_arguments(cfg):
    from research_release import ssh_arguments as configured_ssh
    return configured_ssh()


def remote_phase_sizes(cfg, visits):
    code = """
import json,os,sys
from concurrent.futures import ThreadPoolExecutor
rows=json.load(sys.stdin)
def sizes(row):
    return {phase:os.stat(path).st_size for phase,path in row.items()}
with ThreadPoolExecutor(max_workers=16) as pool:
    print(json.dumps(list(pool.map(sizes,rows))))
"""
    result = subprocess.run([*ssh_arguments(cfg), ssh_host(), "python3", "-c", shlex.quote(code)],
                            input=json.dumps([v["remote_phases"] for v in visits]), text=True,
                            capture_output=True, timeout=300)
    if result.returncode:
        raise RuntimeError(f"Native phase inventory failed: {result.stderr[-1500:]}")
    rows = json.loads(result.stdout)
    if len(rows) != len(visits):
        raise ValueError("Remote inventory length differs")
    return rows


def build_cohort(bundle, native, metadata, aliases):
    rows = {}
    for _, row in metadata.iterrows():
        pid = aliases.get(str(row.pid), str(row.pid))
        if pid in rows:
            raise ValueError("Duplicate canonical patient in phase metadata")
        rows[pid] = row
    native_index = {(r["patient_id"], r["visit"]): r for r in native["records"]}
    patients = defaultdict(dict)
    assignments = {}
    for visit in bundle["visits"]:
        pid = visit["canonical_patient_id"]
        split = visit["split"]
        if split not in ("train", "val") or assignments.setdefault(pid, split) != split:
            raise ValueError("Canonical patient crosses the world-model split")
        tp = int(visit["visit"][1:])
        if tp in patients[pid]:
            raise ValueError("Duplicate canonical visit")
        patients[pid][tp] = visit
    visits, patient_rows, exclusions = [], [], []
    for pid, available in sorted(patients.items()):
        row = rows.get(pid)
        if row is None or pd.isna(row.get("pCR")):
            exclusions.append({"patient_id": pid, "reason": "missing_phase_metadata_or_pcr"})
            continue
        if float(row.pCR) not in (0.0, 1.0):
            raise ValueError("pCR is not binary")
        selected = []
        for tp in range(4):
            v = available.get(tp)
            late = row.get(f"post_late_T{tp}", np.nan)
            if v is None or pd.isna(late):
                if any(t >= tp for t in available):
                    exclusions.append({"patient_id": pid, "from_stage": tp,
                                       "reason": "missing_visit_or_explicit_late_breaks_prefix"})
                break
            if not float(late).is_integer():
                raise ValueError("Late phase index must be an integer")
            late = int(late)
            if late <= 1 or float(row.get(f"pre_T{tp}", np.nan)) != 0:
                raise ValueError("Invalid metadata pre/late phase selection")
            source = native_index[v["patient_id"], v["visit"]]
            first_remote = str(source["source_path"])
            if not first_remote.endswith("_aqc_1.nii.gz"):
                raise ValueError("First-post source does not name aqc_1")
            dt = date.fromisoformat(v["visit_date"])
            if selected and dt <= date.fromisoformat(selected[-1]["visit_date"]):
                exclusions.append({"patient_id": pid, "from_stage": tp,
                                   "reason": "nonpositive_verified_interval_breaks_prefix"})
                break
            keep = {k: copy.deepcopy(v[k]) for k in (
                "patient_id", "canonical_patient_id", "visit", "visit_id", "split", "visit_date",
                "shape_zyx", "image_orientation_patient", "image_position_patient_first",
                "pixel_spacing_yx_mm", "slice_spacing_mm", "crop_geometry", "crop_localization",
                "image_path", "metadata_path", "mask_path")}
            keep.update(timepoint=tp, late_index=late,
                        native_first_post=str(Path(native["destination_root"]) / source["relative_path"]),
                        remote_phases={"pre": first_remote.replace("_aqc_1.nii.gz", "_aqc_0.nii.gz"),
                                       "late": first_remote.replace("_aqc_1.nii.gz", f"_aqc_{late}.nii.gz")})
            selected.append(keep)
        if not selected:
            exclusions.append({"patient_id": pid, "reason": "no_valid_t0"})
            continue
        clinical = {k: row.get(k, np.nan) for k in (
            "pCR", "age", "menopause", "HR", "HER2", "HR_HER2_STATUS", "MP", "Arm", *DRUG_COLS)}
        clinical.update(pid=pid, source="ACRIN-6698" if selected[0]["patient_id"].startswith("ACRIN") else "ISPY2",
                        n_available_timepoints=len(selected), split=assignments[pid])
        for tp in range(4):
            clinical[f"days_T{tp}"] = ((date.fromisoformat(selected[tp]["visit_date"])
                                         - date.fromisoformat(selected[0]["visit_date"])).days
                                        if tp < len(selected) else 0)
        patient_rows.append(clinical)
        visits.extend(selected)
    frame = pd.DataFrame(patient_rows)
    splits = {s: sorted(frame.loc[frame.split == s, "pid"].tolist()) for s in ("train", "val")}
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("pCR patient splits overlap")
    return visits, frame, splits, exclusions


def make_folds(frame, seed):
    development = frame.loc[frame.split == "train"].sort_values("pid").reset_index(drop=True)
    columns = ["pCR", "HR_HER2_STATUS", "source", "n_available_timepoints"]
    if development[columns].isna().any().any():
        raise ValueError("Missing fold stratification field")
    labels = (development.pCR.astype(int).astype(str) + "|"
              + development.HR_HER2_STATUS.astype(str) + "|" + development.source.astype(str)).to_numpy()
    assignment = np.full(len(development), -1, dtype=int)
    for fold, (_, val) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(development, labels)):
        assignment[val] = fold
    counts = development.n_available_timepoints.to_numpy(dtype=int)
    marginals = np.array([[sum((assignment == f) & (counts == c)) for c in (1, 2, 3, 4)]
                          for f in range(5)], dtype=float)
    target = np.array([sum(counts == c) / 5 for c in (1, 2, 3, 4)])
    # Match the existing fold builder: within-stratum swaps preserve stratum and fold sizes.
    while True:
        best = None
        for left in range(len(assignment)):
            for right in range(left + 1, len(assignment)):
                a, b = assignment[left], assignment[right]
                x, y = counts[left] - 1, counts[right] - 1
                if a == b or x == y or labels[left] != labels[right]:
                    continue
                before = sum((marginals[f, c] - target[c]) ** 2 for f, c in ((a, x), (a, y), (b, x), (b, y)))
                after = sum((marginals[f, c] + d - target[c]) ** 2
                            for f, c, d in ((a, x, -1), (a, y, 1), (b, x, 1), (b, y, -1)))
                candidate = (after - before, left, right)
                if candidate[0] < -1e-12 and (best is None or candidate < best):
                    best = candidate
        if best is None:
            break
        _, left, right = best
        a, b, x, y = assignment[left], assignment[right], counts[left] - 1, counts[right] - 1
        marginals[a, x] -= 1
        marginals[a, y] += 1
        marginals[b, x] += 1
        marginals[b, y] -= 1
        assignment[left], assignment[right] = b, a
    ids = development.pid.to_numpy()
    return [{"fold": f, "train_ids": ids[assignment != f].tolist(), "val_ids": ids[assignment == f].tolist()}
            for f in range(5)]


def prepare(cfg):
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    disk_gate(cfg)
    sources = {"native_manifest": identity(cfg["native_manifest"]),
               "phase_metadata": identity(repo_path(cfg["metadata_csv"])),
               "identity_mapping": identity(cfg["identity_mapping"]),
               "world_metadata": identity(Path(cfg["world_run"]) / "shared/metadata_bundle.json")}
    if (output / "cohort.json").exists():
        cohort = read_json(output / "cohort.json")
        if cohort["source_identities"] != sources or cohort["schema"] != SCHEMA:
            raise ValueError("Cohort source contract changed")
        return cohort
    mapping = pd.read_excel(cfg["identity_mapping"])
    aliases = {str(r["TCIA PATIENT ID"]): f"ISPY2-{int(r['I-SPY 2 Research ID'])}"
               for _, r in mapping.iterrows() if pd.notna(r["I-SPY 2 Research ID"])}
    bundle = read_json(Path(cfg["world_run"]) / "shared/metadata_bundle.json")
    native = read_json(cfg["native_manifest"])
    if native["registered"] is not False or native["phase_index"] != 1:
        raise ValueError("Native manifest is not unregistered first-post")
    visits, frame, splits, exclusions = build_cohort(bundle, native, pd.read_csv(repo_path(cfg["metadata_csv"])), aliases)
    sizes = remote_phase_sizes(cfg, visits)
    for visit, size in zip(visits, sizes):
        if not Path(visit["native_first_post"]).is_file():
            raise FileNotFoundError("Local first-post file missing")
        native_meta = read_json(visit["metadata_path"])
        if visit["late_index"] not in native_meta["dce_phase_ids"]:
            raise ValueError("Selected late is absent from native phase IDs")
        visit["phase_bytes"] = size
        visit["native_first_post_identity"] = identity(visit["native_first_post"])
        visit["crop_identity"] = identity(visit["image_path"])
        visit["mask_identity"] = identity(visit["mask_path"])
    counts = {s: {"patients": len(splits[s]), "visits": sum(v["split"] == s for v in visits),
                  "positive_patients": int(frame.loc[frame.split == s, "pCR"].sum()),
                  "complete_prefix_counts": [int(((frame.split == s) & (frame.n_available_timepoints >= t)).sum())
                                             for t in (1, 2, 3, 4)]} for s in splits}
    frame.to_csv(output / "metadata_enriched.csv", index=False)
    write_json(output / "folds.json", make_folds(frame, int(cfg["fold_seed"])))
    cohort = {"schema": SCHEMA, "created_utc": now(), "source_identities": sources,
              "registered": False, "phase_order": ["pre_aqc0", "first_post_aqc1", "metadata_late"],
              "roi_shape_zyx": IMAGE_SHAPE, "pillar_shape_chwd": PILLAR_SHAPE,
              "pillar_spacing_zyx": [1, 1, 1], "pillar_intensity": "per_channel_p1_p99_clip_minmax",
              "prefix_policy": "contiguous_t0_starting", "split": splits, "counts": counts,
              "test_role": "world_selection_validation_not_independent_test",
              "visits": visits, "exclusions": exclusions}
    write_json(output / "cohort.json", cohort)
    print(json.dumps({"stage": "cohort_ready", "counts": counts,
                      "native_transfer_gib": sum(sum(s.values()) for s in sizes) / 2**30}), flush=True)
    return cohort


def stage_phases(cfg, visits):
    staging = Path(cfg["output_dir"]) / "staging"
    staging.mkdir(exist_ok=True)
    root = Path(cfg["remote_root"])
    names = []
    for visit in visits:
        for phase, remote in visit["remote_phases"].items():
            relative = Path(remote).relative_to(root)
            if ".." in relative.parts:
                raise ValueError("Unsafe native phase path")
            local = staging / relative
            if not local.is_file() or local.stat().st_size != visit["phase_bytes"][phase]:
                names.append(str(relative))
    if names:
        disk_gate(cfg, required=sum(sum(v["phase_bytes"].values()) for v in visits))
        workers = int(cfg.get("transfer_workers", 1))
        unique = sorted(set(names))

        def transfer(shard):
            if not shard:
                return
            command = ["rsync", "-rt", "--partial-dir=.rsync-partial", "--from0", "--files-from=-",
                       "-e", shlex.join(ssh_arguments(cfg)), f"{ssh_host()}:{root}/", str(staging) + "/"]
            result = subprocess.run(command, input="\0".join(shard) + "\0",
                                    text=True, capture_output=True, timeout=1800)
            if result.returncode:
                raise RuntimeError(f"Native phase transfer failed: {result.stderr[-1500:]}")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(transfer, [unique[i::workers] for i in range(workers)]))
    for visit in visits:
        for phase, remote in visit["remote_phases"].items():
            path = staging / Path(remote).relative_to(root)
            if not path.is_file() or path.stat().st_size != visit["phase_bytes"][phase]:
                raise ValueError("Transferred phase size differs from native inventory")


def release_phases(cfg, visits):
    staging = Path(cfg["output_dir"]) / "staging"
    for visit in visits:
        for remote in visit["remote_phases"].values():
            (staging / Path(remote).relative_to(cfg["remote_root"])).unlink(missing_ok=True)


def native_image(path, visit):
    import nibabel as nib
    import SimpleITK as sitk
    from mewm_ispy2.first_post_world_data import native_zyx_image
    nii = nib.load(str(path))
    if np.allclose(nii.affine, np.eye(4)):
        image = native_zyx_image(np.asarray(nii.dataobj, dtype=np.float32), visit)
    else:
        image = sitk.ReadImage(str(path), sitk.sitkFloat32)
        if tuple(reversed(image.GetSize())) != tuple(visit["shape_zyx"]):
            raise ValueError("Native phase geometry differs from the first-post series")
    # Match VQ-GAN's operation order, including interpolation at half-voxel boundaries.
    return sitk.DICOMOrient(image, "LPS")


def resample_native_roi(path, visit, support=False):
    import SimpleITK as sitk
    from mewm_ispy2.first_post_world_data import reference_image
    source = native_image(path, visit)
    if support:
        ones = sitk.GetImageFromArray(np.ones(tuple(reversed(source.GetSize())), dtype=np.uint8))
        ones.CopyInformation(source)
        source = ones
    image = sitk.Resample(source, reference_image(visit["crop_geometry"]),
                          sitk.Transform(3, sitk.sitkIdentity),
                          sitk.sitkNearestNeighbor if support else sitk.sitkLinear, 0.0)
    array = sitk.GetArrayFromImage(image)
    if array.shape != IMAGE_SHAPE or not np.isfinite(array).all():
        raise ValueError("Invalid native ROI")
    return array


def pillar_channel(roi, spacing_zyx):
    array = resample_volume(np.asarray(roi, dtype=np.float32), spacing_zyx)
    array = normalize_volume(array)
    return pad_or_crop(torch.from_numpy(array).float()).permute(1, 2, 0).contiguous()


def tumor_retention(visit):
    import SimpleITK as sitk
    from mewm_ispy2.first_post_world_data import reference_image
    if identity(visit["mask_path"]) != visit["mask_identity"]:
        raise ValueError("Cropping mask changed")
    mask = sitk.Resample(sitk.ReadImage(visit["mask_path"], sitk.sitkUInt8),
                         reference_image(visit["crop_geometry"]), sitk.Transform(3, sitk.sitkIdentity),
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    roi = sitk.GetArrayFromImage(mask)
    spacing = visit["crop_geometry"]["spacing_xyz_mm"][::-1]
    isotropic = resample_volume(roi, spacing, interp="nearest")
    before = int(np.count_nonzero(isotropic))
    after = int(torch.count_nonzero(pad_or_crop(torch.from_numpy(isotropic))))
    return {"roi_tumor_voxels_1mm": before, "pillar_tumor_voxels_1mm": after,
            "tumor_retention": after / before if before else None}


def build_volume(cfg, visit, replacement=None, audit=True, replacement_valid=None):
    """Apply the same geometry and channel preprocessing to real and hybrid inputs."""
    world_imports(cfg)
    if identity(visit["native_first_post"]) != visit["native_first_post_identity"]:
        raise ValueError("Native first-post changed")
    root = Path(cfg["output_dir"]) / "staging"
    spacing = visit["crop_geometry"]["spacing_xyz_mm"][::-1]
    channels = []
    crop_error = None
    for phase in ("pre", "first_post", "late"):
        if phase == "first_post" and replacement is not None:
            value = np.asarray(replacement, dtype=np.float32)
            if value.shape != IMAGE_SHAPE or not np.isfinite(value).all():
                raise ValueError("Invalid replacement first-post")
            roi = value * 283.804038 + 142.555409
            support = resample_native_roi(visit["native_first_post"], visit, support=True)
            roi[support == 0] = 0
            if replacement_valid is not None:
                if np.asarray(replacement_valid).shape != IMAGE_SHAPE:
                    raise ValueError("Invalid replacement support")
                roi[~np.asarray(replacement_valid, dtype=bool)] = 0
        else:
            path = (Path(visit["native_first_post"]) if phase == "first_post" else
                    root / Path(visit["remote_phases"][phase]).relative_to(cfg["remote_root"]))
            roi = resample_native_roi(path, visit)
            if phase == "first_post" and audit:
                if identity(visit["image_path"]) != visit["crop_identity"]:
                    raise ValueError("VQ-GAN first-post crop changed")
                cached = np.load(visit["image_path"], allow_pickle=False).astype(np.float32)
                normalized = np.zeros_like(roi)
                foreground = roi != 0
                normalized[foreground] = (roi[foreground] - 142.555409) / 283.804038
                if cached.shape != IMAGE_SHAPE or not np.allclose(cached, normalized, rtol=0.001, atol=0.0001):
                    raise ValueError("Native first-post resampling differs from the frozen VQ-GAN crop")
                crop_error = float(np.abs(cached - normalized).max())
        channels.append(pillar_channel(roi, spacing))
    volume = torch.stack(channels)
    if tuple(volume.shape) != PILLAR_SHAPE or not torch.isfinite(volume).all():
        raise ValueError("Invalid Pillar input")
    report = {"visit_id": visit["visit_id"], "canonical_patient_id": visit["canonical_patient_id"],
              "timepoint": visit["timepoint"], "split": visit["split"],
              "shape_chwd": list(volume.shape), "phase_indices": [0, 1, visit["late_index"]],
              "vq_crop_max_absolute_difference": crop_error}
    if audit:
        report.update(tumor_retention(visit))
    return volume, report


def embedding_path(cfg, source, visit):
    pid = visit["canonical_patient_id"]
    return Path(cfg["output_dir"]) / "embeddings" / source / pid / f"{pid}_T{visit['timepoint']}.pt"


def validate_embedding(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.shape != (1152,) or not torch.isfinite(value).all() or value.norm() <= 0:
        raise ValueError("Invalid first-post Pillar embedding")
    return value
