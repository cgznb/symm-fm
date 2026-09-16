from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, Sampler

SCHEMA = "first_post_unregistered_tumor_roi_v1"
IMAGE_SHAPE = (96, 256, 256)
LATENT_SHAPE = (8, 24, 64, 64)
GIB = 1024**3


def now():
    return datetime.now(timezone.utc).isoformat()


def public(value):
    if isinstance(value, dict):
        return {str(k): public(v) for k, v in value.items()
                if not any(s in str(k).lower() for s in ("sha256", "checksum", "fingerprint", "hash", "revision"))}
    if isinstance(value, (tuple, list)):
        return [public(v) for v in value]
    if isinstance(value, Path):
        return str(value)
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


def read_json(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular artifact: {path}")
    s = path.stat()
    return {"size_bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def config(path):
    value = _release_yaml(Path(path).read_text())
    if value.get("schema") != SCHEMA or value.get("seed") != 2026:
        raise ValueError("Unexpected first-post comparison configuration")
    return value


def disk_gate(root, *, required=0, reserve_gib=15):
    free = shutil.disk_usage(root).free
    if free - required < reserve_gib * GIB:
        raise RuntimeError(f"Disk admission failed: free={free/GIB:.2f} GiB, "
                           f"required={required/GIB:.2f} GiB, reserve={reserve_gib} GiB")


def case_name(record):
    return f"{record['patient_id']}_{record['visit']}_aqc1"


def reference_image(geometry):
    import SimpleITK as sitk
    image = sitk.Image(list(reversed(geometry["shape_zyx"])), sitk.sitkFloat32)
    image.SetSpacing(geometry["spacing_xyz_mm"])
    image.SetOrigin(geometry["origin_lps_mm"])
    image.SetDirection(geometry["direction_lps"])
    return image


def native_zyx_image(array, record):
    import SimpleITK as sitk
    if tuple(array.shape) != tuple(record["shape_zyx"]) or not np.isfinite(array).all():
        raise ValueError("Native SER array differs from the corresponding MRI grid")
    iop = np.asarray(record["image_orientation_patient"], dtype=float)
    direction = np.column_stack((iop[:3], iop[3:], np.cross(iop[:3], iop[3:])))
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError("Native direction is not orthonormal")
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    image.SetOrigin(record["image_position_patient_first"])
    image.SetSpacing([record["pixel_spacing_yx_mm"][1], record["pixel_spacing_yx_mm"][0], record["slice_spacing_mm"]])
    image.SetDirection(direction.ravel().tolist())
    return image


def resample_ser(path, record):
    import nibabel as nib
    import SimpleITK as sitk
    nii = nib.load(path)
    if np.allclose(nii.affine, np.eye(4)):
        source = native_zyx_image(np.asarray(nii.dataobj, dtype=np.float32), record)
    else:
        source = sitk.ReadImage(str(path), sitk.sitkFloat32)
    result = sitk.Resample(source, reference_image(record["crop_geometry"]),
                          sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear, 0.0)
    array = sitk.GetArrayFromImage(result)
    if array.shape != IMAGE_SHAPE or not np.isfinite(array).all():
        raise ValueError("Invalid resampled SER")
    return array


def load_codec(path, device="cpu"):
    from .vqgan import MRILevelVQGAN, VQGANConfig
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("schema") == SCHEMA:
        cfg, state = payload["model_config"], payload["codec_state"]
    else:
        marker = payload.get("mewm_ispy2_vqgan_identity", {})
        if (marker.get("numeric_contract") != "ispy2_first_post_unregistered_train_zscore_v1"
                or marker.get("phase_index") != 1 or marker.get("registered") is not False):
            raise ValueError("Codec is not a first-post unregistered checkpoint")
        cfg = marker["architecture_contract"]["config"]
        state = {k.removeprefix("autoencoder."): v for k, v in payload["state_dict"].items()
                 if k.startswith("autoencoder.")}
    model = MRILevelVQGAN(VQGANConfig(**cfg))
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device)


class PairDataset(Dataset):
    """One shared visit/pair inventory with framework-specific latent scaling."""

    def __init__(self, root, arm, split=None):
        if arm not in ("bifm", "symm"):
            raise ValueError("Unknown comparison arm")
        self.root, self.arm = Path(root), arm
        self.bundle = read_json(self.root / "shared/bundle.json")
        if (self.bundle.get("schema") != SCHEMA or not self.bundle.get("ready")
                or not self.bundle.get("verified_for_world_training")):
            raise ValueError("First-post preparation is not complete")
        self.records = [p for p in self.bundle["pairs"] if split is None or p["split"] == split]
        self.visits = {r["visit_id"]: r for r in self.bundle["visits"]}
        self.stats = self.bundle["normalization"]
        validate_pairs(self.bundle["pairs"], self.visits)
        if identity(self.root / "shared/codec.pt") != self.bundle["codec_identity"]:
            raise ValueError("Frozen codec artifact changed")
        if not self.records:
            raise ValueError("Empty longitudinal dataset")

    def __len__(self):
        return len(self.records)

    def raw_latent(self, visit_id):
        r = self.visits[visit_id]
        path = self.root / r["latent_path"]
        if identity(path) != r["latent_identity"]:
            raise ValueError("Continuous latent cache changed")
        a = np.load(path, allow_pickle=False)
        if a.shape != LATENT_SHAPE or not np.isfinite(a).all():
            raise ValueError("Invalid continuous latent")
        return torch.from_numpy(a.astype(np.float32))

    def normalize(self, value):
        s = self.stats
        if self.arm == "bifm":
            return 2 * (value - s["codebook_min"]) / (s["codebook_max"] - s["codebook_min"]) - 1
        shape = (8, 1, 1, 1)
        return (value - value.new_tensor(s["latent_mean"]).view(shape)) / value.new_tensor(s["latent_std"]).view(shape)

    def denormalize(self, value):
        s = self.stats
        if self.arm == "bifm":
            return (value + 1) / 2 * (s["codebook_max"] - s["codebook_min"]) + s["codebook_min"]
        shape = (1, 8, 1, 1, 1) if value.ndim == 5 else (8, 1, 1, 1)
        return value * value.new_tensor(s["latent_std"]).view(shape) + value.new_tensor(s["latent_mean"]).view(shape)

    def image(self, visit_id):
        r = self.visits[visit_id]
        if identity(r["image_path"]) != r["image_identity"]:
            raise ValueError("First-post crop changed")
        return np.load(r["image_path"], allow_pickle=False).astype(np.float32)

    def source_mri(self, visit_id):
        r = self.visits[visit_id]
        first = self.image(visit_id)
        path = self.root / r["ser_crop_path"]
        if identity(path) != r["ser_crop_identity"]:
            raise ValueError("SER crop changed")
        with np.load(path, allow_pickle=False) as f:
            ser = f["ser"].astype(np.float32)
        foreground = first != 0
        ser[foreground] = (ser[foreground] - self.stats["ser_mean"]) / self.stats["ser_std"]
        ser[~foreground] = 0
        return torch.from_numpy(np.stack((first, ser)))

    def __getitem__(self, index):
        r = self.records[index]
        result = {"source": self.normalize(self.raw_latent(r["earlier_visit_id"])),
                  "target": self.normalize(self.raw_latent(r["later_visit_id"])), "record": r}
        if self.arm == "bifm":
            result["source_mri"] = self.source_mri(r["earlier_visit_id"])
        return result


def collate(items):
    result = {"records": [x["record"] for x in items]}
    for key in items[0]:
        if key != "record":
            result[key] = torch.stack([x[key] for x in items])
    return result


def validate_pairs(pairs, visits):
    assignments, ids = {}, set()
    for p in pairs:
        if p["pair_id"] in ids:
            raise ValueError("Duplicate pair")
        ids.add(p["pair_id"])
        if p["split"] not in ("train", "val") or assignments.setdefault(p["patient_id"], p["split"]) != p["split"]:
            raise ValueError("Patient crosses splits")
        i, j = int(p["earlier_stage"][1:]), int(p["later_stage"][1:])
        if not 0 <= i < j <= 3 or type(p["delta_days"]) is not int or p["delta_days"] <= 0:
            raise ValueError("Invalid temporal supervision")
        for key, stage in (("earlier_visit_id", "earlier_stage"), ("later_visit_id", "later_stage")):
            v = visits[p[key]]
            if (v["canonical_patient_id"] != p["patient_id"] or v["split"] != p["split"]
                    or v["visit"] != p[stage]):
                raise ValueError("Pair endpoint identity differs")
        chain = p.get('adjacent_visit_chain')
        if chain is not None:
            if (len(chain) != j - i + 1 or chain[0] != p['earlier_visit_id']
                    or chain[-1] != p['later_visit_id']):
                raise ValueError('Incomplete adjacent visit chain')
            for stage, visit_id in enumerate(chain, i):
                v = visits[visit_id]
                if v['canonical_patient_id'] != p['patient_id'] or v['split'] != p['split'] or v['visit'] != f'T{stage}':
                    raise ValueError('Invalid adjacent visit chain')
        allowed = {"age", "hr_status", "her2_status", "mammaprint", "menopausal_status"}
        if not set(p["baseline_clinical"]) <= allowed or set(p["treatment"]) != {"treatment_arm"}:
            raise ValueError("Unexpected model-visible conditions")


class PatientBalancedBatchSampler(Sampler):
    """Step-indexed draws make resume independent of worker prefetch."""

    def __init__(self, records, microbatch, effective_batch, start_step, max_steps, seed=2026):
        if effective_batch % microbatch:
            raise ValueError("Effective batch must be divisible by microbatch")
        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[r["patient_id"]].append(i)
        self.groups = [groups[p] for p in sorted(groups)]
        self.microbatch, self.effective_batch = microbatch, effective_batch
        self.start_step, self.max_steps, self.seed = start_step, max_steps, seed

    def __iter__(self):
        for step in range(self.start_step, self.max_steps):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            selected = []
            for index in rng.integers(len(self.groups), size=self.effective_batch):
                group = self.groups[index]
                selected.append(group[int(rng.integers(len(group)))])
            for offset in range(0, self.effective_batch, self.microbatch):
                yield selected[offset:offset + self.microbatch]

    def __len__(self):
        return (self.max_steps - self.start_step) * (self.effective_batch // self.microbatch)
