from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from .backend import sha256_file, validate_registered_strict_a_bundle_contract
from .data import PreparedVisit
from .manifest import ACCEPTED_BACKENDS, VisitRecord
from .preprocessing import (
    NORMALIZATION_PERCENTILES,
    resample_image_and_mask,
    robust_nonzero_scale,
    source_centered_pair_crop,
    tumor_center_crop,
)


CACHE_SCHEMA_VERSION = "mewm_ispy2_dce0_cache_v2"
MOTFM_CACHE_SCHEMA_VERSION = "motfm_ispy2_visit_roi_cache_v2"
MOTFM_CHANNEL_ORDER = ("dce0", "ser")
MOTFM_TARGET_SPACING_XYZ = (0.7032, 0.7032, 2.0)
MOTFM_PAYLOAD_KEYS = {
    "schema_version",
    "cache_key",
    "bundle_contract_sha256",
    "bundle_preprocess_sha256",
    "effective_preprocess_sha256",
    "source_asset_sha256",
    "source_asset_stat_signatures",
    "source_asset_contract_sha256",
    "channel_order",
    "mri",
    "mask",
    "preprocess_metadata",
}
MOTFM_SOURCE_ASSETS = {"dce0", "ser", "mask", "meta"}
REGISTERED_STRICT_A_BUNDLE_SCHEMA = "motfm_ispy2_registered_strict_a_bundle_v1"
REGISTERED_STRICT_A_CACHE_SCHEMA = "motfm_ispy2_t0_fixed_roi_cache_v1"
REGISTERED_STRICT_A_PREPROCESS_SCHEMA = "motfm_ispy2_registered_preprocess_v1"
REGISTERED_STRICT_A_NORMALIZATION_SCHEMA = "motfm_ispy2_train_foreground_stats_v1"
REGISTERED_STRICT_A_COORDINATE_FRAME = "t0_fixed_registered_local_v1"
REGISTERED_STRICT_A_OUTPUT_SHAPE_ZYX = (96, 256, 256)
REGISTERED_STRICT_A_PAYLOAD_KEYS = {
    "schema",
    "cache_key",
    "visit_id",
    "bundle_contract_sha256",
    "preprocess_sha256",
    "normalization_sha256",
    "mri",
    "mask",
    "valid_foreground",
    "crop_plan",
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _spacing_from_meta(meta: dict[str, Any]) -> tuple[float, float, float] | None:
    pixel_spacing = meta.get("pixel_spacing")
    slice_spacing = meta.get("spacing_between_slices")
    if not isinstance(pixel_spacing, list) or len(pixel_spacing) != 2 or slice_spacing is None:
        return None
    return float(slice_spacing), float(pixel_spacing[0]), float(pixel_spacing[1])


class DCE0Cache:
    def __init__(
        self,
        root: str | Path,
        *,
        output_shape_zyx: Sequence[int] = (128, 128, 128),
        percentiles: Sequence[float] = NORMALIZATION_PERCENTILES,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_shape_zyx = tuple(int(value) for value in output_shape_zyx)
        self.percentiles = tuple(float(value) for value in percentiles)

    def _identity(self, visit: VisitRecord, backend: str) -> dict[str, Any]:
        if backend not in ACCEPTED_BACKENDS:
            raise ValueError("data backend is invalid")
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "visit_id": visit.visit_id,
            "backend": backend,
            "dce0_path": str(visit.dce0.path.resolve()),
            "dce0_sha256": sha256_file(visit.dce0.path),
            "mask_sha256": sha256_file(visit.mask_path),
            "meta_sha256": sha256_file(visit.meta_path),
            "phase_index": visit.dce0.phase_index,
            "n_times": visit.dce0.n_times,
            "output_shape_zyx": self.output_shape_zyx,
            "percentiles": self.percentiles,
            "output_range": (-1.0, 1.0),
        }

    def _path(self, visit: VisitRecord, backend: str) -> Path:
        safe_visit = visit.visit_id.replace(":", "_").replace("/", "_")
        return self.root / f"{safe_visit}_{backend}.pt"

    def load_or_create(
        self,
        visit: VisitRecord,
        *,
        backend: str,
        rebuild_stale: bool = True,
    ) -> PreparedVisit:
        identity = self._identity(visit, backend)
        path = self._path(visit, backend)
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            valid = (
                isinstance(payload, dict)
                and payload.get("identity") == identity
                and payload.get("image_sha256") == identity["dce0_sha256"]
            )
            if valid:
                return self._prepared(payload)
            if not rebuild_stale:
                raise ValueError("DCE0 cache identity does not match its source")

        payload = self._build(visit, backend, identity)
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return self._prepared(payload)

    def _build(
        self, visit: VisitRecord, backend: str, identity: dict[str, Any]
    ) -> dict[str, Any]:
        meta = json.loads(visit.meta_path.read_text())
        if meta.get("array_shape_policy") not in {None, "zyx_slices_rows_cols"}:
            raise ValueError("DCE0 array shape policy is unsupported")
        image = np.asarray(nib.load(visit.dce0.path).dataobj, dtype=np.float32)
        mask = np.asarray(nib.load(visit.mask_path).dataobj)
        if image.ndim != 3 or mask.shape != image.shape:
            raise ValueError("DCE0 image and mask grids do not match")
        spacing = _spacing_from_meta(meta)
        resampling = None
        if spacing is not None:
            image, mask, resampling = resample_image_and_mask(
                image, mask, source_spacing_zyx=spacing
            )
        normalized, normalization = robust_nonzero_scale(
            image, percentiles=self.percentiles
        )
        cropped_image, cropped_mask, crop = tumor_center_crop(
            normalized[None], mask, self.output_shape_zyx
        )
        return {
            "identity": identity,
            "visit_id": visit.visit_id,
            "image": torch.from_numpy(cropped_image.astype(np.float32, copy=False)),
            "mask": torch.from_numpy(cropped_mask.astype(np.uint8, copy=False)),
            "phase_index": visit.dce0.phase_index,
            "n_times": visit.dce0.n_times,
            "image_sha256": identity["dce0_sha256"],
            "metadata": {
                "data_backend": backend,
                "source_path": str(visit.dce0.path.resolve()),
                "normalization": normalization,
                "resampling": resampling,
                "crop": crop,
            },
        }

    @staticmethod
    def _prepared(payload: dict[str, Any]) -> PreparedVisit:
        return PreparedVisit(
            visit_id=payload["visit_id"],
            image=payload["image"],
            mask=payload["mask"],
            phase_index=int(payload["phase_index"]),
            n_times=int(payload["n_times"]),
            image_sha256=payload["image_sha256"],
            metadata=payload["metadata"],
        )


class MOTFMROICache:
    """Read verified MOTFM visit tensors without reimplementing preprocessing."""

    def __init__(
        self,
        root: str | Path,
        *,
        bundle_json: str | Path,
        output_shape_zyx: Sequence[int] = (128, 128, 128),
    ) -> None:
        root_path = Path(root)
        if root_path.is_symlink() or not root_path.is_dir():
            raise ValueError("MOTFM ROI cache directory is missing or unsafe")
        self.root = root_path.resolve()
        self.output_shape_zyx = tuple(int(value) for value in output_shape_zyx)
        if len(self.output_shape_zyx) != 3 or any(
            value <= 0 for value in self.output_shape_zyx
        ):
            raise ValueError("MOTFM ROI output shape must contain three positive values")

        bundle_path = Path(bundle_json)
        if bundle_path.is_symlink() or not bundle_path.is_file():
            raise ValueError("MOTFM bundle is missing or unsafe")
        self.bundle_path = bundle_path.resolve()
        bundle = json.loads(self.bundle_path.read_text())
        if bundle.get("schema_version") != "motfm_ispy2_longitudinal_bundle_v1":
            raise ValueError("MOTFM bundle schema is unsupported")
        self.bundle_contract_sha256 = bundle.get("bundle_contract_sha256")
        if not _is_sha256(self.bundle_contract_sha256):
            raise ValueError("MOTFM bundle contract SHA256 is invalid")

        visits_path, _ = self._locked_artifact(bundle, "visits")
        preprocess_path, preprocess_sha256 = self._locked_artifact(
            bundle, "preprocess"
        )
        preprocess = json.loads(preprocess_path.read_text())
        if (
            preprocess.get("schema_version") != "motfm_ispy2_preprocess_v1"
            or preprocess.get("array_order") != "zyx"
            or preprocess.get("target_coordinate_system") != "axis_aligned_RAS"
            or tuple(preprocess.get("target_spacing_xyz", ()))
            != MOTFM_TARGET_SPACING_XYZ
            or tuple(preprocess.get("output_shape_zyx", ()))
            != self.output_shape_zyx
            or preprocess.get("image_interpolation") != "linear"
            or preprocess.get("mask_interpolation") != "nearest"
            or preprocess.get("crop_policy")
            != "bbox_complete_tumor_centered_per_visit"
            or preprocess.get("full_resampled_mask_retention_required") is not True
            or preprocess.get("normalization", {}).get("output_range") != [0.0, 1.0]
        ):
            raise ValueError("MOTFM preprocessing contract is unsupported")
        self.bundle_preprocess_sha256 = preprocess_sha256
        self.effective_preprocess_sha256 = _canonical_sha256(preprocess)

        visits = pd.read_csv(
            visits_path, usecols=["visit_id", "study_instance_uid"]
        )
        if visits.empty or visits["visit_id"].duplicated().any():
            raise ValueError("MOTFM visits artifact has invalid identities")
        self._study_uids = {
            str(row.visit_id): str(row.study_instance_uid)
            for row in visits.itertuples(index=False)
        }

    def _locked_artifact(
        self, bundle: dict[str, Any], name: str
    ) -> tuple[Path, str]:
        descriptor = bundle.get("artifacts", {}).get(name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"MOTFM bundle is missing the {name} artifact")
        path = (self.bundle_path.parent / str(descriptor.get("path", ""))).resolve()
        expected = descriptor.get("sha256")
        if path.is_symlink() or not path.is_file() or not _is_sha256(expected):
            raise ValueError(f"MOTFM {name} artifact is missing or unsafe")
        if sha256_file(path) != expected:
            raise ValueError(f"MOTFM {name} artifact SHA256 does not match")
        return path, expected

    def cache_key(self, visit_id: str) -> str:
        try:
            study_uid = self._study_uids[visit_id]
        except KeyError:
            raise ValueError("visit is absent from the locked MOTFM bundle") from None
        identity_sha256 = _canonical_sha256(
            {"visit_id": visit_id, "study_instance_uid": study_uid}
        )
        return _canonical_sha256(
            {
                "schema_version": MOTFM_CACHE_SCHEMA_VERSION,
                "visit_identity_sha256": identity_sha256,
                "channel_order": list(MOTFM_CHANNEL_ORDER),
                "effective_preprocess_sha256": self.effective_preprocess_sha256,
                "output_shape_zyx": list(self.output_shape_zyx),
                "target_spacing_xyz": list(MOTFM_TARGET_SPACING_XYZ),
            }
        )

    def load(self, visit: VisitRecord) -> PreparedVisit:
        cache_key = self.cache_key(visit.visit_id)
        path = self.root / f"{cache_key}.pt"
        if path.is_symlink() or not path.is_file():
            raise ValueError("locked MOTFM ROI cache payload is missing or unsafe")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self._validate_payload(payload, visit=visit, cache_key=cache_key)
        source_hashes = payload["source_asset_sha256"]
        preprocess_metadata = payload["preprocess_metadata"]
        dce0 = payload["mri"][0:1].to(dtype=torch.float32)
        image = dce0.mul(2.0).sub(1.0).contiguous()
        mask = payload["mask"].clone().contiguous()
        normalization = dict(preprocess_metadata["normalization"]["dce0"])
        normalization.update(
            {
                "cached_output_range": [0.0, 1.0],
                "output_range": [-1.0, 1.0],
                "value_transform": "two_x_minus_one",
            }
        )
        return PreparedVisit(
            visit_id=visit.visit_id,
            image=image,
            mask=mask,
            phase_index=visit.dce0.phase_index,
            n_times=visit.dce0.n_times,
            image_sha256=source_hashes["dce0"],
            metadata={
                "data_backend": "current",
                "source_path": str(visit.dce0.path.resolve()),
                "normalization": normalization,
                "resampling": {
                    "source_shape_zyx": preprocess_metadata["source_shape_zyx"],
                    "resampled_shape_zyx": preprocess_metadata[
                        "resampled_shape_zyx"
                    ],
                    "target_spacing_xyz": preprocess_metadata[
                        "target_spacing_xyz"
                    ],
                },
                "crop": preprocess_metadata["crop"],
                "motfm_roi_cache": {
                    "schema_version": MOTFM_CACHE_SCHEMA_VERSION,
                    "cache_key": cache_key,
                    "bundle_contract_sha256": self.bundle_contract_sha256,
                    "bundle_preprocess_sha256": self.bundle_preprocess_sha256,
                    "effective_preprocess_sha256": self.effective_preprocess_sha256,
                },
            },
        )

    def _validate_payload(
        self,
        payload: Any,
        *,
        visit: VisitRecord,
        cache_key: str,
    ) -> None:
        if not isinstance(payload, dict) or set(payload) != MOTFM_PAYLOAD_KEYS:
            raise ValueError("MOTFM ROI cache payload fields do not match")
        expected = {
            "schema_version": MOTFM_CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "bundle_contract_sha256": self.bundle_contract_sha256,
            "bundle_preprocess_sha256": self.bundle_preprocess_sha256,
            "effective_preprocess_sha256": self.effective_preprocess_sha256,
            "channel_order": list(MOTFM_CHANNEL_ORDER),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("MOTFM ROI cache payload identity does not match")
        source_hashes = payload.get("source_asset_sha256")
        if (
            not isinstance(source_hashes, dict)
            or set(source_hashes) != MOTFM_SOURCE_ASSETS
            or any(not _is_sha256(value) for value in source_hashes.values())
            or source_hashes["dce0"] != sha256_file(visit.dce0.path)
            or source_hashes["mask"] != sha256_file(visit.mask_path)
            or source_hashes["meta"] != sha256_file(visit.meta_path)
        ):
            raise ValueError("MOTFM ROI cache source hashes do not match")
        source_contract = _canonical_sha256(
            {
                "source_asset_sha256": source_hashes,
                "source_asset_stat_signatures": payload.get(
                    "source_asset_stat_signatures"
                ),
            }
        )
        if payload.get("source_asset_contract_sha256") != source_contract:
            raise ValueError("MOTFM ROI cache source contract does not match")

        mri = payload.get("mri")
        mask = payload.get("mask")
        if (
            not isinstance(mri, torch.Tensor)
            or mri.dtype != torch.float16
            or tuple(mri.shape) != (2, *self.output_shape_zyx)
            or not bool(torch.isfinite(mri).all())
            or bool(((mri < 0) | (mri > 1)).any())
        ):
            raise ValueError("MOTFM ROI cache MRI tensor is invalid")
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.uint8
            or tuple(mask.shape) != (1, *self.output_shape_zyx)
            or not bool(((mask == 0) | (mask == 1)).all())
            or int(mask.sum()) <= 0
        ):
            raise ValueError("MOTFM ROI cache mask tensor is invalid")
        preprocess = payload.get("preprocess_metadata")
        crop = preprocess.get("crop") if isinstance(preprocess, dict) else None
        normalization = (
            preprocess.get("normalization") if isinstance(preprocess, dict) else None
        )
        if (
            not isinstance(crop, dict)
            or not isinstance(normalization, dict)
            or not isinstance(normalization.get("dce0"), dict)
            or preprocess.get("output_shape_zyx") != list(self.output_shape_zyx)
            or preprocess.get("target_spacing_xyz")
            != list(MOTFM_TARGET_SPACING_XYZ)
            or crop.get("mask_fully_retained") is not True
            or crop.get("mask_voxel_count_after_crop") != int(mask.sum())
            or crop.get("mask_voxel_count_before_crop") != int(mask.sum())
        ):
            raise ValueError("MOTFM ROI cache preprocessing metadata is invalid")


class RegisteredStrictAROICache:
    """Read the immutable registered Strict-A visit cache."""

    def __init__(
        self,
        root: str | Path,
        *,
        bundle_json: str | Path,
        output_shape_zyx: Sequence[int] = REGISTERED_STRICT_A_OUTPUT_SHAPE_ZYX,
    ) -> None:
        root_path = Path(root)
        if root_path.is_symlink() or not root_path.is_dir():
            raise ValueError(
                "registered Strict-A ROI cache directory is missing or unsafe"
            )
        self.root = root_path.resolve()
        self.output_shape_zyx = tuple(int(value) for value in output_shape_zyx)
        if self.output_shape_zyx != REGISTERED_STRICT_A_OUTPUT_SHAPE_ZYX:
            raise ValueError("registered Strict-A ROI output shape is incompatible")

        bundle_path = Path(bundle_json)
        if bundle_path.is_symlink() or not bundle_path.is_file():
            raise ValueError("registered Strict-A bundle is missing or unsafe")
        self.bundle_path = bundle_path.resolve()
        bundle = json.loads(self.bundle_path.read_text())
        if bundle.get("schema_version") != REGISTERED_STRICT_A_BUNDLE_SCHEMA:
            raise ValueError("registered Strict-A bundle schema is unsupported")
        self.bundle_contract_sha256 = validate_registered_strict_a_bundle_contract(
            bundle, bundle_path=self.bundle_path
        )

        contract = bundle.get("base_cache_contract")
        required_contract = {
            "schema",
            "bundle_contract_sha256",
            "preprocess_sha256",
            "normalization_sha256",
        }
        if not isinstance(contract, dict) or set(contract) != required_contract:
            raise ValueError("registered Strict-A base cache contract is invalid")
        if (
            contract.get("schema") != REGISTERED_STRICT_A_CACHE_SCHEMA
            or any(
                not _is_sha256(contract.get(key))
                for key in required_contract - {"schema"}
            )
        ):
            raise ValueError("registered Strict-A base cache contract is invalid")
        self.cache_contract = {
            key: str(contract[key]) for key in required_contract
        }

        visits_path, _ = self._locked_artifact(bundle, "visits")
        preprocess_path, preprocess_sha256 = self._locked_artifact(
            bundle, "preprocess"
        )
        normalization_path, normalization_sha256 = self._locked_artifact(
            bundle, "normalization"
        )
        crop_plans_path, _ = self._locked_artifact(bundle, "crop_plans")
        if (
            preprocess_sha256 != self.cache_contract["preprocess_sha256"]
            or normalization_sha256 != self.cache_contract["normalization_sha256"]
        ):
            raise ValueError(
                "registered Strict-A cache artifact contract is incompatible"
            )

        preprocess = json.loads(preprocess_path.read_text())
        if (
            preprocess.get("schema") != REGISTERED_STRICT_A_PREPROCESS_SCHEMA
            or preprocess.get("coordinate_frame")
            != REGISTERED_STRICT_A_COORDINATE_FRAME
            or preprocess.get("crop_policy")
            != "single_T0_mask_bbox_center_reused_for_all_visits"
            or preprocess.get("image_interpolation") != "linear"
            or preprocess.get("mask_interpolation") != "nearest"
            or tuple(preprocess.get("output_shape_zyx", ())) != self.output_shape_zyx
            or tuple(preprocess.get("target_spacing_xyz", ()))
            != MOTFM_TARGET_SPACING_XYZ
            or preprocess.get("normalization", {}).get("schema")
            != REGISTERED_STRICT_A_NORMALIZATION_SCHEMA
            or preprocess.get("normalization", {}).get("normalization_sha256")
            != normalization_sha256
        ):
            raise ValueError(
                "registered Strict-A preprocessing contract is unsupported"
            )
        normalization = json.loads(normalization_path.read_text())
        if normalization.get("schema") != REGISTERED_STRICT_A_NORMALIZATION_SCHEMA:
            raise ValueError(
                "registered Strict-A normalization contract is unsupported"
            )

        crop_plans = json.loads(crop_plans_path.read_text())
        if not isinstance(crop_plans, dict) or not crop_plans:
            raise ValueError("registered Strict-A crop plans are invalid")
        self._crop_plans = crop_plans
        for plan in self._crop_plans.values():
            if (
                not isinstance(plan, dict)
                or plan.get("coordinate_frame")
                != REGISTERED_STRICT_A_COORDINATE_FRAME
                or tuple(plan.get("output_shape_zyx", ())) != self.output_shape_zyx
            ):
                raise ValueError("registered Strict-A crop plan is invalid")

        visits = pd.read_csv(
            visits_path,
            usecols=[
                "patient_id",
                "visit_id",
                "visit",
                "dce0_path",
                "mask_path",
                "meta_path",
                "registration_status",
            ],
        )
        descriptor_count = int(bundle["artifacts"]["visits"].get("row_count", -1))
        bundle_count = int(bundle.get("counts", {}).get("visit_count", -1))
        if (
            visits.empty
            or visits["visit_id"].duplicated().any()
            or descriptor_count != len(visits)
            or bundle_count != len(visits)
        ):
            raise ValueError("registered Strict-A visits artifact is invalid")
        self._visit_identities = {
            str(row.visit_id): {
                "patient_id": str(row.patient_id),
                "visit": str(row.visit),
                "dce0_path": str(Path(row.dce0_path).resolve()),
                "mask_path": str(Path(row.mask_path).resolve()),
                "meta_path": str(Path(row.meta_path).resolve()),
                "registration_status": str(row.registration_status),
            }
            for row in visits.itertuples(index=False)
        }
        if set(self._crop_plans) != {
            identity["patient_id"] for identity in self._visit_identities.values()
        }:
            raise ValueError(
                "registered Strict-A crop plans do not match bundle patients"
            )

    def _locked_artifact(
        self, bundle: dict[str, Any], name: str
    ) -> tuple[Path, str]:
        descriptor = bundle.get("artifacts", {}).get(name)
        if not isinstance(descriptor, dict):
            raise ValueError(
                f"registered Strict-A bundle is missing the {name} artifact"
            )
        candidate = self.bundle_path.parent / str(descriptor.get("path", ""))
        expected = descriptor.get("sha256")
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or not _is_sha256(expected)
        ):
            raise ValueError(
                f"registered Strict-A {name} artifact is missing or unsafe"
            )
        path = candidate.resolve()
        if sha256_file(path) != expected:
            raise ValueError(
                f"registered Strict-A {name} artifact SHA256 does not match"
            )
        return path, str(expected)

    def cache_key(self, visit_id: str) -> str:
        if visit_id not in self._visit_identities:
            raise ValueError(
                "visit is absent from the locked registered Strict-A bundle"
            )
        return _canonical_sha256(
            {
                "schema": REGISTERED_STRICT_A_CACHE_SCHEMA,
                "visit_id": visit_id,
                "bundle_contract_sha256": self.cache_contract[
                    "bundle_contract_sha256"
                ],
                "preprocess_sha256": self.cache_contract["preprocess_sha256"],
                "normalization_sha256": self.cache_contract["normalization_sha256"],
            }
        )

    def _load_payload(self, visit: VisitRecord) -> tuple[dict[str, Any], str]:
        self._validate_visit(visit)
        cache_key = self.cache_key(visit.visit_id)
        candidate = self.root / f"{cache_key}.pt"
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(
                f"registered Strict-A cache is unavailable: {visit.visit_id}"
            )
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=True)
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            EOFError,
            pickle.UnpicklingError,
        ):
            raise ValueError(
                f"registered Strict-A cache is unavailable: {visit.visit_id}"
            ) from None
        self._validate_payload(payload, visit=visit, cache_key=cache_key)
        return payload, cache_key

    def load_source_mri(self, visit: VisitRecord) -> torch.Tensor:
        """Return the locked DCE0/SER source tensor without changing its dtype."""
        payload, _ = self._load_payload(visit)
        return payload["mri"].clone().contiguous()

    def load(self, visit: VisitRecord) -> PreparedVisit:
        payload, cache_key = self._load_payload(visit)
        return PreparedVisit(
            visit_id=visit.visit_id,
            image=payload["mri"][0:1].to(dtype=torch.float32).contiguous(),
            mask=payload["mask"].clone().contiguous(),
            phase_index=visit.dce0.phase_index,
            n_times=visit.dce0.n_times,
            image_sha256=cache_key,
            metadata={
                "data_backend": "registered_t0",
                "source_path": str(visit.dce0.path.resolve()),
                "coordinate_frame": REGISTERED_STRICT_A_COORDINATE_FRAME,
                "normalization": {
                    "schema": REGISTERED_STRICT_A_NORMALIZATION_SCHEMA,
                    "sha256": self.cache_contract["normalization_sha256"],
                    "value_transform": "none",
                },
                "resampling": {
                    "target_spacing_xyz": list(MOTFM_TARGET_SPACING_XYZ),
                    "output_shape_zyx": list(self.output_shape_zyx),
                },
                "crop": dict(payload["crop_plan"]),
                "registered_strict_a_roi_cache": {
                    "schema": REGISTERED_STRICT_A_CACHE_SCHEMA,
                    "cache_key": cache_key,
                    "bundle_contract_sha256": self.bundle_contract_sha256,
                    "cache_contract": dict(self.cache_contract),
                },
            },
            valid_foreground=payload["valid_foreground"].clone().contiguous(),
        )

    def _validate_visit(self, visit: VisitRecord) -> None:
        try:
            expected = self._visit_identities[visit.visit_id]
        except KeyError:
            raise ValueError(
                "visit is absent from the locked registered Strict-A bundle"
            ) from None
        actual = {
            "patient_id": visit.patient_id,
            "visit": visit.visit,
            "dce0_path": str(visit.dce0.path.resolve()),
            "mask_path": str(visit.mask_path.resolve()),
            "meta_path": str(visit.meta_path.resolve()),
            "registration_status": visit.registration_status,
        }
        if actual != expected:
            raise ValueError("registered Strict-A visit identity does not match")

    def _validate_payload(
        self, payload: Any, *, visit: VisitRecord, cache_key: str
    ) -> None:
        if (
            not isinstance(payload, dict)
            or set(payload) != REGISTERED_STRICT_A_PAYLOAD_KEYS
        ):
            raise ValueError("registered Strict-A cache fields are invalid")
        expected = {
            "schema": REGISTERED_STRICT_A_CACHE_SCHEMA,
            "cache_key": cache_key,
            "visit_id": visit.visit_id,
            "bundle_contract_sha256": self.cache_contract[
                "bundle_contract_sha256"
            ],
            "preprocess_sha256": self.cache_contract["preprocess_sha256"],
            "normalization_sha256": self.cache_contract["normalization_sha256"],
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("registered Strict-A cache identity is invalid")
        if payload.get("crop_plan") != self._crop_plans[visit.patient_id]:
            raise ValueError("registered Strict-A cache crop identity is invalid")
        mri = payload.get("mri")
        mask = payload.get("mask")
        valid = payload.get("valid_foreground")
        if (
            not isinstance(mri, torch.Tensor)
            or mri.dtype != torch.float16
            or tuple(mri.shape) != (2, *self.output_shape_zyx)
            or not bool(torch.isfinite(mri).all())
        ):
            raise ValueError("registered Strict-A cached MRI is invalid")
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.uint8
            or tuple(mask.shape) != (1, *self.output_shape_zyx)
            or not bool(((mask == 0) | (mask == 1)).all())
            or not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.uint8
            or tuple(valid.shape) != (1, *self.output_shape_zyx)
            or not bool(((valid == 0) | (valid == 1)).all())
        ):
            raise ValueError("registered Strict-A cached mask is invalid")
        outside_foreground = valid[0] == 0
        if bool((mri[:, outside_foreground] != 0).any()):
            raise ValueError(
                "registered Strict-A cached MRI outside-FOV values must be zero"
            )


class RegisteredPairCache:
    def __init__(
        self,
        root: str | Path,
        *,
        output_shape_zyx: Sequence[int] = (128, 128, 128),
        percentiles: Sequence[float] = NORMALIZATION_PERCENTILES,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_shape_zyx = tuple(int(value) for value in output_shape_zyx)
        self.percentiles = tuple(float(value) for value in percentiles)

    def _visit_identity(self, visit: VisitRecord) -> dict[str, Any]:
        if visit.registration_status not in {
            "fixed_reference",
            "rigid_fallback",
            "deformable",
        }:
            raise ValueError("registered pair contains a non-success visit")
        return {
            "visit_id": visit.visit_id,
            "image_path": str(visit.dce0.path.resolve()),
            "image_sha256": sha256_file(visit.dce0.path),
            "mask_sha256": sha256_file(visit.mask_path),
            "meta_sha256": sha256_file(visit.meta_path),
            "phase_index": visit.dce0.phase_index,
            "n_times": visit.dce0.n_times,
            "registration_status": visit.registration_status,
        }

    def _identity(self, source: VisitRecord, target: VisitRecord) -> dict[str, Any]:
        if source.patient_id != target.patient_id:
            raise ValueError("registered pair patient IDs do not match")
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "backend": "registered_t0",
            "source": self._visit_identity(source),
            "target": self._visit_identity(target),
            "output_shape_zyx": self.output_shape_zyx,
            "percentiles": self.percentiles,
        }

    def _path(self, source: VisitRecord, target: VisitRecord) -> Path:
        name = f"{source.visit_id}__{target.visit_id}".replace(":", "_").replace("/", "_")
        return self.root / f"{name}_registered_t0.pt"

    def load_or_create(
        self,
        source: VisitRecord,
        target: VisitRecord,
        *,
        rebuild_stale: bool = True,
    ) -> tuple[PreparedVisit, PreparedVisit]:
        identity = self._identity(source, target)
        path = self._path(source, target)
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and payload.get("identity") == identity:
                return (
                    DCE0Cache._prepared(payload["source"]),
                    DCE0Cache._prepared(payload["target"]),
                )
            if not rebuild_stale:
                raise ValueError("registered pair cache identity does not match its sources")
        payload = self._build(source, target, identity)
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        return (
            DCE0Cache._prepared(payload["source"]),
            DCE0Cache._prepared(payload["target"]),
        )

    def _build(
        self,
        source: VisitRecord,
        target: VisitRecord,
        identity: dict[str, Any],
    ) -> dict[str, Any]:
        source_image = np.asarray(nib.load(source.dce0.path).dataobj, dtype=np.float32)
        target_image = np.asarray(nib.load(target.dce0.path).dataobj, dtype=np.float32)
        source_mask = np.asarray(nib.load(source.mask_path).dataobj)
        target_mask = np.asarray(nib.load(target.mask_path).dataobj)
        if (
            source_image.shape != target_image.shape
            or source_mask.shape != source_image.shape
            or target_mask.shape != target_image.shape
        ):
            raise ValueError("registered source and target must share one voxel grid")
        source_meta = json.loads(source.meta_path.read_text())
        spacing = _spacing_from_meta(source_meta)
        resampling = None
        if spacing is not None:
            source_image, source_mask, resampling = resample_image_and_mask(
                source_image, source_mask, source_spacing_zyx=spacing
            )
            target_image, target_mask, target_resampling = resample_image_and_mask(
                target_image, target_mask, source_spacing_zyx=spacing
            )
            if target_resampling != resampling:
                raise RuntimeError("registered pair resampling plans diverged")
        source_scaled, source_normalization = robust_nonzero_scale(
            source_image, percentiles=self.percentiles
        )
        target_scaled, target_normalization = robust_nonzero_scale(
            target_image, percentiles=self.percentiles
        )
        crop = source_centered_pair_crop(
            source_scaled[None],
            source_mask,
            target_scaled[None],
            target_mask,
            self.output_shape_zyx,
        )

        def visit_payload(
            visit: VisitRecord,
            image: np.ndarray,
            mask: np.ndarray,
            normalization: dict[str, Any],
        ) -> dict[str, Any]:
            return {
                "visit_id": visit.visit_id,
                "image": torch.from_numpy(image.astype(np.float32, copy=False)),
                "mask": torch.from_numpy(mask.astype(np.uint8, copy=False)),
                "phase_index": visit.dce0.phase_index,
                "n_times": visit.dce0.n_times,
                "image_sha256": sha256_file(visit.dce0.path),
                "metadata": {
                    "data_backend": "registered_t0",
                    "source_path": str(visit.dce0.path.resolve()),
                    "registration_status": visit.registration_status,
                    "normalization": normalization,
                    "resampling": resampling,
                    "crop": crop.metadata,
                },
            }

        return {
            "identity": identity,
            "source": visit_payload(
                source, crop.source_image, crop.source_mask, source_normalization
            ),
            "target": visit_payload(
                target, crop.target_image, crop.target_mask, target_normalization
            ),
        }
