"""Import the audited MeWM registered ROI cache into the prepared-volume contract.

The upstream cache is patient-paired, registered to T0, and cropped from one
T0-mask-derived plan.  It is therefore deliberately represented as
registration-assisted research data, not as source-only preprocessing.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ispy2_symmflow.utils.hashing import sha256_file, stable_hash

from .manifest import build_pair_manifest, write_jsonl
from .schema import PairRecord, QCEvent, VisitRecord


BUNDLE_SCHEMA = "motfm_ispy2_registered_strict_a_bundle_v1"
CACHE_SCHEMA = "motfm_ispy2_t0_fixed_roi_cache_v1"
PREPROCESS_SCHEMA = "motfm_ispy2_registered_preprocess_v1"
NORMALIZATION_SCHEMA = "motfm_ispy2_train_foreground_stats_v1"
COORDINATE_FRAME = "t0_fixed_registered_local_v1"
CROP_POLICY = "single_T0_mask_bbox_center_reused_for_all_visits"
IMPORT_PROVENANCE_VERSION = "mewm-registered-roi-import-v1"
NORMALIZATION_SCOPE = "unique_train_transition_visits_per_channel"

_ARTIFACT_NAMES = frozenset(
    {
        "crop_plans",
        "exclusions",
        "normalization",
        "preprocess",
        "split",
        "transitions",
        "visits",
    }
)
_CACHE_KEYS = frozenset(
    {
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
)
_VISIT_COLUMNS = frozenset(
    {
        "patient_id",
        "visit_id",
        "visit",
        "study_instance_uid",
        "visit_date",
        "visit_date_source",
        "qc_status",
        "dce0_path",
        "ser_path",
        "mask_path",
        "meta_path",
        "row_spacing_mm",
        "column_spacing_mm",
        "slice_spacing_mm",
        "ftv_volume_cc",
        "HR",
        "HER2",
        "MP",
        "Age_at_Screening",
        "menopausal_status",
        "trial_arm",
        "clinical_text",
        "fold",
        "orientation_lps_json",
        "resampled_shape_zyx_json",
        "crop_start_zyx_json",
        "registration_status",
        "quality_pass",
    }
)
_TRANSITION_COLUMNS = frozenset(
    {
        "transition_id",
        "patient_id",
        "fold",
        "transition_type",
        "source_visit_id",
        "target_visit_id",
        "source_visit",
        "target_visit",
        "source_study_instance_uid",
        "target_study_instance_uid",
        "delta_days",
        "source_dce0_path",
        "source_ser_path",
        "source_mask_path",
        "source_meta_path",
        "target_dce0_path",
        "target_mask_path",
        "target_meta_path",
        "source_ftv_volume_cc",
        "target_ftv_volume_cc",
        "source_ftv_is_condition",
        "target_ftv_is_condition",
        "target_ftv_is_audit_only",
        "clinical_text",
        "action_text",
    }
)
_EXCLUSION_COLUMNS = frozenset(
    {
        "transition_id",
        "patient_id",
        "source_visit_id",
        "target_visit_id",
        "transition_type",
        "reason",
        "detail",
    }
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_ALLOWED_PHASE_ROLES = (("dce0",), ("dce0", "ser"))


@dataclass(frozen=True)
class MewmImportResult:
    visits: tuple[VisitRecord, ...]
    pairs: tuple[PairRecord, ...]
    visit_manifest_path: str
    pair_manifest_path: str
    intensity_stats_path: str
    audit_path: str
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class MewmVisitGeometry:
    """Immutable physical-grid evidence for one signed MeWM visit."""

    visit_id: str
    patient_id: str
    visit_stage: str
    split: str
    study_uid: str
    affine_lps: tuple[tuple[float, ...], ...]
    base_affine_lps: tuple[tuple[float, ...], ...]
    spacing_dhw: tuple[float, float, float]
    crop_start_zyx: tuple[int, int, int]
    output_shape_zyx: tuple[int, int, int]
    meta_sha256: str
    registration_sha256: str
    transform_artifacts: tuple[tuple[str, str], ...]
    source_geometry_sha256: str
    registered_grid_geometry_sha256: str


@dataclass(frozen=True)
class MewmConnectedPairGeometry:
    """One direct or adjacent-edge-connected MeWM time pair."""

    pair_id: str
    patient_id: str
    split: str
    earlier_stage: str
    later_stage: str
    earlier_visit_id: str
    later_visit_id: str
    delta_days: int
    action_text: str
    edge_transition_ids: tuple[str, ...]
    earlier: MewmVisitGeometry
    later: MewmVisitGeometry


@dataclass(frozen=True)
class MewmMetadataValidationResult:
    """Read-only result of signed bundle and staged-geometry validation."""

    bundle_contract_sha256: str
    bundle_json_sha256: str
    artifact_sha256: Mapping[str, str]
    time_pairs: tuple[tuple[str, str], ...]
    output_shape_zyx: tuple[int, int, int]
    spacing_dhw: tuple[float, float, float]
    visits: tuple[MewmVisitGeometry, ...]
    pairs: tuple[MewmConnectedPairGeometry, ...]
    visit_geometry_by_id: Mapping[str, MewmVisitGeometry]
    pair_geometry_by_endpoints: Mapping[
        tuple[str, str], MewmConnectedPairGeometry
    ]


@dataclass(frozen=True)
class _BundleData:
    root: Path
    document: Mapping[str, Any]
    artifact_paths: Mapping[str, Path]
    artifact_hashes: Mapping[str, str]
    visits: tuple[dict[str, str], ...]
    transitions: tuple[dict[str, str], ...]
    exclusions: tuple[dict[str, str], ...]
    crop_plans: Mapping[str, Mapping[str, Any]]
    normalization: Mapping[str, Any]
    preprocess: Mapping[str, Any]
    split_by_patient: Mapping[str, str]


@dataclass(frozen=True)
class _RegistrationEvidence:
    base_affine_lps: np.ndarray
    cropped_affine_lps: np.ndarray
    meta_sha256: str
    registration_sha256: str
    transform_artifacts: tuple[Mapping[str, str], ...]
    source_geometry_sha256: str
    registered_grid_geometry_sha256: str


@dataclass(frozen=True)
class _CachePayload:
    image: np.ndarray
    cache_key: str
    cache_sha256: str
    crop_plan: Mapping[str, Any]


def _required_text(value: Mapping[str, Any], key: str, *, label: str) -> str:
    raw = value.get(key)
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise ValueError(f"{label} is missing non-empty {key!r}")
    return text


def _sha256(value: Any, *, label: str) -> str:
    text = str(value).strip().lower()
    if _HEX_SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _artifact_path(root: Path, value: Any, *, label: str) -> Path:
    raw = Path(str(value))
    if raw.is_absolute():
        raise ValueError(f"{label} path must be relative to the bundle")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the bundle directory") from exc
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular bundle file: {path}")
    return path


def _read_csv(
    path: Path, *, required_columns: frozenset[str], label: str
) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(required_columns - columns)
        if missing:
            raise ValueError(f"{label} is missing required columns: {missing}")
        rows = tuple(dict(row) for row in reader)
    return rows


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result < 1 or str(result) != str(value).strip():
        raise ValueError(f"{label} must be a positive integer")
    return result


def _parse_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label} must be a boolean")


def _float_array(value: Any, shape: tuple[int, ...], *, label: str) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is not valid JSON") from exc
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _int_triplet(value: Any, *, label: str, positive: bool = False) -> tuple[int, int, int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain three integers")
    if len(value) != 3:
        raise ValueError(f"{label} must contain three integers")
    result: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or int(raw) != raw:
            raise ValueError(f"{label} must contain three integers")
        result.append(int(raw))
    if positive and any(item < 1 for item in result):
        raise ValueError(f"{label} must contain positive integers")
    return tuple(result)  # type: ignore[return-value]


def _validate_bundle(bundle_dir: str | Path) -> _BundleData:
    root = Path(bundle_dir).expanduser().resolve()
    bundle_path = root / "bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    document = _read_json(bundle_path, label="MeWM bundle manifest")
    if document.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError(f"unsupported MeWM bundle schema: {document.get('schema_version')!r}")

    observed_contract = _sha256(
        document.get("bundle_contract_sha256"), label="bundle_contract_sha256"
    )
    unsigned = {key: item for key, item in document.items() if key != "bundle_contract_sha256"}
    if stable_hash(unsigned) != observed_contract:
        raise ValueError("bundle_contract_sha256 does not match bundle.json contents")

    artifacts = document.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _ARTIFACT_NAMES:
        raise ValueError(
            "bundle artifacts must contain exactly " + ", ".join(sorted(_ARTIFACT_NAMES))
        )
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    for name in sorted(_ARTIFACT_NAMES):
        entry = artifacts[name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"bundle artifact {name!r} must be an object")
        path = _artifact_path(root, entry.get("path"), label=f"bundle artifact {name!r}")
        expected = _sha256(entry.get("sha256"), label=f"artifact {name!r} sha256")
        if sha256_file(path) != expected:
            raise ValueError(f"bundle artifact {name!r} SHA-256 mismatch")
        artifact_paths[name] = path
        artifact_hashes[name] = expected

    visits = _read_csv(
        artifact_paths["visits"], required_columns=_VISIT_COLUMNS, label="visits.csv"
    )
    transitions = _read_csv(
        artifact_paths["transitions"],
        required_columns=_TRANSITION_COLUMNS,
        label="transitions.csv",
    )
    exclusions = _read_csv(
        artifact_paths["exclusions"],
        required_columns=_EXCLUSION_COLUMNS,
        label="exclusions.csv",
    )
    for name, rows in (
        ("visits", visits),
        ("transitions", transitions),
        ("exclusions", exclusions),
    ):
        entry = artifacts[name]
        assert isinstance(entry, Mapping)
        expected_count = _positive_int(entry.get("row_count"), label=f"{name} row_count")
        if len(rows) != expected_count:
            raise ValueError(
                f"{name} row count mismatch: observed={len(rows)}, expected={expected_count}"
            )

    crop_plans_raw = _read_json(artifact_paths["crop_plans"], label="crop_plans.json")
    if not crop_plans_raw or any(
        not isinstance(value, Mapping) for value in crop_plans_raw.values()
    ):
        raise ValueError("crop_plans.json must map patients to crop-plan objects")
    crop_plans = {str(key): dict(value) for key, value in crop_plans_raw.items()}
    preprocess = _read_json(artifact_paths["preprocess"], label="preprocess.json")
    normalization = _read_json(
        artifact_paths["normalization"], label="normalization.json"
    )
    split = _read_json(artifact_paths["split"], label="split.json")

    if preprocess.get("schema") != PREPROCESS_SCHEMA:
        raise ValueError("unsupported MeWM preprocessing schema")
    if preprocess.get("coordinate_frame") != COORDINATE_FRAME:
        raise ValueError("MeWM preprocessing coordinate_frame is not T0-fixed")
    if preprocess.get("crop_policy") != CROP_POLICY:
        raise ValueError("MeWM preprocessing crop policy is not the audited T0-mask policy")
    output_shape = _int_triplet(
        preprocess.get("output_shape_zyx"), label="preprocess output_shape_zyx", positive=True
    )
    if output_shape != (96, 256, 256):
        raise ValueError(f"unsupported MeWM ROI shape: {output_shape}")
    spacing_xyz = _float_array(
        preprocess.get("target_spacing_xyz"), (3,), label="preprocess target_spacing_xyz"
    )
    if np.any(spacing_xyz <= 0):
        raise ValueError("preprocess target_spacing_xyz must be positive")
    normalization_contract = preprocess.get("normalization")
    if not isinstance(normalization_contract, Mapping):
        raise ValueError("preprocess normalization contract is missing")
    if normalization_contract.get("schema") != NORMALIZATION_SCHEMA:
        raise ValueError("preprocess normalization schema is unsupported")
    if normalization_contract.get("scope") != "unique_train_transition_visits":
        raise ValueError("preprocess normalization scope is not train-transition-only")
    if normalization_contract.get("foreground") != "finite_nonzero_DCE0":
        raise ValueError("preprocess foreground normalization contract changed")
    if normalization_contract.get("SER_foreground") != "same_as_DCE0":
        raise ValueError("preprocess SER foreground contract changed")
    if int(normalization_contract.get("outside_fov_and_padding", -1)) != 0:
        raise ValueError("preprocess padding contract changed")

    base_cache = document.get("base_cache_contract")
    if not isinstance(base_cache, Mapping) or base_cache.get("schema") != CACHE_SCHEMA:
        raise ValueError("bundle base_cache_contract is missing or unsupported")
    base_bundle_sha = _sha256(
        base_cache.get("bundle_contract_sha256"),
        label="base_cache_contract bundle_contract_sha256",
    )
    preprocess_sha = _sha256(
        base_cache.get("preprocess_sha256"), label="base_cache_contract preprocess_sha256"
    )
    normalization_sha = _sha256(
        base_cache.get("normalization_sha256"),
        label="base_cache_contract normalization_sha256",
    )
    if preprocess_sha != artifact_hashes["preprocess"]:
        raise ValueError("base cache preprocessing SHA differs from the bundle artifact")
    if normalization_sha != artifact_hashes["normalization"]:
        raise ValueError("base cache normalization SHA differs from the bundle artifact")
    if normalization_contract.get("normalization_sha256") != normalization_sha:
        raise ValueError("preprocess normalization SHA differs from the bundle contract")
    # Validating this value as a digest also keeps it from being silently blank.
    _sha256(base_bundle_sha, label="base cache bundle contract")

    if normalization.get("schema") != NORMALIZATION_SCHEMA:
        raise ValueError("normalization.json schema is unsupported")
    channels = normalization.get("channels")
    if not isinstance(channels, Mapping) or set(channels) != {"dce0", "ser"}:
        raise ValueError("normalization channels must be exactly dce0 and ser")
    for name in ("dce0", "ser"):
        stats = channels[name]
        if not isinstance(stats, Mapping):
            raise ValueError(f"normalization channel {name!r} is invalid")
        count = stats.get("count")
        mean = stats.get("mean")
        std = stats.get("std")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"normalization channel {name!r} has invalid count")
        try:
            mean_float, std_float = float(mean), float(std)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"normalization channel {name!r} is not numeric") from exc
        if not math.isfinite(mean_float) or not math.isfinite(std_float) or std_float <= 0:
            raise ValueError(f"normalization channel {name!r} has invalid mean/std")
    fit_visits = normalization.get("visit_ids")
    if not isinstance(fit_visits, list) or any(
        not isinstance(item, str) or not item.strip() for item in fit_visits
    ):
        raise ValueError("normalization visit_ids must be a non-empty string list")
    if not fit_visits or len(fit_visits) != len(set(fit_visits)):
        raise ValueError("normalization visit_ids are empty or duplicated")

    if set(split) != {"train", "val"}:
        raise ValueError("split.json must contain exactly train and val patient lists")
    split_by_patient: dict[str, str] = {}
    for split_name in ("train", "val"):
        patients = split[split_name]
        if not isinstance(patients, list) or any(
            not isinstance(item, str) or not item.strip() for item in patients
        ):
            raise ValueError(f"split {split_name!r} must be a patient-ID list")
        for patient_id in patients:
            if patient_id in split_by_patient:
                raise ValueError(f"patient {patient_id!r} appears in multiple splits")
            split_by_patient[patient_id] = split_name
    if set(crop_plans) != set(split_by_patient):
        raise ValueError("crop plans and patient split do not cover the same cohort")

    visit_ids: set[str] = set()
    visit_patients: set[str] = set()
    for row_number, row in enumerate(visits, start=2):
        label = f"visits.csv row {row_number}"
        visit_id = _required_text(row, "visit_id", label=label)
        patient_id = _required_text(row, "patient_id", label=label)
        if visit_id in visit_ids:
            raise ValueError(f"visits.csv duplicates visit_id {visit_id!r}")
        visit_ids.add(visit_id)
        visit_patients.add(patient_id)
        if patient_id not in split_by_patient:
            raise ValueError(f"visit patient {patient_id!r} is absent from split.json")
        if row.get("fold") != split_by_patient[patient_id]:
            raise ValueError(f"visit {visit_id!r} fold differs from split.json")
        if row.get("qc_status") != "ok" or not _parse_bool(
            row.get("quality_pass"), label=f"visit {visit_id!r} quality_pass"
        ):
            raise ValueError(f"strict-A visit {visit_id!r} is not quality passing")
        stage = _required_text(row, "visit", label=label)
        if visit_id != f"{patient_id}:{stage}":
            raise ValueError(f"visit {visit_id!r} violates the patient:stage identity contract")
        plan = crop_plans[patient_id]
        if _int_triplet(
            row.get("crop_start_zyx_json"), label=f"visit {visit_id!r} crop start"
        ) != _int_triplet(plan.get("crop_start_zyx"), label=f"crop plan {patient_id!r}"):
            raise ValueError(f"visit {visit_id!r} crop start differs from crop_plans.json")
        if _int_triplet(
            row.get("resampled_shape_zyx_json"),
            label=f"visit {visit_id!r} resampled shape",
            positive=True,
        ) != _int_triplet(
            plan.get("input_shape_zyx"),
            label=f"crop plan {patient_id!r} input shape",
            positive=True,
        ):
            raise ValueError(f"visit {visit_id!r} shape differs from crop_plans.json")
        if plan.get("coordinate_frame") != COORDINATE_FRAME:
            raise ValueError(f"crop plan {patient_id!r} has the wrong coordinate frame")
        if _int_triplet(
            plan.get("output_shape_zyx"),
            label=f"crop plan {patient_id!r} output shape",
            positive=True,
        ) != output_shape:
            raise ValueError(f"crop plan {patient_id!r} output shape changed")
    if visit_patients != set(split_by_patient):
        raise ValueError("visits.csv and split.json do not cover the same patients")
    missing_fit_visits = set(fit_visits) - visit_ids
    if missing_fit_visits:
        raise ValueError("normalization references visits absent from visits.csv")
    visits_by_id = {row["visit_id"]: row for row in visits}
    if any(visits_by_id[item]["fold"] != "train" for item in fit_visits):
        raise ValueError("normalization was fitted with a non-training visit")

    transition_ids: set[str] = set()
    endpoint_pairs: set[tuple[str, str]] = set()
    for row_number, row in enumerate(transitions, start=2):
        label = f"transitions.csv row {row_number}"
        transition_id = _required_text(row, "transition_id", label=label)
        if transition_id in transition_ids:
            raise ValueError(f"transitions.csv duplicates transition_id {transition_id!r}")
        transition_ids.add(transition_id)
        source_id = _required_text(row, "source_visit_id", label=label)
        target_id = _required_text(row, "target_visit_id", label=label)
        endpoints = (source_id, target_id)
        if endpoints in endpoint_pairs:
            raise ValueError(f"transitions.csv duplicates endpoints {endpoints!r}")
        endpoint_pairs.add(endpoints)
        if source_id not in visits_by_id or target_id not in visits_by_id:
            raise ValueError(f"transition {transition_id!r} references an unknown visit")
        source, target = visits_by_id[source_id], visits_by_id[target_id]
        patient_id = _required_text(row, "patient_id", label=label)
        if source["patient_id"] != patient_id or target["patient_id"] != patient_id:
            raise ValueError(f"transition {transition_id!r} crosses patients")
        if row.get("fold") != split_by_patient[patient_id]:
            raise ValueError(f"transition {transition_id!r} fold differs from split.json")
        expected_type = f"{source['visit']}->{target['visit']}"
        if row.get("transition_type") != expected_type:
            raise ValueError(f"transition {transition_id!r} has inconsistent stage semantics")
        if transition_id != f"{patient_id}:{expected_type}":
            raise ValueError(f"transition {transition_id!r} violates its identity contract")
        expected_fields = {
            "source_visit": source["visit"],
            "target_visit": target["visit"],
            "source_study_instance_uid": source["study_instance_uid"],
            "target_study_instance_uid": target["study_instance_uid"],
            "source_dce0_path": source["dce0_path"],
            "source_ser_path": source["ser_path"],
            "source_mask_path": source["mask_path"],
            "source_meta_path": source["meta_path"],
            "target_dce0_path": target["dce0_path"],
            "target_mask_path": target["mask_path"],
            "target_meta_path": target["meta_path"],
            "source_ftv_volume_cc": source["ftv_volume_cc"],
            "target_ftv_volume_cc": target["ftv_volume_cc"],
            "clinical_text": source["clinical_text"],
            "action_text": f"treatment arm {source['trial_arm']}",
        }
        for field, expected in expected_fields.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"transition {transition_id!r} {field} differs from visits.csv"
                )
        if not _parse_bool(
            row.get("source_ftv_is_condition"),
            label=f"transition {transition_id!r} source_ftv_is_condition",
        ):
            raise ValueError(f"transition {transition_id!r} source FTV contract changed")
        if _parse_bool(
            row.get("target_ftv_is_condition"),
            label=f"transition {transition_id!r} target_ftv_is_condition",
        ):
            raise ValueError(f"transition {transition_id!r} leaks target FTV as a condition")
        if not _parse_bool(
            row.get("target_ftv_is_audit_only"),
            label=f"transition {transition_id!r} target_ftv_is_audit_only",
        ):
            raise ValueError(f"transition {transition_id!r} target FTV is not audit-only")

    exclusion_ids: set[str] = set()
    for row_number, row in enumerate(exclusions, start=2):
        transition_id = _required_text(
            row, "transition_id", label=f"exclusions.csv row {row_number}"
        )
        if transition_id in exclusion_ids:
            raise ValueError(f"exclusions.csv duplicates transition_id {transition_id!r}")
        exclusion_ids.add(transition_id)
    if transition_ids & exclusion_ids:
        raise ValueError("accepted transitions and exclusions overlap")

    counts = document.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("bundle counts are missing")
    observed_counts = {
        "visit_count": len(visits),
        "transition_count": len(transitions),
        "exclusion_count": len(exclusions),
        "patient_count": len(split_by_patient),
    }
    for key, observed in observed_counts.items():
        if int(counts.get(key, -1)) != observed:
            raise ValueError(f"bundle {key} differs from validated artifacts")

    return _BundleData(
        root=root,
        document=document,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        visits=visits,
        transitions=transitions,
        exclusions=exclusions,
        crop_plans=crop_plans,
        normalization=normalization,
        preprocess=preprocess,
        split_by_patient=split_by_patient,
    )


def _cache_contract(bundle: _BundleData, visit_id: str) -> dict[str, str]:
    base = bundle.document["base_cache_contract"]
    assert isinstance(base, Mapping)
    return {
        "schema": CACHE_SCHEMA,
        "visit_id": visit_id,
        "bundle_contract_sha256": str(base["bundle_contract_sha256"]),
        "preprocess_sha256": str(base["preprocess_sha256"]),
        "normalization_sha256": str(base["normalization_sha256"]),
    }


def _cache_path(bundle: _BundleData, roi_cache_dir: Path, visit_id: str) -> tuple[Path, str]:
    key = stable_hash(_cache_contract(bundle, visit_id))
    path = roi_cache_dir / f"{key}.pt"
    return path, key


def _validate_cache_payload(
    bundle: _BundleData,
    roi_cache_dir: Path,
    row: Mapping[str, str],
) -> _CachePayload:
    visit_id = row["visit_id"]
    path, cache_key = _cache_path(bundle, roi_cache_dir, visit_id)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - training extra supplies torch
        raise RuntimeError("PyTorch is required to read the MeWM ROI cache") from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot safely load ROI cache for visit {visit_id!r}") from exc
    if not isinstance(payload, dict) or set(payload) != _CACHE_KEYS:
        observed = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"ROI cache {visit_id!r} has unexpected payload keys: {observed}")
    expected_scalars = {**_cache_contract(bundle, visit_id), "cache_key": cache_key}
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise ValueError(f"ROI cache {visit_id!r} has mismatched {key}")
    if path.name != f"{cache_key}.pt":
        raise ValueError(f"ROI cache {visit_id!r} filename differs from its cache key")

    expected_spatial = tuple(
        _int_triplet(
            bundle.preprocess["output_shape_zyx"], label="preprocess output shape", positive=True
        )
    )
    tensors = {
        "mri": (torch.float16, (2, *expected_spatial)),
        "mask": (torch.uint8, (1, *expected_spatial)),
        "valid_foreground": (torch.uint8, (1, *expected_spatial)),
    }
    for name, (dtype, shape) in tensors.items():
        value = payload[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != dtype
            or tuple(value.shape) != shape
        ):
            raise ValueError(
                f"ROI cache {visit_id!r} {name} must be {dtype} with shape {shape}"
            )
        if value.device.type != "cpu":
            raise ValueError(f"ROI cache {visit_id!r} {name} must be stored on CPU")
    if not torch.isfinite(payload["mri"]).all():
        raise ValueError(f"ROI cache {visit_id!r} MRI contains non-finite values")
    for name in ("mask", "valid_foreground"):
        values = torch.unique(payload[name])
        if any(int(item) not in (0, 1) for item in values):
            raise ValueError(f"ROI cache {visit_id!r} {name} is not binary")

    crop_plan = payload["crop_plan"]
    patient_plan = bundle.crop_plans[row["patient_id"]]
    if not isinstance(crop_plan, Mapping) or stable_hash(crop_plan) != stable_hash(patient_plan):
        raise ValueError(f"ROI cache {visit_id!r} crop plan differs from the bundle")
    return _CachePayload(
        image=payload["mri"].numpy().astype(np.float32, copy=True),
        cache_key=cache_key,
        cache_sha256=sha256_file(path),
        crop_plan=dict(crop_plan),
    )


def _nested_values(value: Any, key: str) -> list[Any]:
    result: list[Any] = []
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if str(child_key) == key:
                result.append(child)
            result.extend(_nested_values(child, key))
    elif isinstance(value, list):
        for child in value:
            result.extend(_nested_values(child, key))
    return result


def _consistent_nested_text(value: Mapping[str, Any], key: str, *, label: str) -> str:
    values = {str(item).strip() for item in _nested_values(value, key) if str(item).strip()}
    if len(values) != 1:
        raise ValueError(f"{label} must contain one consistent {key!r}")
    return next(iter(values))


def _canonical_upstream_patient_id(value: str) -> str:
    """Normalize the two patient prefixes used by the staged MeWM exports."""

    prefix = "ACRIN-6698-"
    return f"ISPY2-{value[len(prefix):]}" if value.startswith(prefix) else value


def _metadata_directory(root: Path, row: Mapping[str, str]) -> Path:
    path = (root / row["patient_id"] / row["visit"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("upstream metadata identity escapes its staging directory") from exc
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"upstream metadata directory is missing: {path}")
    return path


def _geometry_contract(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    direction = _float_array(value.get("direction"), (9,), label=f"{label} direction")
    spacing = _float_array(value.get("spacing_xyz"), (3,), label=f"{label} spacing")
    shape = _int_triplet(value.get("shape_zyx"), label=f"{label} shape", positive=True)
    if np.any(spacing <= 0):
        raise ValueError(f"{label} spacing must be positive")
    direction[np.abs(direction) < 1e-12] = 0.0
    return {
        "direction": direction.tolist(),
        "shape_zyx": list(shape),
        "spacing_xyz": spacing.tolist(),
    }


def _geometry_from_metadata(
    bundle: _BundleData,
    metadata_root: Path,
    row: Mapping[str, str],
) -> _RegistrationEvidence:
    directory = _metadata_directory(metadata_root, row)
    meta_path = directory / "meta.json"
    registration_path = directory / "registration.json"
    if not meta_path.is_file() or meta_path.is_symlink():
        raise ValueError(f"upstream meta.json is missing for visit {row['visit_id']!r}")
    if not registration_path.is_file() or registration_path.is_symlink():
        raise ValueError(f"upstream registration.json is missing for visit {row['visit_id']!r}")
    meta = _read_json(meta_path, label=f"upstream metadata for {row['visit_id']!r}")
    registration = _read_json(
        registration_path, label=f"upstream registration for {row['visit_id']!r}"
    )

    for document, label in ((meta, "metadata"), (registration, "registration")):
        patient_values = {
            _canonical_upstream_patient_id(str(item).strip())
            for item in _nested_values(document, "patient_id")
            if str(item).strip()
        }
        if patient_values != {row["patient_id"]}:
            raise ValueError(f"upstream {label} patient_id differs from visits.csv")
        visit_values = _nested_values(document, "visit_id")
        if visit_values:
            visits = {str(item).strip() for item in visit_values if str(item).strip()}
            if visits != {row["visit_id"]}:
                raise ValueError(f"upstream {label} visit_id differs from visits.csv")
        stage_values = _nested_values(document, "visit")
        if stage_values:
            stages = {str(item).strip() for item in stage_values if str(item).strip()}
            if stages != {row["visit"]}:
                raise ValueError(f"upstream {label} visit stage differs from visits.csv")

    registered_to = _consistent_nested_text(
        meta, "registered_to_visit", label="upstream metadata"
    )
    if registered_to not in {"T0", f"{row['patient_id']}:T0"}:
        raise ValueError(f"visit {row['visit_id']!r} is not registered to T0")
    study_uid = meta.get("study_uid", meta.get("study_instance_uid"))
    if study_uid is not None and str(study_uid).strip() != row["study_instance_uid"]:
        raise ValueError(f"visit {row['visit_id']!r} study UID differs from upstream metadata")
    if meta.get("qc_status") != row["qc_status"]:
        raise ValueError(f"visit {row['visit_id']!r} QC status differs from upstream metadata")
    if meta.get("registration_status") != row["registration_status"]:
        raise ValueError(
            f"visit {row['visit_id']!r} registration status differs from upstream metadata"
        )
    status_values = {
        str(item).strip()
        for key in ("registration_status", "status")
        for item in _nested_values(registration, key)
        if str(item).strip()
    }
    if row["registration_status"] not in status_values:
        raise ValueError(f"visit {row['visit_id']!r} registration status is inconsistent")

    source_geometry = meta.get("source_geometry")
    registered_geometry = meta.get("target_geometry")
    if not isinstance(source_geometry, Mapping) or not isinstance(registered_geometry, Mapping):
        raise ValueError("upstream metadata must include source_geometry and target_geometry")
    canonical_source_geometry = _geometry_contract(
        source_geometry, label=f"visit {row['visit_id']!r} source geometry"
    )
    canonical_registered_geometry = _geometry_contract(
        registered_geometry, label=f"visit {row['visit_id']!r} registered geometry"
    )
    registered_shape_zyx = _int_triplet(
        canonical_registered_geometry["shape_zyx"],
        label=f"visit {row['visit_id']!r} registered shape",
        positive=True,
    )
    metadata_shape_zyx = _int_triplet(
        (meta.get("n_slices"), meta.get("rows"), meta.get("cols")),
        label=f"visit {row['visit_id']!r} metadata shape",
        positive=True,
    )
    if metadata_shape_zyx != registered_shape_zyx:
        raise ValueError(
            f"visit {row['visit_id']!r} registered shape differs from upstream metadata"
        )
    orientation = _float_array(
        meta.get("image_orientation_patient"),
        (6,),
        label=f"visit {row['visit_id']!r} image_orientation_patient",
    )
    origin = _float_array(
        meta.get("image_position_patient_first"),
        (3,),
        label=f"visit {row['visit_id']!r} image_position_patient_first",
    )
    pixel_spacing = _float_array(
        meta.get("pixel_spacing"), (2,), label=f"visit {row['visit_id']!r} pixel_spacing"
    )
    try:
        slice_spacing = float(meta.get("spacing_between_slices"))
    except (TypeError, ValueError) as exc:
        raise ValueError("upstream spacing_between_slices is invalid") from exc
    if np.any(pixel_spacing <= 0) or not math.isfinite(slice_spacing) or slice_spacing <= 0:
        raise ValueError("upstream source spacing must be finite and positive")
    csv_spacing = np.asarray(
        (
            _optional_float(
                row.get("row_spacing_mm"), label=f"row spacing for {row['visit_id']}"
            ),
            _optional_float(
                row.get("column_spacing_mm"), label=f"column spacing for {row['visit_id']}"
            ),
            _optional_float(
                row.get("slice_spacing_mm"), label=f"slice spacing for {row['visit_id']}"
            ),
        ),
        dtype=np.float64,
    )
    source_spacing_xyz = np.asarray(canonical_source_geometry["spacing_xyz"])
    csv_spacing_xyz = csv_spacing[[1, 0, 2]]
    if not np.allclose(csv_spacing_xyz, source_spacing_xyz, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"visit {row['visit_id']!r} source spacing differs from upstream metadata"
        )
    registered_spacing_xyz = np.asarray((*pixel_spacing[::-1], slice_spacing))
    if not np.allclose(
        registered_spacing_xyz,
        np.asarray(canonical_registered_geometry["spacing_xyz"]),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"visit {row['visit_id']!r} registered spacing differs from upstream metadata"
        )

    x_direction = orientation[:3]
    y_direction = orientation[3:]
    if not np.isclose(np.linalg.norm(x_direction), 1.0, atol=1e-5) or not np.isclose(
        np.linalg.norm(y_direction), 1.0, atol=1e-5
    ):
        raise ValueError("upstream orientation direction cosines are not unit length")
    if not np.isclose(float(x_direction @ y_direction), 0.0, atol=1e-5):
        raise ValueError("upstream orientation direction cosines are not orthogonal")
    normal = np.cross(x_direction, y_direction)
    normal /= np.linalg.norm(normal)
    orientation_xyz = np.column_stack((x_direction, y_direction, normal))
    if not np.allclose(
        orientation_xyz,
        np.asarray(canonical_registered_geometry["direction"]).reshape(3, 3),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"visit {row['visit_id']!r} registered orientation differs from upstream metadata"
        )

    try:
        from nibabel.spaces import vox2out_vox
    except ImportError as exc:
        raise RuntimeError("nibabel is required to validate MeWM geometry") from exc

    lps_to_ras = np.diag((-1.0, -1.0, 1.0))
    registered_affine_ras = np.eye(4, dtype=np.float64)
    registered_affine_ras[:3, :3] = (
        lps_to_ras @ orientation_xyz @ np.diag(registered_spacing_xyz)
    )
    registered_affine_ras[:3, 3] = lps_to_ras @ origin
    spacing_xyz = _float_array(
        bundle.preprocess["target_spacing_xyz"], (3,), label="target_spacing_xyz"
    )
    resampled_shape_xyz, resampled_affine_ras = vox2out_vox(
        (tuple(reversed(registered_shape_zyx)), registered_affine_ras),
        voxel_sizes=spacing_xyz,
    )
    expected_resampled_shape_zyx = _int_triplet(
        row["resampled_shape_zyx_json"],
        label=f"visit {row['visit_id']!r} resampled shape",
        positive=True,
    )
    if tuple(reversed(resampled_shape_xyz)) != expected_resampled_shape_zyx:
        raise ValueError(
            f"visit {row['visit_id']!r} resampled shape differs from upstream metadata"
        )

    resampled_affine_lps_xyz = np.eye(4, dtype=np.float64)
    resampled_affine_lps_xyz[:3, :3] = lps_to_ras @ resampled_affine_ras[:3, :3]
    resampled_affine_lps_xyz[:3, 3] = lps_to_ras @ resampled_affine_ras[:3, 3]
    base_affine = np.eye(4, dtype=np.float64)
    base_affine[:3, :3] = resampled_affine_lps_xyz[:3, :3][:, [2, 1, 0]]
    base_affine[:3, 3] = resampled_affine_lps_xyz[:3, 3]

    resampled_direction_lps = resampled_affine_lps_xyz[:3, :3] / np.linalg.norm(
        resampled_affine_lps_xyz[:3, :3], axis=0
    )[None, :]
    recorded_orientation = _float_array(
        row["orientation_lps_json"],
        (3, 3),
        label=f"visit {row['visit_id']!r} orientation_lps_json",
    )
    if not np.allclose(
        recorded_orientation, resampled_direction_lps, rtol=0.0, atol=1e-6
    ):
        raise ValueError(f"visit {row['visit_id']!r} orientation disagrees with upstream metadata")
    crop_start = np.asarray(
        _int_triplet(row["crop_start_zyx_json"], label="crop_start_zyx"),
        dtype=np.float64,
    )
    cropped_affine = base_affine.copy()
    cropped_affine[:3, 3] = base_affine[:3, 3] + base_affine[:3, :3] @ crop_start

    transform_artifacts: list[dict[str, str]] = []
    transform_manifest = directory / "transform.json"
    if transform_manifest.exists():
        if not transform_manifest.is_file() or transform_manifest.is_symlink():
            raise ValueError(f"visit {row['visit_id']!r} transform.json is not a regular file")
        transform = _read_json(
            transform_manifest, label=f"upstream transform for {row['visit_id']!r}"
        )
        expected_transform = {
            "patient_id": row["patient_id"],
            "fixed_visit": "T0",
            "moving_visit": row["visit"],
            "selected_transform": row["registration_status"],
        }
        for key, expected in expected_transform.items():
            if transform.get(key) != expected:
                raise ValueError(
                    f"visit {row['visit_id']!r} transform {key} differs from its metadata"
                )
        if not _parse_bool(
            transform.get("rigidity_applied"),
            label=f"visit {row['visit_id']!r} transform rigidity_applied",
        ):
            raise ValueError(f"visit {row['visit_id']!r} selected transform is not rigid")
        transform_artifacts.append(
            {"path": "transform.json", "sha256": sha256_file(transform_manifest)}
        )

    transform_root = directory / "transform"
    if transform_root.exists():
        if not transform_root.is_dir() or transform_root.is_symlink():
            raise ValueError(f"visit {row['visit_id']!r} transform is not a directory")
        for path in sorted(transform_root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"visit {row['visit_id']!r} transform contains a symlink")
            if path.is_file():
                if path.stat().st_size == 0:
                    raise ValueError(f"visit {row['visit_id']!r} has an empty transform artifact")
                transform_artifacts.append(
                    {
                        "path": f"transform/{path.relative_to(transform_root).as_posix()}",
                        "sha256": sha256_file(path),
                    }
                )
    if row["visit"] != "T0" and not transform_artifacts:
        raise ValueError(f"registered visit {row['visit_id']!r} has no transform evidence")

    return _RegistrationEvidence(
        base_affine_lps=base_affine,
        cropped_affine_lps=cropped_affine,
        meta_sha256=sha256_file(meta_path),
        registration_sha256=sha256_file(registration_path),
        transform_artifacts=tuple(transform_artifacts),
        source_geometry_sha256=stable_hash(canonical_source_geometry),
        registered_grid_geometry_sha256=stable_hash(canonical_registered_geometry),
    )


def _optional_float(value: Any, *, label: str) -> float | None:
    if value is None or str(value).strip().lower() in {"", "na", "nan", "none", "unknown"}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _binary_code(value: Any, *, label: str) -> str | None:
    if value is None or str(value).strip().lower() in {"", "na", "nan", "none", "unknown"}:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"0", "1"}:
        return normalized
    raise ValueError(f"unrecognized {label} value {value!r}")


def _menopausal_status(value: Any, *, patient_id: str) -> str | None:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return None
    normalized = " ".join(raw.lower().split())
    if normalized.startswith("premenopausal"):
        return "premenopausal"
    if normalized.startswith("perimenopausal"):
        return "perimenopausal"
    if normalized.startswith("postmenopausal"):
        return "postmenopausal"
    if normalized == "above categories not applicable and age < 50":
        return "not_applicable_age_lt_50"
    if normalized == "above categories not applicable and age > 50":
        return "not_applicable_age_gt_50"
    raise ValueError(
        f"unrecognized menopausal_status value {value!r} for patient {patient_id}"
    )


def _visit_clinical(
    row: Mapping[str, str], *, action_text: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    patient = row["patient_id"]
    baseline: dict[str, Any] = {}
    for key, value in (
        ("hr_status", _binary_code(row.get("HR"), label=f"HR for {patient}")),
        ("her2_status", _binary_code(row.get("HER2"), label=f"HER2 for {patient}")),
        ("mammaprint", _binary_code(row.get("MP"), label=f"MP for {patient}")),
    ):
        if value is not None:
            baseline[key] = value
    age = _optional_float(row.get("Age_at_Screening"), label=f"age for {patient}")
    if age is not None:
        if not 0 < age < 120:
            raise ValueError(f"age for {patient!r} is outside the valid human range")
        baseline["age"] = age
    menopause = _menopausal_status(row.get("menopausal_status"), patient_id=patient)
    if menopause is not None:
        baseline["menopausal_status"] = menopause
    source_ftv = _optional_float(row.get("ftv_volume_cc"), label=f"FTV for {row['visit_id']}")
    if source_ftv is not None:
        if source_ftv < 0:
            raise ValueError(f"FTV for {row['visit_id']!r} cannot be negative")
    treatment = {"treatment_arm": action_text}
    evaluation = {
        "upstream_ftv_volume_cc": source_ftv,
        "upstream_registration_status": row["registration_status"],
        "upstream_quality_status": row["qc_status"],
    }
    return baseline, treatment, evaluation


def _validate_pair_metadata(
    transition: Mapping[str, str],
    earlier: Mapping[str, str],
    later: Mapping[str, str],
    earlier_evidence: _RegistrationEvidence,
    later_evidence: _RegistrationEvidence,
) -> None:
    transition_id = transition["transition_id"]
    try:
        observed_delta = int(transition["delta_days"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"transition {transition_id!r} has invalid delta_days") from exc
    if observed_delta <= 0:
        raise ValueError(f"transition {transition_id!r} has nonpositive delta_days")
    try:
        date_delta = (
            date.fromisoformat(later["visit_date"])
            - date.fromisoformat(earlier["visit_date"])
        ).days
    except ValueError as exc:
        raise ValueError(f"transition {transition_id!r} has invalid visit dates") from exc
    if date_delta != observed_delta:
        raise ValueError(f"transition {transition_id!r} delta_days differs from visit dates")
    if earlier["visit_date_source"] != later["visit_date_source"]:
        raise ValueError(f"transition {transition_id!r} visit date sources differ")
    if not earlier["visit_date_source"].strip():
        raise ValueError(f"transition {transition_id!r} lacks visit-date provenance")
    if not np.array_equal(
        earlier_evidence.cropped_affine_lps, later_evidence.cropped_affine_lps
    ):
        raise ValueError(f"transition {transition_id!r} does not share one registered affine")
    if not np.array_equal(earlier_evidence.base_affine_lps, later_evidence.base_affine_lps):
        raise ValueError(f"transition {transition_id!r} registered base grids differ")
    if (
        earlier_evidence.registered_grid_geometry_sha256
        != later_evidence.registered_grid_geometry_sha256
    ):
        raise ValueError(f"transition {transition_id!r} registered geometry metadata differ")
    # Native visit grids may differ before registration.  Their hashes remain
    # part of each visit's provenance; only the shared registered grid must
    # match across a supervised pair.


def _stage_number(value: Any, *, label: str) -> int:
    stage = str(value).strip()
    match = re.fullmatch(r"T([0-9]+)", stage)
    if match is None or stage != f"T{int(match.group(1))}":
        raise ValueError(f"{label} must use canonical T<number> syntax")
    return int(match.group(1))


def _metadata_time_pairs(
    bundle: _BundleData,
    time_pairs: Sequence[Sequence[str]] | None,
    *,
    earlier_stage: str | None,
    later_stage: str | None,
) -> tuple[tuple[str, str], ...]:
    if time_pairs is not None and (earlier_stage is not None or later_stage is not None):
        raise ValueError("time_pairs cannot be combined with earlier_stage/later_stage")
    if (earlier_stage is None) != (later_stage is None):
        raise ValueError("earlier_stage and later_stage must be provided together")
    if earlier_stage is not None and later_stage is not None:
        time_pairs = ((earlier_stage, later_stage),)

    if time_pairs is None:
        stages = sorted(
            {row["visit"] for row in bundle.visits},
            key=lambda value: _stage_number(value, label="bundle visit stage"),
        )
        normalized = tuple(
            (left, right)
            for left_index, left in enumerate(stages)
            for right in stages[left_index + 1 :]
        )
    else:
        pairs: list[tuple[str, str]] = []
        for index, value in enumerate(time_pairs):
            if isinstance(value, (str, bytes)) or len(value) != 2:
                raise ValueError(f"time_pairs[{index}] must contain two stage labels")
            left, right = (str(item).strip() for item in value)
            pairs.append((left, right))
        normalized = tuple(pairs)

    if not normalized:
        raise ValueError("time_pairs must select at least one time interval")
    if len(set(normalized)) != len(normalized):
        raise ValueError("time_pairs must not contain duplicate intervals")
    for left, right in normalized:
        left_number = _stage_number(left, label=f"earlier stage {left!r}")
        right_number = _stage_number(right, label=f"later stage {right!r}")
        if left_number >= right_number:
            raise ValueError(f"time pair {left!r}->{right!r} must preserve Tn<Tm semantics")
    return normalized


def _affine_tuple(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def validate_mewm_bundle_metadata(
    bundle_dir: str | Path,
    upstream_metadata_dir: str | Path,
    *,
    time_pairs: Sequence[Sequence[str]] | None = None,
    earlier_stage: str | None = None,
    later_stage: str | None = None,
    selected_pair_ids: Sequence[str] | None = None,
) -> MewmMetadataValidationResult:
    """Validate signed MeWM metadata and derive direct/all-connected pair grids.

    Non-adjacent intervals are admitted only when every adjacent transition is
    present for that patient. Their real interval is the checked sum of those
    signed edges and is independently checked against the endpoint dates.
    This function never reads or writes an ROI or latent payload.
    """

    bundle = _validate_bundle(bundle_dir)
    requested_pairs = _metadata_time_pairs(
        bundle,
        time_pairs,
        earlier_stage=earlier_stage,
        later_stage=later_stage,
    )
    metadata_root = Path(upstream_metadata_dir).expanduser().resolve()
    if not metadata_root.is_dir() or metadata_root.is_symlink():
        raise ValueError(f"upstream metadata directory is missing: {metadata_root}")

    visits_by_id = {row["visit_id"]: row for row in bundle.visits}
    visits_by_patient_stage = {
        (row["patient_id"], row["visit"]): row for row in bundle.visits
    }
    edges_by_patient_stage = {
        (row["patient_id"], row["source_visit"], row["target_visit"]): row
        for row in bundle.transitions
    }
    candidates: list[tuple[str, str, str, tuple[dict[str, str], ...]]] = []
    candidate_counts: Counter[tuple[str, str]] = Counter()
    for left, right in requested_pairs:
        left_number = _stage_number(left, label="earlier stage")
        right_number = _stage_number(right, label="later stage")
        for patient_id in sorted(bundle.split_by_patient):
            edge_rows: list[dict[str, str]] = []
            for number in range(left_number, right_number):
                edge = edges_by_patient_stage.get(
                    (patient_id, f"T{number}", f"T{number + 1}")
                )
                if edge is None:
                    break
                edge_rows.append(edge)
            else:
                pair_id = f"{patient_id}:{left}->{right}"
                candidates.append((pair_id, left, right, tuple(edge_rows)))
                candidate_counts[(left, right)] += 1
    empty_types = [
        f"{left}->{right}"
        for left, right in requested_pairs
        if candidate_counts[(left, right)] == 0
    ]
    if empty_types:
        raise ValueError(
            "bundle contains no adjacent-edge-connected pairs for "
            + ", ".join(empty_types)
        )

    if selected_pair_ids is not None:
        requested_ids = tuple(str(value).strip() for value in selected_pair_ids)
        if any(not value for value in requested_ids) or len(set(requested_ids)) != len(
            requested_ids
        ):
            raise ValueError("selected_pair_ids must contain unique non-empty IDs")
        candidate_ids = {value[0] for value in candidates}
        missing_ids = sorted(set(requested_ids) - candidate_ids)
        if missing_ids:
            raise ValueError(
                "selected pairs are not adjacent-edge connected for the requested intervals: "
                + ", ".join(missing_ids[:10])
            )
        requested_id_set = set(requested_ids)
        candidates = [value for value in candidates if value[0] in requested_id_set]
    if not candidates:
        raise ValueError("no connected MeWM metadata pairs were selected")

    selected_rows: dict[str, dict[str, str]] = {}
    for _, _, _, edge_rows in candidates:
        for edge in edge_rows:
            for visit_id in (edge["source_visit_id"], edge["target_visit_id"]):
                selected_rows[visit_id] = visits_by_id[visit_id]

    evidence: dict[str, _RegistrationEvidence] = {}
    for visit_id, row in sorted(selected_rows.items()):
        evidence[visit_id] = _geometry_from_metadata(bundle, metadata_root, row)

    spacing_xyz = _float_array(
        bundle.preprocess["target_spacing_xyz"], (3,), label="target_spacing_xyz"
    )
    spacing_dhw = (
        float(spacing_xyz[2]),
        float(spacing_xyz[1]),
        float(spacing_xyz[0]),
    )
    output_shape = _int_triplet(
        bundle.preprocess["output_shape_zyx"],
        label="preprocess output_shape_zyx",
        positive=True,
    )
    visit_geometries: dict[str, MewmVisitGeometry] = {}
    for visit_id, row in sorted(selected_rows.items()):
        item = evidence[visit_id]
        visit_geometries[visit_id] = MewmVisitGeometry(
            visit_id=visit_id,
            patient_id=row["patient_id"],
            visit_stage=row["visit"],
            split=row["fold"],
            study_uid=row["study_instance_uid"],
            affine_lps=_affine_tuple(item.cropped_affine_lps),
            base_affine_lps=_affine_tuple(item.base_affine_lps),
            spacing_dhw=spacing_dhw,
            crop_start_zyx=_int_triplet(
                row["crop_start_zyx_json"], label=f"crop start for {visit_id!r}"
            ),
            output_shape_zyx=output_shape,
            meta_sha256=item.meta_sha256,
            registration_sha256=item.registration_sha256,
            transform_artifacts=tuple(
                (str(artifact["path"]), str(artifact["sha256"]))
                for artifact in item.transform_artifacts
            ),
            source_geometry_sha256=item.source_geometry_sha256,
            registered_grid_geometry_sha256=item.registered_grid_geometry_sha256,
        )

    pair_geometries: list[MewmConnectedPairGeometry] = []
    for pair_id, left, right, edge_rows in candidates:
        action_texts: set[str] = set()
        delta_days = 0
        for edge in edge_rows:
            source_id = edge["source_visit_id"]
            target_id = edge["target_visit_id"]
            _validate_pair_metadata(
                edge,
                visits_by_id[source_id],
                visits_by_id[target_id],
                evidence[source_id],
                evidence[target_id],
            )
            action_texts.add(
                _required_text(
                    edge,
                    "action_text",
                    label=f"transition {edge['transition_id']!r}",
                )
            )
            delta_days += int(edge["delta_days"])
        if len(action_texts) != 1:
            raise ValueError(
                f"connected pair {pair_id!r} has inconsistent treatment action text"
            )

        patient_id = edge_rows[0]["patient_id"]
        earlier_row = visits_by_patient_stage[(patient_id, left)]
        later_row = visits_by_patient_stage[(patient_id, right)]
        earlier_id = earlier_row["visit_id"]
        later_id = later_row["visit_id"]
        _validate_pair_metadata(
            {"transition_id": pair_id, "delta_days": str(delta_days)},
            earlier_row,
            later_row,
            evidence[earlier_id],
            evidence[later_id],
        )
        pair_geometries.append(
            MewmConnectedPairGeometry(
                pair_id=pair_id,
                patient_id=patient_id,
                split=edge_rows[0]["fold"],
                earlier_stage=left,
                later_stage=right,
                earlier_visit_id=earlier_id,
                later_visit_id=later_id,
                delta_days=delta_days,
                action_text=next(iter(action_texts)),
                edge_transition_ids=tuple(edge["transition_id"] for edge in edge_rows),
                earlier=visit_geometries[earlier_id],
                later=visit_geometries[later_id],
            )
        )

    pair_by_endpoints = {
        (pair.earlier_visit_id, pair.later_visit_id): pair for pair in pair_geometries
    }
    return MewmMetadataValidationResult(
        bundle_contract_sha256=str(bundle.document["bundle_contract_sha256"]),
        bundle_json_sha256=sha256_file(bundle.root / "bundle.json"),
        artifact_sha256=MappingProxyType(dict(sorted(bundle.artifact_hashes.items()))),
        time_pairs=requested_pairs,
        output_shape_zyx=output_shape,
        spacing_dhw=spacing_dhw,
        visits=tuple(visit_geometries.values()),
        pairs=tuple(pair_geometries),
        visit_geometry_by_id=MappingProxyType(visit_geometries),
        pair_geometry_by_endpoints=MappingProxyType(pair_by_endpoints),
    )


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "unknown"


def _archive_relative_path(row: Mapping[str, str]) -> Path:
    study_hash = hashlib.sha256(row["study_instance_uid"].encode("utf-8")).hexdigest()[:12]
    return (
        Path("volumes")
        / row["fold"]
        / _safe_component(row["patient_id"])
        / f"{_safe_component(row['visit'])}-{study_hash}.npz"
    )


def _write_archive(
    path: Path,
    *,
    image: np.ndarray,
    row: Mapping[str, str],
    roles: tuple[str, ...],
    evidence: _RegistrationEvidence,
    cache: _CachePayload,
    bundle: _BundleData,
) -> None:
    role_indices = [0 if role == "dce0" else 1 for role in roles]
    selected = np.asarray(image[role_indices], dtype=np.float32)
    spacing_xyz = _float_array(
        bundle.preprocess["target_spacing_xyz"], (3,), label="target spacing"
    )
    spacing_dhw = np.asarray(
        (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]), dtype=np.float32
    )
    original_paths = {role: row[f"{role}_path"] for role in roles}
    provenance = {
        "version": IMPORT_PROVENANCE_VERSION,
        "source_only": False,
        "target_content_used": True,
        "crop_basis": CROP_POLICY,
        "normalization_scope": NORMALIZATION_SCOPE,
        "intensity_stats_hash": bundle.artifact_hashes["normalization"],
        "input_shape_cdhw": list(image.shape),
        "oriented_shape_cdhw": list(image.shape),
        "resampled_shape_cdhw": [2, *cache.crop_plan["input_shape_zyx"]],
        "output_shape_cdhw": list(selected.shape),
        "source_spacing_dhw": spacing_dhw.tolist(),
        "output_spacing_dhw": spacing_dhw.tolist(),
        "input_affine_lps": evidence.base_affine_lps.tolist(),
        "output_affine_lps": evidence.cropped_affine_lps.tolist(),
        "crop_plan": dict(cache.crop_plan),
        "extra": {
            "phase_roles": list(roles),
            "coordinate_frame": COORDINATE_FRAME,
            "geometry_source": "upstream_registered_meta_dicom_absolute_lps",
            "nifti_header_geometry_used": False,
            "registration_assisted": True,
            "deployment_scope": "paired_registered_research_only",
            "crop_anchor_stage": "T0",
            "upstream_bundle_contract_sha256": bundle.document["bundle_contract_sha256"],
            "upstream_base_cache_contract_sha256": bundle.document["base_cache_contract"][
                "bundle_contract_sha256"
            ],
            "upstream_cache_key": cache.cache_key,
            "upstream_cache_sha256": cache.cache_sha256,
            "upstream_meta_sha256": evidence.meta_sha256,
            "upstream_registration_sha256": evidence.registration_sha256,
            "upstream_transform_artifacts": list(evidence.transform_artifacts),
            "upstream_source_geometry_sha256": evidence.source_geometry_sha256,
            "upstream_registered_grid_geometry_sha256": (
                evidence.registered_grid_geometry_sha256
            ),
            "upstream_source_path_sha256": {
                role: hashlib.sha256(value.encode("utf-8")).hexdigest()
                for role, value in original_paths.items()
            },
        },
    }
    source_identifiers = np.asarray(
        [f"upstream:{row['study_instance_uid']}:{role}" for role in roles]
    )
    temporal_positions = np.asarray(
        [0 if role == "dce0" else -1 for role in roles], dtype=np.int32
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image=selected,
        affine_lps=np.asarray(evidence.cropped_affine_lps, dtype=np.float64),
        spacing_dhw=spacing_dhw,
        phase_roles=np.asarray(roles),
        patient_id=np.asarray(row["patient_id"]),
        visit_id=np.asarray(row["visit_id"]),
        study_uid=np.asarray(row["study_instance_uid"]),
        visit_stage=np.asarray(row["visit"]),
        split=np.asarray(row["fold"]),
        source_series_uids=source_identifiers,
        source_temporal_positions=temporal_positions,
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True, separators=(",", ":"))),
    )


def import_mewm_roi_cache(
    bundle_dir: str | Path,
    roi_cache_dir: str | Path,
    upstream_metadata_dir: str | Path,
    output_dir: str | Path,
    *,
    earlier_stage: str = "T0",
    later_stage: str = "T1",
    phase_roles: Sequence[str] = ("dce0",),
) -> MewmImportResult:
    """Validate and import one MeWM time pair into this project's NPZ contract."""

    roles = tuple(str(role).strip().lower() for role in phase_roles)
    if roles not in _ALLOWED_PHASE_ROLES:
        raise ValueError("phase_roles must be ('dce0',) or ('dce0', 'ser')")
    stage_pattern = re.compile(r"T([0-9]+)")
    earlier_match, later_match = stage_pattern.fullmatch(earlier_stage), stage_pattern.fullmatch(
        later_stage
    )
    if not earlier_match or not later_match or int(earlier_match.group(1)) >= int(
        later_match.group(1)
    ):
        raise ValueError("time pair must preserve Tn<Tm semantics")

    bundle = _validate_bundle(bundle_dir)
    cache_root = Path(roi_cache_dir).expanduser().resolve()
    metadata_root = Path(upstream_metadata_dir).expanduser().resolve()
    if not cache_root.is_dir() or cache_root.is_symlink():
        raise ValueError(f"ROI cache directory is missing: {cache_root}")
    if not metadata_root.is_dir() or metadata_root.is_symlink():
        raise ValueError(f"upstream metadata directory is missing: {metadata_root}")

    visits_by_id = {row["visit_id"]: row for row in bundle.visits}
    transition_type = f"{earlier_stage}->{later_stage}"
    candidates = [
        row for row in bundle.transitions if row["transition_type"] == transition_type
    ]
    if not candidates:
        raise ValueError(f"bundle contains no transitions for {transition_type}")

    available: list[dict[str, str]] = []
    missing_cache_counts: Counter[str] = Counter()
    missing_cache_visits: set[str] = set()
    for transition in candidates:
        missing: list[str] = []
        for branch, visit_id in (
            ("earlier", transition["source_visit_id"]),
            ("later", transition["target_visit_id"]),
        ):
            path, _ = _cache_path(bundle, cache_root, visit_id)
            if not path.is_file():
                missing.append(branch)
                missing_cache_visits.add(visit_id)
        if missing:
            missing_cache_counts["+".join(missing)] += 1
        else:
            available.append(transition)
    if not available:
        raise ValueError(f"ROI cache contains no complete {transition_type} pairs")

    selected_rows: dict[str, dict[str, str]] = {}
    action_by_patient: dict[str, str] = {}
    caches: dict[str, _CachePayload] = {}
    evidence: dict[str, _RegistrationEvidence] = {}
    for transition in available:
        patient_id = transition["patient_id"]
        action = _required_text(
            transition, "action_text", label=f"transition {transition['transition_id']!r}"
        )
        previous_action = action_by_patient.setdefault(patient_id, action)
        if previous_action != action:
            raise ValueError(f"patient {patient_id!r} has conflicting treatment action text")
        for visit_id in (transition["source_visit_id"], transition["target_visit_id"]):
            row = visits_by_id[visit_id]
            previous = selected_rows.setdefault(visit_id, row)
            if previous != row:
                raise ValueError(f"visit {visit_id!r} has conflicting metadata")
            if visit_id not in caches:
                caches[visit_id] = _validate_cache_payload(bundle, cache_root, row)
                evidence[visit_id] = _geometry_from_metadata(bundle, metadata_root, row)
        _validate_pair_metadata(
            transition,
            visits_by_id[transition["source_visit_id"]],
            visits_by_id[transition["target_visit_id"]],
            evidence[transition["source_visit_id"]],
            evidence[transition["target_visit_id"]],
        )

    split_pair_counts = Counter(transition["fold"] for transition in available)
    missing_splits = {"train", "val"} - set(split_pair_counts)
    if missing_splits:
        raise ValueError(
            "imported paired cohort needs train and val pairs; missing "
            + ", ".join(sorted(missing_splits))
        )

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"MeWM import output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.import-", dir=destination.parent)
    )
    visit_records: list[VisitRecord] = []
    try:
        for visit_id, row in sorted(selected_rows.items()):
            relative_path = _archive_relative_path(row)
            _write_archive(
                temporary / relative_path,
                image=caches[visit_id].image,
                row=row,
                roles=roles,
                evidence=evidence[visit_id],
                cache=caches[visit_id],
                bundle=bundle,
            )
            baseline, treatment, evaluation = _visit_clinical(
                row, action_text=action_by_patient[row["patient_id"]]
            )
            visit_records.append(
                VisitRecord(
                    visit_id=visit_id,
                    patient_id=row["patient_id"],
                    collection="ISPY2_MeWM_REGISTERED_STRICT_A",
                    study_uid=row["study_instance_uid"],
                    visit_stage=row["visit"],
                    study_date=row["visit_date"],
                    date_source=row["visit_date_source"],
                    relative_date_verified=True,
                    phase_paths={},
                    baseline_clinical=baseline,
                    treatment=treatment,
                    evaluation_metadata=evaluation,
                    qc=(
                        QCEvent(
                            code="external_registration_assisted",
                            severity="warning",
                            message="Visit was externally registered into its patient's T0 frame",
                            details={"deployment_scope": "paired_registered_research_only"},
                        ),
                        QCEvent(
                            code="t0_mask_anchored_crop",
                            severity="warning",
                            message="One T0-mask-derived crop was reused across registered visits",
                        ),
                    ),
                    series_count=len(roles),
                    split=row["fold"],
                    prepared_path=str(destination / relative_path),
                    schema_version="mewm-import-1.0",
                )
            )

        selected_splits = {
            visit.patient_id: str(visit.split) for visit in visit_records
        }
        pairs = build_pair_manifest(
            visit_records,
            selected_splits,
            earlier_stage=earlier_stage,
            later_stage=later_stage,
            require_three_phase=False,
        )
        transitions_by_endpoints = {
            (row["source_visit_id"], row["target_visit_id"]): row for row in available
        }
        if len(pairs) != len(available):
            raise ValueError("generated pair count differs from complete-cache transitions")
        for pair in pairs:
            transition = transitions_by_endpoints.get(
                (pair.earlier_visit_id, pair.later_visit_id)
            )
            if transition is None:
                raise ValueError(f"generated pair {pair.pair_id!r} has no source transition")
            if pair.delta_days != int(transition["delta_days"]):
                raise ValueError(f"generated pair {pair.pair_id!r} has inconsistent delta_days")

        visit_manifest = temporary / "visits.prepared.jsonl"
        pair_manifest = temporary / f"pairs.{earlier_stage}-{later_stage}.jsonl"
        intensity_stats = temporary / "intensity_stats.json"
        audit_path = temporary / "mewm_import_audit.json"
        write_jsonl(visit_records, visit_manifest)
        write_jsonl(pairs, pair_manifest)
        intensity_stats.write_text(
            json.dumps(
                {
                    **bundle.normalization,
                    "artifact_sha256": bundle.artifact_hashes["normalization"],
                    "normalization_scope": NORMALIZATION_SCOPE,
                    "phase_roles": list(roles),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        audit: dict[str, Any] = {
            "schema": "ispy2_symmflow_mewm_import_audit_v1",
            "bundle_schema": BUNDLE_SCHEMA,
            "bundle_contract_sha256": bundle.document["bundle_contract_sha256"],
            "base_cache_contract": dict(bundle.document["base_cache_contract"]),
            "artifact_sha256": dict(sorted(bundle.artifact_hashes.items())),
            "time_pair": [earlier_stage, later_stage],
            "phase_roles": list(roles),
            "candidate_pair_count": len(candidates),
            "imported_pair_count": len(pairs),
            "imported_visit_count": len(visit_records),
            "imported_patient_count": len(selected_splits),
            "imported_pair_counts_by_split": dict(sorted(split_pair_counts.items())),
            "missing_cache_pair_count": len(candidates) - len(available),
            "missing_cache_pattern_counts": dict(sorted(missing_cache_counts.items())),
            "missing_cache_visit_count": len(missing_cache_visits),
            "coordinate_frame": COORDINATE_FRAME,
            "geometry_source": "upstream_registered_meta_dicom_absolute_lps",
            "registration_assisted": True,
            "source_only": False,
            "target_content_used": True,
            "crop_anchor_stage": "T0",
            "deployment_scope": "paired_registered_research_only",
            "normalization_scope": NORMALIZATION_SCOPE,
        }
        audit_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return MewmImportResult(
        visits=tuple(visit_records),
        pairs=tuple(pairs),
        visit_manifest_path=str(destination / "visits.prepared.jsonl"),
        pair_manifest_path=str(destination / f"pairs.{earlier_stage}-{later_stage}.jsonl"),
        intensity_stats_path=str(destination / "intensity_stats.json"),
        audit_path=str(destination / "mewm_import_audit.json"),
        audit=audit,
    )


__all__ = [
    "MewmConnectedPairGeometry",
    "MewmImportResult",
    "MewmMetadataValidationResult",
    "MewmVisitGeometry",
    "import_mewm_roi_cache",
    "validate_mewm_bundle_metadata",
]
