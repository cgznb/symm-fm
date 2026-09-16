from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import DATA_SCHEMA_VERSION, ClinicalTextPolicy, TransitionRecord
from .manifest import ACCEPTED_BACKENDS, VisitRecord, build_visit_index


STAGE_BY_TRANSITION = {"T0->T1": 1, "T1->T2": 2, "T2->T3": 3}
LONGITUDINAL_BUNDLE_SCHEMA = "motfm_ispy2_longitudinal_bundle_v1"
REGISTERED_STRICT_A_BUNDLE_SCHEMA = "motfm_ispy2_registered_strict_a_bundle_v1"
SUPPORTED_BUNDLE_SCHEMAS = frozenset(
    {LONGITUDINAL_BUNDLE_SCHEMA, REGISTERED_STRICT_A_BUNDLE_SCHEMA}
)
REGISTERED_STRICT_A_BASE_CACHE_KEYS = frozenset(
    {
        "schema",
        "bundle_contract_sha256",
        "preprocess_sha256",
        "normalization_sha256",
    }
)
REGISTERED_STRICT_A_BASE_DATA_KEYS = frozenset(
    {
        "schema",
        "bundle_contract_sha256",
        "split_sha256",
        "visits_semantic_sha256",
        "transitions_sha256",
        "exclusion_row_count",
        "exclusions_sha256",
        "crop_plans_sha256",
    }
)
REGISTERED_STRICT_A_ARTIFACT_FILENAMES = {
    "visits": "visits.csv",
    "transitions": "transitions.csv",
    "exclusions": "exclusions.csv",
    "split": "split.json",
    "preprocess": "preprocess.json",
    "normalization": "normalization.json",
    "crop_plans": "crop_plans.json",
}
REGISTERED_STRICT_A_VISIT_COLUMNS = (
    "patient_id",
    "visit_id",
    "visit",
    "visit_index",
    "study_instance_uid",
    "visit_date",
    "visit_date_source",
    "qc_status",
    "dce0_path",
    "ser_path",
    "mask_path",
    "meta_path",
    "ftv_voxel_count",
    "row_spacing_mm",
    "column_spacing_mm",
    "slice_spacing_mm",
    "ftv_voxel_volume_mm3",
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
    "mask_voxel_count_resampled",
    "resampled_bbox_min_zyx_json",
    "resampled_bbox_max_zyx_json",
    "resampled_bbox_shape_zyx_json",
    "crop_start_zyx_json",
    "mask_fully_retained",
    "mask_retention_reason",
    "original_meta_path",
    "registration_status",
    "registration_json_path",
    "is_t0",
    "in_plane_fov_ratio",
    "selected_non_tumor_ncc",
    "selected_protected_ncc",
    "rotation_degrees",
    "translation_mm",
    "rigid_maximum_iterations_reached",
    "algorithm_anomaly",
    "algorithm_anomaly_codes",
    "quality_pass",
    "mask_voxel_count_after_crop",
    "quality_evidence_json",
)
REGISTERED_STRICT_A_TRANSITION_COLUMNS = (
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
    "orientation_angle_degrees",
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
)
REGISTERED_STRICT_A_EXCLUSION_COLUMNS = (
    "transition_id",
    "patient_id",
    "source_visit_id",
    "target_visit_id",
    "transition_type",
    "reason",
    "detail",
)
REGISTERED_STRICT_A_BASE_VISIT_GEOMETRY_FIELDS = frozenset(
    {
        "orientation_lps_json",
        "resampled_shape_zyx_json",
        "resampled_bbox_min_zyx_json",
        "resampled_bbox_max_zyx_json",
        "resampled_bbox_shape_zyx_json",
        "mask_retention_reason",
    }
)
REGISTERED_STRICT_A_INCREMENT_FILENAMES = {
    "new_patient_ids_sha256": "increment/new_patient_ids.txt",
    "new_visit_ids_sha256": "increment/new_visit_ids.txt",
    "new_transition_ids_sha256": "increment/new_transition_ids.txt",
    "new_exclusions_sha256": "increment/new_exclusions.csv",
    "summary_sha256": "increment/summary.json",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("registered Strict-A base record contains non-finite data")
        return float(value)
    if isinstance(value, (bool, int, str)):
        return value
    raise ValueError("registered Strict-A base record contains unsupported data")


def _records_sha256(
    frame: pd.DataFrame, *, identity: str, columns: tuple[str, ...]
) -> str:
    if identity not in frame.columns or any(
        column not in frame.columns for column in columns
    ):
        raise ValueError("registered Strict-A base record schema is invalid")
    records = [
        {column: _canonical_scalar(row[column]) for column in columns}
        for row in frame.sort_values(identity).to_dict(orient="records")
    ]
    return _canonical_sha(records)


def _read_json_object(path: Path, *, artifact_name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(
            f"registered Strict-A {artifact_name} artifact is unreadable"
        ) from None
    if not isinstance(value, dict):
        raise ValueError(
            f"registered Strict-A {artifact_name} artifact is invalid"
        )
    return value


def _load_registered_strict_a_base_artifacts(
    bundle_path: Path, artifacts: dict[str, Any]
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if set(artifacts) != set(REGISTERED_STRICT_A_ARTIFACT_FILENAMES):
        raise ValueError("registered Strict-A base data artifacts are invalid")
    artifact_paths: dict[str, Path] = {}
    for name, filename in REGISTERED_STRICT_A_ARTIFACT_FILENAMES.items():
        descriptor = artifacts.get(name)
        expected_keys = {"path", "sha256"} | (
            {"row_count"} if filename.endswith(".csv") else set()
        )
        path = bundle_path.parent / filename
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != expected_keys
            or descriptor.get("path") != filename
            or not _is_sha256(descriptor.get("sha256"))
            or not path.is_file()
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ValueError("registered Strict-A base data artifact is invalid")
        artifact_paths[name] = path

    frames: list[pd.DataFrame] = []
    for name, columns in (
        ("visits", REGISTERED_STRICT_A_VISIT_COLUMNS),
        ("transitions", REGISTERED_STRICT_A_TRANSITION_COLUMNS),
        ("exclusions", REGISTERED_STRICT_A_EXCLUSION_COLUMNS),
    ):
        try:
            frame = pd.read_csv(artifact_paths[name])
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            raise ValueError(
                f"registered Strict-A {name} artifact is unreadable"
            ) from None
        if (
            tuple(frame.columns) != columns
            or len(frame) != artifacts[name]["row_count"]
        ):
            raise ValueError(f"registered Strict-A {name} artifact is invalid")
        frames.append(frame)

    split = _read_json_object(artifact_paths["split"], artifact_name="split")
    normalization = _read_json_object(
        artifact_paths["normalization"], artifact_name="normalization"
    )
    crop_plans = _read_json_object(
        artifact_paths["crop_plans"], artifact_name="crop plans"
    )
    return frames[0], frames[1], frames[2], split, normalization, crop_plans


def _normalization_visit_ids(normalization: dict[str, Any]) -> tuple[str, ...]:
    channels = normalization.get("channels")
    if (
        normalization.get("schema") != "motfm_ispy2_train_foreground_stats_v1"
        or not isinstance(channels, dict)
        or set(channels) != {"dce0", "ser"}
        or any(
            not isinstance(channels[name], dict)
            or set(channels[name]) != {"count", "mean", "std"}
            or type(channels[name]["count"]) is not int
            for name in ("dce0", "ser")
        )
    ):
        raise ValueError("registered Strict-A normalization artifact is invalid")
    try:
        statistics = [
            (
                channels[name]["count"],
                float(channels[name]["mean"]),
                float(channels[name]["std"]),
            )
            for name in ("dce0", "ser")
        ]
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("registered Strict-A normalization artifact is invalid") from None
    if any(
        count <= 0
        or not math.isfinite(mean)
        or not math.isfinite(std)
        or std <= 0
        for count, mean, std in statistics
    ):
        raise ValueError("registered Strict-A normalization artifact is invalid")
    visit_ids = tuple(str(value) for value in normalization.get("visit_ids", ()))
    if not visit_ids or len(visit_ids) != len(set(visit_ids)):
        raise ValueError("registered Strict-A normalization artifact is invalid")
    return visit_ids


def _expected_registered_strict_a_base_data_contract(
    *,
    base_data: dict[str, Any],
    visits: pd.DataFrame,
    transitions: pd.DataFrame,
    exclusions: pd.DataFrame,
    split: dict[str, Any],
    normalization: dict[str, Any],
    crop_plans: dict[str, Any],
) -> dict[str, Any]:
    if set(split) != {"train", "val"} or any(
        not isinstance(split[fold], list)
        or not split[fold]
        or any(
            not isinstance(value, str) or not value.strip()
            for value in split[fold]
        )
        or len(split[fold]) != len(set(split[fold]))
        for fold in ("train", "val")
    ):
        raise ValueError("registered Strict-A split artifact is invalid")
    base_train_patients = {
        visit_id.rsplit(":", 1)[0]
        for visit_id in _normalization_visit_ids(normalization)
    }
    train_patients = {str(value) for value in split["train"]}
    val_patients = {str(value) for value in split["val"]}
    if (
        not base_train_patients
        or not base_train_patients < train_patients
        or base_train_patients.intersection(val_patients)
        or train_patients.intersection(val_patients)
    ):
        raise ValueError("registered Strict-A base data split is invalid")
    base_patients = base_train_patients | val_patients
    exclusion_row_count = base_data["exclusion_row_count"]
    if exclusion_row_count > len(exclusions):
        raise ValueError("registered Strict-A base data exclusions are invalid")
    try:
        base_plans = {
            patient: crop_plans[patient] for patient in sorted(base_patients)
        }
    except KeyError:
        raise ValueError("registered Strict-A base crop plan is missing") from None
    base_visits = visits.loc[
        visits["patient_id"].astype(str).isin(base_patients)
    ]
    base_transitions = transitions.loc[
        transitions["patient_id"].astype(str).isin(base_patients)
    ]
    base_exclusions = exclusions.iloc[:exclusion_row_count]
    visit_columns = tuple(
        column
        for column in REGISTERED_STRICT_A_VISIT_COLUMNS
        if column not in REGISTERED_STRICT_A_BASE_VISIT_GEOMETRY_FIELDS
    )
    return {
        "schema": "motfm_ispy2_frozen_base_data_v1",
        "bundle_contract_sha256": str(base_data["bundle_contract_sha256"]),
        "split_sha256": _canonical_sha(
            {
                "train": sorted(base_train_patients),
                "val": sorted(val_patients),
            }
        ),
        "visits_semantic_sha256": _records_sha256(
            base_visits,
            identity="visit_id",
            columns=visit_columns,
        ),
        "transitions_sha256": _records_sha256(
            base_transitions,
            identity="transition_id",
            columns=REGISTERED_STRICT_A_TRANSITION_COLUMNS,
        ),
        "exclusion_row_count": len(base_exclusions),
        "exclusions_sha256": _records_sha256(
            base_exclusions,
            identity="transition_id",
            columns=REGISTERED_STRICT_A_EXCLUSION_COLUMNS,
        ),
        "crop_plans_sha256": _canonical_sha(base_plans),
    }


def _read_registered_strict_a_increment_ids(path: Path, *, name: str) -> set[str]:
    try:
        values = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ValueError(
            f"registered Strict-A increment {name} is unreadable"
        ) from None
    if (
        not values
        or values != sorted(values)
        or len(values) != len(set(values))
        or any(not value.strip() for value in values)
    ):
        raise ValueError(f"registered Strict-A increment {name} is invalid")
    return set(values)


def _validate_registered_strict_a_active_relations(
    *,
    payload: dict[str, Any],
    bundle_path: Path,
    base_data: dict[str, Any],
    visits: pd.DataFrame,
    transitions: pd.DataFrame,
    exclusions: pd.DataFrame,
    split: dict[str, Any],
    normalization: dict[str, Any],
) -> None:
    train_patients = {str(value) for value in split["train"]}
    val_patients = {str(value) for value in split["val"]}
    split_patients = train_patients | val_patients
    visit_ids = visits["visit_id"].astype(str)
    transition_ids = transitions["transition_id"].astype(str)
    exclusion_ids = exclusions["transition_id"].astype(str)
    if (
        visit_ids.duplicated().any()
        or transition_ids.duplicated().any()
        or exclusion_ids.duplicated().any()
        or set(transition_ids).intersection(exclusion_ids)
    ):
        raise ValueError("registered Strict-A record identities are invalid")
    if (
        set(visits["patient_id"].astype(str)) != split_patients
        or set(transitions["patient_id"].astype(str)) != split_patients
    ):
        raise ValueError("registered Strict-A split does not cover bundle patients")

    fold_patients = {"train": train_patients, "val": val_patients}
    visit_by_id = {
        str(row["visit_id"]): row for row in visits.to_dict(orient="records")
    }
    for row in visit_by_id.values():
        patient = str(row["patient_id"])
        fold = str(row["fold"])
        if fold not in fold_patients or patient not in fold_patients[fold]:
            raise ValueError("registered Strict-A visit fold is inconsistent")
    for row in transitions.to_dict(orient="records"):
        patient = str(row["patient_id"])
        fold = str(row["fold"])
        source = visit_by_id.get(str(row["source_visit_id"]))
        target = visit_by_id.get(str(row["target_visit_id"]))
        if (
            fold not in fold_patients
            or patient not in fold_patients[fold]
            or source is None
            or target is None
            or any(
                str(endpoint["patient_id"]) != patient
                or str(endpoint["fold"]) != fold
                for endpoint in (source, target)
            )
        ):
            raise ValueError("registered Strict-A transition fold is inconsistent")

    counts = payload.get("counts")
    transition_type_counts = {
        str(name): int(value)
        for name, value in transitions["transition_type"]
        .astype(str)
        .value_counts()
        .sort_index()
        .items()
    }
    empty_endpoints = sum(
        int(visit_by_id[str(row["source_visit_id"])]["mask_voxel_count_resampled"])
        == 0
        or int(visit_by_id[str(row["target_visit_id"])]["mask_voxel_count_resampled"])
        == 0
        for row in transitions.to_dict(orient="records")
    )
    expected_counts = {
        "universe_transition_count": len(transitions) + len(exclusions),
        "evaluable_transition_count": len(transitions)
        + sum(
            "patient_has_no_T0" not in str(row["reason"]).split("+")
            for row in exclusions.to_dict(orient="records")
        ),
        "transition_count": len(transitions),
        "exclusion_count": len(exclusions),
        "patient_count": len(split_patients),
        "visit_count": len(visits),
        "fold_transition_counts": {
            fold: int((transitions["fold"].astype(str) == fold).sum())
            for fold in ("train", "val")
        },
        "fold_patient_counts": {
            "train": len(train_patients),
            "val": len(val_patients),
        },
        "transition_type_counts": transition_type_counts,
        "empty_source_or_target_mask_transitions": empty_endpoints,
    }
    if not isinstance(counts, dict) or counts != expected_counts:
        raise ValueError("registered Strict-A bundle counts are inconsistent")

    source_contracts = payload.get("source_contracts")
    increment = payload.get("increment_contract")
    expansion = payload.get("expansion")
    if (
        not isinstance(source_contracts, dict)
        or not isinstance(increment, dict)
        or set(increment)
        != {"schema", *REGISTERED_STRICT_A_INCREMENT_FILENAMES}
        or increment.get("schema") != "motfm_ispy2_registered_increment_v1"
        or not isinstance(expansion, dict)
        or set(expansion)
        != {
            "candidate_patient_count",
            "candidate_transition_count",
            "excluded_transition_count",
            "selected_patient_count",
            "selected_transition_count",
        }
        or any(type(value) is not int or value < 0 for value in expansion.values())
    ):
        raise ValueError("registered Strict-A increment contract is invalid")
    increment_paths: dict[str, Path] = {}
    for field, relative_path in REGISTERED_STRICT_A_INCREMENT_FILENAMES.items():
        path = bundle_path.parent / relative_path
        if (
            not _is_sha256(increment.get(field))
            or not path.is_file()
            or sha256_file(path) != increment[field]
        ):
            raise ValueError("registered Strict-A increment artifact is invalid")
        increment_paths[field] = path
    if (
        increment["new_patient_ids_sha256"]
        != source_contracts.get("incremental_patient_allowlist_sha256")
    ):
        raise ValueError("registered Strict-A increment allowlist is inconsistent")

    candidate_patients = _read_registered_strict_a_increment_ids(
        increment_paths["new_patient_ids_sha256"], name="patient identities"
    )
    declared_visit_ids = _read_registered_strict_a_increment_ids(
        increment_paths["new_visit_ids_sha256"], name="visit identities"
    )
    declared_transition_ids = _read_registered_strict_a_increment_ids(
        increment_paths["new_transition_ids_sha256"], name="transition identities"
    )
    try:
        new_exclusions = pd.read_csv(increment_paths["new_exclusions_sha256"])
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        raise ValueError("registered Strict-A increment exclusions are unreadable") from None
    base_exclusion_count = base_data["exclusion_row_count"]
    if (
        tuple(new_exclusions.columns) != REGISTERED_STRICT_A_EXCLUSION_COLUMNS
        or _records_sha256(
            new_exclusions,
            identity="transition_id",
            columns=REGISTERED_STRICT_A_EXCLUSION_COLUMNS,
        )
        != _records_sha256(
            exclusions.iloc[base_exclusion_count:],
            identity="transition_id",
            columns=REGISTERED_STRICT_A_EXCLUSION_COLUMNS,
        )
    ):
        raise ValueError("registered Strict-A increment exclusions are inconsistent")
    summary = _read_json_object(
        increment_paths["summary_sha256"], artifact_name="increment summary"
    )
    if summary != expansion:
        raise ValueError("registered Strict-A increment summary is inconsistent")

    base_train_patients = {
        visit_id.rsplit(":", 1)[0]
        for visit_id in _normalization_visit_ids(normalization)
    }
    selected_new_patients = train_patients.difference(base_train_patients)
    new_transitions = transitions.loc[
        transitions["patient_id"].astype(str).isin(selected_new_patients)
    ]
    new_exclusions_frame = exclusions.iloc[base_exclusion_count:]
    published_new_visit_ids = set(
        visits.loc[
            visits["patient_id"].astype(str).isin(selected_new_patients),
            "visit_id",
        ].astype(str)
    )
    if (
        not selected_new_patients <= candidate_patients
        or not set(new_exclusions_frame["patient_id"].astype(str))
        <= candidate_patients
        or set(new_transitions["transition_id"].astype(str))
        != declared_transition_ids
        or published_new_visit_ids != declared_visit_ids
    ):
        raise ValueError("registered Strict-A increment identities are inconsistent")
    expected_expansion = {
        "candidate_patient_count": len(candidate_patients),
        "candidate_transition_count": len(new_transitions)
        + len(new_exclusions_frame),
        "excluded_transition_count": len(new_exclusions_frame),
        "selected_patient_count": len(selected_new_patients),
        "selected_transition_count": len(new_transitions),
    }
    if expansion != expected_expansion:
        raise ValueError("registered Strict-A expansion counts are inconsistent")


def validate_registered_strict_a_bundle_contract(
    payload: dict[str, Any], *, bundle_path: Path
) -> str:
    if payload.get("schema_version") != REGISTERED_STRICT_A_BUNDLE_SCHEMA:
        raise ValueError("registered Strict-A bundle schema is invalid")
    unsigned = dict(payload)
    digest = unsigned.pop("bundle_contract_sha256", None)
    if not _is_sha256(digest) or digest != _canonical_sha(unsigned):
        raise ValueError("registered Strict-A bundle contract SHA is invalid")

    if payload.get("split_policy") != "frozen_base_val_new_train_v1":
        return str(digest)
    base_cache = payload.get("base_cache_contract")
    base_data = payload.get("base_data_contract")
    artifacts = payload.get("artifacts")
    if (
        not isinstance(base_cache, dict)
        or set(base_cache) != REGISTERED_STRICT_A_BASE_CACHE_KEYS
        or base_cache.get("schema") != "motfm_ispy2_t0_fixed_roi_cache_v1"
        or any(
            not _is_sha256(base_cache.get(key))
            for key in REGISTERED_STRICT_A_BASE_CACHE_KEYS - {"schema"}
        )
        or not isinstance(base_data, dict)
        or set(base_data) != REGISTERED_STRICT_A_BASE_DATA_KEYS
        or base_data.get("schema") != "motfm_ispy2_frozen_base_data_v1"
        or not isinstance(artifacts, dict)
    ):
        raise ValueError("registered Strict-A base cache contract is invalid")
    if (
        base_data.get("bundle_contract_sha256")
        != base_cache["bundle_contract_sha256"]
        or not all(
            _is_sha256(base_data.get(key))
            for key in REGISTERED_STRICT_A_BASE_DATA_KEYS
            - {"schema", "exclusion_row_count"}
        )
        or type(base_data.get("exclusion_row_count")) is not int
        or base_data["exclusion_row_count"] < 0
    ):
        raise ValueError("registered Strict-A base cache contract is inconsistent")
    for artifact_name, contract_key in (
        ("preprocess", "preprocess_sha256"),
        ("normalization", "normalization_sha256"),
    ):
        descriptor = artifacts.get(artifact_name)
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("sha256") != base_cache[contract_key]
        ):
            raise ValueError(
                "registered Strict-A base cache contract is inconsistent"
            )
    (
        visits,
        transitions,
        exclusions,
        split,
        normalization,
        crop_plans,
    ) = _load_registered_strict_a_base_artifacts(bundle_path, artifacts)
    expected_base_data = _expected_registered_strict_a_base_data_contract(
        base_data=base_data,
        visits=visits,
        transitions=transitions,
        exclusions=exclusions,
        split=split,
        normalization=normalization,
        crop_plans=crop_plans,
    )
    if base_data != expected_base_data:
        raise ValueError("registered Strict-A base data contract is inconsistent")
    _validate_registered_strict_a_active_relations(
        payload=payload,
        bundle_path=bundle_path,
        base_data=base_data,
        visits=visits,
        transitions=transitions,
        exclusions=exclusions,
        split=split,
        normalization=normalization,
    )
    return str(digest)


@dataclass(frozen=True)
class LoadedTransitions:
    records: tuple[TransitionRecord, ...]
    visits: dict[str, VisitRecord]
    backend: str
    bundle_contract_sha256: str
    data_contract_sha256: str
    split_counts: dict[str, int]
    split_patient_counts: dict[str, int]
    bundle_schema_version: str


@dataclass(frozen=True)
class LoadedSplitVisits:
    visits: dict[str, VisitRecord]
    folds: dict[str, str]
    backend: str
    visits_artifact_sha256: str
    bundle_schema_version: str


def _bundle_schema(
    payload: dict[str, Any], *, backend: str, bundle_path: Path
) -> str:
    schema = payload.get("schema_version")
    if schema not in SUPPORTED_BUNDLE_SCHEMAS:
        raise ValueError("bundle schema version is unsupported")
    if (
        schema == REGISTERED_STRICT_A_BUNDLE_SCHEMA
        and backend != "registered_t0"
    ):
        raise ValueError(
            "registered Strict-A bundle requires the registered_t0 backend"
        )
    if schema == REGISTERED_STRICT_A_BUNDLE_SCHEMA:
        validate_registered_strict_a_bundle_contract(payload, bundle_path=bundle_path)
    return str(schema)


def _validate_strict_a_manifest(
    payload: dict[str, Any], manifest_path: Path, *, schema: str
) -> None:
    if schema != REGISTERED_STRICT_A_BUNDLE_SCHEMA:
        return
    source_contracts = payload.get("source_contracts")
    expected = (
        source_contracts.get("expanded_registered_manifest_sha256")
        if isinstance(source_contracts, dict)
        else None
    )
    if not _is_sha256(expected) or sha256_file(manifest_path) != expected:
        raise ValueError("registered Strict-A phase manifest SHA256 does not match")


def _load_locked_artifact(bundle_path: Path, payload: dict[str, Any], name: str) -> Path:
    descriptor = payload.get("artifacts", {}).get(name)
    if not isinstance(descriptor, dict):
        raise ValueError(f"bundle is missing the {name} artifact")
    path = (bundle_path.parent / str(descriptor.get("path", ""))).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"bundle {name} artifact is missing or unsafe")
    expected = descriptor.get("sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError(f"bundle {name} artifact SHA256 does not match")
    return path


def _required_text(row: pd.Series, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"transition {key} must be a nonempty string")
    return value.strip()


def _strict_a_visit_index(
    visit_frame: pd.DataFrame, *, required_visit_ids: set[str]
) -> dict[str, VisitRecord]:
    required_columns = {
        "patient_id",
        "visit_id",
        "visit",
        "dce0_path",
        "mask_path",
        "meta_path",
        "qc_status",
        "registration_status",
    }
    if not required_columns.issubset(visit_frame.columns):
        raise ValueError("registered Strict-A visits artifact lacks runtime columns")
    identities = visit_frame["visit_id"].astype(str)
    selected = visit_frame.loc[identities.isin(required_visit_ids)].copy()
    found = set(selected["visit_id"].astype(str))
    if found != required_visit_ids:
        raise ValueError("registered Strict-A visits artifact is missing locked visits")
    derived = selected["patient_id"].astype(str) + ":" + selected["visit"].astype(str)
    if not derived.equals(selected["visit_id"].astype(str)):
        raise ValueError("registered Strict-A visit identity is inconsistent")
    runtime_frame = selected.rename(columns={"dce0_path": "dce_paths"})
    runtime_frame["n_times"] = 1
    return build_visit_index(runtime_frame, backend="registered_t0")


def load_transition_records(
    bundle_json: str | Path,
    phase_manifest_csv: str | Path,
    *,
    backend: str,
) -> LoadedTransitions:
    if backend not in ACCEPTED_BACKENDS:
        raise ValueError("data backend is invalid")
    bundle_path = Path(bundle_json).resolve()
    manifest_path = Path(phase_manifest_csv).resolve()
    if not bundle_path.is_file() or not manifest_path.is_file():
        raise ValueError("bundle or phase manifest is missing")
    payload = json.loads(bundle_path.read_text())
    bundle_schema = _bundle_schema(payload, backend=backend, bundle_path=bundle_path)
    _validate_strict_a_manifest(payload, manifest_path, schema=bundle_schema)
    transition_path = _load_locked_artifact(bundle_path, payload, "transitions")
    source_transition_columns = [
        "transition_id",
        "patient_id",
        "fold",
        "transition_type",
        "source_visit_id",
        "target_visit_id",
        "delta_days",
        "clinical_text",
        "action_text",
    ]
    transition_frame = pd.read_csv(transition_path, usecols=source_transition_columns)
    descriptor = payload["artifacts"]["transitions"]
    if int(descriptor.get("row_count", -1)) != len(transition_frame):
        raise ValueError("bundle transition row count does not match")
    if (
        bundle_schema == REGISTERED_STRICT_A_BUNDLE_SCHEMA
        and int(payload.get("counts", {}).get("transition_count", -1))
        != len(transition_frame)
    ):
        raise ValueError("registered Strict-A transition count does not match")

    required_visit_ids = set(transition_frame["source_visit_id"].astype(str)) | set(
        transition_frame["target_visit_id"].astype(str)
    )
    if bundle_schema == REGISTERED_STRICT_A_BUNDLE_SCHEMA:
        visits_path = _load_locked_artifact(bundle_path, payload, "visits")
        visits = _strict_a_visit_index(
            pd.read_csv(visits_path), required_visit_ids=required_visit_ids
        )
    else:
        manifest_columns = [
            "patient_id",
            "visit",
            "n_times",
            "dce_paths",
            "mask_path",
            "meta_path",
            "qc_status",
        ]
        if backend == "registered_t0":
            manifest_columns.append("registration_status")
        manifest_frame = pd.read_csv(manifest_path, usecols=manifest_columns)
        manifest_visit_ids = (
            manifest_frame["patient_id"].astype(str)
            + ":"
            + manifest_frame["visit"].astype(str)
        )
        locked_manifest = manifest_frame.loc[
            manifest_visit_ids.isin(required_visit_ids)
        ].copy()
        locked_ids = set(
            locked_manifest["patient_id"].astype(str)
            + ":"
            + locked_manifest["visit"].astype(str)
        )
        missing_locked = required_visit_ids - locked_ids
        if missing_locked:
            raise ValueError(
                f"phase manifest is missing {len(missing_locked)} locked visits"
            )
        visits = build_visit_index(locked_manifest, backend=backend)
    records: list[TransitionRecord] = []
    patient_folds: dict[str, str] = {}
    for _, row in transition_frame.iterrows():
        transition_type = _required_text(row, "transition_type")
        if transition_type not in STAGE_BY_TRANSITION:
            raise ValueError("transition type is unsupported")
        patient_id = _required_text(row, "patient_id")
        fold = _required_text(row, "fold")
        previous_fold = patient_folds.setdefault(patient_id, fold)
        if previous_fold != fold:
            raise ValueError(f"patient appears in multiple folds: {patient_id}")
        source_visit_id = _required_text(row, "source_visit_id")
        target_visit_id = _required_text(row, "target_visit_id")
        try:
            source = visits[source_visit_id]
            target = visits[target_visit_id]
        except KeyError as exc:
            raise ValueError(f"transition visit is absent from phase manifest: {exc.args[0]}") from None
        if source.patient_id != patient_id or target.patient_id != patient_id:
            raise ValueError("transition patient does not match joined visits")
        clinical_text = ClinicalTextPolicy().validate(
            _required_text(row, "clinical_text")
        )
        records.append(
            TransitionRecord(
                transition_id=_required_text(row, "transition_id"),
                patient_id=patient_id,
                fold=fold,
                transition_type=transition_type,
                source_visit_id=source_visit_id,
                target_visit_id=target_visit_id,
                source_dce_paths=source.dce_paths,
                target_dce_paths=target.dce_paths,
                source_mask_path=source.mask_path,
                target_mask_path=target.mask_path,
                action_text=_required_text(row, "action_text"),
                clinical_text=clinical_text,
                delta_days=int(row["delta_days"]),
                stage_id=STAGE_BY_TRANSITION[transition_type],
            )
        )

    split_counts = {
        fold: sum(record.fold == fold for record in records)
        for fold in ("train", "val", "test")
    }
    expected_counts = payload.get("counts", {}).get("fold_transition_counts")
    if isinstance(expected_counts, dict):
        normalized = {fold: int(expected_counts.get(fold, 0)) for fold in split_counts}
        if normalized != split_counts:
            raise ValueError("bundle split transition counts do not match")
    split_patient_counts = {
        fold: len({record.patient_id for record in records if record.fold == fold})
        for fold in ("train", "val", "test")
    }
    expected_patient_counts = payload.get("counts", {}).get("fold_patient_counts")
    if isinstance(expected_patient_counts, dict):
        normalized = {
            fold: int(expected_patient_counts.get(fold, 0))
            for fold in split_patient_counts
        }
        if normalized != split_patient_counts:
            raise ValueError("bundle split patient counts do not match")
    bundle_contract = str(payload.get("bundle_contract_sha256", ""))
    if not bundle_contract:
        raise ValueError("bundle contract SHA256 is missing")
    if (
        bundle_schema == REGISTERED_STRICT_A_BUNDLE_SCHEMA
        and not _is_sha256(bundle_contract)
    ):
        raise ValueError("registered Strict-A bundle contract SHA256 is invalid")
    contract = _canonical_sha(
        {
            "schema": DATA_SCHEMA_VERSION,
            "bundle_contract_sha256": bundle_contract,
            "transition_sha256": descriptor["sha256"],
            "phase_manifest_sha256": sha256_file(manifest_path),
            "backend": backend,
            "split_counts": split_counts,
        }
    )
    return LoadedTransitions(
        records=tuple(records),
        visits=visits,
        backend=backend,
        bundle_contract_sha256=bundle_contract,
        data_contract_sha256=contract,
        split_counts=split_counts,
        split_patient_counts=split_patient_counts,
        bundle_schema_version=bundle_schema,
    )


def load_source_inference_records(
    bundle_json: str | Path,
    phase_manifest_csv: str | Path,
    *,
    backend: str,
) -> LoadedTransitions:
    """Load transition conditions while never constructing target visit metadata."""

    if backend not in ACCEPTED_BACKENDS:
        raise ValueError("data backend is invalid")
    bundle_path = Path(bundle_json).resolve()
    manifest_path = Path(phase_manifest_csv).resolve()
    if not bundle_path.is_file() or not manifest_path.is_file():
        raise ValueError("bundle or phase manifest is missing")
    payload = json.loads(bundle_path.read_text())
    bundle_schema = _bundle_schema(payload, backend=backend, bundle_path=bundle_path)
    _validate_strict_a_manifest(payload, manifest_path, schema=bundle_schema)
    transition_path = _load_locked_artifact(bundle_path, payload, "transitions")
    transition_frame = pd.read_csv(
        transition_path,
        usecols=[
            "transition_id",
            "patient_id",
            "fold",
            "transition_type",
            "source_visit_id",
            "target_visit_id",
            "delta_days",
            "clinical_text",
            "action_text",
        ],
    )
    descriptor = payload["artifacts"]["transitions"]
    if int(descriptor.get("row_count", -1)) != len(transition_frame):
        raise ValueError("bundle transition row count does not match")
    if (
        bundle_schema == REGISTERED_STRICT_A_BUNDLE_SCHEMA
        and int(payload.get("counts", {}).get("transition_count", -1))
        != len(transition_frame)
    ):
        raise ValueError("registered Strict-A transition count does not match")
    source_ids = set(transition_frame["source_visit_id"].astype(str))
    if bundle_schema == REGISTERED_STRICT_A_BUNDLE_SCHEMA:
        visits_path = _load_locked_artifact(bundle_path, payload, "visits")
        visits = _strict_a_visit_index(
            pd.read_csv(visits_path), required_visit_ids=source_ids
        )
    else:
        identity_frame = pd.read_csv(
            manifest_path, usecols=["patient_id", "visit"]
        )
        identity_ids = (
            identity_frame["patient_id"].astype(str)
            + ":"
            + identity_frame["visit"].astype(str)
        )
        selected_row_indices = set(identity_frame.index[identity_ids.isin(source_ids)])
        manifest_columns = [
            "patient_id",
            "visit",
            "n_times",
            "dce_paths",
            "mask_path",
            "meta_path",
            "qc_status",
        ]
        if backend == "registered_t0":
            manifest_columns.append("registration_status")
        source_manifest = pd.read_csv(
            manifest_path,
            usecols=manifest_columns,
            skiprows=lambda line_number: (
                line_number > 0 and (line_number - 1) not in selected_row_indices
            ),
        )
        found_ids = set(
            source_manifest["patient_id"].astype(str)
            + ":"
            + source_manifest["visit"].astype(str)
        )
        if source_ids - found_ids:
            raise ValueError("phase manifest is missing locked source visits")
        visits = build_visit_index(source_manifest, backend=backend)
    records: list[TransitionRecord] = []
    patient_folds: dict[str, str] = {}
    for _, row in transition_frame.iterrows():
        patient_id = _required_text(row, "patient_id")
        fold = _required_text(row, "fold")
        if patient_folds.setdefault(patient_id, fold) != fold:
            raise ValueError(f"patient appears in multiple folds: {patient_id}")
        transition_type = _required_text(row, "transition_type")
        if transition_type not in STAGE_BY_TRANSITION:
            raise ValueError("transition type is unsupported")
        source_visit_id = _required_text(row, "source_visit_id")
        source = visits[source_visit_id]
        if source.patient_id != patient_id:
            raise ValueError("transition patient does not match source visit")
        records.append(
            TransitionRecord(
                transition_id=_required_text(row, "transition_id"),
                patient_id=patient_id,
                fold=fold,
                transition_type=transition_type,
                source_visit_id=source_visit_id,
                target_visit_id=_required_text(row, "target_visit_id"),
                source_dce_paths=source.dce_paths,
                target_dce_paths=(),
                source_mask_path=source.mask_path,
                target_mask_path=Path(),
                action_text=_required_text(row, "action_text"),
                clinical_text=ClinicalTextPolicy().validate(
                    _required_text(row, "clinical_text")
                ),
                delta_days=int(row["delta_days"]),
                stage_id=STAGE_BY_TRANSITION[transition_type],
            )
        )
    split_counts = {
        fold: sum(record.fold == fold for record in records)
        for fold in ("train", "val", "test")
    }
    expected_counts = payload.get("counts", {}).get("fold_transition_counts")
    if isinstance(expected_counts, dict):
        normalized = {fold: int(expected_counts.get(fold, 0)) for fold in split_counts}
        if normalized != split_counts:
            raise ValueError("bundle split transition counts do not match")
    split_patient_counts = {
        fold: len({record.patient_id for record in records if record.fold == fold})
        for fold in ("train", "val", "test")
    }
    expected_patient_counts = payload.get("counts", {}).get("fold_patient_counts")
    if isinstance(expected_patient_counts, dict):
        normalized = {
            fold: int(expected_patient_counts.get(fold, 0))
            for fold in split_patient_counts
        }
        if normalized != split_patient_counts:
            raise ValueError("bundle split patient counts do not match")
    bundle_contract = str(payload.get("bundle_contract_sha256", ""))
    if not bundle_contract:
        raise ValueError("bundle contract SHA256 is missing")
    if (
        bundle_schema == REGISTERED_STRICT_A_BUNDLE_SCHEMA
        and not _is_sha256(bundle_contract)
    ):
        raise ValueError("registered Strict-A bundle contract SHA256 is invalid")
    contract = _canonical_sha(
        {
            "schema": DATA_SCHEMA_VERSION,
            "bundle_contract_sha256": bundle_contract,
            "transition_sha256": descriptor["sha256"],
            "phase_manifest_sha256": sha256_file(manifest_path),
            "backend": backend,
            "split_counts": split_counts,
        }
    )
    return LoadedTransitions(
        records=tuple(records),
        visits=visits,
        backend=backend,
        bundle_contract_sha256=bundle_contract,
        data_contract_sha256=contract,
        split_counts=split_counts,
        split_patient_counts=split_patient_counts,
        bundle_schema_version=bundle_schema,
    )


def load_split_visit_records(
    bundle_json: str | Path,
    phase_manifest_csv: str | Path,
    *,
    backend: str,
) -> LoadedSplitVisits:
    if backend not in ACCEPTED_BACKENDS:
        raise ValueError("data backend is invalid")
    bundle_path = Path(bundle_json).resolve()
    manifest_path = Path(phase_manifest_csv).resolve()
    if not bundle_path.is_file() or not manifest_path.is_file():
        raise ValueError("bundle or phase manifest is missing")
    payload = json.loads(bundle_path.read_text())
    bundle_schema = _bundle_schema(payload, backend=backend, bundle_path=bundle_path)
    _validate_strict_a_manifest(payload, manifest_path, schema=bundle_schema)
    visits_path = _load_locked_artifact(bundle_path, payload, "visits")
    split_frame = pd.read_csv(visits_path)
    descriptor = payload["artifacts"]["visits"]
    if int(descriptor.get("row_count", -1)) != len(split_frame):
        raise ValueError("bundle visit row count does not match")
    if (
        bundle_schema == REGISTERED_STRICT_A_BUNDLE_SCHEMA
        and int(payload.get("counts", {}).get("visit_count", -1)) != len(split_frame)
    ):
        raise ValueError("registered Strict-A visit count does not match")
    required_columns = {"patient_id", "visit_id", "fold"}
    if not required_columns.issubset(split_frame.columns):
        raise ValueError("bundle visit artifact lacks split identity columns")
    if split_frame["visit_id"].duplicated().any():
        raise ValueError("bundle visit artifact contains duplicate visits")
    required_ids = set(split_frame["visit_id"].astype(str))
    if bundle_schema == REGISTERED_STRICT_A_BUNDLE_SCHEMA:
        visits = _strict_a_visit_index(
            split_frame, required_visit_ids=required_ids
        )
    else:
        manifest_columns = [
            "patient_id",
            "visit",
            "n_times",
            "dce_paths",
            "mask_path",
            "meta_path",
            "qc_status",
        ]
        if backend == "registered_t0":
            manifest_columns.append("registration_status")
        manifest_frame = pd.read_csv(manifest_path, usecols=manifest_columns)
        manifest_ids = (
            manifest_frame["patient_id"].astype(str)
            + ":"
            + manifest_frame["visit"].astype(str)
        )
        locked_manifest = manifest_frame.loc[manifest_ids.isin(required_ids)].copy()
        found_ids = set(
            locked_manifest["patient_id"].astype(str)
            + ":"
            + locked_manifest["visit"].astype(str)
        )
        if required_ids - found_ids:
            raise ValueError("phase manifest is missing bundle split visits")
        visits = build_visit_index(locked_manifest, backend=backend)
    folds: dict[str, str] = {}
    patient_folds: dict[str, str] = {}
    for _, row in split_frame.iterrows():
        visit_id = _required_text(row, "visit_id")
        patient_id = _required_text(row, "patient_id")
        fold = _required_text(row, "fold")
        if fold not in {"train", "val", "test"}:
            raise ValueError("bundle visit fold is invalid")
        if visits[visit_id].patient_id != patient_id:
            raise ValueError("bundle visit patient identity does not match phase manifest")
        if patient_folds.setdefault(patient_id, fold) != fold:
            raise ValueError(f"patient appears in multiple folds: {patient_id}")
        folds[visit_id] = fold
    return LoadedSplitVisits(
        visits=visits,
        folds=folds,
        backend=backend,
        visits_artifact_sha256=descriptor["sha256"],
        bundle_schema_version=bundle_schema,
    )
