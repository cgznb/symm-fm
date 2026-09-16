"""Import the audited MeWM continuous-latent cache for paired training.

The upstream payloads contain unnormalized, pre-quantization VQGAN latents.
This importer validates their complete numeric/data identity, applies the
train-cohort channel statistics, and emits this project's provenance-bound NPZ
contract.  No image, mask, FTV, or outcome is admitted as a model condition.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ispy2_symmflow.data.mewm import (
    MewmMetadataValidationResult,
    validate_mewm_bundle_metadata,
)
from ispy2_symmflow.training.datasets import write_jsonl
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    LATENT_STATISTICS_FINGERPRINT,
    bind_cached_pair_manifest_provenance,
    validate_cached_latent_provenance,
)
from ispy2_symmflow.utils.hashing import sha256_file, stable_hash


CACHE_SCHEMA = "mewm_ispy2_biflow_continuous_latents_v2"
PAYLOAD_SCHEMA = "mewm_ispy2_biflow_continuous_latent_payload_v2"
STORED_REPRESENTATION = "continuous_prequantization_float16_v1"
NORMALIZATION = "continuous_train_unique_visit_channel_zscore_v1"
NUMERIC_CONTRACT = "motfm_registered_global_zscore_v1"
COORDINATE_FRAME = "t0_fixed_registered_local_v1"
CROP_POLICY = "single_T0_mask_bbox_center_reused_for_all_visits"
LATENT_SHAPE = (8, 24, 64, 64)
IMAGE_SHAPE = (96, 256, 256)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "visit_id",
        "split",
        "continuous_latent",
        "vqgan_sha256",
        "codebook_sha256",
        "data_contract_sha256",
        "normalization",
        "latent_statistics_sha256",
    }
)


@dataclass(frozen=True)
class MewmLatentImportResult:
    pair_manifest_path: str
    visit_manifest_path: str
    statistics_path: str
    audit_path: str
    pair_count: int
    visit_count: int
    split_pair_counts: Mapping[str, int]
    autoencoder_id: str


@dataclass(frozen=True)
class _SourceVisit:
    visit_id: str
    patient_id: str
    stage: str
    split: str
    study_uid: str
    date: str | None
    date_source: str
    age: float | None
    hr: str | None
    her2: str | None
    mammaprint: str | None
    menopausal_status: str | None
    treatment_arm: str | None
    registration_status: str
    quality_pass: bool


@dataclass(frozen=True)
class _ValidatedPayload:
    path: Path
    sha256: str
    latent: torch.Tensor


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _read_csv(path: Path, *, required: set[str], label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"{label} is missing columns: {missing}")
            return [dict(row) for row in reader]
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc


def _digest(value: Any, *, label: str) -> str:
    text = str(value).strip().lower()
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if str(result) != str(value).strip() or result < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return result


def _optional_float(value: Any, *, label: str) -> float | None:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "na", "nan", "none", "unknown"}:
        return None
    try:
        result = float(text)
    except ValueError as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _optional_category(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "na", "nan", "none", "unknown"}:
        return None
    if text.endswith(".0"):
        try:
            number = float(text)
        except ValueError:
            pass
        else:
            if number.is_integer():
                text = str(int(number))
    return text


def _boolean(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label} must be boolean")


def _stage_pair(earlier: str, later: str) -> tuple[str, str]:
    pattern = re.compile(r"T([0-9]+)")
    left, right = pattern.fullmatch(earlier), pattern.fullmatch(later)
    if left is None or right is None or int(left.group(1)) >= int(right.group(1)):
        raise ValueError("time pair must use increasing T<number> labels")
    return earlier, later


def _time_pairs(
    value: Sequence[Sequence[str]] | None,
    *,
    earlier_stage: str,
    later_stage: str,
) -> tuple[tuple[str, str], ...]:
    raw_pairs: Sequence[Sequence[str]] = (
        ((earlier_stage, later_stage),) if value is None else value
    )
    if not raw_pairs:
        raise ValueError("at least one time pair is required")
    pairs = tuple(_stage_pair(str(pair[0]), str(pair[1])) for pair in raw_pairs if len(pair) == 2)
    if len(pairs) != len(raw_pairs):
        raise ValueError("each time pair must contain exactly [earlier, later]")
    if len(pairs) != len(set(pairs)):
        raise ValueError("time pairs must be unique")
    return pairs


def _artifact_path(bundle_dir: Path, entry: Mapping[str, Any], *, label: str) -> Path:
    relative = Path(str(entry.get("path", "")))
    if relative.is_absolute() or not relative.parts:
        raise ValueError(f"{label} must use a relative bundle path")
    result = (bundle_dir / relative).resolve()
    try:
        result.relative_to(bundle_dir)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the bundle directory") from exc
    if not result.is_file() or result.is_symlink():
        raise ValueError(f"{label} is not a regular file: {result}")
    expected = _digest(entry.get("sha256"), label=f"{label} sha256")
    if sha256_file(result) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return result


def _validate_sources(
    cache_identity_path: Path,
    bundle_dir: Path,
    vqgan_checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    identity = _read_json(cache_identity_path, label="MeWM cache identity")
    expected_identity = {
        "schema": CACHE_SCHEMA,
        "payload_schema": PAYLOAD_SCHEMA,
        "stored_representation": STORED_REPRESENTATION,
        "normalization": NORMALIZATION,
        "numeric_contract": NUMERIC_CONTRACT,
    }
    for key, expected in expected_identity.items():
        if identity.get(key) != expected:
            raise ValueError(
                f"unsupported cache identity {key}: {identity.get(key)!r}; expected {expected!r}"
            )
    if tuple(identity.get("latent_shape_czyx", ())) != LATENT_SHAPE:
        raise ValueError("MeWM cache latent_shape_czyx is incompatible")
    if tuple(identity.get("input_shape_zyx", ())) != IMAGE_SHAPE:
        raise ValueError("MeWM cache input_shape_zyx is incompatible")
    if identity.get("latent_dtype") != "float16":
        raise ValueError("MeWM source cache must store float16 latents")
    statistics = identity.get("latent_statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("MeWM cache identity has no latent_statistics object")
    stats_sha = _digest(statistics.get("sha256"), label="latent statistics sha256")
    if stable_hash({key: value for key, value in statistics.items() if key != "sha256"}) != stats_sha:
        raise ValueError("MeWM latent statistics SHA-256 does not match its contents")
    if statistics.get("source_split") != "train" or statistics.get("variance_estimator") != "population_ddof0":
        raise ValueError("MeWM latent statistics were not fitted on the expected train contract")
    mean = np.asarray(statistics.get("mean"), dtype=np.float64)
    std = np.asarray(statistics.get("std"), dtype=np.float64)
    if mean.shape != (LATENT_SHAPE[0],) or not np.isfinite(mean).all():
        raise ValueError("MeWM latent channel means are invalid")
    if std.shape != (LATENT_SHAPE[0],) or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("MeWM latent channel standard deviations are invalid")

    if not vqgan_checkpoint.is_file() or vqgan_checkpoint.is_symlink():
        raise ValueError(f"VQGAN checkpoint is not a regular file: {vqgan_checkpoint}")
    expected_vqgan = _digest(identity.get("vqgan_sha256"), label="vqgan_sha256")
    if sha256_file(vqgan_checkpoint) != expected_vqgan:
        raise ValueError("VQGAN checkpoint SHA-256 differs from the latent cache")

    bundle_path = (bundle_dir / "bundle.json").resolve()
    bundle = _read_json(bundle_path, label="MeWM bundle")
    expected_bundle_file = _digest(
        identity.get("bundle_json_sha256"), label="bundle_json_sha256"
    )
    if sha256_file(bundle_path) != expected_bundle_file:
        raise ValueError("bundle.json SHA-256 differs from the latent cache identity")
    contract = _digest(bundle.get("bundle_contract_sha256"), label="bundle contract")
    if stable_hash({key: value for key, value in bundle.items() if key != "bundle_contract_sha256"}) != contract:
        raise ValueError("bundle contract SHA-256 does not match bundle.json contents")
    if contract != _digest(identity.get("bundle_contract_sha256"), label="cache bundle contract"):
        raise ValueError("bundle contract differs from the latent cache identity")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("bundle.json has no artifacts object")
    required_artifacts = {"transitions", "visits", "preprocess", "normalization"}
    if not required_artifacts.issubset(artifacts):
        raise ValueError("bundle.json lacks required transition/visit preprocessing artifacts")
    paths: dict[str, Path] = {}
    for name in required_artifacts:
        entry = artifacts[name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"bundle artifact {name!r} is not an object")
        paths[name] = _artifact_path(bundle_dir, entry, label=f"bundle artifact {name!r}")
    preprocess = _read_json(paths["preprocess"], label="MeWM preprocessing contract")
    if preprocess.get("coordinate_frame") != COORDINATE_FRAME:
        raise ValueError("MeWM preprocessing is not in the audited T0-fixed coordinate frame")
    if preprocess.get("crop_policy") != CROP_POLICY:
        raise ValueError("MeWM preprocessing does not use the audited T0 crop policy")
    if tuple(preprocess.get("output_shape_zyx", ())) != IMAGE_SHAPE:
        raise ValueError("MeWM preprocessing output shape is incompatible")
    return identity, bundle, paths["transitions"], paths["visits"]


def _source_visits(path: Path) -> tuple[list[_SourceVisit], dict[str, _SourceVisit]]:
    required = {
        "patient_id",
        "visit_id",
        "visit",
        "study_instance_uid",
        "visit_date",
        "visit_date_source",
        "fold",
        "HR",
        "HER2",
        "MP",
        "Age_at_Screening",
        "menopausal_status",
        "trial_arm",
        "registration_status",
        "quality_pass",
    }
    rows = _read_csv(path, required=required, label="visits.csv")
    visits: list[_SourceVisit] = []
    by_id: dict[str, _SourceVisit] = {}
    patient_splits: dict[str, str] = {}
    for row in rows:
        visit_id = row["visit_id"].strip()
        patient_id = row["patient_id"].strip()
        split = row["fold"].strip()
        stage = row["visit"].strip()
        if not visit_id or not patient_id or split not in {"train", "val"}:
            raise ValueError("visits.csv contains an invalid identity or split")
        if visit_id != f"{patient_id}:{stage}":
            raise ValueError(f"visit identity is inconsistent: {visit_id!r}")
        previous = patient_splits.setdefault(patient_id, split)
        if previous != split:
            raise ValueError(f"patient {patient_id!r} occurs across splits")
        value = _SourceVisit(
            visit_id=visit_id,
            patient_id=patient_id,
            stage=stage,
            split=split,
            study_uid=row["study_instance_uid"].strip(),
            date=row["visit_date"].strip() or None,
            date_source=row["visit_date_source"].strip(),
            age=_optional_float(row.get("Age_at_Screening"), label=f"age for {patient_id}"),
            hr=_optional_category(row.get("HR")),
            her2=_optional_category(row.get("HER2")),
            mammaprint=_optional_category(row.get("MP")),
            menopausal_status=_optional_category(row.get("menopausal_status")),
            treatment_arm=_optional_category(row.get("trial_arm")),
            registration_status=row["registration_status"].strip(),
            quality_pass=_boolean(row.get("quality_pass"), label=f"quality_pass for {visit_id}"),
        )
        if not value.study_uid or not value.registration_status:
            raise ValueError(f"visit {visit_id!r} lacks study or registration provenance")
        if value.age is not None and not 0 < value.age < 120:
            raise ValueError(f"age for {patient_id!r} is outside the valid human range")
        if visit_id in by_id:
            raise ValueError(f"visits.csv duplicates visit_id {visit_id!r}")
        by_id[visit_id] = value
        visits.append(value)
    return visits, by_id


def _baseline(visit: _SourceVisit) -> dict[str, Any]:
    values = {
        "age": visit.age,
        "hr_status": visit.hr,
        "her2_status": visit.her2,
        "mammaprint": visit.mammaprint,
        "menopausal_status": visit.menopausal_status,
    }
    return {key: value for key, value in values.items() if value is not None}


def _treatment(visit: _SourceVisit) -> dict[str, Any]:
    return {"treatment_arm": visit.treatment_arm} if visit.treatment_arm else {}


def _payload_path(source_dir: Path, visit_id: str) -> Path:
    patient_id, stage = visit_id.rsplit(":", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", patient_id) or not re.fullmatch(r"T[0-9]+", stage):
        raise ValueError(f"unsafe or unsupported visit_id {visit_id!r}")
    result = (source_dir / f"{patient_id}__{stage}.pt").resolve()
    try:
        result.relative_to(source_dir)
    except ValueError as exc:
        raise ValueError(f"latent path escapes source directory for {visit_id!r}") from exc
    return result


def _validate_payload(
    path: Path,
    *,
    visit: _SourceVisit,
    identity: Mapping[str, Any],
) -> _ValidatedPayload:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"source latent is not a regular file: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        raise ValueError(f"cannot safely load MeWM latent payload: {path}") from exc
    if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_KEYS:
        observed = sorted(payload) if isinstance(payload, Mapping) else type(payload).__name__
        raise ValueError(f"MeWM latent payload keys changed for {path}: {observed}")
    expected = {
        "schema": PAYLOAD_SCHEMA,
        "visit_id": visit.visit_id,
        "split": visit.split,
        "vqgan_sha256": identity["vqgan_sha256"],
        "codebook_sha256": identity["codebook_sha256"],
        "data_contract_sha256": identity["data_contract_sha256"],
        "normalization": identity["normalization"],
        "latent_statistics_sha256": identity["latent_statistics"]["sha256"],
    }
    for key, wanted in expected.items():
        if payload.get(key) != wanted:
            raise ValueError(
                f"MeWM payload {key} mismatch for {visit.visit_id!r}: "
                f"observed={payload.get(key)!r}, expected={wanted!r}"
            )
    latent = payload["continuous_latent"]
    if not isinstance(latent, torch.Tensor):
        raise ValueError(f"MeWM payload for {visit.visit_id!r} has no tensor latent")
    if latent.dtype != torch.float16 or tuple(latent.shape) != LATENT_SHAPE:
        raise ValueError(
            f"MeWM payload for {visit.visit_id!r} must be float16 {LATENT_SHAPE}"
        )
    if not torch.isfinite(latent).all():
        raise ValueError(f"MeWM payload for {visit.visit_id!r} contains non-finite values")
    return _ValidatedPayload(path=path, sha256=sha256_file(path), latent=latent)


def _pair_rows(
    path: Path,
    visits: Mapping[str, _SourceVisit],
    *,
    time_pairs: Sequence[tuple[str, str]],
    selected_pair_ids: Sequence[str] | None,
) -> list[dict[str, str]]:
    required = {
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
        "source_ftv_volume_cc",
        "target_ftv_volume_cc",
        "source_ftv_is_condition",
        "target_ftv_is_condition",
        "target_ftv_is_audit_only",
        "action_text",
    }
    rows = _read_csv(path, required=required, label="transitions.csv")
    edges: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        pair_id = row["transition_id"]
        earlier_id, later_id = row["source_visit_id"], row["target_visit_id"]
        if earlier_id not in visits or later_id not in visits:
            raise ValueError(f"transition {pair_id!r} references a missing visit")
        earlier, later = visits[earlier_id], visits[later_id]
        source_stage, target_stage = _stage_pair(
            row["source_visit"].strip(), row["target_visit"].strip()
        )
        source_index = int(source_stage.removeprefix("T"))
        target_index = int(target_stage.removeprefix("T"))
        if (
            target_index != source_index + 1
            or row["transition_type"] != f"{source_stage}->{target_stage}"
            or pair_id != f"{earlier.patient_id}:{source_stage}->{target_stage}"
            or earlier.stage != source_stage
            or later.stage != target_stage
        ):
            raise ValueError(f"transition {pair_id!r} has inconsistent stage semantics")
        if (
            earlier.patient_id != later.patient_id
            or row["patient_id"] != earlier.patient_id
            or row["fold"] != earlier.split
            or earlier.split != later.split
        ):
            raise ValueError(f"transition {pair_id!r} has inconsistent patient/split identity")
        if (
            row["source_study_instance_uid"] != earlier.study_uid
            or row["target_study_instance_uid"] != later.study_uid
        ):
            raise ValueError(f"transition {pair_id!r} has inconsistent study identity")
        if not earlier.quality_pass or not later.quality_pass:
            raise ValueError(f"transition {pair_id!r} contains an endpoint that failed QC")
        if _baseline(earlier) != _baseline(later) or _treatment(earlier) != _treatment(later):
            raise ValueError(f"transition {pair_id!r} has inconsistent baseline conditions")
        arm = earlier.treatment_arm
        expected_action = f"treatment arm {arm}" if arm else ""
        if row["action_text"].strip() != expected_action:
            raise ValueError(f"transition {pair_id!r} action text differs from trial_arm")
        if not _boolean(row["source_ftv_is_condition"], label=f"{pair_id} source FTV flag"):
            raise ValueError(f"transition {pair_id!r} source FTV contract changed")
        if _boolean(row["target_ftv_is_condition"], label=f"{pair_id} target FTV flag"):
            raise ValueError(f"transition {pair_id!r} leaks target FTV as a condition")
        if not _boolean(row["target_ftv_is_audit_only"], label=f"{pair_id} target FTV audit flag"):
            raise ValueError(f"transition {pair_id!r} target FTV is not audit-only")
        _integer(row["delta_days"], label=f"{pair_id} delta_days", minimum=1)
        edge_key = (earlier.patient_id, source_index)
        if edge_key in edges:
            raise ValueError(
                f"transitions.csv duplicates adjacent edge for {earlier.patient_id}:{source_stage}"
            )
        edges[edge_key] = row

    candidates: list[dict[str, str]] = []
    patient_ids = sorted({visit.patient_id for visit in visits.values()})
    ordered_pairs = sorted(
        time_pairs,
        key=lambda pair: (
            int(pair[0].removeprefix("T")),
            int(pair[1].removeprefix("T")),
        ),
    )
    for patient_id in patient_ids:
        for earlier_stage, later_stage in ordered_pairs:
            earlier_index = int(earlier_stage.removeprefix("T"))
            later_index = int(later_stage.removeprefix("T"))
            chain = [edges.get((patient_id, index)) for index in range(earlier_index, later_index)]
            if any(edge is None for edge in chain):
                continue
            typed_chain = [edge for edge in chain if edge is not None]
            first, last = typed_chain[0], typed_chain[-1]
            pair = dict(first)
            pair.update(
                {
                    "transition_id": f"{patient_id}:{earlier_stage}->{later_stage}",
                    "transition_type": f"{earlier_stage}->{later_stage}",
                    "source_visit": earlier_stage,
                    "target_visit": later_stage,
                    "target_visit_id": last["target_visit_id"],
                    "target_study_instance_uid": last["target_study_instance_uid"],
                    "target_ftv_volume_cc": last["target_ftv_volume_cc"],
                    "target_ftv_is_condition": last["target_ftv_is_condition"],
                    "target_ftv_is_audit_only": last["target_ftv_is_audit_only"],
                    "delta_days": str(
                        sum(
                            _integer(
                                edge["delta_days"],
                                label=f"{edge['transition_id']} delta_days",
                                minimum=1,
                            )
                            for edge in typed_chain
                        )
                    ),
                }
            )
            candidates.append(pair)

    by_id = {row["transition_id"]: row for row in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("derived connected pair IDs are duplicated")
    if selected_pair_ids is None:
        selected = candidates
    else:
        requested = [str(value).strip() for value in selected_pair_ids]
        if any(not value for value in requested) or len(set(requested)) != len(requested):
            raise ValueError("selected_pair_ids must contain unique non-empty IDs")
        missing = [value for value in requested if value not in by_id]
        if missing:
            raise ValueError(f"requested pairs are absent from the connected cohort: {missing}")
        requested_set = set(requested)
        selected = [row for row in candidates if row["transition_id"] in requested_set]
    if not selected:
        raise ValueError("no configured connected pairs were selected")
    return selected


def _relative_source(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _validate_geometry_binding(
    geometry: MewmMetadataValidationResult,
    *,
    identity: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
) -> None:
    """Bind the latent-cache cohort to the independently checked image grids."""

    if geometry.bundle_json_sha256 != identity["bundle_json_sha256"]:
        raise ValueError("geometry metadata bundle.json differs from the latent cache")
    if geometry.bundle_contract_sha256 != identity["bundle_contract_sha256"]:
        raise ValueError("geometry metadata bundle contract differs from the latent cache")

    expected_ids = {row["transition_id"] for row in rows}
    observed_ids = {pair.pair_id for pair in geometry.pairs}
    if observed_ids != expected_ids:
        raise ValueError("geometry metadata pair cohort differs from the latent pair cohort")

    for row in rows:
        endpoints = (row["source_visit_id"], row["target_visit_id"])
        pair = geometry.pair_geometry_by_endpoints.get(endpoints)
        if pair is None:
            raise ValueError(
                f"geometry metadata lacks endpoints for {row['transition_id']!r}"
            )
        expected = {
            "pair_id": row["transition_id"],
            "patient_id": row["patient_id"],
            "split": row["fold"],
            "earlier_stage": row["source_visit"],
            "later_stage": row["target_visit"],
            "earlier_visit_id": row["source_visit_id"],
            "later_visit_id": row["target_visit_id"],
            "delta_days": int(row["delta_days"]),
            "action_text": row["action_text"].strip(),
        }
        for field, wanted in expected.items():
            if getattr(pair, field) != wanted:
                raise ValueError(
                    f"geometry metadata {field} differs for {row['transition_id']!r}"
                )


def import_mewm_continuous_latents(
    cache_identity_path: str | Path,
    bundle_dir: str | Path,
    source_latent_dir: str | Path,
    vqgan_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    upstream_metadata_dir: str | Path,
    earlier_stage: str = "T0",
    later_stage: str = "T1",
    time_pairs: Sequence[Sequence[str]] | None = None,
    selected_pair_ids: Sequence[str] | None = None,
) -> MewmLatentImportResult:
    """Convert configured MeWM intervals into normalized, bound latent archives."""

    configured_pairs = _time_pairs(
        time_pairs, earlier_stage=earlier_stage, later_stage=later_stage
    )
    identity_path = Path(cache_identity_path).expanduser().resolve()
    bundle_root = Path(bundle_dir).expanduser().resolve()
    source_root = Path(source_latent_dir).expanduser().resolve()
    checkpoint_path = Path(vqgan_checkpoint).expanduser().resolve()
    metadata_root = Path(upstream_metadata_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"MeWM latent import output already exists: {destination}")
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError(f"source latent directory is missing: {source_root}")

    identity, bundle, transitions_path, visits_path = _validate_sources(
        identity_path, bundle_root, checkpoint_path
    )
    all_visits, visits_by_id = _source_visits(visits_path)
    rows = _pair_rows(
        transitions_path,
        visits_by_id,
        time_pairs=configured_pairs,
        selected_pair_ids=selected_pair_ids,
    )
    geometry = validate_mewm_bundle_metadata(
        bundle_root,
        metadata_root,
        time_pairs=configured_pairs,
        selected_pair_ids=selected_pair_ids,
    )
    _validate_geometry_binding(geometry, identity=identity, rows=rows)
    split_counts = Counter(row["fold"] for row in rows)
    missing_splits = {"train", "val"} - set(split_counts)
    if missing_splits:
        raise ValueError(
            "paired training import requires train and val transitions; missing "
            + ", ".join(sorted(missing_splits))
        )

    identity_visit_ids = identity.get("visit_ids")
    if not isinstance(identity_visit_ids, list) or any(
        not isinstance(value, str) for value in identity_visit_ids
    ):
        raise ValueError("MeWM cache identity has no valid ordered visit_ids")
    if len(identity_visit_ids) != int(identity.get("visit_count", -1)) or len(
        set(identity_visit_ids)
    ) != len(identity_visit_ids):
        raise ValueError("MeWM cache identity visit IDs/count are inconsistent")
    identity_visit_set = set(identity_visit_ids)
    selected_visit_ids = list(
        dict.fromkeys(
            visit_id
            for row in rows
            for visit_id in (row["source_visit_id"], row["target_visit_id"])
        )
    )
    missing_identity = sorted(set(selected_visit_ids) - identity_visit_set)
    if missing_identity:
        raise ValueError(f"selected visits are absent from the latent cache identity: {missing_identity}")

    payloads: dict[str, _ValidatedPayload] = {}
    for visit_id in selected_visit_ids:
        visit = visits_by_id[visit_id]
        payloads[visit_id] = _validate_payload(
            _payload_path(source_root, visit_id), visit=visit, identity=identity
        )

    statistics_source = identity["latent_statistics"]
    mean = torch.tensor(statistics_source["mean"], dtype=torch.float32).view(-1, 1, 1, 1)
    std = torch.tensor(statistics_source["std"], dtype=torch.float32).view(-1, 1, 1, 1)
    autoencoder_id = str(identity["vqgan_sha256"])
    cache_visits = [visits_by_id[value] for value in identity_visit_ids if value in visits_by_id]
    if len(cache_visits) != len(identity_visit_ids):
        raise ValueError("latent cache identity references visits absent from visits.csv")
    split_by_patient: dict[str, str] = {}
    for visit in cache_visits:
        previous = split_by_patient.setdefault(visit.patient_id, visit.split)
        if previous != visit.split:
            raise ValueError(f"patient {visit.patient_id!r} crosses source-cache splits")
    train_cache_visits = [visit for visit in cache_visits if visit.split == "train"]
    if len(train_cache_visits) != int(statistics_source.get("visit_count", -1)):
        raise ValueError("train visit count differs from MeWM latent-statistics provenance")
    source_preprocessing_signature = {
        "kind": "mewm_external_registered_roi",
        "coordinate_frame": COORDINATE_FRAME,
        "crop_policy": CROP_POLICY,
        "crop_anchor_stage": "T0",
        "registration_assisted": True,
        "backward_reconstruction_scope": "T0-registration-and-crop-assisted",
        "image_shape_czyx": [1, *IMAGE_SHAPE],
        "image_channels": ["dce0"],
        "spacing_dhw": list(geometry.spacing_dhw),
        "geometry_source": "upstream_registered_meta_dicom_absolute_lps",
        "header_affine_semantics": "physical_lps_affine_from_signed_upstream_metadata",
        "numeric_contract": identity["numeric_contract"],
        "bundle_contract_sha256": identity["bundle_contract_sha256"],
        "bundle_artifact_sha256": dict(geometry.artifact_sha256),
        "roi_cache_contract": identity["roi_cache_contract"],
    }
    base_statistics = {
        "mean": list(statistics_source["mean"]),
        "std": list(statistics_source["std"]),
        "element_count_per_channel": int(statistics_source["element_count_per_channel"]),
        "fit_split": "train",
        "fit_patient_ids": sorted({visit.patient_id for visit in train_cache_visits}),
        "fit_visit_ids": sorted(visit.visit_id for visit in train_cache_visits),
        "autoencoder_id": autoencoder_id,
        "source_preprocessing_signature": source_preprocessing_signature,
        "source_manifest": str(identity_path),
        "source_split_hash": stable_hash(split_by_patient),
        "source_ordered_manifest_fingerprint": stable_hash(identity_visit_ids),
        "source_manifest_record_count": len(identity_visit_ids),
        "source_cache_identity_sha256": sha256_file(identity_path),
        "source_latent_statistics_sha256": statistics_source["sha256"],
        "source_data_contract_sha256": identity["data_contract_sha256"],
        "source_bundle_contract_sha256": identity["bundle_contract_sha256"],
        "source_codebook_sha256": identity["codebook_sha256"],
        "source_payload_schema": identity["payload_schema"],
        "source_stored_representation": identity["stored_representation"],
        "normalization": identity["normalization"],
        "normalization_application": "imported_npz_equals_(continuous_latent-mean)/std",
        "decoder_contract": "denormalize_then_quantize_then_vqgan_decode",
    }

    final_latent_dir = destination / "visits"
    latent_paths = {
        visit_id: final_latent_dir / f"{visit_id.replace(':', '__')}.npz"
        for visit_id in selected_visit_ids
    }
    unsigned_pairs: list[dict[str, Any]] = []
    for row in rows:
        earlier = visits_by_id[row["source_visit_id"]]
        later = visits_by_id[row["target_visit_id"]]
        pair_geometry = geometry.pair_geometry_by_endpoints[
            (earlier.visit_id, later.visit_id)
        ]
        source_ftv = _optional_float(
            row.get("source_ftv_volume_cc"), label=f"{row['transition_id']} source FTV"
        )
        target_ftv = _optional_float(
            row.get("target_ftv_volume_cc"), label=f"{row['transition_id']} target FTV"
        )
        unsigned_pairs.append(
            {
                "schema_version": "mewm-continuous-latent-import-1.0",
                "pair_id": row["transition_id"],
                "patient_id": earlier.patient_id,
                "collection": "ISPY2_registered_strict_a",
                "earlier_visit_id": earlier.visit_id,
                "later_visit_id": later.visit_id,
                "earlier_stage": earlier.stage,
                "later_stage": later.stage,
                "split": earlier.split,
                "delta_days": int(row["delta_days"]),
                "observed_delta_days": int(row["delta_days"]),
                "interval_missing": False,
                "interval_source": f"source_bundle:{earlier.date_source}",
                "baseline_clinical": _baseline(earlier),
                "treatment": _treatment(earlier),
                "evaluation_metadata": {
                    "source_ftv_volume_cc": source_ftv,
                    "target_ftv_volume_cc": target_ftv,
                    "source_registration_status": earlier.registration_status,
                    "target_registration_status": later.registration_status,
                    "target_ftv_is_audit_only": True,
                    "edge_transition_ids": list(pair_geometry.edge_transition_ids),
                    "earlier_registration_sha256": pair_geometry.earlier.registration_sha256,
                    "later_registration_sha256": pair_geometry.later.registration_sha256,
                },
                "qc": [
                    {
                        "code": "registration_assisted_pair",
                        "severity": "warning",
                        "message": (
                            "Both endpoints use a T0-fixed registration and one T0-mask-derived crop; "
                            "backward output is registration/crop-assisted reconstruction."
                        ),
                    }
                ],
                "coordinate_frame": COORDINATE_FRAME,
                "crop_policy": CROP_POLICY,
                "affine_lps": [list(value) for value in pair_geometry.earlier.affine_lps],
                "spacing_dhw": list(pair_geometry.earlier.spacing_dhw),
                "earlier_latent_path": str(latent_paths[earlier.visit_id]),
                "later_latent_path": str(latent_paths[later.visit_id]),
                "earlier_source_payload_sha256": payloads[earlier.visit_id].sha256,
                "later_source_payload_sha256": payloads[later.visit_id].sha256,
                "autoencoder_id": autoencoder_id,
            }
        )
    bound_pairs, statistics = bind_cached_pair_manifest_provenance(
        unsigned_pairs, base_statistics
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        staging_latents = staging / "visits"
        staging_latents.mkdir()
        visit_records: list[dict[str, Any]] = []
        for visit_id in selected_visit_ids:
            visit = visits_by_id[visit_id]
            source = payloads[visit_id]
            visit_geometry = geometry.visit_geometry_by_id[visit_id]
            affine = np.asarray(visit_geometry.affine_lps, dtype=np.float64)
            base_affine = np.asarray(visit_geometry.base_affine_lps, dtype=np.float64)
            spacing = np.asarray(visit_geometry.spacing_dhw, dtype=np.float32)
            normalized = (source.latent.float() - mean) / std
            if not torch.isfinite(normalized).all():
                raise FloatingPointError(f"normalized latent is non-finite for {visit_id!r}")
            provenance = {
                "schema": "ispy2_symmflow_mewm_continuous_latent_import_v1",
                "source_payload_path": _relative_source(source.path, source_root),
                "source_payload_sha256": source.sha256,
                "source_payload_schema": PAYLOAD_SCHEMA,
                "source_representation": STORED_REPRESENTATION,
                "stored_representation": "channel_zscore_float32_v1",
                "source_latent_statistics_sha256": statistics_source["sha256"],
                "source_data_contract_sha256": identity["data_contract_sha256"],
                "vqgan_sha256": autoencoder_id,
                "codebook_sha256": identity["codebook_sha256"],
                "coordinate_frame": COORDINATE_FRAME,
                "crop_policy": CROP_POLICY,
                "registration_status": visit.registration_status,
                "registration_assisted": True,
                "geometry_source": "upstream_registered_meta_dicom_absolute_lps",
                "header_affine_semantics": "physical_lps_affine_from_signed_upstream_metadata",
                "upstream_meta_sha256": visit_geometry.meta_sha256,
                "upstream_registration_sha256": visit_geometry.registration_sha256,
                "upstream_source_geometry_sha256": (
                    visit_geometry.source_geometry_sha256
                ),
                "upstream_registered_grid_geometry_sha256": (
                    visit_geometry.registered_grid_geometry_sha256
                ),
                "upstream_transform_artifacts": [
                    {"path": path, "sha256": digest}
                    for path, digest in visit_geometry.transform_artifacts
                ],
            }
            local_path = staging_latents / latent_paths[visit_id].name
            np.savez_compressed(
                local_path,
                latent=normalized.numpy(),
                visit_id=visit_id,
                patient_id=visit.patient_id,
                split=visit.split,
                autoencoder_id=autoencoder_id,
                latent_statistics_fingerprint=statistics[LATENT_STATISTICS_FINGERPRINT],
                cached_pair_manifest_fingerprint=statistics[
                    CACHED_PAIR_MANIFEST_FINGERPRINT
                ],
                cached_pair_manifest_record_count=statistics[
                    CACHED_PAIR_MANIFEST_RECORD_COUNT
                ],
                affine_lps=affine,
                base_affine_lps=base_affine,
                spacing_dhw=spacing,
                crop_start_zyx=np.asarray(
                    visit_geometry.crop_start_zyx, dtype=np.int64
                ),
                provenance_json=json.dumps(provenance, sort_keys=True),
                source_payload_sha256=source.sha256,
            )
            visit_records.append(
                {
                    "visit_id": visit_id,
                    "patient_id": visit.patient_id,
                    "visit_stage": visit.stage,
                    "split": visit.split,
                    "study_uid": visit.study_uid,
                    "latent_path": str(latent_paths[visit_id]),
                    "source_payload_path": str(source.path),
                    "source_payload_sha256": source.sha256,
                    "autoencoder_id": autoencoder_id,
                    "affine_lps": affine.tolist(),
                    "base_affine_lps": base_affine.tolist(),
                    "spacing_dhw": spacing.tolist(),
                    "crop_start_zyx": list(visit_geometry.crop_start_zyx),
                    "output_shape_zyx": list(visit_geometry.output_shape_zyx),
                    "meta_sha256": visit_geometry.meta_sha256,
                    "registration_sha256": visit_geometry.registration_sha256,
                    "transform_artifacts": [
                        {"path": path, "sha256": digest}
                        for path, digest in visit_geometry.transform_artifacts
                    ],
                    "source_geometry_sha256": visit_geometry.source_geometry_sha256,
                    "registered_grid_geometry_sha256": (
                        visit_geometry.registered_grid_geometry_sha256
                    ),
                    LATENT_STATISTICS_FINGERPRINT: statistics[
                        LATENT_STATISTICS_FINGERPRINT
                    ],
                    CACHED_PAIR_MANIFEST_FINGERPRINT: statistics[
                        CACHED_PAIR_MANIFEST_FINGERPRINT
                    ],
                    CACHED_PAIR_MANIFEST_RECORD_COUNT: statistics[
                        CACHED_PAIR_MANIFEST_RECORD_COUNT
                    ],
                    "latent_provenance": provenance,
                }
            )
        write_jsonl(staging / "visits.jsonl", visit_records)
        write_jsonl(staging / "pairs.jsonl", bound_pairs)
        (staging / "latent_statistics.json").write_text(
            json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        audit = {
            "schema": "ispy2_symmflow_mewm_continuous_latent_import_audit_v1",
            "time_pairs": [list(pair) for pair in configured_pairs],
            "pair_counts_by_interval": dict(
                sorted(Counter(row["transition_type"] for row in rows).items())
            ),
            "pair_count": len(bound_pairs),
            "visit_count": len(visit_records),
            "split_pair_counts": dict(sorted(split_counts.items())),
            "source_cache_identity": str(identity_path),
            "source_cache_identity_sha256": sha256_file(identity_path),
            "upstream_metadata_dir": str(metadata_root),
            "bundle_json_sha256": identity["bundle_json_sha256"],
            "bundle_contract_sha256": identity["bundle_contract_sha256"],
            "data_contract_sha256": identity["data_contract_sha256"],
            "vqgan_sha256": autoencoder_id,
            "codebook_sha256": identity["codebook_sha256"],
            "latent_statistics_sha256": statistics_source["sha256"],
            "coordinate_frame": COORDINATE_FRAME,
            "crop_policy": CROP_POLICY,
            "geometry_source": "upstream_registered_meta_dicom_absolute_lps",
            "spacing_dhw": list(geometry.spacing_dhw),
            "output_shape_zyx": list(geometry.output_shape_zyx),
            "bundle_artifact_sha256": dict(geometry.artifact_sha256),
            "registration_assisted": True,
            "backward_reconstruction_scope": "T0-registration-and-crop-assisted",
            "condition_fields": [
                "age",
                "hr_status(raw_0_or_1)",
                "her2_status(raw_0_or_1)",
                "mammaprint(raw_0_or_1)",
                "menopausal_status",
                "treatment_arm",
                "stage_i",
                "stage_j",
                "delta_days",
            ],
            "excluded_condition_fields": [
                "source_ftv_volume_cc",
                "target_ftv_volume_cc",
                "target_mask",
                "pCR",
            ],
        }
        (staging / "import_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    try:
        validate_cached_latent_provenance(bound_pairs, statistics)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return MewmLatentImportResult(
        pair_manifest_path=str(destination / "pairs.jsonl"),
        visit_manifest_path=str(destination / "visits.jsonl"),
        statistics_path=str(destination / "latent_statistics.json"),
        audit_path=str(destination / "import_audit.json"),
        pair_count=len(bound_pairs),
        visit_count=len(selected_visit_ids),
        split_pair_counts=dict(sorted(split_counts.items())),
        autoencoder_id=autoencoder_id,
    )


__all__ = ["MewmLatentImportResult", "import_mewm_continuous_latents"]
