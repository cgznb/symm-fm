"""Import the longitudinal MU-Glioma four-modality latent cache.

Pair construction and cache validation follow the audited MeWM MU-Glioma
contract. This adapter deliberately excludes outcome fields and source-mask
conditions, then represents each visit as the ordered concatenation
``t1c,t1n,t2f,t2w`` for the local structured-condition SymmFlow.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ispy2_symmflow.training.datasets import write_jsonl
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    LATENT_STATISTICS_FINGERPRINT,
    bind_cached_pair_manifest_provenance,
    validate_cached_latent_provenance,
)
from ispy2_symmflow.utils.hashing import sha256_file, stable_hash


MU_CACHE_SCHEMA = "mewm_mu_glioma_post_continuous_latents_v1"
MU_PAYLOAD_SCHEMA = "mewm_mu_glioma_post_continuous_latent_payload_v1"
MU_CLINICAL_SCHEMA = "clarity_clinical_timeline_v1"
MU_NUMERIC_CONTRACT = "mu_glioma_post_brainiac_nonzero_zscore_pad_160x256x256_v1"
MU_NORMALIZATION = "continuous_codebook_minmax_v1"
MU_FLOW_NORMALIZATION_TRAIN_ZSCORE = "train_channel_zscore_v1"
MU_FLOW_NORMALIZATIONS = frozenset(
    {MU_NORMALIZATION, MU_FLOW_NORMALIZATION_TRAIN_ZSCORE}
)
MU_MODALITIES = ("t1c", "t1n", "t2f", "t2w")
MU_PER_MODALITY_LATENT_SHAPE = (8, 40, 64, 64)
MU_JOINT_LATENT_SHAPE = (32, 40, 64, 64)
MU_IMAGE_SHAPE = (160, 256, 256)
MU_EXPECTED_SAMPLE_SPLITS = {"train": 2148, "val": 236}
MU_EXPECTED_VISIT_SPLITS = {"train": 537, "val": 59}
MU_EXPECTED_PAIR_SPLITS = {"train": 770, "val": 76}
MU_TREATMENT_WINDOW = "source_inclusive_target_exclusive"
MU_COORDINATE_FRAME = "brainiac_padded_standardized_index_space_v1"

_CACHE_FIELDS = frozenset(
    {
        "codebook_max",
        "codebook_min",
        "codebook_sha256",
        "data_contract_sha256",
        "input_shape_zyx",
        "latent_dtype",
        "latent_shape_czyx",
        "manifest_sha256",
        "normalization",
        "numeric_contract",
        "sample_count",
        "schema",
        "split_counts",
        "vqgan_checkpoint",
        "vqgan_sha256",
    }
)
_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "sample_id",
        "patient_id",
        "timepoint",
        "modality",
        "split",
        "continuous_latent",
        "vqgan_sha256",
        "codebook_sha256",
        "data_contract_sha256",
        "manifest_sha256",
        "numeric_contract",
        "normalization",
    }
)
_PATIENT_PATTERN = re.compile(r"^PatientID_[0-9]{4}$")
_TIMEPOINT_PATTERN = re.compile(r"^Timepoint_([1-9][0-9]*)$")
_CLINICAL_TIMEPOINT_PATTERN = re.compile(r"^T([1-9][0-9]*)$")
_MISSING_TEXT = frozenset({"", "na", "n/a", "nan", "none", "null", "unknown"})
_GENOMIC_MARKERS = (
    "1p19q",
    "atrx",
    "braf_v600e",
    "cdkn2ab_deletion",
    "chr7_gain_chr10_loss",
    "egfr_amplification",
    "h3_3a",
    "idh1",
    "idh2",
    "mgmt",
    "pten",
    "tert_promoter",
    "tp53",
)
_TREATMENT_CATEGORIES = (
    "brachytherapy",
    "chemotherapy",
    "immunotherapy",
    "other_therapy",
    "radiotherapy",
)


@dataclass(frozen=True)
class MUGliomaLatentImportResult:
    pair_manifest_path: str
    visit_manifest_path: str
    statistics_path: str
    audit_path: str
    pair_count: int
    visit_count: int
    split_pair_counts: Mapping[str, int]
    autoencoder_id: str


@dataclass(frozen=True)
class _Sample:
    sample_id: str
    patient_id: str
    timepoint: str
    modality: str
    split: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class _Visit:
    visit_id: str
    patient_id: str
    timepoint: str
    stage: str
    split: str
    samples: tuple[_Sample, ...]


@dataclass(frozen=True)
class _Pair:
    pair_id: str
    patient_id: str
    split: str
    earlier: _Visit
    later: _Visit
    source_timeline_index: int
    target_timeline_index: int
    source_mri_day: int
    target_mri_day: int
    baseline_clinical: Mapping[str, Any]
    treatment: Mapping[str, Any]

    @property
    def delta_days(self) -> int:
        return self.target_mri_day - self.source_mri_day


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact mapping")
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON: {path}") from error
    return _mapping(value, label=label)


def _day(value: Any, *, label: str) -> int:
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value) and value.is_integer():
        return int(value)
    raise ValueError(f"{label} must be an integer day")


def _clean_category(value: Any) -> str | None:
    if value is None or type(value) is float and not math.isfinite(value):
        return None
    if isinstance(value, str):
        result = value.strip()
        return None if result.lower() in _MISSING_TEXT else result
    if type(value) in (bool, int, float):
        return str(value)
    raise ValueError("MU-Glioma condition values must be scalar")


def _genomic_category(value: Any) -> str | None:
    if value is False or type(value) is int and value == 0:
        return "negative_or_wild_type"
    if value is True or type(value) is int and value == 1:
        return "positive_or_altered"
    if value is None or type(value) is int and value == 2:
        return None
    return _clean_category(value)


def _baseline_conditions(context_value: Any) -> dict[str, Any]:
    context = _mapping(context_value or {}, label="clinical context_static")
    raw_age = context.get("age_at_diagnosis_years")
    age: float | None
    try:
        age = None if raw_age is None else float(raw_age)
    except (TypeError, ValueError) as error:
        raise ValueError("MU-Glioma age_at_diagnosis_years is invalid") from error
    if age is not None and not math.isfinite(age):
        age = None
    result: dict[str, Any] = {
        "age_at_diagnosis_years": age,
        "sex_at_birth": _clean_category(context.get("sex_at_birth")),
        "race": _clean_category(context.get("race")),
        "primary_diagnosis": _clean_category(context.get("primary_diagnosis")),
        "who_grade": _clean_category(context.get("who_grade")),
        "previous_brain_tumor": _clean_category(
            context.get("previous_brain_tumor")
        ),
        "stereotactic_biopsy_before_resection": _clean_category(
            context.get("stereotactic_biopsy_before_resection")
        ),
    }
    genomics = _mapping(context.get("genomics") or {}, label="clinical genomics")
    result.update(
        {
            f"genomic_{marker}": _genomic_category(genomics.get(marker))
            for marker in _GENOMIC_MARKERS
        }
    )
    return result


def _timeline_record(value: Any, *, index: int) -> dict[str, Any]:
    record = _mapping(value, label=f"timeline[{index}]")
    if _CLINICAL_TIMEPOINT_PATTERN.fullmatch(str(record.get("tp_id", ""))) is None:
        raise ValueError(f"timeline[{index}].tp_id is invalid")
    _day(record.get("mri_day"), label=f"timeline[{index}].mri_day")
    _mapping(record.get("actions"), label=f"timeline[{index}].actions")
    return record


def _action_bounds(action: Mapping[str, Any], *, label: str) -> tuple[int, int] | None:
    raw_start = action.get("start_day")
    raw_end = action.get("end_day")
    if raw_start is None and raw_end is None:
        point = action.get("day_of_insertion")
        if point is not None:
            raw_start = raw_end = point
        else:
            raw_start = action.get("interval_start_day")
            raw_end = action.get("interval_end_day")
    if raw_start is None and raw_end is None:
        return None
    raw_start = raw_end if raw_start is None else raw_start
    raw_end = raw_start if raw_end is None else raw_end
    start = _day(raw_start, label=f"{label}.start_day")
    end = _day(raw_end, label=f"{label}.end_day")
    if end < start:
        raise ValueError(f"{label} has a reversed interval")
    return start, end


def _treatment_conditions(
    timeline: Sequence[Mapping[str, Any]], source_index: int, target_index: int
) -> dict[str, str]:
    source_day = _day(timeline[source_index]["mri_day"], label="source mri_day")
    target_day = _day(timeline[target_index]["mri_day"], label="target mri_day")
    observed = {category: False for category in _TREATMENT_CATEGORIES}
    for timeline_index in range(source_index, target_index + 1):
        actions = _mapping(
            timeline[timeline_index]["actions"],
            label=f"timeline[{timeline_index}].actions",
        )
        unknown = set(actions).difference(_TREATMENT_CATEGORIES)
        if unknown:
            raise ValueError(
                "MU-Glioma timeline has unsupported treatment categories: "
                + ", ".join(sorted(unknown))
            )
        for category, items in actions.items():
            if type(items) is not list:
                raise ValueError(f"treatment category {category!r} must contain a list")
            for action_index, value in enumerate(items):
                action = _mapping(
                    value,
                    label=f"timeline[{timeline_index}].actions.{category}[{action_index}]",
                )
                bounds = _action_bounds(
                    action,
                    label=f"timeline[{timeline_index}].actions.{category}[{action_index}]",
                )
                if bounds is None:
                    continue
                action_start, action_end = bounds
                if action_end >= source_day and action_start < target_day:
                    observed[category] = True
    return {
        f"{category}_received": "yes" if present else "no"
        for category, present in observed.items()
    }


def _validate_cache_identity(path: Path) -> dict[str, Any]:
    identity = _read_json(path, label="MU-Glioma latent cache identity")
    if set(identity) != _CACHE_FIELDS:
        raise ValueError("MU-Glioma latent cache identity fields are invalid")
    expected = {
        "schema": MU_CACHE_SCHEMA,
        "sample_count": sum(MU_EXPECTED_SAMPLE_SPLITS.values()),
        "split_counts": MU_EXPECTED_SAMPLE_SPLITS,
        "input_shape_zyx": list(MU_IMAGE_SHAPE),
        "latent_shape_czyx": list(MU_PER_MODALITY_LATENT_SHAPE),
        "latent_dtype": "float16",
        "numeric_contract": MU_NUMERIC_CONTRACT,
        "normalization": MU_NORMALIZATION,
    }
    mismatches = {
        key: {"expected": value, "observed": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatches:
        raise ValueError(f"MU-Glioma latent cache contract mismatch: {mismatches}")
    minimum = float(identity["codebook_min"])
    maximum = float(identity["codebook_max"])
    if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= minimum:
        raise ValueError("MU-Glioma codebook range is invalid")
    for key in (
        "vqgan_sha256",
        "codebook_sha256",
        "data_contract_sha256",
        "manifest_sha256",
    ):
        value = identity[key]
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"MU-Glioma cache {key} is invalid")
    return identity


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    except (OSError, RuntimeError, TypeError, ValueError, EOFError) as error:
        raise ValueError(f"MU-Glioma latent payload is unreadable: {path}") from error
    if type(payload) is not dict or set(payload) != _PAYLOAD_FIELDS:
        raise ValueError(f"MU-Glioma latent payload fields are invalid: {path}")
    return payload


def _sample_from_payload(
    path: Path, payload: Mapping[str, Any], identity: Mapping[str, Any]
) -> _Sample:
    split = str(payload.get("split", ""))
    sample_id = str(payload.get("sample_id", ""))
    patient_id = str(payload.get("patient_id", ""))
    timepoint = str(payload.get("timepoint", ""))
    modality = str(payload.get("modality", ""))
    expected_id = f"{patient_id}__{timepoint}__{modality}"
    expected = {
        "schema": MU_PAYLOAD_SCHEMA,
        "sample_id": expected_id,
        "vqgan_sha256": identity["vqgan_sha256"],
        "codebook_sha256": identity["codebook_sha256"],
        "data_contract_sha256": identity["data_contract_sha256"],
        "manifest_sha256": identity["manifest_sha256"],
        "numeric_contract": MU_NUMERIC_CONTRACT,
        "normalization": MU_NORMALIZATION,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"MU-Glioma latent payload identity mismatch: {path}")
    if (
        _PATIENT_PATTERN.fullmatch(patient_id) is None
        or _TIMEPOINT_PATTERN.fullmatch(timepoint) is None
        or modality not in MU_MODALITIES
        or split not in MU_EXPECTED_SAMPLE_SPLITS
        or path.parent.name != split
        or path.name != f"{sample_id}.pt"
    ):
        raise ValueError(f"MU-Glioma latent payload path/identity is invalid: {path}")
    latent = payload.get("continuous_latent")
    if (
        not isinstance(latent, Tensor)
        or latent.dtype != torch.float16
        or tuple(latent.shape) != MU_PER_MODALITY_LATENT_SHAPE
        or not bool(torch.isfinite(latent).all())
    ):
        raise ValueError(f"MU-Glioma continuous latent tensor is invalid: {path}")
    return _Sample(
        sample_id=sample_id,
        patient_id=patient_id,
        timepoint=timepoint,
        modality=modality,
        split=split,
        path=path,
        sha256=sha256_file(path),
    )


def _inventory_samples(
    source_root: Path, identity: Mapping[str, Any]
) -> tuple[tuple[_Sample, ...], str]:
    volumes = source_root / "volumes"
    if not volumes.is_dir() or volumes.is_symlink():
        raise ValueError("MU-Glioma cache volumes directory is missing or unsafe")
    paths = sorted(path for path in volumes.rglob("*") if path.is_file() or path.is_symlink())
    if any(path.is_symlink() or path.suffix != ".pt" for path in paths):
        raise ValueError("MU-Glioma cache contains a symlink or unexpected payload file")
    samples: list[_Sample] = []
    digests: list[dict[str, str]] = []
    for path in paths:
        payload = _load_payload(path)
        sample = _sample_from_payload(path, payload, identity)
        samples.append(sample)
        digests.append(
            {
                "path": path.relative_to(source_root).as_posix(),
                "sha256": sample.sha256,
            }
        )
        del payload
    if len(samples) != int(identity["sample_count"]):
        raise ValueError("MU-Glioma cache payload count differs from cache identity")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("MU-Glioma cache contains duplicate sample IDs")
    split_counts = Counter(sample.split for sample in samples)
    if dict(split_counts) != MU_EXPECTED_SAMPLE_SPLITS:
        raise ValueError(f"MU-Glioma sample split counts changed: {dict(split_counts)}")
    return tuple(samples), stable_hash(digests)


def _group_visits(samples: Sequence[_Sample]) -> dict[tuple[str, str], _Visit]:
    grouped: dict[tuple[str, str], list[_Sample]] = {}
    for sample in samples:
        grouped.setdefault((sample.patient_id, sample.timepoint), []).append(sample)
    visits: dict[tuple[str, str], _Visit] = {}
    for key, values in grouped.items():
        by_modality = {sample.modality: sample for sample in values}
        identities = {(sample.patient_id, sample.timepoint, sample.split) for sample in values}
        if (
            len(values) != len(MU_MODALITIES)
            or set(by_modality) != set(MU_MODALITIES)
            or len(identities) != 1
        ):
            raise ValueError(f"MU-Glioma visit is incomplete or inconsistent: {key}")
        match = _TIMEPOINT_PATTERN.fullmatch(key[1])
        assert match is not None
        visits[key] = _Visit(
            visit_id=f"{key[0]}__{key[1]}",
            patient_id=key[0],
            timepoint=key[1],
            stage=f"T{int(match.group(1))}",
            split=values[0].split,
            samples=tuple(by_modality[name] for name in MU_MODALITIES),
        )
    split_counts = Counter(visit.split for visit in visits.values())
    if dict(split_counts) != MU_EXPECTED_VISIT_SPLITS:
        raise ValueError(f"MU-Glioma visit split counts changed: {dict(split_counts)}")
    patient_splits: dict[str, str] = {}
    for visit in visits.values():
        previous = patient_splits.setdefault(visit.patient_id, visit.split)
        if previous != visit.split:
            raise ValueError(f"MU-Glioma patient crosses splits: {visit.patient_id}")
    return visits


def _build_pairs(
    visits: Mapping[tuple[str, str], _Visit], clinical: Mapping[str, Any]
) -> tuple[tuple[_Pair, ...], dict[str, Any]]:
    if clinical.get("schema_version") != MU_CLINICAL_SCHEMA:
        raise ValueError("MU-Glioma clinical timeline schema is unsupported")
    patients = _mapping(clinical.get("patients"), label="clinical patients")
    mri_patients = sorted({patient_id for patient_id, _ in visits})
    pairs: list[_Pair] = []
    missing_patients: list[str] = []
    missing_mri_visits: list[str] = []
    nonpositive_intervals: list[str] = []
    timeline_without_mri: list[str] = []
    for patient_id in mri_patients:
        patient_visit_keys = {
            timepoint for pid, timepoint in visits if pid == patient_id
        }
        patient_value = patients.get(patient_id)
        if patient_value is None:
            missing_patients.append(patient_id)
            missing_mri_visits.extend(
                f"{patient_id}/{timepoint}" for timepoint in patient_visit_keys
            )
            continue
        patient = _mapping(patient_value, label=f"clinical patient {patient_id}")
        raw_timeline = patient.get("timeline")
        if type(raw_timeline) is not list:
            raise ValueError(f"clinical patient {patient_id} has no valid timeline")
        timeline = tuple(
            sorted(
                (
                    _timeline_record(value, index=index)
                    for index, value in enumerate(raw_timeline)
                ),
                key=lambda value: int(
                    _CLINICAL_TIMEPOINT_PATTERN.fullmatch(value["tp_id"]).group(1)
                ),
            )
        )
        timepoints = [
            f"Timepoint_{int(_CLINICAL_TIMEPOINT_PATTERN.fullmatch(row['tp_id']).group(1))}"
            for row in timeline
        ]
        if len(timepoints) != len(set(timepoints)):
            raise ValueError(f"clinical patient {patient_id} has duplicate timepoints")
        missing_mri_visits.extend(
            f"{patient_id}/{timepoint}"
            for timepoint in patient_visit_keys.difference(timepoints)
        )
        baseline = _baseline_conditions(patient.get("context_static"))
        for timepoint in timepoints:
            if (patient_id, timepoint) not in visits:
                timeline_without_mri.append(f"{patient_id}/{timepoint}")
        for source_index in range(len(timeline) - 1):
            for target_index in range(source_index + 1, len(timeline)):
                source_timepoint = timepoints[source_index]
                target_timepoint = timepoints[target_index]
                earlier = visits.get((patient_id, source_timepoint))
                later = visits.get((patient_id, target_timepoint))
                if earlier is None or later is None:
                    missing_mri_visits.extend(
                        f"{patient_id}/{value}"
                        for value, visit in (
                            (source_timepoint, earlier),
                            (target_timepoint, later),
                        )
                        if visit is None
                    )
                    continue
                source_day = _day(
                    timeline[source_index]["mri_day"], label="source mri_day"
                )
                target_day = _day(
                    timeline[target_index]["mri_day"], label="target mri_day"
                )
                if target_day <= source_day:
                    nonpositive_intervals.append(
                        f"{patient_id}/{source_timepoint}/{target_timepoint}"
                    )
                    continue
                if earlier.split != later.split:
                    raise ValueError(f"MU-Glioma pair crosses splits: {patient_id}")
                pairs.append(
                    _Pair(
                        pair_id=f"{patient_id}__{source_timepoint}__{target_timepoint}",
                        patient_id=patient_id,
                        split=earlier.split,
                        earlier=earlier,
                        later=later,
                        source_timeline_index=source_index,
                        target_timeline_index=target_index,
                        source_mri_day=source_day,
                        target_mri_day=target_day,
                        baseline_clinical=baseline,
                        treatment=_treatment_conditions(
                            timeline, source_index, target_index
                        ),
                    )
                )
    split_counts = Counter(pair.split for pair in pairs)
    if dict(split_counts) != MU_EXPECTED_PAIR_SPLITS:
        raise ValueError(f"MU-Glioma pair split counts changed: {dict(split_counts)}")
    audit = {
        "clinical_missing_patients": sorted(missing_patients),
        "clinical_missing_mri_visits": sorted(set(missing_mri_visits)),
        "timeline_without_mri_visits": sorted(set(timeline_without_mri)),
        "nonpositive_intervals": sorted(nonpositive_intervals),
    }
    return tuple(pairs), audit


def _load_visit_raw_latent(
    visit: _Visit, identity: Mapping[str, Any]
) -> tuple[Tensor, list[dict[str, str]]]:
    values: list[Tensor] = []
    sources: list[dict[str, str]] = []
    for sample in visit.samples:
        payload = _load_payload(sample.path)
        validated = _sample_from_payload(sample.path, payload, identity)
        if validated != sample:
            raise ValueError(f"MU-Glioma payload changed during import: {sample.path}")
        values.append(payload["continuous_latent"].float())
        sources.append(
            {
                "modality": sample.modality,
                "path": sample.path.name,
                "sha256": sample.sha256,
            }
        )
    raw = torch.cat(values, dim=0)
    if tuple(raw.shape) != MU_JOINT_LATENT_SHAPE:
        raise RuntimeError("MU-Glioma joint latent shape changed")
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError(f"MU-Glioma latent is non-finite: {visit.visit_id}")
    return raw.contiguous(), sources


def _fit_train_channel_statistics(
    visits: Sequence[_Visit], identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Fit population moments on unique paired training visits only."""

    training_visits = tuple(visit for visit in visits if visit.split == "train")
    if not training_visits:
        raise ValueError("MU-Glioma z-score fitting requires training visits")
    channels = MU_JOINT_LATENT_SHAPE[0]
    total = torch.zeros(channels, dtype=torch.float64)
    squared_total = torch.zeros(channels, dtype=torch.float64)
    minimum = torch.full((channels,), float("inf"), dtype=torch.float64)
    maximum = torch.full((channels,), float("-inf"), dtype=torch.float64)
    voxel_count = 0
    for visit in training_visits:
        raw, _ = _load_visit_raw_latent(visit, identity)
        value = raw.to(torch.float64)
        flattened = value.flatten(start_dim=1)
        total += flattened.sum(dim=1)
        squared_total += flattened.square().sum(dim=1)
        minimum = torch.minimum(minimum, flattened.amin(dim=1))
        maximum = torch.maximum(maximum, flattened.amax(dim=1))
        voxel_count += math.prod(raw.shape[1:])
        del raw, value, flattened
    mean = total / voxel_count
    variance = squared_total / voxel_count - mean.square()
    std = variance.clamp_min(0.0).sqrt()
    if (
        not bool(torch.isfinite(mean).all())
        or not bool(torch.isfinite(std).all())
        or bool(torch.any(std <= 0))
    ):
        raise FloatingPointError("MU-Glioma training latent statistics are invalid")
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fit_split": "train",
        "statistics_kind": "per_channel_population_moments_unique_train_pair_visits",
        "fit_visit_count": len(training_visits),
        "fit_patient_count": len({visit.patient_id for visit in training_visits}),
        "fit_voxels_per_channel": voxel_count,
        "fit_channel_min": minimum.tolist(),
        "fit_channel_max": maximum.tolist(),
    }


def _normalize_latent(
    raw: Tensor, *, mean: Sequence[float], std: Sequence[float], visit_id: str
) -> Tensor:
    channels = raw.shape[0]
    if len(mean) != channels or len(std) != channels:
        raise ValueError("MU-Glioma latent statistics do not match the channel count")
    center = torch.as_tensor(mean, dtype=raw.dtype).reshape(channels, 1, 1, 1)
    scale = torch.as_tensor(std, dtype=raw.dtype).reshape(channels, 1, 1, 1)
    if not bool(torch.isfinite(scale).all()) or bool(torch.any(scale <= 0)):
        raise ValueError("MU-Glioma latent standard deviations must be finite and positive")
    normalized = (raw - center) / scale
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError(f"normalized MU-Glioma latent is non-finite: {visit_id}")
    return normalized.contiguous()


def import_mu_glioma_continuous_latents(
    cache_identity_path: str | Path,
    source_latent_dir: str | Path,
    clinical_timeline: str | Path,
    vqgan_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    flow_normalization: str = MU_NORMALIZATION,
) -> MUGliomaLatentImportResult:
    """Validate and import all valid MU-Glioma longitudinal pairs."""

    identity_path = Path(cache_identity_path).expanduser().resolve()
    source_root = Path(source_latent_dir).expanduser().resolve()
    clinical_path = Path(clinical_timeline).expanduser().resolve()
    checkpoint_path = Path(vqgan_checkpoint).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    normalization = str(flow_normalization).strip().lower()
    if normalization not in MU_FLOW_NORMALIZATIONS:
        raise ValueError(
            "MU-Glioma flow_normalization must be one of "
            f"{sorted(MU_FLOW_NORMALIZATIONS)}"
        )
    if destination.exists():
        raise FileExistsError(f"MU-Glioma import output already exists: {destination}")
    if identity_path.parent != source_root:
        raise ValueError("MU-Glioma cache identity must be directly inside source_latent_dir")
    if not clinical_path.is_file() or clinical_path.is_symlink():
        raise ValueError("MU-Glioma clinical timeline is missing or unsafe")
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise ValueError("MU-Glioma VQ-GAN checkpoint is missing or unsafe")

    identity = _validate_cache_identity(identity_path)
    autoencoder_id = sha256_file(checkpoint_path)
    if autoencoder_id != identity["vqgan_sha256"]:
        raise ValueError("MU-Glioma VQ-GAN SHA-256 differs from the latent cache")
    clinical = _read_json(clinical_path, label="MU-Glioma clinical timeline")
    clinical_sha256 = sha256_file(clinical_path)
    samples, payload_inventory_sha256 = _inventory_samples(source_root, identity)
    visits = _group_visits(samples)
    pairs, pair_audit = _build_pairs(visits, clinical)
    selected_visits = {
        visit.visit_id: visit
        for pair in pairs
        for visit in (pair.earlier, pair.later)
    }
    selected = tuple(sorted(selected_visits.values(), key=lambda value: value.visit_id))
    final_latent_dir = destination / "visits"
    latent_paths = {
        visit.visit_id: final_latent_dir / f"{visit.visit_id}.npz" for visit in selected
    }
    midpoint = (float(identity["codebook_min"]) + float(identity["codebook_max"])) * 0.5
    half_range = (float(identity["codebook_max"]) - float(identity["codebook_min"])) * 0.5
    patient_splits = {visit.patient_id: visit.split for visit in visits.values()}
    source_signature = {
        "kind": "mewm_mu_glioma_external_latent_cache",
        "coordinate_frame": MU_COORDINATE_FRAME,
        "image_shape_czyx": [len(MU_MODALITIES), *MU_IMAGE_SHAPE],
        "image_channels": list(MU_MODALITIES),
        "latent_shape_czyx": list(MU_JOINT_LATENT_SHAPE),
        "longitudinal_registration_verified": False,
        "geometry_semantics": "shared_preprocessed_index_shape_not_physical_registration",
        "numeric_contract": identity["numeric_contract"],
        "data_contract_sha256": identity["data_contract_sha256"],
    }
    if normalization == MU_FLOW_NORMALIZATION_TRAIN_ZSCORE:
        fitted_statistics = _fit_train_channel_statistics(selected, identity)
        normalization_application = (
            "(continuous-train_channel_mean)/train_channel_population_std"
        )
    else:
        fitted_statistics = {
            "mean": [midpoint] * MU_JOINT_LATENT_SHAPE[0],
            "std": [half_range] * MU_JOINT_LATENT_SHAPE[0],
            "fit_split": None,
            "statistics_kind": "fixed_codebook_minmax_affine_not_data_fitted",
        }
        normalization_application = (
            "2*(continuous-codebook_min)/(codebook_max-codebook_min)-1"
        )
    base_statistics = {
        **fitted_statistics,
        "autoencoder_id": autoencoder_id,
        "source_preprocessing_signature": source_signature,
        "source_manifest": str(identity_path),
        "source_split_hash": stable_hash(patient_splits),
        "source_manifest_record_count": len(samples),
        "source_cache_identity_sha256": sha256_file(identity_path),
        "source_payload_inventory_sha256": payload_inventory_sha256,
        "source_clinical_timeline_sha256": clinical_sha256,
        "source_data_contract_sha256": identity["data_contract_sha256"],
        "source_manifest_sha256": identity["manifest_sha256"],
        "source_codebook_sha256": identity["codebook_sha256"],
        "source_payload_schema": MU_PAYLOAD_SCHEMA,
        "normalization": normalization,
        "source_payload_normalization": MU_NORMALIZATION,
        "normalization_application": normalization_application,
        "normalization_clamp": False,
        "modality_order": list(MU_MODALITIES),
        "decoder_contract": "split_modalities_then_denormalize_quantize_shared_vqgan_decode",
    }

    warning = {
        "code": "longitudinal_registration_unverified",
        "severity": "warning",
        "message": (
            "Endpoints share the BRAINiac padded index shape, but explicit cross-visit "
            "physical registration was not established."
        ),
    }
    unsigned_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        unsigned_pairs.append(
            {
                "schema_version": "mewm-mu-glioma-continuous-latent-import-1.0",
                "pair_id": pair.pair_id,
                "patient_id": pair.patient_id,
                "collection": "MU_Glioma_Post",
                "earlier_visit_id": pair.earlier.visit_id,
                "later_visit_id": pair.later.visit_id,
                "earlier_stage": pair.earlier.stage,
                "later_stage": pair.later.stage,
                "split": pair.split,
                "delta_days": pair.delta_days,
                "observed_delta_days": pair.delta_days,
                "interval_missing": False,
                "interval_source": f"clarity_timeline:{MU_TREATMENT_WINDOW}",
                "baseline_clinical": dict(pair.baseline_clinical),
                "treatment": dict(pair.treatment),
                "evaluation_metadata": {
                    "source_mri_day": pair.source_mri_day,
                    "target_mri_day": pair.target_mri_day,
                    "source_timeline_index": pair.source_timeline_index,
                    "target_timeline_index": pair.target_timeline_index,
                    "modality_order": list(MU_MODALITIES),
                    "target_outcomes_are_conditions": False,
                },
                "qc": [warning],
                "coordinate_frame": MU_COORDINATE_FRAME,
                "affine_lps": np.eye(4, dtype=np.float64).tolist(),
                "spacing_dhw": [1.0, 1.0, 1.0],
                "earlier_latent_path": str(latent_paths[pair.earlier.visit_id]),
                "later_latent_path": str(latent_paths[pair.later.visit_id]),
                "earlier_source_visit_sha256": stable_hash(
                    [sample.sha256 for sample in pair.earlier.samples]
                ),
                "later_source_visit_sha256": stable_hash(
                    [sample.sha256 for sample in pair.later.samples]
                ),
                "autoencoder_id": autoencoder_id,
            }
        )
    bound_pairs, statistics = bind_cached_pair_manifest_provenance(
        unsigned_pairs, base_statistics
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        staging_latents = staging / "visits"
        staging_latents.mkdir()
        visit_records: list[dict[str, Any]] = []
        for visit in selected:
            raw, source_payloads = _load_visit_raw_latent(visit, identity)
            normalized = _normalize_latent(
                raw,
                mean=statistics["mean"],
                std=statistics["std"],
                visit_id=visit.visit_id,
            )
            provenance = {
                "schema": "ispy2_symmflow_mewm_mu_glioma_joint4_import_v1",
                "source_payloads": source_payloads,
                "source_payload_inventory_sha256": payload_inventory_sha256,
                "source_data_contract_sha256": identity["data_contract_sha256"],
                "source_manifest_sha256": identity["manifest_sha256"],
                "vqgan_sha256": autoencoder_id,
                "codebook_sha256": identity["codebook_sha256"],
                "modality_order": list(MU_MODALITIES),
                "coordinate_frame": MU_COORDINATE_FRAME,
                "longitudinal_registration_verified": False,
                "normalization": normalization,
                "source_payload_normalization": MU_NORMALIZATION,
                "normalization_clamp": False,
            }
            local_path = staging_latents / latent_paths[visit.visit_id].name
            np.savez(
                local_path,
                latent=normalized.numpy(),
                visit_id=visit.visit_id,
                patient_id=visit.patient_id,
                split=visit.split,
                autoencoder_id=autoencoder_id,
                latent_statistics_fingerprint=statistics[
                    LATENT_STATISTICS_FINGERPRINT
                ],
                cached_pair_manifest_fingerprint=statistics[
                    CACHED_PAIR_MANIFEST_FINGERPRINT
                ],
                cached_pair_manifest_record_count=statistics[
                    CACHED_PAIR_MANIFEST_RECORD_COUNT
                ],
                affine_lps=np.eye(4, dtype=np.float64),
                spacing_dhw=np.ones(3, dtype=np.float32),
                provenance_json=json.dumps(provenance, sort_keys=True),
            )
            visit_records.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": visit.patient_id,
                    "visit_stage": visit.stage,
                    "split": visit.split,
                    "latent_path": str(latent_paths[visit.visit_id]),
                    "source_payloads": source_payloads,
                    "autoencoder_id": autoencoder_id,
                    "modality_order": list(MU_MODALITIES),
                    "latent_shape_czyx": list(MU_JOINT_LATENT_SHAPE),
                    "coordinate_frame": MU_COORDINATE_FRAME,
                    "longitudinal_registration_verified": False,
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
            json.dumps(statistics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        interval_counts = Counter(
            f"{pair.earlier.stage}->{pair.later.stage}" for pair in pairs
        )
        audit = {
            "schema": "ispy2_symmflow_mewm_mu_glioma_import_audit_v1",
            "pair_count": len(pairs),
            "visit_count": len(selected),
            "cache_visit_count": len(visits),
            "cache_patient_count": len({visit.patient_id for visit in visits.values()}),
            "paired_patient_count": len({pair.patient_id for pair in pairs}),
            "split_pair_counts": dict(sorted(Counter(pair.split for pair in pairs).items())),
            "pair_counts_by_interval": dict(sorted(interval_counts.items())),
            "source_cache_identity": str(identity_path),
            "source_cache_identity_sha256": sha256_file(identity_path),
            "source_payload_inventory_sha256": payload_inventory_sha256,
            "source_clinical_timeline_sha256": clinical_sha256,
            "vqgan_sha256": autoencoder_id,
            "codebook_sha256": identity["codebook_sha256"],
            "data_contract_sha256": identity["data_contract_sha256"],
            "modality_order": list(MU_MODALITIES),
            "latent_shape_czyx": list(MU_JOINT_LATENT_SHAPE),
            "normalization": normalization,
            "source_payload_normalization": MU_NORMALIZATION,
            "normalization_clamp": False,
            "treatment_window": MU_TREATMENT_WINDOW,
            "coordinate_frame": MU_COORDINATE_FRAME,
            "longitudinal_registration_verified": False,
            "condition_fields": sorted(
                set(pairs[0].baseline_clinical)
                | set(pairs[0].treatment)
                | {"stage_i", "stage_j", "delta_days", "interval_missing", "interval_source"}
            ),
            "excluded_condition_fields": [
                "patient_id",
                "survival",
                "progression",
                "hospice",
                "target_image",
                "target_mask",
                "source_mask",
            ],
            **pair_audit,
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
    split_pair_counts = dict(sorted(Counter(pair.split for pair in pairs).items()))
    return MUGliomaLatentImportResult(
        pair_manifest_path=str(destination / "pairs.jsonl"),
        visit_manifest_path=str(destination / "visits.jsonl"),
        statistics_path=str(destination / "latent_statistics.json"),
        audit_path=str(destination / "import_audit.json"),
        pair_count=len(pairs),
        visit_count=len(selected),
        split_pair_counts=split_pair_counts,
        autoencoder_id=autoencoder_id,
    )


__all__ = [
    "MU_COORDINATE_FRAME",
    "MU_EXPECTED_PAIR_SPLITS",
    "MU_IMAGE_SHAPE",
    "MU_JOINT_LATENT_SHAPE",
    "MU_MODALITIES",
    "MU_FLOW_NORMALIZATIONS",
    "MU_FLOW_NORMALIZATION_TRAIN_ZSCORE",
    "MU_NORMALIZATION",
    "MUGliomaLatentImportResult",
    "import_mu_glioma_continuous_latents",
]
