"""Serializable schemas shared by data preparation and training.

The schemas deliberately keep an earlier visit and a later visit fixed. A
backward generation request changes the integration direction, not the field
meaning in a pair manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class QCEvent:
    code: str
    severity: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class Geometry:
    """DICOM geometry in LPS coordinates for an array ordered as D, H, W."""

    shape_dhw: tuple[int, int, int]
    spacing_dhw: tuple[float, float, float]
    orientation_lps: tuple[float, float, float, float, float, float]
    origin_lps: tuple[float, float, float]
    affine_lps: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
    source: str = "dicom_patient_geometry"

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape_dhw": list(self.shape_dhw),
            "spacing_dhw": list(self.spacing_dhw),
            "orientation_lps": list(self.orientation_lps),
            "origin_lps": list(self.origin_lps),
            "affine_lps": [list(row) for row in self.affine_lps],
            "source": self.source,
        }


@dataclass(frozen=True)
class PhaseRef:
    """Reference to one DCE phase without expanding every file into JSONL."""

    role: str
    series_uid: str
    series_path: str
    source_kind: str
    temporal_position: int | None
    acquisition_time: str | None
    instance_count: int
    reliability: str
    evidence: tuple[str, ...] = ()
    geometry: Geometry | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "series_uid": self.series_uid,
            "series_path": self.series_path,
            "source_kind": self.source_kind,
            "temporal_position": self.temporal_position,
            "acquisition_time": self.acquisition_time,
            "instance_count": self.instance_count,
            "reliability": self.reliability,
            "evidence": list(self.evidence),
        }
        result["geometry"] = self.geometry.to_dict() if self.geometry else None
        return result


@dataclass(frozen=True)
class MaskRef:
    series_uid: str
    series_path: str
    sop_class_uid: str
    mask_type: str
    semantic_status: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_uid": self.series_uid,
            "series_path": self.series_path,
            "sop_class_uid": self.sop_class_uid,
            "mask_type": self.mask_type,
            "semantic_status": self.semantic_status,
            "source": self.source,
        }


@dataclass(frozen=True)
class VisitRecord:
    visit_id: str
    patient_id: str
    collection: str
    study_uid: str
    visit_stage: str
    study_date: str | None
    date_source: str
    relative_date_verified: bool
    phase_paths: Mapping[str, PhaseRef]
    mask: MaskRef | None = None
    baseline_clinical: Mapping[str, Any] = field(default_factory=dict)
    treatment: Mapping[str, Any] = field(default_factory=dict)
    evaluation_metadata: Mapping[str, Any] = field(default_factory=dict)
    qc: tuple[QCEvent, ...] = ()
    series_count: int = 0
    split: str | None = None
    prepared_path: str | None = None
    schema_version: str = SCHEMA_VERSION

    @property
    def three_phase_ready(self) -> bool:
        required = {"pre", "early", "late"}
        return required.issubset(self.phase_paths) and not any(
            event.severity == "error" for event in self.qc
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "visit_id": self.visit_id,
            "patient_id": self.patient_id,
            "collection": self.collection,
            "study_uid": self.study_uid,
            "visit_stage": self.visit_stage,
            "study_date": self.study_date,
            "date_source": self.date_source,
            "relative_date_verified": self.relative_date_verified,
            "phase_paths": {
                role: ref.to_dict() for role, ref in sorted(self.phase_paths.items())
            },
            "mask": self.mask.to_dict() if self.mask else None,
            "baseline_clinical": dict(self.baseline_clinical),
            "treatment": dict(self.treatment),
            "evaluation_metadata": dict(self.evaluation_metadata),
            "qc": [event.to_dict() for event in self.qc],
            "series_count": self.series_count,
            "split": self.split,
            "prepared_path": self.prepared_path,
            "three_phase_ready": self.three_phase_ready,
        }


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    patient_id: str
    collection: str
    earlier_visit_id: str
    later_visit_id: str
    earlier_stage: str
    later_stage: str
    split: str
    delta_days: int | None
    observed_delta_days: int | None
    interval_missing: bool
    interval_source: str
    earlier_prepared_path: str | None = None
    later_prepared_path: str | None = None
    baseline_clinical: Mapping[str, Any] = field(default_factory=dict)
    treatment: Mapping[str, Any] = field(default_factory=dict)
    qc: tuple[QCEvent, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "patient_id": self.patient_id,
            "collection": self.collection,
            "earlier_visit_id": self.earlier_visit_id,
            "later_visit_id": self.later_visit_id,
            "earlier_stage": self.earlier_stage,
            "later_stage": self.later_stage,
            "split": self.split,
            "delta_days": self.delta_days,
            "observed_delta_days": self.observed_delta_days,
            "interval_missing": self.interval_missing,
            "interval_source": self.interval_source,
            "earlier_prepared_path": self.earlier_prepared_path,
            "later_prepared_path": self.later_prepared_path,
            "baseline_clinical": dict(self.baseline_clinical),
            "treatment": dict(self.treatment),
            "qc": [event.to_dict() for event in self.qc],
        }
