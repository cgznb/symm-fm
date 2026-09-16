"""Build visit and fixed-semantics longitudinal pair manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .clinical import ClinicalRecord, load_clinical_records
from .dicom import (
    SEGMENTATION_STORAGE_UID,
    SeriesSummary,
    iter_series_directories,
    read_series_headers,
    summarize_series,
)
from .phases import choose_dce_candidate, identify_temporal_dce_phases
from .schema import Geometry, MaskRef, PairRecord, PhaseRef, QCEvent, VisitRecord


VISIT_PATTERN = re.compile(r"(?:^|[^A-Z0-9])T([0-3])(?:$|[^A-Z0-9])", re.IGNORECASE)
COMPACT_VISIT_PATTERN = re.compile(r"ISPY2MRI\s*T([0-3])", re.IGNORECASE)


def parse_visit_stage(study_description: str) -> str | None:
    compact = COMPACT_VISIT_PATTERN.search(study_description)
    match = compact or VISIT_PATTERN.search(study_description)
    return f"T{match.group(1)}" if match else None


def parse_dicom_date(value: str | None) -> date | None:
    if not value:
        return None
    for pattern in ("%Y%m%d", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            pass
    return None


def index_dicom_series(data_root: str | Path) -> list[SeriesSummary]:
    summaries: list[SeriesSummary] = []
    for collection, patient, study, path in iter_series_directories(data_root):
        summaries.append(summarize_series(collection, patient, study, path))
    return summaries


def _mask_from_series(series: list[SeriesSummary]) -> tuple[MaskRef | None, list[QCEvent]]:
    candidates = [
        item
        for item in series
        if item.header.sop_class_uid == SEGMENTATION_STORAGE_UID
        or item.header.modality == "SEG"
    ]
    if not candidates:
        return None, [
            QCEvent(
                code="missing_analysis_mask",
                severity="warning",
                message="No DICOM Segmentation Storage object was found",
            )
        ]
    if len(candidates) > 1:
        return None, [
            QCEvent(
                code="multiple_analysis_masks",
                severity="error",
                message="Multiple SEG series require explicit disambiguation",
                details={"series_uids": [item.header.series_uid for item in candidates]},
            )
        ]
    candidate = candidates[0]
    return (
        MaskRef(
            series_uid=candidate.header.series_uid,
            series_path=str(candidate.path.resolve()),
            sop_class_uid=candidate.header.sop_class_uid,
            mask_type="TCIA_VOLSER_analysis_mask",
            semantic_status="unverified_ftv_encoding",
            source="DICOM_SEG_derived_analysis",
        ),
        [
            QCEvent(
                code="mask_semantics_unverified",
                severity="warning",
                message="FTV analysis-mask segment meanings have not been verified",
            )
        ],
    )


def _clinical_for_patient(
    records: Mapping[str, ClinicalRecord], patient_id: str
) -> ClinicalRecord:
    return records.get(
        patient_id,
        ClinicalRecord(patient_id, baseline={}, treatment={}, evaluation_only={}),
    )


def build_visit_manifest(
    data_root: str | Path,
    *,
    clinical_path: str | Path | None = None,
    relative_dates_verified: bool = False,
    derived_phase_convention_verified: bool = False,
    scan_phase_headers: bool = True,
) -> list[VisitRecord]:
    """Index one visit per StudyInstanceUID without reading pixel data."""

    summaries = index_dicom_series(data_root)
    clinical = load_clinical_records(clinical_path) if clinical_path else {}
    by_study: dict[str, list[SeriesSummary]] = defaultdict(list)
    for summary in summaries:
        by_study[summary.header.study_uid].append(summary)

    visits: list[VisitRecord] = []
    for study_uid, series in sorted(by_study.items()):
        first = series[0]
        patient_ids = {item.header.patient_id for item in series}
        collections = {item.collection for item in series}
        descriptions = {item.header.study_description for item in series}
        qc: list[QCEvent] = [event for item in series for event in item.qc]
        if len(patient_ids) != 1:
            qc.append(
                QCEvent(
                    code="mixed_patient_study",
                    severity="error",
                    message="One StudyInstanceUID contains multiple PatientID values",
                    details={"patient_ids": sorted(patient_ids)},
                )
            )
        if len(collections) != 1:
            qc.append(
                QCEvent(
                    code="mixed_collection_study",
                    severity="error",
                    message="One study appears in multiple collections",
                    details={"collections": sorted(collections)},
                )
            )
        patient_id = first.header.patient_id or first.patient_directory
        stage_values = {
            stage
            for description in descriptions
            if (stage := parse_visit_stage(description)) is not None
        }
        if len(stage_values) != 1:
            qc.append(
                QCEvent(
                    code="ambiguous_visit_stage",
                    severity="error",
                    message="StudyDescription does not identify exactly one T0-T3 visit",
                    details={"descriptions": sorted(descriptions)},
                )
            )
            visit_stage = "UNKNOWN"
        else:
            visit_stage = next(iter(stage_values))

        study_dates = {
            parsed
            for item in series
            if (parsed := parse_dicom_date(item.header.study_date)) is not None
        }
        if len(study_dates) > 1:
            qc.append(
                QCEvent(
                    code="inconsistent_study_date",
                    severity="error",
                    message="Series within one study contain different StudyDate values",
                    details={"dates": sorted(value.isoformat() for value in study_dates)},
                )
            )
        study_date = min(study_dates).isoformat() if study_dates else None
        if not study_dates:
            qc.append(
                QCEvent(
                    code="missing_study_date",
                    severity="warning",
                    message="No parseable DICOM StudyDate was found",
                )
            )

        dce_candidate, candidate_qc = choose_dce_candidate(series)
        qc.extend(candidate_qc)
        phases = {}
        if dce_candidate is not None and scan_phase_headers:
            assessment = identify_temporal_dce_phases(
                dce_candidate,
                read_series_headers(dce_candidate.path),
                derived_phase_convention_verified=derived_phase_convention_verified,
            )
            phases = assessment.phases
            qc.extend(assessment.qc)
        elif dce_candidate is not None:
            qc.append(
                QCEvent(
                    code="phase_headers_not_scanned",
                    severity="error",
                    message="DCE candidate exists but temporal instance headers were not scanned",
                )
            )

        mask, mask_qc = _mask_from_series(series)
        qc.extend(mask_qc)
        clinical_record = _clinical_for_patient(clinical, patient_id)
        if not clinical_record.baseline:
            qc.append(
                QCEvent(
                    code="baseline_clinical_missing",
                    severity="warning",
                    message="No whitelisted baseline clinical predictors are available",
                )
            )
        if not clinical_record.treatment:
            qc.append(
                QCEvent(
                    code="treatment_missing",
                    severity="warning",
                    message="No verified treatment field is available",
                )
            )

        visits.append(
            VisitRecord(
                visit_id=f"{first.collection}:{study_uid}",
                patient_id=patient_id,
                collection=first.collection,
                study_uid=study_uid,
                visit_stage=visit_stage,
                study_date=study_date,
                date_source="dicom_study_date_deidentified",
                relative_date_verified=relative_dates_verified,
                phase_paths=phases,
                mask=mask,
                baseline_clinical=clinical_record.baseline,
                treatment=clinical_record.treatment,
                evaluation_metadata=clinical_record.evaluation_only,
                qc=tuple(qc),
                series_count=len(series),
            )
        )
    stage_counts = Counter((visit.patient_id, visit.visit_stage) for visit in visits)
    duplicate_keys = {key for key, count in stage_counts.items() if count > 1}
    if duplicate_keys:
        duplicate_event = QCEvent(
            code="duplicate_patient_visit_stage",
            severity="error",
            message="A patient has multiple StudyInstanceUID values for one visit stage",
        )
        visits = [
            replace(visit, qc=(*visit.qc, duplicate_event))
            if (visit.patient_id, visit.visit_stage) in duplicate_keys
            else visit
            for visit in visits
        ]
    return visits


def _pair_id(earlier: VisitRecord, later: VisitRecord) -> str:
    value = "\0".join(
        (
            earlier.collection,
            earlier.patient_id,
            earlier.study_uid,
            later.study_uid,
            earlier.visit_stage,
            later.visit_stage,
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def build_pair_manifest(
    visits: Iterable[VisitRecord],
    split_by_patient: Mapping[str, str],
    *,
    earlier_stage: str = "T0",
    later_stage: str = "T1",
    require_three_phase: bool = True,
) -> list[PairRecord]:
    """Pair fixed earlier/later records; backward use does not swap them."""

    stage_order = {f"T{index}": index for index in range(100)}
    if (
        earlier_stage not in stage_order
        or later_stage not in stage_order
        or stage_order[earlier_stage] >= stage_order[later_stage]
    ):
        raise ValueError("pair stages must preserve an earlier Tn and later Tm with n < m")
    records = list(visits)
    by_patient: dict[str, dict[str, list[VisitRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for visit in records:
        by_patient[visit.patient_id][visit.visit_stage].append(visit)

    pairs: list[PairRecord] = []
    for patient_id, stages in sorted(by_patient.items()):
        if earlier_stage not in stages or later_stage not in stages:
            continue
        if len(stages[earlier_stage]) != 1 or len(stages[later_stage]) != 1:
            raise ValueError(
                f"patient {patient_id} has an ambiguous {earlier_stage}-{later_stage} pair"
            )
        earlier = stages[earlier_stage][0]
        later = stages[later_stage][0]
        qc: list[QCEvent] = []
        if patient_id not in split_by_patient:
            raise ValueError(f"no split assignment for patient {patient_id}")
        if earlier.collection != later.collection:
            qc.append(
                QCEvent(
                    code="cross_collection_pair",
                    severity="error",
                    message="Longitudinal visits resolve to different collections",
                )
            )
        if require_three_phase and not (earlier.three_phase_ready and later.three_phase_ready):
            qc.append(
                QCEvent(
                    code="pair_not_three_phase_ready",
                    severity="error",
                    message="Both visits need reliable pre/early/late phase references",
                )
            )

        earlier_date = parse_dicom_date(earlier.study_date)
        later_date = parse_dicom_date(later.study_date)
        observed_delta = (
            (later_date - earlier_date).days
            if earlier_date is not None and later_date is not None
            else None
        )
        if observed_delta is not None and observed_delta <= 0:
            qc.append(
                QCEvent(
                    code="nonpositive_visit_interval",
                    severity="error",
                    message="The later visit date is not after the earlier visit date",
                    details={"observed_delta_days": observed_delta},
                )
            )
        verified = earlier.relative_date_verified and later.relative_date_verified
        delta_days = observed_delta if verified and observed_delta and observed_delta > 0 else None
        interval_missing = delta_days is None
        interval_source = (
            "verified_relative_dicom_study_date"
            if delta_days is not None
            else "stage_only_unverified_deidentified_date"
        )
        baseline = earlier.baseline_clinical
        treatment = earlier.treatment or later.treatment
        pairs.append(
            PairRecord(
                pair_id=_pair_id(earlier, later),
                patient_id=patient_id,
                collection=earlier.collection,
                earlier_visit_id=earlier.visit_id,
                later_visit_id=later.visit_id,
                earlier_stage=earlier_stage,
                later_stage=later_stage,
                split=split_by_patient[patient_id],
                delta_days=delta_days,
                observed_delta_days=observed_delta,
                interval_missing=interval_missing,
                interval_source=interval_source,
                earlier_prepared_path=earlier.prepared_path,
                later_prepared_path=later.prepared_path,
                baseline_clinical=baseline,
                treatment=treatment,
                qc=tuple(qc),
            )
        )
    return pairs


def write_jsonl(records: Iterable[Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in records:
        value = record.to_dict() if hasattr(record, "to_dict") else record
        lines.append(json.dumps(value, sort_keys=True, separators=(",", ":")))
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _geometry_from_dict(value: Mapping[str, Any] | None) -> Geometry | None:
    if value is None:
        return None
    affine = tuple(tuple(float(item) for item in row) for row in value["affine_lps"])
    if len(affine) != 4 or any(len(row) != 4 for row in affine):
        raise ValueError("geometry affine_lps must have shape [4,4]")
    return Geometry(
        shape_dhw=tuple(int(item) for item in value["shape_dhw"]),  # type: ignore[arg-type]
        spacing_dhw=tuple(float(item) for item in value["spacing_dhw"]),  # type: ignore[arg-type]
        orientation_lps=tuple(float(item) for item in value["orientation_lps"]),  # type: ignore[arg-type]
        origin_lps=tuple(float(item) for item in value["origin_lps"]),  # type: ignore[arg-type]
        affine_lps=affine,  # type: ignore[arg-type]
        source=str(value.get("source", "dicom_patient_geometry")),
    )


def _qc_from_dict(value: Mapping[str, Any]) -> QCEvent:
    return QCEvent(
        code=str(value["code"]),
        severity=str(value["severity"]),
        message=str(value["message"]),
        details=dict(value.get("details") or {}),
    )


def visit_from_dict(value: Mapping[str, Any]) -> VisitRecord:
    """Deserialize and validate a visit JSONL object for ``prepare_dataset``."""

    phase_values = value.get("phase_paths") or {}
    phases = {
        str(role): PhaseRef(
            role=str(item.get("role", role)),
            series_uid=str(item["series_uid"]),
            series_path=str(item["series_path"]),
            source_kind=str(item["source_kind"]),
            temporal_position=(
                int(item["temporal_position"])
                if item.get("temporal_position") is not None
                else None
            ),
            acquisition_time=(
                str(item["acquisition_time"])
                if item.get("acquisition_time") is not None
                else None
            ),
            instance_count=int(item["instance_count"]),
            reliability=str(item["reliability"]),
            evidence=tuple(str(entry) for entry in item.get("evidence", ())),
            geometry=_geometry_from_dict(item.get("geometry")),
        )
        for role, item in phase_values.items()
    }
    mask_value = value.get("mask")
    mask = (
        MaskRef(
            series_uid=str(mask_value["series_uid"]),
            series_path=str(mask_value["series_path"]),
            sop_class_uid=str(mask_value["sop_class_uid"]),
            mask_type=str(mask_value["mask_type"]),
            semantic_status=str(mask_value["semantic_status"]),
            source=str(mask_value["source"]),
        )
        if mask_value
        else None
    )
    return VisitRecord(
        visit_id=str(value["visit_id"]),
        patient_id=str(value["patient_id"]),
        collection=str(value["collection"]),
        study_uid=str(value["study_uid"]),
        visit_stage=str(value["visit_stage"]),
        study_date=str(value["study_date"]) if value.get("study_date") else None,
        date_source=str(value.get("date_source", "unknown")),
        relative_date_verified=bool(value.get("relative_date_verified", False)),
        phase_paths=phases,
        mask=mask,
        baseline_clinical=dict(value.get("baseline_clinical") or {}),
        treatment=dict(value.get("treatment") or {}),
        evaluation_metadata=dict(value.get("evaluation_metadata") or {}),
        qc=tuple(_qc_from_dict(item) for item in value.get("qc", ())),
        series_count=int(value.get("series_count", 0)),
        split=str(value["split"]) if value.get("split") else None,
        prepared_path=(
            str(value["prepared_path"]) if value.get("prepared_path") else None
        ),
        schema_version=str(value.get("schema_version", "1.0")),
    )


def read_visit_manifest(path: str | Path) -> list[VisitRecord]:
    return [visit_from_dict(value) for value in read_jsonl(path)]


def manifest_counts(visits: Iterable[VisitRecord]) -> dict[str, Any]:
    items = list(visits)
    return {
        "patients": len({item.patient_id for item in items}),
        "visits": len(items),
        "visits_by_stage": dict(Counter(item.visit_stage for item in items)),
        "three_phase_ready": sum(item.three_phase_ready for item in items),
    }
