"""Fast dataset audit using TCIA metadata plus optional pixel-free headers."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from .clinical import load_clinical_records
from .dicom import iter_series_directories, summarize_series
from .manifest import parse_visit_stage
from .schema import QCEvent


@dataclass(frozen=True)
class DataAudit:
    data_root: str
    dataset_kind: str
    collection_counts: Mapping[str, int]
    patient_count: int
    study_count: int
    series_count: int
    dicom_instance_count: int
    dicom_bytes: int
    visit_counts: Mapping[str, int]
    visit_patterns: Mapping[str, int]
    pair_counts: Mapping[str, int]
    observed_pair_intervals_days: Mapping[str, Mapping[str, float | int]]
    modality_counts: Mapping[str, int]
    sop_class_counts: Mapping[str, int]
    manufacturer_counts: Mapping[str, int]
    transfer_syntax_sample_counts: Mapping[str, int]
    derived_series_counts: Mapping[str, int]
    dce_candidate_studies: int
    segmentation_studies: int
    clinical_patient_count: int
    baseline_clinical_fields: tuple[str, ...]
    treatment_fields: tuple[str, ...]
    evaluation_only_fields: tuple[str, ...]
    date_status: str
    phase_status: str
    mask_status: str
    duplicate_series_uid_count: int
    header_sample_size: int
    qc: tuple[QCEvent, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_root": self.data_root,
            "dataset_kind": self.dataset_kind,
            "collection_counts": dict(self.collection_counts),
            "patient_count": self.patient_count,
            "study_count": self.study_count,
            "series_count": self.series_count,
            "dicom_instance_count": self.dicom_instance_count,
            "dicom_bytes": self.dicom_bytes,
            "visit_counts": dict(self.visit_counts),
            "visit_patterns": dict(self.visit_patterns),
            "pair_counts": dict(self.pair_counts),
            "observed_pair_intervals_days": {
                key: dict(value)
                for key, value in self.observed_pair_intervals_days.items()
            },
            "modality_counts": dict(self.modality_counts),
            "sop_class_counts": dict(self.sop_class_counts),
            "manufacturer_counts": dict(self.manufacturer_counts),
            "transfer_syntax_sample_counts": dict(self.transfer_syntax_sample_counts),
            "derived_series_counts": dict(self.derived_series_counts),
            "dce_candidate_studies": self.dce_candidate_studies,
            "segmentation_studies": self.segmentation_studies,
            "clinical_patient_count": self.clinical_patient_count,
            "baseline_clinical_fields": list(self.baseline_clinical_fields),
            "treatment_fields": list(self.treatment_fields),
            "evaluation_only_fields": list(self.evaluation_only_fields),
            "date_status": self.date_status,
            "phase_status": self.phase_status,
            "mask_status": self.mask_status,
            "duplicate_series_uid_count": self.duplicate_series_uid_count,
            "header_sample_size": self.header_sample_size,
            "qc": [event.to_dict() for event in self.qc],
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _date(value: str) -> datetime | None:
    for pattern in ("%m-%d-%Y", "%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, pattern)
        except (TypeError, ValueError):
            pass
    return None


def _interval_summary(
    studies: Mapping[str, Mapping[str, dict[str, str]]], earlier: str, later: str
) -> tuple[int, dict[str, float | int]]:
    deltas: list[int] = []
    pair_count = 0
    for patient_studies in studies.values():
        if earlier not in patient_studies or later not in patient_studies:
            continue
        pair_count += 1
        left = _date(patient_studies[earlier].get("Study Date", ""))
        right = _date(patient_studies[later].get("Study Date", ""))
        if left is not None and right is not None:
            deltas.append((right - left).days)
    if not deltas:
        return pair_count, {"dated_pairs": 0}
    return pair_count, {
        "dated_pairs": len(deltas),
        "min": min(deltas),
        "median": float(median(deltas)),
        "max": max(deltas),
        "nonpositive": sum(value <= 0 for value in deltas),
    }


def _derived_kind(description: str) -> str | None:
    value = description.lower()
    if "volser" not in value:
        return None
    if "original dce" in value:
        return "volser_original_dce"
    if "analysis mask" in value:
        return "volser_analysis_mask"
    if "pe2" in value:
        return "volser_pe2"
    if "pe5" in value:
        return "volser_pe5"
    if "pe6" in value:
        return "volser_pe6"
    if value.rstrip().endswith("ser") or ": ser" in value:
        return "volser_ser"
    return "volser_other"


def audit_dataset(
    data_root: str | Path,
    clinical_path: str | Path | None = None,
    *,
    metadata_csv: str | Path | None = None,
    header_sample_limit: int = 32,
) -> DataAudit:
    """Audit a TCIA-style tree without reading or decoding Pixel Data."""

    root = Path(data_root).resolve()
    metadata_path = (
        Path(metadata_csv).expanduser().resolve()
        if metadata_csv is not None
        else root / "metadata.csv"
    )
    qc: list[QCEvent] = []
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"series metadata CSV is required for the fast audit: {metadata_path}"
        )
    rows = _read_metadata(metadata_path)
    if not rows:
        raise ValueError("metadata.csv contains no series rows")

    series_uids = [row.get("Series UID", "") for row in rows]
    duplicate_series_uid_count = sum(
        count - 1 for count in Counter(series_uids).values() if count > 1
    )
    if duplicate_series_uid_count:
        qc.append(
            QCEvent(
                code="duplicate_series_uid",
                severity="error",
                message="metadata.csv repeats SeriesInstanceUID values",
                details={"duplicate_rows": duplicate_series_uid_count},
            )
        )

    study_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        study_rows.setdefault(row.get("Study UID", ""), row)
    patient_studies: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    unknown_stage_studies = 0
    duplicate_stage_studies = 0
    for row in study_rows.values():
        patient_id = row.get("Subject ID", "")
        stage = parse_visit_stage(row.get("Study Description", ""))
        if stage is None:
            unknown_stage_studies += 1
            continue
        if stage in patient_studies[patient_id]:
            duplicate_stage_studies += 1
        patient_studies[patient_id][stage] = row
    if unknown_stage_studies:
        qc.append(
            QCEvent(
                code="unknown_visit_stage",
                severity="error",
                message="Some studies lack a recognized T0-T3 description",
                details={"studies": unknown_stage_studies},
            )
        )
    if duplicate_stage_studies:
        qc.append(
            QCEvent(
                code="duplicate_patient_stage",
                severity="error",
                message="A patient has multiple studies for the same visit stage",
                details={"studies": duplicate_stage_studies},
            )
        )

    visit_counts = Counter(
        stage for stages in patient_studies.values() for stage in stages
    )
    visit_patterns = Counter(
        "+".join(stage for stage in ("T0", "T1", "T2", "T3") if stage in stages)
        for stages in patient_studies.values()
    )
    pair_counts: dict[str, int] = {}
    intervals: dict[str, dict[str, float | int]] = {}
    for earlier, later in (
        ("T0", "T1"),
        ("T0", "T2"),
        ("T0", "T3"),
        ("T1", "T2"),
        ("T1", "T3"),
        ("T2", "T3"),
    ):
        key = f"{earlier}-{later}"
        count, summary = _interval_summary(patient_studies, earlier, later)
        pair_counts[key] = count
        intervals[key] = summary

    derived_counts = Counter()
    dce_studies: set[str] = set()
    seg_studies: set[str] = set()
    for row in rows:
        kind = _derived_kind(row.get("Series Description", ""))
        if kind:
            derived_counts[kind] += 1
        if kind == "volser_original_dce":
            dce_studies.add(row.get("Study UID", ""))
        if row.get("Modality") == "SEG":
            seg_studies.add(row.get("Study UID", ""))

    if len(dce_studies) != len(study_rows):
        qc.append(
            QCEvent(
                code="missing_dce_candidate_study",
                severity="error",
                message="Not every study has one consolidated original-DCE candidate",
                details={"candidate_studies": len(dce_studies), "studies": len(study_rows)},
            )
        )
    if len(seg_studies) != len(study_rows):
        qc.append(
            QCEvent(
                code="missing_seg_study",
                severity="warning",
                message="Not every study has a DICOM SEG object",
                details={"segmentation_studies": len(seg_studies), "studies": len(study_rows)},
            )
        )

    transfer_syntax_counts: Counter[str] = Counter()
    sampled = 0
    if header_sample_limit > 0:
        for collection, patient, study, series_path in iter_series_directories(root):
            if sampled >= header_sample_limit:
                break
            try:
                summary = summarize_series(collection, patient, study, series_path)
            except (OSError, ValueError, EOFError) as exc:
                qc.append(
                    QCEvent(
                        code="dicom_header_read_failed",
                        severity="error",
                        message="A sampled DICOM series header could not be read",
                        details={"series_path": str(series_path), "error": type(exc).__name__},
                    )
                )
                continue
            transfer_syntax_counts[summary.header.transfer_syntax_uid or "missing"] += 1
            sampled += 1

    clinical_records = load_clinical_records(clinical_path) if clinical_path else {}
    baseline_fields = sorted(
        {field for record in clinical_records.values() for field in record.baseline}
    )
    treatment_fields = sorted(
        {field for record in clinical_records.values() for field in record.treatment}
    )
    evaluation_fields = sorted(
        {field for record in clinical_records.values() for field in record.evaluation_only}
    )
    if not clinical_records:
        qc.append(
            QCEvent(
                code="clinical_data_absent",
                severity="warning",
                message="No clinical table was supplied; treatment effects cannot be trained or validated",
            )
        )

    descriptions = [row.get("Series Description", "") for row in rows]
    dataset_kind = (
        "TCIA ISPY2 with original MR and VOLSER-derived DCE/SEG series"
        if any("VOLSER" in value.upper() for value in descriptions)
        else "TCIA ISPY2 imaging series"
    )
    collection_counts = Counter(row.get("Collection", "<missing>") for row in rows)
    missing_study_dates = sum(
        _date(row.get("Study Date", "")) is None for row in study_rows.values()
    )
    nonpositive_intervals = sum(
        int(summary.get("nonpositive", 0)) for summary in intervals.values()
    )
    date_status = (
        "missing or non-monotonic study dates require stage-only intervals"
        if missing_study_dates or nonpositive_intervals
        else "stage order and observed intervals are internally consistent; relative date preservation remains unverified"
    )
    return DataAudit(
        data_root=str(root),
        dataset_kind=dataset_kind,
        collection_counts=dict(collection_counts),
        patient_count=len(patient_studies),
        study_count=len(study_rows),
        series_count=len(rows),
        dicom_instance_count=sum(int(row.get("Number of Images", 0) or 0) for row in rows),
        dicom_bytes=sum(int(row.get("File Size", 0) or 0) for row in rows),
        visit_counts=dict(sorted(visit_counts.items())),
        visit_patterns=dict(sorted(visit_patterns.items())),
        pair_counts=pair_counts,
        observed_pair_intervals_days=intervals,
        modality_counts=dict(Counter(row.get("Modality", "") for row in rows)),
        sop_class_counts=dict(Counter(row.get("SOP Class UID", "") for row in rows)),
        manufacturer_counts=dict(Counter(row.get("Manufacturer", "") for row in rows)),
        transfer_syntax_sample_counts=dict(transfer_syntax_counts),
        derived_series_counts=dict(sorted(derived_counts.items())),
        dce_candidate_studies=len(dce_studies),
        segmentation_studies=len(seg_studies),
        clinical_patient_count=len(clinical_records),
        baseline_clinical_fields=tuple(baseline_fields),
        treatment_fields=tuple(treatment_fields),
        evaluation_only_fields=tuple(evaluation_fields),
        date_status=date_status,
        phase_status=(
            "consolidated DCE candidates exist; pre/early/late require temporal-header QC and verified phase semantics"
        ),
        mask_status=(
            "DICOM SEG VOLSER analysis masks exist; FTV segment encoding is unverified and disabled for metrics/cropping"
        ),
        duplicate_series_uid_count=duplicate_series_uid_count,
        header_sample_size=sampled,
        qc=tuple(qc),
    )
