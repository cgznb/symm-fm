"""Clinical table ingestion with explicit condition/outcome separation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .schema import QCEvent


PATIENT_ID_FIELDS = (
    "patient_id",
    "patientid",
    "subject_id",
    "subjectid",
    "case_id",
)

BASELINE_FIELDS = {
    "age",
    "hr_status",
    "her2_status",
    "mammaprint",
    "molecular_subtype",
    "clinical_stage",
    "tumor_grade",
}

TREATMENT_FIELDS = {
    "treatment_arm",
    "regimen",
    "drug",
    "drug_name",
    "therapy",
    "treatment_start_date",
    "treatment_end_date",
}

OUTCOME_FIELDS = {
    "pcr",
    "rcb",
    "rcb_class",
    "response",
    "outcome",
    "survival",
    "recurrence",
}


def normalize_field_name(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


_POSITIVE_STATUS_VALUES = {
    "1",
    "detected",
    "pos",
    "positive",
    "receptor_positive",
    "true",
    "y",
    "yes",
}
_NEGATIVE_STATUS_VALUES = {
    "0",
    "absent",
    "false",
    "n",
    "neg",
    "negative",
    "no",
    "not_detected",
    "receptor_negative",
}
_UNAVAILABLE_STATUS_VALUES = {
    "equivocal",
    "indeterminate",
    "missing",
    "n_a",
    "na",
    "not_assessed",
    "not_available",
    "not_tested",
    "unk",
    "unknown",
}


def _canonical_alias(
    values: dict[str, Any],
    *,
    canonical: str,
    alias: str,
    patient_id: str,
) -> None:
    present = [(name, values[name]) for name in (canonical, alias) if name in values]
    if not present:
        return
    normalized = {str(value).strip().casefold() for _, value in present}
    if len(normalized) != 1:
        raise ValueError(
            f"clinical aliases {canonical!r}/{alias!r} disagree for patient {patient_id}"
        )
    values[canonical] = present[0][1]
    values.pop(alias, None)


def _binary_receptor_status(value: Any, *, field: str, patient_id: str) -> str | None:
    if isinstance(value, bool):
        return "positive" if value else "negative"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return "positive"
        if value == 0:
            return "negative"
    raw = str(value).strip().lower()
    if raw in {"+", "er+", "pr+", "hr+"}:
        return "positive"
    if raw in {"-", "er-", "pr-", "hr-"}:
        return "negative"
    normalized = normalize_field_name(raw)
    if normalized in _POSITIVE_STATUS_VALUES:
        return "positive"
    if normalized in _NEGATIVE_STATUS_VALUES:
        return "negative"
    if normalized in _UNAVAILABLE_STATUS_VALUES:
        return None
    raise ValueError(
        f"unrecognized {field} value {value!r} for patient {patient_id}; "
        f"cannot derive a trustworthy {field}"
    )


def _canonicalize_hr_status(values: dict[str, Any], *, patient_id: str) -> None:
    explicit_present = "hr_status" in values
    explicit = (
        _binary_receptor_status(
            values["hr_status"], field="hr_status", patient_id=patient_id
        )
        if explicit_present
        else None
    )
    receptor_fields = [name for name in ("er_status", "pr_status") if name in values]
    receptors = {
        name: _binary_receptor_status(values[name], field=name, patient_id=patient_id)
        for name in receptor_fields
    }
    known_receptors = [value for value in receptors.values() if value is not None]
    inferred: str | None = None
    if "positive" in known_receptors:
        inferred = "positive"
    elif len(receptor_fields) == 2 and known_receptors == ["negative", "negative"]:
        inferred = "negative"

    if explicit is not None and inferred is not None and explicit != inferred:
        raise ValueError(
            f"hr_status conflicts with ER/PR status for patient {patient_id}"
        )
    canonical = explicit or inferred
    for name in ("er_status", "pr_status", "hr_status"):
        values.pop(name, None)
    if canonical is not None:
        values["hr_status"] = canonical


def _canonicalize_binary_status(
    values: dict[str, Any], *, field: str, patient_id: str
) -> None:
    if field not in values:
        return
    canonical = _binary_receptor_status(
        values.pop(field), field=field, patient_id=patient_id
    )
    if canonical is not None:
        values[field] = canonical


@dataclass(frozen=True)
class ClinicalRecord:
    patient_id: str
    baseline: Mapping[str, Any]
    treatment: Mapping[str, Any]
    evaluation_only: Mapping[str, Any]
    qc: tuple[QCEvent, ...] = ()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            value = value.get("records", value.get("patients", value))
        if isinstance(value, dict):
            return [dict(row, patient_id=patient_id) for patient_id, row in value.items()]
        if not isinstance(value, list):
            raise ValueError("clinical JSON must contain a list or patient-keyed object")
        return [dict(row) for row in value]
    delimiter = "\t" if suffix in {".tsv", ".tab"} else ","
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _first_matching_field(fieldnames: Iterable[str]) -> str | None:
    normalized = {normalize_field_name(name): name for name in fieldnames}
    for candidate in PATIENT_ID_FIELDS:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_clinical_records(path: str | Path) -> dict[str, ClinicalRecord]:
    """Load whitelisted predictors and quarantine outcomes from conditions."""

    source = Path(path)
    rows = _read_rows(source)
    if not rows:
        return {}
    patient_field = _first_matching_field(rows[0].keys())
    if patient_field is None:
        raise ValueError("clinical table has no recognized patient identifier column")

    records: dict[str, ClinicalRecord] = {}
    for row in rows:
        patient_id = str(row.get(patient_field, "")).strip()
        if not patient_id:
            raise ValueError("clinical table contains a blank patient identifier")
        if patient_id in records:
            raise ValueError(f"duplicate clinical row for patient {patient_id}")
        normalized = {
            normalize_field_name(name): value
            for name, value in row.items()
            if name != patient_field
            and value is not None
            and (not isinstance(value, str) or value.strip())
        }
        _canonical_alias(
            normalized,
            canonical="age",
            alias="age_at_diagnosis",
            patient_id=patient_id,
        )
        _canonical_alias(
            normalized,
            canonical="treatment_arm",
            alias="arm",
            patient_id=patient_id,
        )
        _canonicalize_hr_status(normalized, patient_id=patient_id)
        _canonicalize_binary_status(
            normalized, field="her2_status", patient_id=patient_id
        )
        baseline = {key: normalized[key] for key in BASELINE_FIELDS & normalized.keys()}
        treatment = {
            key: normalized[key] for key in TREATMENT_FIELDS & normalized.keys()
        }
        evaluation_only = {
            key: value
            for key, value in normalized.items()
            if key in OUTCOME_FIELDS
            or any(token in key for token in ("pcr", "rcb", "outcome", "survival"))
        }
        records[patient_id] = ClinicalRecord(
            patient_id=patient_id,
            baseline=baseline,
            treatment=treatment,
            evaluation_only=evaluation_only,
        )
    return records
