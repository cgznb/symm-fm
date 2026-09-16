from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import DCE0Phase, select_dce0


ACCEPTED_REGISTRATION_STATUSES = frozenset(
    {"fixed_reference", "rigid_fallback", "deformable"}
)
ACCEPTED_BACKENDS = frozenset({"current", "registered_t0"})


@dataclass(frozen=True)
class VisitRecord:
    visit_id: str
    patient_id: str
    visit: str
    dce_paths: tuple[Path, ...]
    dce0: DCE0Phase
    mask_path: Path
    meta_path: Path
    qc_status: str
    registration_status: str | None


def _required_string(row: pd.Series, field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest {field} must be a nonempty string")
    return value.strip()


def _required_path(row: pd.Series, field: str) -> Path:
    path = Path(_required_string(row, field))
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"manifest {field} is missing or unsafe")
    return path.resolve()


def _n_times(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("manifest n_times is invalid") from None
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise ValueError("manifest n_times is invalid")
    return int(numeric)


def build_visit_index(frame: pd.DataFrame, *, backend: str) -> dict[str, VisitRecord]:
    if backend not in ACCEPTED_BACKENDS:
        raise ValueError("data backend is invalid")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("visit manifest must be a nonempty table")
    required = {"patient_id", "visit", "n_times", "dce_paths", "mask_path", "meta_path", "qc_status"}
    if backend == "registered_t0":
        required.add("registration_status")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"visit manifest is missing columns: {sorted(missing)}")

    result: dict[str, VisitRecord] = {}
    for _, row in frame.iterrows():
        patient_id = _required_string(row, "patient_id")
        visit = _required_string(row, "visit")
        count = _n_times(row["n_times"])
        dce_paths = tuple(Path(value) for value in _required_string(row, "dce_paths").split(";"))
        dce0 = select_dce0(dce_paths, n_times=count)
        status: str | None = None
        if backend == "registered_t0":
            status = _required_string(row, "registration_status")
            if status not in ACCEPTED_REGISTRATION_STATUSES:
                raise ValueError(f"registration status is not a terminal success: {status}")
        visit_id = f"{patient_id}:{visit}"
        if visit_id in result:
            raise ValueError(f"visit manifest contains duplicate visit: {visit_id}")
        result[visit_id] = VisitRecord(
            visit_id=visit_id,
            patient_id=patient_id,
            visit=visit,
            dce_paths=tuple(path.resolve() for path in dce_paths),
            dce0=dce0,
            mask_path=_required_path(row, "mask_path"),
            meta_path=_required_path(row, "meta_path"),
            qc_status=_required_string(row, "qc_status"),
            registration_status=status,
        )
    return result
