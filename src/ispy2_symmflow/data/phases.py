"""Conservative DCE phase identification from DICOM temporal metadata."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping

from .dicom import (
    DicomHeader,
    MR_IMAGE_STORAGE_UID,
    SeriesSummary,
    geometries_match,
    geometry_from_headers,
)
from .schema import PhaseRef, QCEvent


@dataclass(frozen=True)
class PhaseAssessment:
    phases: Mapping[str, PhaseRef]
    usable: bool
    strategy: str
    qc: tuple[QCEvent, ...]


def parse_dicom_time(value: str | None) -> float | None:
    """Convert a DICOM TM value to seconds since midnight."""

    if not value:
        return None
    compact = value.replace(":", "").strip()
    try:
        hours = int(compact[0:2])
        minutes = int(compact[2:4]) if len(compact) >= 4 else 0
        seconds = float(compact[4:]) if len(compact) > 4 else 0.0
    except (TypeError, ValueError):
        return None
    if not (0 <= hours <= 23 and 0 <= minutes <= 59 and 0 <= seconds < 61):
        return None
    return hours * 3600.0 + minutes * 60.0 + seconds


def _chronological_order(
    representative_times: Mapping[int, float],
) -> list[int] | None:
    """Order short acquisitions, allowing a single midnight rollover."""

    if len(set(representative_times.values())) != len(representative_times):
        return None
    values = list(representative_times.values())
    if max(values) - min(values) <= 12 * 3600:
        return sorted(representative_times, key=representative_times.get)  # type: ignore[arg-type]
    adjusted = {
        key: value + (24 * 3600 if value < 12 * 3600 else 0)
        for key, value in representative_times.items()
    }
    if max(adjusted.values()) - min(adjusted.values()) > 12 * 3600:
        return None
    return sorted(adjusted, key=adjusted.get)  # type: ignore[arg-type]


def _event(code: str, severity: str, message: str, **details: object) -> QCEvent:
    return QCEvent(code=code, severity=severity, message=message, details=details)


def identify_temporal_dce_phases(
    series: SeriesSummary,
    headers: Iterable[DicomHeader],
    *,
    derived_phase_convention_verified: bool = False,
) -> PhaseAssessment:
    """Identify pre/early/late phases in one spatially matched DCE series.

    Temporal Position Identifier is used to group instances, while actual
    acquisition time orders the groups. It does not itself establish which
    phase is pre-contrast. Unless a bolus time is present, the first/second/last
    convention must be explicitly verified for the derived dataset.
    """

    items = list(headers)
    qc: list[QCEvent] = list(series.qc)
    if not items:
        return PhaseAssessment(
            phases={},
            usable=False,
            strategy="dicom_temporal_position",
            qc=(
                _event("empty_dce_series", "error", "DCE candidate has no instances"),
            ),
        )

    series_uids = {item.series_uid for item in items}
    if len(series_uids) != 1 or series.header.series_uid not in series_uids:
        qc.append(
            _event(
                "mixed_series_uid",
                "error",
                "Files in a DCE directory do not share one SeriesInstanceUID",
                series_uids=sorted(series_uids),
            )
        )
    if any(item.sop_class_uid != MR_IMAGE_STORAGE_UID for item in items):
        qc.append(
            _event(
                "unexpected_dce_sop_class",
                "error",
                "DCE phase source is not entirely MR Image Storage",
            )
        )
    if any(item.temporal_position is None for item in items):
        qc.append(
            _event(
                "missing_temporal_position",
                "error",
                "At least one DCE instance lacks TemporalPositionIdentifier",
                missing=sum(item.temporal_position is None for item in items),
            )
        )
        return PhaseAssessment({}, False, "dicom_temporal_position", tuple(qc))

    groups: dict[int, list[DicomHeader]] = defaultdict(list)
    for item in items:
        assert item.temporal_position is not None
        groups[item.temporal_position].append(item)
    if len(groups) < 3:
        qc.append(
            _event(
                "insufficient_dce_phases",
                "error",
                "At least three temporal positions are required",
                observed=len(groups),
            )
        )
        return PhaseAssessment({}, False, "dicom_temporal_position", tuple(qc))

    declared_counts = {
        item.number_of_temporal_positions
        for item in items
        if item.number_of_temporal_positions is not None
    }
    if len(declared_counts) > 1 or (
        declared_counts and len(groups) not in declared_counts
    ):
        qc.append(
            _event(
                "temporal_position_count_mismatch",
                "warning",
                "Observed temporal groups differ from the DICOM declaration",
                observed=len(groups),
                declared=sorted(declared_counts),
            )
        )

    geometries = {position: geometry_from_headers(group) for position, group in groups.items()}
    first_position = next(iter(groups))
    reference_geometry = geometries[first_position]
    if reference_geometry is None or any(
        not geometries_match(reference_geometry, geometry)
        for geometry in geometries.values()
    ):
        qc.append(
            _event(
                "phase_geometry_mismatch",
                "error",
                "DCE temporal positions do not have matching spatial geometry",
            )
        )

    duplicate_positions: dict[int, int] = {}
    for position, group in groups.items():
        locations = [item.position_lps for item in group if item.position_lps is not None]
        duplicate_count = len(locations) - len(set(locations))
        if duplicate_count:
            duplicate_positions[position] = duplicate_count
    if duplicate_positions:
        qc.append(
            _event(
                "duplicate_spatial_instances",
                "error",
                "A temporal phase contains duplicate ImagePositionPatient values",
                counts=duplicate_positions,
            )
        )

    counts = {position: len(group) for position, group in groups.items()}
    if len(set(counts.values())) != 1:
        qc.append(
            _event(
                "phase_instance_count_mismatch",
                "error",
                "DCE temporal positions contain different instance counts",
                counts=counts,
            )
        )

    representative_times: dict[int, float] = {}
    acquisition_strings: dict[int, str] = {}
    for position, group in groups.items():
        parsed = [
            parsed_time
            for item in group
            if (parsed_time := parse_dicom_time(item.acquisition_time)) is not None
        ]
        if parsed:
            representative_times[position] = median(parsed)
            acquisition_strings[position] = next(
                item.acquisition_time
                for item in group
                if parse_dicom_time(item.acquisition_time) is not None
            )  # type: ignore[assignment]
    order = (
        _chronological_order(representative_times)
        if len(representative_times) == len(groups)
        else None
    )
    order_evidence: str
    if order is not None:
        order_evidence = "ordered_by_dicom_acquisition_time"
    else:
        positions = sorted(groups)
        contiguous = positions == list(range(positions[0], positions[-1] + 1))
        if derived_phase_convention_verified and contiguous and positions[0] in (0, 1):
            order = positions
            order_evidence = "ordered_by_verified_volser_temporal_position_convention"
            qc.append(
                _event(
                    "acquisition_time_not_discriminating",
                    "warning",
                    "AcquisitionTime could not order phases; verified VOLSER temporal positions were used",
                    available_times=len(representative_times),
                    observed_positions=positions,
                )
            )
        else:
            qc.append(
                _event(
                    "ambiguous_phase_time_order",
                    "error",
                    "Temporal positions lack a unique acquisition-time order and no verified convention applies",
                    available_times=len(representative_times),
                    observed_positions=positions,
                )
            )
            return PhaseAssessment({}, False, "dicom_temporal_position", tuple(qc))

    bolus_times = [
        parsed
        for item in items
        if (parsed := parse_dicom_time(item.contrast_bolus_start_time)) is not None
    ]
    role_positions: dict[str, int]
    evidence = [
        "grouped_by_dicom_temporal_position",
        order_evidence,
        "matched_patient_geometry",
    ]
    reliability = "candidate"
    if bolus_times and len(representative_times) == len(groups):
        bolus_time = median(bolus_times)
        before = [p for p in order if representative_times[p] < bolus_time]
        after = [p for p in order if representative_times[p] >= bolus_time]
        if before and len(after) >= 2:
            role_positions = {"pre": before[-1], "early": after[0], "late": after[-1]}
            evidence.append("ordered_against_contrast_bolus_start_time")
            reliability = "high"
        else:
            qc.append(
                _event(
                    "bolus_time_inconsistent",
                    "error",
                    "Bolus time does not separate one pre and two post phases",
                )
            )
            role_positions = {"pre": order[0], "early": order[1], "late": order[-1]}
    else:
        role_positions = {"pre": order[0], "early": order[1], "late": order[-1]}
        if derived_phase_convention_verified:
            evidence.append("verified_derived_first_second_last_convention")
            reliability = "medium"
            if bolus_times:
                qc.append(
                    _event(
                        "bolus_comparison_unavailable",
                        "warning",
                        "Bolus time exists but some phases lack usable acquisition times; "
                        "the verified derived temporal-position convention was used",
                    )
                )
        else:
            qc.append(
                _event(
                    "phase_semantics_unverified",
                    "error",
                    "No bolus time is present and the derived phase convention is unverified",
                )
            )

    phases = {
        role: PhaseRef(
            role=role,
            series_uid=series.header.series_uid,
            series_path=str(series.path.resolve()),
            source_kind="dicom_temporal_position",
            temporal_position=position,
            acquisition_time=acquisition_strings.get(position),
            instance_count=len(groups[position]),
            reliability=reliability,
            evidence=tuple(evidence),
            geometry=geometries[position],
        )
        for role, position in role_positions.items()
    }
    errors = any(event.severity == "error" for event in qc)
    return PhaseAssessment(phases, not errors, "dicom_temporal_position", tuple(qc))


def is_volser_original_dce(series: SeriesSummary) -> bool:
    description = series.header.series_description.lower().replace(":", " ")
    tokens = set(description.split())
    return (
        series.header.modality == "MR"
        and "volser" in tokens
        and "original" in tokens
        and "dce" in tokens
    )


def choose_dce_candidate(series: Iterable[SeriesSummary]) -> tuple[SeriesSummary | None, tuple[QCEvent, ...]]:
    """Choose only an unambiguous consolidated VOLSER DCE candidate."""

    candidates = [item for item in series if is_volser_original_dce(item)]
    if len(candidates) == 1:
        return candidates[0], ()
    if not candidates:
        return None, (
            _event(
                "no_consolidated_dce",
                "error",
                "No consolidated VOLSER original DCE series was found",
            ),
        )
    return None, (
        _event(
            "duplicate_consolidated_dce",
            "error",
            "More than one consolidated VOLSER original DCE series was found",
            series_uids=[item.header.series_uid for item in candidates],
        ),
    )
