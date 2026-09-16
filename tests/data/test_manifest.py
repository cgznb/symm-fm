from __future__ import annotations

from pathlib import Path

from ispy2_symmflow.data.manifest import (
    build_pair_manifest,
    read_visit_manifest,
    write_jsonl,
)
from ispy2_symmflow.data.schema import Geometry, PhaseRef, VisitRecord


def _visit(
    patient: str,
    stage: str,
    study_date: str,
    *,
    verified_date: bool,
    prepared_path: str | None = None,
) -> VisitRecord:
    geometry = Geometry(
        shape_dhw=(2, 3, 4),
        spacing_dhw=(1.0, 1.0, 1.0),
        orientation_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        origin_lps=(0.0, 0.0, 0.0),
        affine_lps=(
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    phases = {
        role: PhaseRef(
            role=role,
            series_uid=f"{patient}.{stage}.series",
            series_path=f"/data/{patient}/{stage}",
            source_kind="dicom_temporal_position",
            temporal_position=index,
            acquisition_time=f"100{index}00",
            instance_count=2,
            reliability="high",
            evidence=("test",),
            geometry=geometry,
        )
        for index, role in enumerate(("pre", "early", "late"))
    }
    return VisitRecord(
        visit_id=f"ISPY2:{patient}.{stage}",
        patient_id=patient,
        collection="ISPY2",
        study_uid=f"{patient}.{stage}",
        visit_stage=stage,
        study_date=study_date,
        date_source="dicom_study_date_deidentified",
        relative_date_verified=verified_date,
        phase_paths=phases,
        baseline_clinical={"age": 50},
        treatment={"treatment_arm": "A"},
        split="train" if prepared_path else None,
        prepared_path=prepared_path,
    )


def test_visit_jsonl_round_trip(tmp_path: Path) -> None:
    source = _visit("P1", "T0", "2020-01-01", verified_date=False)
    path = tmp_path / "visits.jsonl"
    write_jsonl([source], path)
    loaded = read_visit_manifest(path)[0]
    assert loaded.visit_id == source.visit_id
    assert loaded.phase_paths["early"].geometry == source.phase_paths["early"].geometry
    assert loaded.baseline_clinical == {"age": 50}


def test_pair_keeps_fixed_semantics_and_direct_paths() -> None:
    earlier = _visit(
        "P1", "T0", "2020-01-01", verified_date=False, prepared_path="/p1-t0.npz"
    )
    later = _visit(
        "P1", "T1", "2020-02-05", verified_date=False, prepared_path="/p1-t1.npz"
    )
    pair = build_pair_manifest([later, earlier], {"P1": "train"})[0]
    assert pair.earlier_visit_id == earlier.visit_id
    assert pair.later_visit_id == later.visit_id
    assert pair.earlier_prepared_path == "/p1-t0.npz"
    assert pair.later_prepared_path == "/p1-t1.npz"
    assert pair.observed_delta_days == 35
    assert pair.delta_days is None
    assert pair.interval_missing


def test_verified_relative_dates_enable_delta_days() -> None:
    earlier = _visit("P1", "T0", "2020-01-01", verified_date=True)
    later = _visit("P1", "T1", "2020-02-05", verified_date=True)
    pair = build_pair_manifest([earlier, later], {"P1": "val"})[0]
    assert pair.delta_days == 35
    assert not pair.interval_missing
