from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import numpy as np

from ispy2_symmflow.data.schema import PhaseRef, VisitRecord
from ispy2_symmflow.preprocessing import NPZ_KEYS, PreprocessConfig, prepare_dataset
from ispy2_symmflow.preprocessing.intensity import fit_intensity_stats
from ispy2_symmflow.preprocessing.loader import LoadedVisit
from ispy2_symmflow.preprocessing.pipeline import preprocess_source


def _visit(patient: str, stage: str, date: str) -> VisitRecord:
    phases = {
        role: PhaseRef(
            role=role,
            series_uid=f"{patient}.{stage}",
            series_path="/unused",
            source_kind="test",
            temporal_position=index,
            acquisition_time=None,
            instance_count=4,
            reliability="high",
        )
        for index, role in enumerate(("pre", "early", "late"))
    }
    return VisitRecord(
        visit_id=f"ISPY2:{patient}.{stage}",
        patient_id=patient,
        collection="ISPY2",
        study_uid=f"{patient}.{stage}",
        visit_stage=stage,
        study_date=date,
        date_source="test",
        relative_date_verified=True,
        phase_paths=phases,
    )


def _loader(visit: VisitRecord, roles: tuple[str, ...]) -> LoadedVisit:
    offset = 10 if visit.visit_stage == "T1" else 0
    image = np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6) + offset
    return LoadedVisit(
        image=image,
        affine_lps=np.eye(4, dtype=np.float64),
        spacing_dhw=(1.0, 1.0, 1.0),
        phase_roles=tuple(roles),
        source_series_uids=tuple(visit.phase_paths[role].series_uid for role in roles),
        source_temporal_positions=tuple(
            visit.phase_paths[role].temporal_position or 0 for role in roles
        ),
    )


def test_validation_values_do_not_affect_training_statistics() -> None:
    splits = {"train": "train", "val": "val"}
    stats = fit_intensity_stats(
        [
            ("train", np.asarray([1.0, 2.0, 3.0], dtype=np.float32).reshape(1, 1, 3)),
            ("val", np.asarray([1e9], dtype=np.float32).reshape(1, 1, 1)),
        ],
        splits,
        lower_percentile=0,
        upper_percentile=100,
    )
    assert stats.lower == 1.0
    assert stats.upper == 3.0
    assert stats.fit_patient_count == 1


def test_preprocess_source_has_no_target_argument() -> None:
    assert "target" not in inspect.signature(preprocess_source).parameters


def test_prepare_dataset_writes_npz_and_resolvable_pair_contract(tmp_path: Path) -> None:
    visits = [
        _visit("P1", "T0", "2020-01-01"),
        _visit("P1", "T1", "2020-02-05"),
        _visit("P2", "T0", "2020-01-02"),
        _visit("P2", "T1", "2020-02-06"),
    ]
    result = prepare_dataset(
        visits,
        tmp_path,
        split_by_patient={"P1": "train", "P2": "test"},
        config=PreprocessConfig(
            target_spacing_dhw=None,
            target_axis_codes=None,
            output_shape_dhw=(4, 5, 6),
            lower_percentile=0,
            upper_percentile=100,
        ),
        volume_loader=_loader,
    )
    assert len(result.visits) == 4
    assert len(result.pairs) == 2
    pair = next(item for item in result.pairs if item.patient_id == "P1")
    assert Path(pair.earlier_prepared_path or "").is_file()
    assert Path(pair.later_prepared_path or "").is_file()
    with np.load(pair.earlier_prepared_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(NPZ_KEYS)
        assert archive["image"].shape == (3, 4, 5, 6)
        provenance = json.loads(str(archive["provenance_json"].item()))
        assert provenance["target_content_used"] is False
        assert provenance["normalization_scope"] == "global_train_patients_shared_phases"


def test_prepare_preserves_audited_splits_and_flags_pair_grid_mismatch(tmp_path: Path) -> None:
    visits = [
        replace(_visit("P1", "T0", "2020-01-01"), split="train"),
        replace(_visit("P1", "T1", "2020-02-05"), split="train"),
        replace(_visit("P2", "T0", "2020-01-02"), split="val"),
        replace(_visit("P2", "T1", "2020-02-06"), split="val"),
    ]

    def shifted_loader(visit: VisitRecord, roles: tuple[str, ...]) -> LoadedVisit:
        loaded = _loader(visit, roles)
        affine = loaded.affine_lps.copy()
        if visit.patient_id == "P1" and visit.visit_stage == "T1":
            affine[0, 3] = 10.0
        return replace(loaded, affine_lps=affine)

    result = prepare_dataset(
        visits,
        tmp_path,
        split_seed=999,
        config=PreprocessConfig(
            target_spacing_dhw=None,
            target_axis_codes=None,
            output_shape_dhw=(4, 5, 6),
            lower_percentile=0,
            upper_percentile=100,
        ),
        volume_loader=shifted_loader,
    )
    assert {visit.patient_id: visit.split for visit in result.visits} == {
        "P1": "train",
        "P2": "val",
    }
    mismatched = next(pair for pair in result.pairs if pair.patient_id == "P1")
    assert any(event.code == "pair_physical_grid_mismatch" for event in mismatched.qc)
    assert result.invalid_pair_ids == (mismatched.pair_id,)
