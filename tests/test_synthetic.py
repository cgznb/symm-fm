from __future__ import annotations

from datetime import date

import numpy as np

from ispy2_symmflow.data.manifest import build_pair_manifest, read_visit_manifest
from ispy2_symmflow.data.synthetic import create_synthetic_dataset
from ispy2_symmflow.training.datasets import read_jsonl


def test_synthetic_dataset_is_explicit_and_patient_split_safe(tmp_path) -> None:
    result = create_synthetic_dataset(tmp_path, patient_count=5, image_shape=(3, 8, 8, 8))
    visits = read_jsonl(result["visit_manifest"])
    pairs = read_jsonl(result["pair_manifest"])
    assert result["synthetic"] is True
    assert len(visits) == 10 and len(pairs) == 5
    assignments = {}
    for visit in visits:
        assignments.setdefault(visit["patient_id"], set()).add(visit["split"])
        assert visit["relative_date_verified"] is True
        assert visit["date_source"] == "synthetic_generator_known_date"
        date.fromisoformat(visit["study_date"])
        with np.load(visit["prepared_path"], allow_pickle=False) as archive:
            assert archive["image"].shape == (3, 8, 8, 8)
            assert bool(archive["provenance_json"])
            assert str(archive["study_date"].item()) == visit["study_date"]
            assert bool(archive["relative_date_verified"].item()) is True
    assert all(len(splits) == 1 for splits in assignments.values())


def test_synthetic_pair_manifest_is_strictly_reconstructable_from_visits(
    tmp_path,
) -> None:
    result = create_synthetic_dataset(
        tmp_path, patient_count=5, image_shape=(3, 8, 8, 8)
    )
    visits = read_visit_manifest(result["visit_manifest"])
    assignments = {visit.patient_id: str(visit.split) for visit in visits}
    rebuilt = build_pair_manifest(
        visits,
        assignments,
        earlier_stage="T0",
        later_stage="T1",
        require_three_phase=False,
    )
    stored = read_jsonl(result["pair_manifest"])

    assert len(rebuilt) == len(stored) == 5
    for index, (expected_pair, observed_pair) in enumerate(
        zip(rebuilt, stored, strict=True)
    ):
        expected = expected_pair.to_dict()
        assert {key: observed_pair[key] for key in expected} == expected
        assert observed_pair["delta_days"] == 35 + index
        assert observed_pair["observed_delta_days"] == 35 + index
        assert observed_pair["interval_missing"] is False
        assert (
            observed_pair["interval_source"]
            == "verified_relative_dicom_study_date"
        )
