from __future__ import annotations

from ispy2_symmflow.evaluation.aggregate import aggregate_by_patient, patient_bootstrap_mean_ci


def test_repeated_pairs_are_aggregated_within_patient() -> None:
    values = aggregate_by_patient(
        [
            {"patient_id": "a", "mae": 1.0},
            {"patient_id": "a", "mae": 3.0},
            {"patient_id": "b", "mae": 4.0},
        ],
        metric="mae",
    )
    assert values == {"a": 2.0, "b": 4.0}
    interval = patient_bootstrap_mean_ci(values, samples=200, seed=3)
    assert interval["patient_count"] == 2
    assert interval["lower"] is not None and interval["upper"] is not None


def test_single_patient_does_not_claim_confidence_interval() -> None:
    result = patient_bootstrap_mean_ci({"a": 2.0})
    assert result["lower"] is None
    assert str(result["status"]).startswith("not_computed")
