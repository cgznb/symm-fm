from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.evaluation.diagnostics import (
    DiagnosticCase,
    patient_derangement,
    run_usage_diagnostics,
)


def test_patient_derangement_is_reproducible_and_has_no_fixed_points() -> None:
    patients = ["a", "b", "c", "d"]
    first = patient_derangement(patients, seed=4)
    assert first == patient_derangement(patients, seed=4)
    assert all(index != mapped for index, mapped in enumerate(first))


def test_usage_diagnostics_run_source_condition_and_pair_perturbations() -> None:
    cases = [
        DiagnosticCase(
            patient_id=f"p{index}",
            source=torch.full((1, 2, 2, 2), float(index)),
            target=torch.full((1, 2, 2, 2), float(index + 1)),
            conditions={"treatment_arm": "A", "age": float(index), "delta_days": 30.0},
        )
        for index in range(3)
    ]

    def predictor(source, conditions, seed):
        age = conditions.get("age")
        effect = 0.0 if age is None else 0.1 * float(age)
        return source + 1.0 + effect

    report = run_usage_diagnostics(cases, predictor, seed=9, bootstrap_samples=50)
    assert report["patient_count"] == 3
    assert all(
        record["shuffled_source_patient_id"] != record["patient_id"]
        for record in report["records"]
    )
    assert "source_output_change_mae" in report["patient_level"]
    assert "clinical_output_change_mae" in report["patient_level"]
    assert any("causal" in warning for warning in report["warnings"])


def test_usage_diagnostics_require_multiple_unique_patients() -> None:
    case = DiagnosticCase("p", torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, 2), {})
    with pytest.raises(ValueError, match="two unique"):
        run_usage_diagnostics([case], lambda source, conditions, seed: source)
