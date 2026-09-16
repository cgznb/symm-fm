from __future__ import annotations

import pytest

pytest.importorskip("torch")

from ispy2_symmflow.training.schema import (
    fit_condition_schema,
    validate_sampling_condition_availability,
)


def _condition_config() -> dict[str, object]:
    return {
        "token_dim": 16,
        "categorical": {
            "treatment_arm": [],
            "hr_status": ["negative", "positive"],
            "stage_i": ["T0"],
            "stage_j": ["T1"],
            "interval_missing": ["no", "yes"],
            "interval_source": [],
        },
        "numeric": ["age", "delta_days"],
    }


def test_schema_statistics_ignore_validation_values() -> None:
    records = [
        {
            "pair_id": "a",
            "split": "train",
            "earlier_stage": "T0",
            "later_stage": "T1",
            "delta_days": 30,
            "baseline_clinical": {"age": 40},
            "treatment": {"treatment_arm": "A"},
        },
        {
            "pair_id": "b",
            "split": "train",
            "earlier_stage": "T0",
            "later_stage": "T1",
            "delta_days": 40,
            "baseline_clinical": {"age": 60},
            "treatment": {"treatment_arm": "B"},
        },
        {
            "pair_id": "c",
            "split": "val",
            "earlier_stage": "T0",
            "later_stage": "T1",
            "delta_days": 1000,
            "baseline_clinical": {"age": 1000},
            "treatment": {"treatment_arm": "C"},
        },
    ]
    schema, provenance = fit_condition_schema(
        records,
        _condition_config(),
    )
    assert schema.categorical_fields[0].categories == ("A", "B")
    assert schema.numeric_fields[0].mean == pytest.approx(50)
    assert schema.numeric_fields[1].mean == pytest.approx(35)
    assert provenance["fit_split"] == "train"
    assert "hr_status" in provenance["unavailable_fields"]
    assert provenance["train_observed_coverage"]["treatment_arm"] == {
        "kind": "categorical",
        "observation_count": 2,
        "observed_categories": ["A", "B"],
    }
    assert provenance["train_observed_coverage"]["age"]["observation_count"] == 2
    interval_field = next(
        field for field in schema.categorical_fields if field.name == "interval_missing"
    )
    assert interval_field.categories == ("no", "yes")
    assert "interval_missing" not in provenance["unavailable_fields"]


def test_schema_rejects_observed_but_unconfigured_conditions() -> None:
    record = {
        "pair_id": "a",
        "split": "train",
        "earlier_stage": "T0",
        "later_stage": "T1",
        "baseline_clinical": {"age": 40, "tumor_grade": "high"},
        "treatment": {"treatment_arm": "A"},
    }

    with pytest.raises(ValueError, match="not configured: tumor_grade"):
        fit_condition_schema([record], _condition_config())


def test_configured_categorical_with_no_training_values_is_unavailable() -> None:
    record = {
        "pair_id": "a",
        "split": "train",
        "earlier_stage": "T0",
        "later_stage": "T1",
        "baseline_clinical": {"age": 40},
        "treatment": {"treatment_arm": "A"},
    }

    schema, provenance = fit_condition_schema([record], _condition_config())
    hr_field = next(field for field in schema.categorical_fields if field.name == "hr_status")
    assert hr_field.categories == ("negative", "positive")
    assert "hr_status" in provenance["unavailable_fields"]

    validate_sampling_condition_availability(
        {"hr_status": None, "delta_days": "missing"}, provenance
    )
    with pytest.raises(ValueError, match="unavailable in training: hr_status"):
        validate_sampling_condition_availability(
            {"hr_status": "positive"}, provenance
        )


def test_sampling_condition_availability_requires_checkpoint_provenance() -> None:
    with pytest.raises(ValueError, match="no valid unavailable_fields"):
        validate_sampling_condition_availability({"age": 40}, {})
