from __future__ import annotations

import pytest
import torch
from torch import nn

from ispy2_symmflow.models.conditioning import (
    CategoricalField,
    ConditionSchema,
    NumericField,
    StructuredConditionEncoder,
    build_condition_encoder_from_config,
)


def _schema(token_dim: int = 12) -> ConditionSchema:
    return ConditionSchema(
        categorical_fields=(
            CategoricalField("treatment_arm", ("A", "B")),
            CategoricalField("stage_i", ("T0", "T1")),
            CategoricalField("stage_j", ("T1", "T2")),
        ),
        numeric_fields=(
            NumericField("age", mean=50.0, std=10.0),
            NumericField("delta_days", mean=35.0, std=7.0),
        ),
        token_dim=token_dim,
    )


def test_schema_round_trip_and_fingerprint_are_stable() -> None:
    schema = _schema()
    restored = ConditionSchema.from_dict(schema.to_dict())

    assert restored == schema
    assert restored.fingerprint == schema.fingerprint
    assert restored.field_names == (
        "treatment_arm",
        "stage_i",
        "stage_j",
        "age",
        "delta_days",
    )


@pytest.mark.parametrize("name", ["pCR", "RCB", "target_mask", "future-image"])
def test_schema_rejects_outcome_and_target_fields(name: str) -> None:
    with pytest.raises(ValueError, match="cannot be model conditions"):
        ConditionSchema(
            categorical_fields=(CategoricalField(name, ("a",)),),
            numeric_fields=(),
        )


def test_reserved_unknown_is_not_a_learned_ordered_category() -> None:
    with pytest.raises(ValueError, match="reserved category"):
        CategoricalField("arm", ("unknown", "A"))

    schema = ConditionSchema.from_dict(
        {
            "token_dim": 8,
            "categorical": {"arm": ["unknown", "A"]},
            "numeric": [],
        }
    )
    assert schema.categorical_fields[0].categories == ("A",)
    assert schema.categorical_fields[0].num_embeddings == 3


@pytest.mark.parametrize(
    "field_name",
    ["pcr_status", "RCB_class", "outcome_label", "pathology_result", "target_path"],
)
def test_outcome_and_target_derived_name_variants_are_rejected(field_name: str) -> None:
    with pytest.raises(ValueError, match="cannot be model conditions"):
        ConditionSchema(
            categorical_fields=(),
            numeric_fields=(NumericField(field_name, mean=0.0, std=1.0),),
            token_dim=8,
        )


def test_integer_categories_are_values_not_implicit_embedding_indices() -> None:
    schema = ConditionSchema(
        categorical_fields=(CategoricalField("status", ("0", "1")),),
        numeric_fields=(),
        token_dim=2,
    )
    encoder = StructuredConditionEncoder(schema)
    encoder.output_norm = nn.Identity()
    with torch.no_grad():
        encoder.field_embeddings.weight.zero_()
        encoder.type_embeddings.weight.zero_()
        encoder.categorical_embeddings["status"].weight.copy_(
            torch.tensor([[9.0, 9.0], [8.0, 8.0], [0.0, 0.0], [1.0, 1.0]])
        )

    tokens = encoder({"status": torch.tensor([0, 1])})[:, 0]
    torch.testing.assert_close(tokens, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))


def test_encoder_emits_typed_tokens_and_handles_missing_values() -> None:
    torch.manual_seed(4)
    encoder = StructuredConditionEncoder(_schema())
    tokens = encoder(
        {
            "treatment_arm": ["A", "not-in-training", None],
            "stage_i": ["T0", "T0", "T1"],
            "stage_j": ["T1", "T2", "T2"],
            "age": [60.0, None, float("nan")],
            "delta_days": torch.tensor([[28.0], [35.0], [42.0]]),
        }
    )

    assert tokens.shape == (3, 5, 12)
    assert tokens.dtype == torch.float32
    assert torch.isfinite(tokens).all()
    assert not torch.allclose(tokens[0, 0], tokens[1, 0])
    assert not torch.allclose(tokens[1, 0], tokens[2, 0])
    assert not torch.allclose(tokens[0, 3], tokens[1, 3])


def test_unknown_and_missing_category_have_distinct_embedding_rows() -> None:
    schema = ConditionSchema(
        categorical_fields=(CategoricalField("arm", ("A",)),),
        numeric_fields=(),
        token_dim=2,
    )
    encoder = StructuredConditionEncoder(schema)
    encoder.output_norm = nn.Identity()
    with torch.no_grad():
        encoder.field_embeddings.weight.zero_()
        encoder.type_embeddings.weight.zero_()
        encoder.categorical_embeddings["arm"].weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        )

    tokens = encoder({"arm": [None, "unseen", "A"]})[:, 0]

    torch.testing.assert_close(tokens[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(tokens[1], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(tokens[2], torch.tensor([2.0, 2.0]))


def test_condition_changes_are_differentiable() -> None:
    encoder = StructuredConditionEncoder(_schema(token_dim=8))
    tokens = encoder(
        {
            "treatment_arm": ["A", "B"],
            "stage_i": ["T0", "T0"],
            "stage_j": ["T1", "T1"],
            "age": [45.0, 55.0],
            "delta_days": [30.0, 40.0],
        }
    )
    weights = torch.arange(tokens.numel(), dtype=tokens.dtype).reshape_as(tokens)
    (tokens * weights).sum().backward()

    arm_gradient = encoder.categorical_embeddings["treatment_arm"].weight.grad
    numeric_gradient = encoder.numeric_projections["age"][0].weight.grad
    assert arm_gradient is not None and torch.count_nonzero(arm_gradient) > 0
    assert numeric_gradient is not None and torch.count_nonzero(numeric_gradient) > 0


def test_strict_schema_rejects_extra_fields() -> None:
    encoder = StructuredConditionEncoder(_schema())
    with pytest.raises(KeyError, match="not present"):
        encoder({"pcr": [0, 1]}, batch_size=2)


def test_numeric_fields_need_fitted_training_statistics() -> None:
    with pytest.raises(ValueError, match="training-set mean/std"):
        ConditionSchema.from_dict(
            {"categorical": {"stage_i": ["T0"]}, "numeric": ["age"]}
        )


def test_factory_accepts_nested_project_condition_config() -> None:
    encoder = build_condition_encoder_from_config(
        {
            "conditions": {
                "token_dim": 10,
                "categorical": {"stage_i": ["T0", "T1"]},
                "numeric": ["age"],
                "numeric_statistics": {"age": {"mean": 52.0, "std": 9.0}},
            }
        }
    )
    assert encoder.output_dim == 10
    assert encoder.num_tokens == 2


def test_infinite_numeric_value_is_an_error() -> None:
    encoder = StructuredConditionEncoder(_schema())
    with pytest.raises(ValueError, match="infinite"):
        encoder(
            {
                "treatment_arm": ["A"],
                "stage_i": ["T0"],
                "stage_j": ["T1"],
                "age": [float("inf")],
                "delta_days": [35.0],
            }
        )
