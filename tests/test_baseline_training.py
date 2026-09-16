from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.models.baselines import (
    DeterministicLatentPredictor,
    UnidirectionalConditionalFMUNet,
)
from ispy2_symmflow.models.conditioning import (
    CategoricalField,
    ConditionSchema,
    StructuredConditionEncoder,
)
from ispy2_symmflow.models.velocity import JointVelocityUNetConfig
from ispy2_symmflow.training.baselines import validate_baseline


class DeterministicBackbone(torch.nn.Module):
    def forward(self, x, timesteps, context):
        return torch.zeros_like(x)


class OneWayBackbone(torch.nn.Module):
    def forward(self, x, timesteps, context):
        return torch.zeros_like(x[:, :1])


def _architecture() -> JointVelocityUNetConfig:
    return JointVelocityUNetConfig(
        latent_channels=1,
        channels=(8, 16),
        num_res_blocks=1,
        attention_levels=(False, True),
        num_head_channels=(0, 8),
        condition_dim=4,
        norm_num_groups=8,
    )


def _encoder() -> StructuredConditionEncoder:
    return StructuredConditionEncoder(
        ConditionSchema(
            categorical_fields=(CategoricalField("stage_i", ("T0",)),),
            numeric_fields=(),
            token_dim=4,
        )
    )


def _batch():
    return {
        "earlier_latent": torch.zeros(2, 1, 2, 2, 2),
        "later_latent": torch.ones(2, 1, 2, 2, 2),
        "conditions": {"stage_i": ["T0", "T0"]},
    }


def test_deterministic_and_cfm_validation_are_repeatable() -> None:
    deterministic = DeterministicLatentPredictor(
        _architecture(), predict_residual=False, backbone=DeterministicBackbone()
    )
    deterministic_metrics = validate_baseline(
        "deterministic",
        deterministic,
        _encoder(),
        [_batch()],
        device=torch.device("cpu"),
        precision="fp32",
        sigma_min=0.0,
        seed=4,
    )
    assert deterministic_metrics["loss"] == pytest.approx(1.0)

    cfm = UnidirectionalConditionalFMUNet(
        _architecture(), backbone=OneWayBackbone()
    )
    first = validate_baseline(
        "unidirectional_cfm",
        cfm,
        _encoder(),
        [_batch()],
        device=torch.device("cpu"),
        precision="fp32",
        sigma_min=0.0,
        seed=9,
    )
    repeated = validate_baseline(
        "unidirectional_cfm",
        cfm,
        _encoder(),
        [_batch()],
        device=torch.device("cpu"),
        precision="fp32",
        sigma_min=0.0,
        seed=9,
    )
    assert first == repeated


def test_baseline_validation_does_not_filter_conditions_outside_schema() -> None:
    model = DeterministicLatentPredictor(
        _architecture(), predict_residual=False, backbone=DeterministicBackbone()
    )
    batch = _batch()
    batch["conditions"]["unconfigured"] = [1, 2]

    with pytest.raises(KeyError, match="not present"):
        validate_baseline(
            "deterministic",
            model,
            _encoder(),
            [batch],
            device=torch.device("cpu"),
            precision="fp32",
            sigma_min=0.0,
            seed=4,
        )
