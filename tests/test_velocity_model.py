from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import nn

from ispy2_symmflow.models.conditioning import (
    CategoricalField,
    ConditionSchema,
    NumericField,
    StructuredConditionEncoder,
)
from ispy2_symmflow.models.velocity import (
    ConditionalVelocityModel,
    JointVelocityUNet,
    JointVelocityUNetConfig,
    build_velocity_model_from_config,
    join_branches,
    split_branches,
)


class RecordingBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_scale = nn.Parameter(torch.tensor(1.5))
        self.last_timesteps: torch.Tensor | None = None
        self.last_context: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        self.last_timesteps = timesteps
        self.last_context = context
        context_effect = context.mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
        time_effect = timesteps.reshape(-1, 1, 1, 1, 1)
        return self.input_scale * x + context_effect + time_effect


def _config(condition_dim: int = 6) -> JointVelocityUNetConfig:
    return JointVelocityUNetConfig(
        latent_channels=2,
        channels=(8, 16),
        num_res_blocks=1,
        attention_levels=(False, True),
        num_head_channels=(0, 8),
        condition_dim=condition_dim,
        norm_num_groups=8,
    )


def test_join_and_split_preserve_later_earlier_channel_order() -> None:
    later = torch.full((1, 2, 2, 2, 2), 3.0)
    earlier = torch.full((1, 2, 2, 2, 2), -4.0)

    joint = join_branches(later, earlier)
    restored_later, restored_earlier = split_branches(joint, latent_channels=2)

    torch.testing.assert_close(joint[:, :2], later)
    torch.testing.assert_close(joint[:, 2:], earlier)
    torch.testing.assert_close(restored_later, later)
    torch.testing.assert_close(restored_earlier, earlier)


def test_joint_forward_preserves_continuous_tau_and_shape() -> None:
    backbone = RecordingBackbone()
    model = JointVelocityUNet(_config(), backbone=backbone)
    joint = torch.randn(2, 4, 3, 4, 5)
    tau = torch.tensor([[0.125], [0.875]], dtype=torch.float32)
    context = torch.randn(2, 4, 6)

    velocity = model(joint, tau, context)

    assert velocity.shape == joint.shape
    assert backbone.last_timesteps is not None
    torch.testing.assert_close(
        backbone.last_timesteps, torch.tensor([0.125, 0.875])
    )
    assert backbone.last_context is context


def test_condition_tokens_change_velocity_and_receive_gradients() -> None:
    model = JointVelocityUNet(_config(), backbone=RecordingBackbone())
    joint = torch.randn(2, 4, 2, 2, 2, requires_grad=True)
    tau = torch.tensor([0.2, 0.8])
    first_context = torch.zeros(2, 3, 6, requires_grad=True)
    second_context = torch.ones(2, 3, 6)

    first = model(joint, tau, first_context)
    second = model(joint.detach(), tau, second_context)

    assert not torch.allclose(first.detach(), second)
    first.square().mean().backward()
    assert joint.grad is not None and torch.count_nonzero(joint.grad) > 0
    assert first_context.grad is not None
    assert torch.count_nonzero(first_context.grad) > 0


def test_both_velocity_branches_have_gradients_in_one_forward() -> None:
    model = JointVelocityUNet(_config(), backbone=RecordingBackbone())
    later = torch.randn(2, 2, 2, 2, 2, requires_grad=True)
    earlier = torch.randn(2, 2, 2, 2, 2, requires_grad=True)
    context = torch.randn(2, 2, 6)

    velocity_later, velocity_earlier = model.forward_branches(
        later, earlier, torch.tensor([0.3, 0.7]), context
    )
    (velocity_later.square().mean() + velocity_earlier.square().mean()).backward()

    assert later.grad is not None and torch.count_nonzero(later.grad) > 0
    assert earlier.grad is not None and torch.count_nonzero(earlier.grad) > 0


def test_velocity_has_no_output_squashing() -> None:
    model = JointVelocityUNet(_config(), backbone=RecordingBackbone())
    joint = torch.full((1, 4, 2, 2, 2), 10.0)
    velocity = model(joint, torch.tensor([0.5]), torch.zeros(1, 1, 6))
    assert velocity.min() > 1.0


@pytest.mark.parametrize(
    ("tau", "error"),
    [
        (torch.tensor([0]), TypeError),
        (torch.tensor([[0.1, 0.2]]), ValueError),
        (torch.tensor([1.2]), ValueError),
    ],
)
def test_invalid_tau_is_rejected(tau: torch.Tensor, error: type[Exception]) -> None:
    model = JointVelocityUNet(_config(), backbone=RecordingBackbone())
    with pytest.raises(error):
        model(torch.randn(1, 4, 2, 2, 2), tau, torch.randn(1, 2, 6))


def test_composed_model_encodes_raw_conditions() -> None:
    schema = ConditionSchema(
        categorical_fields=(CategoricalField("stage_i", ("T0", "T1")),),
        numeric_fields=(NumericField("delta_days", mean=35.0, std=5.0),),
        token_dim=6,
    )
    encoder = StructuredConditionEncoder(schema)
    backbone = RecordingBackbone()
    model = ConditionalVelocityModel(
        encoder, JointVelocityUNet(_config(), backbone=backbone)
    )
    conditions: Mapping[str, Any] = {
        "stage_i": ["T0", "T1"],
        "delta_days": [30.0, None],
    }

    result = model(
        torch.randn(2, 4, 2, 2, 2),
        torch.tensor([0.25, 0.75]),
        conditions,
    )

    assert result.shape == (2, 4, 2, 2, 2)
    assert backbone.last_context is not None
    assert backbone.last_context.shape == (2, 2, 6)


def test_factory_accepts_cross_attention_alias_and_training_config() -> None:
    model = build_velocity_model_from_config(
        {
            "velocity": {
                "spatial_dims": 3,
                "latent_channels": 2,
                "channels": [8, 16],
                "num_res_blocks": 1,
                "attention_levels": [False, True],
                "num_head_channels": [0, 8],
                "norm_num_groups": 8,
                "with_conditioning": True,
                "cross_attention_dim": 6,
                "batch_size": 2,
            }
        },
        backbone=RecordingBackbone(),
    )
    assert model.config.condition_dim == 6
    assert model.config.joint_channels == 4


@pytest.mark.skipif(
    importlib.util.find_spec("monai") is None,
    reason="MONAI train dependency is not installed",
)
def test_real_monai_joint_velocity_unet_3d_shape() -> None:
    model = JointVelocityUNet(_config()).eval()
    joint = torch.randn(1, 4, 8, 8, 8)
    tau = torch.tensor([0.375])
    context = torch.randn(1, 3, 6)

    with torch.no_grad():
        velocity = model(joint, tau, context)

    assert velocity.shape == joint.shape
