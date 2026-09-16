from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

from ispy2_symmflow.models.baselines import (
    CopySourceBaseline,
    DeterministicImagePredictor,
    DeterministicLatentObjective,
    DeterministicLatentPredictor,
    UnidirectionalConditionalFMObjective,
    UnidirectionalConditionalFMUNet,
    build_deterministic_baseline_from_config,
    build_unidirectional_cfm_from_config,
    copy_source_baseline,
    make_unidirectional_cfm_path,
)
from ispy2_symmflow.models.conditioning import (
    CategoricalField,
    ConditionSchema,
    NumericField,
    StructuredConditionEncoder,
)
from ispy2_symmflow.models.velocity import JointVelocityUNetConfig


class DeterministicBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.timesteps: torch.Tensor | None = None
        self.context: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        self.timesteps = timesteps
        self.context = context
        effect = context.mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
        return self.scale * x + effect


class OneWayBackbone(nn.Module):
    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.model_input: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        self.model_input = x
        self.timesteps = timesteps
        effect = context.mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
        return (
            self.scale * x[:, : self.latent_channels]
            + 0.1 * x[:, self.latent_channels :]
            + effect
        )


class TrackingIdentityAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoded_with_normalization = False
        self.decoded_with_denormalization = False

    def encode(
        self, image: torch.Tensor, *, normalize: bool = True
    ) -> torch.Tensor:
        self.encoded_with_normalization = normalize
        return image

    def decode(
        self, latent: torch.Tensor, *, denormalize: bool = True
    ) -> torch.Tensor:
        self.decoded_with_denormalization = denormalize
        return latent


def _config() -> JointVelocityUNetConfig:
    return JointVelocityUNetConfig(
        latent_channels=2,
        channels=(8, 16),
        num_res_blocks=1,
        attention_levels=(False, True),
        num_head_channels=(0, 8),
        condition_dim=6,
        norm_num_groups=8,
    )


def _condition_encoder() -> StructuredConditionEncoder:
    return StructuredConditionEncoder(
        ConditionSchema(
            categorical_fields=(CategoricalField("stage_i", ("T0",)),),
            numeric_fields=(NumericField("delta_days", mean=35.0, std=5.0),),
            token_dim=6,
        )
    )


def test_deterministic_predictor_is_repeatable_and_conditioned() -> None:
    backbone = DeterministicBackbone()
    model = DeterministicLatentPredictor(_config(), backbone=backbone)
    source = torch.randn(2, 2, 2, 3, 4)
    first_context = torch.zeros(2, 3, 6)
    second_context = torch.ones(2, 3, 6)

    first = model(source, first_context)
    repeated = model(source, first_context)
    changed = model(source, second_context)

    torch.testing.assert_close(first, repeated)
    assert not torch.allclose(first, changed)
    assert backbone.timesteps is not None
    torch.testing.assert_close(backbone.timesteps, torch.zeros(2))
    assert backbone.context is second_context


def test_deterministic_objective_trains_one_target_prediction() -> None:
    model = DeterministicLatentPredictor(
        _config(), predict_residual=False, backbone=DeterministicBackbone()
    )
    source = torch.randn(2, 2, 2, 2, 2, requires_grad=True)
    target = torch.randn_like(source)
    context = torch.randn(2, 2, 6, requires_grad=True)

    loss = DeterministicLatentObjective("mse")(model, source, target, context)
    loss.total.backward()

    assert loss.total.ndim == 0
    assert source.grad is not None and torch.count_nonzero(source.grad) > 0
    assert context.grad is not None and torch.count_nonzero(context.grad) > 0


def test_image_predictor_uses_shared_latent_scaling_and_structured_context() -> None:
    autoencoder = TrackingIdentityAutoencoder()
    latent_predictor = DeterministicLatentPredictor(
        _config(), predict_residual=True, backbone=DeterministicBackbone()
    )
    model = DeterministicImagePredictor(
        autoencoder, _condition_encoder(), latent_predictor
    )
    image = torch.randn(2, 2, 2, 2, 2)

    output = model(
        image,
        {"stage_i": ["T0", "T0"], "delta_days": [30.0, 40.0]},
    )

    assert output.shape == image.shape
    assert autoencoder.encoded_with_normalization
    assert autoencoder.decoded_with_denormalization
    assert latent_predictor.backbone.context is not None
    assert latent_predictor.backbone.context.shape == (2, 2, 6)


@pytest.mark.parametrize("sigma", [0.0, 0.05])
def test_unidirectional_cfm_path_endpoints_and_derivative(sigma: float) -> None:
    target = torch.tensor([[[[[2.0]]]]])
    noise = torch.tensor([[[[[-1.0]]]]])
    at_zero = make_unidirectional_cfm_path(
        target, torch.tensor([0.0]), noise=noise, sigma_min=sigma
    )
    at_one = make_unidirectional_cfm_path(
        target, torch.tensor([1.0]), noise=noise, sigma_min=sigma
    )

    torch.testing.assert_close(at_zero.state, noise)
    torch.testing.assert_close(at_one.state, target + sigma * noise)
    tau = torch.tensor([0.37], requires_grad=True)
    middle = make_unidirectional_cfm_path(
        target, tau, noise=noise, sigma_min=sigma
    )
    derivative = torch.autograd.grad(middle.state.sum(), tau)[0]
    torch.testing.assert_close(
        derivative, middle.target_velocity.sum().reshape_as(derivative)
    )


def test_one_way_unet_uses_source_as_condition_but_outputs_one_branch() -> None:
    backbone = OneWayBackbone(latent_channels=2)
    model = UnidirectionalConditionalFMUNet(_config(), backbone=backbone)
    target_state = torch.full((1, 2, 2, 2, 2), 3.0)
    source = torch.full_like(target_state, -4.0)
    tau = torch.tensor([[0.375]])

    velocity = model(target_state, source, tau, torch.zeros(1, 2, 6))

    assert velocity.shape == target_state.shape
    assert backbone.model_input is not None
    torch.testing.assert_close(backbone.model_input[:, :2], target_state)
    torch.testing.assert_close(backbone.model_input[:, 2:], source)
    assert backbone.timesteps is not None
    torch.testing.assert_close(backbone.timesteps, torch.tensor([0.375]))
    assert not hasattr(model, "sample_backward")


def test_one_way_objective_is_single_branch_and_differentiable() -> None:
    model = UnidirectionalConditionalFMUNet(
        _config(), backbone=OneWayBackbone(latent_channels=2)
    )
    source = torch.randn(2, 2, 2, 2, 2, requires_grad=True)
    target = torch.randn_like(source)
    noise = torch.randn_like(source)
    context = torch.randn(2, 3, 6, requires_grad=True)
    objective = UnidirectionalConditionalFMObjective(sigma_min=0.0)

    loss = objective(
        model,
        source,
        target,
        context,
        tau=torch.tensor([0.2, 0.8]),
        noise=noise,
    )
    loss.total.backward()

    assert loss.total.ndim == 0
    assert source.grad is not None and torch.count_nonzero(source.grad) > 0
    assert context.grad is not None and torch.count_nonzero(context.grad) > 0


def test_copy_source_baseline_is_exact_and_does_not_alias_input() -> None:
    source = torch.randn(1, 3, 2, 3, 4)
    module_output = CopySourceBaseline()(source)
    function_output = copy_source_baseline(source)

    torch.testing.assert_close(module_output, source)
    torch.testing.assert_close(function_output, source)
    assert module_output.data_ptr() != source.data_ptr()
    assert function_output.data_ptr() != source.data_ptr()


def test_baseline_factories_can_share_main_velocity_architecture_config() -> None:
    config = {
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
        }
    }
    deterministic = build_deterministic_baseline_from_config(
        config, backbone=DeterministicBackbone()
    )
    one_way = build_unidirectional_cfm_from_config(
        config, backbone=OneWayBackbone(2)
    )

    assert deterministic.config.channels == one_way.config.channels == (8, 16)
    assert deterministic.config.condition_dim == one_way.config.condition_dim == 6


@pytest.mark.skipif(
    importlib.util.find_spec("monai") is None,
    reason="MONAI train dependency is not installed",
)
def test_real_monai_baselines_are_3d_and_have_distinct_outputs() -> None:
    deterministic = DeterministicLatentPredictor(_config()).eval()
    one_way = UnidirectionalConditionalFMUNet(_config()).eval()
    latent = torch.randn(1, 2, 8, 8, 8)
    context = torch.randn(1, 2, 6)

    with torch.no_grad():
        deterministic_output = deterministic(latent, context)
        velocity = one_way(latent, latent, torch.tensor([0.4]), context)

    assert deterministic_output.shape == latent.shape
    assert velocity.shape == latent.shape
