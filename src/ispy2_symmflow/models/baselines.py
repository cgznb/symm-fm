"""Explicit non-SymmFlow baselines for controlled comparisons.

These models share the 3D MONAI backbone family and structured condition tokens
with the main model. They do not implement the two-branch SymmFlow objective and
must not be used as substitutes for it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .conditioning import StructuredConditionEncoder
from .velocity import JointVelocityUNetConfig, _load_monai_unet_class


CFM_BASE_DISTRIBUTIONS = ("standard_normal", "source_gaussian")


def _build_conditioned_unet(
    config: JointVelocityUNetConfig,
    *,
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    unet_class = _load_monai_unet_class()
    return unet_class(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        num_res_blocks=config.num_res_blocks,
        channels=config.channels,
        attention_levels=config.attention_levels,
        norm_num_groups=config.norm_num_groups,
        norm_eps=config.norm_eps,
        resblock_updown=config.resblock_updown,
        num_head_channels=config.num_head_channels,
        with_conditioning=True,
        transformer_num_layers=config.transformer_num_layers,
        cross_attention_dim=config.condition_dim,
        upcast_attention=config.upcast_attention,
        dropout_cattn=config.dropout_cattn,
        include_fc=config.include_fc,
        use_combined_linear=config.use_combined_linear,
        use_flash_attention=config.use_flash_attention,
    )


def _validate_latent(latent: Tensor, latent_channels: int, *, name: str) -> None:
    if latent.ndim != 5:
        raise ValueError(f"{name} must have shape [B, C, D, H, W]")
    if latent.shape[1] != latent_channels:
        raise ValueError(
            f"{name} must contain {latent_channels} channels, got {latent.shape[1]}"
        )
    if not latent.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def _validate_context(context: Tensor, reference: Tensor, condition_dim: int) -> None:
    if context.ndim != 3 or context.shape[0] != reference.shape[0]:
        raise ValueError("condition tokens must have shape [B, L, condition_dim]")
    if context.shape[1] == 0 or context.shape[2] != condition_dim:
        raise ValueError(
            f"condition tokens require L > 0 and width {condition_dim}"
        )
    if context.device != reference.device:
        raise ValueError("condition tokens and latent tensors must share a device")
    if not context.is_floating_point():
        raise TypeError("condition tokens must use a floating-point dtype")


def _continuous_tau(tau: Tensor, reference: Tensor) -> Tensor:
    if not isinstance(tau, Tensor) or not tau.is_floating_point():
        raise TypeError("tau must be a floating-point tensor with shape [B] or [B, 1]")
    if tau.ndim == 2 and tau.shape[1] == 1:
        tau = tau[:, 0]
    if tau.ndim != 1 or tau.shape[0] != reference.shape[0]:
        raise ValueError("tau must have shape [B] or [B, 1]")
    if not torch.all(torch.isfinite(tau)):
        raise ValueError("tau must be finite")
    if torch.any(tau < -1e-6) or torch.any(tau > 1.0 + 1e-6):
        raise ValueError("tau must lie in [0, 1]")
    return tau.to(device=reference.device)


class DeterministicLatentPredictor(nn.Module):
    """Deterministic source-to-target latent regression baseline.

    The MONAI U-Net requires a timestep input, so this baseline always supplies a
    fixed architectural timestep. It is not a flow time and all real interval
    information remains in the structured condition tokens.
    """

    def __init__(
        self,
        config: JointVelocityUNetConfig,
        *,
        predict_residual: bool = True,
        fixed_timestep: float = 0.0,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not math.isfinite(fixed_timestep):
            raise ValueError("fixed_timestep must be finite")
        self.config = config
        self.predict_residual = bool(predict_residual)
        self.fixed_timestep = float(fixed_timestep)
        self.backbone = (
            backbone
            if backbone is not None
            else _build_conditioned_unet(
                config,
                in_channels=config.latent_channels,
                out_channels=config.latent_channels,
            )
        )

    def forward(self, source_latent: Tensor, condition_tokens: Tensor) -> Tensor:
        _validate_latent(
            source_latent, self.config.latent_channels, name="source_latent"
        )
        _validate_context(condition_tokens, source_latent, self.config.condition_dim)
        timesteps = torch.full(
            (source_latent.shape[0],),
            self.fixed_timestep,
            device=source_latent.device,
            dtype=source_latent.dtype,
        )
        prediction = self.backbone(
            x=source_latent,
            timesteps=timesteps,
            context=condition_tokens,
        )
        if not isinstance(prediction, Tensor) or prediction.shape != source_latent.shape:
            raise RuntimeError(
                "deterministic MONAI backbone must return the source latent shape"
            )
        return source_latent + prediction if self.predict_residual else prediction

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class DeterministicLatentLoss:
    total: Tensor


class DeterministicLatentObjective(nn.Module):
    """Supervised latent regression objective for the deterministic baseline."""

    def __init__(self, loss: str = "mse") -> None:
        super().__init__()
        normalized = loss.lower()
        if normalized not in {"mse", "l1"}:
            raise ValueError("deterministic loss must be 'mse' or 'l1'")
        self.loss = normalized

    def forward(
        self,
        model: DeterministicLatentPredictor,
        source_latent: Tensor,
        target_latent: Tensor,
        condition_tokens: Tensor,
    ) -> DeterministicLatentLoss:
        if source_latent.shape != target_latent.shape:
            raise ValueError("paired source and target latents must have identical shapes")
        prediction = model(source_latent, condition_tokens)
        objective = F.mse_loss if self.loss == "mse" else F.l1_loss
        return DeterministicLatentLoss(
            total=objective(prediction, target_latent, reduction="mean")
        )


class LatentAutoencoder(Protocol):
    def encode(self, image: Tensor, *, normalize: bool = True) -> Tensor: ...

    def decode(self, latent: Tensor, *, denormalize: bool = True) -> Tensor: ...


class DeterministicImagePredictor(nn.Module):
    """End-to-end deterministic baseline using the shared autoencoder and schema."""

    def __init__(
        self,
        autoencoder: nn.Module,
        condition_encoder: StructuredConditionEncoder,
        latent_predictor: DeterministicLatentPredictor,
    ) -> None:
        super().__init__()
        if condition_encoder.output_dim != latent_predictor.config.condition_dim:
            raise ValueError(
                "condition encoder width must equal predictor cross-attention width"
            )
        self.autoencoder = autoencoder
        self.condition_encoder = condition_encoder
        self.latent_predictor = latent_predictor

    def predict_latent(
        self, source_image: Tensor, conditions: Mapping[str, Any]
    ) -> Tensor:
        source_latent = self.autoencoder.encode(source_image, normalize=True)
        tokens = self.condition_encoder(conditions, batch_size=source_latent.shape[0])
        return self.latent_predictor(source_latent, tokens)

    def forward(self, source_image: Tensor, conditions: Mapping[str, Any]) -> Tensor:
        target_latent = self.predict_latent(source_image, conditions)
        return self.autoencoder.decode(target_latent, denormalize=True)


@dataclass(frozen=True)
class UnidirectionalCFMPath:
    """One target branch from a configured base state to the later latent."""

    state: Tensor
    target_velocity: Tensor
    initial_state: Tensor
    noise: Tensor
    tau: Tensor
    sigma_min: float
    base_distribution: str
    noise_scale: float


def _validate_sigma(sigma_min: float) -> float:
    sigma = float(sigma_min)
    if not 0.0 <= sigma < 1.0:
        raise ValueError("sigma_min must satisfy 0 <= sigma_min < 1")
    return sigma


def validate_cfm_base_distribution(
    base_distribution: str, noise_scale: float
) -> tuple[str, float]:
    distribution = str(base_distribution).strip().lower()
    if distribution not in CFM_BASE_DISTRIBUTIONS:
        raise ValueError(
            f"CFM base_distribution must be one of {CFM_BASE_DISTRIBUTIONS}"
        )
    scale = float(noise_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("CFM noise_scale must be positive and finite")
    return distribution, scale


def make_unidirectional_cfm_initial_state(
    source_latent: Tensor,
    noise: Tensor,
    *,
    base_distribution: str = "standard_normal",
    noise_scale: float = 1.0,
) -> Tensor:
    """Construct the conditional base sample used at both training and inference."""

    distribution, scale = validate_cfm_base_distribution(
        base_distribution, noise_scale
    )
    if source_latent.shape != noise.shape:
        raise ValueError("source latent and CFM noise must have identical shapes")
    if source_latent.dtype != noise.dtype or source_latent.device != noise.device:
        raise ValueError("source latent and CFM noise must share dtype and device")
    if not source_latent.is_floating_point() or not noise.is_floating_point():
        raise TypeError("source latent and CFM noise must be floating tensors")
    if not bool(torch.isfinite(source_latent).all()) or not bool(
        torch.isfinite(noise).all()
    ):
        raise ValueError("source latent and CFM noise must contain only finite values")
    scaled_noise = scale * noise
    if distribution == "source_gaussian":
        return source_latent + scaled_noise
    return scaled_noise


def make_unidirectional_cfm_path(
    target_latent: Tensor,
    tau: Tensor,
    *,
    source_latent: Tensor | None = None,
    noise: Tensor | None = None,
    sigma_min: float = 0.0,
    base_distribution: str = "standard_normal",
    noise_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> UnidirectionalCFMPath:
    """Build a one-way CFM path from a fixed or source-conditioned base."""

    if target_latent.ndim != 5 or not target_latent.is_floating_point():
        raise ValueError("target_latent must be a floating 3D tensor [B, C, D, H, W]")
    if not bool(torch.isfinite(target_latent).all()):
        raise ValueError("target_latent must contain only finite values")
    sigma = _validate_sigma(sigma_min)
    distribution, scale = validate_cfm_base_distribution(
        base_distribution, noise_scale
    )
    tau = _continuous_tau(tau, target_latent)
    if noise is None:
        noise = torch.randn(
            target_latent.shape,
            dtype=target_latent.dtype,
            device=target_latent.device,
            generator=generator,
        )
    if noise.shape != target_latent.shape:
        raise ValueError("noise and target_latent must have identical shapes")
    if source_latent is None:
        if distribution == "source_gaussian":
            raise ValueError("source_gaussian CFM requires source_latent")
        source_latent = torch.zeros_like(target_latent)
    initial_state = make_unidirectional_cfm_initial_state(
        source_latent,
        noise,
        base_distribution=distribution,
        noise_scale=scale,
    )
    broadcast_tau = tau.reshape(tau.shape[0], 1, 1, 1, 1)
    attenuation = 1.0 - (1.0 - sigma) * broadcast_tau
    state = attenuation * initial_state + broadcast_tau * target_latent
    target_velocity = target_latent - (1.0 - sigma) * initial_state
    return UnidirectionalCFMPath(
        state=state,
        target_velocity=target_velocity,
        initial_state=initial_state,
        noise=noise,
        tau=tau,
        sigma_min=sigma,
        base_distribution=distribution,
        noise_scale=scale,
    )


class UnidirectionalConditionalFMUNet(nn.Module):
    """Forward-only conditional FM baseline predicting one velocity branch.

    Input channels are ``[noised_later_target, clean_earlier_source]`` and output
    channels contain only the later-target velocity. There is no retrospective
    reconstruction branch or joint-state update.
    """

    input_order = ("noised_later_target", "clean_earlier_source")

    def __init__(
        self,
        config: JointVelocityUNetConfig,
        *,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = (
            backbone
            if backbone is not None
            else _build_conditioned_unet(
                config,
                in_channels=2 * config.latent_channels,
                out_channels=config.latent_channels,
            )
        )

    def forward(
        self,
        noised_target: Tensor,
        source_latent: Tensor,
        tau: Tensor,
        condition_tokens: Tensor,
    ) -> Tensor:
        _validate_latent(
            noised_target, self.config.latent_channels, name="noised_target"
        )
        _validate_latent(
            source_latent, self.config.latent_channels, name="source_latent"
        )
        if noised_target.shape != source_latent.shape:
            raise ValueError("source and noised target latents must have identical shapes")
        if (
            noised_target.dtype != source_latent.dtype
            or noised_target.device != source_latent.device
        ):
            raise ValueError("source and noised target must share dtype and device")
        tau = _continuous_tau(tau, noised_target)
        _validate_context(condition_tokens, noised_target, self.config.condition_dim)
        model_input = torch.cat((noised_target, source_latent), dim=1)
        velocity = self.backbone(
            x=model_input,
            timesteps=tau,
            context=condition_tokens,
        )
        if not isinstance(velocity, Tensor) or velocity.shape != noised_target.shape:
            raise RuntimeError(
                "unidirectional FM backbone must return exactly one velocity branch"
            )
        return velocity

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class UnidirectionalCFMLoss:
    total: Tensor


class _OneWayVelocity(Protocol):
    def __call__(
        self,
        noised_target: Tensor,
        source_latent: Tensor,
        tau: Tensor,
        condition_tokens: Tensor,
    ) -> Tensor: ...


class UnidirectionalConditionalFMObjective(nn.Module):
    """MSE objective for the forward-only conditional FM comparison."""

    def __init__(
        self,
        sigma_min: float = 0.0,
        *,
        base_distribution: str = "standard_normal",
        noise_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.sigma_min = _validate_sigma(sigma_min)
        self.base_distribution, self.noise_scale = validate_cfm_base_distribution(
            base_distribution, noise_scale
        )

    def forward(
        self,
        model: _OneWayVelocity,
        source_latent: Tensor,
        target_latent: Tensor,
        condition_tokens: Tensor,
        *,
        tau: Tensor | None = None,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> UnidirectionalCFMLoss:
        if source_latent.shape != target_latent.shape:
            raise ValueError("paired source and target latents must have identical shapes")
        if tau is None:
            tau = torch.rand(
                target_latent.shape[0],
                dtype=target_latent.dtype,
                device=target_latent.device,
                generator=generator,
            )
        path = make_unidirectional_cfm_path(
            target_latent,
            tau,
            source_latent=source_latent,
            noise=noise,
            sigma_min=self.sigma_min,
            base_distribution=self.base_distribution,
            noise_scale=self.noise_scale,
            generator=generator,
        )
        prediction = model(path.state, source_latent, path.tau, condition_tokens)
        if prediction.shape != path.target_velocity.shape:
            raise ValueError("one-way velocity prediction has an incompatible shape")
        return UnidirectionalCFMLoss(
            total=F.mse_loss(prediction, path.target_velocity, reduction="mean")
        )


class CopySourceBaseline(nn.Module):
    """No-change baseline returning the aligned source MRI unchanged."""

    def forward(self, source_image: Tensor) -> Tensor:
        if source_image.ndim != 5:
            raise ValueError("source MRI must have shape [B, C, D, H, W]")
        return source_image.clone()


def copy_source_baseline(source_image: Tensor) -> Tensor:
    return CopySourceBaseline()(source_image)


def _baseline_config_section(
    config: Mapping[str, Any], baseline_name: str
) -> Mapping[str, Any]:
    section = config
    if "model" in section and isinstance(section["model"], Mapping):
        section = section["model"]
    baselines = section.get("baselines")
    if isinstance(baselines, Mapping) and isinstance(baselines.get(baseline_name), Mapping):
        return baselines[baseline_name]
    if isinstance(section.get(baseline_name), Mapping):
        return section[baseline_name]
    if isinstance(section.get("velocity"), Mapping):
        return section["velocity"]
    return section


def build_deterministic_baseline_from_config(
    config: Mapping[str, Any] | JointVelocityUNetConfig,
    *,
    backbone: nn.Module | None = None,
) -> DeterministicLatentPredictor:
    if isinstance(config, JointVelocityUNetConfig):
        architecture = config
        options: Mapping[str, Any] = {}
    else:
        section = _baseline_config_section(config, "deterministic")
        options = section
        architecture_payload = {
            key: value
            for key, value in section.items()
            if key not in {"predict_residual", "fixed_timestep"}
        }
        architecture = JointVelocityUNetConfig.from_mapping(architecture_payload)
    return DeterministicLatentPredictor(
        architecture,
        predict_residual=bool(options.get("predict_residual", True)),
        fixed_timestep=float(options.get("fixed_timestep", 0.0)),
        backbone=backbone,
    )


def build_unidirectional_cfm_from_config(
    config: Mapping[str, Any] | JointVelocityUNetConfig,
    *,
    backbone: nn.Module | None = None,
) -> UnidirectionalConditionalFMUNet:
    architecture = (
        config
        if isinstance(config, JointVelocityUNetConfig)
        else JointVelocityUNetConfig.from_mapping(
            _baseline_config_section(config, "unidirectional_cfm")
        )
    )
    return UnidirectionalConditionalFMUNet(architecture, backbone=backbone)


__all__ = [
    "CFM_BASE_DISTRIBUTIONS",
    "CopySourceBaseline",
    "DeterministicImagePredictor",
    "DeterministicLatentLoss",
    "DeterministicLatentObjective",
    "DeterministicLatentPredictor",
    "UnidirectionalCFMLoss",
    "UnidirectionalCFMPath",
    "UnidirectionalConditionalFMObjective",
    "UnidirectionalConditionalFMUNet",
    "build_deterministic_baseline_from_config",
    "build_unidirectional_cfm_from_config",
    "copy_source_baseline",
    "make_unidirectional_cfm_initial_state",
    "make_unidirectional_cfm_path",
    "validate_cfm_base_distribution",
]
