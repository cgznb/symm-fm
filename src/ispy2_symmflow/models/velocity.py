"""Conditional 3D joint velocity network for SymmFlow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn

from .conditioning import ConditionSchema, StructuredConditionEncoder


@dataclass(frozen=True)
class JointVelocityUNetConfig:
    """Arguments for a MONAI 1.5.1 conditional DiffusionModelUNet."""

    latent_channels: int = 8
    channels: tuple[int, ...] = (128, 256, 384)
    num_res_blocks: int | tuple[int, ...] = 2
    attention_levels: tuple[bool, ...] = (False, True, True)
    num_head_channels: int | tuple[int, ...] = (32, 64, 64)
    condition_dim: int = 256
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    resblock_updown: bool = False
    transformer_num_layers: int = 1
    upcast_attention: bool = False
    dropout_cattn: float = 0.0
    include_fc: bool = True
    use_combined_linear: bool = False
    use_flash_attention: bool = False

    def __post_init__(self) -> None:
        channels = tuple(int(value) for value in self.channels)
        attention = tuple(bool(value) for value in self.attention_levels)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "attention_levels", attention)
        if isinstance(self.num_res_blocks, Sequence) and not isinstance(
            self.num_res_blocks, (str, bytes)
        ):
            blocks: int | tuple[int, ...] = tuple(
                int(value) for value in self.num_res_blocks
            )
            object.__setattr__(self, "num_res_blocks", blocks)
            if len(blocks) != len(channels) or any(value <= 0 for value in blocks):
                raise ValueError(
                    "num_res_blocks must contain one positive value per channel level"
                )
        elif int(self.num_res_blocks) <= 0:
            raise ValueError("num_res_blocks must be positive")

        head_channels = self.num_head_channels
        if isinstance(head_channels, Sequence) and not isinstance(
            head_channels, (str, bytes)
        ):
            heads: int | tuple[int, ...] = tuple(int(value) for value in head_channels)
            object.__setattr__(self, "num_head_channels", heads)
            if len(heads) != len(channels):
                raise ValueError(
                    "num_head_channels must have one value per channel level"
                )
            for channel, enabled, head_width in zip(channels, attention, heads, strict=True):
                if enabled and (head_width <= 0 or channel % head_width != 0):
                    raise ValueError(
                        "attention channel widths must be positive divisors of channels"
                    )
        else:
            head_width = int(head_channels)
            if head_width <= 0:
                raise ValueError("num_head_channels must be positive")
            if any(
                enabled and channel % head_width != 0
                for channel, enabled in zip(channels, attention, strict=True)
            ):
                raise ValueError("attention channels must be divisible by num_head_channels")

        if len(channels) < 2 or any(value <= 0 for value in channels):
            raise ValueError("channels must contain at least two positive values")
        if len(attention) != len(channels):
            raise ValueError("attention_levels must have one value per channel level")
        if self.latent_channels <= 0 or self.condition_dim <= 0:
            raise ValueError("latent_channels and condition_dim must be positive")
        if self.norm_num_groups <= 0:
            raise ValueError("norm_num_groups must be positive")
        if any(channel % self.norm_num_groups != 0 for channel in channels):
            raise ValueError("all channels must be divisible by norm_num_groups")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if self.transformer_num_layers <= 0:
            raise ValueError("transformer_num_layers must be positive")
        if not 0.0 <= self.dropout_cattn <= 1.0:
            raise ValueError("dropout_cattn must be in [0, 1]")

    @property
    def joint_channels(self) -> int:
        return 2 * self.latent_channels

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "JointVelocityUNetConfig":
        values = dict(payload)
        spatial_dims = int(values.pop("spatial_dims", 3))
        if spatial_dims != 3:
            raise ValueError("this project requires spatial_dims=3")
        with_conditioning = bool(values.pop("with_conditioning", True))
        if not with_conditioning:
            raise ValueError("the conditional SymmFlow U-Net requires with_conditioning=true")
        if "cross_attention_dim" in values:
            if "condition_dim" in values:
                raise ValueError(
                    "use only one of condition_dim and cross_attention_dim"
                )
            values["condition_dim"] = values.pop("cross_attention_dim")
        for training_key in ("batch_size", "learning_rate", "weight_decay", "max_steps"):
            values.pop(training_key, None)
        valid = {field.name for field in fields(cls)}
        unknown = set(values).difference(valid)
        if unknown:
            raise ValueError(f"unsupported velocity U-Net config keys: {sorted(unknown)}")
        for name in ("channels", "attention_levels"):
            if name in values:
                values[name] = tuple(values[name])
        for name in ("num_res_blocks", "num_head_channels"):
            if name in values and isinstance(values[name], list):
                values[name] = tuple(values[name])
        return cls(**values)


def join_branches(later: Tensor, earlier: Tensor) -> Tensor:
    """Concatenate branches in the invariant order ``[later(x), earlier(y)]``."""

    if later.shape != earlier.shape:
        raise ValueError("later and earlier branch tensors must have identical shapes")
    if later.ndim != 5:
        raise ValueError("branch tensors must have shape [B, C, D, H, W]")
    if later.device != earlier.device or later.dtype != earlier.dtype:
        raise ValueError("later and earlier branches must share device and dtype")
    return torch.cat((later, earlier), dim=1)


def split_branches(joint: Tensor, latent_channels: int) -> tuple[Tensor, Tensor]:
    """Return ``(later_branch, earlier_branch)`` without changing semantics."""

    if joint.ndim != 5:
        raise ValueError("joint tensors must have shape [B, 2*C, D, H, W]")
    if latent_channels <= 0:
        raise ValueError("latent_channels must be positive")
    if joint.shape[1] != 2 * latent_channels:
        raise ValueError(
            f"expected {2 * latent_channels} joint channels, got {joint.shape[1]}"
        )
    return joint[:, :latent_channels], joint[:, latent_channels:]


def _load_monai_unet_class() -> type[nn.Module]:
    try:
        from monai.networks.nets import DiffusionModelUNet
    except ImportError as error:
        raise ImportError(
            "JointVelocityUNet requires MONAI 1.5.1; install the project's train extra"
        ) from error
    return DiffusionModelUNet


def _build_monai_unet(config: JointVelocityUNetConfig) -> nn.Module:
    unet_class = _load_monai_unet_class()
    return unet_class(
        spatial_dims=3,
        in_channels=config.joint_channels,
        out_channels=config.joint_channels,
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


class JointVelocityUNet(nn.Module):
    """Shared 3D U-Net that predicts both SymmFlow velocity branches at once."""

    branch_order = ("later", "earlier")

    def __init__(
        self,
        config: JointVelocityUNetConfig,
        *,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone if backbone is not None else _build_monai_unet(config)

    def _validate_inputs(
        self, joint_state: Tensor, tau: Tensor, condition_tokens: Tensor
    ) -> Tensor:
        if joint_state.ndim != 5:
            raise ValueError("joint state must have shape [B, 2*C, D, H, W]")
        if joint_state.shape[1] != self.config.joint_channels:
            raise ValueError(
                f"expected {self.config.joint_channels} joint channels, "
                f"got {joint_state.shape[1]}"
            )
        if not joint_state.is_floating_point():
            raise TypeError("joint state must use a floating-point dtype")
        if not torch.isfinite(joint_state).all():
            raise ValueError("joint state must contain only finite values")
        if not isinstance(tau, Tensor) or not tau.is_floating_point():
            raise TypeError("tau must be a floating-point tensor with shape [B] or [B, 1]")
        if tau.ndim == 2 and tau.shape[1] == 1:
            tau = tau[:, 0]
        if tau.ndim != 1 or tau.shape[0] != joint_state.shape[0]:
            raise ValueError("tau must have shape [B] or [B, 1]")
        if not torch.all(torch.isfinite(tau)):
            raise ValueError("tau must be finite")
        if torch.any(tau < -1e-6) or torch.any(tau > 1.0 + 1e-6):
            raise ValueError("tau must lie in [0, 1]")
        if condition_tokens.ndim != 3:
            raise ValueError("condition tokens must have shape [B, L, condition_dim]")
        if condition_tokens.shape[0] != joint_state.shape[0]:
            raise ValueError("condition tokens and joint state must share batch size")
        if condition_tokens.shape[1] == 0:
            raise ValueError("at least one condition token is required")
        if condition_tokens.shape[2] != self.config.condition_dim:
            raise ValueError(
                f"expected condition token width {self.config.condition_dim}, "
                f"got {condition_tokens.shape[2]}"
            )
        if not condition_tokens.is_floating_point():
            raise TypeError("condition tokens must use a floating-point dtype")
        if not torch.isfinite(condition_tokens).all():
            raise ValueError("condition tokens must contain only finite values")
        if condition_tokens.device != joint_state.device:
            raise ValueError("condition tokens and joint state must be on the same device")
        return tau.to(device=joint_state.device)

    def forward(
        self, joint_state: Tensor, tau: Tensor, condition_tokens: Tensor
    ) -> Tensor:
        """Predict joint velocity with unchanged ``[v_x, v_y]`` channel order."""

        tau = self._validate_inputs(joint_state, tau, condition_tokens)
        velocity = self.backbone(
            x=joint_state,
            timesteps=tau,
            context=condition_tokens,
        )
        if not isinstance(velocity, Tensor) or velocity.shape != joint_state.shape:
            raise RuntimeError(
                "DiffusionModelUNet must return a tensor with the joint state's shape"
            )
        if not torch.isfinite(velocity).all():
            raise FloatingPointError("DiffusionModelUNet returned non-finite velocity")
        return velocity

    def forward_branches(
        self,
        later_state: Tensor,
        earlier_state: Tensor,
        tau: Tensor,
        condition_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        joint_state = join_branches(later_state, earlier_state)
        joint_velocity = self(joint_state, tau, condition_tokens)
        return split_branches(joint_velocity, self.config.latent_channels)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class ConditionalVelocityModel(nn.Module):
    """Compose raw structured conditions with the joint velocity U-Net."""

    def __init__(
        self,
        condition_encoder: StructuredConditionEncoder,
        velocity_unet: JointVelocityUNet,
    ) -> None:
        super().__init__()
        if condition_encoder.output_dim != velocity_unet.config.condition_dim:
            raise ValueError(
                "condition encoder width must equal the U-Net cross-attention width"
            )
        self.condition_encoder = condition_encoder
        self.velocity_unet = velocity_unet

    def forward(
        self,
        joint_state: Tensor,
        tau: Tensor,
        conditions: Mapping[str, Any],
    ) -> Tensor:
        tokens = self.condition_encoder(conditions, batch_size=joint_state.shape[0])
        if tokens.device != joint_state.device:
            raise ValueError(
                "condition encoder and velocity U-Net must be moved to the same device"
            )
        return self.velocity_unet(joint_state, tau, tokens)

    def forward_branches(
        self,
        later_state: Tensor,
        earlier_state: Tensor,
        tau: Tensor,
        conditions: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor]:
        joint = join_branches(later_state, earlier_state)
        velocity = self(joint, tau, conditions)
        return split_branches(velocity, self.velocity_unet.config.latent_channels)


def _velocity_config_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    section = config
    if "model" in section and isinstance(section["model"], Mapping):
        section = section["model"]
    for key in ("velocity", "velocity_unet", "symmflow"):
        if key in section and isinstance(section[key], Mapping):
            return section[key]
    return section


def build_velocity_model_from_config(
    config: Mapping[str, Any] | JointVelocityUNetConfig,
    *,
    backbone: nn.Module | None = None,
) -> JointVelocityUNet:
    """Build the MONAI-backed joint velocity model from configuration."""

    parsed = (
        config
        if isinstance(config, JointVelocityUNetConfig)
        else JointVelocityUNetConfig.from_mapping(_velocity_config_section(config))
    )
    return JointVelocityUNet(parsed, backbone=backbone)


def build_conditional_velocity_model_from_config(
    config: Mapping[str, Any],
    *,
    backbone: nn.Module | None = None,
) -> ConditionalVelocityModel:
    """Build both condition encoder and joint velocity U-Net from one config."""

    model_section: Mapping[str, Any] = config
    if "model" in model_section and isinstance(model_section["model"], Mapping):
        model_section = model_section["model"]
    conditioning_section = model_section.get(
        "conditioning", model_section.get("conditions")
    )
    if not isinstance(conditioning_section, Mapping):
        raise ValueError("model config must include a conditioning mapping")
    schema_payload = conditioning_section.get("schema", conditioning_section)
    if not isinstance(schema_payload, Mapping):
        raise TypeError("conditioning schema must be a mapping")
    schema = ConditionSchema.from_dict(schema_payload)
    encoder = StructuredConditionEncoder(
        schema,
        hidden_dim=(
            None
            if conditioning_section.get("hidden_dim") is None
            else int(conditioning_section["hidden_dim"])
        ),
        strict=bool(conditioning_section.get("strict", True)),
    )
    velocity = build_velocity_model_from_config(model_section, backbone=backbone)
    return ConditionalVelocityModel(encoder, velocity)


build_velocity_unet_from_config = build_velocity_model_from_config


__all__ = [
    "ConditionalVelocityModel",
    "JointVelocityUNet",
    "JointVelocityUNetConfig",
    "build_conditional_velocity_model_from_config",
    "build_velocity_model_from_config",
    "build_velocity_unet_from_config",
    "join_branches",
    "split_branches",
]
