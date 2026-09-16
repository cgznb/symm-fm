from __future__ import annotations

import torch
import torch.nn as nn

from .contracts import (
    PAPER_FAITHFUL_ARCHITECTURE,
    REGISTERED_LARGE_PAPER_ARCHITECTURE,
)
from .ct_denoiser import (
    CTStyleDenoiser3D,
    ChannelLayerNorm3D,
    DenoiserResult,
    _sinusoidal_embedding,
)
from .paper_conditioning import PaperConditionOutput


class VolumetricCrossAttention3D(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        context_dim: int = 512,
        heads: int = 8,
        head_dim: int = 32,
    ) -> None:
        super().__init__()
        dimensions = {
            "channels": channels,
            "context_dim": context_dim,
            "heads": heads,
            "head_dim": head_dim,
        }
        for name, value in dimensions.items():
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact positive integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.channels = channels
        self.context_dim = context_dim
        self.heads = heads
        self.head_dim = head_dim
        attention_dim = self.heads * self.head_dim
        self.scale = self.head_dim**-0.5

        self.normalization = ChannelLayerNorm3D(self.channels)
        self.query_projection = nn.Linear(self.channels, attention_dim, bias=False)
        self.key_projection = nn.Linear(self.context_dim, attention_dim, bias=False)
        self.value_projection = nn.Linear(self.context_dim, attention_dim, bias=False)
        self.output_projection = nn.Linear(attention_dim, self.channels)

    def _validate_inputs(
        self,
        hidden: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> None:
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("hidden must be a tensor")
        if hidden.ndim != 5 or hidden.shape[1] != self.channels:
            raise ValueError(
                f"hidden must have shape [B,{self.channels},D,H,W]"
            )
        if not hidden.is_floating_point():
            raise TypeError("hidden must use a floating-point dtype")
        if not isinstance(context_tokens, torch.Tensor):
            raise TypeError("context tokens must be a tensor")
        if context_tokens.ndim != 3:
            raise ValueError(
                f"context tokens must have shape [B,7,{self.context_dim}]"
            )
        if context_tokens.shape[0] != hidden.shape[0]:
            raise ValueError("hidden and context batch sizes must match")
        if context_tokens.shape[1] != 7:
            raise ValueError("cross-attention requires exactly seven context tokens")
        if context_tokens.shape[2] != self.context_dim:
            raise ValueError(f"context token width must be {self.context_dim}")
        if not context_tokens.is_floating_point():
            raise TypeError("context tokens must use a floating-point dtype")
        if hidden.device != context_tokens.device:
            raise ValueError("hidden and context tokens must use the same device")
        if not isinstance(context_mask, torch.Tensor):
            raise TypeError("context mask must be a tensor")
        if context_mask.shape != (hidden.shape[0], 7):
            raise ValueError("context mask must have shape [B,7]")
        if context_mask.dtype != torch.bool:
            raise TypeError("context mask must have boolean dtype")
        if context_mask.device != hidden.device:
            raise ValueError("context mask and hidden must use the same device")
        if not torch.all(context_mask.any(dim=1)):
            raise ValueError("every context mask row must contain at least one token")
        if not torch.all(torch.isfinite(hidden)):
            raise ValueError("hidden must contain only finite values")
        if not torch.all(torch.isfinite(context_tokens)):
            raise ValueError("context tokens must contain only finite values")

    @staticmethod
    def _require_finite_computation(value: torch.Tensor, stage: str) -> None:
        if not bool(torch.isfinite(value).all()):
            raise ValueError(
                f"cross-attention produced nonfinite {stage} from finite inputs"
            )

    def forward(
        self,
        hidden: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(hidden, context_tokens, context_mask)
        batch, _channels, depth, height, width = hidden.shape
        normalized = self.normalization(hidden)
        flattened = normalized.permute(0, 2, 3, 4, 1).reshape(
            batch, depth * height * width, self.channels
        )

        query = self.query_projection(flattened)
        self._require_finite_computation(query, "query projection")
        compute_dtype = query.dtype
        masked_context = context_tokens.masked_fill(
            ~context_mask[:, :, None], 0.0
        ).to(dtype=compute_dtype)
        self._require_finite_computation(masked_context, "context conversion")
        key = self.key_projection(masked_context)
        value = self.value_projection(masked_context)
        self._require_finite_computation(key, "key projection")
        self._require_finite_computation(value, "value projection")

        query = query.reshape(
            batch, -1, self.heads, self.head_dim
        )
        key = key.reshape(batch, 7, self.heads, self.head_dim)
        value = value.reshape(batch, 7, self.heads, self.head_dim)
        query = query.transpose(1, 2) * self.scale
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-1, -2))
        self._require_finite_computation(scores, "attention scores")
        scores = scores.masked_fill(~context_mask[:, None, None, :], -torch.inf)
        weights = torch.softmax(scores.float(), dim=-1)
        self._require_finite_computation(weights, "attention weights")
        weights = weights.to(dtype=value.dtype)
        attended = torch.matmul(weights, value)
        self._require_finite_computation(attended, "attended values")
        attended = attended.transpose(1, 2).reshape(
            batch, depth * height * width, self.heads * self.head_dim
        )
        projected = self.output_projection(attended)
        self._require_finite_computation(projected, "output projection")
        projected = projected.reshape(batch, depth, height, width, self.channels)
        projected = projected.permute(0, 4, 1, 2, 3).to(dtype=hidden.dtype)
        output = hidden + projected
        self._require_finite_computation(output, "output")
        return output


class PaperFaithfulDenoiser3D(CTStyleDenoiser3D):
    def __init__(self, *, activation_checkpointing: bool = True) -> None:
        super().__init__(activation_checkpointing=activation_checkpointing)
        self.architecture = PAPER_FAITHFUL_ARCHITECTURE
        self.down_cross_attentions = nn.ModuleList(
            VolumetricCrossAttention3D(ch, context_dim=512, heads=8, head_dim=32)
            for ch in self.channels
        )
        self.middle_cross_attention = VolumetricCrossAttention3D(
            self.channels[-1], context_dim=512, heads=8, head_dim=32
        )
        self.up_cross_attentions = nn.ModuleList(
            VolumetricCrossAttention3D(ch, context_dim=512, heads=8, head_dim=32)
            for ch in reversed((self.channels[0], *self.channels[:-1]))
        )

    @staticmethod
    def _validate_forward_inputs(
        noisy_latent: torch.Tensor,
        spatial_condition: torch.Tensor,
        condition: PaperConditionOutput,
        timesteps: torch.Tensor,
    ) -> None:
        if not isinstance(noisy_latent, torch.Tensor):
            raise TypeError("noisy latent must be a tensor")
        if not isinstance(spatial_condition, torch.Tensor):
            raise TypeError("spatial condition must be a tensor")
        if not isinstance(condition, PaperConditionOutput):
            raise TypeError("condition must be a PaperConditionOutput")
        condition_tensors = (
            ("global condition", condition.global_condition),
            ("context tokens", condition.context_tokens),
            ("context mask", condition.context_mask),
        )
        for name, value in condition_tensors:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
        if not isinstance(timesteps, torch.Tensor):
            raise TypeError("timesteps must be a tensor")

        if noisy_latent.ndim != 5 or noisy_latent.shape[1] != 8:
            raise ValueError("noisy latent must have shape [B,8,D,H,W]")
        if spatial_condition.ndim != 5 or spatial_condition.shape[1] != 9:
            raise ValueError("spatial condition must have shape [B,9,D,H,W]")
        if noisy_latent.shape[0] != spatial_condition.shape[0] or (
            noisy_latent.shape[-3:] != spatial_condition.shape[-3:]
        ):
            raise ValueError("noisy latent and spatial condition shapes must align")
        batch_size = noisy_latent.shape[0]
        if batch_size <= 0 or any(size <= 0 for size in noisy_latent.shape[2:]):
            raise ValueError(
                "noisy latent batch and spatial dimensions must be positive"
            )
        if noisy_latent.shape[-2] % 8 or noisy_latent.shape[-1] % 8:
            raise ValueError("noisy latent height and width must be divisible by 8")
        if not noisy_latent.is_floating_point():
            raise TypeError("noisy latent must use a floating-point dtype")
        if not spatial_condition.is_floating_point():
            raise TypeError("spatial condition must use a floating-point dtype")
        if spatial_condition.device != noisy_latent.device:
            raise ValueError(
                "noisy latent and spatial condition must use the same device"
            )
        if not bool(torch.isfinite(noisy_latent).all()):
            raise ValueError("noisy latent must contain only finite values")
        if not bool(torch.isfinite(spatial_condition).all()):
            raise ValueError("spatial condition must contain only finite values")

        if condition.global_condition.shape != (batch_size, 512):
            raise ValueError("global condition must have shape [B,512]")
        if condition.context_tokens.shape != (batch_size, 7, 512):
            raise ValueError("context tokens must have shape [B,7,512]")
        if condition.context_mask.shape != (batch_size, 7):
            raise ValueError("context mask must have shape [B,7]")
        if not condition.global_condition.is_floating_point():
            raise TypeError("global condition must use a floating-point dtype")
        if not condition.context_tokens.is_floating_point():
            raise TypeError("context tokens must use a floating-point dtype")
        if condition.context_mask.dtype != torch.bool:
            raise TypeError("context mask must have boolean dtype")
        if any(
            value.device != noisy_latent.device
            for _name, value in condition_tensors
        ):
            raise ValueError(
                "paper condition tensors and noisy latent must use the same device"
            )
        if not bool(condition.context_mask.any(dim=1).all()):
            raise ValueError(
                "every context mask row must contain at least one token"
            )
        if not bool(torch.isfinite(condition.global_condition).all()):
            raise ValueError("global condition must contain only finite values")
        if not bool(torch.isfinite(condition.context_tokens).all()):
            raise ValueError("context tokens must contain only finite values")

        if timesteps.shape != (batch_size,):
            raise ValueError("timesteps must have shape [B]")
        numeric_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }
        if timesteps.dtype not in numeric_dtypes:
            raise TypeError("timesteps must use a real numeric dtype")
        if timesteps.device != noisy_latent.device:
            raise ValueError("timesteps and noisy latent must use the same device")
        if not bool(torch.isfinite(timesteps).all()):
            raise ValueError("timesteps must contain only finite values")
        if timesteps.is_floating_point() and not bool(
            (timesteps == timesteps.round()).all()
        ):
            raise ValueError("timesteps must contain integer values")
        if bool((timesteps < 0).any()):
            raise ValueError("timesteps must be nonnegative")

    def forward(
        self,
        noisy_latent: torch.Tensor,
        spatial_condition: torch.Tensor,
        condition: PaperConditionOutput,
        timesteps: torch.Tensor,
    ) -> DenoiserResult:
        self._validate_forward_inputs(
            noisy_latent, spatial_condition, condition, timesteps
        )
        global_condition = condition.global_condition.to(
            dtype=self.semantic_projection.weight.dtype
        )
        network_input, spatial_semantic = self.prepare_input(
            noisy_latent, spatial_condition, global_condition
        )
        hidden = self.input(network_input)
        residual = hidden.clone()
        positional_bias = self.relative_position_bias(hidden.shape[2], hidden.device)
        hidden = self._run_module(
            self.initial_temporal_attention, hidden, positional_bias
        )
        time_embedding = self.time_mlp(
            _sinusoidal_embedding(timesteps, self.channels[0])
        )

        skips: list[torch.Tensor] = []
        for stage, cross_attention in zip(
            self.downs, self.down_cross_attentions, strict=True
        ):
            block1, block2, spatial_attention, temporal_attention, downsample = stage
            hidden = self._run_module(block1, hidden, time_embedding)
            hidden = self._run_module(block2, hidden, time_embedding)
            hidden = self._run_module(
                cross_attention,
                hidden,
                condition.context_tokens,
                condition.context_mask,
            )
            hidden = self._run_module(spatial_attention, hidden)
            hidden = self._run_module(temporal_attention, hidden, positional_bias)
            skips.append(hidden)
            hidden = downsample(hidden)

        hidden = self._run_module(self.middle_block1, hidden, time_embedding)
        hidden = self._run_module(
            self.middle_cross_attention,
            hidden,
            condition.context_tokens,
            condition.context_mask,
        )
        hidden = self._run_module(self.middle_spatial_attention, hidden)
        hidden = self._run_module(
            self.middle_temporal_attention, hidden, positional_bias
        )
        hidden = self._run_module(self.middle_block2, hidden, time_embedding)

        for stage, cross_attention in zip(
            self.ups, self.up_cross_attentions, strict=True
        ):
            block1, block2, spatial_attention, temporal_attention, upsample = stage
            hidden = torch.cat((hidden, skips.pop()), dim=1)
            hidden = self._run_module(block1, hidden, time_embedding)
            hidden = self._run_module(block2, hidden, time_embedding)
            hidden = self._run_module(
                cross_attention,
                hidden,
                condition.context_tokens,
                condition.context_mask,
            )
            hidden = self._run_module(spatial_attention, hidden)
            hidden = self._run_module(temporal_attention, hidden, positional_bias)
            hidden = upsample(hidden)

        predicted_noise = self.final(torch.cat((hidden, residual), dim=1))
        return DenoiserResult(
            predicted_noise=predicted_noise,
            network_input=network_input,
            spatial_semantic=spatial_semantic,
        )


class PaperRegisteredLargeDenoiser3D(PaperFaithfulDenoiser3D):
    _WIDTHS = (32, 64, 128, 256)

    def __init__(self, *, activation_checkpointing: bool = True) -> None:
        super().__init__(activation_checkpointing=activation_checkpointing)
        self.architecture = REGISTERED_LARGE_PAPER_ARCHITECTURE


__all__ = [
    "PaperFaithfulDenoiser3D",
    "PaperRegisteredLargeDenoiser3D",
    "VolumetricCrossAttention3D",
]
