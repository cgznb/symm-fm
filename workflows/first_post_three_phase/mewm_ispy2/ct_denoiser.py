from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .contracts import CT_DENOISER_ARCHITECTURE


@dataclass(frozen=True)
class DenoiserResult:
    predicted_noise: torch.Tensor
    network_input: torch.Tensor
    spatial_semantic: torch.Tensor | None

    @property
    def prediction(self) -> torch.Tensor:
        """Return the model output independent of diffusion parameterization."""
        return self.predicted_noise


def _sinusoidal_embedding(values: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = values.float()[:, None] * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=1)


class ChannelLayerNorm3D(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(dim=1, keepdim=True)
        variance = value.var(dim=1, unbiased=False, keepdim=True)
        return (value - mean) * torch.rsqrt(variance + self.epsilon) * self.scale


class ResidualPreNorm3D(nn.Module):
    def __init__(self, channels: int, module: nn.Module) -> None:
        super().__init__()
        self.norm = ChannelLayerNorm3D(channels)
        self.module = module

    def forward(
        self, value: torch.Tensor, positional_bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = self.norm(value)
        if positional_bias is None:
            result = self.module(normalized)
        else:
            result = self.module(normalized, positional_bias)
        return value + result


class CTConvBlock3D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.norm = nn.GroupNorm(groups, output_channels)
        self.activation = nn.SiLU()

    def forward(
        self,
        value: torch.Tensor,
        scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        hidden = self.norm(self.conv(value))
        if scale_shift is not None:
            scale, shift = scale_shift
            hidden = hidden * (1.0 + scale) + shift
        return self.activation(hidden)


class CTResnetBlock3D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        time_embedding_dim: int | None,
        groups: int,
    ) -> None:
        super().__init__()
        self.time_projection = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dim, output_channels * 2))
            if time_embedding_dim is not None
            else None
        )
        self.block1 = CTConvBlock3D(input_channels, output_channels, groups)
        self.block2 = CTConvBlock3D(output_channels, output_channels, groups)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, 1)
        )

    def forward(
        self, value: torch.Tensor, time_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        scale_shift = None
        if self.time_projection is not None:
            if time_embedding is None:
                raise ValueError("time embedding is required by CT residual block")
            projected = self.time_projection(time_embedding)[:, :, None, None, None]
            scale_shift = projected.chunk(2, dim=1)
        hidden = self.block1(value, scale_shift)
        hidden = self.block2(hidden)
        return hidden + self.skip(value)


class SpatialLinearAttention3D(nn.Module):
    def __init__(self, channels: int, *, heads: int, head_dim: int = 32) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        hidden_channels = self.heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.to_qkv = nn.Conv2d(channels, hidden_channels * 3, 1, bias=False)
        self.output = nn.Conv2d(hidden_channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, _channels, depth, height, width = value.shape
        slices = value.permute(0, 2, 1, 3, 4).reshape(
            batch * depth, value.shape[1], height, width
        )
        query, key, content = self.to_qkv(slices).chunk(3, dim=1)
        query = query.reshape(batch * depth, self.heads, self.head_dim, height * width)
        key = key.reshape(batch * depth, self.heads, self.head_dim, height * width)
        content = content.reshape(
            batch * depth, self.heads, self.head_dim, height * width
        )
        query = query.softmax(dim=-2) * self.scale
        key = key.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", key, content)
        attended = torch.einsum("bhde,bhdn->bhen", context, query)
        attended = attended.reshape(
            batch * depth, self.heads * self.head_dim, height, width
        )
        attended = self.output(attended)
        return attended.reshape(batch, depth, -1, height, width).permute(0, 2, 1, 3, 4)


def _apply_rotary_embedding(value: torch.Tensor) -> torch.Tensor:
    dimension = value.shape[-1]
    if dimension % 2:
        raise ValueError("rotary attention head dimension must be even")
    positions = torch.arange(value.shape[-2], device=value.device, dtype=torch.float32)
    inverse_frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, dimension, 2, device=value.device, dtype=torch.float32)
        / dimension
    )
    angles = torch.outer(positions, inverse_frequencies).to(value.dtype)
    cosine = angles.cos()
    sine = angles.sin()
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, odd * cosine + even * sine), dim=-1
    )
    return rotated.flatten(start_dim=-2)


class SequenceAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        heads: int,
        head_dim: int = 32,
        rotary: bool = False,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.rotary = bool(rotary)
        hidden_channels = self.heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.to_qkv = nn.Linear(channels, hidden_channels * 3, bias=False)
        self.output = nn.Linear(hidden_channels, channels, bias=False)

    def forward(
        self, value: torch.Tensor, positional_bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        query, key, content = self.to_qkv(value).chunk(3, dim=-1)
        base_shape = query.shape[:-2]
        sequence_length = query.shape[-2]
        query = query.reshape(*base_shape, sequence_length, self.heads, self.head_dim)
        key = key.reshape(*base_shape, sequence_length, self.heads, self.head_dim)
        content = content.reshape(
            *base_shape, sequence_length, self.heads, self.head_dim
        )
        query = query.transpose(-3, -2)
        key = key.transpose(-3, -2)
        content = content.transpose(-3, -2)
        if self.rotary:
            query = _apply_rotary_embedding(query)
            key = _apply_rotary_embedding(key)
        similarity = torch.matmul(query * self.scale, key.transpose(-2, -1))
        if positional_bias is not None:
            bias_shape = (1,) * len(base_shape) + positional_bias.shape
            similarity = similarity + positional_bias.reshape(bias_shape)
        attention = similarity.softmax(dim=-1)
        attended = torch.matmul(attention, content).transpose(-3, -2)
        attended = attended.reshape(*base_shape, sequence_length, -1)
        return self.output(attended)


class TemporalAttention3D(nn.Module):
    def __init__(self, channels: int, *, heads: int, head_dim: int) -> None:
        super().__init__()
        self.attention = SequenceAttention(
            channels, heads=heads, head_dim=head_dim, rotary=True
        )

    def forward(
        self, value: torch.Tensor, positional_bias: torch.Tensor
    ) -> torch.Tensor:
        sequence = value.permute(0, 3, 4, 2, 1)
        attended = self.attention(sequence, positional_bias)
        return attended.permute(0, 4, 3, 1, 2)


class SpatialAttention3D(nn.Module):
    def __init__(self, channels: int, *, heads: int, head_dim: int = 32) -> None:
        super().__init__()
        self.attention = SequenceAttention(
            channels, heads=heads, head_dim=head_dim, rotary=False
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = value.shape
        sequence = value.permute(0, 2, 3, 4, 1).reshape(
            batch, depth, height * width, channels
        )
        attended = self.attention(sequence)
        return attended.reshape(batch, depth, height, width, channels).permute(
            0, 4, 1, 2, 3
        )


class RelativePositionBias(nn.Module):
    def __init__(
        self, *, heads: int, buckets: int = 32, max_distance: int = 32
    ) -> None:
        super().__init__()
        self.buckets = int(buckets)
        self.max_distance = int(max_distance)
        self.embedding = nn.Embedding(self.buckets, heads)

    def forward(self, length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(length, device=device)
        relative = positions[None, :] - positions[:, None]
        buckets = self._bucket(relative)
        return self.embedding(buckets).permute(2, 0, 1)

    def _bucket(self, relative: torch.Tensor) -> torch.Tensor:
        half_buckets = self.buckets // 2
        distance = -relative
        result = (distance < 0).long() * half_buckets
        distance = distance.abs()
        max_exact = half_buckets // 2
        is_small = distance < max_exact
        safe_distance = distance.float().clamp_min(1.0)
        large = max_exact + (
            torch.log(safe_distance / max_exact)
            / math.log(self.max_distance / max_exact)
            * (half_buckets - max_exact)
        ).long()
        large = large.clamp(max=half_buckets - 1)
        return result + torch.where(is_small, distance, large)


def _downsample(channels: int) -> nn.Module:
    return nn.Conv3d(
        channels,
        channels,
        kernel_size=(1, 4, 4),
        stride=(1, 2, 2),
        padding=(0, 1, 1),
    )


def _upsample(channels: int) -> nn.Module:
    return nn.ConvTranspose3d(
        channels,
        channels,
        kernel_size=(1, 4, 4),
        stride=(1, 2, 2),
        padding=(0, 1, 1),
    )


class CTStyleDenoiser3D(nn.Module):
    _WIDTHS = (24, 48, 96, 192)

    def __init__(self, *, activation_checkpointing: bool = True) -> None:
        super().__init__()
        channels = self._WIDTHS
        if type(channels) is not tuple or len(channels) != 4:
            raise TypeError("channels must be an exact four-integer tuple")
        if any(type(channel) is not int for channel in channels):
            raise TypeError("channels must be an exact four-integer tuple")
        if any(channel <= 0 or channel % 8 for channel in channels):
            raise ValueError("channels must be positive and divisible by eight")
        self.architecture = CT_DENOISER_ARCHITECTURE
        self.base_input_channels = 17
        self.semantic_condition_channels = 512
        self.spatial_semantic_channels = 32
        self.network_input_channels = 49
        self.output_channels = 8
        self.channels = channels
        self.activation_checkpointing = bool(activation_checkpointing)
        self.semantic_projection = nn.Linear(
            self.semantic_condition_channels, self.spatial_semantic_channels
        )
        self.input = nn.Conv3d(
            self.network_input_channels,
            self.channels[0],
            kernel_size=(1, 7, 7),
            padding=(0, 3, 3),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(self.channels[0], self.channels[0] * 4),
            nn.GELU(),
            nn.Linear(self.channels[0] * 4, self.channels[0] * 4),
        )
        self.relative_position_bias = RelativePositionBias(heads=8, max_distance=32)
        self.initial_temporal_attention = ResidualPreNorm3D(
            self.channels[0],
            TemporalAttention3D(self.channels[0], heads=8, head_dim=32),
        )

        dimensions = (self.channels[0], *self.channels)
        pairs = tuple(zip(dimensions[:-1], dimensions[1:], strict=True))
        self.downs = nn.ModuleList()
        for index, (input_channels, output_channels) in enumerate(pairs):
            is_last = index == len(pairs) - 1
            self.downs.append(
                nn.ModuleList(
                    (
                        CTResnetBlock3D(
                            input_channels,
                            output_channels,
                            time_embedding_dim=self.channels[0] * 4,
                            groups=8,
                        ),
                        CTResnetBlock3D(
                            output_channels,
                            output_channels,
                            time_embedding_dim=self.channels[0] * 4,
                            groups=8,
                        ),
                        ResidualPreNorm3D(
                            output_channels,
                            SpatialLinearAttention3D(output_channels, heads=8),
                        ),
                        ResidualPreNorm3D(
                            output_channels,
                            TemporalAttention3D(
                                output_channels, heads=8, head_dim=32
                            ),
                        ),
                        nn.Identity() if is_last else _downsample(output_channels),
                    )
                )
            )

        middle_channels = self.channels[-1]
        self.middle_block1 = CTResnetBlock3D(
            middle_channels,
            middle_channels,
            time_embedding_dim=self.channels[0] * 4,
            groups=8,
        )
        self.middle_spatial_attention = ResidualPreNorm3D(
            middle_channels,
            SpatialAttention3D(middle_channels, heads=8),
        )
        self.middle_temporal_attention = ResidualPreNorm3D(
            middle_channels,
            TemporalAttention3D(middle_channels, heads=8, head_dim=32),
        )
        self.middle_block2 = CTResnetBlock3D(
            middle_channels,
            middle_channels,
            time_embedding_dim=self.channels[0] * 4,
            groups=8,
        )

        self.ups = nn.ModuleList()
        reversed_pairs = tuple(reversed(pairs))
        for index, (input_channels, output_channels) in enumerate(reversed_pairs):
            is_last = index == len(reversed_pairs) - 1
            self.ups.append(
                nn.ModuleList(
                    (
                        CTResnetBlock3D(
                            output_channels * 2,
                            input_channels,
                            time_embedding_dim=self.channels[0] * 4,
                            groups=8,
                        ),
                        CTResnetBlock3D(
                            input_channels,
                            input_channels,
                            time_embedding_dim=self.channels[0] * 4,
                            groups=8,
                        ),
                        ResidualPreNorm3D(
                            input_channels,
                            SpatialLinearAttention3D(input_channels, heads=8),
                        ),
                        ResidualPreNorm3D(
                            input_channels,
                            TemporalAttention3D(input_channels, heads=8, head_dim=32),
                        ),
                        nn.Identity() if is_last else _upsample(input_channels),
                    )
                )
            )
        self.final = nn.Sequential(
            CTResnetBlock3D(
                self.channels[0] * 2,
                self.channels[0],
                time_embedding_dim=None,
                groups=8,
            ),
            nn.Conv3d(self.channels[0], self.output_channels, 1),
        )

    def prepare_input(
        self,
        noisy_latent: torch.Tensor,
        spatial_condition: torch.Tensor,
        semantic_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noisy_latent.ndim != 5 or noisy_latent.shape[1] != 8:
            raise ValueError("noisy latent must have shape [B,8,D,H,W]")
        if spatial_condition.ndim != 5 or spatial_condition.shape[1] != 9:
            raise ValueError("spatial condition must have shape [B,9,D,H,W]")
        if noisy_latent.shape[0] != spatial_condition.shape[0] or (
            noisy_latent.shape[-3:] != spatial_condition.shape[-3:]
        ):
            raise ValueError("noisy latent and spatial condition shapes must align")
        if semantic_condition.shape != (
            noisy_latent.shape[0],
            self.semantic_condition_channels,
        ):
            raise ValueError("semantic condition must have shape [B,512]")
        projected = self.semantic_projection(semantic_condition)
        spatial_semantic = projected[:, :, None, None, None].expand(
            -1, -1, *noisy_latent.shape[-3:]
        )
        network_input = torch.cat(
            (noisy_latent, spatial_condition, spatial_semantic), dim=1
        )
        return network_input, spatial_semantic

    def _run_module(self, module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        spatial_condition: torch.Tensor,
        semantic_condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> DenoiserResult:
        network_input, spatial_semantic = self.prepare_input(
            noisy_latent, spatial_condition, semantic_condition
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
        for block1, block2, spatial_attention, temporal_attention, downsample in self.downs:
            hidden = self._run_module(block1, hidden, time_embedding)
            hidden = self._run_module(block2, hidden, time_embedding)
            hidden = self._run_module(spatial_attention, hidden)
            hidden = self._run_module(temporal_attention, hidden, positional_bias)
            skips.append(hidden)
            hidden = downsample(hidden)

        hidden = self._run_module(self.middle_block1, hidden, time_embedding)
        hidden = self._run_module(self.middle_spatial_attention, hidden)
        hidden = self._run_module(
            self.middle_temporal_attention, hidden, positional_bias
        )
        hidden = self._run_module(self.middle_block2, hidden, time_embedding)

        for block1, block2, spatial_attention, temporal_attention, upsample in self.ups:
            hidden = torch.cat((hidden, skips.pop()), dim=1)
            hidden = self._run_module(block1, hidden, time_embedding)
            hidden = self._run_module(block2, hidden, time_embedding)
            hidden = self._run_module(spatial_attention, hidden)
            hidden = self._run_module(temporal_attention, hidden, positional_bias)
            hidden = upsample(hidden)

        predicted_noise = self.final(torch.cat((hidden, residual), dim=1))
        return DenoiserResult(
            predicted_noise=predicted_noise,
            network_input=network_input,
            spatial_semantic=spatial_semantic,
        )
