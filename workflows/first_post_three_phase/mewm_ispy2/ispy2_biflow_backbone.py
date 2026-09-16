from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


ISPY2_BIFLOW_PRESET = "biflownet_original_single_res_ispy2_v1"


@dataclass(frozen=True)
class ISPY2BiFlowPreset:
    """Versioned BiFlowNet architecture; run configs select but do not edit it."""

    name: str = ISPY2_BIFLOW_PRESET
    dim: int = 72
    dim_mults: tuple[int, ...] = (1, 1, 2, 4, 8)
    sub_volume_size: tuple[int, int, int] = (8, 8, 8)
    patch_size: tuple[int, int, int] = (1, 1, 1)
    local_encoder_blocks: int = 2
    local_mid_blocks: int = 1
    local_decoder_blocks: int = 2
    dit_heads: int = 8
    attention_heads: int = 8
    mlp_ratio: float = 4.0
    norm_groups: int = 24
    attention_levels: tuple[bool, ...] = (False, False, False, True, True)
    downsample_after: tuple[bool, ...] = (False, True, True, True, False)
    upsample_after: tuple[bool, ...] = (True, True, True, False, False)
    init_kernel_size: int = 3
    context_pool_heads: int = 8

    def validate(self) -> None:
        levels = len(self.dim_mults)
        if (
            self.name != ISPY2_BIFLOW_PRESET
            or self.dim <= 0
            or self.dim % 6
            or levels == 0
            or len(self.attention_levels) != levels
            or len(self.downsample_after) != levels
            or len(self.upsample_after) != levels
            or self.local_encoder_blocks != 2
            or self.local_decoder_blocks != 2
            or self.local_mid_blocks != 1
            or self.init_kernel_size % 2 != 1
        ):
            raise ValueError("I-SPY2 BiFlowNet preset contract is invalid")
        widths = (self.dim, *(self.dim * value for value in self.dim_mults))
        if any(value <= 0 or value % self.norm_groups for value in widths):
            raise ValueError("I-SPY2 BiFlowNet widths must support GroupNorm")
        if self.dim % self.dit_heads or self.dim % self.context_pool_heads:
            raise ValueError("I-SPY2 BiFlowNet attention heads do not divide width")
        if any(value <= 0 for value in (*self.sub_volume_size, *self.patch_size)):
            raise ValueError("I-SPY2 BiFlowNet patch sizes must be positive")
        if any(
            sub % patch
            for sub, patch in zip(
                self.sub_volume_size, self.patch_size, strict=True
            )
        ):
            raise ValueError("I-SPY2 DiT patches must tile each sub-volume axis")
        if sum(self.downsample_after) != sum(self.upsample_after):
            raise ValueError("I-SPY2 BiFlowNet down/up sampling is asymmetric")

    def payload(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


ORIGINAL_ISPY2_BIFLOW_PRESET = ISPY2BiFlowPreset()
ORIGINAL_ISPY2_BIFLOW_PRESET.validate()


@dataclass(frozen=True)
class ISPY2BiFlowControlResiduals:
    stem: torch.Tensor
    down: tuple[torch.Tensor, ...]
    middle: torch.Tensor


def get_ispy2_biflow_preset(name: str) -> ISPY2BiFlowPreset:
    if name != ISPY2_BIFLOW_PRESET:
        raise ValueError(f"unsupported I-SPY2 BiFlowNet preset: {name}")
    return ORIGINAL_ISPY2_BIFLOW_PRESET


class _SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("sinusoidal time width must be even and at least four")
        self.dim = dim

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10_000.0) / (half - 1)
        frequencies = torch.exp(
            -scale
            * torch.arange(half, device=value.device, dtype=torch.float32)
        ).to(dtype=value.dtype)
        phase = value[:, None] * frequencies[None]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


def _modulate(
    value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return value * (1.0 + scale[:, None]) + shift[:, None]


class _TokenMLP(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class _DiTBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        condition_dim: int,
        heads: int,
        mlp_ratio: float,
        use_skip: bool,
    ) -> None:
        super().__init__()
        self.skip = nn.Linear(2 * dim, dim) if use_skip else None
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attention_subvolume_batch = 0
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = _TokenMLP(dim, mlp_ratio)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(condition_dim, 6 * dim)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        value: torch.Tensor,
        condition: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.skip is not None:
            if skip is None or skip.shape != value.shape:
                raise ValueError("I-SPY2 decoder DiT skip is invalid")
            value = self.skip(torch.cat((value, skip), dim=-1))
        elif skip is not None:
            raise ValueError("I-SPY2 encoder DiT received an unexpected skip")
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(
            condition
        ).chunk(6, dim=-1)
        attention_input = _modulate(self.norm1(value), shift_a, scale_a)
        if self.attention_subvolume_batch:
            outputs = []
            for chunk in attention_input.split(self.attention_subvolume_batch, dim=0):
                outputs.append(
                    activation_checkpoint(self._self_attention, chunk, use_reentrant=False)
                    if torch.is_grad_enabled()
                    else self._self_attention(chunk)
                )
            attended = torch.cat(outputs, dim=0)
        else:
            attended = self._self_attention(attention_input)
        value = value + gate_a[:, None] * attended
        value = value + gate_m[:, None] * self.mlp(
            _modulate(self.norm2(value), shift_m, scale_m)
        )
        return value

    def _self_attention(self, value: torch.Tensor) -> torch.Tensor:
        return self.attention(value, value, value, need_weights=False)[0]


class _LocalFeatureHead(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        condition_dim: int,
        patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(condition_dim, 2 * dim)
        )
        self.patch_size = patch_size
        self.linear = nn.Linear(dim, dim * math.prod(patch_size))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm(value), shift, scale))


class _ContextPool(nn.Module):
    def __init__(self, *, context_dim: int, output_dim: int, heads: int) -> None:
        super().__init__()
        self.context_norm = nn.LayerNorm(context_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, output_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(
            output_dim,
            heads,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        key_padding_mask = None
        if mask is not None:
            if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
                raise ValueError("I-SPY2 context mask must be boolean [B,N]")
            if not bool(mask.any(dim=1).all()):
                raise ValueError("I-SPY2 context mask must retain at least one token")
            key_padding_mask = ~mask
        query = self.query.expand(batch, -1, -1).to(dtype=tokens.dtype)
        pooled = self.attention(
            query,
            self.context_norm(tokens),
            self.context_norm(tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0][:, 0]
        return self.output_norm(pooled)


class _ChannelNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        variance, mean = torch.var_mean(value, dim=1, correction=0, keepdim=True)
        return (value - mean) * torch.rsqrt(variance + self.eps) * self.scale


class _Attention3d(nn.Module):
    def __init__(self, channels: int, heads: int, head_dim: int = 32) -> None:
        super().__init__()
        self.norm = _ChannelNorm(channels)
        self.heads = heads
        self.head_dim = head_dim
        hidden = heads * head_dim
        self.qkv = nn.Linear(channels, 3 * hidden, bias=False)
        self.output = nn.Conv3d(hidden, channels, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        batch, _, depth, height, width = value.shape
        tokens = self.norm(value).flatten(2).transpose(1, 2)
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)

        def split_heads(item: torch.Tensor) -> torch.Tensor:
            return item.reshape(
                batch, -1, self.heads, self.head_dim
            ).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            split_heads(q), split_heads(k), split_heads(v)
        )
        attended = attended.transpose(1, 2).reshape(batch, -1, self.heads * self.head_dim)
        attended = attended.transpose(1, 2).reshape(
            batch, self.heads * self.head_dim, depth, height, width
        )
        return residual + self.output(attended)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU()

    def forward(
        self,
        value: torch.Tensor,
        scale_shift: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        value = self.norm(self.conv(value))
        if scale_shift is not None:
            scale, shift = scale_shift
            value = value * (1.0 + scale[:, :, None, None, None])
            value = value + shift[:, :, None, None, None]
        return self.activation(value)


class _ResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        groups: int,
        condition_dim: int | None,
    ) -> None:
        super().__init__()
        self.condition = (
            nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2 * out_channels))
            if condition_dim is not None
            else None
        )
        self.block1 = _ConvBlock(in_channels, out_channels, groups)
        self.block2 = _ConvBlock(out_channels, out_channels, groups)
        self.residual = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, value: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        scale_shift = None
        if self.condition is not None:
            if condition is None:
                raise ValueError("I-SPY2 BiFlowNet condition is missing")
            scale_shift = self.condition(condition).chunk(2, dim=-1)
        elif condition is not None:
            raise ValueError("unconditioned I-SPY2 block received a condition")
        hidden = self.block1(value, scale_shift)
        hidden = self.block2(hidden)
        return hidden + self.residual(value)


class _DownStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        condition_dim: int,
        groups: int,
        attention: bool,
        heads: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.block1 = _ResnetBlock3d(
            in_channels,
            out_channels,
            groups=groups,
            condition_dim=condition_dim,
        )
        self.attention1 = _Attention3d(out_channels, heads) if attention else nn.Identity()
        self.block2 = _ResnetBlock3d(
            out_channels,
            out_channels,
            groups=groups,
            condition_dim=condition_dim,
        )
        self.attention2 = _Attention3d(out_channels, heads) if attention else nn.Identity()
        self.downsample = (
            nn.Conv3d(out_channels, out_channels, 4, stride=2, padding=1)
            if downsample
            else nn.Identity()
        )

    def forward(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = self.attention1(self.block1(value, condition))
        second = self.attention2(self.block2(first, condition))
        return self.downsample(second), first, second


class _UpStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        condition_dim: int,
        groups: int,
        attention: bool,
        heads: int,
        upsample: bool,
    ) -> None:
        super().__init__()
        self.block1 = _ResnetBlock3d(
            out_channels * 2,
            out_channels,
            groups=groups,
            condition_dim=condition_dim,
        )
        self.attention1 = _Attention3d(out_channels, heads) if attention else nn.Identity()
        self.block2 = _ResnetBlock3d(
            out_channels * 2,
            in_channels,
            groups=groups,
            condition_dim=condition_dim,
        )
        self.attention2 = _Attention3d(in_channels, heads) if attention else nn.Identity()
        self.upsample = (
            nn.ConvTranspose3d(in_channels, in_channels, 4, stride=2, padding=1)
            if upsample
            else nn.Identity()
        )

    def forward(
        self,
        value: torch.Tensor,
        skip_second: torch.Tensor,
        skip_first: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        value = self.attention1(
            self.block1(torch.cat((value, skip_second), dim=1), condition)
        )
        value = self.attention2(
            self.block2(torch.cat((value, skip_first), dim=1), condition)
        )
        return self.upsample(value)


class ISPY2ConditionalBiFlowNet(nn.Module):
    """Original single-resolution BiFlowNet adapted to I-SPY2 RF conditions."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        context_dim: int,
        preset: ISPY2BiFlowPreset = ORIGINAL_ISPY2_BIFLOW_PRESET,
    ) -> None:
        super().__init__()
        preset.validate()
        if input_channels <= 0 or output_channels <= 0 or context_dim <= 0:
            raise ValueError("I-SPY2 BiFlowNet channel dimensions must be positive")
        self.preset = preset
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.context_dim = context_dim
        self.time_dim = preset.dim * 4
        self.condition_dim = self.time_dim * 2
        self.local_dit_fp32 = False

        padding = preset.init_kernel_size // 2
        self.input_stem = nn.Conv3d(
            input_channels,
            preset.dim,
            preset.init_kernel_size,
            padding=padding,
        )
        self.time_mlp = nn.Sequential(
            _SinusoidalTimeEmbedding(preset.dim),
            nn.Linear(preset.dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.context_pool = _ContextPool(
            context_dim=context_dim,
            output_dim=self.time_dim,
            heads=preset.context_pool_heads,
        )

        self.local_embed = nn.Conv3d(
            input_channels,
            preset.dim,
            kernel_size=preset.patch_size,
            stride=preset.patch_size,
        )
        local_grid = tuple(
            sub // patch
            for sub, patch in zip(
                preset.sub_volume_size, preset.patch_size, strict=True
            )
        )
        self.register_buffer(
            "local_position",
            self._position_encoding(preset.dim, local_grid),
            persistent=True,
        )
        self.local_encoder = nn.ModuleList(
            [
                _DiTBlock(
                    dim=preset.dim,
                    condition_dim=self.condition_dim,
                    heads=preset.dit_heads,
                    mlp_ratio=preset.mlp_ratio,
                    use_skip=False,
                )
                for _ in range(preset.local_encoder_blocks)
            ]
        )
        self.local_encoder_heads = nn.ModuleList(
            [
                _LocalFeatureHead(
                    dim=preset.dim,
                    condition_dim=self.condition_dim,
                    patch_size=preset.patch_size,
                )
                for _ in range(preset.local_encoder_blocks)
            ]
        )
        self.local_mid = nn.ModuleList(
            [
                _DiTBlock(
                    dim=preset.dim,
                    condition_dim=self.condition_dim,
                    heads=preset.dit_heads,
                    mlp_ratio=preset.mlp_ratio,
                    use_skip=False,
                )
                for _ in range(preset.local_mid_blocks)
            ]
        )
        self.local_decoder = nn.ModuleList(
            [
                _DiTBlock(
                    dim=preset.dim,
                    condition_dim=self.condition_dim,
                    heads=preset.dit_heads,
                    mlp_ratio=preset.mlp_ratio,
                    use_skip=True,
                )
                for _ in range(preset.local_decoder_blocks)
            ]
        )
        self.local_decoder_heads = nn.ModuleList(
            [
                _LocalFeatureHead(
                    dim=preset.dim,
                    condition_dim=self.condition_dim,
                    patch_size=preset.patch_size,
                )
                for _ in range(preset.local_decoder_blocks)
            ]
        )

        widths = (preset.dim, *(preset.dim * value for value in preset.dim_mults))
        in_out = tuple(zip(widths[:-1], widths[1:], strict=True))
        self.down_stages = nn.ModuleList(
            [
                _DownStage(
                    dim_in,
                    dim_out,
                    condition_dim=self.condition_dim,
                    groups=preset.norm_groups,
                    attention=preset.attention_levels[index],
                    heads=preset.attention_heads,
                    downsample=preset.downsample_after[index],
                )
                for index, (dim_in, dim_out) in enumerate(in_out)
            ]
        )
        mid_dim = widths[-1]
        self.mid_block1 = _ResnetBlock3d(
            mid_dim,
            mid_dim,
            groups=preset.norm_groups,
            condition_dim=self.condition_dim,
        )
        self.mid_attention = _Attention3d(mid_dim, preset.attention_heads)
        self.mid_block2 = _ResnetBlock3d(
            mid_dim,
            mid_dim,
            groups=preset.norm_groups,
            condition_dim=self.condition_dim,
        )
        reversed_in_out = tuple(reversed(in_out))
        self.up_stages = nn.ModuleList(
            [
                _UpStage(
                    dim_in,
                    dim_out,
                    condition_dim=self.condition_dim,
                    groups=preset.norm_groups,
                    attention=preset.attention_levels[-index - 1],
                    heads=preset.attention_heads,
                    upsample=preset.upsample_after[index],
                )
                for index, (dim_in, dim_out) in enumerate(reversed_in_out)
            ]
        )
        self.output_head = nn.Sequential(
            _ResnetBlock3d(
                2 * preset.dim,
                preset.dim,
                groups=preset.norm_groups,
                condition_dim=None,
            ),
            nn.Conv3d(preset.dim, output_channels, kernel_size=1),
        )
        self._initialize_original_weights()

    def _initialize_original_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(
            self.local_embed.weight.reshape(self.local_embed.weight.shape[0], -1)
        )
        for block in (*self.local_encoder, *self.local_mid, *self.local_decoder):
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        for head in (*self.local_encoder_heads, *self.local_decoder_heads):
            nn.init.zeros_(head.modulation[-1].weight)
            nn.init.zeros_(head.modulation[-1].bias)
            nn.init.zeros_(head.linear.weight)
            nn.init.zeros_(head.linear.bias)

    @staticmethod
    def _position_encoding(
        dim: int, grid: tuple[int, int, int]
    ) -> torch.Tensor:
        axis_dim = dim // 3
        if dim % 6 or axis_dim % 2:
            raise ValueError("I-SPY2 3D position width must be divisible by six")
        coordinates = torch.meshgrid(
            *(torch.arange(size, dtype=torch.float32) for size in grid),
            indexing="ij",
        )
        omega = torch.arange(axis_dim // 2, dtype=torch.float32)
        omega = torch.pow(10_000.0, -omega / (axis_dim / 2))
        axes = []
        for coordinate in coordinates:
            phase = coordinate.reshape(-1, 1) * omega.reshape(1, -1)
            axes.append(torch.cat((phase.sin(), phase.cos()), dim=-1))
        return torch.cat(axes, dim=-1).unsqueeze(0)

    @property
    def architecture_contract(self) -> dict[str, object]:
        return {
            "preset": self.preset.payload(),
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "context_dim": self.context_dim,
            "spatial_input": "flow_state_only",
        }

    def _padding(self, spatial: tuple[int, int, int]) -> tuple[int, int, int]:
        downsample = 2 ** sum(self.preset.downsample_after)
        multiples = tuple(
            math.lcm(sub, downsample) for sub in self.preset.sub_volume_size
        )
        return tuple((-size) % multiple for size, multiple in zip(spatial, multiples))

    def _split_sub_volumes(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        batch, channels, depth, height, width = value.shape
        sub_d, sub_h, sub_w = self.preset.sub_volume_size
        if depth % sub_d or height % sub_h or width % sub_w:
            raise ValueError("I-SPY2 padded latent is not tiled by sub-volumes")
        grid = (depth // sub_d, height // sub_h, width // sub_w)
        value = value.reshape(
            batch, channels, grid[0], sub_d, grid[1], sub_h, grid[2], sub_w
        )
        value = value.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return value.reshape(batch * math.prod(grid), channels, sub_d, sub_h, sub_w), grid

    @staticmethod
    def _stitch_sub_volumes(
        value: torch.Tensor,
        *,
        batch: int,
        grid: tuple[int, int, int],
    ) -> torch.Tensor:
        _, channels, sub_d, sub_h, sub_w = value.shape
        value = value.reshape(
            batch, grid[0], grid[1], grid[2], channels, sub_d, sub_h, sub_w
        )
        value = value.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return value.reshape(
            batch,
            channels,
            grid[0] * sub_d,
            grid[1] * sub_h,
            grid[2] * sub_w,
        )

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        patch_d, patch_h, patch_w = self.preset.patch_size
        sub_d, sub_h, sub_w = self.preset.sub_volume_size
        grid = (sub_d // patch_d, sub_h // patch_h, sub_w // patch_w)
        expected_tokens = math.prod(grid)
        expected_features = self.preset.dim * math.prod(self.preset.patch_size)
        if tokens.shape[1:] != (expected_tokens, expected_features):
            raise RuntimeError("I-SPY2 local feature token shape changed")
        value = tokens.reshape(
            tokens.shape[0],
            grid[0],
            grid[1],
            grid[2],
            patch_d,
            patch_h,
            patch_w,
            self.preset.dim,
        )
        value = value.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return value.reshape(tokens.shape[0], self.preset.dim, sub_d, sub_h, sub_w)

    def _local_features(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> list[torch.Tensor]:
        if self.local_dit_fp32:
            with torch.autocast(value.device.type, enabled=False):
                if torch.is_grad_enabled():
                    return activation_checkpoint(
                        self._local_features_impl, value.float(), condition.float(),
                        use_reentrant=False,
                    )
                return self._local_features_impl(value.float(), condition.float())
        return self._local_features_impl(value, condition)

    def _local_features_impl(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> list[torch.Tensor]:
        batch = value.shape[0]
        sub_volumes, grid = self._split_sub_volumes(value)
        local_condition = condition[:, None].expand(
            batch, math.prod(grid), self.condition_dim
        ).reshape(-1, self.condition_dim)
        tokens = self.local_embed(sub_volumes).flatten(2).transpose(1, 2)
        tokens = tokens + self.local_position.to(device=tokens.device, dtype=tokens.dtype)
        skips: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        for block, head in zip(
            self.local_encoder, self.local_encoder_heads, strict=True
        ):
            tokens = block(tokens, local_condition)
            skips.append(tokens)
            features.append(
                self._stitch_sub_volumes(
                    self._unpatchify(head(tokens, local_condition)),
                    batch=batch,
                    grid=grid,
                )
            )
        for block in self.local_mid:
            tokens = block(tokens, local_condition)
        for block, head in zip(
            self.local_decoder, self.local_decoder_heads, strict=True
        ):
            tokens = block(tokens, local_condition, skips.pop())
            features.append(
                self._stitch_sub_volumes(
                    self._unpatchify(head(tokens, local_condition)),
                    batch=batch,
                    grid=grid,
                )
            )
        if skips or len(features) != 4:
            raise RuntimeError("I-SPY2 BiFlowNet local fusion contract changed")
        return features

    def forward(
        self,
        sample: torch.Tensor,
        flow_time: torch.Tensor,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        control: ISPY2BiFlowControlResiduals | None = None,
    ) -> torch.Tensor:
        if sample.ndim != 5 or sample.shape[1] != self.input_channels:
            raise ValueError("I-SPY2 BiFlowNet sample must be [B,C,D,H,W]")
        batch = sample.shape[0]
        if (
            flow_time.shape != (batch,)
            or not bool(torch.isfinite(flow_time).all())
            or not bool(((flow_time >= 0.0) & (flow_time <= 1.0)).all())
        ):
            raise ValueError("I-SPY2 RF time must be finite [B] in [0,1]")
        if (
            context.ndim != 3
            or context.shape[0] != batch
            or context.shape[2] != self.context_dim
            or not bool(torch.isfinite(context).all())
        ):
            raise ValueError("I-SPY2 context tokens are invalid")
        original_spatial = tuple(int(value) for value in sample.shape[2:])
        pad_d, pad_h, pad_w = self._padding(original_spatial)
        padded = F.pad(sample, (0, pad_w, 0, pad_h, 0, pad_d))
        time = self.time_mlp(flow_time.to(device=sample.device, dtype=sample.dtype))
        pooled = self.context_pool(
            context.to(device=sample.device, dtype=sample.dtype), context_mask
        )
        condition = torch.cat((time, pooled), dim=-1)

        local_features = self._local_features(padded, condition)
        value = self.input_stem(padded)
        residual = value
        if control is not None:
            if control.stem.shape != residual.shape:
                raise ValueError("I-SPY2 ControlNet stem residual shape is invalid")
            if len(control.down) != 2 * len(self.down_stages):
                raise ValueError("I-SPY2 ControlNet down residual count is invalid")
            # The control branch modifies decoder skips, not the backbone encoder state.
            residual = residual + control.stem
        global_skips: list[torch.Tensor] = []
        for index, stage in enumerate(self.down_stages):
            if index < 2:
                value = value + local_features[index]
            value, first, second = stage(value, condition)
            if control is not None:
                first_control = control.down[2 * index]
                second_control = control.down[2 * index + 1]
                if (
                    first_control.shape != first.shape
                    or second_control.shape != second.shape
                ):
                    raise ValueError(
                        "I-SPY2 ControlNet encoder residual shape is invalid"
                    )
                first = first + first_control
                second = second + second_control
            global_skips.extend((first, second))
        value = self.mid_block1(value, condition)
        value = self.mid_attention(value)
        value = self.mid_block2(value, condition)
        if control is not None:
            if control.middle.shape != value.shape:
                raise ValueError("I-SPY2 ControlNet middle residual shape is invalid")
            value = value + control.middle
        for index, stage in enumerate(self.up_stages):
            if index >= len(self.up_stages) - 2:
                value = value + local_features[index - (len(self.up_stages) - 4)]
            value = stage(
                value,
                global_skips.pop(),
                global_skips.pop(),
                condition,
            )
        if global_skips:
            raise RuntimeError("I-SPY2 BiFlowNet global skip contract changed")
        velocity = self.output_head(torch.cat((value, residual), dim=1))
        return velocity[
            :, :, : original_spatial[0], : original_spatial[1], : original_spatial[2]
        ]


def _zero_conv3d(channels: int) -> nn.Conv3d:
    layer = nn.Conv3d(channels, channels, kernel_size=1)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class ISPY2BiFlowControlNet(nn.Module):
    """BiFlowNet encoder clone conditioned by raw DCE0 and SER volumes."""

    def __init__(
        self,
        backbone: ISPY2ConditionalBiFlowNet,
        *,
        spatial_condition_channels: int = 2,
    ) -> None:
        super().__init__()
        if spatial_condition_channels <= 0:
            raise ValueError("I-SPY2 ControlNet condition channels must be positive")
        self.preset = backbone.preset
        self.input_channels = backbone.input_channels
        self.context_dim = backbone.context_dim
        self.condition_dim = backbone.condition_dim
        self.spatial_condition_channels = int(spatial_condition_channels)
        self.local_dit_fp32 = False

        dim = self.preset.dim
        self.condition_encoder = nn.Sequential(
            nn.Conv3d(spatial_condition_channels, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(32, dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(dim, dim, 3, padding=1),
        )
        nn.init.zeros_(self.condition_encoder[-1].weight)
        nn.init.zeros_(self.condition_encoder[-1].bias)

        self.time_mlp = copy.deepcopy(backbone.time_mlp)
        self.context_pool = copy.deepcopy(backbone.context_pool)
        self.input_stem = copy.deepcopy(backbone.input_stem)
        self.local_embed = copy.deepcopy(backbone.local_embed)
        self.local_encoder = copy.deepcopy(backbone.local_encoder)
        self.local_encoder_heads = copy.deepcopy(backbone.local_encoder_heads)
        self.register_buffer(
            "local_position", backbone.local_position.detach().clone(), persistent=True
        )
        self.down_stages = copy.deepcopy(backbone.down_stages)
        self.mid_block1 = copy.deepcopy(backbone.mid_block1)
        self.mid_attention = copy.deepcopy(backbone.mid_attention)
        self.mid_block2 = copy.deepcopy(backbone.mid_block2)

        widths = tuple(self.preset.dim * value for value in self.preset.dim_mults)
        self.zero_stem = _zero_conv3d(self.preset.dim)
        self.zero_down = nn.ModuleList(
            [
                nn.ModuleList((_zero_conv3d(width), _zero_conv3d(width)))
                for width in widths
            ]
        )
        self.zero_middle = _zero_conv3d(widths[-1])

    @property
    def architecture_contract(self) -> dict[str, object]:
        return {
            "type": "biflownet_encoder_clone_zero_conv",
            "spatial_condition": "raw_dce0_ser",
            "spatial_condition_channels": self.spatial_condition_channels,
            "condition_downsample_factor": 4,
            "condition_embedding_channels": self.preset.dim,
            "zero_initialized": True,
            "residual_sites": 2 + 2 * len(self.down_stages),
        }

    def encode_spatial_condition(self, source_mri: torch.Tensor) -> torch.Tensor:
        if (
            source_mri.ndim != 5
            or source_mri.shape[1] != self.spatial_condition_channels
            or not source_mri.is_floating_point()
            or not bool(torch.isfinite(source_mri).all())
        ):
            raise ValueError("I-SPY2 ControlNet condition must be finite [B,2,D,H,W]")
        return self.condition_encoder(source_mri)

    def _local_encoder_features(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> list[torch.Tensor]:
        if self.local_dit_fp32:
            with torch.autocast(value.device.type, enabled=False):
                if torch.is_grad_enabled():
                    return activation_checkpoint(
                        self._local_encoder_features_impl,
                        value.float(), condition.float(), use_reentrant=False,
                    )
                return self._local_encoder_features_impl(value.float(), condition.float())
        return self._local_encoder_features_impl(value, condition)

    def _local_encoder_features_impl(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> list[torch.Tensor]:
        batch = value.shape[0]
        sub_volumes, grid = ISPY2ConditionalBiFlowNet._split_sub_volumes(self, value)
        local_condition = condition[:, None].expand(
            batch, math.prod(grid), self.condition_dim
        ).reshape(-1, self.condition_dim)
        tokens = self.local_embed(sub_volumes).flatten(2).transpose(1, 2)
        tokens = tokens + self.local_position.to(
            device=tokens.device, dtype=tokens.dtype
        )
        features: list[torch.Tensor] = []
        for block, head in zip(
            self.local_encoder, self.local_encoder_heads, strict=True
        ):
            tokens = block(tokens, local_condition)
            local = ISPY2ConditionalBiFlowNet._unpatchify(
                self, head(tokens, local_condition)
            )
            features.append(
                ISPY2ConditionalBiFlowNet._stitch_sub_volumes(
                    local, batch=batch, grid=grid
                )
            )
        return features

    def forward(
        self,
        sample: torch.Tensor,
        flow_time: torch.Tensor,
        *,
        context: torch.Tensor,
        spatial_condition: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> ISPY2BiFlowControlResiduals:
        if sample.ndim != 5 or sample.shape[1] != self.input_channels:
            raise ValueError("I-SPY2 ControlNet state must be [B,C,D,H,W]")
        if (
            flow_time.shape != (sample.shape[0],)
            or not bool(torch.isfinite(flow_time).all())
            or not bool(((flow_time >= 0.0) & (flow_time <= 1.0)).all())
        ):
            raise ValueError("I-SPY2 ControlNet RF time must be finite [B] in [0,1]")
        if (
            context.ndim != 3
            or context.shape[0] != sample.shape[0]
            or context.shape[2] != self.context_dim
            or not bool(torch.isfinite(context).all())
        ):
            raise ValueError("I-SPY2 ControlNet context tokens are invalid")
        guided_hint = self.encode_spatial_condition(
            spatial_condition.to(device=sample.device, dtype=sample.dtype)
        )
        if (
            guided_hint.shape[0] != sample.shape[0]
            or guided_hint.shape[2:] != sample.shape[2:]
        ):
            raise ValueError(
                "I-SPY2 DCE0+SER condition must be four times the latent spatial shape"
            )

        original_spatial = tuple(int(value) for value in sample.shape[2:])
        pad_d, pad_h, pad_w = ISPY2ConditionalBiFlowNet._padding(
            self, original_spatial
        )
        padded = F.pad(sample, (0, pad_w, 0, pad_h, 0, pad_d))
        guided_hint = F.pad(guided_hint, (0, pad_w, 0, pad_h, 0, pad_d))
        time = self.time_mlp(
            flow_time.to(device=sample.device, dtype=sample.dtype)
        )
        pooled = self.context_pool(
            context.to(device=sample.device, dtype=sample.dtype), context_mask
        )
        condition = torch.cat((time, pooled), dim=-1)

        local_features = self._local_encoder_features(padded, condition)
        value = self.input_stem(padded) + guided_hint
        stem = self.zero_stem(value)
        down: list[torch.Tensor] = []
        for index, (stage, zero_layers) in enumerate(
            zip(self.down_stages, self.zero_down, strict=True)
        ):
            if index < len(local_features):
                value = value + local_features[index]
            value, first, second = stage(value, condition)
            down.extend((zero_layers[0](first), zero_layers[1](second)))
        value = self.mid_block1(value, condition)
        value = self.mid_attention(value)
        value = self.mid_block2(value, condition)
        return ISPY2BiFlowControlResiduals(
            stem=stem,
            down=tuple(down),
            middle=self.zero_middle(value),
        )


class ISPY2ControlledBiFlowNet(nn.Module):
    """BiFlowNet dynamics with a raw-image ControlNet side branch."""

    def __init__(
        self,
        backbone: ISPY2ConditionalBiFlowNet,
        controlnet: ISPY2BiFlowControlNet,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.controlnet = controlnet

    @property
    def architecture_contract(self) -> dict[str, object]:
        return {
            "spatial_input": "flow_state_only",
            "input_channels": self.backbone.input_channels,
            "output_channels": self.backbone.output_channels,
            "backbone": self.backbone.architecture_contract,
            "controlnet": self.controlnet.architecture_contract,
        }

    def forward(
        self,
        sample: torch.Tensor,
        flow_time: torch.Tensor,
        *,
        context: torch.Tensor,
        spatial_condition: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        control = self.controlnet(
            sample,
            flow_time,
            context=context,
            spatial_condition=spatial_condition,
            context_mask=context_mask,
        )
        return self.backbone(
            sample,
            flow_time,
            context=context,
            context_mask=context_mask,
            control=control,
        )


def enable_local_dit_fp32(dynamics: nn.Module, *, subvolume_batch: int) -> None:
    if type(subvolume_batch) is not int or subvolume_batch <= 0:
        raise ValueError("FP32 local attention requires a positive sub-volume batch")
    branches = (dynamics.backbone, dynamics.controlnet)
    if not isinstance(branches[0], ISPY2ConditionalBiFlowNet) or not isinstance(
        branches[1], ISPY2BiFlowControlNet
    ):
        raise TypeError("FP32 local computation requires the original BiFlow branches")
    for branch in branches:
        branch.local_dit_fp32 = True
        for module in branch.modules():
            if isinstance(module, _DiTBlock):
                module.attention_subvolume_batch = subvolume_batch


def build_original_ispy2_biflownet(
    *, input_channels: int, output_channels: int, context_dim: int
) -> ISPY2ConditionalBiFlowNet:
    return ISPY2ConditionalBiFlowNet(
        input_channels=input_channels,
        output_channels=output_channels,
        context_dim=context_dim,
        preset=ORIGINAL_ISPY2_BIFLOW_PRESET,
    )


def build_original_ispy2_controlled_biflownet(
    *,
    input_channels: int,
    output_channels: int,
    context_dim: int,
    spatial_condition_channels: int = 2,
) -> ISPY2ControlledBiFlowNet:
    backbone = build_original_ispy2_biflownet(
        input_channels=input_channels,
        output_channels=output_channels,
        context_dim=context_dim,
    )
    return ISPY2ControlledBiFlowNet(
        backbone,
        ISPY2BiFlowControlNet(
            backbone, spatial_condition_channels=spatial_condition_channels
        ),
    )


__all__ = [
    "ISPY2_BIFLOW_PRESET",
    "ISPY2BiFlowControlNet",
    "ISPY2BiFlowControlResiduals",
    "ISPY2BiFlowPreset",
    "ISPY2ConditionalBiFlowNet",
    "ISPY2ControlledBiFlowNet",
    "ORIGINAL_ISPY2_BIFLOW_PRESET",
    "build_original_ispy2_biflownet",
    "build_original_ispy2_controlled_biflownet",
    "get_ispy2_biflow_preset",
]
