"""Adapter for the MRI VQ-GAN used by the MeWM I-SPY2 latent cache.

The architecture in this module is compatible with ``mewm_ispy2.vqgan`` from
the MeWM source snapshot at commit ``9620f58261346603548cd51a1aad70a6a154077f``.
That source is distributed under CC BY-NC 4.0.  The wrapper adds strict
checkpoint and tensor contracts required by this project.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ispy2_symmflow.utils.hashing import sha256_file


MEWM_VQGAN_CHECKPOINT_SCHEMA = "mewm_ispy2_mri_vqgan_v1"
MEWM_VQGAN_ARCHITECTURE_SCHEMA = "mewm_ispy2_vqgan_architecture_v1"
MEWM_REGISTERED_NUMERIC_CONTRACT = "motfm_registered_global_zscore_v1"
MEWM_REGISTERED_DATA_BACKEND = "registered_t0"
MEWM_REGISTERED_VQGAN_SHA256 = (
    "a91133b264d4aee28f47440067563aab0cef13145c8555272e564329b0ab21c1"
)
MEWM_REGISTERED_VQGAN_DATA_CONTRACT_SHA256 = (
    "52fd47f608f0d777647340a868341cbc6b0b60165b046aaef18fa636b91b0067"
)
MEWM_REGISTERED_CODEBOOK_SHA256 = (
    "b9c2b6a38faa276c5f7be8f955ae5f664293c1f87b5f8d96a2566f985ed9db3a"
)
MEWM_REGISTERED_IMAGE_SHAPE = (96, 256, 256)
MEWM_REGISTERED_LATENT_SHAPE = (8, 24, 64, 64)
MEWM_MU_GLIOMA_MODALITIES = ("t1c", "t1n", "t2f", "t2w")

_HEX = frozenset("0123456789abcdef")


class MeWMVQGANCheckpointError(ValueError):
    """Raised before using an incompatible or incomplete MeWM checkpoint."""


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class MeWMVQGANConfig:
    """Exact generator architecture recorded in a MeWM checkpoint identity."""

    image_channels: int = 1
    hidden_channels: int = 16
    embedding_dim: int = 8
    n_codes: int = 16384
    downsample_factor: int = 4
    bottleneck_blocks: int = 1
    num_groups: int = 32
    commitment_weight: float = 0.25
    ema_decay: float = 0.99
    ema_smoothing: float = 1e-7
    restart_threshold: float = 1.0
    nearest_chunk_size: int = 4096

    def __post_init__(self) -> None:
        integer_fields = (
            "image_channels",
            "hidden_channels",
            "embedding_dim",
            "n_codes",
            "downsample_factor",
            "bottleneck_blocks",
            "num_groups",
            "nearest_chunk_size",
        )
        if any(type(getattr(self, name)) is not int for name in integer_fields):
            raise TypeError("MeWM VQ-GAN integer config fields must be exact integers")
        if self.image_channels != 1:
            raise ValueError("MeWM I-SPY2 VQ-GAN must be single-channel")
        if self.hidden_channels <= 0 or self.embedding_dim <= 0 or self.n_codes <= 0:
            raise ValueError("MeWM VQ-GAN channel and code counts must be positive")
        if self.downsample_factor != 4:
            raise ValueError("MeWM I-SPY2 VQ-GAN downsampling must be exactly 4x")
        if self.bottleneck_blocks < 0:
            raise ValueError("MeWM VQ-GAN bottleneck block count cannot be negative")
        if self.num_groups <= 0 or (self.hidden_channels * 2) % self.num_groups:
            raise ValueError("MeWM VQ-GAN normalized channels must divide into groups")
        if self.nearest_chunk_size <= 0:
            raise ValueError("MeWM VQ-GAN nearest-neighbor chunk size must be positive")
        real_fields = (
            "commitment_weight",
            "ema_decay",
            "ema_smoothing",
            "restart_threshold",
        )
        if any(
            isinstance(getattr(self, name), bool)
            or not isinstance(getattr(self, name), (int, float))
            or not math.isfinite(float(getattr(self, name)))
            for name in real_fields
        ):
            raise TypeError("MeWM VQ-GAN scalar config fields must be finite numbers")
        if self.commitment_weight < 0 or self.restart_threshold < 0:
            raise ValueError("MeWM VQ-GAN loss and restart values cannot be negative")
        if not 0.0 <= self.ema_decay < 1.0 or self.ema_smoothing <= 0:
            raise ValueError("MeWM VQ-GAN EMA parameters are invalid")

    @property
    def in_channels(self) -> int:
        return self.image_channels

    @property
    def out_channels(self) -> int:
        return self.image_channels

    @property
    def latent_channels(self) -> int:
        return self.embedding_dim

    @property
    def compression_factor(self) -> int:
        return self.downsample_factor

    def latent_shape(
        self, spatial_shape: Sequence[int]
    ) -> tuple[int, int, int, int]:
        shape = tuple(int(size) for size in spatial_shape)
        if len(shape) != 3 or any(size <= 0 for size in shape):
            raise ValueError("MeWM VQ-GAN input spatial shape must contain three sizes")
        if any(size % self.downsample_factor for size in shape):
            raise ValueError("input shape must be divisible by the downsampling factor")
        return (
            self.embedding_dim,
            *(size // self.downsample_factor for size in shape),
        )

    def architecture_contract(self) -> dict[str, Any]:
        return {
            "schema_version": MEWM_VQGAN_ARCHITECTURE_SCHEMA,
            "config": asdict(self),
        }

    @classmethod
    def from_architecture_contract(
        cls, payload: Mapping[str, Any]
    ) -> "MeWMVQGANConfig":
        if type(payload) is not dict:
            raise MeWMVQGANCheckpointError(
                "VQ-GAN checkpoint architecture contract must be an exact dictionary"
            )
        if payload.get("schema_version") != MEWM_VQGAN_ARCHITECTURE_SCHEMA:
            raise MeWMVQGANCheckpointError(
                "VQ-GAN checkpoint architecture schema is incompatible"
            )
        raw = payload.get("config")
        expected_fields = {field.name for field in fields(cls)}
        if type(raw) is not dict or set(raw) != expected_fields:
            raise MeWMVQGANCheckpointError(
                "VQ-GAN checkpoint architecture config is incomplete or has extra fields"
            )
        try:
            return cls(**raw)
        except (TypeError, ValueError) as error:
            raise MeWMVQGANCheckpointError(
                "VQ-GAN checkpoint architecture config is invalid"
            ) from error


MEWM_REGISTERED_VQGAN_CONFIG = MeWMVQGANConfig()


def _triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    return (value, value, value) if isinstance(value, int) else value


def _same_pad(
    kernel_size: tuple[int, int, int], stride: tuple[int, int, int]
) -> tuple[int, ...]:
    padding: list[int] = []
    for total in (
        kernel - step for kernel, step in zip(kernel_size, stride, strict=True)
    ):
        padding.extend((total // 2 + total % 2, total // 2))
    depth, height, width = (
        tuple(padding[index : index + 2]) for index in range(0, len(padding), 2)
    )
    return (*width, *height, *depth)


class SamePadConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        bias: bool = True,
        padding_type: str = "replicate",
    ) -> None:
        kernel = _triple(kernel_size)
        steps = _triple(stride)
        super().__init__(
            in_channels,
            out_channels,
            kernel,
            stride=steps,
            padding=0,
            bias=bias,
        )
        self.pad_input = _same_pad(kernel, steps)
        self.padding_type = padding_type

    def forward(self, value: Tensor) -> Tensor:
        if any(self.pad_input):
            value = F.pad(value, self.pad_input, mode=self.padding_type)
        return super().forward(value)


class SamePadConvTranspose3d(nn.ConvTranspose3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        bias: bool = True,
        padding_type: str = "replicate",
    ) -> None:
        kernel = _triple(kernel_size)
        steps = _triple(stride)
        super().__init__(
            in_channels,
            out_channels,
            kernel,
            stride=steps,
            padding=tuple(size - 1 for size in kernel),
            bias=bias,
        )
        self.pad_input = _same_pad(kernel, steps)
        self.padding_type = padding_type

    def forward(
        self, value: Tensor, output_size: list[int] | None = None
    ) -> Tensor:
        if any(self.pad_input):
            value = F.pad(value, self.pad_input, mode=self.padding_type)
        return super().forward(value, output_size=output_size)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.conv1 = SamePadConv3d(channels, channels, 3)
        self.norm2 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.conv2 = SamePadConv3d(channels, channels, 3)

    def forward(self, value: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return value + hidden


def _identity_residual_blocks(
    count: int, *, channels: int, groups: int
) -> nn.ModuleList:
    blocks = nn.ModuleList(
        ResidualBlock3D(channels, groups) for _ in range(count)
    )
    for block in blocks:
        nn.init.zeros_(block.conv2.weight)
        nn.init.zeros_(block.conv2.bias)
    return blocks


class MRIEncoder(nn.Module):
    def __init__(self, config: MeWMVQGANConfig) -> None:
        super().__init__()
        hidden = config.hidden_channels
        groups = config.num_groups
        self.input = SamePadConv3d(1, hidden, 3)
        self.downsamples = nn.ModuleList(
            (
                SamePadConv3d(hidden, hidden * 2, 4, stride=2),
                SamePadConv3d(hidden * 2, hidden * 4, 4, stride=2),
            )
        )
        self.residuals = nn.ModuleList(
            (ResidualBlock3D(hidden * 2, groups), ResidualBlock3D(hidden * 4, groups))
        )
        self.bottleneck = _identity_residual_blocks(
            config.bottleneck_blocks, channels=hidden * 4, groups=groups
        )
        self.final_norm = nn.GroupNorm(groups, hidden * 4, eps=1e-6)
        self.output_channels = hidden * 4

    def forward(self, value: Tensor) -> Tensor:
        hidden = self.input(value)
        for downsample, residual in zip(
            self.downsamples, self.residuals, strict=True
        ):
            hidden = residual(downsample(hidden))
        for block in self.bottleneck:
            hidden = block(hidden)
        return F.silu(self.final_norm(hidden))


class MRIDecoder(nn.Module):
    def __init__(self, config: MeWMVQGANConfig) -> None:
        super().__init__()
        hidden = config.hidden_channels
        groups = config.num_groups
        self.final_norm = nn.GroupNorm(groups, hidden * 4, eps=1e-6)
        self.bottleneck = _identity_residual_blocks(
            config.bottleneck_blocks, channels=hidden * 4, groups=groups
        )
        self.upsamples = nn.ModuleList(
            (
                SamePadConvTranspose3d(hidden * 4, hidden * 4, 4, stride=2),
                SamePadConvTranspose3d(hidden * 4, hidden * 2, 4, stride=2),
            )
        )
        self.residual_one = nn.ModuleList(
            (ResidualBlock3D(hidden * 4, groups), ResidualBlock3D(hidden * 2, groups))
        )
        self.residual_two = nn.ModuleList(
            (ResidualBlock3D(hidden * 4, groups), ResidualBlock3D(hidden * 2, groups))
        )
        self.output = SamePadConv3d(hidden * 2, 1, 3)

    def forward(self, value: Tensor) -> Tensor:
        hidden = F.silu(self.final_norm(value))
        for block in self.bottleneck:
            hidden = block(hidden)
        for upsample, residual_one, residual_two in zip(
            self.upsamples, self.residual_one, self.residual_two, strict=True
        ):
            hidden = residual_two(residual_one(upsample(hidden)))
        return self.output(hidden)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, config: MeWMVQGANConfig) -> None:
        super().__init__()
        embeddings = F.normalize(
            torch.randn(config.n_codes, config.embedding_dim), dim=1
        )
        self.register_buffer("embeddings", embeddings)
        self.register_buffer("cluster_size", torch.ones(config.n_codes))
        self.register_buffer("embedding_sum", embeddings.clone())
        self.n_codes = config.n_codes
        self.embedding_dim = config.embedding_dim
        self.decay = float(config.ema_decay)
        self.smoothing = float(config.ema_smoothing)
        self.restart_threshold = float(config.restart_threshold)
        self.commitment_weight = float(config.commitment_weight)
        self.chunk_size = config.nearest_chunk_size

    def _nearest(self, flat: Tensor) -> Tensor:
        embedding_norm = self.embeddings.square().sum(dim=1)
        indices = []
        for chunk in flat.split(self.chunk_size):
            distances = (
                chunk.square().sum(dim=1, keepdim=True)
                - 2.0 * chunk @ self.embeddings.t()
                + embedding_norm[None]
            )
            indices.append(distances.argmin(dim=1))
        return torch.cat(indices)

    def _restart_candidates(self, flat: Tensor) -> Tensor:
        candidates = flat.detach().to(self.embeddings.dtype)
        if candidates.shape[0] < self.n_codes:
            repeats = math.ceil(self.n_codes / candidates.shape[0])
            candidates = candidates.repeat(repeats, 1)
            candidates = candidates + torch.randn_like(candidates) * (
                0.01 / math.sqrt(self.embedding_dim)
            )
        order = torch.randperm(candidates.shape[0], device=candidates.device)
        return candidates[order[: self.n_codes]]

    def forward(self, latent: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if (
            latent.ndim != 5
            or latent.shape[1] != self.embedding_dim
            or not latent.is_floating_point()
        ):
            raise ValueError("VQ-GAN quantizer input must be floating [B,E,Z,Y,X]")
        flat = latent.movedim(1, -1).reshape(-1, self.embedding_dim)
        indices = self._nearest(flat)
        quantized_flat = F.embedding(indices, self.embeddings)
        if self.training:
            counts = torch.bincount(indices, minlength=self.n_codes).to(
                self.cluster_size.dtype
            )
            sums = torch.zeros_like(self.embedding_sum)
            sums.index_add_(0, indices, flat.detach().to(sums.dtype))
            self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
            self.embedding_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
            total = self.cluster_size.sum()
            smoothed = (
                (self.cluster_size + self.smoothing)
                / (total + self.n_codes * self.smoothing)
                * total
            )
            self.embeddings.copy_(self.embedding_sum / smoothed[:, None])
            underused = self.cluster_size < self.restart_threshold
            if underused.any():
                replacements = self._restart_candidates(flat)
                self.embeddings[underused] = replacements[underused]
        quantized = quantized_flat.reshape(
            *latent.shape[0:1], *latent.shape[2:], self.embedding_dim
        ).movedim(-1, 1)
        commitment = self.commitment_weight * F.mse_loss(
            latent, quantized.detach()
        )
        straight_through = latent + (quantized - latent).detach()
        probabilities = torch.bincount(indices, minlength=self.n_codes).float()
        probabilities = probabilities / probabilities.sum().clamp_min(1.0)
        perplexity = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-10).log()).sum()
        )
        return straight_through, {
            "indices": indices.reshape(latent.shape[0], *latent.shape[2:]),
            "commitment_loss": commitment,
            "perplexity": perplexity,
        }


class MRILevelVQGAN(nn.Module):
    """Generator portion of the upstream MRI-level VQ-GAN."""

    def __init__(
        self, config: MeWMVQGANConfig = MEWM_REGISTERED_VQGAN_CONFIG
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = MRIEncoder(config)
        self.pre_quant = SamePadConv3d(
            self.encoder.output_channels, config.embedding_dim, 1
        )
        self.quantizer = VectorQuantizerEMA(config)
        self.post_quant = SamePadConv3d(
            config.embedding_dim, self.encoder.output_channels, 1
        )
        self.decoder = MRIDecoder(config)

    def encode_continuous(self, image: Tensor) -> Tensor:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("VQ-GAN input must have shape [B,1,Z,Y,X]")
        return self.pre_quant(self.encoder(image))

    def encode(self, image: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        return self.quantizer(self.encode_continuous(image))

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(self.post_quant(latent))

    def forward(self, image: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        latent, diagnostics = self.encode(image)
        return self.decode(latent), diagnostics


def mewm_codebook_sha256(embeddings: Tensor) -> str:
    """Reproduce the fingerprint stored by the paired MeWM latent cache."""

    if (
        not isinstance(embeddings, Tensor)
        or embeddings.ndim != 2
        or embeddings.shape[1] != 8
        or not embeddings.is_floating_point()
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("MeWM codebook embeddings must be finite [N,8] floating data")
    value = embeddings.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(b"mewm_ispy2_v5_codebook_float32_n8_v1\n")
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


class MeWMVQGANCodec(nn.Module):
    """Frozen continuous-latent codec matching SymmFlow's autoencoder API.

    Flow latents are continuous encoder outputs.  Decoding therefore always
    projects them through the learned codebook before invoking the decoder.
    """

    def __init__(
        self,
        config: MeWMVQGANConfig = MEWM_REGISTERED_VQGAN_CONFIG,
        *,
        backend: nn.Module | None = None,
        image_shape: Sequence[int] = MEWM_REGISTERED_IMAGE_SHAPE,
        checkpoint_sha256: str | None = None,
        checkpoint_identity: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.image_shape = tuple(int(size) for size in image_shape)
        self.latent_shape = config.latent_shape(self.image_shape)
        self.backend = backend if backend is not None else MRILevelVQGAN(config)
        shape = (1, config.embedding_dim, 1, 1, 1)
        self.register_buffer("latent_mean", torch.zeros(shape))
        self.register_buffer("latent_std", torch.ones(shape))
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_identity = dict(checkpoint_identity or {})
        self.freeze()

    def train(self, mode: bool = True) -> "MeWMVQGANCodec":
        # The EMA codebook mutates in train mode, so a reused codec stays frozen
        # even when it is registered below another module whose train() is called.
        super().train(False)
        return self

    def _validate_image(self, image: Tensor) -> None:
        if image.ndim != 5:
            raise ValueError("3D MRI tensors must have shape [B, C, D, H, W]")
        if image.shape[1] != self.config.image_channels:
            raise ValueError(
                f"expected {self.config.image_channels} DCE0 channel, got {image.shape[1]}"
            )
        if tuple(image.shape[2:]) != self.image_shape:
            raise ValueError(
                f"expected DCE0 spatial shape {self.image_shape}, got {tuple(image.shape[2:])}"
            )
        if not image.is_floating_point():
            raise TypeError("DCE0 tensors must use a floating-point dtype")

    def _validate_latent(self, latent: Tensor) -> None:
        expected = (self.config.embedding_dim, *self.latent_shape[1:])
        if latent.ndim != 5:
            raise ValueError("3D latent tensors must have shape [B, C, D, H, W]")
        if tuple(latent.shape[1:]) != expected:
            raise ValueError(
                f"expected continuous latent shape [B,{','.join(map(str, expected))}], "
                f"got {tuple(latent.shape)}"
            )
        if not latent.is_floating_point():
            raise TypeError("latent tensors must use a floating-point dtype")

    def encode(self, image: Tensor, *, normalize: bool = False) -> Tensor:
        """Return the continuous pre-quantization latent for one DCE0 visit."""

        self._validate_image(image)
        latent = self.backend.encode_continuous(image)
        self._validate_latent(latent)
        if not bool(torch.isfinite(latent).all()):
            raise RuntimeError("MeWM VQ-GAN encoder returned a non-finite latent")
        return self.normalize_latent(latent) if normalize else latent

    def quantize(self, continuous: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        self._validate_latent(continuous)
        result = self.backend.quantizer(continuous)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError("MeWM VQ-GAN quantizer returned an invalid result")
        quantized, diagnostics = result
        self._validate_latent(quantized)
        if not isinstance(diagnostics, Mapping):
            raise RuntimeError("MeWM VQ-GAN quantizer diagnostics are invalid")
        if not bool(torch.isfinite(quantized).all()):
            raise RuntimeError("MeWM VQ-GAN quantizer returned a non-finite latent")
        return quantized, dict(diagnostics)

    def decode(self, latent: Tensor, *, denormalize: bool = False) -> Tensor:
        """Inverse z-score, quantize, and decode a continuous flow latent."""

        self._validate_latent(latent)
        continuous = self.denormalize_latent(latent) if denormalize else latent
        quantized, _ = self.quantize(continuous)
        reconstruction = self.backend.decode(quantized)
        expected = (latent.shape[0], self.config.image_channels, *self.image_shape)
        if tuple(reconstruction.shape) != expected:
            raise RuntimeError(
                "MeWM VQ-GAN decoder returned an incompatible tensor: "
                f"expected {expected}, got {tuple(reconstruction.shape)}"
            )
        if not bool(torch.isfinite(reconstruction).all()):
            raise RuntimeError("MeWM VQ-GAN decoder returned a non-finite image")
        return reconstruction

    def _canonical_statistic(self, value: Tensor | Sequence[float] | float) -> Tensor:
        statistic = torch.as_tensor(
            value, dtype=self.latent_mean.dtype, device=self.latent_mean.device
        )
        if statistic.numel() == 1:
            statistic = statistic.expand(self.config.embedding_dim)
        if statistic.numel() != self.config.embedding_dim:
            raise ValueError(
                "latent statistics must be scalar or contain one value per latent channel"
            )
        return statistic.reshape(1, self.config.embedding_dim, 1, 1, 1)

    @torch.no_grad()
    def set_latent_statistics(
        self,
        mean: Tensor | Sequence[float] | float,
        std: Tensor | Sequence[float] | float,
    ) -> None:
        canonical_mean = self._canonical_statistic(mean)
        canonical_std = self._canonical_statistic(std)
        if not torch.all(torch.isfinite(canonical_mean)):
            raise ValueError("latent mean must be finite")
        if not torch.all(torch.isfinite(canonical_std)) or torch.any(canonical_std <= 0):
            raise ValueError("latent std must be positive and finite")
        self.latent_mean.copy_(canonical_mean)
        self.latent_std.copy_(canonical_std)

    def normalize_latent(self, latent: Tensor) -> Tensor:
        self._validate_latent(latent)
        return (latent - self.latent_mean.to(latent)) / self.latent_std.to(latent)

    def denormalize_latent(self, latent: Tensor) -> Tensor:
        self._validate_latent(latent)
        return latent * self.latent_std.to(latent) + self.latent_mean.to(latent)

    def encode_normalized(self, image: Tensor) -> Tensor:
        return self.encode(image, normalize=True)

    def decode_normalized(self, latent: Tensor) -> Tensor:
        return self.decode(latent, denormalize=True)

    def forward(self, image: Tensor) -> Tensor:
        return self.decode(self.encode(image))

    def freeze(self) -> "MeWMVQGANCodec":
        self.requires_grad_(False)
        self.train(False)
        return self

    def unfreeze(self) -> "MeWMVQGANCodec":
        raise RuntimeError("the imported MeWM VQ-GAN is a frozen pretrained codec")


class MeWMMultimodalVQGANCodec(nn.Module):
    """Apply one frozen single-channel MeWM VQ-GAN to an ordered MRI state.

    MU-Glioma uses one shared VQ-GAN for four independently encoded MRI
    modalities. SymmFlow sees their continuous latents concatenated along the
    channel dimension. Decoding splits them back into the fixed modality order
    and invokes the shared codebook and decoder once per modality.
    """

    def __init__(
        self,
        codec: MeWMVQGANCodec,
        *,
        modalities: Sequence[str] = MEWM_MU_GLIOMA_MODALITIES,
        latent_mean: Tensor | Sequence[float] | float = 0.0,
        latent_std: Tensor | Sequence[float] | float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(codec, MeWMVQGANCodec):
            raise TypeError("multimodal MeWM codec requires a single-channel codec")
        ordered = tuple(str(value).strip().lower() for value in modalities)
        if ordered != MEWM_MU_GLIOMA_MODALITIES:
            raise ValueError("MU-Glioma modality order must be t1c,t1n,t2f,t2w")
        self.codec = codec.freeze()
        self.modalities = ordered
        self.config = codec.config
        self.image_shape = codec.image_shape
        self.per_modality_latent_shape = codec.latent_shape
        self.latent_shape = (
            len(ordered) * codec.config.embedding_dim,
            *codec.latent_shape[1:],
        )
        statistic_shape = (1, self.latent_shape[0], 1, 1, 1)
        self.register_buffer("latent_mean", torch.zeros(statistic_shape))
        self.register_buffer("latent_std", torch.ones(statistic_shape))
        self.checkpoint_sha256 = codec.checkpoint_sha256
        self.checkpoint_identity = dict(codec.checkpoint_identity)
        self.set_latent_statistics(latent_mean, latent_std)
        self.freeze()

    @property
    def image_channels(self) -> int:
        return len(self.modalities)

    @property
    def latent_channels(self) -> int:
        return self.latent_shape[0]

    def train(self, mode: bool = True) -> "MeWMMultimodalVQGANCodec":
        super().train(False)
        return self

    def _validate_image(self, image: Tensor) -> None:
        expected = (self.image_channels, *self.image_shape)
        if image.ndim != 5 or tuple(image.shape[1:]) != expected:
            raise ValueError(
                f"expected multimodal MRI shape [B,{','.join(map(str, expected))}], "
                f"got {tuple(image.shape)}"
            )
        if not image.is_floating_point():
            raise TypeError("multimodal MRI tensors must use a floating-point dtype")

    def _validate_latent(self, latent: Tensor) -> None:
        if latent.ndim != 5 or tuple(latent.shape[1:]) != self.latent_shape:
            raise ValueError(
                "expected joint MU-Glioma latent shape "
                f"[B,{','.join(map(str, self.latent_shape))}], got {tuple(latent.shape)}"
            )
        if not latent.is_floating_point():
            raise TypeError("multimodal latent tensors must use a floating-point dtype")

    def _canonical_statistic(
        self, value: Tensor | Sequence[float] | float
    ) -> Tensor:
        statistic = torch.as_tensor(
            value, dtype=self.latent_mean.dtype, device=self.latent_mean.device
        )
        if statistic.numel() == 1:
            statistic = statistic.expand(self.latent_channels)
        if statistic.numel() != self.latent_channels:
            raise ValueError(
                "multimodal latent statistics must be scalar or contain one value "
                "per joint latent channel"
            )
        return statistic.reshape(1, self.latent_channels, 1, 1, 1)

    @torch.no_grad()
    def set_latent_statistics(
        self,
        mean: Tensor | Sequence[float] | float,
        std: Tensor | Sequence[float] | float,
    ) -> None:
        canonical_mean = self._canonical_statistic(mean)
        canonical_std = self._canonical_statistic(std)
        if not bool(torch.isfinite(canonical_mean).all()):
            raise ValueError("multimodal latent mean must be finite")
        if not bool(torch.isfinite(canonical_std).all()) or bool(
            torch.any(canonical_std <= 0)
        ):
            raise ValueError("multimodal latent std must be positive and finite")
        self.latent_mean.copy_(canonical_mean)
        self.latent_std.copy_(canonical_std)

    def normalize_latent(self, latent: Tensor) -> Tensor:
        self._validate_latent(latent)
        return (latent - self.latent_mean.to(latent)) / self.latent_std.to(latent)

    def denormalize_latent(self, latent: Tensor) -> Tensor:
        self._validate_latent(latent)
        return latent * self.latent_std.to(latent) + self.latent_mean.to(latent)

    def encode(self, image: Tensor, *, normalize: bool = False) -> Tensor:
        self._validate_image(image)
        encoded = torch.cat(
            tuple(
                self.codec.encode(image[:, index : index + 1], normalize=False)
                for index in range(self.image_channels)
            ),
            dim=1,
        )
        self._validate_latent(encoded)
        if not bool(torch.isfinite(encoded).all()):
            raise RuntimeError("multimodal MeWM encoder returned a non-finite latent")
        return self.normalize_latent(encoded) if normalize else encoded

    def decode(self, latent: Tensor, *, denormalize: bool = False) -> Tensor:
        self._validate_latent(latent)
        continuous = self.denormalize_latent(latent) if denormalize else latent
        width = self.config.embedding_dim
        decoded = torch.cat(
            tuple(
                self.codec.decode(
                    continuous[:, index * width : (index + 1) * width],
                    denormalize=False,
                )
                for index in range(self.image_channels)
            ),
            dim=1,
        )
        expected = (latent.shape[0], self.image_channels, *self.image_shape)
        if tuple(decoded.shape) != expected or not bool(torch.isfinite(decoded).all()):
            raise RuntimeError("multimodal MeWM decoder returned an invalid tensor")
        return decoded

    def encode_normalized(self, image: Tensor) -> Tensor:
        return self.encode(image, normalize=True)

    def decode_normalized(self, latent: Tensor) -> Tensor:
        return self.decode(latent, denormalize=True)

    def forward(self, image: Tensor) -> Tensor:
        return self.decode(self.encode(image))

    def freeze(self) -> "MeWMMultimodalVQGANCodec":
        self.requires_grad_(False)
        self.train(False)
        return self

    def unfreeze(self) -> "MeWMMultimodalVQGANCodec":
        raise RuntimeError("the imported multimodal MeWM VQ-GAN is frozen")


def _validated_checkpoint_identity(
    payload: Mapping[str, Any],
    *,
    expected_config: MeWMVQGANConfig,
    expected_data_contract_sha256: str,
    expected_data_backend: str,
    expected_numeric_contract: str,
) -> tuple[dict[str, Any], MeWMVQGANConfig]:
    identity = payload.get("mewm_ispy2_vqgan_identity")
    if type(identity) is not dict:
        raise MeWMVQGANCheckpointError(
            "VQ-GAN checkpoint has no exact MeWM identity dictionary"
        )
    if identity.get("schema_version") != MEWM_VQGAN_CHECKPOINT_SCHEMA:
        raise MeWMVQGANCheckpointError("VQ-GAN checkpoint identity schema is incompatible")
    if identity.get("mri_finetuned") is not True:
        raise MeWMVQGANCheckpointError(
            "VQ-GAN checkpoint is not marked as I-SPY2 MRI-finetuned"
        )
    if identity.get("numeric_contract") != expected_numeric_contract:
        raise MeWMVQGANCheckpointError("VQ-GAN numeric contract is incompatible")
    if identity.get("data_backend") != expected_data_backend:
        raise MeWMVQGANCheckpointError("VQ-GAN data backend is incompatible")
    if identity.get("data_contract_sha256") != expected_data_contract_sha256:
        raise MeWMVQGANCheckpointError("VQ-GAN data contract SHA-256 is incompatible")
    actual_config = MeWMVQGANConfig.from_architecture_contract(
        identity.get("architecture_contract")
    )
    if actual_config != expected_config:
        raise MeWMVQGANCheckpointError(
            "VQ-GAN checkpoint architecture differs from the pinned configuration"
        )
    return dict(identity), actual_config


def _strict_autoencoder_state(
    payload: Mapping[str, Any], model: MRILevelVQGAN
) -> dict[str, Tensor]:
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise MeWMVQGANCheckpointError("VQ-GAN checkpoint has no state mapping")
    if any(type(key) is not str for key in state):
        raise MeWMVQGANCheckpointError("VQ-GAN checkpoint state keys must be exact strings")
    prefix = "autoencoder."
    source = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    target = model.state_dict()
    missing = sorted(set(target).difference(source))
    unexpected = sorted(set(source).difference(target))
    invalid_values = sorted(
        key for key in set(source).intersection(target) if type(source[key]) is not Tensor
    )
    shape_mismatches = sorted(
        key
        for key in set(source).intersection(target)
        if isinstance(source[key], Tensor) and source[key].shape != target[key].shape
    )
    dtype_mismatches = sorted(
        key
        for key in set(source).intersection(target)
        if isinstance(source[key], Tensor) and source[key].dtype != target[key].dtype
    )
    nonfinite = sorted(
        key
        for key in set(source).intersection(target)
        if isinstance(source[key], Tensor)
        and source[key].is_floating_point()
        and not bool(torch.isfinite(source[key]).all())
    )
    if (
        missing
        or unexpected
        or invalid_values
        or shape_mismatches
        or dtype_mismatches
        or nonfinite
    ):
        raise MeWMVQGANCheckpointError(
            "VQ-GAN autoencoder state is incomplete or incompatible: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"invalid={len(invalid_values)}, shape_mismatches={len(shape_mismatches)}, "
            f"dtype_mismatches={len(dtype_mismatches)}, nonfinite={len(nonfinite)}"
        )
    return source


def load_mewm_vqgan_codec(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str = MEWM_REGISTERED_VQGAN_SHA256,
    expected_config: MeWMVQGANConfig = MEWM_REGISTERED_VQGAN_CONFIG,
    expected_data_contract_sha256: str = MEWM_REGISTERED_VQGAN_DATA_CONTRACT_SHA256,
    expected_data_backend: str = MEWM_REGISTERED_DATA_BACKEND,
    expected_numeric_contract: str = MEWM_REGISTERED_NUMERIC_CONTRACT,
    expected_codebook_sha256: str | None = MEWM_REGISTERED_CODEBOOK_SHA256,
    image_shape: Sequence[int] = MEWM_REGISTERED_IMAGE_SHAPE,
    latent_mean: Tensor | Sequence[float] | float | None = None,
    latent_std: Tensor | Sequence[float] | float | None = None,
) -> MeWMVQGANCodec:
    """Load the pinned generator namespace from a full MeWM Lightning checkpoint."""

    source = Path(checkpoint_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    expected_digest = _require_sha256(expected_sha256, label="expected checkpoint SHA-256")
    actual_digest = sha256_file(source)
    if actual_digest != expected_digest:
        raise MeWMVQGANCheckpointError(
            "MeWM VQ-GAN checkpoint SHA-256 mismatch: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    expected_data_digest = _require_sha256(
        expected_data_contract_sha256, label="expected data contract SHA-256"
    )
    try:
        payload = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError, EOFError) as error:
        raise MeWMVQGANCheckpointError("could not read MeWM VQ-GAN checkpoint") from error
    if type(payload) is not dict:
        raise MeWMVQGANCheckpointError(
            "MeWM VQ-GAN checkpoint payload must be an exact dictionary"
        )
    identity, config = _validated_checkpoint_identity(
        payload,
        expected_config=expected_config,
        expected_data_contract_sha256=expected_data_digest,
        expected_data_backend=expected_data_backend,
        expected_numeric_contract=expected_numeric_contract,
    )
    model = MRILevelVQGAN(config).float().eval()
    state = _strict_autoencoder_state(payload, model)
    model.load_state_dict(state, strict=True)
    if expected_codebook_sha256 is not None:
        expected_codebook = _require_sha256(
            expected_codebook_sha256, label="expected codebook SHA-256"
        )
        actual_codebook = mewm_codebook_sha256(model.quantizer.embeddings)
        if actual_codebook != expected_codebook:
            raise MeWMVQGANCheckpointError(
                "MeWM VQ-GAN codebook SHA-256 differs from the latent cache contract"
            )
    if (latent_mean is None) != (latent_std is None):
        raise ValueError("latent mean and std must be supplied together")
    codec = MeWMVQGANCodec(
        config,
        backend=model,
        image_shape=image_shape,
        checkpoint_sha256=actual_digest,
        checkpoint_identity=identity,
    )
    if latent_mean is not None and latent_std is not None:
        codec.set_latent_statistics(latent_mean, latent_std)
    return codec


__all__ = [
    "MEWM_MU_GLIOMA_MODALITIES",
    "MEWM_REGISTERED_CODEBOOK_SHA256",
    "MEWM_REGISTERED_DATA_BACKEND",
    "MEWM_REGISTERED_IMAGE_SHAPE",
    "MEWM_REGISTERED_LATENT_SHAPE",
    "MEWM_REGISTERED_NUMERIC_CONTRACT",
    "MEWM_REGISTERED_VQGAN_CONFIG",
    "MEWM_REGISTERED_VQGAN_DATA_CONTRACT_SHA256",
    "MEWM_REGISTERED_VQGAN_SHA256",
    "MEWM_VQGAN_ARCHITECTURE_SCHEMA",
    "MEWM_VQGAN_CHECKPOINT_SCHEMA",
    "MRILevelVQGAN",
    "MeWMVQGANCheckpointError",
    "MeWMVQGANCodec",
    "MeWMMultimodalVQGANCodec",
    "MeWMVQGANConfig",
    "VectorQuantizerEMA",
    "load_mewm_vqgan_codec",
    "mewm_codebook_sha256",
]
