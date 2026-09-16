from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


VQGAN_NUMERIC_CONTRACT = "dce0_neg_one_one_ct_training_aligned_v1"
REGISTERED_VQGAN_NUMERIC_CONTRACT = "motfm_registered_global_zscore_v1"
MU_GLIOMA_VQGAN_NUMERIC_CONTRACT = (
    "mu_glioma_post_brainiac_nonzero_zscore_pad_160x256x256_v1"
)
CLAMPED_LPIPS_NUMERIC_CONTRACTS = frozenset(
    {
        REGISTERED_VQGAN_NUMERIC_CONTRACT,
        MU_GLIOMA_VQGAN_NUMERIC_CONTRACT,
    }
)
VQGAN_TRAINING_CONTRACT = "mewm_ispy2_weak_gan_v1"
VQGAN_DEEP_TRAINING_CONTRACT = "mewm_ispy2_deep_b1_training_v1"
VQGAN_ARCHITECTURE_CONTRACT = "mewm_ispy2_vqgan_architecture_v1"
WIDTH_COMPATIBLE_MRI_INITIALIZATION = (
    "width_compatible_mri_codebook_discriminators_v1"
)


@dataclass(frozen=True)
class VQGANConfig:
    image_channels: int = 1
    hidden_channels: int = 16
    embedding_dim: int = 8
    n_codes: int = 16384
    downsample_factor: int = 4
    bottleneck_blocks: int = 0
    num_groups: int = 32
    commitment_weight: float = 0.25
    ema_decay: float = 0.99
    ema_smoothing: float = 1e-7
    restart_threshold: float = 1.0
    nearest_chunk_size: int = 4096

    def __post_init__(self) -> None:
        if self.image_channels != 1:
            raise ValueError("I-SPY2 VQGAN must be single-channel")
        if self.downsample_factor != 4:
            raise ValueError("I-SPY2 VQGAN downsampling must be exactly 4x")
        if self.bottleneck_blocks < 0:
            raise ValueError("VQGAN bottleneck block count cannot be negative")
        if (self.hidden_channels * 2) % self.num_groups:
            raise ValueError("VQGAN normalized channels must be divisible by group count")

    def latent_shape(self, spatial_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
        if any(size % self.downsample_factor for size in spatial_shape):
            raise ValueError("input shape must be divisible by the downsampling factor")
        return (
            self.embedding_dim,
            *(size // self.downsample_factor for size in spatial_shape),
        )


MRI_VQGAN_DEFAULTS = VQGANConfig()
REGISTERED_LARGE_VQGAN_CONFIG = VQGANConfig(
    image_channels=1,
    hidden_channels=32,
    embedding_dim=8,
    n_codes=16384,
    downsample_factor=4,
    bottleneck_blocks=1,
    num_groups=32,
)


def vqgan_architecture_contract(config: VQGANConfig) -> dict[str, Any]:
    return {
        "schema_version": VQGAN_ARCHITECTURE_CONTRACT,
        "config": asdict(config),
    }


def vqgan_config_from_checkpoint_identity(identity: dict[str, Any]) -> VQGANConfig:
    architecture = identity.get("architecture_contract")
    if architecture is None:
        return MRI_VQGAN_DEFAULTS
    if not isinstance(architecture, dict):
        raise ValueError("VQGAN checkpoint architecture contract is invalid")
    if architecture.get("schema_version") != VQGAN_ARCHITECTURE_CONTRACT:
        raise ValueError("VQGAN checkpoint architecture contract is incompatible")
    raw_config = architecture.get("config")
    expected_fields = {field.name for field in fields(VQGANConfig)}
    if not isinstance(raw_config, dict) or set(raw_config) != expected_fields:
        raise ValueError("VQGAN checkpoint architecture config is incomplete")
    return VQGANConfig(**raw_config)


def _triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    return value


def _same_pad(
    kernel_size: tuple[int, int, int], stride: tuple[int, int, int]
) -> tuple[int, ...]:
    padding: list[int] = []
    for total in (kernel - step for kernel, step in zip(kernel_size, stride, strict=True)):
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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
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

    def forward(self, value: torch.Tensor, output_size: list[int] | None = None) -> torch.Tensor:
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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, config: VQGANConfig) -> None:
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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.input(value)
        for downsample, residual in zip(self.downsamples, self.residuals, strict=True):
            hidden = residual(downsample(hidden))
        for block in self.bottleneck:
            hidden = block(hidden)
        return F.silu(self.final_norm(hidden))


class MRIDecoder(nn.Module):
    def __init__(self, config: VQGANConfig) -> None:
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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.final_norm(value))
        for block in self.bottleneck:
            hidden = block(hidden)
        for upsample, residual_one, residual_two in zip(
            self.upsamples, self.residual_one, self.residual_two, strict=True
        ):
            hidden = residual_two(residual_one(upsample(hidden)))
        return self.output(hidden)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, config: VQGANConfig) -> None:
        super().__init__()
        embeddings = torch.randn(config.n_codes, config.embedding_dim)
        embeddings = F.normalize(embeddings, dim=1)
        self.register_buffer("embeddings", embeddings)
        self.register_buffer("cluster_size", torch.ones(config.n_codes))
        self.register_buffer("embedding_sum", embeddings.clone())
        self.n_codes = config.n_codes
        self.embedding_dim = config.embedding_dim
        self.decay = config.ema_decay
        self.smoothing = config.ema_smoothing
        self.restart_threshold = config.restart_threshold
        self.commitment_weight = config.commitment_weight
        self.chunk_size = config.nearest_chunk_size

    def _nearest(self, flat: torch.Tensor) -> torch.Tensor:
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

    def _restart_candidates(self, flat: torch.Tensor) -> torch.Tensor:
        candidates = flat.detach().to(self.embeddings.dtype)
        if candidates.shape[0] < self.n_codes:
            repeats = math.ceil(self.n_codes / candidates.shape[0])
            candidates = candidates.repeat(repeats, 1)
            candidates = candidates + torch.randn_like(candidates) * (
                0.01 / math.sqrt(self.embedding_dim)
            )
        order = torch.randperm(candidates.shape[0], device=candidates.device)
        return candidates[order[: self.n_codes]]

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        quantized = quantized_flat.reshape(*latent.shape[0:1], *latent.shape[2:], -1)
        quantized = quantized.movedim(-1, 1)
        commitment = self.commitment_weight * F.mse_loss(latent, quantized.detach())
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
    def __init__(self, config: VQGANConfig = MRI_VQGAN_DEFAULTS) -> None:
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

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("VQGAN input must have shape [B,1,Z,Y,X]")
        return self.quantizer(self.pre_quant(self.encoder(image)))

    def encode_continuous(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("VQGAN input must have shape [B,1,Z,Y,X]")
        return self.pre_quant(self.encoder(image))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant(latent))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        latent, info = self.encode(image)
        return self.decode(latent), info


class PatchDiscriminator(nn.Module):
    def __init__(self, dimensions: int, channels: int, layers: int) -> None:
        super().__init__()
        convolution = nn.Conv2d if dimensions == 2 else nn.Conv3d
        normalization = nn.BatchNorm2d if dimensions == 2 else nn.BatchNorm3d
        blocks: list[nn.Module] = [
            nn.Sequential(
                convolution(1, channels, 4, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        ]
        output_channels = channels
        for _ in range(1, layers):
            input_channels = output_channels
            output_channels = min(output_channels * 2, 512)
            blocks.append(
                nn.Sequential(
                    convolution(
                        input_channels, output_channels, 4, stride=2, padding=2
                    ),
                    normalization(output_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        input_channels = output_channels
        output_channels = min(output_channels * 2, 512)
        blocks.append(
            nn.Sequential(
                convolution(input_channels, output_channels, 4, stride=1, padding=2),
                normalization(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )
        blocks.append(
            nn.Sequential(
                convolution(output_channels, 1, 4, stride=1, padding=2)
            )
        )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = []
        for block in self.blocks:
            value = block(value)
            features.append(value)
        return value, features


def _hinge_discriminator(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean())


SliceIndices = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _random_slice_indices(volume: torch.Tensor) -> SliceIndices:
    batch, _, depth, height, width = volume.shape
    return (
        torch.randint(depth, (batch,), device=volume.device),
        torch.randint(height, (batch,), device=volume.device),
        torch.randint(width, (batch,), device=volume.device),
    )


def _orthogonal_slices(
    volume: torch.Tensor, indices: SliceIndices
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, channels, depth, height, width = volume.shape
    depth_indices, height_indices, width_indices = indices
    depth_slice = torch.gather(
        volume,
        2,
        depth_indices.reshape(batch, 1, 1, 1, 1).expand(
            batch, channels, 1, height, width
        ),
    ).squeeze(2)
    height_slice = torch.gather(
        volume,
        3,
        height_indices.reshape(batch, 1, 1, 1, 1).expand(
            batch, channels, depth, 1, width
        ),
    ).squeeze(3)
    width_slice = torch.gather(
        volume,
        4,
        width_indices.reshape(batch, 1, 1, 1, 1).expand(
            batch, channels, depth, height, 1
        ),
    ).squeeze(4)
    return depth_slice, height_slice, width_slice


class VQGANTrainingSystem(pl.LightningModule):
    def __init__(
        self,
        autoencoder: MRILevelVQGAN,
        *,
        learning_rate: float = 1.875e-5,
        discriminator_learning_rate: float = 9.375e-6,
        discriminator_channels: int = 64,
        discriminator_layers: int = 3,
        discriminator_start: int = 10000,
        discriminator_ramp_steps: int = 10000,
        adversarial_always_on: bool = False,
        reconstruction_weight: float = 4.0,
        perceptual_weight: float = 4.0,
        image_gan_weight: float = 0.1,
        volume_gan_weight: float = 0.1,
        feature_matching_weight: float = 1.0,
        gradient_clip_val: float = 1.0,
        batch_size: int = 2,
        precision: str = "16-mixed",
        max_epochs: int = -1,
        max_steps: int = -1,
        early_stopping_patience: int = 8,
        perceptual_model: nn.Module | None = None,
        checkpoint_identity: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.image_discriminator = PatchDiscriminator(
            2, discriminator_channels, discriminator_layers
        )
        self.volume_discriminator = PatchDiscriminator(
            3, discriminator_channels, discriminator_layers
        )
        self.perceptual_model = perceptual_model
        self.learning_rate = learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.discriminator_start = discriminator_start
        self.discriminator_ramp_steps = discriminator_ramp_steps
        self.adversarial_always_on = adversarial_always_on
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.image_gan_weight = image_gan_weight
        self.volume_gan_weight = volume_gan_weight
        self.feature_matching_weight = feature_matching_weight
        self.gradient_clip_val = gradient_clip_val
        self.discriminator_channels = discriminator_channels
        self.discriminator_layers = discriminator_layers
        self.batch_size = batch_size
        self.precision = precision
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_identity = dict(checkpoint_identity or {})
        self.automatic_optimization = False

    @property
    def adversarial_full_step(self) -> int:
        if self.adversarial_always_on:
            return 0
        return self.discriminator_start + self.discriminator_ramp_steps

    @property
    def training_contract(self) -> dict[str, Any]:
        if self.adversarial_always_on:
            return {
                "schema_version": VQGAN_DEEP_TRAINING_CONTRACT,
                "generator_learning_rate": self.learning_rate,
                "discriminator_learning_rate": self.discriminator_learning_rate,
                "adam_betas": [0.5, 0.9],
                "batch_size": self.batch_size,
                "precision": self.precision,
                "gradient_clip_val": self.gradient_clip_val,
                "gradient_clip_algorithm": "norm",
                "max_epochs": self.max_epochs,
                "max_steps": self.max_steps,
                "adversarial_always_on": True,
                "discriminator_start": self.discriminator_start,
                "discriminator_ramp_steps": self.discriminator_ramp_steps,
                "reconstruction_weight": self.reconstruction_weight,
                "perceptual_weight": self.perceptual_weight,
                "image_gan_weight": self.image_gan_weight,
                "volume_gan_weight": self.volume_gan_weight,
                "feature_matching_weight": self.feature_matching_weight,
                "discriminator_channels": self.discriminator_channels,
                "discriminator_layers": self.discriminator_layers,
                "scheduler_milestones": [60000, 120000],
                "generator_scheduler_gamma": 0.5,
                "discriminator_scheduler_gamma": 0.1,
                "early_stopping_monitor": "val/composite",
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_minimum_global_step": self.adversarial_full_step,
                "reconstruction_loss": "volume_l1_mean",
                "perceptual_loss": "three_random_orthogonal_slices_lpips_sum",
                "discriminator_loss": "2d_and_3d_hinge",
                "feature_matching_loss": "all_intermediate_features_l1_sum",
            }
        return {
            "schema_version": VQGAN_TRAINING_CONTRACT,
            "discriminator_start": self.discriminator_start,
            "discriminator_ramp_steps": self.discriminator_ramp_steps,
            "image_gan_weight": self.image_gan_weight,
            "volume_gan_weight": self.volume_gan_weight,
            "feature_matching_weight": self.feature_matching_weight,
            "generator_learning_rate": self.learning_rate,
            "discriminator_learning_rate": self.discriminator_learning_rate,
        }

    def adversarial_factor(self, global_step: int) -> float:
        if self.adversarial_always_on:
            return 1.0
        if global_step <= self.discriminator_start:
            return 0.0
        if self.discriminator_ramp_steps <= 0:
            return 1.0
        progress = (global_step - self.discriminator_start) / float(
            self.discriminator_ramp_steps
        )
        return min(max(progress, 0.0), 1.0)

    def _perceptual_view(self, image_slice: torch.Tensor) -> torch.Tensor:
        view = image_slice.repeat(1, 3, 1, 1)
        if self.checkpoint_identity.get("numeric_contract") in (
            CLAMPED_LPIPS_NUMERIC_CONTRACTS
        ):
            view = view.clamp(-1.0, 1.0)
        return view

    def compute_losses(
        self, image: torch.Tensor, *, global_step: int
    ) -> dict[str, torch.Tensor]:
        reconstruction, quantizer = self.autoencoder(image)
        slice_indices = _random_slice_indices(image)
        real_slices = _orthogonal_slices(image, slice_indices)
        fake_slices = _orthogonal_slices(reconstruction, slice_indices)
        fake_2d, fake_features_2d = self.image_discriminator(fake_slices[0])
        real_2d, real_features_2d = self.image_discriminator(real_slices[0])
        fake_3d, fake_features_3d = self.volume_discriminator(reconstruction)
        real_3d, real_features_3d = self.volume_discriminator(image)
        reconstruction_loss = F.l1_loss(reconstruction, image)
        if self.perceptual_model is None:
            slice_lpips = reconstruction_loss.new_zeros(())
        else:
            axis_losses = [
                self.perceptual_model(
                    self._perceptual_view(fake_slice),
                    self._perceptual_view(real_slice),
                ).mean()
                for fake_slice, real_slice in zip(
                    fake_slices, real_slices, strict=True
                )
            ]
            slice_lpips = torch.stack(axis_losses).sum()
        generator_2d = -fake_2d.mean()
        generator_3d = -fake_3d.mean()
        feature_matching = sum(
            F.l1_loss(fake, real.detach())
            for fake, real in zip(
                fake_features_2d[:-1] + fake_features_3d[:-1],
                real_features_2d[:-1] + real_features_3d[:-1],
                strict=True,
            )
        )
        adversarial_factor = self.adversarial_factor(global_step)
        generator_total = (
            self.reconstruction_weight * reconstruction_loss
            + quantizer["commitment_loss"]
            + self.perceptual_weight * slice_lpips
            + adversarial_factor
            * (
                self.image_gan_weight * generator_2d
                + self.volume_gan_weight * generator_3d
                + self.feature_matching_weight * feature_matching
            )
        )
        discriminator_2d = _hinge_discriminator(real_2d.detach(), fake_2d.detach())
        discriminator_3d = _hinge_discriminator(real_3d.detach(), fake_3d.detach())
        discriminator_total = adversarial_factor * (
            self.image_gan_weight * discriminator_2d
            + self.volume_gan_weight * discriminator_3d
        )
        return {
            "reconstruction": reconstruction_loss,
            "commitment": quantizer["commitment_loss"],
            "slice_lpips": slice_lpips,
            "generator_2d": generator_2d,
            "generator_3d": generator_3d,
            "feature_matching": feature_matching,
            "discriminator_2d": discriminator_2d,
            "discriminator_3d": discriminator_3d,
            "generator_total": generator_total,
            "discriminator_total": discriminator_total,
        }

    def compute_discriminator_losses(
        self, image: torch.Tensor, *, global_step: int
    ) -> dict[str, torch.Tensor]:
        was_training = self.autoencoder.training
        self.autoencoder.eval()
        with torch.no_grad():
            reconstruction, _ = self.autoencoder(image)
        self.autoencoder.train(was_training)
        slice_indices = _random_slice_indices(image)
        real_slices = _orthogonal_slices(image, slice_indices)
        fake_slices = _orthogonal_slices(reconstruction, slice_indices)
        real_2d, _ = self.image_discriminator(real_slices[0])
        fake_2d, _ = self.image_discriminator(fake_slices[0])
        real_3d, _ = self.volume_discriminator(image)
        fake_3d, _ = self.volume_discriminator(reconstruction)
        discriminator_2d = _hinge_discriminator(real_2d, fake_2d)
        discriminator_3d = _hinge_discriminator(real_3d, fake_3d)
        factor = self.adversarial_factor(global_step)
        total = factor * (
            self.image_gan_weight * discriminator_2d
            + self.volume_gan_weight * discriminator_3d
        )
        return {
            "discriminator_2d": discriminator_2d,
            "discriminator_3d": discriminator_3d,
            "total": total,
        }

    def train(self, mode: bool = True) -> "VQGANTrainingSystem":
        super().train(mode)
        if self.perceptual_model is not None:
            self.perceptual_model.eval()
        return self

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        image = batch.get("image")
        if image is None:
            image = batch["model_inputs"]["source_dce0"]
        generator_optimizer, discriminator_optimizer = self.optimizers()
        generator_scheduler, discriminator_scheduler = self.lr_schedulers()
        optimization_step = int(self.global_step)
        losses = self.compute_losses(image, global_step=optimization_step)

        generator_optimizer.zero_grad()
        self.manual_backward(losses["generator_total"])
        self.clip_gradients(
            generator_optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )
        generator_optimizer.step()
        generator_scheduler.step()

        discriminator_optimizer.zero_grad()
        discriminator_loss = self.compute_discriminator_losses(
            image.detach(), global_step=optimization_step
        )["total"]
        self.manual_backward(discriminator_loss)
        self.clip_gradients(
            discriminator_optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )
        discriminator_optimizer.step()
        discriminator_scheduler.step()
        self.log_dict(
            {f"train/{key}": value.detach() for key, value in losses.items()},
            on_step=True,
            on_epoch=True,
        )
        return losses["generator_total"].detach()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        image = batch.get("image")
        if image is None:
            image = batch["model_inputs"]["source_dce0"]
        losses = self.compute_losses(image, global_step=self.global_step)
        composite = (
            self.reconstruction_weight * losses["reconstruction"]
            + self.perceptual_weight * losses["slice_lpips"]
            + losses["commitment"]
        )
        self.log("val/reconstruction", losses["reconstruction"], prog_bar=True, on_epoch=True)
        self.log("val/slice_lpips", losses["slice_lpips"], on_epoch=True)
        self.log("val/commitment", losses["commitment"], on_epoch=True)
        self.log("val/composite", composite, prog_bar=True, on_epoch=True)

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer],
        list[torch.optim.lr_scheduler.MultiStepLR],
    ]:
        generator = torch.optim.Adam(
            self.autoencoder.parameters(), lr=self.learning_rate, betas=(0.5, 0.9)
        )
        discriminator = torch.optim.Adam(
            list(self.image_discriminator.parameters())
            + list(self.volume_discriminator.parameters()),
            lr=self.discriminator_learning_rate,
            betas=(0.5, 0.9),
        )
        generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            generator, milestones=[60000, 120000], gamma=0.5
        )
        discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            discriminator, milestones=[60000, 120000], gamma=0.1
        )
        return [generator, discriminator], [
            generator_scheduler,
            discriminator_scheduler,
        ]

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["mewm_ispy2_vqgan_identity"] = {
            "schema_version": "mewm_ispy2_mri_vqgan_v1",
            "mri_finetuned": True,
            "discriminator_initialization": "random_mri",
            **self.checkpoint_identity,
            "numeric_contract": self.checkpoint_identity.get(
                "numeric_contract", VQGAN_NUMERIC_CONTRACT
            ),
            "architecture_contract": vqgan_architecture_contract(
                self.autoencoder.config
            ),
            "training_contract": self.training_contract,
        }


@dataclass(frozen=True)
class CTInitializationReport:
    checkpoint_path: str
    checkpoint_sha256: str
    loaded_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]


@dataclass(frozen=True)
class MRIInitializationReport:
    checkpoint_path: str
    checkpoint_sha256: str
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    source_epoch: int | None
    source_global_step: int | None


@dataclass(frozen=True)
class WidthCompatibleMRIInitializationReport:
    initialization_method: str
    checkpoint_path: str
    checkpoint_sha256: str
    source_epoch: int
    source_global_step: int
    source_numeric_contract: str
    source_data_contract_sha256: str
    source_data_backend: str
    source_architecture_contract: dict[str, Any]
    loaded_keys: tuple[str, ...]
    rejected_keys: tuple[tuple[str, str], ...]
    loaded_key_count: int
    rejected_key_count: int
    source_key_count: int
    report_digest: str

    def identity_payload(self) -> dict[str, Any]:
        return {
            "initialization_method": self.initialization_method,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_epoch": self.source_epoch,
            "source_global_step": self.source_global_step,
            "source_numeric_contract": self.source_numeric_contract,
            "source_data_contract_sha256": self.source_data_contract_sha256,
            "source_data_backend": self.source_data_backend,
            "source_architecture_contract": self.source_architecture_contract,
            "loaded_keys": list(self.loaded_keys),
            "rejected_keys": [
                {"key": key, "reason": reason}
                for key, reason in self.rejected_keys
            ],
            "counts": {
                "loaded": self.loaded_key_count,
                "rejected": self.rejected_key_count,
                "source": self.source_key_count,
            },
            "report_digest": self.report_digest,
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_report_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _width_compatible_required_keys(
    system: VQGANTrainingSystem,
) -> tuple[str, ...]:
    codebook_keys = {
        "autoencoder.quantizer.embeddings",
        "autoencoder.quantizer.cluster_size",
        "autoencoder.quantizer.embedding_sum",
    }
    return tuple(
        sorted(
            codebook_keys
            | {
                key
                for key in system.state_dict()
                if key.startswith(("image_discriminator.", "volume_discriminator."))
            }
        )
    )


def validate_width_compatible_initialization_report(
    system: VQGANTrainingSystem,
    payload: Any,
    *,
    expected_checkpoint_sha256: str,
    expected_report_digest: str,
) -> WidthCompatibleMRIInitializationReport:
    error = "VQGAN resume identity has invalid width-compatible provenance"
    expected_fields = {
        "initialization_method",
        "checkpoint_path",
        "checkpoint_sha256",
        "source_epoch",
        "source_global_step",
        "source_numeric_contract",
        "source_data_contract_sha256",
        "source_data_backend",
        "source_architecture_contract",
        "loaded_keys",
        "rejected_keys",
        "counts",
        "report_digest",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError(error)
    if (
        payload.get("initialization_method") != WIDTH_COMPATIBLE_MRI_INITIALIZATION
        or payload.get("checkpoint_sha256") != expected_checkpoint_sha256
        or payload.get("source_numeric_contract") != VQGAN_NUMERIC_CONTRACT
        or payload.get("source_data_backend") != "current"
        or not isinstance(payload.get("checkpoint_path"), str)
        or not Path(payload["checkpoint_path"]).is_absolute()
    ):
        raise ValueError(error)
    for field_name in ("source_epoch", "source_global_step"):
        value = payload.get(field_name)
        if type(value) is not int or value < 0:
            raise ValueError(error)
    source_data_contract = payload.get("source_data_contract_sha256")
    if (
        not isinstance(source_data_contract, str)
        or len(source_data_contract) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_data_contract
        )
    ):
        raise ValueError(error)
    architecture = payload.get("source_architecture_contract")
    try:
        source_config = vqgan_config_from_checkpoint_identity(
            {"architecture_contract": architecture}
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if architecture != vqgan_architecture_contract(source_config):
        raise ValueError(error)

    loaded_raw = payload.get("loaded_keys")
    rejected_raw = payload.get("rejected_keys")
    counts = payload.get("counts")
    if (
        not isinstance(loaded_raw, list)
        or not all(isinstance(key, str) for key in loaded_raw)
        or loaded_raw != sorted(set(loaded_raw))
        or not isinstance(rejected_raw, list)
        or not isinstance(counts, dict)
        or set(counts) != {"loaded", "rejected", "source"}
    ):
        raise ValueError(error)
    rejected: list[tuple[str, str]] = []
    for entry in rejected_raw:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"key", "reason"}
            or not isinstance(entry.get("key"), str)
            or entry.get("reason") != "outside_allowlist"
        ):
            raise ValueError(error)
        rejected.append((entry["key"], entry["reason"]))
    if rejected != sorted(set(rejected)):
        raise ValueError(error)
    expected_loaded = list(_width_compatible_required_keys(system))
    rejected_keys = {key for key, _ in rejected}
    if (
        loaded_raw != expected_loaded
        or rejected_keys.intersection(loaded_raw)
        or any(type(value) is not int or value < 0 for value in counts.values())
        or counts["loaded"] != len(loaded_raw)
        or counts["rejected"] != len(rejected)
        or counts["source"] != len(loaded_raw) + len(rejected)
    ):
        raise ValueError(error)
    report_body = {
        key: value for key, value in payload.items() if key != "report_digest"
    }
    report_digest = payload.get("report_digest")
    if (
        not isinstance(report_digest, str)
        or report_digest != _canonical_report_digest(report_body)
        or report_digest != expected_report_digest
    ):
        raise ValueError(error)
    return WidthCompatibleMRIInitializationReport(
        initialization_method=payload["initialization_method"],
        checkpoint_path=payload["checkpoint_path"],
        checkpoint_sha256=payload["checkpoint_sha256"],
        source_epoch=payload["source_epoch"],
        source_global_step=payload["source_global_step"],
        source_numeric_contract=payload["source_numeric_contract"],
        source_data_contract_sha256=payload["source_data_contract_sha256"],
        source_data_backend=payload["source_data_backend"],
        source_architecture_contract=architecture,
        loaded_keys=tuple(loaded_raw),
        rejected_keys=tuple(rejected),
        loaded_key_count=counts["loaded"],
        rejected_key_count=counts["rejected"],
        source_key_count=counts["source"],
        report_digest=report_digest,
    )


def _prepare_width_compatible_mri_codebook_discriminators(
    system: VQGANTrainingSystem,
    checkpoint_path: str | Path,
    *,
    expected_checkpoint_sha256: str,
) -> tuple[WidthCompatibleMRIInitializationReport, dict[str, torch.Tensor]]:
    path = Path(checkpoint_path).resolve()
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "width-compatible MRI initialization checkpoint SHA256 mismatch"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("width-compatible MRI initialization payload is invalid")
    identity = payload.get("mewm_ispy2_vqgan_identity")
    if not isinstance(identity, dict) or identity.get("mri_finetuned") is not True:
        raise ValueError(
            "width-compatible MRI initialization source is not MRI-finetuned"
        )
    if identity.get("schema_version") != "mewm_ispy2_mri_vqgan_v1":
        raise ValueError(
            "width-compatible MRI initialization source schema is incompatible"
        )
    if identity.get("numeric_contract") != VQGAN_NUMERIC_CONTRACT:
        raise ValueError(
            "width-compatible MRI initialization source numeric contract "
            "is incompatible"
        )
    source_data_contract = identity.get("data_contract_sha256")
    if (
        not isinstance(source_data_contract, str)
        or len(source_data_contract) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_data_contract
        )
    ):
        raise ValueError(
            "width-compatible MRI initialization source data contract is invalid"
        )
    if identity.get("data_backend") != "current":
        raise ValueError(
            "width-compatible MRI initialization source data backend is incompatible"
        )
    source_config = vqgan_config_from_checkpoint_identity(identity)
    source_architecture = vqgan_architecture_contract(source_config)
    source_epoch = payload.get("epoch")
    source_global_step = payload.get("global_step")
    if (
        type(source_epoch) is not int
        or source_epoch < 0
        or type(source_global_step) is not int
        or source_global_step < 0
    ):
        raise ValueError(
            "width-compatible MRI initialization source step provenance is invalid"
        )
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(
            "width-compatible MRI initialization checkpoint has no state dict"
        )

    source: dict[str, Any] = {}
    duplicates: set[str] = set()
    for source_key, value in state.items():
        if type(source_key) is not str:
            raise ValueError(
                "width-compatible MRI initialization state dict keys must be exact strings"
            )
        key = source_key
        if key in source:
            duplicates.add(key)
        else:
            source[key] = value

    target = system.state_dict()
    required_keys = set(_width_compatible_required_keys(system))
    transfer_namespaces = (
        "autoencoder.quantizer.",
        "image_discriminator.",
        "volume_discriminator.",
    )
    missing = required_keys.difference(source)
    unexpected = {
        key
        for key in source
        if key.startswith(transfer_namespaces) and key not in required_keys
    }
    shape_mismatches = {
        key
        for key in required_keys.intersection(source)
        if not isinstance(source[key], torch.Tensor)
        or source[key].shape != target[key].shape
    }
    if duplicates or missing or unexpected or shape_mismatches:
        raise ValueError(
            "width-compatible MRI initialization is incomplete or incompatible: "
            f"missing={len(missing)}, duplicated={len(duplicates)}, "
            f"unexpected={len(unexpected)}, shape_mismatches={len(shape_mismatches)}"
        )

    loaded_keys = tuple(sorted(required_keys))
    rejected_keys = tuple(
        (key, "outside_allowlist")
        for key in sorted(set(source).difference(required_keys))
    )
    report_body = {
        "initialization_method": WIDTH_COMPATIBLE_MRI_INITIALIZATION,
        "checkpoint_path": str(path),
        "checkpoint_sha256": actual_sha256,
        "source_epoch": source_epoch,
        "source_global_step": source_global_step,
        "source_numeric_contract": VQGAN_NUMERIC_CONTRACT,
        "source_data_contract_sha256": source_data_contract,
        "source_data_backend": "current",
        "source_architecture_contract": source_architecture,
        "loaded_keys": list(loaded_keys),
        "rejected_keys": [
            {"key": key, "reason": reason}
            for key, reason in rejected_keys
        ],
        "counts": {
            "loaded": len(loaded_keys),
            "rejected": len(rejected_keys),
            "source": len(source),
        },
    }
    report = WidthCompatibleMRIInitializationReport(
        initialization_method=WIDTH_COMPATIBLE_MRI_INITIALIZATION,
        checkpoint_path=str(path),
        checkpoint_sha256=actual_sha256,
        source_epoch=source_epoch,
        source_global_step=source_global_step,
        source_numeric_contract=VQGAN_NUMERIC_CONTRACT,
        source_data_contract_sha256=source_data_contract,
        source_data_backend="current",
        source_architecture_contract=source_architecture,
        loaded_keys=loaded_keys,
        rejected_keys=rejected_keys,
        loaded_key_count=len(loaded_keys),
        rejected_key_count=len(rejected_keys),
        source_key_count=len(source),
        report_digest=_canonical_report_digest(report_body),
    )
    mapped = {
        key: source[key].to(dtype=target[key].dtype)
        for key in loaded_keys
    }
    return report, mapped


def load_width_compatible_mri_codebook_discriminators(
    system: VQGANTrainingSystem,
    checkpoint_path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_report_digest: str | None = None,
) -> WidthCompatibleMRIInitializationReport:
    if (
        not isinstance(expected_report_digest, str)
        or len(expected_report_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_report_digest
        )
    ):
        raise ValueError(
            "width-compatible MRI initialization expected report digest is invalid"
        )
    report, mapped = _prepare_width_compatible_mri_codebook_discriminators(
        system,
        checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    if report.report_digest != expected_report_digest:
        raise ValueError(
            "width-compatible MRI initialization report digest mismatch"
        )
    system.load_state_dict(mapped, strict=False)
    return report


def load_mri_training_weights(
    system: VQGANTrainingSystem,
    checkpoint_path: str | Path,
) -> MRIInitializationReport:
    path = Path(checkpoint_path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = payload.get("mewm_ispy2_vqgan_identity", {})
    if identity.get("mri_finetuned") is not True:
        raise ValueError("MRI warm-start checkpoint is not marked as MRI-finetuned")
    if identity.get("numeric_contract") != VQGAN_NUMERIC_CONTRACT:
        raise ValueError("MRI warm-start checkpoint numeric contract is incompatible")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("MRI warm-start checkpoint has no state dict")

    prefixes = (
        "autoencoder.",
        "image_discriminator.",
        "volume_discriminator.",
    )
    target = system.state_dict()
    source_network = {
        str(key): value
        for key, value in state.items()
        if str(key).startswith(prefixes) and isinstance(value, torch.Tensor)
    }
    target_network_keys = {key for key in target if key.startswith(prefixes)}
    unexpected = set(source_network) - target_network_keys
    missing = target_network_keys - set(source_network)
    allowed_missing_prefixes = (
        "autoencoder.encoder.bottleneck.",
        "autoencoder.decoder.bottleneck.",
    )
    disallowed_missing = {
        key for key in missing if not key.startswith(allowed_missing_prefixes)
    }
    shape_mismatches = {
        key
        for key in set(source_network) & target_network_keys
        if source_network[key].shape != target[key].shape
    }
    if unexpected or disallowed_missing or shape_mismatches:
        raise ValueError(
            "MRI warm-start checkpoint is incomplete or incompatible: "
            f"missing={len(disallowed_missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatches={len(shape_mismatches)}"
        )

    mapped = {
        key: value.to(dtype=target[key].dtype)
        for key, value in source_network.items()
    }
    system.load_state_dict(mapped, strict=False)
    return MRIInitializationReport(
        checkpoint_path=str(path),
        checkpoint_sha256=_file_sha256(path),
        loaded_keys=tuple(sorted(mapped)),
        missing_keys=tuple(sorted(missing)),
        source_epoch=(int(payload["epoch"]) if "epoch" in payload else None),
        source_global_step=(
            int(payload["global_step"]) if "global_step" in payload else None
        ),
    )


def load_ct_autoencoder_weights(
    model: MRILevelVQGAN,
    checkpoint_path: str | Path,
    *,
    require_complete: bool = True,
) -> CTInitializationReport:
    path = Path(checkpoint_path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("CT autoencoder checkpoint has no state dict")
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    prefixes = ("module.", "model.", "autoencoder.", "vqgan.")

    def legacy_key(key: str) -> str:
        direct_prefixes = {
            "encoder.conv_first.conv.": "encoder.input.",
            "encoder.final_block.0.": "encoder.final_norm.",
            "decoder.final_block.0.": "decoder.final_norm.",
            "decoder.conv_last.conv.": "decoder.output.",
            "pre_vq_conv.conv.": "pre_quant.",
            "post_vq_conv.conv.": "post_quant.",
            "codebook.embeddings": "quantizer.embeddings",
            "codebook.N": "quantizer.cluster_size",
            "codebook.z_avg": "quantizer.embedding_sum",
        }
        for source, destination in direct_prefixes.items():
            if key == source or key.startswith(source):
                return destination + key[len(source) :]
        for index in range(2):
            replacements = {
                f"encoder.conv_blocks.{index}.down.conv.": f"encoder.downsamples.{index}.",
                f"encoder.conv_blocks.{index}.res.": f"encoder.residuals.{index}.",
                f"decoder.conv_blocks.{index}.up.convt.": f"decoder.upsamples.{index}.",
                f"decoder.conv_blocks.{index}.res1.": f"decoder.residual_one.{index}.",
                f"decoder.conv_blocks.{index}.res2.": f"decoder.residual_two.{index}.",
            }
            for source, destination in replacements.items():
                if key.startswith(source):
                    key = destination + key[len(source) :]
                    break
        return key.replace(".conv1.conv.", ".conv1.").replace(
            ".conv2.conv.", ".conv2."
        )

    for source_key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = str(source_key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        key = legacy_key(key)
        if key not in target:
            skipped.append(str(source_key))
            continue
        candidate = value
        if (
            key == "encoder.input.weight"
            and value.ndim == 5
            and value.shape[1] == 3
            and target[key].shape[1] == 1
            and value.shape[0] == target[key].shape[0]
            and value.shape[2:] == target[key].shape[2:]
        ):
            candidate = value.mean(dim=1, keepdim=True)
        if candidate.shape != target[key].shape:
            skipped.append(str(source_key))
            continue
        mapped[key] = candidate.to(dtype=target[key].dtype)
    if not mapped:
        raise ValueError("CT autoencoder checkpoint has no compatible MRI VQGAN weights")
    missing = set(target) - set(mapped)
    if require_complete and missing:
        raise ValueError(
            f"CT autoencoder checkpoint is not a complete initialization; missing {len(missing)} states"
        )
    model.load_state_dict(mapped, strict=False)
    return CTInitializationReport(
        checkpoint_path=str(path),
        checkpoint_sha256=_file_sha256(path),
        loaded_keys=tuple(sorted(mapped)),
        skipped_keys=tuple(sorted(skipped)),
    )
