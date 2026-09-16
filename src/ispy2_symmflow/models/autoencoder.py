"""Shared MONAI AutoencoderKL wrapper for all longitudinal visits."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AutoencoderKLConfig:
    """Arguments supported by MONAI 1.5.1 ``AutoencoderKL``."""

    in_channels: int = 3
    out_channels: int = 3
    channels: tuple[int, ...] = (64, 128, 256)
    num_res_blocks: int | tuple[int, ...] = 2
    attention_levels: tuple[bool, ...] = (False, False, True)
    latent_channels: int = 8
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    with_encoder_nonlocal_attn: bool = True
    with_decoder_nonlocal_attn: bool = True
    use_checkpoint: bool = False
    use_convtranspose: bool = False
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
            if len(blocks) != len(channels):
                raise ValueError("num_res_blocks must have one value per channel level")
            if any(value <= 0 for value in blocks):
                raise ValueError("num_res_blocks values must be positive")
        elif int(self.num_res_blocks) <= 0:
            raise ValueError("num_res_blocks must be positive")
        if len(channels) < 2 or any(value <= 0 for value in channels):
            raise ValueError("channels must contain at least two positive values")
        if len(attention) != len(channels):
            raise ValueError("attention_levels must have one value per channel level")
        if self.in_channels <= 0 or self.out_channels <= 0 or self.latent_channels <= 0:
            raise ValueError("input, output, and latent channel counts must be positive")
        if self.norm_num_groups <= 0:
            raise ValueError("norm_num_groups must be positive")
        incompatible = [
            channel for channel in channels if channel % self.norm_num_groups != 0
        ]
        if incompatible:
            raise ValueError(
                "all channels must be divisible by norm_num_groups; invalid values: "
                f"{incompatible}"
            )
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")

    @property
    def compression_factor(self) -> int:
        """Spatial compression implied by MONAI's N-1 downsampling blocks."""

        return 2 ** (len(self.channels) - 1)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AutoencoderKLConfig":
        values = dict(payload)
        spatial_dims = int(values.pop("spatial_dims", 3))
        if spatial_dims != 3:
            raise ValueError("this project requires spatial_dims=3")
        for training_key in (
            "batch_size",
            "gradient_clip_norm",
            "gradient_weight",
            "learning_rate",
            "weight_decay",
            "kl_weight",
            "max_epochs",
            "validation_interval_epochs",
            "warmup_steps",
        ):
            values.pop(training_key, None)
        valid = {field.name for field in fields(cls)}
        unknown = set(values).difference(valid)
        if unknown:
            raise ValueError(f"unsupported autoencoder config keys: {sorted(unknown)}")
        for name in ("channels", "attention_levels"):
            if name in values:
                values[name] = tuple(values[name])
        if "num_res_blocks" in values and isinstance(values["num_res_blocks"], list):
            values["num_res_blocks"] = tuple(values["num_res_blocks"])
        return cls(**values)


@dataclass(frozen=True)
class PosteriorStats:
    """Gaussian posterior parameters returned by MONAI AutoencoderKL."""

    mean: Tensor
    scale: Tensor


@dataclass(frozen=True)
class AutoencoderLoss:
    """Autoencoder objective components before any optional auxiliary losses."""

    total: Tensor
    reconstruction: Tensor
    kl: Tensor


def _load_monai_autoencoder_class() -> type[nn.Module]:
    try:
        from monai.networks.nets import AutoencoderKL
    except ImportError as error:
        raise ImportError(
            "SharedAutoencoderKL requires MONAI 1.5.1; install the project's train extra"
        ) from error
    return AutoencoderKL


def _build_monai_autoencoder(config: AutoencoderKLConfig) -> nn.Module:
    autoencoder_class = _load_monai_autoencoder_class()
    return autoencoder_class(
        spatial_dims=3,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_res_blocks=config.num_res_blocks,
        channels=config.channels,
        attention_levels=config.attention_levels,
        latent_channels=config.latent_channels,
        norm_num_groups=config.norm_num_groups,
        norm_eps=config.norm_eps,
        with_encoder_nonlocal_attn=config.with_encoder_nonlocal_attn,
        with_decoder_nonlocal_attn=config.with_decoder_nonlocal_attn,
        use_checkpoint=config.use_checkpoint,
        use_convtranspose=config.use_convtranspose,
        include_fc=config.include_fc,
        use_combined_linear=config.use_combined_linear,
        use_flash_attention=config.use_flash_attention,
    )


class SharedAutoencoderKL(nn.Module):
    """One AutoencoderKL shared across visits, with explicit latent scaling.

    ``encode`` always returns the posterior mean. This is the deterministic latent
    used by flow training, caching, and inference. Autoencoder training can request
    posterior samples through ``forward(sample_posterior=True)``.
    """

    def __init__(
        self,
        config: AutoencoderKLConfig,
        *,
        backend: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backend = backend if backend is not None else _build_monai_autoencoder(config)
        statistic_shape = (1, config.latent_channels, 1, 1, 1)
        self.register_buffer("latent_mean", torch.zeros(statistic_shape))
        self.register_buffer("latent_std", torch.ones(statistic_shape))

    def _validate_image(self, image: Tensor) -> None:
        if image.ndim != 5:
            raise ValueError("3D MRI tensors must have shape [B, C, D, H, W]")
        if image.shape[1] != self.config.in_channels:
            raise ValueError(
                f"expected {self.config.in_channels} MRI channels, got {image.shape[1]}"
            )
        if not image.is_floating_point():
            raise TypeError("MRI tensors must use a floating-point dtype")

    def _validate_latent(self, latent: Tensor) -> None:
        if latent.ndim != 5:
            raise ValueError("3D latent tensors must have shape [B, C, D, H, W]")
        if latent.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"expected {self.config.latent_channels} latent channels, got {latent.shape[1]}"
            )
        if not latent.is_floating_point():
            raise TypeError("latent tensors must use a floating-point dtype")

    def posterior(self, image: Tensor) -> PosteriorStats:
        """Return the unnormalized posterior mean and scale."""

        self._validate_image(image)
        encoded = self.backend.encode(image)
        if not isinstance(encoded, (tuple, list)) or len(encoded) != 2:
            raise RuntimeError("AutoencoderKL.encode must return (mean, scale)")
        mean, scale = encoded
        self._validate_latent(mean)
        if scale.shape != mean.shape:
            raise RuntimeError("posterior mean and scale must have identical shapes")
        if torch.any(scale < 0):
            raise RuntimeError("posterior scale cannot be negative")
        return PosteriorStats(mean=mean, scale=scale)

    def encode(
        self,
        image: Tensor,
        *,
        normalize: bool = False,
        return_posterior: bool = False,
    ) -> Tensor | tuple[Tensor, PosteriorStats]:
        """Encode one visit independently using its posterior mean."""

        stats = self.posterior(image)
        latent = self.normalize_latent(stats.mean) if normalize else stats.mean
        if return_posterior:
            return latent, stats
        return latent

    def sample_posterior(
        self,
        stats: PosteriorStats,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw a posterior sample explicitly; flow code should use ``encode``."""

        noise = torch.randn(
            stats.mean.shape,
            dtype=stats.mean.dtype,
            device=stats.mean.device,
            generator=generator,
        )
        return stats.mean + noise * stats.scale

    def decode(self, latent: Tensor, *, denormalize: bool = False) -> Tensor:
        """Decode a latent, optionally undoing training-set normalization first."""

        self._validate_latent(latent)
        if denormalize:
            latent = self.denormalize_latent(latent)
        reconstruction = self.backend.decode(latent)
        if reconstruction.ndim != 5 or reconstruction.shape[1] != self.config.out_channels:
            raise RuntimeError("AutoencoderKL.decode returned an incompatible tensor")
        return reconstruction

    def _canonical_statistic(self, value: Tensor | Sequence[float] | float) -> Tensor:
        statistic = torch.as_tensor(
            value, dtype=self.latent_mean.dtype, device=self.latent_mean.device
        )
        if statistic.numel() == 1:
            statistic = statistic.expand(self.config.latent_channels)
        if statistic.numel() != self.config.latent_channels:
            raise ValueError(
                "latent statistics must be scalar or contain one value per latent channel"
            )
        return statistic.reshape(1, self.config.latent_channels, 1, 1, 1)

    @torch.no_grad()
    def set_latent_statistics(
        self,
        mean: Tensor | Sequence[float] | float,
        std: Tensor | Sequence[float] | float,
    ) -> None:
        """Store statistics fitted once from training patients across all visits."""

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

    def forward(
        self, image: Tensor, *, sample_posterior: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(reconstruction, posterior_mean, posterior_scale)``."""

        stats = self.posterior(image)
        latent = self.sample_posterior(stats) if sample_posterior else stats.mean
        reconstruction = self.decode(latent)
        return reconstruction, stats.mean, stats.scale

    def reconstruction_loss(
        self,
        image: Tensor,
        *,
        kl_weight: float = 1e-6,
        sample_posterior: bool = True,
    ) -> AutoencoderLoss:
        """Compute an L1 reconstruction objective plus Gaussian posterior KL."""

        if kl_weight < 0 or not math.isfinite(kl_weight):
            raise ValueError("kl_weight must be finite and non-negative")
        reconstruction, mean, scale = self(
            image, sample_posterior=sample_posterior
        )
        reconstruction_term = F.l1_loss(reconstruction, image)
        variance = scale.square().clamp_min(torch.finfo(scale.dtype).tiny)
        kl_per_sample = 0.5 * (
            mean.square() + variance - variance.log() - 1.0
        ).flatten(start_dim=1).sum(dim=1)
        kl_term = kl_per_sample.mean()
        return AutoencoderLoss(
            total=reconstruction_term + kl_weight * kl_term,
            reconstruction=reconstruction_term,
            kl=kl_term,
        )

    def freeze(self) -> "SharedAutoencoderKL":
        self.requires_grad_(False)
        self.eval()
        return self

    def unfreeze(self) -> "SharedAutoencoderKL":
        self.requires_grad_(True)
        self.train()
        return self


def build_autoencoder_from_config(
    config: Mapping[str, Any] | AutoencoderKLConfig,
    *,
    backend: nn.Module | None = None,
) -> SharedAutoencoderKL:
    """Build the shared 3D autoencoder from a direct or nested mapping."""

    if isinstance(config, AutoencoderKLConfig):
        parsed = config
    else:
        section: Mapping[str, Any] = config
        if "model" in section and isinstance(section["model"], Mapping):
            section = section["model"]
        if "autoencoder" in section and isinstance(section["autoencoder"], Mapping):
            section = section["autoencoder"]
        parsed = AutoencoderKLConfig.from_mapping(section)
    return SharedAutoencoderKL(parsed, backend=backend)


__all__ = [
    "AutoencoderLoss",
    "AutoencoderKLConfig",
    "PosteriorStats",
    "SharedAutoencoderKL",
    "build_autoencoder_from_config",
]
