from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.modules.module import _IncompatibleKeys

from .contracts import (
    ANCESTRAL_DDPM_SAMPLER,
    CONTINUOUS_CODEBOOK_MINMAX_CONTRACT,
    CT_DENOISER_ARCHITECTURE,
    DIFFUSION_CHECKPOINT_SCHEMA_VERSION,
    EPSILON_PREDICTION_TYPE,
    FILM_DENOISER_ARCHITECTURE,
)
from .ct_denoiser import CTStyleDenoiser3D, DenoiserResult
from .vqgan import MRILevelVQGAN


@dataclass(frozen=True)
class DiffusionConfig:
    denoiser_architecture: str = FILM_DENOISER_ARCHITECTURE
    timesteps: int = 200
    sampler: str = ANCESTRAL_DDPM_SAMPLER
    latent_contract: str = CONTINUOUS_CODEBOOK_MINMAX_CONTRACT
    ema_decay: float = 0.995
    noisy_channels: int = 8
    spatial_condition_channels: int = 9
    denoiser_input_channels: int = 17
    semantic_channels: int = 0
    prediction_type: str = EPSILON_PREDICTION_TYPE

    def __post_init__(self) -> None:
        if self.noisy_channels != 8 or self.spatial_condition_channels != 9:
            raise ValueError("latent and spatial condition channel contracts are fixed")
        expected_channels = {
            FILM_DENOISER_ARCHITECTURE: (17, 0),
            CT_DENOISER_ARCHITECTURE: (49, 32),
        }
        if expected_channels.get(self.denoiser_architecture) != (
            self.denoiser_input_channels,
            self.semantic_channels,
        ):
            raise ValueError("diffusion denoiser architecture and channel contract mismatch")
        if self.prediction_type != EPSILON_PREDICTION_TYPE:
            raise ValueError("diffusion prediction type must be epsilon")
        if self.timesteps <= 0:
            raise ValueError("diffusion timesteps must be positive")
        if self.sampler != ANCESTRAL_DDPM_SAMPLER:
            raise ValueError("diffusion sampler must be ancestral_ddpm")
        if self.latent_contract != CONTINUOUS_CODEBOOK_MINMAX_CONTRACT:
            raise ValueError(
                "diffusion latent contract must be continuous_codebook_minmax_v1"
            )
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("EMA decay must be between zero and one")


DIFFUSION_DEFAULTS = DiffusionConfig()


@dataclass(frozen=True)
class DiffusionSample:
    image: torch.Tensor
    normalized_latent: torch.Tensor
    continuous_latent: torch.Tensor
    code_indices: torch.Tensor
    unique_code_count: int
    effective_code_count: float


def _sinusoidal_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=1)


class FiLMResBlock3D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, input_channels)
        self.conv1 = nn.Conv3d(input_channels, output_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, output_channels)
        self.conv2 = nn.Conv3d(output_channels, output_channels, 3, padding=1)
        self.film = nn.Linear(512, output_channels * 2)
        self.time = nn.Linear(512, output_channels * 2)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, 1)
        )

    def forward(
        self, value: torch.Tensor, time_embedding: torch.Tensor, film_condition: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        scale, shift = (self.film(film_condition) + self.time(time_embedding)).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None, None])
        hidden = hidden + shift[:, :, None, None, None]
        hidden = self.conv2(F.silu(hidden))
        return self.skip(value) + hidden


class FiLMDenoiser3D(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int = 17,
        output_channels: int = 8,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        groups: int = 8,
    ) -> None:
        super().__init__()
        if input_channels != 17 or output_channels != 8:
            raise ValueError("FiLM denoiser channel contract is 17 -> 8")
        channels = [base_channels * multiplier for multiplier in channel_multipliers]
        if any(channel % groups for channel in channels):
            raise ValueError("denoiser channels must be divisible by group count")
        self.input_channels = input_channels
        self.architecture = FILM_DENOISER_ARCHITECTURE
        self.base_input_channels = input_channels
        self.semantic_condition_channels = 512
        self.spatial_semantic_channels = 0
        self.network_input_channels = input_channels
        self.input = nn.Conv3d(input_channels, channels[0], 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(256, 512), nn.SiLU(), nn.Linear(512, 512)
        )
        self.down_blocks = nn.ModuleList(
            [FiLMResBlock3D(channel, channel, groups=groups) for channel in channels[:-1]]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Conv3d(channels[i], channels[i + 1], 4, stride=2, padding=1)
                for i in range(len(channels) - 1)
            ]
        )
        self.middle_blocks = nn.ModuleList(
            [
                FiLMResBlock3D(channels[-1], channels[-1], groups=groups),
                FiLMResBlock3D(channels[-1], channels[-1], groups=groups),
            ]
        )
        self.upsamples = nn.ModuleList(
            [
                nn.ConvTranspose3d(
                    channels[i + 1], channels[i], 4, stride=2, padding=1
                )
                for i in reversed(range(len(channels) - 1))
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                FiLMResBlock3D(channels[i] * 2, channels[i], groups=groups)
                for i in reversed(range(len(channels) - 1))
            ]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(groups, channels[0]),
            nn.SiLU(),
            nn.Conv3d(channels[0], output_channels, 3, padding=1),
        )

    @property
    def conditional_blocks(self) -> tuple[FiLMResBlock3D, ...]:
        return tuple(self.down_blocks) + tuple(self.middle_blocks) + tuple(self.up_blocks)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        spatial_condition: torch.Tensor,
        semantic_condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> DenoiserResult:
        value = torch.cat((noisy_latent, spatial_condition), dim=1)
        if value.shape[1] != self.input_channels:
            raise ValueError("denoiser input must have exactly 17 channels")
        time_embedding = self.time_mlp(_sinusoidal_embedding(timesteps, 256))
        hidden = self.input(value)
        skips = []
        for block, downsample in zip(self.down_blocks, self.downsamples, strict=True):
            hidden = block(hidden, time_embedding, semantic_condition)
            skips.append(hidden)
            hidden = downsample(hidden)
        for block in self.middle_blocks:
            hidden = block(hidden, time_embedding, semantic_condition)
        for upsample, block, skip in zip(
            self.upsamples, self.up_blocks, reversed(skips), strict=True
        ):
            hidden = upsample(hidden)
            hidden = block(
                torch.cat((hidden, skip), dim=1),
                time_embedding,
                semantic_condition,
            )
        return DenoiserResult(
            predicted_noise=self.output(hidden),
            network_input=value,
            spatial_semantic=None,
        )


def build_denoiser(architecture: str, *, smoke: bool = False) -> nn.Module:
    if architecture == FILM_DENOISER_ARCHITECTURE:
        if smoke:
            return FiLMDenoiser3D(
                base_channels=4, channel_multipliers=(1, 2), groups=2
            )
        return FiLMDenoiser3D()
    if architecture == CT_DENOISER_ARCHITECTURE:
        return CTStyleDenoiser3D()
    raise ValueError(f"unsupported denoiser architecture: {architecture}")


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    values = torch.cos(((steps / timesteps) + offset) / (1 + offset) * math.pi / 2).square()
    values = values / values[0]
    return (1.0 - values[1:] / values[:-1]).clamp(0.0001, 0.9999).float()


class ConditionalLatentDiffusion(nn.Module):
    def __init__(
        self,
        vqgan: MRILevelVQGAN,
        conditioner: nn.Module,
        denoiser: nn.Module,
        config: DiffusionConfig = DIFFUSION_DEFAULTS,
    ) -> None:
        super().__init__()
        self.vqgan = vqgan.eval()
        for parameter in self.vqgan.parameters():
            parameter.requires_grad_(False)
        embeddings = self.vqgan.quantizer.embeddings
        latent_min = embeddings.amin().detach()
        latent_max = embeddings.amax().detach()
        if not bool(torch.isfinite(torch.stack((latent_min, latent_max))).all()):
            raise ValueError("VQGAN codebook bounds must be finite")
        if not bool(latent_max > latent_min):
            raise ValueError("VQGAN codebook bounds must have a positive range")
        self.conditioner = conditioner
        denoiser_contract = (
            getattr(denoiser, "architecture", None),
            getattr(denoiser, "network_input_channels", None),
            getattr(denoiser, "spatial_semantic_channels", None),
        )
        expected_denoiser_contract = (
            config.denoiser_architecture,
            config.denoiser_input_channels,
            config.semantic_channels,
        )
        if denoiser_contract != expected_denoiser_contract:
            raise ValueError("denoiser implementation does not match diffusion config")
        self.denoiser = denoiser
        self.config = config
        self.ema_denoiser = copy.deepcopy(denoiser).eval()
        for parameter in self.ema_denoiser.parameters():
            parameter.requires_grad_(False)
        betas = cosine_beta_schedule(config.timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
        )
        posterior_mean_coefficient_start = (
            betas * alpha_bars_previous.sqrt() / (1.0 - alpha_bars)
        )
        posterior_mean_coefficient_current = (
            (1.0 - alpha_bars_previous) * alphas.sqrt() / (1.0 - alpha_bars)
        )
        self.register_buffer("latent_min", latent_min.clone())
        self.register_buffer("latent_max", latent_max.clone())
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_previous", alpha_bars_previous)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance",
            posterior_variance.clamp_min(1e-20).log(),
        )
        self.register_buffer(
            "posterior_mean_coefficient_start", posterior_mean_coefficient_start
        )
        self.register_buffer(
            "posterior_mean_coefficient_current", posterior_mean_coefficient_current
        )

    def train(self, mode: bool = True) -> "ConditionalLatentDiffusion":
        super().train(mode)
        self.vqgan.eval()
        self.ema_denoiser.eval()
        return self

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return (
            2.0 * (latent - self.latent_min) / (self.latent_max - self.latent_min)
            - 1.0
        )

    def denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return (
            (latent + 1.0)
            * 0.5
            * (self.latent_max - self.latent_min)
            + self.latent_min
        )

    @torch.no_grad()
    def spatial_condition(
        self, source_dce0: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        masked_source = source_dce0 * (1.0 - source_mask)
        source_latent = self.normalize_latent(
            self.vqgan.encode_continuous(masked_source)
        )
        latent_mask = F.interpolate(
            source_mask.float(), size=source_latent.shape[-3:], mode="nearest"
        )
        condition = torch.cat((source_latent, latent_mask), dim=1)
        if condition.shape[1] != 9:
            raise RuntimeError("spatial condition must have exactly 9 channels")
        return condition

    @torch.no_grad()
    def target_latent(self, target_dce0: torch.Tensor) -> torch.Tensor:
        return self.normalize_latent(self.vqgan.encode_continuous(target_dce0))

    def q_sample(
        self, target_latent: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha = self.alpha_bars[timesteps].reshape(-1, 1, 1, 1, 1)
        return alpha.sqrt() * target_latent + (1.0 - alpha).sqrt() * noise

    @staticmethod
    def _extract(
        values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size
    ) -> torch.Tensor:
        return values[timesteps].reshape(
            timesteps.shape[0], *((1,) * (len(shape) - 1))
        )

    def training_loss(
        self,
        source_dce0: torch.Tensor,
        source_mask: torch.Tensor,
        target_dce0: torch.Tensor,
        action_text: Sequence[str],
        clinical_text: Sequence[str],
        delta_days: torch.Tensor,
        stage_id: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = self.target_latent(target_dce0)
        spatial = self.spatial_condition(source_dce0, source_mask)
        batch = target.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0, self.config.timesteps, (batch,), device=target.device
            )
        else:
            timesteps = timesteps.to(device=target.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(target)
        else:
            noise = noise.to(target.device)
        noisy = self.q_sample(target, timesteps, noise)
        film_condition = self.conditioner(
            action_text, clinical_text, delta_days, stage_id
        ).to(target.device)
        denoiser_result = self.denoiser(noisy, spatial, film_condition, timesteps)
        predicted_noise = denoiser_result.predicted_noise
        loss = F.l1_loss(predicted_noise, noise)
        details = {
            "target_latent": target,
            "spatial_condition": spatial,
            "film_condition": film_condition,
            "denoiser_input": denoiser_result.network_input,
            "predicted_noise": predicted_noise,
            "noise": noise,
            "timesteps": timesteps,
        }
        if denoiser_result.spatial_semantic is not None:
            details["spatial_semantic"] = denoiser_result.spatial_semantic
        return loss, details

    @torch.no_grad()
    def update_ema(self) -> None:
        decay = self.config.ema_decay
        for ema_parameter, parameter in zip(
            self.ema_denoiser.parameters(), self.denoiser.parameters(), strict=True
        ):
            ema_parameter.lerp_(parameter, 1.0 - decay)
        for ema_buffer, buffer in zip(
            self.ema_denoiser.buffers(), self.denoiser.buffers(), strict=True
        ):
            ema_buffer.copy_(buffer)

    @torch.no_grad()
    def p_sample_latent_step(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        spatial_condition: torch.Tensor,
        film_condition: torch.Tensor,
        *,
        posterior_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        predicted_noise = self.ema_denoiser(
            latent, spatial_condition, film_condition, timesteps
        ).predicted_noise
        alpha_bar = self._extract(self.alpha_bars, timesteps, latent.shape)
        predicted_start = (
            latent - (1.0 - alpha_bar).sqrt() * predicted_noise
        ) / alpha_bar.sqrt()
        predicted_start = predicted_start.clamp(-1.0, 1.0)
        posterior_mean = self._extract(
            self.posterior_mean_coefficient_start, timesteps, latent.shape
        ) * predicted_start + self._extract(
            self.posterior_mean_coefficient_current, timesteps, latent.shape
        ) * latent
        if posterior_noise is None:
            posterior_noise = torch.randn_like(latent)
        else:
            posterior_noise = posterior_noise.to(latent.device)
        nonzero = (timesteps != 0).to(latent.dtype).reshape(
            timesteps.shape[0], *((1,) * (latent.ndim - 1))
        )
        posterior_standard_deviation = (
            0.5
            * self._extract(self.posterior_log_variance, timesteps, latent.shape)
        ).exp()
        return (
            posterior_mean
            + nonzero * posterior_standard_deviation * posterior_noise
        )

    @torch.no_grad()
    def ancestral_sample_latent(
        self,
        source_dce0: torch.Tensor,
        source_mask: torch.Tensor,
        action_text: Sequence[str],
        clinical_text: Sequence[str],
        delta_days: torch.Tensor,
        stage_id: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        spatial = self.spatial_condition(source_dce0, source_mask)
        film = self.conditioner(action_text, clinical_text, delta_days, stage_id)
        film = film.to(spatial.device)
        latent = torch.randn(
            spatial.shape[0], 8, *spatial.shape[-3:], device=spatial.device
        ) if noise is None else noise.to(spatial.device)
        for timestep in reversed(range(self.config.timesteps)):
            times = torch.full(
                (latent.shape[0],), int(timestep), device=latent.device, dtype=torch.long
            )
            latent = self.p_sample_latent_step(
                latent,
                times,
                spatial,
                film,
            )
        return latent

    @torch.no_grad()
    def sample_with_diagnostics(self, *args: Any, **kwargs: Any) -> DiffusionSample:
        normalized = self.ancestral_sample_latent(*args, **kwargs)
        continuous = self.denormalize_latent(normalized)
        quantized, diagnostics = self.vqgan.quantizer(continuous)
        indices = diagnostics.get("indices")
        perplexity = diagnostics.get("perplexity")
        if not isinstance(indices, torch.Tensor) or not isinstance(
            perplexity, torch.Tensor
        ):
            raise RuntimeError("VQGAN quantizer did not return codebook diagnostics")
        return DiffusionSample(
            image=self.vqgan.decode(quantized),
            normalized_latent=normalized,
            continuous_latent=continuous,
            code_indices=indices,
            unique_code_count=int(torch.unique(indices).numel()),
            effective_code_count=float(perplexity),
        )

    @torch.no_grad()
    def sample(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.sample_with_diagnostics(*args, **kwargs).image


class DiffusionTrainingSystem(pl.LightningModule):
    def __init__(
        self,
        model: ConditionalLatentDiffusion,
        *,
        learning_rate: float = 1e-4,
        lora_learning_rate: float = 2e-5,
        optimizer_name: str = "adamw",
        checkpoint_identity: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.lora_learning_rate = lora_learning_rate
        if optimizer_name not in {"adam", "adamw"}:
            raise ValueError("diffusion optimizer must be adam or adamw")
        self.optimizer_name = optimizer_name
        self.checkpoint_identity = dict(checkpoint_identity or {})

    @staticmethod
    def _checkpoint_key(key: str) -> bool:
        fixed_prefixes = (
            "model.denoiser.",
            "model.ema_denoiser.",
            "model.conditioner.text_projection.",
            "model.conditioner.stage_embedding.",
            "model.conditioner.fusion.",
        )
        return key.startswith(fixed_prefixes) or (
            key.startswith("model.conditioner.text_tower.") and "lora_" in key
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        full = super().state_dict(*args, **kwargs)
        filtered = {key: value for key, value in full.items() if self._checkpoint_key(key)}
        if not any("lora_" in key for key in filtered):
            raise ValueError("diffusion checkpoint has no MedGemma LoRA adapter state")
        return filtered

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> _IncompatibleKeys:
        del strict
        expected = set(self.state_dict())
        actual = set(state_dict)
        if actual != expected:
            raise ValueError(
                "diffusion checkpoint required state mismatch: "
                f"missing={sorted(expected - actual)[:5]}, "
                f"unexpected={sorted(actual - expected)[:5]}"
            )
        super().load_state_dict(state_dict, strict=False, assign=assign)
        return _IncompatibleKeys([], [])

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["schema_version"] = DIFFUSION_CHECKPOINT_SCHEMA_VERSION
        checkpoint["identity"] = dict(self.checkpoint_identity)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("schema_version") != DIFFUSION_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("diffusion resume checkpoint schema mismatch")
        actual = checkpoint.get("identity")
        if not isinstance(actual, dict):
            raise ValueError("diffusion resume checkpoint identity is missing")
        for key, expected in self.checkpoint_identity.items():
            if actual.get(key) != expected:
                raise ValueError(f"diffusion resume identity mismatch: {key}")

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        denoiser = [parameter for parameter in self.model.denoiser.parameters() if parameter.requires_grad]
        lora = [
            parameter
            for parameter in self.model.conditioner.text_tower.parameters()
            if parameter.requires_grad
        ]
        conditioning_modules = (
            self.model.conditioner.text_projection,
            self.model.conditioner.stage_embedding,
            self.model.conditioner.fusion,
        )
        conditioning = [
            parameter
            for module in conditioning_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        groups = {"denoiser": denoiser, "lora": lora, "conditioning": conditioning}
        identifiers = [id(parameter) for values in groups.values() for parameter in values]
        if len(identifiers) != len(set(identifiers)):
            raise RuntimeError("diffusion optimizer parameter groups overlap")
        return groups

    def _batch_loss(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs = batch["model_inputs"]
        supervision = batch["supervision"]
        return self.model.training_loss(
            inputs["source_dce0"],
            inputs["source_mask"],
            supervision["target_dce0"],
            inputs["action_text"],
            inputs["clinical_text"],
            inputs["delta_days"],
            inputs["stage_id"],
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss, _ = self._batch_loss(batch)
        self.log("train/epsilon_l1", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        loss, _ = self._batch_loss(batch)
        self.log("val/epsilon_l1", loss, prog_bar=True, on_epoch=True, sync_dist=True)

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        super().optimizer_step(*args, **kwargs)
        self.model.update_ema()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        groups = self.trainable_parameter_groups()
        optimizer_groups = [
            {"params": groups["denoiser"], "lr": self.learning_rate},
            {"params": groups["conditioning"], "lr": self.learning_rate},
        ]
        if groups["lora"]:
            optimizer_groups.append(
                {"params": groups["lora"], "lr": self.lora_learning_rate}
            )
        if self.optimizer_name == "adam":
            return torch.optim.Adam(optimizer_groups, betas=(0.9, 0.999))
        return torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.99))
