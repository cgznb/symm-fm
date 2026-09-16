from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import _IncompatibleKeys

from .contracts import (
    ANCESTRAL_DDPM_SAMPLER,
    CONTINUOUS_CODEBOOK_MINMAX_CONTRACT,
    CONTINUOUS_TRAIN_CHANNEL_ZSCORE_CONTRACT,
    EPSILON_PREDICTION_TYPE,
    PAPER_FAITHFUL_ARCHITECTURE,
    REGISTERED_LARGE_PAPER_ARCHITECTURE,
    X0_PREDICTION_TYPE,
)
from .ct_denoiser import DenoiserResult
from .diffusion import DiffusionSample, cosine_beta_schedule
from .latent_statistics import LatentChannelStatistics
from .paper_ccl import (
    PAPER_CCL_MODEL_INPUT_KEYS,
    _downsample_mask_with_occupancy_fallback,
    combo_contrastive_loss,
    latent_tumor_neighborhood,
    masked_latent_representation,
)
from .paper_conditioning import (
    PaperConditionOutput,
    parse_action_text,
    parse_clinical_text,
)
from .paper_contracts import (
    PaperRuntimeContract,
    RegisteredLargePaperRuntimeContract,
    default_paper_runtime_contract,
)
from .paper_checkpoint import (
    PAPER_CHECKPOINT_SCHEMA_VERSION,
    PaperCheckpointIdentity,
    _load_paper_state,
    _paper_identity_payload,
    _paper_state_dict,
    _validate_checkpoint_metadata,
    _validate_model_identity,
    _validate_paper_state,
)
from .vqgan import MRI_VQGAN_DEFAULTS
from .registered_large_checkpoint import (
    REGISTERED_LARGE_CHECKPOINT_SCHEMA_VERSION,
    RegisteredLargeCheckpointIdentity,
    _identity_payload as _registered_large_identity_payload,
    _validate_metadata as _validate_registered_large_metadata,
    _validate_model_identity as _validate_registered_large_model_identity,
)
from .registered_x0_checkpoint import (
    REGISTERED_X0_CHECKPOINT_SCHEMA_VERSION,
    RegisteredX0CheckpointIdentity,
    _identity_payload as _registered_x0_identity_payload,
    _validate_metadata as _validate_registered_x0_metadata,
    _validate_model_identity as _validate_registered_x0_model_identity,
)


PAPER_DENOISER_ARCHITECTURE = PAPER_FAITHFUL_ARCHITECTURE
DDIM_ETA0_SAMPLER = "ddim_eta0"
RESPACED_DDPM_SAMPLER = "respaced_ddpm"
_PAPER_CONTEXT_DIM = 512
_PAPER_CONTEXT_TOKENS = 7
_USED_RUNTIME_FIELD_TYPES = (
    ("architecture_id", str),
    ("epsilon_objective", str),
    ("context_dim", int),
    ("context_tokens", int),
    ("ccl_negative_count", int),
    ("ccl_timestep_min", int),
    ("ccl_timestep_max", int),
    ("ccl_temperature", float),
    ("ccl_weight", float),
)


_ValidationCheck = tuple[torch.Tensor, type[Exception], str]


def _raise_on_failed_checks(checks: Sequence[_ValidationCheck]) -> None:
    failures = torch.stack(tuple(failed.reshape(()) for failed, _, _ in checks))
    sentinel = failures.new_ones(1)
    first_failure = torch.cat((failures, sentinel)).to(torch.uint8).argmax().item()
    if first_failure < len(checks):
        _, error_type, message = checks[first_failure]
        raise error_type(message)


@dataclass(frozen=True)
class PaperDiffusionConfig:
    denoiser_architecture: str = PAPER_DENOISER_ARCHITECTURE
    timesteps: int = 200
    sampler: str = ANCESTRAL_DDPM_SAMPLER
    latent_contract: str = CONTINUOUS_CODEBOOK_MINMAX_CONTRACT
    ema_decay: float = 0.995
    noisy_channels: int = 8
    spatial_condition_channels: int = 9
    denoiser_input_channels: int = 49
    semantic_channels: int = 32
    prediction_type: str = EPSILON_PREDICTION_TYPE
    x0_objective: str | None = None
    latent_statistics_sha256: str | None = None
    runtime: PaperRuntimeContract | RegisteredLargePaperRuntimeContract = field(
        default_factory=default_paper_runtime_contract
    )

    def __post_init__(self) -> None:
        for name in (
            "denoiser_architecture",
            "sampler",
            "latent_contract",
            "prediction_type",
        ):
            if type(getattr(self, name)) is not str:
                raise TypeError(f"{name} must be an exact string")
        for name in (
            "timesteps",
            "noisy_channels",
            "spatial_condition_channels",
            "denoiser_input_channels",
            "semantic_channels",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if type(self.ema_decay) is not float:
            raise TypeError("ema_decay must be an exact float")
        expected_runtime_type = (
            RegisteredLargePaperRuntimeContract
            if self.denoiser_architecture == REGISTERED_LARGE_PAPER_ARCHITECTURE
            else PaperRuntimeContract
        )
        if type(self.runtime) is not expected_runtime_type:
            raise TypeError(
                f"runtime must be an exact {expected_runtime_type.__name__}"
            )
        for name, expected_type in _USED_RUNTIME_FIELD_TYPES:
            if type(getattr(self.runtime, name)) is not expected_type:
                raise TypeError(
                    f"{name} must be an exact {expected_type.__name__}"
                )

        if self.timesteps <= 0:
            raise ValueError("paper diffusion timesteps must be positive")
        if self.denoiser_architecture not in {
            PAPER_DENOISER_ARCHITECTURE,
            REGISTERED_LARGE_PAPER_ARCHITECTURE,
        }:
            raise ValueError("paper diffusion denoiser architecture is unsupported")
        if self.sampler != ANCESTRAL_DDPM_SAMPLER:
            raise ValueError("paper diffusion sampler must be ancestral_ddpm")
        if self.prediction_type == EPSILON_PREDICTION_TYPE:
            if self.latent_contract != CONTINUOUS_CODEBOOK_MINMAX_CONTRACT:
                raise ValueError("epsilon diffusion latent contract is fixed")
            if self.x0_objective is not None:
                raise ValueError("epsilon diffusion cannot define an x0 objective")
            if self.latent_statistics_sha256 is not None:
                raise ValueError("epsilon diffusion cannot bind latent statistics")
        elif self.prediction_type == X0_PREDICTION_TYPE:
            if self.denoiser_architecture != REGISTERED_LARGE_PAPER_ARCHITECTURE:
                raise ValueError("x0 prediction requires registered-large diffusion")
            if self.latent_contract != CONTINUOUS_TRAIN_CHANNEL_ZSCORE_CONTRACT:
                raise ValueError("x0 diffusion requires train-channel z-score latents")
            if self.x0_objective != "l2":
                raise ValueError("x0 diffusion objective must be l2")
            if (
                type(self.latent_statistics_sha256) is not str
                or len(self.latent_statistics_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.latent_statistics_sha256
                )
            ):
                raise ValueError("x0 diffusion latent statistics SHA256 is invalid")
        else:
            raise ValueError("paper diffusion prediction type must be epsilon or x0")
        if (self.noisy_channels, self.spatial_condition_channels) != (8, 9):
            raise ValueError("paper latent and spatial channel contracts are fixed")
        if (self.denoiser_input_channels, self.semantic_channels) != (49, 32):
            raise ValueError("paper denoiser channel contract is fixed")
        if not math.isfinite(self.ema_decay) or not 0.0 < self.ema_decay < 1.0:
            raise ValueError("paper EMA decay must be finite and between zero and one")
        if self.runtime.architecture_id != self.denoiser_architecture:
            raise ValueError("paper runtime architecture does not match diffusion config")
        if self.runtime.epsilon_objective not in {"l1", "l2"}:
            raise ValueError("paper epsilon objective must be l1 or l2")
        if self.runtime.context_dim != _PAPER_CONTEXT_DIM:
            raise ValueError("paper runtime context_dim must equal 512")
        if self.runtime.context_tokens != _PAPER_CONTEXT_TOKENS:
            raise ValueError("paper runtime context_tokens must equal 7")
        if self.runtime.ccl_negative_count != 2:
            raise ValueError("paper CCL requires exactly two negatives")
        if (
            not math.isfinite(self.runtime.ccl_temperature)
            or self.runtime.ccl_temperature <= 0.0
        ):
            raise ValueError("paper CCL temperature must be finite and positive")
        if (
            not math.isfinite(self.runtime.ccl_weight)
            or self.runtime.ccl_weight != 0.1
        ):
            raise ValueError("paper CCL weight must equal 0.1")
        if not (
            0
            <= self.runtime.ccl_timestep_min
            <= self.runtime.ccl_timestep_max
            < self.timesteps
        ):
            raise ValueError("paper CCL timestep range is outside the diffusion schedule")


PAPER_DIFFUSION_DEFAULTS = PaperDiffusionConfig()


@dataclass(frozen=True)
class PaperLossOutput:
    total_loss: torch.Tensor
    epsilon_loss: torch.Tensor
    epsilon_mse: torch.Tensor
    ccl_loss: torch.Tensor
    positive_similarity: torch.Tensor
    negative_similarity: torch.Tensor
    valid_ccl_fraction: torch.Tensor
    attenuation_level: torch.Tensor


@dataclass(frozen=True)
class X0LossOutput:
    total_loss: torch.Tensor
    x0_mse: torch.Tensor
    x0_mae: torch.Tensor
    attenuation_level: torch.Tensor


@dataclass(frozen=True)
class _TransitionMetadata:
    patient_id: str
    fold: str
    transition_type: str
    transition_id: str


@dataclass(frozen=True)
class _ValidatedTransition:
    value: dict[str, Any]
    actions: tuple[str, ...]
    clinical_subtypes: tuple[tuple[int, int, int], ...]
    stage_ids: torch.Tensor
    metadata: tuple[_TransitionMetadata, ...]
    image_shape: torch.Size
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class _ValidatedPaperBatch:
    anchor: dict[str, Any]
    positive: dict[str, Any] | None
    negative_action_texts: tuple[tuple[str, ...], tuple[str, ...]]
    valid_mask: torch.Tensor
    valid_indices: torch.Tensor
    valid_positions: tuple[int, ...]
    batch_size: int
    valid_count: int


class PaperFaithfulLatentDiffusion(nn.Module):
    def __init__(
        self,
        vqgan: nn.Module,
        conditioner: nn.Module,
        denoiser: nn.Module,
        attenuator: nn.Module,
        config: PaperDiffusionConfig = PAPER_DIFFUSION_DEFAULTS,
        latent_statistics: LatentChannelStatistics | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, PaperDiffusionConfig):
            raise TypeError("config must be a PaperDiffusionConfig")
        if (
            getattr(denoiser, "architecture", None),
            getattr(denoiser, "network_input_channels", None),
            getattr(denoiser, "spatial_semantic_channels", None),
        ) != (
            config.denoiser_architecture,
            config.denoiser_input_channels,
            config.semantic_channels,
        ):
            raise ValueError("denoiser implementation does not match paper config")

        self.vqgan = vqgan.eval()
        self.vqgan.requires_grad_(False)
        if config.prediction_type == EPSILON_PREDICTION_TYPE:
            if latent_statistics is not None:
                raise ValueError("epsilon diffusion cannot use channel statistics")
            embeddings = self.vqgan.quantizer.embeddings
            latent_min = embeddings.amin().detach()
            latent_max = embeddings.amax().detach()
            bounds = torch.stack((latent_min, latent_max))
            if not bool(torch.isfinite(bounds).all()):
                raise ValueError("VQGAN codebook bounds must be finite")
            if not bool(latent_max > latent_min):
                raise ValueError("VQGAN codebook bounds must have a positive range")
        else:
            if type(latent_statistics) is not LatentChannelStatistics:
                raise TypeError("x0 diffusion requires exact latent channel statistics")
            mean, std = latent_statistics.tensors(device="cpu", dtype=torch.float32)

        self.conditioner = conditioner
        self.denoiser = denoiser
        self.attenuator = attenuator
        self.config = config
        self.ema_denoiser = copy.deepcopy(denoiser).eval()
        self.ema_denoiser.requires_grad_(False)

        betas = cosine_beta_schedule(config.timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
        )
        if config.prediction_type == EPSILON_PREDICTION_TYPE:
            self.register_buffer("latent_min", latent_min.clone())
            self.register_buffer("latent_max", latent_max.clone())
        else:
            self.register_buffer("latent_channel_mean", mean)
            self.register_buffer("latent_channel_std", std)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_previous", alpha_bars_previous)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance", posterior_variance.clamp_min(1e-20).log()
        )
        self.register_buffer(
            "posterior_mean_coefficient_start",
            betas * alpha_bars_previous.sqrt() / (1.0 - alpha_bars),
        )
        self.register_buffer(
            "posterior_mean_coefficient_current",
            (1.0 - alpha_bars_previous) * alphas.sqrt() / (1.0 - alpha_bars),
        )

    def train(self, mode: bool = True) -> PaperFaithfulLatentDiffusion:
        super().train(mode)
        self.vqgan.eval()
        self.ema_denoiser.eval()
        return self

    @property
    def latent_device(self) -> torch.device:
        if self.config.prediction_type == X0_PREDICTION_TYPE:
            return self.latent_channel_mean.device
        return self.latent_min.device

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.config.prediction_type == X0_PREDICTION_TYPE:
            return (latent - self.latent_channel_mean) / self.latent_channel_std
        return (
            2.0 * (latent - self.latent_min) / (self.latent_max - self.latent_min)
            - 1.0
        )

    def denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.config.prediction_type == X0_PREDICTION_TYPE:
            return latent * self.latent_channel_std + self.latent_channel_mean
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
        source_latent = self.normalize_latent(
            self.vqgan.encode_continuous(source_dce0)
        )
        latent_mask = _downsample_mask_with_occupancy_fallback(
            source_mask.to(dtype=source_latent.dtype),
            source_latent.shape[-3:],
        )
        condition = torch.cat((source_latent, latent_mask), dim=1)
        if condition.shape[1] != self.config.spatial_condition_channels:
            raise RuntimeError("spatial condition must have exactly 9 channels")
        if not bool(torch.isfinite(condition).all()):
            raise ValueError("spatial condition must be finite")
        return condition

    @torch.no_grad()
    def target_latent(self, target_dce0: torch.Tensor) -> torch.Tensor:
        latent = self.normalize_latent(self.vqgan.encode_continuous(target_dce0))
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("target latent must be finite")
        return latent

    def q_sample(
        self, target_latent: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha = self._extract(self.alpha_bars, timesteps, target_latent.shape)
        return alpha.sqrt() * target_latent + (1.0 - alpha).sqrt() * noise

    def predicted_x0(
        self,
        noisy_latent: torch.Tensor,
        predicted_epsilon: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._extract(self.alpha_bars, timesteps, noisy_latent.shape)
        return (
            noisy_latent - (1.0 - alpha).sqrt() * predicted_epsilon
        ) / alpha.sqrt()

    def _model_output_to_x0(
        self,
        noisy_latent: torch.Tensor,
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.prediction_type == X0_PREDICTION_TYPE:
            return model_output
        return self.predicted_x0(noisy_latent, model_output, timesteps).clamp(
            -1.0, 1.0
        )

    def _epsilon_from_x0(
        self,
        noisy_latent: torch.Tensor,
        predicted_start: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._extract(self.alpha_bars, timesteps, noisy_latent.shape)
        denominator = (1.0 - alpha).sqrt().clamp_min(
            torch.finfo(noisy_latent.dtype).tiny
        )
        epsilon = (noisy_latent - alpha.sqrt() * predicted_start) / denominator
        self._finite(epsilon, "epsilon recovered from x0")
        return epsilon

    @staticmethod
    def _extract(
        values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size
    ) -> torch.Tensor:
        return values[timesteps].reshape(
            timesteps.shape[0], *((1,) * (len(shape) - 1))
        )

    @staticmethod
    def _finite(value: torch.Tensor, name: str) -> None:
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} must be finite")

    @staticmethod
    def _condition_to(
        condition: PaperConditionOutput, device: torch.device
    ) -> PaperConditionOutput:
        if not isinstance(condition, PaperConditionOutput):
            raise TypeError("conditioner must return PaperConditionOutput")
        moved = PaperConditionOutput(
            global_condition=condition.global_condition.to(device=device),
            context_tokens=condition.context_tokens.to(device=device),
            context_mask=condition.context_mask.to(device=device),
        )
        if moved.context_mask.dtype != torch.bool:
            raise TypeError("paper context mask must have boolean dtype")
        if not bool(torch.isfinite(moved.global_condition).all()):
            raise FloatingPointError("global condition must be finite")
        if not bool(torch.isfinite(moved.context_tokens).all()):
            raise FloatingPointError("context tokens must be finite")
        return moved

    def _condition(
        self,
        inputs: dict[str, Any],
        *,
        action_text: Sequence[str] | None = None,
        device: torch.device | None = None,
    ) -> PaperConditionOutput:
        actions = inputs["action_text"] if action_text is None else action_text
        condition = self.conditioner(
            actions,
            inputs["clinical_text"],
            inputs["delta_days"],
            inputs["stage_id"],
        )
        condition_device = (
            inputs["source_dce0"].device if device is None else device
        )
        return self._condition_to(condition, condition_device)

    @staticmethod
    def _generator(device: torch.device, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        if type(seed) is not int:
            raise TypeError("random_seed must be an exact integer")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    @staticmethod
    def _noise(
        supplied: torch.Tensor | None,
        reference: torch.Tensor,
        generator: torch.Generator | None,
        *,
        name: str,
    ) -> torch.Tensor:
        if supplied is None:
            return torch.randn(
                reference.shape,
                dtype=reference.dtype,
                device=reference.device,
                generator=generator,
            )
        if not isinstance(supplied, torch.Tensor) or supplied.shape != reference.shape:
            raise ValueError(f"{name} must match the target latent shape")
        noise = supplied.to(device=reference.device, dtype=reference.dtype)
        if not bool(torch.isfinite(noise).all()):
            raise ValueError(f"{name} must be finite")
        return noise

    def _timesteps(
        self,
        supplied: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        minimum: int,
        maximum: int,
        generator: torch.Generator | None,
        name: str,
    ) -> torch.Tensor:
        if supplied is None:
            return torch.randint(
                minimum,
                maximum + 1,
                (batch_size,),
                device=device,
                generator=generator,
            )
        if not isinstance(supplied, torch.Tensor) or supplied.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [B]")
        if supplied.dtype == torch.bool or supplied.is_floating_point() or supplied.is_complex():
            raise TypeError(f"{name} must use an integer dtype")
        timesteps = supplied.to(device=device, dtype=torch.long)
        if bool(torch.any((timesteps < minimum) | (timesteps > maximum))):
            raise ValueError(f"{name} is outside its configured range")
        return timesteps

    def _predict_epsilon(
        self,
        noisy: torch.Tensor,
        spatial: torch.Tensor,
        condition: PaperConditionOutput,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.prediction_type != EPSILON_PREDICTION_TYPE:
            raise RuntimeError("epsilon prediction helper requires epsilon parameterization")
        return self._predict_model_output(noisy, spatial, condition, timesteps)

    def _predict_model_output(
        self,
        noisy: torch.Tensor,
        spatial: torch.Tensor,
        condition: PaperConditionOutput,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self._validated_prediction(
            self.denoiser(noisy, spatial, condition, timesteps),
            reference=noisy,
        )

    def _validated_prediction(
        self, result: Any, *, reference: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(result, DenoiserResult):
            raise TypeError("denoiser must return DenoiserResult")
        predicted = result.prediction
        parameterization = self.config.prediction_type
        if not isinstance(predicted, torch.Tensor):
            raise TypeError(f"predicted {parameterization} must be a tensor")
        if predicted.shape != reference.shape:
            raise ValueError(
                f"predicted {parameterization} must match the noisy latent shape"
            )
        if predicted.device != reference.device:
            raise ValueError(
                f"predicted {parameterization} must use the noisy latent device"
            )
        expected_dtypes = {reference.dtype}
        if torch.is_autocast_enabled(reference.device.type):
            expected_dtypes.add(torch.get_autocast_dtype(reference.device.type))
        if predicted.dtype not in expected_dtypes:
            raise TypeError(f"predicted {parameterization} has the wrong runtime dtype")
        self._finite(predicted, f"predicted {parameterization}")
        return predicted

    @staticmethod
    def _validate_exact_noise(
        value: Any,
        *,
        expected_shape: tuple[int, ...] | torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have the exact latent shape")
        if value.device != device:
            raise ValueError(f"{name} must use the latent device")
        if value.dtype != dtype:
            raise TypeError(f"{name} must use the latent dtype")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} must be finite")
        return value

    def _validate_reverse_inputs(
        self,
        latent: Any,
        timesteps: Any,
        spatial_condition: Any,
        condition: Any,
        posterior_noise: Any,
    ) -> torch.Tensor:
        self._validate_reverse_state(
            latent,
            timesteps,
            spatial_condition,
            condition,
        )
        if posterior_noise is None:
            return torch.randn_like(latent)
        return self._validate_exact_noise(
            posterior_noise,
            expected_shape=latent.shape,
            device=latent.device,
            dtype=latent.dtype,
            name="posterior noise",
        )

    def _validate_reverse_state(
        self,
        latent: Any,
        timesteps: Any,
        spatial_condition: Any,
        condition: Any,
    ) -> None:
        if not isinstance(latent, torch.Tensor):
            raise TypeError("latent must be a tensor")
        if latent.ndim != 5 or latent.shape[1] != self.config.noisy_channels:
            raise ValueError("latent must have shape [B,8,D,H,W]")
        if latent.shape[0] <= 0 or any(size <= 0 for size in latent.shape[2:]):
            raise ValueError("latent dimensions must be positive")
        if latent.shape[-2] % 8 or latent.shape[-1] % 8:
            raise ValueError("latent height and width must be divisible by 8")
        if not latent.is_floating_point():
            raise TypeError("latent must use a floating dtype")
        if latent.device != self.latent_device:
            raise ValueError("latent is on the wrong model device")
        self._finite(latent, "latent")

        expected_spatial_shape = (
            latent.shape[0],
            self.config.spatial_condition_channels,
            *latent.shape[-3:],
        )
        if not isinstance(spatial_condition, torch.Tensor):
            raise TypeError("spatial condition must be a tensor")
        if spatial_condition.shape != expected_spatial_shape:
            raise ValueError("spatial condition must have shape [B,9,D,H,W]")
        if not spatial_condition.is_floating_point():
            raise TypeError("spatial condition must use a floating dtype")
        if spatial_condition.device != latent.device:
            raise ValueError("spatial condition must use the latent device")
        if spatial_condition.dtype != latent.dtype:
            raise TypeError("spatial condition must use the latent dtype")
        self._finite(spatial_condition, "spatial condition")

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
        batch_size = latent.shape[0]
        if condition.global_condition.shape != (
            batch_size,
            self.config.runtime.context_dim,
        ):
            raise ValueError("global condition must have shape [B,512]")
        expected_context_shape = (
            batch_size,
            self.config.runtime.context_tokens,
            self.config.runtime.context_dim,
        )
        if condition.context_tokens.shape != expected_context_shape:
            raise ValueError("context tokens must have shape [B,7,512]")
        if condition.context_mask.shape != expected_context_shape[:2]:
            raise ValueError("context mask must have shape [B,7]")
        if not condition.global_condition.is_floating_point():
            raise TypeError("global condition must use a floating dtype")
        if not condition.context_tokens.is_floating_point():
            raise TypeError("context tokens must use a floating dtype")
        if condition.context_mask.dtype != torch.bool:
            raise TypeError("context mask must have boolean dtype")
        if any(value.device != latent.device for _name, value in condition_tensors):
            raise ValueError("condition tensors must use the latent device")
        if not bool(condition.context_mask.any(dim=1).all()):
            raise ValueError("every context mask row must contain a token")
        self._finite(condition.global_condition, "global condition")
        self._finite(condition.context_tokens, "context tokens")

        if not isinstance(timesteps, torch.Tensor):
            raise TypeError("timesteps must be a tensor")
        if timesteps.shape != (batch_size,):
            raise ValueError("timesteps must have shape [B]")
        if (
            timesteps.dtype == torch.bool
            or timesteps.is_floating_point()
            or timesteps.is_complex()
        ):
            raise TypeError("timesteps must use an integer dtype")
        if timesteps.device != latent.device:
            raise ValueError("timesteps must use the latent device")
        if bool(torch.any((timesteps < 0) | (timesteps >= self.config.timesteps))):
            raise ValueError("timesteps are outside the diffusion schedule")

    def _prepared_spatial(
        self,
        inputs: dict[str, Any],
        *,
        augment_source: bool,
        level: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = inputs["source_dce0"]
        mask = inputs["source_mask"]
        if not augment_source:
            return self.spatial_condition(source, mask), mask
        result = self.attenuator(source, mask, level)
        if result.level != level:
            raise RuntimeError("attenuator returned the wrong shared level")
        return self.spatial_condition(result.source, mask), mask

    @staticmethod
    def _validate_metadata(
        value: Any, *, batch_size: int, name: str
    ) -> tuple[_TransitionMetadata, ...]:
        if type(value) is not list or len(value) != batch_size:
            raise ValueError(
                f"paper objective {name} metadata must be a list with length B"
            )
        required_fields = (
            "patient_id",
            "fold",
            "transition_type",
            "transition_id",
        )
        validated = []
        for index, row in enumerate(value):
            if type(row) is not dict:
                raise TypeError(
                    f"paper objective {name} metadata row {index} must be an exact dict"
                )
            if not set(required_fields).issubset(row):
                raise ValueError(
                    f"paper objective {name} metadata row {index} is incomplete"
                )
            fields = {field: row[field] for field in required_fields}
            if any(
                type(field_value) is not str or not field_value.strip()
                for field_value in fields.values()
            ):
                raise ValueError(
                    f"paper objective {name} metadata identity fields "
                    "must be nonempty exact strings"
                )
            validated.append(_TransitionMetadata(**fields))
        return tuple(validated)

    def _validate_model_inputs(
        self,
        inputs: Any,
        *,
        target: Any | None,
        name: str,
    ) -> _ValidatedTransition:
        if (
            not isinstance(inputs, dict)
            or set(inputs) != PAPER_CCL_MODEL_INPUT_KEYS
        ):
            raise ValueError(f"paper objective {name} model inputs are malformed")
        source = inputs["source_dce0"]
        mask = inputs["source_mask"]
        if not isinstance(source, torch.Tensor):
            raise TypeError(f"paper objective {name} source must be a tensor")
        if source.ndim != 5 or source.shape[0] <= 0 or source.shape[1] != 1:
            raise ValueError(
                f"paper objective {name} source must have shape [B,1,D,H,W]"
            )
        if any(size <= 0 for size in source.shape[2:]):
            raise ValueError(
                f"paper objective {name} source dimensions must be positive"
            )
        if not source.is_floating_point():
            raise TypeError(f"paper objective {name} source must use a floating dtype")
        if source.device != self.latent_device:
            raise ValueError(f"paper objective {name} source is on the wrong device")
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"paper objective {name} source mask must be a tensor")
        if mask.shape != source.shape:
            raise ValueError(
                f"paper objective {name} source and mask shapes must align"
            )
        if mask.device != source.device:
            raise ValueError(
                f"paper objective {name} source and mask devices must align"
            )
        if mask.dtype != torch.bool and not mask.is_floating_point():
            raise TypeError(
                f"paper objective {name} source mask must be bool or floating"
            )
        if mask.requires_grad:
            raise ValueError(
                f"paper objective {name} source mask cannot require gradients"
            )

        if target is not None:
            if not isinstance(target, torch.Tensor):
                raise TypeError(f"paper objective {name} target must be a tensor")
            if target.shape != source.shape:
                raise ValueError(
                    f"paper objective {name} source and target shapes must align"
                )
            if target.device != source.device:
                raise ValueError(
                    f"paper objective {name} source and target devices must align"
                )
            if target.dtype != source.dtype or not target.is_floating_point():
                raise TypeError(
                    f"paper objective {name} source and target dtypes must align"
                )

        batch_size = source.shape[0]
        actions = inputs["action_text"]
        clinical = inputs["clinical_text"]
        for field_name, texts in (("action", actions), ("clinical", clinical)):
            if (
                isinstance(texts, (str, bytes))
                or not isinstance(texts, Sequence)
                or len(texts) != batch_size
                or not all(type(text) is str and text for text in texts)
            ):
                raise ValueError(
                    f"paper objective {name} {field_name} text must match batch"
                )
        canonical_actions = tuple(actions)
        parsed_clinical = []
        try:
            for action in canonical_actions:
                parse_action_text(action)
            for text in clinical:
                parsed_clinical.append(parse_clinical_text(text))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"paper objective {name} text must be canonical"
            ) from error

        delta_days = inputs["delta_days"]
        if (
            not isinstance(delta_days, torch.Tensor)
            or delta_days.shape != (batch_size,)
        ):
            raise ValueError(f"paper objective {name} delta_days must have shape [B]")
        if not delta_days.is_floating_point():
            raise TypeError(
                f"paper objective {name} delta_days must use a floating dtype"
            )
        if delta_days.device != source.device:
            raise ValueError(
                f"paper objective {name} delta_days is on the wrong device"
            )
        stage_id = inputs["stage_id"]
        if not isinstance(stage_id, torch.Tensor) or stage_id.shape != (batch_size,):
            raise ValueError(f"paper objective {name} stage_id must have shape [B]")
        if (
            stage_id.dtype == torch.bool
            or stage_id.is_floating_point()
            or stage_id.is_complex()
        ):
            raise TypeError(
                f"paper objective {name} stage_id must use an integer dtype"
            )
        if stage_id.device != source.device:
            raise ValueError(
                f"paper objective {name} stage_id is on the wrong device"
            )

        value_checks: list[_ValidationCheck] = [
            (
                torch.any(~torch.isfinite(source)),
                ValueError,
                f"paper objective {name} source must be finite",
            ),
            (
                torch.any(~torch.isfinite(mask)),
                ValueError,
                f"paper objective {name} source mask must be finite",
            ),
            (
                torch.any((mask != 0) & (mask != 1)),
                ValueError,
                f"paper objective {name} source mask must be binary",
            ),
        ]
        if target is not None:
            value_checks.append(
                (
                    torch.any(~torch.isfinite(target)),
                    ValueError,
                    f"paper objective {name} target must be finite",
                )
            )
        delta_message = (
            f"paper objective {name} delta_days must be finite and positive"
        )
        value_checks.extend(
            (
                (
                    torch.any(~torch.isfinite(delta_days)),
                    ValueError,
                    delta_message,
                ),
                (torch.any(delta_days <= 0), ValueError, delta_message),
                (
                    torch.any((stage_id < 1) | (stage_id > 3)),
                    ValueError,
                    f"paper objective {name} stage_id must be in 1..3",
                ),
            )
        )
        _raise_on_failed_checks(value_checks)

        return _ValidatedTransition(
            value=inputs,
            actions=canonical_actions,
            clinical_subtypes=tuple(
                (fields.hr, fields.her2, fields.mp) for fields in parsed_clinical
            ),
            stage_ids=stage_id,
            metadata=(),
            image_shape=source.shape,
            device=source.device,
            dtype=source.dtype,
        )

    def _validate_transition(self, value: Any, *, name: str) -> _ValidatedTransition:
        if not isinstance(value, dict) or not {
            "model_inputs",
            "metadata",
            "supervision",
        }.issubset(value):
            raise ValueError(f"paper objective {name} must be a supervised transition")
        inputs = value["model_inputs"]
        supervision = value["supervision"]
        if not isinstance(inputs, dict) or not isinstance(supervision, dict):
            raise ValueError(f"paper objective {name} transition is malformed")
        if "target_dce0" not in supervision:
            raise ValueError(f"paper objective {name} target is missing")
        validated = self._validate_model_inputs(
            inputs, target=supervision["target_dce0"], name=name
        )
        metadata = self._validate_metadata(
            value["metadata"], batch_size=validated.image_shape[0], name=name
        )
        return _ValidatedTransition(
            value=value,
            actions=validated.actions,
            clinical_subtypes=validated.clinical_subtypes,
            stage_ids=validated.stage_ids,
            metadata=metadata,
            image_shape=validated.image_shape,
            device=validated.device,
            dtype=validated.dtype,
        )

    def _validate_nested_batch(
        self, batch: Any
    ) -> _ValidatedPaperBatch:
        expected_keys = {
            "anchor",
            "positive",
            "negative_action_texts",
            "ccl_valid_mask",
        }
        if not isinstance(batch, dict) or set(batch) != expected_keys:
            raise ValueError("paper objective batch has a malformed nested structure")
        anchor = self._validate_transition(batch["anchor"], name="anchor")
        valid_mask = batch["ccl_valid_mask"]
        batch_size = anchor.image_shape[0]
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("ccl_valid_mask must be a tensor")
        if valid_mask.dtype != torch.bool:
            raise TypeError("ccl_valid_mask must use a boolean dtype")
        if valid_mask.shape != (batch_size,):
            raise ValueError("ccl_valid_mask must have shape [B]")
        if valid_mask.device != anchor.device:
            raise ValueError("ccl_valid_mask must use the anchor device")
        valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
        valid_positions = tuple(
            valid_indices.detach().cpu().tolist()
        )
        valid_count = len(valid_positions)
        negatives = batch["negative_action_texts"]
        if (
            type(negatives) is not tuple
            or len(negatives) != 2
            or not all(type(arm) is tuple for arm in negatives)
        ):
            raise ValueError("CCL batch requires exactly two negative action arms")
        if valid_count:
            positive = self._validate_transition(batch["positive"], name="positive")
            if (
                positive.image_shape[0] != valid_count
                or positive.image_shape[1:] != anchor.image_shape[1:]
                or positive.device != anchor.device
                or positive.dtype != anchor.dtype
            ):
                raise ValueError(
                    "paper objective anchor and positive tensors must be compatible"
                )
            valid_anchor_actions = tuple(
                anchor.actions[index] for index in valid_positions
            )
            valid_anchor_subtypes = tuple(
                anchor.clinical_subtypes[index] for index in valid_positions
            )
            if (
                positive.actions != valid_anchor_actions
                or positive.clinical_subtypes != valid_anchor_subtypes
            ):
                raise ValueError(
                    "paper objective anchor and positive pairing fields must match"
                )
            for anchor_index, positive_row in zip(
                valid_positions, positive.metadata, strict=True
            ):
                anchor_row = anchor.metadata[anchor_index]
                if anchor_row.patient_id == positive_row.patient_id:
                    raise ValueError(
                        "paper objective positive must use a different patient"
                    )
                if anchor_row.fold != positive_row.fold:
                    raise ValueError(
                        "paper objective anchor and positive folds must match"
                    )
                if anchor_row.transition_type != positive_row.transition_type:
                    raise ValueError(
                        "paper objective anchor and positive transition types must match"
                    )
            _raise_on_failed_checks(
                (
                    (
                        torch.any(
                            positive.stage_ids
                            != anchor.stage_ids.index_select(0, valid_indices)
                        ),
                        ValueError,
                        "paper objective anchor and positive pairing fields must match",
                    ),
                )
            )
            if (
                any(len(arm) != valid_count for arm in negatives)
                or not all(
                    type(action) is str and action
                    for arm in negatives
                    for action in arm
                )
            ):
                raise ValueError(
                    "valid CCL batch negative arms must each have length V"
                )
            for action in (action for arm in negatives for action in arm):
                try:
                    parse_action_text(action)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "valid CCL batch negative actions must be canonical"
                    ) from error
            for compact_row, anchor_action in enumerate(valid_anchor_actions):
                row_negatives = tuple(arm[compact_row] for arm in negatives)
                if len(set(row_negatives)) != 2:
                    raise ValueError(
                        "valid CCL batch requires two distinct negative actions per row"
                    )
                if anchor_action in row_negatives:
                    raise ValueError(
                        "valid CCL batch negative cannot equal anchor action"
                    )
            return _ValidatedPaperBatch(
                anchor=anchor.value,
                positive=positive.value,
                negative_action_texts=negatives,
                valid_mask=valid_mask,
                valid_indices=valid_indices,
                valid_positions=valid_positions,
                batch_size=batch_size,
                valid_count=valid_count,
            )
        if batch["positive"] is not None or negatives != ((), ()):
            raise ValueError(
                "invalid CCL batch cannot contain a positive or negatives"
            )
        return _ValidatedPaperBatch(
            anchor=anchor.value,
            positive=None,
            negative_action_texts=negatives,
            valid_mask=valid_mask,
            valid_indices=valid_indices,
            valid_positions=valid_positions,
            batch_size=batch_size,
            valid_count=valid_count,
        )

    @staticmethod
    def _select_condition(
        condition: PaperConditionOutput, valid_indices: torch.Tensor
    ) -> PaperConditionOutput:
        return PaperConditionOutput(
            global_condition=condition.global_condition.index_select(
                0, valid_indices
            ),
            context_tokens=condition.context_tokens.index_select(0, valid_indices),
            context_mask=condition.context_mask.index_select(0, valid_indices),
        )

    @staticmethod
    def _select_condition_inputs(
        inputs: dict[str, Any],
        *,
        valid_indices: torch.Tensor,
        valid_positions: tuple[int, ...],
    ) -> dict[str, Any]:
        return {
            "action_text": [
                inputs["action_text"][index] for index in valid_positions
            ],
            "clinical_text": [
                inputs["clinical_text"][index] for index in valid_positions
            ],
            "delta_days": inputs["delta_days"].index_select(0, valid_indices),
            "stage_id": inputs["stage_id"].index_select(0, valid_indices),
        }

    def _x0_training_objective(
        self,
        batch: dict[str, Any],
        *,
        augment_source: bool,
        random_seed: int | None,
        base_noise: torch.Tensor | None,
        base_timesteps: torch.Tensor | None,
        attenuation_level: int | None,
    ) -> X0LossOutput:
        if not isinstance(batch, dict) or set(batch) != {"anchor"}:
            raise ValueError("x0 objective batch must contain only the anchor transition")
        anchor = self._validate_transition(batch["anchor"], name="anchor")
        anchor_value = anchor.value
        anchor_inputs = anchor_value["model_inputs"]
        target = self.target_latent(anchor_value["supervision"]["target_dce0"])
        generator = self._generator(target.device, random_seed)

        if augment_source:
            if attenuation_level is None:
                level = self.attenuator.sample_level(generator=generator)
            else:
                if type(attenuation_level) is not int:
                    raise TypeError("attenuation_level must be an exact integer")
                level = attenuation_level
        else:
            if attenuation_level is not None:
                raise ValueError("attenuation_level requires augment_source=True")
            level = 0

        spatial, _neighborhood_mask = self._prepared_spatial(
            anchor_inputs, augment_source=augment_source, level=level
        )
        condition = self._condition(anchor_inputs)
        batch_size = anchor.image_shape[0]
        timesteps = self._timesteps(
            base_timesteps,
            batch_size=batch_size,
            device=target.device,
            minimum=0,
            maximum=self.config.timesteps - 1,
            generator=generator,
            name="base_timesteps",
        )
        noise = self._noise(base_noise, target, generator, name="base_noise")
        noisy = self.q_sample(target, timesteps, noise)
        prediction = self._predict_model_output(noisy, spatial, condition, timesteps)

        # CompVis x0 and OpenAI START_X both supervise the clean x_start directly.
        x0_mse = F.mse_loss(prediction, target)
        output = X0LossOutput(
            total_loss=x0_mse,
            x0_mse=x0_mse,
            x0_mae=F.l1_loss(prediction, target).detach(),
            attenuation_level=x0_mse.new_tensor(level, dtype=torch.long),
        )
        self._validate_x0_output(output)
        return output

    def training_objective(
        self,
        batch: dict[str, Any],
        *,
        augment_source: bool,
        random_seed: int | None = None,
        base_noise: torch.Tensor | None = None,
        base_timesteps: torch.Tensor | None = None,
        ccl_noise: torch.Tensor | None = None,
        ccl_timesteps: torch.Tensor | None = None,
        attenuation_level: int | None = None,
    ) -> PaperLossOutput | X0LossOutput:
        if type(augment_source) is not bool:
            raise TypeError("augment_source must be a bool")
        if self.config.prediction_type == X0_PREDICTION_TYPE:
            if ccl_noise is not None or ccl_timesteps is not None:
                raise ValueError("x0 objective does not accept CCL noise or timesteps")
            return self._x0_training_objective(
                batch,
                augment_source=augment_source,
                random_seed=random_seed,
                base_noise=base_noise,
                base_timesteps=base_timesteps,
                attenuation_level=attenuation_level,
            )
        validated_batch = self._validate_nested_batch(batch)
        anchor = validated_batch.anchor
        positive = validated_batch.positive
        negatives = validated_batch.negative_action_texts

        anchor_inputs = anchor["model_inputs"]
        target = self.target_latent(anchor["supervision"]["target_dce0"])
        generator = self._generator(target.device, random_seed)

        if augment_source:
            if attenuation_level is None:
                level = self.attenuator.sample_level(generator=generator)
            else:
                if type(attenuation_level) is not int:
                    raise TypeError("attenuation_level must be an exact integer")
                level = attenuation_level
        else:
            if attenuation_level is not None:
                raise ValueError("attenuation_level requires augment_source=True")
            level = 0

        anchor_spatial, anchor_neighborhood_mask = self._prepared_spatial(
            anchor_inputs, augment_source=augment_source, level=level
        )
        anchor_condition = self._condition(anchor_inputs)
        batch_size = validated_batch.batch_size
        base_t = self._timesteps(
            base_timesteps,
            batch_size=batch_size,
            device=target.device,
            minimum=0,
            maximum=self.config.timesteps - 1,
            generator=generator,
            name="base_timesteps",
        )
        base_eps = self._noise(
            base_noise, target, generator, name="base_noise"
        )
        base_noisy = self.q_sample(target, base_t, base_eps)
        base_prediction = self._predict_epsilon(
            base_noisy, anchor_spatial, anchor_condition, base_t
        )
        epsilon_mse_train = F.mse_loss(base_prediction, base_eps)
        if self.config.runtime.epsilon_objective == "l2":
            epsilon_loss = epsilon_mse_train
        else:
            epsilon_loss = F.l1_loss(base_prediction, base_eps)
        epsilon_mse = epsilon_mse_train.detach()

        valid_count = validated_batch.valid_count
        if valid_count == 0:
            zero = epsilon_loss.new_zeros(())
            output = PaperLossOutput(
                total_loss=epsilon_loss,
                epsilon_loss=epsilon_loss,
                epsilon_mse=epsilon_mse,
                ccl_loss=zero,
                positive_similarity=zero,
                negative_similarity=zero,
                valid_ccl_fraction=zero,
                attenuation_level=epsilon_loss.new_tensor(level, dtype=torch.long),
            )
            self._validate_output(output)
            return output

        if positive is None:
            raise RuntimeError("validated CCL batch omitted its positive transition")

        positive_inputs = positive["model_inputs"]
        valid_target = target.index_select(0, validated_batch.valid_indices)
        positive_target = self.target_latent(
            positive["supervision"]["target_dce0"]
        )
        if positive_target.shape != valid_target.shape:
            raise ValueError("anchor and positive target latent shapes must match")
        positive_spatial, positive_neighborhood_mask = self._prepared_spatial(
            positive_inputs, augment_source=augment_source, level=level
        )
        positive_condition = self._condition(positive_inputs)
        valid_anchor_inputs = self._select_condition_inputs(
            anchor_inputs,
            valid_indices=validated_batch.valid_indices,
            valid_positions=validated_batch.valid_positions,
        )
        valid_anchor_spatial = anchor_spatial.index_select(
            0, validated_batch.valid_indices
        )
        valid_anchor_condition = self._select_condition(
            anchor_condition, validated_batch.valid_indices
        )
        valid_anchor_neighborhood_mask = anchor_neighborhood_mask.index_select(
            0, validated_batch.valid_indices
        )
        ccl_t = self._timesteps(
            ccl_timesteps,
            batch_size=valid_count,
            device=target.device,
            minimum=self.config.runtime.ccl_timestep_min,
            maximum=self.config.runtime.ccl_timestep_max,
            generator=generator,
            name="ccl_timesteps",
        )
        shared_eps = self._noise(
            ccl_noise, valid_target, generator, name="ccl_noise"
        )
        anchor_noisy = self.q_sample(valid_target, ccl_t, shared_eps)
        positive_noisy = self.q_sample(positive_target, ccl_t, shared_eps)

        anchor_prediction = self._predict_epsilon(
            anchor_noisy, valid_anchor_spatial, valid_anchor_condition, ccl_t
        )
        positive_prediction = self._predict_epsilon(
            positive_noisy, positive_spatial, positive_condition, ccl_t
        )
        negative_predictions = []
        for action_arm in negatives:
            negative_condition = self._condition(
                valid_anchor_inputs,
                action_text=action_arm,
                device=target.device,
            )
            negative_predictions.append(
                self._predict_epsilon(
                    anchor_noisy, valid_anchor_spatial, negative_condition, ccl_t
                )
            )

        anchor_x0 = self.predicted_x0(anchor_noisy, anchor_prediction, ccl_t)
        positive_x0 = self.predicted_x0(
            positive_noisy, positive_prediction, ccl_t
        )
        negative_x0 = [
            self.predicted_x0(anchor_noisy, prediction, ccl_t)
            for prediction in negative_predictions
        ]
        anchor_neighborhood = latent_tumor_neighborhood(
            valid_anchor_neighborhood_mask, target.shape[-3:]
        )
        positive_neighborhood = latent_tumor_neighborhood(
            positive_neighborhood_mask, target.shape[-3:]
        )
        anchor_representation = masked_latent_representation(
            anchor_x0, anchor_neighborhood
        )
        positive_representation = masked_latent_representation(
            positive_x0, positive_neighborhood
        )
        negative_representations = torch.stack(
            [
                masked_latent_representation(value, anchor_neighborhood)
                for value in negative_x0
            ],
            dim=1,
        )
        ccl = combo_contrastive_loss(
            anchor_representation,
            positive_representation,
            negative_representations,
            temperature=self.config.runtime.ccl_temperature,
        )
        valid_fraction = epsilon_loss.new_tensor(valid_count / batch_size)
        ccl_loss = ccl.loss * valid_fraction
        total = epsilon_loss + self.config.runtime.ccl_weight * ccl_loss
        output = PaperLossOutput(
            total_loss=total,
            epsilon_loss=epsilon_loss,
            epsilon_mse=epsilon_mse,
            ccl_loss=ccl_loss,
            positive_similarity=(
                ccl.positive_similarity.detach() * valid_fraction
            ),
            negative_similarity=(
                ccl.negative_similarity.detach() * valid_fraction
            ),
            valid_ccl_fraction=valid_fraction,
            attenuation_level=epsilon_loss.new_tensor(level, dtype=torch.long),
        )
        self._validate_output(output)
        return output

    def _validate_output(self, output: PaperLossOutput) -> None:
        named_values = (
            ("total loss", output.total_loss),
            ("epsilon loss", output.epsilon_loss),
            ("epsilon MSE", output.epsilon_mse),
            ("CCL loss", output.ccl_loss),
            ("positive similarity", output.positive_similarity),
            ("negative similarity", output.negative_similarity),
            ("valid CCL fraction", output.valid_ccl_fraction),
            ("attenuation level", output.attenuation_level),
        )
        for name, value in named_values:
            if value.ndim != 0:
                raise ValueError(f"{name} must be a scalar")
        finite_values = torch.stack([value.detach() for _name, value in named_values])
        if not bool(torch.isfinite(finite_values).all()):
            for name, value in named_values:
                self._finite(value, name)

    def _validate_x0_output(self, output: X0LossOutput) -> None:
        named_values = (
            ("total loss", output.total_loss),
            ("x0 MSE", output.x0_mse),
            ("x0 MAE", output.x0_mae),
            ("attenuation level", output.attenuation_level),
        )
        for name, value in named_values:
            if value.ndim != 0:
                raise ValueError(f"{name} must be a scalar")
        for name, value in named_values:
            self._finite(value, name)

    @torch.no_grad()
    def update_ema(self) -> None:
        parameters = dict(self.denoiser.named_parameters())
        ema_parameters = dict(self.ema_denoiser.named_parameters())
        buffers = dict(self.denoiser.named_buffers())
        ema_buffers = dict(self.ema_denoiser.named_buffers())
        if parameters.keys() != ema_parameters.keys():
            raise RuntimeError("EMA and online parameter keys must match")
        if buffers.keys() != ema_buffers.keys():
            raise RuntimeError("EMA and online buffer keys must match")
        for name, parameter in parameters.items():
            ema_parameter = ema_parameters[name]
            if parameter.shape != ema_parameter.shape:
                raise RuntimeError(f"EMA parameter shape mismatch for {name}")
            if parameter.device != ema_parameter.device:
                raise RuntimeError(f"EMA parameter device mismatch for {name}")
        for name, buffer in buffers.items():
            ema_buffer = ema_buffers[name]
            if buffer.shape != ema_buffer.shape:
                raise RuntimeError(f"EMA buffer shape mismatch for {name}")
            if buffer.device != ema_buffer.device:
                raise RuntimeError(f"EMA buffer device mismatch for {name}")

        decay = self.config.ema_decay
        for name, parameter in parameters.items():
            ema_parameters[name].lerp_(parameter, 1.0 - decay)
        for name, buffer in buffers.items():
            ema_buffers[name].copy_(buffer)

    @torch.no_grad()
    def p_sample_latent_step(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        spatial_condition: torch.Tensor,
        condition: PaperConditionOutput,
        *,
        posterior_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        posterior_noise = self._validate_reverse_inputs(
            latent,
            timesteps,
            spatial_condition,
            condition,
            posterior_noise,
        )
        model_output = self._validated_prediction(
            self.ema_denoiser(latent, spatial_condition, condition, timesteps),
            reference=latent,
        )
        predicted_start = self._model_output_to_x0(
            latent, model_output, timesteps
        )
        posterior_mean = self._extract(
            self.posterior_mean_coefficient_start, timesteps, latent.shape
        ) * predicted_start + self._extract(
            self.posterior_mean_coefficient_current, timesteps, latent.shape
        ) * latent
        nonzero = (timesteps != 0).to(latent.dtype).reshape(
            timesteps.shape[0], *((1,) * (latent.ndim - 1))
        )
        posterior_standard_deviation = (
            0.5
            * self._extract(self.posterior_log_variance, timesteps, latent.shape)
        ).exp()
        result = (
            posterior_mean
            + nonzero * posterior_standard_deviation * posterior_noise
        )
        self._finite(result, "sampled latent")
        return result

    def _uniform_sampling_timesteps(
        self, sampling_steps: int, *, sampler_name: str
    ) -> tuple[int, ...]:
        if type(sampling_steps) is not int:
            raise TypeError(f"{sampler_name} sampling_steps must be an exact integer")
        if not 1 <= sampling_steps <= self.config.timesteps:
            raise ValueError(
                f"{sampler_name} sampling_steps must be within the diffusion schedule"
            )
        if sampling_steps == 1:
            if self.config.timesteps != 1:
                raise ValueError(
                    f"{sampler_name} requires at least two steps to include both endpoints"
                )
            return (0,)
        ascending = torch.linspace(
            0,
            self.config.timesteps - 1,
            steps=sampling_steps,
            dtype=torch.float64,
        ).round().to(dtype=torch.long)
        ascending[0] = 0
        ascending[-1] = self.config.timesteps - 1
        if int(torch.unique(ascending).numel()) != sampling_steps:
            raise RuntimeError(
                f"{sampler_name} timestep construction produced duplicate steps"
            )
        return tuple(reversed(ascending.tolist()))

    def _ddim_sampling_timesteps(self, sampling_steps: int) -> tuple[int, ...]:
        return self._uniform_sampling_timesteps(
            sampling_steps, sampler_name="DDIM"
        )

    def _respaced_ddpm_sampling_timesteps(
        self, sampling_steps: int
    ) -> tuple[int, ...]:
        return self._uniform_sampling_timesteps(
            sampling_steps, sampler_name="respaced DDPM"
        )

    @torch.no_grad()
    def ddim_eta0_latent_step(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        previous_timesteps: torch.Tensor,
        spatial_condition: torch.Tensor,
        condition: PaperConditionOutput,
    ) -> torch.Tensor:
        self._validate_reverse_state(
            latent,
            timesteps,
            spatial_condition,
            condition,
        )
        if not isinstance(previous_timesteps, torch.Tensor):
            raise TypeError("previous timesteps must be a tensor")
        if previous_timesteps.shape != timesteps.shape:
            raise ValueError("previous timesteps must have shape [B]")
        if (
            previous_timesteps.dtype == torch.bool
            or previous_timesteps.is_floating_point()
            or previous_timesteps.is_complex()
        ):
            raise TypeError("previous timesteps must use an integer dtype")
        if previous_timesteps.device != latent.device:
            raise ValueError("previous timesteps must use the latent device")
        if bool(
            torch.any(
                (previous_timesteps < -1)
                | (previous_timesteps >= timesteps)
                | ((previous_timesteps == -1) != (timesteps == 0))
            )
        ):
            raise ValueError("previous timesteps do not precede the current timesteps")

        model_output = self._validated_prediction(
            self.ema_denoiser(latent, spatial_condition, condition, timesteps),
            reference=latent,
        )
        predicted_start = self._model_output_to_x0(
            latent, model_output, timesteps
        )
        predicted_epsilon = (
            model_output
            if self.config.prediction_type == EPSILON_PREDICTION_TYPE
            else self._epsilon_from_x0(latent, predicted_start, timesteps)
        )
        safe_previous = previous_timesteps.clamp_min(0).to(dtype=torch.long)
        previous_alpha = self.alpha_bars[safe_previous]
        previous_alpha = torch.where(
            previous_timesteps >= 0,
            previous_alpha,
            torch.ones_like(previous_alpha),
        ).to(dtype=latent.dtype)
        previous_alpha = previous_alpha.reshape(
            previous_timesteps.shape[0], *((1,) * (latent.ndim - 1))
        )
        result = (
            previous_alpha.sqrt() * predicted_start
            + (1.0 - previous_alpha).sqrt() * predicted_epsilon
        )
        self._finite(result, "DDIM sampled latent")
        return result

    @torch.no_grad()
    def respaced_ddpm_latent_step(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        previous_timesteps: torch.Tensor,
        spatial_condition: torch.Tensor,
        condition: PaperConditionOutput,
        *,
        posterior_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        posterior_noise = self._validate_reverse_inputs(
            latent,
            timesteps,
            spatial_condition,
            condition,
            posterior_noise,
        )
        if not isinstance(previous_timesteps, torch.Tensor):
            raise TypeError("previous timesteps must be a tensor")
        if previous_timesteps.shape != timesteps.shape:
            raise ValueError("previous timesteps must have shape [B]")
        if (
            previous_timesteps.dtype == torch.bool
            or previous_timesteps.is_floating_point()
            or previous_timesteps.is_complex()
        ):
            raise TypeError("previous timesteps must use an integer dtype")
        if previous_timesteps.device != latent.device:
            raise ValueError("previous timesteps must use the latent device")
        if bool(
            torch.any(
                (previous_timesteps < -1)
                | (previous_timesteps >= timesteps)
                | ((previous_timesteps == -1) != (timesteps == 0))
            )
        ):
            raise ValueError("previous timesteps do not precede the current timesteps")

        model_output = self._validated_prediction(
            self.ema_denoiser(latent, spatial_condition, condition, timesteps),
            reference=latent,
        )
        predicted_start = self._model_output_to_x0(
            latent, model_output, timesteps
        )
        predicted_epsilon = self._epsilon_from_x0(
            latent, predicted_start, timesteps
        )
        current_alpha = self._extract(self.alpha_bars, timesteps, latent.shape)
        safe_previous = previous_timesteps.clamp_min(0).to(dtype=torch.long)
        previous_alpha = self.alpha_bars[safe_previous]
        previous_alpha = torch.where(
            previous_timesteps >= 0,
            previous_alpha,
            torch.ones_like(previous_alpha),
        ).to(dtype=latent.dtype)
        previous_alpha = previous_alpha.reshape(
            previous_timesteps.shape[0], *((1,) * (latent.ndim - 1))
        )
        variance = (
            (1.0 - previous_alpha)
            / (1.0 - current_alpha).clamp_min(torch.finfo(latent.dtype).tiny)
            * (1.0 - current_alpha / previous_alpha)
        ).clamp_min(0.0)
        nonzero = (previous_timesteps >= 0).to(latent.dtype).reshape(
            previous_timesteps.shape[0], *((1,) * (latent.ndim - 1))
        )
        result = (
            previous_alpha.sqrt() * predicted_start
            + (1.0 - previous_alpha - variance).clamp_min(0.0).sqrt()
            * predicted_epsilon
            + nonzero * variance.sqrt() * posterior_noise
        )
        self._finite(result, "respaced DDPM sampled latent")
        return result

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
        inputs = {
            "source_dce0": source_dce0,
            "source_mask": source_mask,
            "action_text": action_text,
            "clinical_text": clinical_text,
            "delta_days": delta_days,
            "stage_id": stage_id,
        }
        self._validate_model_inputs(inputs, target=None, name="sampling")
        expected_noise_shape = (
            source_dce0.shape[0],
            *MRI_VQGAN_DEFAULTS.latent_shape(tuple(source_dce0.shape[-3:])),
        )
        if noise is not None:
            self._validate_exact_noise(
                noise,
                expected_shape=expected_noise_shape,
                device=source_dce0.device,
                dtype=source_dce0.dtype,
                name="sampling noise",
            )

        spatial = self.spatial_condition(source_dce0, source_mask)
        expected_spatial_shape = (
            expected_noise_shape[0],
            self.config.spatial_condition_channels,
            *expected_noise_shape[-3:],
        )
        if spatial.shape != expected_spatial_shape:
            raise RuntimeError("VQGAN returned an unexpected paper latent shape")
        condition = self._condition(inputs)
        if noise is None:
            latent = torch.randn(
                spatial.shape[0],
                self.config.noisy_channels,
                *spatial.shape[-3:],
                device=spatial.device,
                dtype=spatial.dtype,
            )
        else:
            if noise.dtype != spatial.dtype:
                raise TypeError("sampling noise must use the spatial condition dtype")
            latent = noise
        for timestep in reversed(range(self.config.timesteps)):
            times = torch.full(
                (latent.shape[0],),
                timestep,
                device=latent.device,
                dtype=torch.long,
            )
            latent = self.p_sample_latent_step(
                latent, times, spatial, condition
            )
        return latent

    @torch.no_grad()
    def ddim_sample_latent(
        self,
        source_dce0: torch.Tensor,
        source_mask: torch.Tensor,
        action_text: Sequence[str],
        clinical_text: Sequence[str],
        delta_days: torch.Tensor,
        stage_id: torch.Tensor,
        *,
        sampling_steps: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        schedule = self._ddim_sampling_timesteps(sampling_steps)
        inputs = {
            "source_dce0": source_dce0,
            "source_mask": source_mask,
            "action_text": action_text,
            "clinical_text": clinical_text,
            "delta_days": delta_days,
            "stage_id": stage_id,
        }
        self._validate_model_inputs(inputs, target=None, name="sampling")
        expected_noise_shape = (
            source_dce0.shape[0],
            *MRI_VQGAN_DEFAULTS.latent_shape(tuple(source_dce0.shape[-3:])),
        )
        if noise is not None:
            self._validate_exact_noise(
                noise,
                expected_shape=expected_noise_shape,
                device=source_dce0.device,
                dtype=source_dce0.dtype,
                name="sampling noise",
            )

        spatial = self.spatial_condition(source_dce0, source_mask)
        expected_spatial_shape = (
            expected_noise_shape[0],
            self.config.spatial_condition_channels,
            *expected_noise_shape[-3:],
        )
        if spatial.shape != expected_spatial_shape:
            raise RuntimeError("VQGAN returned an unexpected paper latent shape")
        condition = self._condition(inputs)
        if noise is None:
            latent = torch.randn(
                spatial.shape[0],
                self.config.noisy_channels,
                *spatial.shape[-3:],
                device=spatial.device,
                dtype=spatial.dtype,
            )
        else:
            if noise.dtype != spatial.dtype:
                raise TypeError("sampling noise must use the spatial condition dtype")
            latent = noise
        for index, timestep in enumerate(schedule):
            previous_timestep = schedule[index + 1] if index + 1 < len(schedule) else -1
            times = torch.full(
                (latent.shape[0],),
                timestep,
                device=latent.device,
                dtype=torch.long,
            )
            previous_times = torch.full_like(times, previous_timestep)
            latent = self.ddim_eta0_latent_step(
                latent,
                times,
                previous_times,
                spatial,
                condition,
            )
        return latent

    @torch.no_grad()
    def respaced_ddpm_sample_latent(
        self,
        source_dce0: torch.Tensor,
        source_mask: torch.Tensor,
        action_text: Sequence[str],
        clinical_text: Sequence[str],
        delta_days: torch.Tensor,
        stage_id: torch.Tensor,
        *,
        sampling_steps: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        schedule = self._respaced_ddpm_sampling_timesteps(sampling_steps)
        inputs = {
            "source_dce0": source_dce0,
            "source_mask": source_mask,
            "action_text": action_text,
            "clinical_text": clinical_text,
            "delta_days": delta_days,
            "stage_id": stage_id,
        }
        self._validate_model_inputs(inputs, target=None, name="sampling")
        expected_noise_shape = (
            source_dce0.shape[0],
            *MRI_VQGAN_DEFAULTS.latent_shape(tuple(source_dce0.shape[-3:])),
        )
        if noise is not None:
            self._validate_exact_noise(
                noise,
                expected_shape=expected_noise_shape,
                device=source_dce0.device,
                dtype=source_dce0.dtype,
                name="sampling noise",
            )

        spatial = self.spatial_condition(source_dce0, source_mask)
        expected_spatial_shape = (
            expected_noise_shape[0],
            self.config.spatial_condition_channels,
            *expected_noise_shape[-3:],
        )
        if spatial.shape != expected_spatial_shape:
            raise RuntimeError("VQGAN returned an unexpected paper latent shape")
        condition = self._condition(inputs)
        if noise is None:
            latent = torch.randn(
                spatial.shape[0],
                self.config.noisy_channels,
                *spatial.shape[-3:],
                device=spatial.device,
                dtype=spatial.dtype,
            )
        else:
            if noise.dtype != spatial.dtype:
                raise TypeError("sampling noise must use the spatial condition dtype")
            latent = noise
        for index, timestep in enumerate(schedule):
            previous_timestep = schedule[index + 1] if index + 1 < len(schedule) else -1
            times = torch.full(
                (latent.shape[0],),
                timestep,
                device=latent.device,
                dtype=torch.long,
            )
            previous_times = torch.full_like(times, previous_timestep)
            latent = self.respaced_ddpm_latent_step(
                latent,
                times,
                previous_times,
                spatial,
                condition,
            )
        return latent

    def _resolve_sampling_override(
        self, sampler: str, sampling_steps: int | None
    ) -> tuple[str, int]:
        if type(sampler) is not str:
            raise TypeError("sampler must be an exact string")
        if sampling_steps is not None and type(sampling_steps) is not int:
            raise TypeError("sampling_steps must be an exact integer")
        if sampler == ANCESTRAL_DDPM_SAMPLER:
            if sampling_steps not in {None, self.config.timesteps}:
                raise ValueError("ancestral DDPM must use the full diffusion schedule")
            return sampler, self.config.timesteps
        if sampler not in {DDIM_ETA0_SAMPLER, RESPACED_DDPM_SAMPLER}:
            raise ValueError(
                "sampler must be ancestral_ddpm, respaced_ddpm, or ddim_eta0"
            )
        resolved_steps = self.config.timesteps if sampling_steps is None else sampling_steps
        if sampler == DDIM_ETA0_SAMPLER:
            self._ddim_sampling_timesteps(resolved_steps)
        else:
            self._respaced_ddpm_sampling_timesteps(resolved_steps)
        return sampler, resolved_steps

    @torch.no_grad()
    def sample_with_diagnostics(
        self,
        *args: Any,
        sampler: str = ANCESTRAL_DDPM_SAMPLER,
        sampling_steps: int | None = None,
        **kwargs: Any,
    ) -> DiffusionSample:
        sampler, resolved_steps = self._resolve_sampling_override(
            sampler, sampling_steps
        )
        if sampler == ANCESTRAL_DDPM_SAMPLER:
            normalized = self.ancestral_sample_latent(*args, **kwargs)
        elif sampler == DDIM_ETA0_SAMPLER:
            normalized = self.ddim_sample_latent(
                *args,
                sampling_steps=resolved_steps,
                **kwargs,
            )
        else:
            normalized = self.respaced_ddpm_sample_latent(
                *args,
                sampling_steps=resolved_steps,
                **kwargs,
            )
        continuous = self.denormalize_latent(normalized)
        self._finite(normalized, "normalized latent")
        self._finite(continuous, "continuous latent")
        quantized, diagnostics = self.vqgan.quantizer(continuous)
        indices = diagnostics.get("indices")
        perplexity = diagnostics.get("perplexity")
        if not isinstance(indices, torch.Tensor) or not isinstance(
            perplexity, torch.Tensor
        ):
            raise RuntimeError("VQGAN quantizer did not return codebook diagnostics")
        self._finite(quantized, "quantized latent")
        self._finite(perplexity, "codebook perplexity")
        image = self.vqgan.decode(quantized)
        self._finite(image, "decoded image")
        return DiffusionSample(
            image=image,
            normalized_latent=normalized,
            continuous_latent=continuous,
            code_indices=indices,
            unique_code_count=int(torch.unique(indices).numel()),
            effective_code_count=float(perplexity),
        )

    @torch.no_grad()
    def sample(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.sample_with_diagnostics(*args, **kwargs).image


class PaperDiffusionTrainingSystem(pl.LightningModule):
    def __init__(
        self,
        model: PaperFaithfulLatentDiffusion,
        *,
        learning_rate: float = 1e-4,
        checkpoint_identity: PaperCheckpointIdentity | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, PaperFaithfulLatentDiffusion):
            raise TypeError("model must be a PaperFaithfulLatentDiffusion")
        if type(learning_rate) is not float:
            raise TypeError("learning_rate must be an exact float")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            checkpoint_identity is not None
            and type(checkpoint_identity) is not PaperCheckpointIdentity
        ):
            raise TypeError(
                "checkpoint_identity must be an exact PaperCheckpointIdentity"
            )
        self.model = model
        self.learning_rate = learning_rate
        self.checkpoint_identity = checkpoint_identity

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        destination = kwargs.pop("destination", None)
        prefix = kwargs.pop("prefix", "")
        kwargs.pop("keep_vars", False)
        if kwargs:
            raise TypeError(f"unsupported state_dict arguments: {sorted(kwargs)}")
        if args:
            if len(args) > 3:
                raise TypeError("state_dict accepts at most three positional arguments")
            destination = args[0]
            if len(args) >= 2:
                prefix = args[1]
        if not isinstance(prefix, str):
            raise TypeError("state_dict prefix must be a string")
        state = _paper_state_dict(self.model, prefix=f"{prefix}model.")
        if destination is not None:
            destination.update(state)
            return destination
        return state

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> _IncompatibleKeys:
        del strict, assign
        _load_paper_state(self.model, state_dict, prefix="model.")
        return _IncompatibleKeys([], [])

    def _required_checkpoint_identity(self) -> PaperCheckpointIdentity:
        if self.checkpoint_identity is None:
            raise ValueError("paper Lightning checkpoint identity is required")
        return self.checkpoint_identity

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        identity = self._required_checkpoint_identity()
        _validate_model_identity(self.model, identity)
        checkpoint["schema_version"] = PAPER_CHECKPOINT_SCHEMA_VERSION
        checkpoint["identity"] = _paper_identity_payload(identity)
        _validate_checkpoint_metadata(checkpoint, identity)
        _validate_paper_state(
            self.model,
            checkpoint["state_dict"],
            prefix="model.",
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        identity = self._required_checkpoint_identity()
        _, _, state = _validate_checkpoint_metadata(checkpoint, identity)
        _validate_model_identity(self.model, identity)
        _validate_paper_state(self.model, state, prefix="model.")

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = {
            "denoiser": [
                parameter
                for parameter in self.model.denoiser.parameters()
                if parameter.requires_grad
            ],
            "conditioner": [
                parameter
                for parameter in self.model.conditioner.parameters()
                if parameter.requires_grad
            ],
        }
        identifiers = [
            id(parameter)
            for parameters in groups.values()
            for parameter in parameters
        ]
        if len(identifiers) != len(set(identifiers)):
            raise RuntimeError("paper optimizer parameter groups overlap")
        if not groups["denoiser"]:
            raise ValueError("paper denoiser has no trainable parameters")
        if not groups["conditioner"]:
            raise ValueError("paper conditioner has no trainable parameters")
        return groups

    @staticmethod
    def _validation_seed(batch: dict[str, Any]) -> int:
        if type(batch) is not dict:
            raise TypeError("paper validation batch must be an exact dictionary")
        anchor = batch.get("anchor")
        if type(anchor) is not dict:
            raise ValueError("paper validation batch is missing its anchor")
        metadata = anchor.get("metadata")
        if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)):
            raise ValueError("paper validation anchor metadata is invalid")
        transition_ids: list[str] = []
        for row in metadata:
            if type(row) is not dict or type(row.get("transition_id")) is not str:
                raise ValueError("paper validation transition_id is invalid")
            transition_id = row["transition_id"]
            if not transition_id:
                raise ValueError("paper validation transition_id must be nonempty")
            transition_ids.append(transition_id)
        if not transition_ids:
            raise ValueError("paper validation transition_id is missing")
        encoded = "\x1f".join(transition_ids).encode("utf-8")
        return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")

    def _log_output(
        self,
        stage: str,
        output: PaperLossOutput | X0LossOutput,
        *,
        batch_size: int,
    ) -> None:
        if isinstance(output, X0LossOutput):
            fields = (
                ("x0_mse", output.x0_mse.detach()),
                ("x0_mae", output.x0_mae),
                ("total_loss", output.total_loss.detach()),
            )
        else:
            fields = (
                ("epsilon_mse", output.epsilon_mse),
                ("ccl_loss", output.ccl_loss),
                ("total_loss", output.total_loss),
                ("positive_similarity", output.positive_similarity),
                ("negative_similarity", output.negative_similarity),
                ("valid_ccl_fraction", output.valid_ccl_fraction),
            )
        if stage == "train":
            fields = (
                *fields,
                ("attenuation_level", output.attenuation_level.float()),
            )
        for name, value in fields:
            self.log(
                f"{stage}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=stage == "val",
                batch_size=batch_size,
            )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        output = self.model.training_objective(batch, augment_source=True)
        batch_size = batch["anchor"]["model_inputs"]["source_dce0"].shape[0]
        self._log_output("train", output, batch_size=batch_size)
        return output.total_loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        output = self.model.training_objective(
            batch,
            augment_source=False,
            random_seed=self._validation_seed(batch),
        )
        batch_size = batch["anchor"]["model_inputs"]["source_dce0"].shape[0]
        self._log_output("val", output, batch_size=batch_size)

    def on_after_backward(self) -> None:
        for name, parameter in self.named_parameters():
            gradient = parameter.grad
            if parameter.requires_grad and gradient is not None and not bool(
                torch.isfinite(gradient).all()
            ):
                raise FloatingPointError(f"nonfinite gradient: {name}")

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Any = None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.model.update_ema()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        groups = self.trainable_parameter_groups()
        parameters = [*groups["denoiser"], *groups["conditioner"]]
        return torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
        )


class RegisteredLargePaperDiffusionTrainingSystem(PaperDiffusionTrainingSystem):
    def __init__(
        self,
        model: PaperFaithfulLatentDiffusion,
        *,
        learning_rate: float = 1e-4,
        checkpoint_identity: RegisteredLargeCheckpointIdentity | None = None,
    ) -> None:
        pl.LightningModule.__init__(self)
        if not isinstance(model, PaperFaithfulLatentDiffusion):
            raise TypeError("model must be a PaperFaithfulLatentDiffusion")
        if type(learning_rate) is not float:
            raise TypeError("learning_rate must be an exact float")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            checkpoint_identity is not None
            and type(checkpoint_identity) is not RegisteredLargeCheckpointIdentity
        ):
            raise TypeError(
                "checkpoint_identity must be an exact "
                "RegisteredLargeCheckpointIdentity"
            )
        self.model = model
        self.learning_rate = learning_rate
        self.checkpoint_identity = checkpoint_identity

    def _required_registered_identity(self) -> RegisteredLargeCheckpointIdentity:
        if self.checkpoint_identity is None:
            raise ValueError(
                "registered-large Lightning checkpoint identity is required"
            )
        return self.checkpoint_identity

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        identity = self._required_registered_identity()
        _validate_registered_large_model_identity(self.model, identity)
        checkpoint["schema_version"] = REGISTERED_LARGE_CHECKPOINT_SCHEMA_VERSION
        checkpoint["identity"] = _registered_large_identity_payload(identity)
        _validate_registered_large_metadata(
            checkpoint,
            identity,
        )
        _validate_paper_state(
            self.model,
            checkpoint["state_dict"],
            prefix="model.",
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        identity = self._required_registered_identity()
        _, _, state = _validate_registered_large_metadata(
            checkpoint,
            identity,
        )
        _validate_registered_large_model_identity(self.model, identity)
        _validate_paper_state(self.model, state, prefix="model.")


class RegisteredX0PaperDiffusionTrainingSystem(PaperDiffusionTrainingSystem):
    def __init__(
        self,
        model: PaperFaithfulLatentDiffusion,
        *,
        learning_rate: float = 1e-4,
        checkpoint_identity: RegisteredX0CheckpointIdentity | None = None,
    ) -> None:
        pl.LightningModule.__init__(self)
        if not isinstance(model, PaperFaithfulLatentDiffusion):
            raise TypeError("model must be a PaperFaithfulLatentDiffusion")
        if type(learning_rate) is not float:
            raise TypeError("learning_rate must be an exact float")
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            checkpoint_identity is not None
            and type(checkpoint_identity) is not RegisteredX0CheckpointIdentity
        ):
            raise TypeError(
                "checkpoint_identity must be an exact RegisteredX0CheckpointIdentity"
            )
        self.model = model
        self.learning_rate = learning_rate
        self.checkpoint_identity = checkpoint_identity

    def _required_x0_identity(self) -> RegisteredX0CheckpointIdentity:
        if self.checkpoint_identity is None:
            raise ValueError("registered x0 Lightning checkpoint identity is required")
        return self.checkpoint_identity

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        identity = self._required_x0_identity()
        _validate_registered_x0_model_identity(self.model, identity)
        checkpoint["schema_version"] = REGISTERED_X0_CHECKPOINT_SCHEMA_VERSION
        checkpoint["identity"] = _registered_x0_identity_payload(identity)
        _validate_registered_x0_metadata(checkpoint, identity)
        _validate_paper_state(self.model, checkpoint["state_dict"], prefix="model.")

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        identity = self._required_x0_identity()
        _, _, state = _validate_registered_x0_metadata(checkpoint, identity)
        _validate_registered_x0_model_identity(self.model, identity)
        _validate_paper_state(self.model, state, prefix="model.")
