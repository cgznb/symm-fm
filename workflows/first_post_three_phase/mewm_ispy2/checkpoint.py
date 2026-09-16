from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .conditioning import MEDGEMMA_MODEL_ID, MEDGEMMA_REVISION
from .contracts import (
    ANCESTRAL_DDPM_SAMPLER,
    CONTINUOUS_CODEBOOK_MINMAX_CONTRACT,
    CT_DENOISER_ARCHITECTURE,
    DIFFUSION_CHECKPOINT_SCHEMA_VERSION,
    EPSILON_PREDICTION_TYPE,
    FILM_DENOISER_ARCHITECTURE,
)
from .diffusion import ConditionalLatentDiffusion
from .manifest import ACCEPTED_BACKENDS


CHECKPOINT_SCHEMA_VERSION = DIFFUSION_CHECKPOINT_SCHEMA_VERSION


@dataclass(frozen=True)
class CheckpointIdentity:
    medgemma_model_id: str
    medgemma_revision: str
    vqgan_sha256: str
    data_contract_sha256: str
    data_backend: str
    denoiser_architecture: str
    denoiser_input_channels: int
    semantic_channels: int
    prediction_type: str
    timesteps: int
    sampler: str
    latent_contract: str
    ema_decay: float

    def __post_init__(self) -> None:
        if self.medgemma_model_id != MEDGEMMA_MODEL_ID:
            raise ValueError("MedGemma model ID does not match the locked identity")
        if self.medgemma_revision != MEDGEMMA_REVISION:
            raise ValueError("MedGemma revision does not match the locked identity")
        if len(self.vqgan_sha256) != 64 or len(self.data_contract_sha256) != 64:
            raise ValueError("checkpoint SHA256 identities must contain 64 hex characters")
        if self.data_backend not in ACCEPTED_BACKENDS:
            raise ValueError("checkpoint data backend is invalid")
        expected_channels = {
            FILM_DENOISER_ARCHITECTURE: (17, 0),
            CT_DENOISER_ARCHITECTURE: (49, 32),
        }
        if expected_channels.get(self.denoiser_architecture) != (
            self.denoiser_input_channels,
            self.semantic_channels,
        ):
            raise ValueError("checkpoint denoiser architecture is invalid")
        if self.prediction_type != EPSILON_PREDICTION_TYPE:
            raise ValueError("checkpoint prediction type is invalid")
        if self.timesteps <= 0:
            raise ValueError("checkpoint diffusion timesteps are invalid")
        if self.sampler != ANCESTRAL_DDPM_SAMPLER:
            raise ValueError("checkpoint diffusion sampler is invalid")
        if self.latent_contract != CONTINUOUS_CODEBOOK_MINMAX_CONTRACT:
            raise ValueError("checkpoint diffusion latent contract is invalid")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("checkpoint EMA decay is invalid")


def _prefix_state(module: torch.nn.Module, prefix: str) -> dict[str, torch.Tensor]:
    return {f"{prefix}{key}": value for key, value in module.state_dict().items()}


def _adapter_state(model: ConditionalLatentDiffusion) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    state.update(_prefix_state(model.denoiser, "denoiser."))
    state.update(_prefix_state(model.ema_denoiser, "ema_denoiser."))
    state.update(
        _prefix_state(model.conditioner.text_projection, "conditioner.text_projection.")
    )
    state.update(
        _prefix_state(model.conditioner.stage_embedding, "conditioner.stage_embedding.")
    )
    state.update(_prefix_state(model.conditioner.fusion, "conditioner.fusion."))
    for key, value in model.conditioner.text_tower.state_dict().items():
        if "lora_" in key:
            state[f"conditioner.text_tower.{key}"] = value
    if not any("lora_" in key for key in state):
        raise ValueError("diffusion checkpoint has no MedGemma LoRA adapter state")
    return state


def _validate_model_schedule(
    model: ConditionalLatentDiffusion, identity: CheckpointIdentity
) -> None:
    expected = (
        model.config.denoiser_architecture,
        model.config.denoiser_input_channels,
        model.config.semantic_channels,
        model.config.prediction_type,
        model.config.timesteps,
        model.config.sampler,
        model.config.latent_contract,
        model.config.ema_decay,
    )
    actual = (
        identity.denoiser_architecture,
        identity.denoiser_input_channels,
        identity.semantic_channels,
        identity.prediction_type,
        identity.timesteps,
        identity.sampler,
        identity.latent_contract,
        identity.ema_decay,
    )
    if actual != expected:
        raise ValueError("checkpoint identity does not match the model diffusion schedule")


def build_diffusion_checkpoint(
    model: ConditionalLatentDiffusion,
    identity: CheckpointIdentity,
    *,
    global_step: int,
) -> dict[str, Any]:
    _validate_model_schedule(model, identity)
    state = _adapter_state(model)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity": asdict(identity),
        "global_step": int(global_step),
        "state_dict": state,
    }


def _validate_identity(payload: dict[str, Any], expected: CheckpointIdentity) -> None:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version does not match")
    actual = payload.get("identity")
    if not isinstance(actual, dict):
        raise ValueError("checkpoint identity is missing")
    for field, expected_value in asdict(expected).items():
        if actual.get(field) != expected_value:
            raise ValueError(f"checkpoint identity mismatch: {field}")


def load_diffusion_checkpoint(
    model: ConditionalLatentDiffusion,
    checkpoint_path: str | Path,
    expected_identity: CheckpointIdentity,
) -> int:
    payload = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict):
        raise ValueError("diffusion checkpoint payload is invalid")
    _validate_identity(payload, expected_identity)
    _validate_model_schedule(model, expected_identity)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("diffusion checkpoint state dict is missing")
    normalized = {
        key.removeprefix("model."): value for key, value in state.items()
    }
    required = set(_adapter_state(model))
    actual = set(normalized)
    if actual != required:
        missing = sorted(required - actual)
        unexpected_state = sorted(actual - required)
        raise ValueError(
            "diffusion checkpoint required state mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected_state[:5]}"
        )
    unexpected = model.load_state_dict(normalized, strict=False).unexpected_keys
    if unexpected:
        raise ValueError(f"diffusion checkpoint contains unexpected state: {unexpected}")
    return int(payload.get("global_step", 0))
