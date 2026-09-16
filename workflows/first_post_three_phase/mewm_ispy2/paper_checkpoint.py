from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from .contracts import (
    ANCESTRAL_DDPM_SAMPLER,
    CONTINUOUS_CODEBOOK_MINMAX_CONTRACT,
    EPSILON_PREDICTION_TYPE,
)
from .manifest import ACCEPTED_BACKENDS
from .paper_contracts import (
    UNKNOWN_AGE_POLICY,
    default_paper_runtime_contract,
    paper_contract_payload,
    paper_contract_sha256,
)


PAPER_CHECKPOINT_SCHEMA_VERSION = "mewm_ispy2_diffusion_checkpoint_v5"
_PAPER_DENOISER_ARCHITECTURE = "mewm_paper_faithful_v1"
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("paper_contract must be canonical JSON data") from error
    return hashlib.sha256(encoded).hexdigest()


def _validate_exact_structure(actual: Any, expected: Any, *, path: str) -> None:
    if type(actual) is not type(expected):
        raise TypeError(f"{path} must have exact type {type(expected).__name__}")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{path} keys do not match the canonical paper contract")
        for key, value in expected.items():
            _validate_exact_structure(actual[key], value, path=f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{path} length does not match the canonical paper contract")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _validate_exact_structure(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
    elif isinstance(actual, float) and not math.isfinite(actual):
        raise ValueError(f"{path} must be finite")


def _validate_paper_contract_payload(
    payload: dict[str, Any], expected_sha256: str
) -> None:
    if type(payload) is not dict:
        raise TypeError("paper_contract must be an exact dictionary")
    canonical = paper_contract_payload(default_paper_runtime_contract())
    _validate_exact_structure(payload, canonical, path="paper_contract")
    epsilon_objective = payload["epsilon_objective"]
    if epsilon_objective not in {"l1", "l2"}:
        raise ValueError("paper_contract.epsilon_objective must be l1 or l2")
    expected = copy.deepcopy(canonical)
    expected["epsilon_objective"] = epsilon_objective
    for key, expected_value in expected.items():
        if payload[key] != expected_value:
            raise ValueError(f"paper_contract mismatch: {key}")
    if payload["unknown_age_policy"] != UNKNOWN_AGE_POLICY:
        raise ValueError("paper_contract mismatch: unknown_age_policy")
    if _canonical_sha256(payload) != expected_sha256:
        raise ValueError("paper_contract_sha256 does not match paper_contract")


@dataclass(frozen=True)
class PaperCheckpointIdentity:
    denoiser_architecture: str
    vqgan_sha256: str
    data_contract_sha256: str
    data_backend: str
    denoiser_input_channels: int
    semantic_channels: int
    prediction_type: str
    timesteps: int
    sampler: str
    latent_contract: str
    ema_decay: float
    paper_contract: dict[str, Any]
    paper_contract_sha256: str

    def __post_init__(self) -> None:
        string_fields = (
            "denoiser_architecture",
            "vqgan_sha256",
            "data_contract_sha256",
            "data_backend",
            "prediction_type",
            "sampler",
            "latent_contract",
            "paper_contract_sha256",
        )
        for name in string_fields:
            if type(getattr(self, name)) is not str:
                raise TypeError(f"{name} must be an exact string")
        for name in ("denoiser_input_channels", "semantic_channels", "timesteps"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if type(self.ema_decay) is not float:
            raise TypeError("ema_decay must be an exact float")
        if type(self.paper_contract) is not dict:
            raise TypeError("paper_contract must be an exact dictionary")

        for name in ("vqgan_sha256", "data_contract_sha256", "paper_contract_sha256"):
            if _HEX_SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must contain 64 lowercase hex characters")
        if self.data_backend not in ACCEPTED_BACKENDS:
            raise ValueError("data_backend is invalid")
        locked_values = {
            "denoiser_architecture": _PAPER_DENOISER_ARCHITECTURE,
            "denoiser_input_channels": 49,
            "semantic_channels": 32,
            "prediction_type": EPSILON_PREDICTION_TYPE,
            "timesteps": 200,
            "sampler": ANCESTRAL_DDPM_SAMPLER,
            "latent_contract": CONTINUOUS_CODEBOOK_MINMAX_CONTRACT,
            "ema_decay": 0.995,
        }
        for name, expected in locked_values.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} does not match the locked paper identity")

        contract = copy.deepcopy(self.paper_contract)
        _validate_paper_contract_payload(contract, self.paper_contract_sha256)
        if contract["vqgan_sha256"] != self.vqgan_sha256:
            raise ValueError("vqgan_sha256 does not match paper_contract")
        if contract["data_contract_sha256"] != self.data_contract_sha256:
            raise ValueError("data_contract_sha256 does not match paper_contract")
        object.__setattr__(self, "paper_contract", contract)


def _paper_identity_payload(identity: PaperCheckpointIdentity) -> dict[str, Any]:
    if type(identity) is not PaperCheckpointIdentity:
        raise TypeError("identity must be an exact PaperCheckpointIdentity")
    return {
        field.name: copy.deepcopy(getattr(identity, field.name))
        for field in fields(identity)
    }


def _parse_paper_identity(value: Any) -> PaperCheckpointIdentity:
    if type(value) is not dict:
        raise ValueError("paper checkpoint identity is missing")
    expected_keys = {field.name for field in fields(PaperCheckpointIdentity)}
    if set(value) != expected_keys:
        raise ValueError("paper checkpoint identity fields do not match v5")
    try:
        return PaperCheckpointIdentity(
            **{key: copy.deepcopy(value[key]) for key in expected_keys}
        )
    except (TypeError, ValueError) as error:
        raise type(error)(f"paper checkpoint identity is invalid: {error}") from error


def _validate_identity_match(
    actual: PaperCheckpointIdentity, expected: PaperCheckpointIdentity
) -> None:
    if type(expected) is not PaperCheckpointIdentity:
        raise TypeError("expected_identity must be an exact PaperCheckpointIdentity")
    actual_payload = _paper_identity_payload(actual)
    expected_payload = _paper_identity_payload(expected)
    for field in fields(PaperCheckpointIdentity):
        if actual_payload[field.name] != expected_payload[field.name]:
            raise ValueError(f"paper checkpoint identity mismatch: {field.name}")


def _validate_model_identity(model: torch.nn.Module, identity: PaperCheckpointIdentity) -> None:
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("paper checkpoint model must expose its diffusion config")
    expected = (
        config.denoiser_architecture,
        config.denoiser_input_channels,
        config.semantic_channels,
        config.prediction_type,
        config.timesteps,
        config.sampler,
        config.latent_contract,
        config.ema_decay,
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
    names = (
        "denoiser_architecture",
        "denoiser_input_channels",
        "semantic_channels",
        "prediction_type",
        "timesteps",
        "sampler",
        "latent_contract",
        "ema_decay",
    )
    for name, actual_value, expected_value in zip(names, actual, expected, strict=True):
        if actual_value != expected_value:
            raise ValueError(f"paper checkpoint identity does not match model: {name}")
    runtime_payload = paper_contract_payload(config.runtime)
    if identity.paper_contract != runtime_payload:
        raise ValueError("paper checkpoint identity does not match model: paper_contract")
    if identity.paper_contract_sha256 != paper_contract_sha256(config.runtime):
        raise ValueError("paper checkpoint identity does not match model: paper_contract_sha256")


def _conditioner_state_keys(conditioner: torch.nn.Module) -> tuple[str, ...]:
    exporter = getattr(conditioner, "trainable_state_dict", None)
    if not callable(exporter):
        raise TypeError("paper conditioner must expose trainable_state_dict()")
    exported = exporter()
    if type(exported) is not dict:
        raise TypeError("conditioner trainable_state_dict must return an exact dictionary")
    expected = tuple(
        name
        for name, parameter in conditioner.named_parameters()
        if parameter.requires_grad
    )
    if tuple(exported) != expected:
        raise ValueError("conditioner trainable_state_dict keys are not exact")
    for name, value in exported.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"conditioner state tensor is invalid: {name}")
    return expected


def _paper_state_references(
    model: torch.nn.Module, *, prefix: str = ""
) -> dict[str, torch.Tensor]:
    modules = {
        "denoiser.": getattr(model, "denoiser", None),
        "ema_denoiser.": getattr(model, "ema_denoiser", None),
    }
    if any(not isinstance(module, torch.nn.Module) for module in modules.values()):
        raise TypeError("paper checkpoint model is missing a denoiser")
    references: dict[str, torch.Tensor] = {}
    for state_prefix, module in modules.items():
        assert isinstance(module, torch.nn.Module)
        for name, value in module.state_dict(keep_vars=True).items():
            references[f"{prefix}{state_prefix}{name}"] = value

    conditioner = getattr(model, "conditioner", None)
    if not isinstance(conditioner, torch.nn.Module):
        raise TypeError("paper checkpoint model is missing its conditioner")
    conditioner_keys = _conditioner_state_keys(conditioner)
    conditioner_state = conditioner.state_dict(keep_vars=True)
    if not set(conditioner_keys).issubset(conditioner_state):
        raise ValueError("conditioner trainable state is not registered")
    for name in conditioner_keys:
        references[f"{prefix}conditioner.{name}"] = conditioner_state[name]
    return references


def _validate_tensor_finite(value: torch.Tensor, *, name: str) -> None:
    if value.device.type == "meta":
        raise ValueError(f"paper checkpoint tensor device is invalid: {name}")
    if (value.is_floating_point() or value.is_complex()) and not bool(
        torch.isfinite(value).all()
    ):
        raise FloatingPointError(f"paper checkpoint tensor is nonfinite: {name}")


def _paper_state_dict(
    model: torch.nn.Module,
    *,
    prefix: str = "",
    cpu: bool = False,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, value in _paper_state_references(model, prefix=prefix).items():
        _validate_tensor_finite(value, name=name)
        clone = value.detach().clone()
        state[name] = clone.cpu() if cpu else clone
    return state


def _normalize_state_keys(state: dict[str, Any]) -> dict[str, Any]:
    keys = tuple(state)
    if keys and all(key.startswith("model.") for key in keys):
        return {key.removeprefix("model."): value for key, value in state.items()}
    return dict(state)


def _validate_paper_state(
    model: torch.nn.Module,
    state: Any,
    *,
    prefix: str = "",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if type(state) is not dict:
        raise ValueError("paper checkpoint state_dict is missing")
    expected = _paper_state_references(model, prefix=prefix)
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(
            "paper checkpoint required state mismatch: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    validated: dict[str, torch.Tensor] = {}
    for name, target in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"paper checkpoint state is not a tensor: {name}")
        if value.shape != target.shape:
            raise ValueError(f"paper checkpoint tensor shape mismatch: {name}")
        if value.dtype != target.dtype:
            raise TypeError(f"paper checkpoint tensor dtype mismatch: {name}")
        _validate_tensor_finite(value, name=name)
        if target.device.type == "meta":
            raise ValueError(f"paper model tensor device is invalid: {name}")
        validated[name] = value
    return expected, validated


@torch.no_grad()
def _load_paper_state(
    model: torch.nn.Module,
    state: Any,
    *,
    prefix: str = "",
) -> None:
    expected, validated = _validate_paper_state(model, state, prefix=prefix)
    for name, target in expected.items():
        target.copy_(validated[name].to(device=target.device))


def _validate_checkpoint_metadata(
    payload: Any,
    expected_identity: PaperCheckpointIdentity | None = None,
) -> tuple[PaperCheckpointIdentity, int, dict[str, Any]]:
    if type(payload) is not dict:
        raise ValueError("paper checkpoint payload must be an exact dictionary")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not str:
        raise TypeError("paper checkpoint schema_version must be an exact string")
    if schema_version != PAPER_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("paper checkpoint schema version does not match v5")
    actual_identity = _parse_paper_identity(payload.get("identity"))
    if expected_identity is not None:
        _validate_identity_match(actual_identity, expected_identity)
    global_step = payload.get("global_step")
    if type(global_step) is not int:
        raise TypeError("paper checkpoint global_step must be an exact integer")
    if global_step < 0:
        raise ValueError("paper checkpoint global_step must be nonnegative")
    state = payload.get("state_dict")
    if type(state) is not dict:
        raise ValueError("paper checkpoint state_dict is missing")
    return actual_identity, global_step, state


def _checkpoint_path(path: str | Path) -> Path:
    if type(path) is str:
        if not path:
            raise ValueError("paper checkpoint path must be nonempty")
        return Path(path)
    if not isinstance(path, Path):
        raise TypeError("paper checkpoint path must be an exact string or Path")
    if not str(path):
        raise ValueError("paper checkpoint path must be nonempty")
    return path


def _load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    loaded = torch.load(
        _checkpoint_path(path),
        map_location="cpu",
        weights_only=True,
    )
    if type(loaded) is not dict:
        raise ValueError("paper checkpoint payload must be an exact dictionary")
    return loaded


def peek_paper_checkpoint_identity(path: str | Path) -> PaperCheckpointIdentity:
    identity, _, _ = _validate_checkpoint_metadata(_load_checkpoint_payload(path))
    return identity


def build_paper_checkpoint(
    model: torch.nn.Module,
    identity: PaperCheckpointIdentity,
    *,
    global_step: int,
) -> dict[str, Any]:
    if type(global_step) is not int:
        raise TypeError("global_step must be an exact integer")
    if global_step < 0:
        raise ValueError("global_step must be nonnegative")
    if type(identity) is not PaperCheckpointIdentity:
        raise TypeError("identity must be an exact PaperCheckpointIdentity")
    _validate_model_identity(model, identity)
    return {
        "schema_version": PAPER_CHECKPOINT_SCHEMA_VERSION,
        "identity": _paper_identity_payload(identity),
        "global_step": global_step,
        "state_dict": _paper_state_dict(model, cpu=True),
    }


def load_paper_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    expected_identity: PaperCheckpointIdentity,
) -> int:
    if type(expected_identity) is not PaperCheckpointIdentity:
        raise TypeError("expected_identity must be an exact PaperCheckpointIdentity")
    payload = _load_checkpoint_payload(path)
    _, global_step, raw_state = _validate_checkpoint_metadata(
        payload, expected_identity
    )
    _validate_model_identity(model, expected_identity)
    state = _normalize_state_keys(raw_state)
    _load_paper_state(model, state)
    return global_step
