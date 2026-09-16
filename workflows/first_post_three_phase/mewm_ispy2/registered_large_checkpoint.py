from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from .contracts import (
    ANCESTRAL_DDPM_SAMPLER,
    CONTINUOUS_CODEBOOK_MINMAX_CONTRACT,
    EPSILON_PREDICTION_TYPE,
    REGISTERED_LARGE_PAPER_ARCHITECTURE,
)
from .paper_checkpoint import (
    _checkpoint_path,
    _load_paper_state,
    _normalize_state_keys,
    _paper_state_dict,
    _validate_paper_state,
)
from .paper_contracts import (
    RegisteredLargePaperRuntimeContract,
    registered_large_paper_contract_payload,
    registered_large_paper_contract_sha256,
    registered_large_paper_runtime_contract,
)
from .vqgan import REGISTERED_VQGAN_NUMERIC_CONTRACT


REGISTERED_LARGE_CHECKPOINT_SCHEMA_VERSION = (
    "mewm_ispy2_diffusion_checkpoint_v6"
)
REGISTERED_LARGE_NORMALIZATION_SHA256 = (
    "f904d923c518dab98d42a0b17685bfaedb1ebca5d56c36b4e3a19038a1a28042"
)
REGISTERED_LARGE_PHYSICAL_BATCH_SIZE = 12
REGISTERED_LARGE_ACCUMULATE_GRAD_BATCHES = 1
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
        raise ValueError(
            "registered-large paper_contract must be canonical JSON data"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _validate_exact_structure(actual: Any, expected: Any, *, path: str) -> None:
    if type(actual) is not type(expected):
        raise TypeError(f"{path} must have exact type {type(expected).__name__}")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{path} keys do not match the registered contract")
        for key, value in expected.items():
            _validate_exact_structure(actual[key], value, path=f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{path} length does not match the registered contract")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _validate_exact_structure(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
    elif isinstance(actual, float) and not math.isfinite(actual):
        raise ValueError(f"{path} must be finite")


@dataclass(frozen=True)
class RegisteredLargeCheckpointIdentity:
    denoiser_architecture: str
    denoiser_widths: tuple[int, int, int, int]
    input_shape_zyx: tuple[int, int, int]
    latent_shape_czyx: tuple[int, int, int, int]
    vqgan_sha256: str
    vqgan_numeric_contract: str
    data_contract_sha256: str
    data_backend: str
    bundle_contract_sha256: str
    phase_manifest_sha256: str
    normalization_sha256: str
    denoiser_input_channels: int
    semantic_channels: int
    prediction_type: str
    timesteps: int
    sampler: str
    latent_contract: str
    ema_decay: float
    paper_contract: dict[str, Any]
    paper_contract_sha256: str
    initialization_method: str
    warm_start_source: None
    physical_batch_size: int = REGISTERED_LARGE_PHYSICAL_BATCH_SIZE
    accumulate_grad_batches: int = REGISTERED_LARGE_ACCUMULATE_GRAD_BATCHES

    def __post_init__(self) -> None:
        string_fields = (
            "denoiser_architecture",
            "vqgan_sha256",
            "vqgan_numeric_contract",
            "data_contract_sha256",
            "data_backend",
            "bundle_contract_sha256",
            "phase_manifest_sha256",
            "normalization_sha256",
            "prediction_type",
            "sampler",
            "latent_contract",
            "paper_contract_sha256",
            "initialization_method",
        )
        for name in string_fields:
            if type(getattr(self, name)) is not str:
                raise TypeError(f"{name} must be an exact string")
        tuple_fields = (
            ("denoiser_widths", 4),
            ("input_shape_zyx", 3),
            ("latent_shape_czyx", 4),
        )
        for name, length in tuple_fields:
            value = getattr(self, name)
            if type(value) is not tuple:
                raise TypeError(f"{name} must be an exact tuple")
            if len(value) != length or any(type(item) is not int for item in value):
                raise ValueError(f"{name} must contain {length} exact integers")
        for name in (
            "denoiser_input_channels",
            "semantic_channels",
            "timesteps",
            "physical_batch_size",
            "accumulate_grad_batches",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if type(self.ema_decay) is not float:
            raise TypeError("ema_decay must be an exact float")
        if type(self.paper_contract) is not dict:
            raise TypeError("paper_contract must be an exact dictionary")
        if self.warm_start_source is not None:
            raise ValueError("warm_start_source must be None for random initialization")

        sha_fields = (
            "vqgan_sha256",
            "data_contract_sha256",
            "bundle_contract_sha256",
            "phase_manifest_sha256",
            "normalization_sha256",
            "paper_contract_sha256",
        )
        for name in sha_fields:
            if _HEX_SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must contain 64 lowercase hex characters")
        locked = {
            "denoiser_architecture": REGISTERED_LARGE_PAPER_ARCHITECTURE,
            "denoiser_widths": (32, 64, 128, 256),
            "input_shape_zyx": (96, 256, 256),
            "latent_shape_czyx": (8, 24, 64, 64),
            "vqgan_numeric_contract": REGISTERED_VQGAN_NUMERIC_CONTRACT,
            "data_backend": "registered_t0",
            "normalization_sha256": REGISTERED_LARGE_NORMALIZATION_SHA256,
            "denoiser_input_channels": 49,
            "semantic_channels": 32,
            "prediction_type": EPSILON_PREDICTION_TYPE,
            "timesteps": 200,
            "sampler": ANCESTRAL_DDPM_SAMPLER,
            "latent_contract": CONTINUOUS_CODEBOOK_MINMAX_CONTRACT,
            "ema_decay": 0.995,
            "initialization_method": "random",
            "physical_batch_size": REGISTERED_LARGE_PHYSICAL_BATCH_SIZE,
            "accumulate_grad_batches": REGISTERED_LARGE_ACCUMULATE_GRAD_BATCHES,
        }
        for name, expected in locked.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"{name} does not match the locked registered-large identity"
                )

        canonical_contract = registered_large_paper_runtime_contract(
            vqgan_sha256=self.vqgan_sha256
        )
        canonical_payload = registered_large_paper_contract_payload(
            canonical_contract
        )
        contract = copy.deepcopy(self.paper_contract)
        _validate_exact_structure(
            contract,
            canonical_payload,
            path="paper_contract",
        )
        for key, expected in canonical_payload.items():
            if contract[key] != expected:
                raise ValueError(f"paper_contract mismatch: {key}")
        if _canonical_sha256(contract) != self.paper_contract_sha256:
            raise ValueError("paper_contract_sha256 does not match paper_contract")
        bound_fields = {
            "data_contract_sha256": "data_contract_sha256",
            "bundle_contract_sha256": "bundle_contract_sha256",
            "phase_manifest_sha256": "phase_manifest_sha256",
            "vqgan_numeric_contract": "vqgan_numeric_contract",
            "initialization_method": "initialization_method",
        }
        for identity_name, contract_name in bound_fields.items():
            if getattr(self, identity_name) != contract[contract_name]:
                raise ValueError(f"{identity_name} does not match paper_contract")
        object.__setattr__(self, "paper_contract", contract)


def _identity_payload(
    identity: RegisteredLargeCheckpointIdentity,
) -> dict[str, Any]:
    if type(identity) is not RegisteredLargeCheckpointIdentity:
        raise TypeError(
            "identity must be an exact RegisteredLargeCheckpointIdentity"
        )
    return {
        field.name: copy.deepcopy(getattr(identity, field.name))
        for field in fields(identity)
    }


def _parse_identity(value: Any) -> RegisteredLargeCheckpointIdentity:
    if type(value) is not dict:
        raise ValueError("registered-large checkpoint identity is missing")
    expected = {field.name for field in fields(RegisteredLargeCheckpointIdentity)}
    if set(value) != expected:
        raise ValueError("registered-large checkpoint identity fields do not match v6")
    try:
        return RegisteredLargeCheckpointIdentity(
            **{name: copy.deepcopy(value[name]) for name in expected}
        )
    except (TypeError, ValueError) as error:
        raise type(error)(
            f"registered-large checkpoint identity is invalid: {error}"
        ) from error


def _validate_identity_match(
    actual: RegisteredLargeCheckpointIdentity,
    expected: RegisteredLargeCheckpointIdentity,
) -> None:
    if type(expected) is not RegisteredLargeCheckpointIdentity:
        raise TypeError(
            "expected_identity must be an exact RegisteredLargeCheckpointIdentity"
        )
    actual_payload = _identity_payload(actual)
    expected_payload = _identity_payload(expected)
    for field in fields(RegisteredLargeCheckpointIdentity):
        if actual_payload[field.name] != expected_payload[field.name]:
            raise ValueError(
                f"registered-large checkpoint identity mismatch: {field.name}"
            )


def _validate_model_identity(
    model: torch.nn.Module,
    identity: RegisteredLargeCheckpointIdentity,
) -> None:
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("registered-large checkpoint model must expose its config")
    denoiser = getattr(model, "denoiser", None)
    checks = {
        "denoiser_architecture": getattr(config, "denoiser_architecture", None),
        "denoiser_widths": getattr(denoiser, "channels", None),
        "denoiser_input_channels": getattr(config, "denoiser_input_channels", None),
        "semantic_channels": getattr(config, "semantic_channels", None),
        "prediction_type": getattr(config, "prediction_type", None),
        "timesteps": getattr(config, "timesteps", None),
        "sampler": getattr(config, "sampler", None),
        "latent_contract": getattr(config, "latent_contract", None),
        "ema_decay": getattr(config, "ema_decay", None),
    }
    for name, model_value in checks.items():
        if getattr(identity, name) != model_value:
            raise ValueError(
                f"registered-large checkpoint identity does not match model: {name}"
            )
    runtime = getattr(config, "runtime", None)
    if type(runtime) is not RegisteredLargePaperRuntimeContract:
        raise TypeError("registered-large model runtime contract is invalid")
    if identity.paper_contract != registered_large_paper_contract_payload(runtime):
        raise ValueError(
            "registered-large checkpoint identity does not match model: paper_contract"
        )
    if identity.paper_contract_sha256 != registered_large_paper_contract_sha256(
        runtime
    ):
        raise ValueError(
            "registered-large checkpoint identity does not match model: "
            "paper_contract_sha256"
        )


def _validate_metadata(
    payload: Any,
    expected_identity: RegisteredLargeCheckpointIdentity | None = None,
) -> tuple[RegisteredLargeCheckpointIdentity, int, dict[str, Any]]:
    if type(payload) is not dict:
        raise ValueError("registered-large checkpoint payload must be an exact dictionary")
    schema = payload.get("schema_version")
    if type(schema) is not str:
        raise TypeError("registered-large checkpoint schema_version must be an exact string")
    if schema != REGISTERED_LARGE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("registered-large checkpoint schema version does not match v6")
    identity = _parse_identity(payload.get("identity"))
    if expected_identity is not None:
        _validate_identity_match(identity, expected_identity)
    global_step = payload.get("global_step")
    if type(global_step) is not int:
        raise TypeError("registered-large checkpoint global_step must be an exact integer")
    if global_step < 0:
        raise ValueError("registered-large checkpoint global_step must be nonnegative")
    state = payload.get("state_dict")
    if type(state) is not dict:
        raise ValueError("registered-large checkpoint state_dict is missing")
    return identity, global_step, state


def _load_payload(path: str | Path) -> dict[str, Any]:
    loaded = torch.load(
        _checkpoint_path(path),
        map_location="cpu",
        weights_only=True,
    )
    if type(loaded) is not dict:
        raise ValueError("registered-large checkpoint payload must be an exact dictionary")
    return loaded


def peek_registered_large_checkpoint_identity(
    path: str | Path,
) -> RegisteredLargeCheckpointIdentity:
    identity, _, _ = _validate_metadata(_load_payload(path))
    return identity


def build_registered_large_checkpoint(
    model: torch.nn.Module,
    identity: RegisteredLargeCheckpointIdentity,
    *,
    global_step: int,
) -> dict[str, Any]:
    if type(global_step) is not int:
        raise TypeError("global_step must be an exact integer")
    if global_step < 0:
        raise ValueError("global_step must be nonnegative")
    if type(identity) is not RegisteredLargeCheckpointIdentity:
        raise TypeError(
            "identity must be an exact RegisteredLargeCheckpointIdentity"
        )
    _validate_model_identity(model, identity)
    return {
        "schema_version": REGISTERED_LARGE_CHECKPOINT_SCHEMA_VERSION,
        "identity": _identity_payload(identity),
        "global_step": global_step,
        "state_dict": _paper_state_dict(model, cpu=True),
    }


def save_registered_large_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    identity: RegisteredLargeCheckpointIdentity,
    *,
    global_step: int,
) -> None:
    target = _checkpoint_path(path)
    payload = build_registered_large_checkpoint(
        model,
        identity,
        global_step=global_step,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_registered_large_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    expected_identity: RegisteredLargeCheckpointIdentity,
) -> int:
    if type(expected_identity) is not RegisteredLargeCheckpointIdentity:
        raise TypeError(
            "expected_identity must be an exact RegisteredLargeCheckpointIdentity"
        )
    payload = _load_payload(path)
    _, global_step, raw_state = _validate_metadata(
        payload,
        expected_identity,
    )
    _validate_model_identity(model, expected_identity)
    state = _normalize_state_keys(raw_state)
    _load_paper_state(model, state)
    return global_step


__all__ = [
    "REGISTERED_LARGE_CHECKPOINT_SCHEMA_VERSION",
    "REGISTERED_LARGE_NORMALIZATION_SHA256",
    "REGISTERED_LARGE_PHYSICAL_BATCH_SIZE",
    "REGISTERED_LARGE_ACCUMULATE_GRAD_BATCHES",
    "RegisteredLargeCheckpointIdentity",
    "build_registered_large_checkpoint",
    "load_registered_large_checkpoint",
    "peek_registered_large_checkpoint_identity",
    "save_registered_large_checkpoint",
]
