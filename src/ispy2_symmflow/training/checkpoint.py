"""Provenance-rich, resumable training checkpoint serialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
from torch import nn

from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
)
from ispy2_symmflow.utils.reproducibility import capture_rng_state, restore_rng_state


CHECKPOINT_FORMAT_VERSION = 1


class CheckpointCompatibilityError(ValueError):
    """Raised before loading state with incompatible data/model provenance."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _latent_signature(value: Mapping[str, Any] | None) -> dict[str, Any]:
    mapping = value or {}
    keys = (
        "mean",
        "std",
        "element_count_per_channel",
        "fit_split",
        "fit_patient_ids",
        "fit_visit_ids",
        "autoencoder_id",
        "latent_statistics_fingerprint",
        "source_split_hash",
        "source_ordered_manifest_fingerprint",
        "source_manifest_record_count",
        "source_preprocessing_signature",
        CACHED_PAIR_MANIFEST_FINGERPRINT,
        CACHED_PAIR_MANIFEST_RECORD_COUNT,
    )
    return {key: mapping.get(key) for key in keys}


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    ema: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    config: Mapping[str, Any],
    autoencoder_id: str | None,
    latent_statistics: Mapping[str, Any] | None,
    feature_schema: Mapping[str, Any] | None,
    split_hash: str,
    upstream_commits: Mapping[str, str],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write all state needed for an exact local resume."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "epoch": int(epoch),
        "step": int(step),
        "rng_state": capture_rng_state(),
        "config": dict(config),
        "autoencoder_id": autoencoder_id,
        "latent_statistics": dict(latent_statistics or {}),
        "feature_schema": dict(feature_schema or {}),
        "split_hash": str(split_hash),
        "upstream_commits": dict(upstream_commits),
        "extra": dict(extra or {}),
    }
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    ema: Any | None = None,
    expected_autoencoder_id: str | None = None,
    expected_feature_schema: Mapping[str, Any] | None = None,
    expected_split_hash: str | None = None,
    expected_latent_statistics: Mapping[str, Any] | None = None,
    expected_sigma_min: float | None = None,
    expected_training_signature: Mapping[str, Any] | None = None,
    required_extra_keys: tuple[str, ...] = (),
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Validate schema/provenance before restoring mutable training state."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint format {payload.get('format_version')!r}"
        )
    checks = (
        ("autoencoder_id", expected_autoencoder_id),
        ("split_hash", expected_split_hash),
    )
    for key, expected in checks:
        if expected is not None and str(payload.get(key)) != str(expected):
            raise CheckpointCompatibilityError(
                f"{key} mismatch: checkpoint={payload.get(key)!r}, expected={expected!r}"
            )
    if expected_feature_schema is not None and _canonical(payload.get("feature_schema")) != _canonical(
        expected_feature_schema
    ):
        raise CheckpointCompatibilityError("condition feature schema mismatch")
    if expected_latent_statistics is not None and _canonical(
        _latent_signature(payload.get("latent_statistics"))
    ) != _canonical(_latent_signature(expected_latent_statistics)):
        raise CheckpointCompatibilityError("latent normalization statistics mismatch")
    if expected_sigma_min is not None:
        checkpoint_sigma = float(payload.get("config", {}).get("flow", {}).get("sigma_min", 0.0))
        if checkpoint_sigma != float(expected_sigma_min):
            raise CheckpointCompatibilityError(
                f"sigma_min mismatch: checkpoint={checkpoint_sigma}, expected={expected_sigma_min}"
            )
    extra = payload.get("extra")
    if not isinstance(extra, Mapping):
        extra = {}
    if expected_training_signature is not None and _canonical(
        extra.get("training_signature")
    ) != _canonical(expected_training_signature):
        raise CheckpointCompatibilityError("training plan or ordered manifest signature mismatch")
    missing_extra = [key for key in required_extra_keys if key not in extra]
    if missing_extra:
        raise CheckpointCompatibilityError(
            f"checkpoint is missing required resume state: {missing_extra}"
        )
    required_states = (
        ("model", model),
        ("optimizer", optimizer),
        ("scheduler", scheduler),
        ("ema", ema),
    )
    for key, consumer in required_states:
        if consumer is not None and payload.get(key) is None:
            label = "EMA" if key == "ema" else key
            raise CheckpointCompatibilityError(f"checkpoint contains no {label} state")
    if restore_rng and payload.get("rng_state") is None:
        raise CheckpointCompatibilityError("checkpoint contains no RNG state")

    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if ema is not None:
        ema.load_state_dict(payload["ema"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload
