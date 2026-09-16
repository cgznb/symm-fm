"""Configuration loading with explicit, reproducible overrides."""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is missing or internally inconsistent."""


def resolve_time_pairs(data: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return the ordered longitudinal intervals declared by a data config.

    ``time_pair`` remains supported for existing experiments.  A multi-interval
    experiment uses ``time_pairs`` and must set the inherited ``time_pair`` to
    null so the checkpoint contract is unambiguous.
    """

    single = data.get("time_pair")
    multiple = data.get("time_pairs")
    if single is not None and multiple is not None:
        raise ConfigError("data must define only one of time_pair and time_pairs")
    raw_pairs: Any
    if multiple is not None:
        raw_pairs = multiple
    elif single is not None:
        raw_pairs = [single]
    else:
        raw_pairs = [["T0", "T1"]]
    if (
        not isinstance(raw_pairs, (list, tuple))
        or not raw_pairs
        or any(
            not isinstance(pair, (list, tuple)) or len(pair) != 2
            for pair in raw_pairs
        )
    ):
        raise ConfigError("data.time_pairs must contain one or more [earlier, later] pairs")

    normalized: list[tuple[str, str]] = []
    for raw_pair in raw_pairs:
        pair = tuple(str(stage).strip().upper() for stage in raw_pair)
        if any(not stage for stage in pair):
            raise ConfigError("visit stages must be non-empty")
        try:
            indices = [int(stage.removeprefix("T")) for stage in pair]
        except ValueError as exc:
            raise ConfigError("visit stages must use T<number> labels") from exc
        if any(stage != f"T{index}" for stage, index in zip(pair, indices, strict=True)):
            raise ConfigError("visit stages must use canonical T<number> labels")
        if indices[0] >= indices[1]:
            raise ConfigError("each data time pair must keep earlier stage before later stage")
        normalized.append(pair)
    if len(normalized) != len(set(normalized)):
        raise ConfigError("data time pairs must not contain duplicates")
    return tuple(normalized)


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ConfigError(f"override must be KEY=VALUE, received {raw!r}")
    dotted, raw_value = raw.split("=", 1)
    keys = [part for part in dotted.split(".") if part]
    if not keys:
        raise ConfigError(f"override has an empty key: {raw!r}")
    try:
        value = _release_yaml(raw_value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid override value in {raw!r}: {exc}") from exc
    return keys, value


def _apply_override(config: dict[str, Any], raw: str) -> None:
    keys, value = _parse_override(raw)
    cursor = config
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, dict):
            raise ConfigError(f"cannot assign below non-mapping key {key!r}")
        cursor = child
    cursor[keys[-1]] = value


def load_config(path: str | Path, overrides: tuple[str, ...] = (), *, resolve_assets: bool = True) -> dict[str, Any]:
    """Load YAML, resolving an optional ``extends`` file relative to the file."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        parsed = _release_yaml(handle, resolve_assets=resolve_assets) or {}
    if not isinstance(parsed, dict):
        raise ConfigError(f"configuration root must be a mapping: {config_path}")
    extends = parsed.pop("extends", None)
    if extends:
        parent = load_config(config_path.parent / str(extends), resolve_assets=resolve_assets)
        parsed = _merge(parent, parsed)
    for raw in overrides:
        _apply_override(parsed, raw)
    parsed["_config_path"] = str(config_path)
    validate_config(parsed)
    return parsed


def validate_config(config: Mapping[str, Any]) -> None:
    flow = config.get("flow", {})
    sigma = float(flow.get("sigma_min", 0.0))
    if not math.isfinite(sigma) or not 0.0 <= sigma < 1.0:
        raise ConfigError("flow.sigma_min must satisfy 0 <= sigma_min < 1")
    data = config.get("data", {})
    if not isinstance(data, Mapping):
        raise ConfigError("data must be a mapping")
    time_pairs = resolve_time_pairs(data)
    pair_sampling = str(
        data.get(
            "pair_sampling",
            "patient_balanced" if len(time_pairs) > 1 else "uniform_pairs",
        )
    )
    if pair_sampling not in {"uniform_pairs", "patient_balanced"}:
        raise ConfigError(
            "data.pair_sampling must be 'uniform_pairs' or 'patient_balanced'"
        )
    latent_normalization = data.get("latent_normalization")
    if latent_normalization is not None and str(latent_normalization) not in {
        "continuous_codebook_minmax_v1",
        "train_channel_zscore_v1",
    }:
        raise ConfigError(
            "data.latent_normalization must be "
            "'continuous_codebook_minmax_v1' or 'train_channel_zscore_v1'"
        )
    channels = data.get("phase_channels", [])
    if not channels:
        raise ConfigError("data.phase_channels cannot be empty")
    registration = str(data.get("registration", "none")).lower()
    if registration not in {"none", "external_registered_t0"}:
        raise ConfigError(
            "data.registration must be 'none' or the audited 'external_registered_t0' mode"
        )
    if registration == "external_registered_t0":
        if data.get("registration_assisted") is not True:
            raise ConfigError(
                "external_registered_t0 requires data.registration_assisted=true"
            )
        if data.get("backward_task_label") != "registration-assisted reconstruction":
            raise ConfigError(
                "external_registered_t0 requires the exact registration-assisted backward label"
            )
        if data.get("crop_policy") != "single_T0_mask_bbox_center_reused_for_all_visits":
            raise ConfigError(
                "external_registered_t0 requires the audited single-T0-mask crop policy"
            )
    weights = (
        float(flow.get("loss_weight_x", 1.0)),
        float(flow.get("loss_weight_y", 1.0)),
    )
    if any(not math.isfinite(value) or value < 0 for value in weights) or not any(
        value > 0 for value in weights
    ):
        raise ConfigError("flow branch loss weights must be finite, non-negative, and not both zero")
    positive_integer_fields = (
        ("flow.max_steps", flow.get("max_steps", 1)),
        ("flow.gradient_accumulation", flow.get("gradient_accumulation", 1)),
        ("flow.validation_interval_steps", flow.get("validation_interval_steps", 1)),
        ("flow.validation_repeats", flow.get("validation_repeats", 1)),
        ("flow.log_interval_steps", flow.get("log_interval_steps", 10)),
        ("autoencoder.max_epochs", config.get("autoencoder", {}).get("max_epochs", 1)),
        (
            "autoencoder.validation_interval_epochs",
            config.get("autoencoder", {}).get("validation_interval_epochs", 1),
        ),
        ("sampling.steps", config.get("sampling", {}).get("steps", 1)),
        ("sampling.num_samples", config.get("sampling", {}).get("num_samples", 1)),
    )
    for name, raw_value in positive_integer_fields:
        if isinstance(raw_value, bool) or int(raw_value) != raw_value or int(raw_value) < 1:
            raise ConfigError(f"{name} must be a positive integer")
    endpoint_pairs = int(flow.get("endpoint_validation_pairs", 0))
    if endpoint_pairs < 0:
        raise ConfigError("flow.endpoint_validation_pairs must be non-negative")
    if endpoint_pairs:
        if sigma != 0.0:
            raise ConfigError(
                "flow endpoint validation currently requires flow.sigma_min=0"
            )
        for name, raw_value in (
            ("flow.endpoint_validation_samples", flow.get("endpoint_validation_samples", 1)),
            ("flow.endpoint_validation_steps", flow.get("endpoint_validation_steps", 1)),
        ):
            if (
                isinstance(raw_value, bool)
                or int(raw_value) != raw_value
                or int(raw_value) < 1
            ):
                raise ConfigError(f"{name} must be a positive integer")
        if str(flow.get("endpoint_validation_solver", "heun")) not in {
            "euler",
            "heun",
        }:
            raise ConfigError(
                "flow.endpoint_validation_solver must be 'euler' or 'heun'"
            )
    checkpoint_metric = str(flow.get("checkpoint_metric", "loss"))
    supported_checkpoint_metrics = {
        "loss",
        "endpoint_candidate_mse",
        "endpoint_candidate_mae",
        "endpoint_mean_mse",
        "endpoint_mean_mae",
    }
    if checkpoint_metric not in supported_checkpoint_metrics:
        raise ConfigError(
            "flow.checkpoint_metric must be a validation loss or generated endpoint metric"
        )
    if checkpoint_metric != "loss" and not endpoint_pairs:
        raise ConfigError(
            "a non-loss flow.checkpoint_metric requires endpoint validation"
        )


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Return stable JSON text suitable for hashing or checkpoint metadata."""

    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
