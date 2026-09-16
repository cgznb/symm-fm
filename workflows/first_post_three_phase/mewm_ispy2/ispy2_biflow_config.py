from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .ispy2_biflow_backbone import ISPY2_BIFLOW_PRESET, get_ispy2_biflow_preset



ISPY2_BIFLOW_CONFIG_SCHEMA_LEGACY = "mewm_ispy2_dce0_biflow_rectified_flow_config_v1"
ISPY2_BIFLOW_CONFIG_SCHEMA = "mewm_ispy2_dce0_biflow_rectified_flow_config_v2"
ISPY2_BIFLOW_STABLE_CONFIG_SCHEMA = (
    "mewm_ispy2_dce0_biflow_rectified_flow_stable_config_v3"
)
ISPY2_BIFLOW_BASE_CONFIG_SCHEMA_LEGACY = "mewm_ispy2_dce0_biflow_data_text_config_v1"
ISPY2_BIFLOW_BASE_CONFIG_SCHEMA = "mewm_ispy2_dce0_biflow_data_text_config_v2"


@dataclass(frozen=True)
class ISPY2BiFlowDataConfig:
    backend: str
    bundle_json: Path
    bundle_contract_sha256: str
    phase_manifest_csv: Path
    roi_cache_dir: Path
    vqgan_config: Path
    vqgan_checkpoint: Path
    vqgan_sha256: str
    continuous_root: Path
    source_modalities: tuple[str, ...]
    output_shape_zyx: tuple[int, ...]
    pair_mode: str
    codebook_min: float
    codebook_max: float


@dataclass(frozen=True)
class ISPY2BiFlowTextConfig:
    model_id: str
    revision: str
    max_length: int
    load_in_4bit: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]


@dataclass(frozen=True)
class ISPY2BiFlowTensorConfig:
    latent_normalization: str
    context_dim: int
    context_tokens: int
    source_mri_shape_czyx: tuple[int, ...]
    latent_channels: int
    latent_shape_czyx: tuple[int, ...]


@dataclass(frozen=True)
class ISPY2BiFlowBaseConfig:
    path: Path
    schema_version: str
    data: ISPY2BiFlowDataConfig
    text: ISPY2BiFlowTextConfig
    model: ISPY2BiFlowTensorConfig
    raw: dict[str, Any]


@dataclass(frozen=True)
class ISPY2BiFlowTrainingConfig:
    batch_size: int
    accumulate_grad_batches: int
    max_epochs: int
    precision: str
    num_workers: int
    dynamics_learning_rate: float
    backbone_learning_rate: float
    controlnet_learning_rate: float
    conditioner_learning_rate: float
    text_lora_learning_rate: float
    weight_decay: float
    gradient_clip_val: float
    evaluation_solver_steps: int
    early_stopping_patience: int
    optimizer_layout: str
    min_learning_rate: float
    warmup_fraction: float
    divergence_reference_metric: float | None
    divergence_multiplier: float
    divergence_patience: int


@dataclass(frozen=True)
class ISPY2BiFlowRuntimeConfig:
    output_root: Path
    seed: int
    warm_start_checkpoint: Path | None
    warm_start_sha256: str | None


@dataclass(frozen=True)
class ISPY2BiFlowConfig:
    path: Path
    schema_version: str
    base: ISPY2BiFlowBaseConfig
    preset: str
    training: ISPY2BiFlowTrainingConfig
    runtime: ISPY2BiFlowRuntimeConfig
    raw: dict[str, Any]
    sha256: str

    def conditioning_data_payload(self) -> dict[str, Any]:
        model = self.base.model
        return {
            "base_schema_version": self.base.schema_version,
            "data": self.base.raw["data"],
            "text": self.base.raw["text"],
            "tensor_contract": {
                "latent_normalization": model.latent_normalization,
                "context_dim": model.context_dim,
                "context_tokens": model.context_tokens,
                "source_mri_shape_czyx": list(model.source_mri_shape_czyx),
                "latent_channels": model.latent_channels,
                "latent_shape_czyx": list(model.latent_shape_czyx),
            },
        }

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "conditioning_data": self.conditioning_data_payload(),
            "preset": get_ispy2_biflow_preset(self.preset).payload(),
            "spatial_input": "flow_state_only",
            "image_condition_path": "controlnet_only",
            "prediction_target": "full_future_state",
            "latent_channels": self.base.model.latent_channels,
            "context_dim": self.base.model.context_dim,
        }

    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"I-SPY2 BiFlowNet {name} must be positive")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"I-SPY2 BiFlowNet {name} must be nonnegative")
    return value


def _finite_float(
    value: Any, name: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"I-SPY2 BiFlowNet {name} must be numeric")
    result = float(value)
    if (
        not math.isfinite(result)
        or positive
        and result <= 0.0
        or nonnegative
        and result < 0.0
    ):
        raise ValueError(f"I-SPY2 BiFlowNet {name} is invalid")
    return result


def _mapping(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"I-SPY2 BiFlowNet {name} fields are invalid")
    return value


def _path(value: Any, name: str) -> Path:
    if type(value) is not str or not value.strip():
        raise ValueError(f"I-SPY2 BiFlowNet {name} path is invalid")
    return Path(value).expanduser().resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"I-SPY2 BiFlowNet {name} SHA256 is invalid")
    return value


def _int_tuple(value: Any, name: str, length: int) -> tuple[int, ...]:
    if (
        type(value) is not list
        or len(value) != length
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise ValueError(f"I-SPY2 BiFlowNet {name} is invalid")
    return tuple(value)


def _load_biflow_base_config(path: Path) -> ISPY2BiFlowBaseConfig:
    try:
        raw = _release_yaml(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ValueError("I-SPY2 BiFlowNet base config is unreadable") from None
    root = _mapping(
        raw, "base root", {"schema_version", "data", "text", "model"}
    )
    if root["schema_version"] not in (
        ISPY2_BIFLOW_BASE_CONFIG_SCHEMA_LEGACY, ISPY2_BIFLOW_BASE_CONFIG_SCHEMA
    ):
        raise ValueError("I-SPY2 BiFlowNet base config schema is unsupported")

    data_raw = _mapping(
        root["data"],
        "base data",
        {
            "backend",
            "bundle_json",
            "bundle_contract_sha256",
            "phase_manifest_csv",
            "roi_cache_dir",
            "vqgan_config",
            "vqgan_checkpoint",
            "vqgan_sha256",
            "continuous_root",
            "source_modalities",
            "output_shape_zyx",
            "pair_mode",
            "codebook_min",
            "codebook_max",
        },
    )
    if data_raw["backend"] != "registered_t0":
        raise ValueError("I-SPY2 BiFlowNet data backend is unsupported")
    source_modalities = tuple(data_raw["source_modalities"])
    output_shape = _int_tuple(data_raw["output_shape_zyx"], "output shape", 3)
    if (
        source_modalities != ("dce0", "ser")
        or output_shape != (96, 256, 256)
        or data_raw["pair_mode"] != "all_connected_later_endpoints"
    ):
        raise ValueError("I-SPY2 BiFlowNet data tensor contract changed")
    data = ISPY2BiFlowDataConfig(
        backend="registered_t0",
        bundle_json=_path(data_raw["bundle_json"], "bundle_json"),
        bundle_contract_sha256=_sha256(
            data_raw["bundle_contract_sha256"], "bundle contract"
        ),
        phase_manifest_csv=_path(data_raw["phase_manifest_csv"], "phase manifest"),
        roi_cache_dir=_path(data_raw["roi_cache_dir"], "ROI cache"),
        vqgan_config=_path(data_raw["vqgan_config"], "VQGAN config"),
        vqgan_checkpoint=_path(data_raw["vqgan_checkpoint"], "VQGAN checkpoint"),
        vqgan_sha256=_sha256(data_raw["vqgan_sha256"], "VQGAN"),
        continuous_root=_path(data_raw["continuous_root"], "continuous latent root"),
        source_modalities=source_modalities,
        output_shape_zyx=output_shape,
        pair_mode=str(data_raw["pair_mode"]),
        codebook_min=_finite_float(data_raw["codebook_min"], "codebook_min"),
        codebook_max=_finite_float(data_raw["codebook_max"], "codebook_max"),
    )
    required_files = (
        data.bundle_json,
        data.phase_manifest_csv,
        data.vqgan_config,
        data.vqgan_checkpoint,
    )
    if any(not value.is_file() for value in required_files) or not data.roi_cache_dir.is_dir():
        raise FileNotFoundError("I-SPY2 BiFlowNet data assets are missing")

    text_raw = _mapping(
        root["text"],
        "base text",
        {
            "model_id",
            "revision",
            "max_length",
            "load_in_4bit",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_targets",
        },
    )
    text = ISPY2BiFlowTextConfig(
        model_id=str(text_raw["model_id"]),
        revision=str(text_raw["revision"]),
        max_length=_positive_int(text_raw["max_length"], "text max_length"),
        load_in_4bit=text_raw["load_in_4bit"] is True,
        lora_rank=_positive_int(text_raw["lora_rank"], "LoRA rank"),
        lora_alpha=_positive_int(text_raw["lora_alpha"], "LoRA alpha"),
        lora_dropout=_finite_float(text_raw["lora_dropout"], "LoRA dropout"),
        lora_targets=tuple(text_raw["lora_targets"]),
    )
    if (
        text.model_id != "google/medgemma-4b-it"
        or len(text.revision) != 40
        or not text.load_in_4bit
        or text.lora_targets != ("q_proj", "v_proj")
        or not 0.0 <= text.lora_dropout < 1.0
    ):
        raise ValueError("I-SPY2 BiFlowNet text contract changed")

    model_raw = _mapping(
        root["model"],
        "base model",
        {
            "latent_normalization",
            "context_dim",
            "context_tokens",
            "source_mri_shape_czyx",
            "latent_channels",
            "latent_shape_czyx",
        },
    )
    model = ISPY2BiFlowTensorConfig(
        latent_normalization=str(model_raw["latent_normalization"]),
        context_dim=_positive_int(model_raw["context_dim"], "context_dim"),
        context_tokens=_positive_int(model_raw["context_tokens"], "context_tokens"),
        source_mri_shape_czyx=_int_tuple(
            model_raw["source_mri_shape_czyx"], "source MRI shape", 4
        ),
        latent_channels=_positive_int(
            model_raw["latent_channels"], "latent_channels"
        ),
        latent_shape_czyx=_int_tuple(
            model_raw["latent_shape_czyx"], "latent shape", 4
        ),
    )
    if (
        model.latent_normalization not in (
            "continuous_codebook_minmax_v1",
            "continuous_train_unique_visit_channel_zscore_v1",
        )
        or model.context_dim != 768
        or model.context_tokens != 4
        or model.source_mri_shape_czyx != (2, 96, 256, 256)
        or model.latent_channels != 8
        or model.latent_shape_czyx != (8, 24, 64, 64)
    ):
        raise ValueError("I-SPY2 BiFlowNet tensor contract changed")
    return ISPY2BiFlowBaseConfig(
        path=path,
        schema_version=str(root["schema_version"]),
        data=data,
        text=text,
        model=model,
        raw=root,
    )


def load_ispy2_biflow_config(path: str | Path) -> ISPY2BiFlowConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"I-SPY2 BiFlowNet config is missing: {source}")
    try:
        root = _release_yaml(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ValueError("I-SPY2 BiFlowNet config is unreadable") from None
    root = _mapping(
        root,
        "root",
        {"schema_version", "base_config", "model", "training", "runtime"},
    )
    schema_version = str(root["schema_version"])
    if schema_version not in (
        ISPY2_BIFLOW_CONFIG_SCHEMA_LEGACY,
        ISPY2_BIFLOW_CONFIG_SCHEMA,
        ISPY2_BIFLOW_STABLE_CONFIG_SCHEMA,
    ):
        raise ValueError("I-SPY2 BiFlowNet config schema is unsupported")
    base_path = _path(root["base_config"], "base_config")
    base = _load_biflow_base_config(base_path)
    raw_model = _mapping(root["model"], "model", {"preset"})
    preset = str(raw_model["preset"])
    get_ispy2_biflow_preset(preset)

    legacy_training_fields = {
            "batch_size",
            "accumulate_grad_batches",
            "max_epochs",
            "precision",
            "num_workers",
            "dynamics_learning_rate",
            "text_lora_learning_rate",
            "weight_decay",
            "gradient_clip_val",
            "evaluation_solver_steps",
            "early_stopping_patience",
    }
    stable_training_fields = (
        legacy_training_fields
        - {"dynamics_learning_rate"}
        | {
            "backbone_learning_rate",
            "controlnet_learning_rate",
            "conditioner_learning_rate",
            "min_learning_rate",
            "warmup_fraction",
            "divergence_reference_metric",
            "divergence_multiplier",
            "divergence_patience",
        }
    )
    stable = schema_version == ISPY2_BIFLOW_STABLE_CONFIG_SCHEMA
    raw_training = _mapping(
        root["training"],
        "training",
        stable_training_fields if stable else legacy_training_fields,
    )
    precision = str(raw_training["precision"])
    if not precision.strip():
        raise ValueError("I-SPY2 BiFlowNet precision must be nonempty")
    dynamics_learning_rate = (
        _finite_float(
            raw_training["conditioner_learning_rate"],
            "conditioner_learning_rate",
            positive=True,
        )
        if stable
        else _finite_float(
            raw_training["dynamics_learning_rate"],
            "dynamics_learning_rate",
            positive=True,
        )
    )
    backbone_learning_rate = (
        _finite_float(
            raw_training["backbone_learning_rate"],
            "backbone_learning_rate",
            positive=True,
        )
        if stable
        else dynamics_learning_rate
    )
    controlnet_learning_rate = (
        _finite_float(
            raw_training["controlnet_learning_rate"],
            "controlnet_learning_rate",
            positive=True,
        )
        if stable
        else dynamics_learning_rate
    )
    conditioner_learning_rate = (
        _finite_float(
            raw_training["conditioner_learning_rate"],
            "conditioner_learning_rate",
            positive=True,
        )
        if stable
        else dynamics_learning_rate
    )
    warmup_fraction = (
        _finite_float(
            raw_training["warmup_fraction"], "warmup_fraction", positive=True
        )
        if stable
        else 0.0
    )
    if stable and not 0.0 < warmup_fraction < 1.0:
        raise ValueError("I-SPY2 BiFlowNet warmup_fraction must be in (0,1)")
    divergence_reference_metric = (
        _finite_float(
            raw_training["divergence_reference_metric"],
            "divergence_reference_metric",
            positive=True,
        )
        if stable
        else None
    )
    training = ISPY2BiFlowTrainingConfig(
        batch_size=_positive_int(raw_training["batch_size"], "batch_size"),
        accumulate_grad_batches=_positive_int(
            raw_training["accumulate_grad_batches"], "accumulate_grad_batches"
        ),
        max_epochs=_positive_int(raw_training["max_epochs"], "max_epochs"),
        precision=precision,
        num_workers=_nonnegative_int(raw_training["num_workers"], "num_workers"),
        dynamics_learning_rate=dynamics_learning_rate,
        backbone_learning_rate=backbone_learning_rate,
        controlnet_learning_rate=controlnet_learning_rate,
        conditioner_learning_rate=conditioner_learning_rate,
        text_lora_learning_rate=_finite_float(
            raw_training["text_lora_learning_rate"],
            "text_lora_learning_rate",
            positive=True,
        ),
        weight_decay=_finite_float(
            raw_training["weight_decay"], "weight_decay", nonnegative=True
        ),
        gradient_clip_val=_finite_float(
            raw_training["gradient_clip_val"], "gradient_clip_val", positive=True
        ),
        evaluation_solver_steps=_positive_int(
            raw_training["evaluation_solver_steps"], "evaluation_solver_steps"
        ),
        early_stopping_patience=_positive_int(
            raw_training["early_stopping_patience"], "early_stopping_patience"
        ),
        optimizer_layout="split_backbone_v1" if stable else "legacy_joint_v1",
        min_learning_rate=(
            _finite_float(
                raw_training["min_learning_rate"],
                "min_learning_rate",
                positive=True,
            )
            if stable
            else dynamics_learning_rate
        ),
        warmup_fraction=warmup_fraction,
        divergence_reference_metric=divergence_reference_metric,
        divergence_multiplier=(
            _finite_float(
                raw_training["divergence_multiplier"],
                "divergence_multiplier",
                positive=True,
            )
            if stable
            else 1.5
        ),
        divergence_patience=(
            _positive_int(
                raw_training["divergence_patience"], "divergence_patience"
            )
            if stable
            else 2
        ),
    )
    if stable and training.min_learning_rate > min(
        training.backbone_learning_rate,
        training.controlnet_learning_rate,
        training.conditioner_learning_rate,
        training.text_lora_learning_rate,
    ):
        raise ValueError(
            "I-SPY2 BiFlowNet min_learning_rate exceeds a peak learning rate"
        )
    runtime_fields = {"output_root", "seed"}
    if stable:
        runtime_fields |= {"warm_start_checkpoint", "warm_start_sha256"}
    raw_runtime = _mapping(root["runtime"], "runtime", runtime_fields)
    warm_start_checkpoint = (
        _path(raw_runtime["warm_start_checkpoint"], "warm_start_checkpoint")
        if stable
        else None
    )
    warm_start_sha256 = str(raw_runtime["warm_start_sha256"]) if stable else None
    if stable and (
        not warm_start_checkpoint.is_file()
        or len(warm_start_sha256) != 64
        or any(value not in "0123456789abcdef" for value in warm_start_sha256)
    ):
        raise ValueError("I-SPY2 BiFlowNet warm-start contract is invalid")
    runtime = ISPY2BiFlowRuntimeConfig(
        output_root=_path(raw_runtime["output_root"], "output_root"),
        seed=_nonnegative_int(raw_runtime["seed"], "seed"),
        warm_start_checkpoint=warm_start_checkpoint,
        warm_start_sha256=warm_start_sha256,
    )
    return ISPY2BiFlowConfig(
        path=source,
        schema_version=schema_version,
        base=base,
        preset=preset,
        training=training,
        runtime=runtime,
        raw=root,
        sha256=_sha256_file(source),
    )


__all__ = [
    "ISPY2_BIFLOW_BASE_CONFIG_SCHEMA",
    "ISPY2_BIFLOW_CONFIG_SCHEMA",
    "ISPY2_BIFLOW_STABLE_CONFIG_SCHEMA",
    "ISPY2BiFlowBaseConfig",
    "ISPY2_BIFLOW_PRESET",
    "ISPY2BiFlowConfig",
    "ISPY2BiFlowRuntimeConfig",
    "ISPY2BiFlowTrainingConfig",
    "load_ispy2_biflow_config",
]
