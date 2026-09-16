from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .latent_statistics import LATENT_CHANNEL_STATISTICS_SCHEMA


LATENT_PCR_CONFIG_SCHEMA = "mewm_ispy2_latent_pcr_config_v1"
LATENT_PCR_CONFIG_SCHEMA_V2 = "mewm_ispy2_latent_pcr_config_v2"
LATENT_PCR_ARCHITECTURE = "fmbcmri_vit_base_anisotropic_unetr_v1"
LATENT_PCR_ARCHITECTURE_V2 = "fmbcmri_vit_base_anisotropic_unetr_pooled_wide_ffn_v2"
LATENT_PCR_MODES = frozenset({"parallel", "cascade_detach", "cascade_joint"})
_HEX = frozenset("0123456789abcdef")


def _mapping(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"latent-pCR config {name} fields are invalid")
    return value


def _path(value: Any, name: str) -> Path:
    if type(value) is not str or not value.strip():
        raise ValueError(f"latent-pCR config {name} path is invalid")
    return Path(value).expanduser().resolve()


def _sha256(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in _HEX for c in value):
        raise ValueError(f"latent-pCR config {name} SHA256 is invalid")
    return value


def _git_sha(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 40 or any(c not in _HEX for c in value):
        raise ValueError(f"latent-pCR config {name} Git revision is invalid")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"latent-pCR config {name} must be a positive integer")
    return value


def _positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"latent-pCR config {name} must be numeric")
    result = float(value)
    minimum_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not minimum_ok:
        raise ValueError(f"latent-pCR config {name} is invalid")
    return result


@dataclass(frozen=True)
class LatentPCRDataConfig:
    backend: str
    bundle_json: Path
    phase_manifest_csv: Path
    roi_cache_dir: Path
    clinical_xlsx: Path
    clinical_sha256: str
    output_shape_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class LatentPCRLatentConfig:
    cache_dir: Path
    statistics_path: Path
    statistics_sha256: str
    statistics_schema: str
    vqgan_checkpoint: Path
    vqgan_sha256: str


@dataclass(frozen=True)
class LatentPCRExternalConfig:
    fmbcmri_root: Path
    fmbcmri_revision: str
    fmbcmri_checkpoint: Path
    fmbcmri_checkpoint_sha256: str


@dataclass(frozen=True)
class LatentPCRModelConfig:
    architecture: str
    latent_channels: int
    input_channels: int
    latent_shape_czyx: tuple[int, int, int, int]
    patch_size_zyx: tuple[int, int, int]
    embed_dim: int
    depth: int
    num_heads: int
    decoder_feature_size: int
    pcr_head_type: str
    pcr_token_dim: int
    pcr_hidden_dim: int
    pcr_attention_dim: int | None
    pcr_ffn_dim: int
    pcr_dropout: float
    roi_dilation: int
    roi_weight: float


@dataclass(frozen=True)
class LatentPCRTrainingConfig:
    batch_size: int
    accumulate_grad_batches: int
    max_epochs: int
    precision: str
    num_workers: int
    new_module_learning_rate: float
    late_vit_learning_rate: float
    pcr_head_learning_rate: float
    new_module_warmup_epochs: int
    late_vit_warmup_epochs: int
    pcr_head_lr_warmup_epochs: int
    learning_rate_warmup_start_ratio: float
    learning_rate_minimum_ratio: float
    weight_decay: float
    gradient_clip_val: float
    vit_unfreeze_epoch: int
    pcr_warmup_epochs: int
    pcr_ramp_epochs: int
    pcr_weight: float


@dataclass(frozen=True)
class LatentPCRRuntimeConfig:
    output_root: Path
    seed: int


@dataclass(frozen=True)
class LatentPCRExperimentConfig:
    schema_version: str
    path: Path
    data: LatentPCRDataConfig
    latents: LatentPCRLatentConfig
    external: LatentPCRExternalConfig
    model: LatentPCRModelConfig
    training: LatentPCRTrainingConfig
    runtime: LatentPCRRuntimeConfig
    raw: dict[str, Any]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "architecture": self.model.architecture,
            "bundle_json": str(self.data.bundle_json),
            "clinical_sha256": self.data.clinical_sha256,
            "latent_statistics_sha256": self.latents.statistics_sha256,
            "vqgan_sha256": self.latents.vqgan_sha256,
            "fmbcmri_revision": self.external.fmbcmri_revision,
            "fmbcmri_checkpoint_sha256": self.external.fmbcmri_checkpoint_sha256,
            "model": self.raw["model"],
        }

    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.identity_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        import hashlib

        return hashlib.sha256(encoded).hexdigest()


def load_latent_pcr_config(path: str | Path) -> LatentPCRExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    payload = _release_yaml(config_path.read_text(encoding="utf-8"))
    root = _mapping(
        payload,
        "root",
        {"schema_version", "data", "latents", "external", "model", "training", "runtime"},
    )
    schema_version = root["schema_version"]
    if schema_version not in {LATENT_PCR_CONFIG_SCHEMA, LATENT_PCR_CONFIG_SCHEMA_V2}:
        raise ValueError("latent-pCR config schema is unsupported")

    data = _mapping(
        root["data"],
        "data",
        {
            "backend",
            "bundle_json",
            "phase_manifest_csv",
            "roi_cache_dir",
            "clinical_xlsx",
            "clinical_sha256",
            "output_shape_zyx",
        },
    )
    shape = tuple(data["output_shape_zyx"])
    if shape != (96, 256, 256) or data["backend"] != "registered_t0":
        raise ValueError("latent-pCR data must use registered Strict-A geometry")
    data_config = LatentPCRDataConfig(
        backend="registered_t0",
        bundle_json=_path(data["bundle_json"], "data.bundle_json"),
        phase_manifest_csv=_path(data["phase_manifest_csv"], "data.phase_manifest_csv"),
        roi_cache_dir=_path(data["roi_cache_dir"], "data.roi_cache_dir"),
        clinical_xlsx=_path(data["clinical_xlsx"], "data.clinical_xlsx"),
        clinical_sha256=_sha256(data["clinical_sha256"], "data.clinical"),
        output_shape_zyx=shape,
    )

    latents = _mapping(
        root["latents"],
        "latents",
        {
            "cache_dir",
            "statistics_path",
            "statistics_sha256",
            "statistics_schema",
            "vqgan_checkpoint",
            "vqgan_sha256",
        },
    )
    if latents["statistics_schema"] != LATENT_CHANNEL_STATISTICS_SCHEMA:
        raise ValueError("latent-pCR statistics schema is unsupported")
    latent_config = LatentPCRLatentConfig(
        cache_dir=_path(latents["cache_dir"], "latents.cache_dir"),
        statistics_path=_path(latents["statistics_path"], "latents.statistics_path"),
        statistics_sha256=_sha256(latents["statistics_sha256"], "latents.statistics"),
        statistics_schema=latents["statistics_schema"],
        vqgan_checkpoint=_path(latents["vqgan_checkpoint"], "latents.vqgan_checkpoint"),
        vqgan_sha256=_sha256(latents["vqgan_sha256"], "latents.vqgan"),
    )

    external = _mapping(
        root["external"],
        "external",
        {
            "fmbcmri_root",
            "fmbcmri_revision",
            "fmbcmri_checkpoint",
            "fmbcmri_checkpoint_sha256",
        },
    )
    revision = _git_sha(external["fmbcmri_revision"], "external.fmbcmri_revision")
    external_config = LatentPCRExternalConfig(
        fmbcmri_root=_path(external["fmbcmri_root"], "external.fmbcmri_root"),
        fmbcmri_revision=revision,
        fmbcmri_checkpoint=_path(external["fmbcmri_checkpoint"], "external.fmbcmri_checkpoint"),
        fmbcmri_checkpoint_sha256=_sha256(
            external["fmbcmri_checkpoint_sha256"], "external.fmbcmri_checkpoint"
        ),
    )

    legacy_model_fields = {
        "architecture",
        "latent_channels",
        "input_channels",
        "latent_shape_czyx",
        "patch_size_zyx",
        "embed_dim",
        "depth",
        "num_heads",
        "decoder_feature_size",
        "pcr_token_dim",
        "pcr_hidden_dim",
        "pcr_attention_dim",
        "roi_dilation",
        "roi_weight",
    }
    wide_ffn_model_fields = {
        "architecture",
        "latent_channels",
        "input_channels",
        "latent_shape_czyx",
        "patch_size_zyx",
        "embed_dim",
        "depth",
        "num_heads",
        "decoder_feature_size",
        "pcr_head_type",
        "pcr_token_dim",
        "pcr_hidden_dim",
        "pcr_ffn_dim",
        "pcr_dropout",
        "roi_dilation",
        "roi_weight",
    }
    model_fields = (
        legacy_model_fields
        if schema_version == LATENT_PCR_CONFIG_SCHEMA
        else wide_ffn_model_fields
    )
    model = _mapping(root["model"], "model", model_fields)
    fixed = {
        "architecture": (
            LATENT_PCR_ARCHITECTURE
            if schema_version == LATENT_PCR_CONFIG_SCHEMA
            else LATENT_PCR_ARCHITECTURE_V2
        ),
        "latent_channels": 8,
        "input_channels": 9,
        "latent_shape_czyx": [8, 24, 64, 64],
        "patch_size_zyx": [4, 8, 8],
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
    }
    if any(model[key] != value for key, value in fixed.items()):
        raise ValueError("latent-pCR FM-BCMRI architecture fields are not locked")
    if type(model["roi_dilation"]) is not int or model["roi_dilation"] < 0:
        raise ValueError("latent-pCR roi_dilation is invalid")
    if schema_version == LATENT_PCR_CONFIG_SCHEMA_V2:
        if model["pcr_head_type"] != "pooled_wide_ffn":
            raise ValueError("latent-pCR v2 pCR head type is invalid")
        pcr_dropout = _positive_float(model["pcr_dropout"], "model.pcr_dropout")
        if not 0.0 < pcr_dropout < 1.0:
            raise ValueError("latent-pCR pCR dropout is invalid")
    else:
        pcr_dropout = 0.1
    model_config = LatentPCRModelConfig(
        architecture=model["architecture"],
        latent_channels=8,
        input_channels=9,
        latent_shape_czyx=(8, 24, 64, 64),
        patch_size_zyx=(4, 8, 8),
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_feature_size=_positive_int(model["decoder_feature_size"], "model.decoder_feature_size"),
        pcr_head_type=(
            "gated_attention"
            if schema_version == LATENT_PCR_CONFIG_SCHEMA
            else model["pcr_head_type"]
        ),
        pcr_token_dim=_positive_int(model["pcr_token_dim"], "model.pcr_token_dim"),
        pcr_hidden_dim=_positive_int(model["pcr_hidden_dim"], "model.pcr_hidden_dim"),
        pcr_attention_dim=(
            _positive_int(model["pcr_attention_dim"], "model.pcr_attention_dim")
            if schema_version == LATENT_PCR_CONFIG_SCHEMA
            else None
        ),
        pcr_ffn_dim=(
            0
            if schema_version == LATENT_PCR_CONFIG_SCHEMA
            else _positive_int(model["pcr_ffn_dim"], "model.pcr_ffn_dim")
        ),
        pcr_dropout=pcr_dropout,
        roi_dilation=model["roi_dilation"],
        roi_weight=_positive_float(model["roi_weight"], "model.roi_weight", allow_zero=True),
    )

    legacy_training_fields = {
        "batch_size",
        "accumulate_grad_batches",
        "max_epochs",
        "precision",
        "num_workers",
        "new_module_learning_rate",
        "late_vit_learning_rate",
        "new_module_warmup_epochs",
        "late_vit_warmup_epochs",
        "learning_rate_warmup_start_ratio",
        "learning_rate_minimum_ratio",
        "weight_decay",
        "gradient_clip_val",
        "vit_unfreeze_epoch",
        "pcr_warmup_epochs",
        "pcr_ramp_epochs",
        "pcr_weight",
    }
    wide_ffn_training_fields = legacy_training_fields | {
        "pcr_head_learning_rate",
        "pcr_head_lr_warmup_epochs",
    }
    training_fields = (
        legacy_training_fields
        if schema_version == LATENT_PCR_CONFIG_SCHEMA
        else wide_ffn_training_fields
    )
    training = _mapping(root["training"], "training", training_fields)
    if training["precision"] != "bf16-mixed" or training["batch_size"] not in {8, 32, 64}:
        raise ValueError("latent-pCR training must use a registered BF16 physical batch size")
    if type(training["num_workers"]) is not int or training["num_workers"] < 0:
        raise ValueError("latent-pCR num_workers is invalid")
    if type(training["vit_unfreeze_epoch"]) is not int or training["vit_unfreeze_epoch"] < 0:
        raise ValueError("latent-pCR vit_unfreeze_epoch is invalid")
    warmup_start_ratio = _positive_float(
        training["learning_rate_warmup_start_ratio"],
        "training.learning_rate_warmup_start_ratio",
    )
    minimum_ratio = _positive_float(
        training["learning_rate_minimum_ratio"], "training.learning_rate_minimum_ratio"
    )
    if not 0.0 < minimum_ratio <= warmup_start_ratio < 1.0:
        raise ValueError("latent-pCR learning-rate ratios are invalid")
    training_config = LatentPCRTrainingConfig(
        batch_size=training["batch_size"],
        accumulate_grad_batches=_positive_int(
            training["accumulate_grad_batches"], "training.accumulate_grad_batches"
        ),
        max_epochs=_positive_int(training["max_epochs"], "training.max_epochs"),
        precision="bf16-mixed",
        num_workers=training["num_workers"],
        new_module_learning_rate=_positive_float(
            training["new_module_learning_rate"], "training.new_module_learning_rate"
        ),
        late_vit_learning_rate=_positive_float(
            training["late_vit_learning_rate"], "training.late_vit_learning_rate"
        ),
        pcr_head_learning_rate=(
            _positive_float(
                training["new_module_learning_rate"], "training.new_module_learning_rate"
            )
            if schema_version == LATENT_PCR_CONFIG_SCHEMA
            else _positive_float(
                training["pcr_head_learning_rate"], "training.pcr_head_learning_rate"
            )
        ),
        new_module_warmup_epochs=_positive_int(
            training["new_module_warmup_epochs"], "training.new_module_warmup_epochs"
        ),
        late_vit_warmup_epochs=_positive_int(
            training["late_vit_warmup_epochs"], "training.late_vit_warmup_epochs"
        ),
        pcr_head_lr_warmup_epochs=(
            _positive_int(
                training["new_module_warmup_epochs"], "training.new_module_warmup_epochs"
            )
            if schema_version == LATENT_PCR_CONFIG_SCHEMA
            else _positive_int(
                training["pcr_head_lr_warmup_epochs"],
                "training.pcr_head_lr_warmup_epochs",
            )
        ),
        learning_rate_warmup_start_ratio=warmup_start_ratio,
        learning_rate_minimum_ratio=minimum_ratio,
        weight_decay=_positive_float(training["weight_decay"], "training.weight_decay", allow_zero=True),
        gradient_clip_val=_positive_float(training["gradient_clip_val"], "training.gradient_clip_val"),
        vit_unfreeze_epoch=training["vit_unfreeze_epoch"],
        pcr_warmup_epochs=_positive_int(training["pcr_warmup_epochs"], "training.pcr_warmup_epochs"),
        pcr_ramp_epochs=_positive_int(training["pcr_ramp_epochs"], "training.pcr_ramp_epochs"),
        pcr_weight=_positive_float(training["pcr_weight"], "training.pcr_weight"),
    )

    runtime = _mapping(root["runtime"], "runtime", {"output_root", "seed"})
    if type(runtime["seed"]) is not int or runtime["seed"] < 0:
        raise ValueError("latent-pCR runtime seed is invalid")
    runtime_config = LatentPCRRuntimeConfig(
        output_root=_path(runtime["output_root"], "runtime.output_root"), seed=runtime["seed"]
    )
    return LatentPCRExperimentConfig(
        schema_version=schema_version,
        path=config_path,
        data=data_config,
        latents=latent_config,
        external=external_config,
        model=model_config,
        training=training_config,
        runtime=runtime_config,
        raw=root,
    )


__all__ = [
    "LATENT_PCR_ARCHITECTURE",
    "LATENT_PCR_ARCHITECTURE_V2",
    "LATENT_PCR_CONFIG_SCHEMA",
    "LATENT_PCR_CONFIG_SCHEMA_V2",
    "LATENT_PCR_MODES",
    "LatentPCRExperimentConfig",
    "load_latent_pcr_config",
]
