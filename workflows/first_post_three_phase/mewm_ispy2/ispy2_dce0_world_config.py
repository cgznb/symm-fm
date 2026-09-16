from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .backend import (
    REGISTERED_STRICT_A_BUNDLE_SCHEMA,
    validate_registered_strict_a_bundle_contract,
)


ISPY2_DCE0_WORLD_CONFIG_SCHEMA = "mewm_ispy2_dce0_ser_post_world_model_config_v1"
ISPY2_DCE0_WORLD_ARCHITECTURE = "ispy2_dce0_ser_fmbcmri_single_unet_flow_v1"
ISPY2_DCE0_WORLD_PAIR_MODE = "all_connected_later_endpoints"
ISPY2_DCE0_WORLD_LATENT_NORMALIZATION = "continuous_codebook_minmax_v1"
ISPY2_DCE0_WORLD_SOURCE_MODALITIES = ("dce0", "ser")
ISPY2_DCE0_WORLD_SOURCE_MRI_SHAPE = (2, 96, 256, 256)
ISPY2_DCE0_WORLD_LATENT_SHAPE = (8, 24, 64, 64)
ISPY2_DCE0_WORLD_IMAGE_CONTEXT_GRID = (3, 4, 4)
ISPY2_DCE0_WORLD_IMAGE_CONTEXT_TOKENS = 96
ISPY2_DCE0_WORLD_CONTEXT_TOKENS = 100
ISPY2_DCE0_WORLD_CODEBOOK_MIN = -33.624992
ISPY2_DCE0_WORLD_CODEBOOK_MAX = 51.080605


@dataclass(frozen=True)
class ISPY2DCE0WorldDataConfig:
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
    output_shape_zyx: tuple[int, int, int]
    pair_mode: str
    codebook_min: float
    codebook_max: float


@dataclass(frozen=True)
class ISPY2DCE0WorldFMConfig:
    root: Path
    revision: str
    checkpoint: Path
    checkpoint_sha256: str
    fully_trainable: bool
    stem_input_channels: int
    patch_size_zyx: tuple[int, int, int]
    feature_pool_zyx: tuple[int, int, int]
    output_pool_zyx: tuple[int, int, int]
    embed_dim: int
    depth: int
    num_heads: int


@dataclass(frozen=True)
class ISPY2DCE0WorldTextConfig:
    model_id: str
    revision: str
    max_length: int
    load_in_4bit: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]


@dataclass(frozen=True)
class ISPY2DCE0WorldModelConfig:
    architecture: str
    prediction_target: str
    latent_normalization: str
    context_dim: int
    image_context_grid: tuple[int, int, int]
    image_context_tokens: int
    context_tokens: int
    source_mri_shape_czyx: tuple[int, int, int, int]
    latent_channels: int
    latent_shape_czyx: tuple[int, int, int, int]
    flow_channels: tuple[int, ...]
    flow_attention_levels: tuple[bool, ...]
    flow_res_blocks: int
    flow_head_channels: int


@dataclass(frozen=True)
class ISPY2DCE0WorldTrainingConfig:
    batch_size: int
    accumulate_grad_batches: int
    max_epochs: int
    precision: str
    num_workers: int
    dynamics_learning_rate: float
    fm_learning_rate: float
    text_lora_learning_rate: float
    weight_decay: float
    gradient_clip_val: float
    velocity_loss: str
    evaluation_solver_steps: int
    early_stopping_patience: int


@dataclass(frozen=True)
class ISPY2DCE0WorldRuntimeConfig:
    output_root: Path
    seed: int


@dataclass(frozen=True)
class ISPY2DCE0WorldConfig:
    path: Path
    schema_version: str
    data: ISPY2DCE0WorldDataConfig
    fm: ISPY2DCE0WorldFMConfig
    text: ISPY2DCE0WorldTextConfig
    model: ISPY2DCE0WorldModelConfig
    training: ISPY2DCE0WorldTrainingConfig
    runtime: ISPY2DCE0WorldRuntimeConfig
    sha256: str
    raw: dict[str, Any]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bundle_contract_sha256": self.data.bundle_contract_sha256,
            "vqgan_sha256": self.data.vqgan_sha256,
            "fm_revision": self.fm.revision,
            "fm_checkpoint_sha256": self.fm.checkpoint_sha256,
            "text_revision": self.text.revision,
            "data": self.raw["data"],
            "fm": self.raw["fm"],
            "text": self.raw["text"],
            "model": self.raw["model"],
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


def _mapping(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"I-SPY2 DCE0 world config {name} fields are invalid")
    return value


def _path(value: Any, name: str) -> Path:
    if type(value) is not str or not value.strip():
        raise ValueError(f"I-SPY2 DCE0 world config {name} path is invalid")
    return Path(value).expanduser().resolve()


def _sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"I-SPY2 DCE0 world config {name} SHA256 is invalid")
    return value


def _git_sha(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"I-SPY2 DCE0 world config {name} must be a pinned Git SHA")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"I-SPY2 DCE0 world config {name} must be positive")
    return value


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"I-SPY2 DCE0 world config {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or positive and result <= 0.0:
        raise ValueError(f"I-SPY2 DCE0 world config {name} is invalid")
    return result


def _int_tuple(value: Any, name: str, length: int) -> tuple[int, ...]:
    if type(value) is not list or len(value) != length or any(
        type(item) is not int or item <= 0 for item in value
    ):
        raise ValueError(f"I-SPY2 DCE0 world config {name} is invalid")
    return tuple(value)


def _bool_tuple(value: Any, name: str, length: int) -> tuple[bool, ...]:
    if type(value) is not list or len(value) != length or any(
        type(item) is not bool for item in value
    ):
        raise ValueError(f"I-SPY2 DCE0 world config {name} is invalid")
    return tuple(value)


@lru_cache(maxsize=16)
def _sha256_file_cached(path: Path, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_file_cached(path, stat.st_size, stat.st_mtime_ns)


def load_ispy2_dce0_world_config(path: str | Path) -> ISPY2DCE0WorldConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"I-SPY2 DCE0 world config is missing: {source}")
    try:
        parsed = _release_yaml(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ValueError("I-SPY2 DCE0 world config is unreadable") from None
    root = _mapping(
        parsed,
        "root",
        {"schema_version", "data", "fm", "text", "model", "training", "runtime"},
    )
    if root["schema_version"] != ISPY2_DCE0_WORLD_CONFIG_SCHEMA:
        raise ValueError("I-SPY2 DCE0 world config schema is unsupported")

    raw_data = _mapping(
        root["data"],
        "data",
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
    bundle_json = _path(raw_data["bundle_json"], "data.bundle_json")
    phase_manifest = _path(raw_data["phase_manifest_csv"], "data.phase_manifest_csv")
    roi_cache = _path(raw_data["roi_cache_dir"], "data.roi_cache_dir")
    vqgan_config = _path(raw_data["vqgan_config"], "data.vqgan_config")
    vqgan_checkpoint = _path(raw_data["vqgan_checkpoint"], "data.vqgan_checkpoint")
    if raw_data["backend"] != "registered_t0":
        raise ValueError("I-SPY2 DCE0 world data backend must be registered_t0")
    if not bundle_json.is_file() or not phase_manifest.is_file():
        raise FileNotFoundError("I-SPY2 DCE0 world bundle or phase manifest is missing")
    if not roi_cache.is_dir() or roi_cache.is_symlink():
        raise FileNotFoundError("I-SPY2 DCE0 world ROI cache is missing or unsafe")
    if not vqgan_config.is_file() or not vqgan_checkpoint.is_file():
        raise FileNotFoundError("I-SPY2 DCE0 world VQGAN files are missing")
    bundle = json.loads(bundle_json.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != REGISTERED_STRICT_A_BUNDLE_SCHEMA:
        raise ValueError("I-SPY2 DCE0 world bundle schema is unsupported")
    expected_bundle_sha = _sha256(
        raw_data["bundle_contract_sha256"], "data.bundle_contract"
    )
    actual_bundle_sha = validate_registered_strict_a_bundle_contract(
        bundle, bundle_path=bundle_json
    )
    if actual_bundle_sha != expected_bundle_sha:
        raise ValueError("I-SPY2 DCE0 world bundle contract SHA256 mismatch")
    counts = bundle.get("counts", {})
    if (
        counts.get("fold_patient_counts") != {"train": 765, "val": 102}
        or counts.get("fold_transition_counts") != {"train": 1868, "val": 278}
        or counts.get("transition_type_counts")
        != {"T0->T1": 837, "T1->T2": 698, "T2->T3": 611}
    ):
        raise ValueError("I-SPY2 DCE0 world extended Strict-A counts changed")
    source_modalities = tuple(raw_data["source_modalities"])
    output_shape = _int_tuple(raw_data["output_shape_zyx"], "data.output_shape_zyx", 3)
    minimum = _finite_float(raw_data["codebook_min"], "data.codebook_min")
    maximum = _finite_float(raw_data["codebook_max"], "data.codebook_max")
    if source_modalities != ISPY2_DCE0_WORLD_SOURCE_MODALITIES:
        raise ValueError("I-SPY2 DCE0 world source modality order must be dce0,ser")
    if output_shape != ISPY2_DCE0_WORLD_SOURCE_MRI_SHAPE[1:]:
        raise ValueError("I-SPY2 DCE0 world source MRI shape changed")
    if raw_data["pair_mode"] != ISPY2_DCE0_WORLD_PAIR_MODE:
        raise ValueError("I-SPY2 DCE0 world pair mode is unsupported")
    if (minimum, maximum) != (
        ISPY2_DCE0_WORLD_CODEBOOK_MIN,
        ISPY2_DCE0_WORLD_CODEBOOK_MAX,
    ):
        raise ValueError("I-SPY2 DCE0 world codebook range changed")
    vqgan_sha = _sha256(raw_data["vqgan_sha256"], "data.vqgan")
    if _sha256_file(vqgan_checkpoint) != vqgan_sha:
        raise ValueError("I-SPY2 DCE0 world VQGAN checkpoint SHA256 mismatch")
    data = ISPY2DCE0WorldDataConfig(
        backend="registered_t0",
        bundle_json=bundle_json,
        bundle_contract_sha256=expected_bundle_sha,
        phase_manifest_csv=phase_manifest,
        roi_cache_dir=roi_cache,
        vqgan_config=vqgan_config,
        vqgan_checkpoint=vqgan_checkpoint,
        vqgan_sha256=vqgan_sha,
        continuous_root=_path(raw_data["continuous_root"], "data.continuous_root"),
        source_modalities=source_modalities,
        output_shape_zyx=output_shape,
        pair_mode=ISPY2_DCE0_WORLD_PAIR_MODE,
        codebook_min=minimum,
        codebook_max=maximum,
    )

    raw_fm = _mapping(
        root["fm"],
        "fm",
        {
            "root",
            "revision",
            "checkpoint",
            "checkpoint_sha256",
            "fully_trainable",
            "stem_input_channels",
            "patch_size_zyx",
            "feature_pool_zyx",
            "output_pool_zyx",
            "embed_dim",
            "depth",
            "num_heads",
        },
    )
    fm_root = _path(raw_fm["root"], "fm.root")
    fm_checkpoint = _path(raw_fm["checkpoint"], "fm.checkpoint")
    if not fm_root.is_dir() or not fm_checkpoint.is_file():
        raise FileNotFoundError("I-SPY2 DCE0 world FM-BCMRI files are missing")
    fm_sha = _sha256(raw_fm["checkpoint_sha256"], "fm.checkpoint")
    if _sha256_file(fm_checkpoint) != fm_sha:
        raise ValueError("I-SPY2 DCE0 world FM-BCMRI checkpoint SHA256 mismatch")
    fm = ISPY2DCE0WorldFMConfig(
        root=fm_root,
        revision=_git_sha(raw_fm["revision"], "fm.revision"),
        checkpoint=fm_checkpoint,
        checkpoint_sha256=fm_sha,
        fully_trainable=raw_fm["fully_trainable"] is True,
        stem_input_channels=_positive_int(
            raw_fm["stem_input_channels"], "fm.stem_input_channels"
        ),
        patch_size_zyx=_int_tuple(raw_fm["patch_size_zyx"], "fm.patch_size_zyx", 3),
        feature_pool_zyx=_int_tuple(
            raw_fm["feature_pool_zyx"], "fm.feature_pool_zyx", 3
        ),
        output_pool_zyx=_int_tuple(
            raw_fm["output_pool_zyx"], "fm.output_pool_zyx", 3
        ),
        embed_dim=_positive_int(raw_fm["embed_dim"], "fm.embed_dim"),
        depth=_positive_int(raw_fm["depth"], "fm.depth"),
        num_heads=_positive_int(raw_fm["num_heads"], "fm.num_heads"),
    )
    if (
        not fm.fully_trainable
        or fm.stem_input_channels != 1
        or fm.patch_size_zyx != (8, 8, 8)
        or fm.feature_pool_zyx != (2, 4, 4)
        or fm.output_pool_zyx != (2, 2, 2)
        or fm.embed_dim != 768
        or fm.depth != 6
        or fm.num_heads != 12
    ):
        raise ValueError("I-SPY2 DCE0 world FM architecture contract changed")

    raw_text = _mapping(
        root["text"],
        "text",
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
    dropout = _finite_float(raw_text["lora_dropout"], "text.lora_dropout")
    targets = tuple(raw_text["lora_targets"])
    text_config = ISPY2DCE0WorldTextConfig(
        model_id=str(raw_text["model_id"]),
        revision=_git_sha(raw_text["revision"], "text.revision"),
        max_length=_positive_int(raw_text["max_length"], "text.max_length"),
        load_in_4bit=raw_text["load_in_4bit"] is True,
        lora_rank=_positive_int(raw_text["lora_rank"], "text.lora_rank"),
        lora_alpha=_positive_int(raw_text["lora_alpha"], "text.lora_alpha"),
        lora_dropout=dropout,
        lora_targets=targets,
    )
    if (
        text_config.model_id != "google/medgemma-4b-it"
        or text_config.max_length != 512
        or not text_config.load_in_4bit
        or not 0.0 <= dropout < 1.0
        or targets != ("q_proj", "v_proj")
    ):
        raise ValueError("I-SPY2 DCE0 world text contract changed")

    raw_model = _mapping(
        root["model"],
        "model",
        {
            "architecture",
            "prediction_target",
            "latent_normalization",
            "context_dim",
            "image_context_grid",
            "image_context_tokens",
            "context_tokens",
            "source_mri_shape_czyx",
            "latent_channels",
            "latent_shape_czyx",
            "flow_channels",
            "flow_attention_levels",
            "flow_res_blocks",
            "flow_head_channels",
        },
    )
    model = ISPY2DCE0WorldModelConfig(
        architecture=str(raw_model["architecture"]),
        prediction_target=str(raw_model["prediction_target"]),
        latent_normalization=str(raw_model["latent_normalization"]),
        context_dim=_positive_int(raw_model["context_dim"], "model.context_dim"),
        image_context_grid=_int_tuple(
            raw_model["image_context_grid"], "model.image_context_grid", 3
        ),
        image_context_tokens=_positive_int(
            raw_model["image_context_tokens"], "model.image_context_tokens"
        ),
        context_tokens=_positive_int(raw_model["context_tokens"], "model.context_tokens"),
        source_mri_shape_czyx=_int_tuple(
            raw_model["source_mri_shape_czyx"], "model.source_mri_shape_czyx", 4
        ),
        latent_channels=_positive_int(
            raw_model["latent_channels"], "model.latent_channels"
        ),
        latent_shape_czyx=_int_tuple(
            raw_model["latent_shape_czyx"], "model.latent_shape_czyx", 4
        ),
        flow_channels=_int_tuple(raw_model["flow_channels"], "model.flow_channels", 4),
        flow_attention_levels=_bool_tuple(
            raw_model["flow_attention_levels"], "model.flow_attention_levels", 4
        ),
        flow_res_blocks=_positive_int(
            raw_model["flow_res_blocks"], "model.flow_res_blocks"
        ),
        flow_head_channels=_positive_int(
            raw_model["flow_head_channels"], "model.flow_head_channels"
        ),
    )
    if (
        model.architecture != ISPY2_DCE0_WORLD_ARCHITECTURE
        or model.prediction_target != "state"
        or model.latent_normalization != ISPY2_DCE0_WORLD_LATENT_NORMALIZATION
        or model.context_dim != 768
        or model.image_context_grid != ISPY2_DCE0_WORLD_IMAGE_CONTEXT_GRID
        or model.image_context_tokens != ISPY2_DCE0_WORLD_IMAGE_CONTEXT_TOKENS
        or model.context_tokens != ISPY2_DCE0_WORLD_CONTEXT_TOKENS
        or model.source_mri_shape_czyx != ISPY2_DCE0_WORLD_SOURCE_MRI_SHAPE
        or model.latent_channels != 8
        or model.latent_shape_czyx != ISPY2_DCE0_WORLD_LATENT_SHAPE
        or model.flow_channels != (64, 128, 256, 256)
        or model.flow_attention_levels != (False, False, True, True)
        or model.flow_res_blocks != 2
        or model.flow_head_channels != 32
    ):
        raise ValueError("I-SPY2 DCE0 world model architecture contract changed")

    raw_training = _mapping(
        root["training"],
        "training",
        {
            "batch_size",
            "accumulate_grad_batches",
            "max_epochs",
            "precision",
            "num_workers",
            "dynamics_learning_rate",
            "fm_learning_rate",
            "text_lora_learning_rate",
            "weight_decay",
            "gradient_clip_val",
            "velocity_loss",
            "evaluation_solver_steps",
            "early_stopping_patience",
        },
    )
    training = ISPY2DCE0WorldTrainingConfig(
        batch_size=_positive_int(raw_training["batch_size"], "training.batch_size"),
        accumulate_grad_batches=_positive_int(
            raw_training["accumulate_grad_batches"], "training.accumulate_grad_batches"
        ),
        max_epochs=_positive_int(raw_training["max_epochs"], "training.max_epochs"),
        precision=str(raw_training["precision"]),
        num_workers=_positive_int(raw_training["num_workers"], "training.num_workers"),
        dynamics_learning_rate=_finite_float(
            raw_training["dynamics_learning_rate"],
            "training.dynamics_learning_rate",
            positive=True,
        ),
        fm_learning_rate=_finite_float(
            raw_training["fm_learning_rate"], "training.fm_learning_rate", positive=True
        ),
        text_lora_learning_rate=_finite_float(
            raw_training["text_lora_learning_rate"],
            "training.text_lora_learning_rate",
            positive=True,
        ),
        weight_decay=_finite_float(raw_training["weight_decay"], "training.weight_decay"),
        gradient_clip_val=_finite_float(
            raw_training["gradient_clip_val"], "training.gradient_clip_val", positive=True
        ),
        velocity_loss=str(raw_training["velocity_loss"]),
        evaluation_solver_steps=_positive_int(
            raw_training["evaluation_solver_steps"], "training.evaluation_solver_steps"
        ),
        early_stopping_patience=_positive_int(
            raw_training["early_stopping_patience"], "training.early_stopping_patience"
        ),
    )
    if (
        (
            training.batch_size, training.accumulate_grad_batches, training.max_epochs
        )
        not in {(1, 8, 200), (4, 1, 250)}
        or training.precision != "bf16-mixed"
        or training.dynamics_learning_rate != 1e-4
        or training.fm_learning_rate != 1e-5
        or training.text_lora_learning_rate != 1e-4
        or training.weight_decay != 0.05
        or training.gradient_clip_val != 1.0
        or training.velocity_loss != "l1"
    ):
        raise ValueError("I-SPY2 DCE0 world training contract changed")

    raw_runtime = _mapping(root["runtime"], "runtime", {"output_root", "seed"})
    runtime = ISPY2DCE0WorldRuntimeConfig(
        output_root=_path(raw_runtime["output_root"], "runtime.output_root"),
        seed=_positive_int(raw_runtime["seed"], "runtime.seed"),
    )
    return ISPY2DCE0WorldConfig(
        path=source,
        schema_version=ISPY2_DCE0_WORLD_CONFIG_SCHEMA,
        data=data,
        fm=fm,
        text=text_config,
        model=model,
        training=training,
        runtime=runtime,
        sha256=_sha256_file(source),
        raw=root,
    )


__all__ = [
    "ISPY2_DCE0_WORLD_ARCHITECTURE",
    "ISPY2_DCE0_WORLD_CODEBOOK_MAX",
    "ISPY2_DCE0_WORLD_CODEBOOK_MIN",
    "ISPY2_DCE0_WORLD_CONFIG_SCHEMA",
    "ISPY2_DCE0_WORLD_CONTEXT_TOKENS",
    "ISPY2_DCE0_WORLD_IMAGE_CONTEXT_GRID",
    "ISPY2_DCE0_WORLD_IMAGE_CONTEXT_TOKENS",
    "ISPY2_DCE0_WORLD_LATENT_NORMALIZATION",
    "ISPY2_DCE0_WORLD_LATENT_SHAPE",
    "ISPY2_DCE0_WORLD_PAIR_MODE",
    "ISPY2_DCE0_WORLD_SOURCE_MODALITIES",
    "ISPY2_DCE0_WORLD_SOURCE_MRI_SHAPE",
    "ISPY2DCE0WorldConfig",
    "ISPY2DCE0WorldDataConfig",
    "ISPY2DCE0WorldFMConfig",
    "ISPY2DCE0WorldModelConfig",
    "ISPY2DCE0WorldRuntimeConfig",
    "ISPY2DCE0WorldTextConfig",
    "ISPY2DCE0WorldTrainingConfig",
    "load_ispy2_dce0_world_config",
]
