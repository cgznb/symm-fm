from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .latent_pcr_config import LatentPCRExperimentConfig, load_latent_pcr_config
from .latent_statistics import sha256_file


LATENT_WORLD_V4_CONFIG_SCHEMA = "mewm_ispy2_treatment_flow_world_model_config_v4"
LATENT_WORLD_V4_CHECKPOINT_SCHEMA = "mewm_ispy2_treatment_flow_world_model_checkpoint_v4"
LATENT_WORLD_V4_ARCHITECTURE = "fmbcmri_monai_controlnet_treatment_flow_v4"
V4_CONDITIONING_MODES = frozenset({"id", "clip_lora", "medgemma_lora"})
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
MEDGEMMA_MODEL_ID = "google/medgemma-4b-it"
MEDGEMMA_REVISION = "290cda5eeccbee130f987c4ad74a59ae6f196408"


def _mapping(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"latent-world v4 config {name} fields are invalid")
    return value


def _path(value: Any, name: str) -> Path:
    if type(value) is not str or not value.strip():
        raise ValueError(f"latent-world v4 config {name} path is invalid")
    return Path(value).expanduser().resolve()


def _sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"latent-world v4 config {name} SHA256 is invalid")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"latent-world v4 config {name} is invalid")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"latent-world v4 config {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"latent-world v4 config {name} is invalid")
    return result


def _integer_tuple(value: Any, name: str, *, length: int | None = None) -> tuple[int, ...]:
    if type(value) is not list or (length is not None and len(value) != length):
        raise ValueError(f"latent-world v4 config {name} is invalid")
    result = tuple(_integer(item, name, minimum=1) for item in value)
    if not result:
        raise ValueError(f"latent-world v4 config {name} is empty")
    return result


@dataclass(frozen=True)
class V4TextModelConfig:
    clip_model_id: str
    clip_revision: str
    medgemma_model_id: str
    medgemma_revision: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]


@dataclass(frozen=True)
class V4ConditioningConfig:
    mode: str
    context_dim: int
    arm_residual_scale: float
    text: V4TextModelConfig


@dataclass(frozen=True)
class V4ModelConfig:
    architecture: str
    latent_channels: int
    latent_shape_czyx: tuple[int, int, int, int]
    flow_channels: tuple[int, ...]
    flow_attention_levels: tuple[bool, ...]
    flow_res_blocks: int
    flow_head_channels: int
    fm_context_grid: tuple[int, int, int]
    gate_init_filters: int


@dataclass(frozen=True)
class V4LossConfig:
    flow_weight: float
    gate_weight: float
    rollout_weight: float
    roi_dilation: int


@dataclass(frozen=True)
class V4TrainingConfig:
    batch_size: int
    accumulate_grad_batches: int
    max_epochs: int
    precision: str
    num_workers: int
    new_module_learning_rate: float
    fm_learning_rate: float
    text_lora_learning_rate: float
    fm_layer_decay: float
    weight_decay: float
    gradient_clip_val: float
    warmup_epochs: int
    early_stopping_patience: int
    scheduled_sampling_start_epoch: int
    scheduled_sampling_end_epoch: int
    predicted_state_probability_max: float
    rollout_solver_steps: int
    evaluation_solver_steps: int


@dataclass(frozen=True)
class V4TrajectoryConfig:
    token_dim: int
    hidden_dim: int
    layers: int
    heads: int
    ffn_dim: int
    dropout: float
    cv_folds: int
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class LatentWorldV4ExperimentConfig:
    path: Path
    base_config_path: Path
    base_config_sha256: str
    base: LatentPCRExperimentConfig
    conditioning: V4ConditioningConfig
    model: V4ModelConfig
    losses: V4LossConfig
    training: V4TrainingConfig
    trajectory: V4TrajectoryConfig
    output_root: Path
    seed: int
    raw: dict[str, Any]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": LATENT_WORLD_V4_CONFIG_SCHEMA,
            "base_config_sha256": self.base_config_sha256,
            "base_identity_sha256": self.base.identity_sha256(),
            "conditioning": asdict(self.conditioning),
            "model": asdict(self.model),
            "losses": asdict(self.losses),
            "training": asdict(self.training),
            "trajectory": asdict(self.trajectory),
            "seed": self.seed,
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


def load_latent_world_v4_config(path: str | Path) -> LatentWorldV4ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    payload = _release_yaml(config_path.read_text(encoding="utf-8"))
    root = _mapping(
        payload,
        "root",
        {
            "schema_version",
            "base_config",
            "conditioning",
            "model",
            "losses",
            "training",
            "trajectory",
            "runtime",
        },
    )
    if root["schema_version"] != LATENT_WORLD_V4_CONFIG_SCHEMA:
        raise ValueError("latent-world v4 config schema is unsupported")

    base_raw = _mapping(root["base_config"], "base_config", {"path", "sha256"})
    base_path = _path(base_raw["path"], "base_config.path")
    base_sha256 = _sha256(base_raw["sha256"], "base_config.sha256")
    if sha256_file(base_path) != base_sha256:
        raise ValueError("latent-world v4 base config SHA256 mismatch")
    base = load_latent_pcr_config(base_path)

    conditioning_raw = _mapping(
        root["conditioning"],
        "conditioning",
        {"mode", "context_dim", "arm_residual_scale", "text"},
    )
    mode = str(conditioning_raw["mode"])
    if mode not in V4_CONDITIONING_MODES:
        raise ValueError("latent-world v4 conditioning mode is unsupported")
    text_raw = _mapping(
        conditioning_raw["text"],
        "conditioning.text",
        {
            "clip_model_id",
            "clip_revision",
            "medgemma_model_id",
            "medgemma_revision",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_targets",
        },
    )
    if (
        text_raw["clip_model_id"] != CLIP_MODEL_ID
        or text_raw["clip_revision"] != CLIP_REVISION
        or text_raw["medgemma_model_id"] != MEDGEMMA_MODEL_ID
        or text_raw["medgemma_revision"] != MEDGEMMA_REVISION
        or text_raw["lora_targets"] != ["q_proj", "v_proj"]
    ):
        raise ValueError("latent-world v4 text model identity is unsupported")
    lora_dropout = _number(text_raw["lora_dropout"], "conditioning.text.lora_dropout")
    if lora_dropout >= 1.0:
        raise ValueError("latent-world v4 LoRA dropout must be below one")
    text = V4TextModelConfig(
        clip_model_id=CLIP_MODEL_ID,
        clip_revision=CLIP_REVISION,
        medgemma_model_id=MEDGEMMA_MODEL_ID,
        medgemma_revision=MEDGEMMA_REVISION,
        lora_rank=_integer(text_raw["lora_rank"], "conditioning.text.lora_rank", minimum=1),
        lora_alpha=_integer(text_raw["lora_alpha"], "conditioning.text.lora_alpha", minimum=1),
        lora_dropout=lora_dropout,
        lora_targets=("q_proj", "v_proj"),
    )
    context_dim = _integer(conditioning_raw["context_dim"], "conditioning.context_dim", minimum=1)
    if context_dim != 768:
        raise ValueError("latent-world v4 context dimension must match FM-BCMRI")
    conditioning = V4ConditioningConfig(
        mode=mode,
        context_dim=context_dim,
        arm_residual_scale=_number(
            conditioning_raw["arm_residual_scale"], "conditioning.arm_residual_scale"
        ),
        text=text,
    )

    model_raw = _mapping(
        root["model"],
        "model",
        {
            "architecture",
            "latent_channels",
            "latent_shape_czyx",
            "flow_channels",
            "flow_attention_levels",
            "flow_res_blocks",
            "flow_head_channels",
            "fm_context_grid",
            "gate_init_filters",
        },
    )
    if model_raw["architecture"] != LATENT_WORLD_V4_ARCHITECTURE:
        raise ValueError("latent-world v4 architecture is unsupported")
    if model_raw["latent_channels"] != 8 or model_raw["latent_shape_czyx"] != [8, 24, 64, 64]:
        raise ValueError("latent-world v4 latent contract changed")
    flow_channels = _integer_tuple(model_raw["flow_channels"], "model.flow_channels")
    levels = model_raw["flow_attention_levels"]
    if type(levels) is not list or len(levels) != len(flow_channels) or any(type(item) is not bool for item in levels):
        raise ValueError("latent-world v4 attention levels are invalid")
    model = V4ModelConfig(
        architecture=LATENT_WORLD_V4_ARCHITECTURE,
        latent_channels=8,
        latent_shape_czyx=(8, 24, 64, 64),
        flow_channels=flow_channels,
        flow_attention_levels=tuple(levels),
        flow_res_blocks=_integer(model_raw["flow_res_blocks"], "model.flow_res_blocks", minimum=1),
        flow_head_channels=_integer(model_raw["flow_head_channels"], "model.flow_head_channels", minimum=1),
        fm_context_grid=_integer_tuple(model_raw["fm_context_grid"], "model.fm_context_grid", length=3),  # type: ignore[arg-type]
        gate_init_filters=_integer(model_raw["gate_init_filters"], "model.gate_init_filters", minimum=1),
    )

    loss_raw = _mapping(
        root["losses"],
        "losses",
        {"flow_weight", "gate_weight", "rollout_weight", "roi_dilation"},
    )
    losses = V4LossConfig(
        flow_weight=_number(loss_raw["flow_weight"], "losses.flow_weight"),
        gate_weight=_number(loss_raw["gate_weight"], "losses.gate_weight"),
        rollout_weight=_number(loss_raw["rollout_weight"], "losses.rollout_weight"),
        roi_dilation=_integer(loss_raw["roi_dilation"], "losses.roi_dilation"),
    )

    training_fields = {
        "batch_size",
        "accumulate_grad_batches",
        "max_epochs",
        "precision",
        "num_workers",
        "new_module_learning_rate",
        "fm_learning_rate",
        "text_lora_learning_rate",
        "fm_layer_decay",
        "weight_decay",
        "gradient_clip_val",
        "warmup_epochs",
        "early_stopping_patience",
        "scheduled_sampling_start_epoch",
        "scheduled_sampling_end_epoch",
        "predicted_state_probability_max",
        "rollout_solver_steps",
        "evaluation_solver_steps",
    }
    training_raw = _mapping(root["training"], "training", training_fields)
    if training_raw["precision"] not in {"bf16-mixed", "32-true"}:
        raise ValueError("latent-world v4 precision is unsupported")
    start = _integer(
        training_raw["scheduled_sampling_start_epoch"],
        "training.scheduled_sampling_start_epoch",
    )
    end = _integer(
        training_raw["scheduled_sampling_end_epoch"],
        "training.scheduled_sampling_end_epoch",
        minimum=1,
    )
    probability = _number(
        training_raw["predicted_state_probability_max"],
        "training.predicted_state_probability_max",
    )
    layer_decay = _number(training_raw["fm_layer_decay"], "training.fm_layer_decay")
    if end <= start or probability > 1.0 or not 0.0 < layer_decay <= 1.0:
        raise ValueError("latent-world v4 schedule is invalid")
    training = V4TrainingConfig(
        batch_size=_integer(training_raw["batch_size"], "training.batch_size", minimum=1),
        accumulate_grad_batches=_integer(
            training_raw["accumulate_grad_batches"],
            "training.accumulate_grad_batches",
            minimum=1,
        ),
        max_epochs=_integer(training_raw["max_epochs"], "training.max_epochs", minimum=1),
        precision=training_raw["precision"],
        num_workers=_integer(training_raw["num_workers"], "training.num_workers"),
        new_module_learning_rate=_number(
            training_raw["new_module_learning_rate"],
            "training.new_module_learning_rate",
            minimum=1e-12,
        ),
        fm_learning_rate=_number(
            training_raw["fm_learning_rate"], "training.fm_learning_rate", minimum=1e-12
        ),
        text_lora_learning_rate=_number(
            training_raw["text_lora_learning_rate"],
            "training.text_lora_learning_rate",
            minimum=1e-12,
        ),
        fm_layer_decay=layer_decay,
        weight_decay=_number(training_raw["weight_decay"], "training.weight_decay"),
        gradient_clip_val=_number(
            training_raw["gradient_clip_val"], "training.gradient_clip_val", minimum=1e-12
        ),
        warmup_epochs=_integer(training_raw["warmup_epochs"], "training.warmup_epochs"),
        early_stopping_patience=_integer(
            training_raw["early_stopping_patience"],
            "training.early_stopping_patience",
            minimum=1,
        ),
        scheduled_sampling_start_epoch=start,
        scheduled_sampling_end_epoch=end,
        predicted_state_probability_max=probability,
        rollout_solver_steps=_integer(
            training_raw["rollout_solver_steps"], "training.rollout_solver_steps", minimum=1
        ),
        evaluation_solver_steps=_integer(
            training_raw["evaluation_solver_steps"],
            "training.evaluation_solver_steps",
            minimum=1,
        ),
    )

    trajectory_fields = {
        "token_dim",
        "hidden_dim",
        "layers",
        "heads",
        "ffn_dim",
        "dropout",
        "cv_folds",
        "batch_size",
        "max_epochs",
        "patience",
        "learning_rate",
        "weight_decay",
    }
    trajectory_raw = _mapping(root["trajectory"], "trajectory", trajectory_fields)
    trajectory_dropout = _number(trajectory_raw["dropout"], "trajectory.dropout")
    if trajectory_dropout >= 1.0:
        raise ValueError("latent-world v4 trajectory dropout must be below one")
    hidden_dim = _integer(trajectory_raw["hidden_dim"], "trajectory.hidden_dim", minimum=1)
    heads = _integer(trajectory_raw["heads"], "trajectory.heads", minimum=1)
    if hidden_dim % heads:
        raise ValueError("latent-world v4 trajectory heads must divide hidden dim")
    trajectory = V4TrajectoryConfig(
        token_dim=_integer(trajectory_raw["token_dim"], "trajectory.token_dim", minimum=1),
        hidden_dim=hidden_dim,
        layers=_integer(trajectory_raw["layers"], "trajectory.layers", minimum=1),
        heads=heads,
        ffn_dim=_integer(trajectory_raw["ffn_dim"], "trajectory.ffn_dim", minimum=1),
        dropout=trajectory_dropout,
        cv_folds=_integer(trajectory_raw["cv_folds"], "trajectory.cv_folds", minimum=2),
        batch_size=_integer(trajectory_raw["batch_size"], "trajectory.batch_size", minimum=1),
        max_epochs=_integer(trajectory_raw["max_epochs"], "trajectory.max_epochs", minimum=1),
        patience=_integer(trajectory_raw["patience"], "trajectory.patience", minimum=1),
        learning_rate=_number(
            trajectory_raw["learning_rate"], "trajectory.learning_rate", minimum=1e-12
        ),
        weight_decay=_number(trajectory_raw["weight_decay"], "trajectory.weight_decay"),
    )

    runtime_raw = _mapping(root["runtime"], "runtime", {"output_root", "seed"})
    return LatentWorldV4ExperimentConfig(
        path=config_path,
        base_config_path=base_path,
        base_config_sha256=base_sha256,
        base=base,
        conditioning=conditioning,
        model=model,
        losses=losses,
        training=training,
        trajectory=trajectory,
        output_root=_path(runtime_raw["output_root"], "runtime.output_root"),
        seed=_integer(runtime_raw["seed"], "runtime.seed"),
        raw=root,
    )


__all__ = [
    "CLIP_MODEL_ID",
    "CLIP_REVISION",
    "LATENT_WORLD_V4_ARCHITECTURE",
    "LATENT_WORLD_V4_CHECKPOINT_SCHEMA",
    "LATENT_WORLD_V4_CONFIG_SCHEMA",
    "MEDGEMMA_MODEL_ID",
    "MEDGEMMA_REVISION",
    "V4_CONDITIONING_MODES",
    "LatentWorldV4ExperimentConfig",
    "V4ConditioningConfig",
    "V4LossConfig",
    "V4ModelConfig",
    "V4TextModelConfig",
    "V4TrainingConfig",
    "V4TrajectoryConfig",
    "load_latent_world_v4_config",
]
