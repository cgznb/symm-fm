from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .contracts import (
    ANCESTRAL_DDPM_SAMPLER,
    CONTINUOUS_CODEBOOK_MINMAX_CONTRACT,
    CONTINUOUS_TRAIN_CHANNEL_ZSCORE_CONTRACT,
    CT_DENOISER_ARCHITECTURE,
    EPSILON_PREDICTION_TYPE,
    FILM_DENOISER_ARCHITECTURE,
    PAPER_FAITHFUL_ARCHITECTURE,
    REGISTERED_LARGE_PAPER_ARCHITECTURE,
    X0_PREDICTION_TYPE,
)
from .latent_statistics import LATENT_CHANNEL_STATISTICS_SCHEMA
from .manifest import ACCEPTED_BACKENDS
from .paper_contracts import (
    PaperRuntimeContract,
    RegisteredLargePaperRuntimeContract,
    default_paper_runtime_contract,
    paper_contract_payload,
    registered_large_paper_contract_payload,
    registered_large_paper_runtime_contract,
)


MODEL_CONDITIONS = frozenset(
    {
        "source_dce0",
        "source_mask",
        "action_text",
        "clinical_text",
        "delta_days",
        "stage_id",
    }
)


@dataclass(frozen=True)
class DataConfig:
    backend: str
    bundle_json: Path
    phase_manifest_csv: Path
    cache_dir: Path
    roi_cache_dir: Path | None
    output_shape_zyx: tuple[int, int, int]
    model_conditions: tuple[str, ...]


@dataclass(frozen=True)
class LatentStatisticsConfig:
    path: Path
    sha256: str
    schema: str = LATENT_CHANNEL_STATISTICS_SCHEMA


@dataclass(frozen=True)
class DiffusionRuntimeConfig:
    denoiser_architecture: str
    input_channels: int
    semantic_channels: int
    prediction_type: str
    samples_per_transition: int
    timesteps: int = 200
    sampler: str = ANCESTRAL_DDPM_SAMPLER
    latent_contract: str = CONTINUOUS_CODEBOOK_MINMAX_CONTRACT
    ema_decay: float = 0.995
    paper: PaperRuntimeContract | RegisteredLargePaperRuntimeContract | None = None
    x0_objective: str | None = None
    latent_statistics: LatentStatisticsConfig | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    smoke: bool
    output_dir: Path
    seed: int = 2026


@dataclass(frozen=True)
class VQGANTrainingConfig:
    generator_learning_rate: float = 1.875e-5
    discriminator_learning_rate: float = 9.375e-6
    batch_size: int = 2
    precision: str = "16-mixed"
    gradient_clip_val: float = 1.0
    adversarial_always_on: bool = False
    image_gan_weight: float = 0.1
    volume_gan_weight: float = 0.1
    feature_matching_weight: float = 1.0
    early_stopping_patience: int = 8
    init_mri_checkpoint_sha256: str | None = None


@dataclass(frozen=True)
class DiffusionTrainingConfig:
    batch_size: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    data: DataConfig
    diffusion: DiffusionRuntimeConfig
    runtime: RuntimeConfig
    vqgan_training: VQGANTrainingConfig
    diffusion_training: DiffusionTrainingConfig
    raw: dict[str, Any]


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"config section {section} must be a mapping")
    return value


def _paper_field_json(value: Any, field_name: str) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"config diffusion paper field {field_name} is invalid"
        ) from error


def _paper_fixed_integer(
    diffusion: dict[str, Any], field_name: str, expected: int
) -> int:
    value = diffusion.get(field_name)
    if type(value) is not int or value <= 0 or value != expected:
        raise ValueError(
            f"config diffusion paper field {field_name} must be integer {expected}"
        )
    return value


def _paper_fixed_float(
    diffusion: dict[str, Any], field_name: str, expected: float
) -> float:
    value = diffusion.get(field_name)
    if (
        type(value) is not float
        or not math.isfinite(value)
        or not 0.0 < value < 1.0
        or value != expected
    ):
        raise ValueError(
            f"config diffusion paper field {field_name} must be finite {expected}"
        )
    return value


def _parse_latent_statistics(payload: Any) -> LatentStatisticsConfig:
    value = _mapping(payload, "diffusion.latent_statistics")
    if set(value) != {"path", "sha256", "schema"}:
        raise ValueError("config diffusion latent_statistics fields are invalid")
    path = Path(str(value["path"])).expanduser()
    sha256 = value["sha256"]
    schema = value["schema"]
    if (
        type(sha256) is not str
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("config diffusion latent statistics SHA256 is invalid")
    if schema != LATENT_CHANNEL_STATISTICS_SCHEMA:
        raise ValueError("config diffusion latent statistics schema is unsupported")
    return LatentStatisticsConfig(path=path, sha256=sha256, schema=schema)


def _parse_paper_contract(payload: Any) -> PaperRuntimeContract:
    paper = _mapping(payload, "diffusion.paper")
    default = default_paper_runtime_contract()
    expected = paper_contract_payload(default)
    for field_name, expected_value in expected.items():
        if field_name == "epsilon_objective":
            continue
        if field_name not in paper or _paper_field_json(
            paper[field_name], field_name
        ) != _paper_field_json(expected_value, field_name):
            raise ValueError(f"config diffusion paper field {field_name} is not locked")
    extra_fields = set(paper).difference(expected)
    if extra_fields:
        field_name = sorted(extra_fields, key=str)[0]
        raise ValueError(f"config diffusion paper field {field_name} is unsupported")
    objective = paper.get("epsilon_objective")
    if not isinstance(objective, str) or objective not in {"l1", "l2"}:
        raise ValueError("config diffusion paper field epsilon_objective must be l1 or l2")
    return replace(default, epsilon_objective=objective)


def _parse_registered_large_paper_contract(
    payload: Any,
) -> RegisteredLargePaperRuntimeContract:
    paper = _mapping(payload, "diffusion.paper")
    vqgan_sha256 = paper.get("vqgan_sha256")
    if type(vqgan_sha256) is not str:
        raise TypeError(
            "config diffusion paper field vqgan_sha256 must be an exact string"
        )
    default = registered_large_paper_runtime_contract(
        vqgan_sha256=vqgan_sha256
    )
    expected = registered_large_paper_contract_payload(default)
    for field_name, expected_value in expected.items():
        if field_name not in paper or _paper_field_json(
            paper[field_name], field_name
        ) != _paper_field_json(expected_value, field_name):
            raise ValueError(
                f"config diffusion registered paper field {field_name} is not locked"
            )
    extra_fields = set(paper).difference(expected)
    if extra_fields:
        field_name = sorted(extra_fields, key=str)[0]
        raise ValueError(
            f"config diffusion registered paper field {field_name} is unsupported"
        )
    return default


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    payload = _release_yaml(config_path.read_text())
    root = _mapping(payload, "root")
    data = _mapping(root.get("data"), "data")
    diffusion = _mapping(root.get("diffusion"), "diffusion")
    runtime = _mapping(root.get("runtime"), "runtime")
    vqgan_training = _mapping(root.get("vqgan_training", {}), "vqgan_training")
    diffusion_training = _mapping(
        root.get("diffusion_training", {}), "diffusion_training"
    )
    backend = str(data.get("backend", ""))
    if backend not in ACCEPTED_BACKENDS:
        raise ValueError("config data backend is invalid")
    shape = tuple(int(value) for value in data.get("output_shape_zyx", ()))
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("config output shape must contain three positive values")
    if shape == (96, 256, 256):
        bundle_path = Path(str(data.get("bundle_json", ""))).expanduser()
        try:
            bundle_schema = json.loads(bundle_path.read_text()).get("schema_version")
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            bundle_schema = None
        if (
            backend != "registered_t0"
            or bundle_schema != "motfm_ispy2_registered_strict_a_bundle_v1"
            or data.get("roi_cache_dir") is None
        ):
            raise ValueError(
                "96x256x256 is only valid for the registered Strict-A ROI cache"
            )
    elif shape != (128, 128, 128):
        raise ValueError(
            "I-SPY2 configs require 128x128x128 or the registered Strict-A shape"
        )
    conditions = tuple(str(value) for value in data.get("model_conditions", ()))
    if set(conditions) != MODEL_CONDITIONS or len(conditions) != len(MODEL_CONDITIONS):
        raise ValueError("config model condition contract is not the six allowed source fields")
    denoiser_architecture = str(diffusion.get("denoiser_architecture", ""))
    prediction_type = str(diffusion.get("prediction_type", ""))
    paper_architectures = {
        PAPER_FAITHFUL_ARCHITECTURE,
        REGISTERED_LARGE_PAPER_ARCHITECTURE,
    }
    if denoiser_architecture in paper_architectures:
        input_channels = _paper_fixed_integer(diffusion, "input_channels", 49)
        semantic_channels = _paper_fixed_integer(diffusion, "semantic_channels", 32)
        sample_count = _paper_fixed_integer(diffusion, "samples_per_transition", 8)
        timesteps = _paper_fixed_integer(diffusion, "timesteps", 200)
        ema_decay = _paper_fixed_float(diffusion, "ema_decay", 0.995)
    else:
        input_channels = int(diffusion.get("input_channels", 0))
        semantic_channels = int(diffusion.get("semantic_channels", -1))
        sample_count = int(diffusion.get("samples_per_transition", 0))
        timesteps = int(diffusion.get("timesteps", 200))
        ema_decay = float(diffusion.get("ema_decay", 0.995))
    expected_channels = {
        FILM_DENOISER_ARCHITECTURE: (17, 0),
        CT_DENOISER_ARCHITECTURE: (49, 32),
        PAPER_FAITHFUL_ARCHITECTURE: (49, 32),
        REGISTERED_LARGE_PAPER_ARCHITECTURE: (49, 32),
    }
    if expected_channels.get(denoiser_architecture) != (
        input_channels,
        semantic_channels,
    ):
        raise ValueError("config denoiser architecture and channel contract mismatch")
    if prediction_type not in {EPSILON_PREDICTION_TYPE, X0_PREDICTION_TYPE}:
        raise ValueError("config diffusion prediction type must be epsilon or x0")
    if (
        prediction_type == X0_PREDICTION_TYPE
        and denoiser_architecture != REGISTERED_LARGE_PAPER_ARCHITECTURE
    ):
        raise ValueError("x0 prediction requires registered-large paper diffusion")
    if sample_count != 8:
        raise ValueError("config diffusion contract must use eight samples")
    if denoiser_architecture == PAPER_FAITHFUL_ARCHITECTURE:
        paper_contract = _parse_paper_contract(diffusion.get("paper"))
    elif denoiser_architecture == REGISTERED_LARGE_PAPER_ARCHITECTURE:
        paper_contract = _parse_registered_large_paper_contract(
            diffusion.get("paper")
        )
    elif "paper" in diffusion:
        raise ValueError("config diffusion paper is only valid for the paper architecture")
    else:
        paper_contract = None
    if "ddim_steps" in diffusion:
        raise ValueError("DDIM configuration is retired; use ancestral DDPM")
    sampler = str(diffusion.get("sampler", ""))
    if sampler != ANCESTRAL_DDPM_SAMPLER:
        raise ValueError("config diffusion sampler must be ancestral_ddpm")
    latent_contract = str(diffusion.get("latent_contract", ""))
    expected_latent_contract = (
        CONTINUOUS_TRAIN_CHANNEL_ZSCORE_CONTRACT
        if prediction_type == X0_PREDICTION_TYPE
        else CONTINUOUS_CODEBOOK_MINMAX_CONTRACT
    )
    if latent_contract != expected_latent_contract:
        raise ValueError(
            "config diffusion latent contract does not match prediction type"
        )
    x0_objective = diffusion.get("x0_objective")
    latent_statistics = diffusion.get("latent_statistics")
    if prediction_type == X0_PREDICTION_TYPE:
        if x0_objective != "l2":
            raise ValueError("config diffusion x0 objective must be l2")
        latent_statistics_config = _parse_latent_statistics(latent_statistics)
    else:
        if x0_objective is not None or latent_statistics is not None:
            raise ValueError("epsilon diffusion cannot define x0-only fields")
        latent_statistics_config = None
    data_config = DataConfig(
        backend=backend,
        bundle_json=Path(str(data["bundle_json"])).expanduser(),
        phase_manifest_csv=Path(str(data["phase_manifest_csv"])).expanduser(),
        cache_dir=Path(str(data["cache_dir"])).expanduser(),
        roi_cache_dir=(
            Path(str(data["roi_cache_dir"])).expanduser()
            if data.get("roi_cache_dir") is not None
            else None
        ),
        output_shape_zyx=shape,
        model_conditions=conditions,
    )
    diffusion_config = DiffusionRuntimeConfig(
        denoiser_architecture=denoiser_architecture,
        input_channels=input_channels,
        semantic_channels=semantic_channels,
        prediction_type=prediction_type,
        samples_per_transition=sample_count,
        timesteps=timesteps,
        sampler=sampler,
        latent_contract=latent_contract,
        ema_decay=ema_decay,
        paper=paper_contract,
        x0_objective=x0_objective,
        latent_statistics=latent_statistics_config,
    )
    runtime_config = RuntimeConfig(
        smoke=bool(runtime.get("smoke", False)),
        output_dir=Path(str(runtime.get("output_dir", "runs/mewm_ispy2"))).expanduser(),
        seed=int(runtime.get("seed", 2026)),
    )
    vqgan_training_config = VQGANTrainingConfig(
        generator_learning_rate=float(
            vqgan_training.get("generator_learning_rate", 1.875e-5)
        ),
        discriminator_learning_rate=float(
            vqgan_training.get("discriminator_learning_rate", 9.375e-6)
        ),
        batch_size=int(vqgan_training.get("batch_size", 2)),
        precision=str(vqgan_training.get("precision", "16-mixed")),
        gradient_clip_val=float(vqgan_training.get("gradient_clip_val", 1.0)),
        adversarial_always_on=bool(
            vqgan_training.get("adversarial_always_on", False)
        ),
        image_gan_weight=float(vqgan_training.get("image_gan_weight", 0.1)),
        volume_gan_weight=float(vqgan_training.get("volume_gan_weight", 0.1)),
        feature_matching_weight=float(
            vqgan_training.get("feature_matching_weight", 1.0)
        ),
        early_stopping_patience=int(
            vqgan_training.get("early_stopping_patience", 8)
        ),
        init_mri_checkpoint_sha256=(
            str(vqgan_training["init_mri_checkpoint_sha256"])
            if vqgan_training.get("init_mri_checkpoint_sha256") is not None
            else None
        ),
    )
    if vqgan_training_config.batch_size <= 0:
        raise ValueError("VQGAN batch size must be positive")
    if vqgan_training_config.precision != "16-mixed":
        raise ValueError("VQGAN CUDA precision must be 16-mixed")
    if vqgan_training_config.gradient_clip_val <= 0:
        raise ValueError("VQGAN gradient clip must be positive")
    if vqgan_training_config.early_stopping_patience < 0:
        raise ValueError("VQGAN early stopping patience cannot be negative")
    expected_init_sha = vqgan_training_config.init_mri_checkpoint_sha256
    if expected_init_sha is not None and (
        len(expected_init_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_init_sha)
    ):
        raise ValueError("VQGAN MRI initialization SHA256 is invalid")
    expected_diffusion_batch_size = (
        12 if denoiser_architecture == REGISTERED_LARGE_PAPER_ARCHITECTURE else 1
    )
    if (
        denoiser_architecture == REGISTERED_LARGE_PAPER_ARCHITECTURE
        and prediction_type == EPSILON_PREDICTION_TYPE
        and "batch_size" not in diffusion_training
    ):
        raise ValueError(
            "registered-large config diffusion_training.batch_size must be twelve"
        )
    diffusion_batch_size = diffusion_training.get(
        "batch_size", expected_diffusion_batch_size
    )
    if type(diffusion_batch_size) is not int or diffusion_batch_size <= 0:
        raise ValueError("config diffusion_training.batch_size must be positive")
    if (
        prediction_type != X0_PREDICTION_TYPE
        and diffusion_batch_size != expected_diffusion_batch_size
    ):
        raise ValueError(
            f"config diffusion_training.batch_size must be {expected_diffusion_batch_size}"
        )
    diffusion_training_config = DiffusionTrainingConfig(
        batch_size=diffusion_batch_size
    )
    return ExperimentConfig(
        path=config_path,
        data=data_config,
        diffusion=diffusion_config,
        runtime=runtime_config,
        vqgan_training=vqgan_training_config,
        diffusion_training=diffusion_training_config,
        raw=root,
    )
