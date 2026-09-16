from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import copy
import json
from argparse import Namespace
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from .backend import (
    REGISTERED_STRICT_A_BUNDLE_SCHEMA,
    LoadedSplitVisits,
    LoadedTransitions,
    load_source_inference_records,
    load_split_visit_records,
    load_transition_records,
    sha256_file,
)
from .cache import (
    DCE0Cache,
    MOTFMROICache,
    RegisteredPairCache,
    RegisteredStrictAROICache,
)
from .checkpoint import (
    CheckpointIdentity,
    load_diffusion_checkpoint,
)
from .conditioning import (
    MEDGEMMA_MODEL_ID,
    MEDGEMMA_REVISION,
    ISPY2Conditioner,
    SharedMedGemmaTower,
)
from .config import DiffusionRuntimeConfig, ExperimentConfig, load_experiment_config
from .contracts import (
    CT_DENOISER_ARCHITECTURE,
    EPSILON_PREDICTION_TYPE,
    PAPER_FAITHFUL_ARCHITECTURE,
    REGISTERED_LARGE_PAPER_ARCHITECTURE,
    X0_PREDICTION_TYPE,
)
from .data import DCE0TransitionDataset, PreparedVisit, collate_transitions
from .diffusion import (
    ConditionalLatentDiffusion,
    DiffusionConfig,
    DiffusionTrainingSystem,
    build_denoiser,
)
from .inference import (
    assert_prediction_complete,
    publish_prediction,
    validate_prediction_identity,
)
from .metrics import evaluate_samples
from .perceptual import UncheckedLPIPSLoss
from .paper_attenuation import MorphoGaussianAttenuator
from .paper_ccl import (
    CCLPairIndex,
    PaperAnchorTransitionDataset,
    PaperCCLTransitionDataset,
    collate_paper_anchor,
    collate_paper_ccl,
)
from .paper_checkpoint import (
    PaperCheckpointIdentity,
    _validate_identity_match,
    load_paper_checkpoint,
    peek_paper_checkpoint_identity,
)
from .paper_conditioning import FrozenCLIPTextTower, PaperActionConditioner
from .paper_contracts import (
    AttenuationLevelConfig,
    PaperRuntimeContract,
    default_paper_runtime_contract,
    paper_contract_payload,
    paper_contract_sha256,
    registered_large_paper_contract_payload,
    registered_large_paper_contract_sha256,
)
from .paper_denoiser import (
    PaperFaithfulDenoiser3D,
    PaperRegisteredLargeDenoiser3D,
)
from .paper_diffusion import (
    PaperDiffusionConfig,
    PaperDiffusionTrainingSystem,
    PaperFaithfulLatentDiffusion,
    RegisteredLargePaperDiffusionTrainingSystem,
    RegisteredX0PaperDiffusionTrainingSystem,
)
from .latent_statistics import LatentChannelStatistics, load_latent_statistics
from .paper_warm_start import import_v4_paper_warm_start
from .registered_large_checkpoint import (
    REGISTERED_LARGE_ACCUMULATE_GRAD_BATCHES,
    REGISTERED_LARGE_NORMALIZATION_SHA256,
    REGISTERED_LARGE_PHYSICAL_BATCH_SIZE,
    RegisteredLargeCheckpointIdentity,
    _validate_identity_match as _validate_registered_large_identity_match,
    load_registered_large_checkpoint,
    peek_registered_large_checkpoint_identity,
)
from .registered_x0_checkpoint import (
    REGISTERED_X0_ACCUMULATE_GRAD_BATCHES,
    RegisteredX0CheckpointIdentity,
    _validate_identity_match as _validate_registered_x0_identity_match,
    load_registered_x0_checkpoint,
    peek_registered_x0_checkpoint_identity,
)
from .preflight import CT_EXPECTED_SHA256
from .segmenter import (
    DynUNetSegmentationSystem,
    build_dynunet,
    segment_generated_samples,
)
from .vqgan import (
    MRILevelVQGAN,
    REGISTERED_LARGE_VQGAN_CONFIG,
    REGISTERED_VQGAN_NUMERIC_CONTRACT,
    VQGAN_NUMERIC_CONTRACT,
    VQGANConfig,
    VQGANTrainingSystem,
    WIDTH_COMPATIBLE_MRI_INITIALIZATION,
    load_ct_autoencoder_weights,
    load_mri_training_weights,
    load_width_compatible_mri_codebook_discriminators,
    validate_width_compatible_initialization_report,
    vqgan_config_from_checkpoint_identity,
)

if TYPE_CHECKING:
    from .paper_ccl import PaperAnchorTransitionDataset, PaperCCLTransitionDataset


VQGAN_BATCH_SIZE = 2
REGISTERED_LARGE_VQGAN_SOURCE_CHECKPOINT = Path(
    _release_path('@artifacts/mewm/runs/current_dce0_ct_aligned_weak_gan_deep_b1/vqgan/checkpoints/composite/best-composite-42-50654.ckpt')
)
REGISTERED_LARGE_VQGAN_SOURCE_CHECKPOINT_SHA256 = (
    "ff639cb5b456e531d20c1a67b0e3cec47b43cbdc12f38348034b7daed764f33c"
)
REGISTERED_LARGE_VQGAN_INITIALIZATION_REPORT_DIGEST = (
    "69b5821636e9cd93cbdf149a0919c201dfdc1f23eab4427dd6d50bf9b2905f40"
)


class PaperDatasetEpochCallback(Callback):
    def __init__(
        self,
        train_dataset: PaperCCLTransitionDataset | PaperAnchorTransitionDataset,
        validation_dataset: PaperCCLTransitionDataset
        | PaperAnchorTransitionDataset
        | None = None,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        if validation_dataset is not None:
            validation_dataset.set_epoch(0)

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self.train_dataset.set_epoch(int(trainer.current_epoch))


class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, *, minimum_global_step: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.minimum_global_step = int(minimum_global_step)

    def should_check(self, trainer: Any) -> bool:
        return int(trainer.global_step) >= self.minimum_global_step

    def _run_early_stopping_check(self, trainer: Any) -> None:
        if self.should_check(trainer):
            super()._run_early_stopping_check(trainer)


class CachedVisitDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        visit_ids: Sequence[str],
        loader: Any,
    ) -> None:
        self.visit_ids = tuple(visit_ids)
        self.loader = loader

    def __len__(self) -> int:
        return len(self.visit_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        visit: PreparedVisit = self.loader(self.visit_ids[index])
        return {"image": visit.image.float(), "mask": visit.mask.float()}


def _vqgan_dataloaders(
    train: Dataset[Any],
    val: Dataset[Any],
    *,
    batch_size: int = VQGAN_BATCH_SIZE,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        ),
        DataLoader(val, batch_size=batch_size, num_workers=2),
    )


def _vqgan_callbacks(
    output: Path,
    *,
    adversarial_full_step: int,
    early_stopping_patience: int = 8,
) -> list[Callback]:
    checkpoint_root = output / "checkpoints"
    return [
        ModelCheckpoint(
            dirpath=checkpoint_root / "reconstruction",
            filename="best-reconstruction-{epoch:02d}-{step}",
            auto_insert_metric_name=False,
            monitor="val/reconstruction",
            mode="min",
            save_last=False,
            save_top_k=1,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_root / "composite",
            filename="best-composite-{epoch:02d}-{step}",
            auto_insert_metric_name=False,
            monitor="val/composite",
            mode="min",
            save_last=False,
            save_top_k=1,
        ),
        DelayedEarlyStopping(
            minimum_global_step=adversarial_full_step,
            monitor="val/composite",
            mode="min",
            patience=early_stopping_patience,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_root / "reconstruction",
            save_last=True,
            save_top_k=0,
        ),
    ]


def _load_data(config: ExperimentConfig) -> LoadedTransitions:
    return load_transition_records(
        config.data.bundle_json,
        config.data.phase_manifest_csv,
        backend=config.data.backend,
    )


def _build_diffusion_model(
    vqgan: MRILevelVQGAN,
    conditioner: torch.nn.Module,
    runtime: DiffusionRuntimeConfig,
) -> ConditionalLatentDiffusion:
    model_config = DiffusionConfig(
        denoiser_architecture=runtime.denoiser_architecture,
        timesteps=runtime.timesteps,
        sampler=runtime.sampler,
        latent_contract=runtime.latent_contract,
        ema_decay=runtime.ema_decay,
        denoiser_input_channels=runtime.input_channels,
        semantic_channels=runtime.semantic_channels,
        prediction_type=runtime.prediction_type,
    )
    return ConditionalLatentDiffusion(
        vqgan,
        conditioner,
        build_denoiser(runtime.denoiser_architecture),
        model_config,
    )


def _diffusion_checkpoint_identity(
    model: ConditionalLatentDiffusion,
    *,
    vqgan_sha256: str,
    loaded: LoadedTransitions,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        medgemma_model_id=MEDGEMMA_MODEL_ID,
        medgemma_revision=MEDGEMMA_REVISION,
        vqgan_sha256=vqgan_sha256,
        data_contract_sha256=loaded.data_contract_sha256,
        data_backend=loaded.backend,
        denoiser_architecture=model.config.denoiser_architecture,
        denoiser_input_channels=model.config.denoiser_input_channels,
        semantic_channels=model.config.semantic_channels,
        prediction_type=model.config.prediction_type,
        timesteps=model.config.timesteps,
        sampler=model.config.sampler,
        latent_contract=model.config.latent_contract,
        ema_decay=model.config.ema_decay,
    )


def _build_paper_diffusion_model(
    vqgan: MRILevelVQGAN,
    conditioner: PaperActionConditioner,
    runtime: DiffusionRuntimeConfig,
    *,
    latent_statistics: LatentChannelStatistics | None = None,
) -> PaperFaithfulLatentDiffusion:
    if runtime.paper is None:
        raise ValueError("paper diffusion runtime contract is required")
    config = PaperDiffusionConfig(
        denoiser_architecture=runtime.denoiser_architecture,
        timesteps=runtime.timesteps,
        sampler=runtime.sampler,
        latent_contract=runtime.latent_contract,
        ema_decay=runtime.ema_decay,
        denoiser_input_channels=runtime.input_channels,
        semantic_channels=runtime.semantic_channels,
        prediction_type=runtime.prediction_type,
        x0_objective=getattr(runtime, "x0_objective", None),
        latent_statistics_sha256=(
            getattr(runtime, "latent_statistics").sha256
            if getattr(runtime, "latent_statistics", None) is not None
            else None
        ),
        runtime=runtime.paper,
    )
    denoiser: PaperFaithfulDenoiser3D
    if runtime.denoiser_architecture == REGISTERED_LARGE_PAPER_ARCHITECTURE:
        denoiser = PaperRegisteredLargeDenoiser3D()
    else:
        denoiser = PaperFaithfulDenoiser3D()
    return PaperFaithfulLatentDiffusion(
        vqgan,
        conditioner,
        denoiser,
        MorphoGaussianAttenuator(runtime.paper),
        config,
        latent_statistics=latent_statistics,
    )


def _load_runtime_latent_statistics(
    runtime: DiffusionRuntimeConfig,
    *,
    vqgan_sha256: str,
    data_contract_sha256: str,
) -> LatentChannelStatistics | None:
    descriptor = getattr(runtime, "latent_statistics", None)
    prediction_type = getattr(runtime, "prediction_type", EPSILON_PREDICTION_TYPE)
    if prediction_type == EPSILON_PREDICTION_TYPE:
        if descriptor is not None:
            raise ValueError("epsilon diffusion cannot load latent statistics")
        return None
    if prediction_type != X0_PREDICTION_TYPE or descriptor is None:
        raise ValueError("x0 diffusion requires bound latent statistics")
    return load_latent_statistics(
        descriptor.path,
        expected_sha256=descriptor.sha256,
        expected_vqgan_sha256=vqgan_sha256,
        expected_data_contract_sha256=data_contract_sha256,
    )


def _paper_checkpoint_identity(
    model: PaperFaithfulLatentDiffusion,
    *,
    vqgan_sha256: str,
    loaded: LoadedTransitions,
) -> PaperCheckpointIdentity:
    config = model.config
    return PaperCheckpointIdentity(
        denoiser_architecture=config.denoiser_architecture,
        vqgan_sha256=vqgan_sha256,
        data_contract_sha256=loaded.data_contract_sha256,
        data_backend=loaded.backend,
        denoiser_input_channels=config.denoiser_input_channels,
        semantic_channels=config.semantic_channels,
        prediction_type=config.prediction_type,
        timesteps=config.timesteps,
        sampler=config.sampler,
        latent_contract=config.latent_contract,
        ema_decay=config.ema_decay,
        paper_contract=paper_contract_payload(config.runtime),
        paper_contract_sha256=paper_contract_sha256(config.runtime),
    )


def _registered_large_checkpoint_identity_from_runtime(
    runtime: DiffusionRuntimeConfig,
    *,
    vqgan_sha256: str,
    loaded: LoadedTransitions,
) -> RegisteredLargeCheckpointIdentity:
    contract = getattr(runtime, "paper", None)
    if contract is None:
        contract = getattr(runtime, "runtime", None)
    if contract is None:
        raise ValueError("registered-large runtime contract is required")
    return RegisteredLargeCheckpointIdentity(
        denoiser_architecture=runtime.denoiser_architecture,
        denoiser_widths=(32, 64, 128, 256),
        input_shape_zyx=(96, 256, 256),
        latent_shape_czyx=(8, 24, 64, 64),
        vqgan_sha256=vqgan_sha256,
        vqgan_numeric_contract=contract.vqgan_numeric_contract,
        data_contract_sha256=loaded.data_contract_sha256,
        data_backend=loaded.backend,
        bundle_contract_sha256=contract.bundle_contract_sha256,
        phase_manifest_sha256=contract.phase_manifest_sha256,
        normalization_sha256=REGISTERED_LARGE_NORMALIZATION_SHA256,
        denoiser_input_channels=getattr(
            runtime,
            "input_channels",
            getattr(runtime, "denoiser_input_channels", None),
        ),
        semantic_channels=runtime.semantic_channels,
        prediction_type=runtime.prediction_type,
        timesteps=runtime.timesteps,
        sampler=runtime.sampler,
        latent_contract=runtime.latent_contract,
        ema_decay=runtime.ema_decay,
        paper_contract=registered_large_paper_contract_payload(contract),
        paper_contract_sha256=registered_large_paper_contract_sha256(contract),
        initialization_method=contract.initialization_method,
        warm_start_source=None,
    )


def _registered_large_checkpoint_identity(
    model: PaperFaithfulLatentDiffusion,
    *,
    vqgan_sha256: str,
    loaded: LoadedTransitions,
) -> RegisteredLargeCheckpointIdentity:
    identity = _registered_large_checkpoint_identity_from_runtime(
        model.config,
        vqgan_sha256=vqgan_sha256,
        loaded=loaded,
    )
    if getattr(model.denoiser, "channels", None) != identity.denoiser_widths:
        raise ValueError("registered-large denoiser widths do not match v6 identity")
    return identity


def _registered_x0_checkpoint_identity_from_runtime(
    runtime: DiffusionRuntimeConfig,
    *,
    vqgan_sha256: str,
    loaded: LoadedTransitions,
    physical_batch_size: int,
) -> RegisteredX0CheckpointIdentity:
    contract = getattr(runtime, "paper", None)
    if contract is None:
        contract = getattr(runtime, "runtime", None)
    statistics = getattr(runtime, "latent_statistics", None)
    if contract is None:
        raise ValueError("registered x0 runtime contract is required")
    if statistics is None:
        raise ValueError("registered x0 latent statistics are required")
    return RegisteredX0CheckpointIdentity(
        denoiser_architecture=runtime.denoiser_architecture,
        denoiser_widths=(32, 64, 128, 256),
        input_shape_zyx=(96, 256, 256),
        latent_shape_czyx=(8, 24, 64, 64),
        vqgan_sha256=vqgan_sha256,
        vqgan_numeric_contract=contract.vqgan_numeric_contract,
        data_contract_sha256=loaded.data_contract_sha256,
        data_backend=loaded.backend,
        bundle_contract_sha256=contract.bundle_contract_sha256,
        phase_manifest_sha256=contract.phase_manifest_sha256,
        normalization_sha256=REGISTERED_LARGE_NORMALIZATION_SHA256,
        latent_statistics_sha256=statistics.sha256,
        denoiser_input_channels=getattr(
            runtime,
            "input_channels",
            getattr(runtime, "denoiser_input_channels", None),
        ),
        semantic_channels=runtime.semantic_channels,
        prediction_type=runtime.prediction_type,
        x0_objective=runtime.x0_objective,
        timesteps=runtime.timesteps,
        sampler=runtime.sampler,
        latent_contract=runtime.latent_contract,
        ema_decay=runtime.ema_decay,
        paper_contract=registered_large_paper_contract_payload(contract),
        paper_contract_sha256=registered_large_paper_contract_sha256(contract),
        initialization_method=contract.initialization_method,
        warm_start_source=None,
        physical_batch_size=physical_batch_size,
        accumulate_grad_batches=REGISTERED_X0_ACCUMULATE_GRAD_BATCHES,
    )


def _registered_x0_checkpoint_identity(
    model: PaperFaithfulLatentDiffusion,
    *,
    runtime: DiffusionRuntimeConfig,
    vqgan_sha256: str,
    loaded: LoadedTransitions,
    physical_batch_size: int,
) -> RegisteredX0CheckpointIdentity:
    identity = _registered_x0_checkpoint_identity_from_runtime(
        runtime,
        vqgan_sha256=vqgan_sha256,
        loaded=loaded,
        physical_batch_size=physical_batch_size,
    )
    if getattr(model.denoiser, "channels", None) != identity.denoiser_widths:
        raise ValueError("registered x0 denoiser widths do not match v7 identity")
    for name in (
        "denoiser_architecture",
        "denoiser_input_channels",
        "semantic_channels",
        "prediction_type",
        "x0_objective",
        "timesteps",
        "sampler",
        "latent_contract",
        "ema_decay",
        "latent_statistics_sha256",
    ):
        if getattr(model.config, name) != getattr(identity, name):
            raise ValueError(f"registered x0 model does not match v7 identity: {name}")
    return identity


def _paper_runtime_from_checkpoint_identity(
    identity: PaperCheckpointIdentity,
) -> PaperRuntimeContract:
    if type(identity) is not PaperCheckpointIdentity:
        raise TypeError("identity must be an exact PaperCheckpointIdentity")
    payload = copy.deepcopy(identity.paper_contract)
    levels = payload.pop("attenuation_levels")
    payload["attenuation_levels"] = tuple(
        AttenuationLevelConfig(**copy.deepcopy(level)) for level in levels
    )
    payload["warm_start_path"] = Path(payload["warm_start_path"])
    runtime = PaperRuntimeContract(**payload)
    if paper_contract_payload(runtime) != identity.paper_contract:
        raise ValueError("checkpoint paper contract reconstruction is not exact")
    if paper_contract_sha256(runtime) != identity.paper_contract_sha256:
        raise ValueError("checkpoint paper contract SHA256 is not canonical")
    return runtime


def _validate_paper_inference_identity(
    runtime: DiffusionRuntimeConfig,
    loaded: LoadedTransitions,
    identity: PaperCheckpointIdentity,
    *,
    vqgan_sha256: str,
) -> PaperRuntimeContract:
    if type(runtime) is not DiffusionRuntimeConfig:
        raise TypeError("runtime must be an exact DiffusionRuntimeConfig")
    if runtime.paper is None:
        raise ValueError("paper inference requires a paper runtime contract")
    if type(runtime.paper) is not PaperRuntimeContract:
        raise TypeError("paper inference contract must be an exact PaperRuntimeContract")
    canonical_path_type = type(default_paper_runtime_contract().warm_start_path)
    if type(runtime.paper.warm_start_path) is not canonical_path_type:
        raise TypeError(
            "paper inference warm_start_path must have exact type "
            f"{canonical_path_type.__name__}"
        )
    if type(runtime.samples_per_transition) is not int:
        raise TypeError("paper inference samples_per_transition must be an exact integer")
    if runtime.samples_per_transition != 8:
        raise ValueError("paper inference samples_per_transition is locked to 8")
    for index, level in enumerate(runtime.paper.attenuation_levels):
        if type(level) is not AttenuationLevelConfig:
            raise TypeError(
                "paper inference attenuation level "
                f"{index} must be an exact AttenuationLevelConfig"
            )
    expected_identity = PaperCheckpointIdentity(
        denoiser_architecture=runtime.denoiser_architecture,
        vqgan_sha256=vqgan_sha256,
        data_contract_sha256=loaded.data_contract_sha256,
        data_backend=loaded.backend,
        denoiser_input_channels=runtime.input_channels,
        semantic_channels=runtime.semantic_channels,
        prediction_type=runtime.prediction_type,
        timesteps=runtime.timesteps,
        sampler=runtime.sampler,
        latent_contract=runtime.latent_contract,
        ema_decay=runtime.ema_decay,
        paper_contract=paper_contract_payload(runtime.paper),
        paper_contract_sha256=paper_contract_sha256(runtime.paper),
    )
    _validate_identity_match(identity, expected_identity)
    return _paper_runtime_from_checkpoint_identity(identity)


def load_paper_diffusion_for_inference(
    *,
    runtime: DiffusionRuntimeConfig,
    loaded: LoadedTransitions,
    vqgan_checkpoint_path: str | Path,
    diffusion_checkpoint_path: str | Path,
    vqgan_sha256: str | None = None,
    vqgan: MRILevelVQGAN | None = None,
    text_tower: torch.nn.Module | None = None,
) -> tuple[PaperFaithfulLatentDiffusion, PaperCheckpointIdentity, int]:
    identity = peek_paper_checkpoint_identity(diffusion_checkpoint_path)
    actual_vqgan_sha256 = (
        sha256_file(Path(vqgan_checkpoint_path))
        if vqgan_sha256 is None
        else vqgan_sha256
    )
    checkpoint_runtime = _validate_paper_inference_identity(
        runtime,
        loaded,
        identity,
        vqgan_sha256=actual_vqgan_sha256,
    )
    if vqgan is None:
        vqgan = load_mri_vqgan(
            vqgan_checkpoint_path,
            expected_data_contract_sha256=loaded.data_contract_sha256,
            expected_data_backend=loaded.backend,
            **_vqgan_load_contract_kwargs(loaded),
        )
    if text_tower is None:
        text_tower = FrozenCLIPTextTower.from_pretrained(
            model_id=checkpoint_runtime.clip_model_id,
            revision=checkpoint_runtime.clip_revision,
            local_files_only=True,
        )
    conditioner = PaperActionConditioner(text_tower, contract=checkpoint_runtime)
    checkpoint_bound_runtime = replace(runtime, paper=checkpoint_runtime)
    model = _build_paper_diffusion_model(vqgan, conditioner, checkpoint_bound_runtime)
    global_step = load_paper_checkpoint(model, diffusion_checkpoint_path, identity)
    return model, identity, global_step


def load_registered_paper_diffusion_for_inference(
    *,
    runtime: DiffusionRuntimeConfig,
    loaded: LoadedTransitions,
    vqgan_checkpoint_path: str | Path,
    diffusion_checkpoint_path: str | Path,
    physical_batch_size: int,
) -> tuple[
    PaperFaithfulLatentDiffusion,
    RegisteredLargeCheckpointIdentity | RegisteredX0CheckpointIdentity,
    int,
]:
    if runtime.denoiser_architecture != REGISTERED_LARGE_PAPER_ARCHITECTURE:
        raise ValueError("registered paper inference requires registered-large diffusion")
    contract = runtime.paper
    if contract is None:
        raise ValueError("registered paper inference requires its runtime contract")
    vqgan_path = Path(vqgan_checkpoint_path)
    vqgan_sha256 = sha256_file(vqgan_path)
    if vqgan_sha256 != contract.vqgan_sha256:
        raise ValueError("registered paper VQGAN does not match the pinned SHA256")
    vqgan = load_mri_vqgan(
        vqgan_path,
        expected_data_contract_sha256=loaded.data_contract_sha256,
        expected_data_backend=loaded.backend,
        **_vqgan_load_contract_kwargs(loaded),
    )
    tower = FrozenCLIPTextTower.from_pretrained(
        model_id=contract.clip_model_id,
        revision=contract.clip_revision,
        local_files_only=True,
    )
    conditioner = PaperActionConditioner(tower, contract=contract)
    statistics = _load_runtime_latent_statistics(
        runtime,
        vqgan_sha256=vqgan_sha256,
        data_contract_sha256=loaded.data_contract_sha256,
    )
    model = _build_paper_diffusion_model(
        vqgan,
        conditioner,
        runtime,
        latent_statistics=statistics,
    )
    if runtime.prediction_type == X0_PREDICTION_TYPE:
        identity = _registered_x0_checkpoint_identity(
            model,
            runtime=runtime,
            vqgan_sha256=vqgan_sha256,
            loaded=loaded,
            physical_batch_size=physical_batch_size,
        )
        global_step = load_registered_x0_checkpoint(
            model, diffusion_checkpoint_path, identity
        )
    else:
        identity = _registered_large_checkpoint_identity(
            model,
            vqgan_sha256=vqgan_sha256,
            loaded=loaded,
        )
        global_step = load_registered_large_checkpoint(
            model, diffusion_checkpoint_path, identity
        )
    return model, identity, global_step


def _is_registered_strict_a(loaded: Any) -> bool:
    return (
        getattr(loaded, "bundle_schema_version", None)
        == REGISTERED_STRICT_A_BUNDLE_SCHEMA
    )


def _vqgan_numeric_contract(loaded: Any) -> str:
    if _is_registered_strict_a(loaded):
        return REGISTERED_VQGAN_NUMERIC_CONTRACT
    return VQGAN_NUMERIC_CONTRACT


def _vqgan_load_contract_kwargs(loaded: Any) -> dict[str, str]:
    if _is_registered_strict_a(loaded):
        return {"expected_numeric_contract": REGISTERED_VQGAN_NUMERIC_CONTRACT}
    return {}


def _visit_loader(
    config: ExperimentConfig, loaded: LoadedTransitions | LoadedSplitVisits
) -> Any:
    if _is_registered_strict_a(loaded):
        if config.data.backend != "registered_t0":
            raise ValueError("registered Strict-A data requires registered_t0")
        if config.data.roi_cache_dir is None:
            raise ValueError(
                "registered Strict-A data requires its read-only ROI cache"
            )
        roi_cache = RegisteredStrictAROICache(
            config.data.roi_cache_dir,
            bundle_json=config.data.bundle_json,
            output_shape_zyx=config.data.output_shape_zyx,
        )
        return lambda visit_id: roi_cache.load(loaded.visits[visit_id])
    if config.data.roi_cache_dir is not None:
        if config.data.backend != "current":
            raise ValueError("MOTFM ROI cache is only valid for the current backend")
        roi_cache = MOTFMROICache(
            config.data.roi_cache_dir,
            bundle_json=config.data.bundle_json,
            output_shape_zyx=config.data.output_shape_zyx,
        )
        return lambda visit_id: roi_cache.load(loaded.visits[visit_id])
    cache = DCE0Cache(
        config.data.cache_dir, output_shape_zyx=config.data.output_shape_zyx
    )
    return lambda visit_id: cache.load_or_create(
        loaded.visits[visit_id], backend=config.data.backend
    )


def _transition_dataset(
    config: ExperimentConfig,
    loaded: LoadedTransitions,
    fold: str,
) -> DCE0TransitionDataset:
    records = [record for record in loaded.records if record.fold == fold]
    visit_loader = _visit_loader(config, loaded)
    pair_loader = None
    if (
        config.data.backend == "registered_t0"
        and not _is_registered_strict_a(loaded)
    ):
        pair_cache = RegisteredPairCache(
            config.data.cache_dir, output_shape_zyx=config.data.output_shape_zyx
        )
        pair_loader = lambda record: pair_cache.load_or_create(
            loaded.visits[record.source_visit_id], loaded.visits[record.target_visit_id]
        )
    return DCE0TransitionDataset(
        records,
        visit_loader=visit_loader,
        pair_loader=pair_loader,
        source_only=False,
        backend=config.data.backend,
    )


def _paper_transition_datasets(
    config: ExperimentConfig,
    loaded: LoadedTransitions,
) -> tuple[
    PaperCCLTransitionDataset | PaperAnchorTransitionDataset,
    PaperCCLTransitionDataset | PaperAnchorTransitionDataset,
]:
    visit_loader = _visit_loader(config, loaded)
    pair_loader = None
    if (
        config.data.backend == "registered_t0"
        and not _is_registered_strict_a(loaded)
    ):
        pair_cache = RegisteredPairCache(
            config.data.cache_dir, output_shape_zyx=config.data.output_shape_zyx
        )
        pair_loader = lambda record: pair_cache.load_or_create(
            loaded.visits[record.source_visit_id],
            loaded.visits[record.target_visit_id],
        )

    datasets: list[PaperCCLTransitionDataset | PaperAnchorTransitionDataset] = []
    for fold in ("train", "val"):
        records = tuple(record for record in loaded.records if record.fold == fold)
        if not records:
            raise ValueError(f"paper diffusion {fold} fold is empty")
        if (
            getattr(config.diffusion, "prediction_type", EPSILON_PREDICTION_TYPE)
            == X0_PREDICTION_TYPE
        ):
            dataset = PaperAnchorTransitionDataset(
                records,
                visit_loader=visit_loader,
                pair_loader=pair_loader,
                backend=config.data.backend,
            )
        else:
            pair_index = CCLPairIndex(records, seed=config.runtime.seed)
            dataset = PaperCCLTransitionDataset(
                records,
                visit_loader=visit_loader,
                pair_index=pair_index,
                pair_loader=pair_loader,
                backend=config.data.backend,
            )
        datasets.append(dataset)
    datasets[1].set_epoch(0)
    return datasets[0], datasets[1]


def _paper_dataloaders(
    train: PaperCCLTransitionDataset | PaperAnchorTransitionDataset,
    val: PaperCCLTransitionDataset | PaperAnchorTransitionDataset,
    *,
    batch_size: int = 1,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("paper diffusion batch size must be a positive integer")
    if isinstance(train, PaperAnchorTransitionDataset):
        if not isinstance(val, PaperAnchorTransitionDataset):
            raise TypeError("paper train and validation dataset modes must match")
        collate_fn = collate_paper_anchor
    else:
        if isinstance(train, PaperCCLTransitionDataset) != isinstance(
            val, PaperCCLTransitionDataset
        ):
            raise TypeError("paper train and validation dataset modes must match")
        collate_fn = collate_paper_ccl
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False,
        ),
        DataLoader(
            val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False,
        ),
    )


def _training_precision(
    stage: str,
    *,
    cuda_available: bool,
    vqgan_precision: str = "16-mixed",
    denoiser_architecture: str | None = None,
) -> str:
    if not cuda_available:
        return "32-true"
    if stage == "diffusion" and denoiser_architecture == CT_DENOISER_ARCHITECTURE:
        return "32-true"
    return vqgan_precision if stage == "vqgan" else "bf16-mixed"


def _diffusion_callbacks(
    output: Path,
    *,
    denoiser_architecture: str,
    prediction_type: str = EPSILON_PREDICTION_TYPE,
) -> list[Callback]:
    checkpoint_root = output / "checkpoints"
    if denoiser_architecture in {
        PAPER_FAITHFUL_ARCHITECTURE,
        REGISTERED_LARGE_PAPER_ARCHITECTURE,
    }:
        if prediction_type == X0_PREDICTION_TYPE:
            return [
                ModelCheckpoint(
                    dirpath=checkpoint_root / "total",
                    monitor="val/total_loss",
                    mode="min",
                    save_last=True,
                    save_top_k=1,
                ),
                ModelCheckpoint(
                    dirpath=checkpoint_root / "x0_mae",
                    monitor="val/x0_mae",
                    mode="min",
                    save_last=False,
                    save_top_k=1,
                ),
            ]
        return [
            ModelCheckpoint(
                dirpath=checkpoint_root / "total",
                monitor="val/total_loss",
                mode="min",
                save_last=True,
                save_top_k=1,
            ),
            ModelCheckpoint(
                dirpath=checkpoint_root / "epsilon",
                monitor="val/epsilon_mse",
                mode="min",
                save_last=False,
                save_top_k=1,
            ),
        ]
    if denoiser_architecture == CT_DENOISER_ARCHITECTURE:
        return [
            ModelCheckpoint(
                dirpath=checkpoint_root,
                filename="step-{step:07d}",
                auto_insert_metric_name=False,
                every_n_train_steps=200,
                save_last=True,
                save_top_k=-1,
            )
        ]
    return [
        ModelCheckpoint(
            dirpath=checkpoint_root,
            monitor="val/epsilon_l1",
            mode="min",
            save_last=True,
            save_top_k=1,
        )
    ]


def _trainer(
    args: Namespace,
    output: Path,
    callbacks: list[Callback],
    *,
    stage: str,
    vqgan_precision: str = "16-mixed",
    denoiser_architecture: str | None = None,
    accumulate_grad_batches: int = 1,
    gradient_clip_val: float | None = None,
) -> pl.Trainer:
    devices: str | int = args.devices
    if isinstance(devices, str) and devices.isdigit():
        devices = int(devices)
    trainer_options: dict[str, Any] = {}
    if gradient_clip_val is not None:
        trainer_options["gradient_clip_val"] = gradient_clip_val
    return pl.Trainer(
        accelerator="auto",
        devices=devices,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        default_root_dir=output,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=_training_precision(
            stage,
            cuda_available=torch.cuda.is_available(),
            vqgan_precision=vqgan_precision,
            denoiser_architecture=denoiser_architecture,
        ),
        log_every_n_steps=10,
        **trainer_options,
    )


def _unique_visit_ids(loaded: LoadedTransitions, fold: str) -> list[str]:
    result = {
        visit_id
        for record in loaded.records
        if record.fold == fold
        for visit_id in (record.source_visit_id, record.target_visit_id)
    }
    return sorted(result)


def _vqgan_visit_ids(
    loaded: LoadedTransitions | LoadedSplitVisits, fold: str
) -> list[str]:
    if isinstance(getattr(loaded, "folds", None), dict):
        return sorted(
            visit_id
            for visit_id, visit_fold in loaded.folds.items()
            if visit_fold == fold
        )
    return _unique_visit_ids(loaded, fold)


def load_mri_vqgan(
    checkpoint_path: str | Path,
    *,
    require_mri_finetuned: bool = True,
    expected_data_contract_sha256: str | None = None,
    expected_data_backend: str | None = None,
    expected_numeric_contract: str = VQGAN_NUMERIC_CONTRACT,
) -> MRILevelVQGAN:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = payload.get("mewm_ispy2_vqgan_identity", {})
    if require_mri_finetuned and identity.get("mri_finetuned") is not True:
        raise ValueError("VQGAN checkpoint is not marked as I-SPY2 MRI-finetuned")
    if expected_numeric_contract not in {
        VQGAN_NUMERIC_CONTRACT,
        REGISTERED_VQGAN_NUMERIC_CONTRACT,
    }:
        raise ValueError("expected VQGAN numeric contract is unsupported")
    if identity.get("numeric_contract") != expected_numeric_contract:
        raise ValueError("VQGAN checkpoint numeric contract is incompatible")
    if (expected_data_contract_sha256 is None) != (expected_data_backend is None):
        raise ValueError("expected VQGAN data identity must be complete")
    if expected_data_contract_sha256 is not None:
        if identity.get("data_contract_sha256") != expected_data_contract_sha256:
            raise ValueError("VQGAN data contract does not match diffusion data")
        if identity.get("data_backend") != expected_data_backend:
            raise ValueError("VQGAN data backend does not match diffusion data")
    state = payload.get("state_dict", payload)
    model = MRILevelVQGAN(vqgan_config_from_checkpoint_identity(identity))
    model_keys = set(model.state_dict())
    mapped = {}
    for key, value in state.items():
        if key.startswith("autoencoder."):
            mapped[key.removeprefix("autoencoder.")] = value
        elif key in model_keys:
            mapped[key] = value
    incompatible = model.load_state_dict(mapped, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "MRI VQGAN checkpoint is incomplete: "
            f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
        )
    return model.eval()


def _vqgan_model_config(config: ExperimentConfig) -> VQGANConfig:
    models = config.raw.get("models", {})
    raw_vqgan = models.get("vqgan", {}) if isinstance(models, dict) else {}
    if not isinstance(raw_vqgan, dict):
        raise ValueError("config models.vqgan must be a mapping")
    accepted = {field.name for field in fields(VQGANConfig)}
    values = {key: value for key, value in raw_vqgan.items() if key in accepted}
    return VQGANConfig(**values)


def _build_vqgan_training_system(
    config: ExperimentConfig,
    args: Namespace,
    loaded: LoadedTransitions,
    *,
    perceptual_model: torch.nn.Module | None,
) -> VQGANTrainingSystem:
    ct_checkpoint = getattr(args, "ct_checkpoint", None)
    mri_checkpoint = getattr(args, "init_mri_checkpoint", None)
    resume_checkpoint = getattr(args, "resume", None)
    if resume_checkpoint and (ct_checkpoint or mri_checkpoint):
        raise ValueError("VQGAN resume cannot be combined with initialization")
    if not resume_checkpoint and bool(ct_checkpoint) == bool(mri_checkpoint):
        raise ValueError("select exactly one VQGAN initialization checkpoint")

    model_config = _vqgan_model_config(config)
    training = config.vqgan_training
    is_registered_large = (
        _is_registered_strict_a(loaded)
        and model_config == REGISTERED_LARGE_VQGAN_CONFIG
    )
    if is_registered_large:
        if (
            training.init_mri_checkpoint_sha256
            != REGISTERED_LARGE_VQGAN_SOURCE_CHECKPOINT_SHA256
        ):
            raise ValueError(
                "registered Strict-A hidden32 VQGAN requires its pinned source SHA256"
            )
        if mri_checkpoint and (
            Path(mri_checkpoint).resolve()
            != REGISTERED_LARGE_VQGAN_SOURCE_CHECKPOINT.resolve()
        ):
            raise ValueError(
                "registered Strict-A hidden32 VQGAN requires its pinned source path"
            )
        if resume_checkpoint:
            resume_payload = torch.load(
                resume_checkpoint, map_location="cpu", weights_only=False
            )
            resume_identity = resume_payload.get("mewm_ispy2_vqgan_identity", {})
            report = (
                resume_identity.get("width_compatible_initialization_report")
                if isinstance(resume_identity, dict)
                else None
            )
            if (
                not isinstance(report, dict)
                or report.get("report_digest")
                != REGISTERED_LARGE_VQGAN_INITIALIZATION_REPORT_DIGEST
            ):
                raise ValueError(
                    "VQGAN resume identity has invalid width-compatible provenance"
                )

    model = MRILevelVQGAN(model_config)
    checkpoint_identity: dict[str, Any] = {
        "data_contract_sha256": loaded.data_contract_sha256,
        "data_backend": loaded.backend,
        "numeric_contract": _vqgan_numeric_contract(loaded),
    }
    system = VQGANTrainingSystem(
        model,
        learning_rate=training.generator_learning_rate,
        discriminator_learning_rate=training.discriminator_learning_rate,
        adversarial_always_on=training.adversarial_always_on,
        image_gan_weight=training.image_gan_weight,
        volume_gan_weight=training.volume_gan_weight,
        feature_matching_weight=training.feature_matching_weight,
        gradient_clip_val=training.gradient_clip_val,
        batch_size=training.batch_size,
        precision=training.precision,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        early_stopping_patience=training.early_stopping_patience,
        perceptual_model=perceptual_model,
        checkpoint_identity=checkpoint_identity,
    )

    if ct_checkpoint:
        ct_sha256 = sha256_file(Path(ct_checkpoint))
        if ct_sha256 != CT_EXPECTED_SHA256:
            raise ValueError("CT initialization checkpoint does not match the pinned SHA256")
        report = load_ct_autoencoder_weights(
            model, ct_checkpoint, require_complete=True
        )
        system.checkpoint_identity.update(
            {
                "ct_checkpoint_sha256": report.checkpoint_sha256,
                "initialization_method": "ct_checkpoint",
                "discriminator_initialization": "random_mri",
            }
        )
        return system

    if mri_checkpoint:
        if (
            _is_registered_strict_a(loaded)
            and model.config == REGISTERED_LARGE_VQGAN_CONFIG
        ):
            expected_sha256 = training.init_mri_checkpoint_sha256
            if expected_sha256 is None:
                raise ValueError(
                    "width-compatible MRI initialization requires a pinned SHA256"
                )
            report = load_width_compatible_mri_codebook_discriminators(
                system,
                mri_checkpoint,
                expected_checkpoint_sha256=expected_sha256,
                expected_report_digest=(
                    REGISTERED_LARGE_VQGAN_INITIALIZATION_REPORT_DIGEST
                ),
            )
            system.checkpoint_identity.update(
                {
                    "initialization_method": WIDTH_COMPATIBLE_MRI_INITIALIZATION,
                    "mri_initialization_checkpoint_sha256": report.checkpoint_sha256,
                    "discriminator_initialization": WIDTH_COMPATIBLE_MRI_INITIALIZATION,
                    "width_compatible_initialization_report": report.identity_payload(),
                }
            )
            return system
        report = load_mri_training_weights(system, mri_checkpoint)
        expected_sha256 = training.init_mri_checkpoint_sha256
        if expected_sha256 is not None and report.checkpoint_sha256 != expected_sha256:
            raise ValueError("MRI initialization checkpoint does not match the pinned SHA256")
        system.checkpoint_identity.update(
            {
                "initialization_method": (
                    "mri_checkpoint_zero_initialized_identity_bottlenecks"
                ),
                "mri_initialization_checkpoint_sha256": report.checkpoint_sha256,
                "source_checkpoint_epoch": report.source_epoch,
                "source_checkpoint_global_step": report.source_global_step,
                "discriminator_initialization": "mri_checkpoint",
            }
        )
        return system

    if not is_registered_large:
        resume_payload = torch.load(
            resume_checkpoint, map_location="cpu", weights_only=False
        )
    resume_identity = resume_payload.get("mewm_ispy2_vqgan_identity", {})
    resume_config = vqgan_config_from_checkpoint_identity(resume_identity)
    if resume_config != model.config:
        raise ValueError("VQGAN resume identity mismatch: architecture_contract")
    expected_resume = {
        **checkpoint_identity,
        "training_contract": system.training_contract,
    }
    for key, value in expected_resume.items():
        if resume_identity.get(key) != value:
            raise ValueError(f"VQGAN resume identity mismatch: {key}")
    if is_registered_large:
        expected_provenance = {
            "initialization_method": WIDTH_COMPATIBLE_MRI_INITIALIZATION,
            "mri_initialization_checkpoint_sha256": (
                training.init_mri_checkpoint_sha256
            ),
            "discriminator_initialization": WIDTH_COMPATIBLE_MRI_INITIALIZATION,
        }
        if any(
            resume_identity.get(key) != value
            for key, value in expected_provenance.items()
        ):
            raise ValueError(
                "VQGAN resume identity has invalid width-compatible provenance"
            )
        validate_width_compatible_initialization_report(
            system,
            resume_identity.get("width_compatible_initialization_report"),
            expected_checkpoint_sha256=training.init_mri_checkpoint_sha256 or "",
            expected_report_digest=(
                REGISTERED_LARGE_VQGAN_INITIALIZATION_REPORT_DIGEST
            ),
        )
    elif model.config.bottleneck_blocks:
        expected_provenance = {
            "initialization_method": (
                "mri_checkpoint_zero_initialized_identity_bottlenecks"
            ),
            "mri_initialization_checkpoint_sha256": (
                training.init_mri_checkpoint_sha256
            ),
            "discriminator_initialization": "mri_checkpoint",
        }
        invalid_provenance = any(
            resume_identity.get(key) != value
            for key, value in expected_provenance.items()
        ) or any(
            resume_identity.get(key) is None
            for key in ("source_checkpoint_epoch", "source_checkpoint_global_step")
        )
        if invalid_provenance:
            raise ValueError("VQGAN resume identity has invalid warm-start provenance")
    provenance_keys = (
        "ct_checkpoint_sha256",
        "initialization_method",
        "mri_initialization_checkpoint_sha256",
        "source_checkpoint_epoch",
        "source_checkpoint_global_step",
        "discriminator_initialization",
        "width_compatible_initialization_report",
    )
    system.checkpoint_identity.update(
        {key: resume_identity[key] for key in provenance_keys if key in resume_identity}
    )
    return system


def _load_segmenter(checkpoint_path: str | Path) -> torch.nn.Module:
    model = build_dynunet()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    mapped = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    if not mapped:
        mapped = state
    model.load_state_dict(mapped, strict=True)
    return model.eval()


def _build_paper_training_components(
    config: ExperimentConfig,
    args: Namespace,
    loaded: LoadedTransitions,
    output: Path,
) -> tuple[
    PaperDiffusionTrainingSystem
    | RegisteredLargePaperDiffusionTrainingSystem
    | RegisteredX0PaperDiffusionTrainingSystem,
    DataLoader[Any],
    DataLoader[Any],
    list[Callback],
]:
    runtime = config.diffusion
    contract = runtime.paper
    if contract is None:
        raise ValueError("paper training requires a paper runtime contract")
    vqgan_path = Path(args.vqgan_checkpoint)
    vqgan = load_mri_vqgan(
        vqgan_path,
        expected_data_contract_sha256=loaded.data_contract_sha256,
        expected_data_backend=loaded.backend,
        **_vqgan_load_contract_kwargs(loaded),
    )
    vqgan_sha256 = sha256_file(vqgan_path)
    if vqgan_sha256 != contract.vqgan_sha256:
        raise ValueError("paper VQGAN checkpoint does not match the pinned SHA256")
    tower = FrozenCLIPTextTower.from_pretrained(
        model_id=contract.clip_model_id,
        revision=contract.clip_revision,
        local_files_only=True,
    )
    conditioner = PaperActionConditioner(tower, contract=contract)
    latent_statistics = _load_runtime_latent_statistics(
        runtime,
        vqgan_sha256=vqgan_sha256,
        data_contract_sha256=loaded.data_contract_sha256,
    )
    model = _build_paper_diffusion_model(
        vqgan,
        conditioner,
        runtime,
        latent_statistics=latent_statistics,
    )
    registered_large = (
        runtime.denoiser_architecture == REGISTERED_LARGE_PAPER_ARCHITECTURE
    )
    prediction_type = getattr(runtime, "prediction_type", EPSILON_PREDICTION_TYPE)
    configured_training = getattr(config, "diffusion_training", None)
    paper_batch_size = (
        getattr(configured_training, "batch_size", REGISTERED_LARGE_PHYSICAL_BATCH_SIZE)
        if registered_large
        else 1
    )
    if registered_large and prediction_type == X0_PREDICTION_TYPE:
        x0_identity = _registered_x0_checkpoint_identity(
            model,
            runtime=runtime,
            vqgan_sha256=vqgan_sha256,
            loaded=loaded,
            physical_batch_size=paper_batch_size,
        )
        system = RegisteredX0PaperDiffusionTrainingSystem(
            model,
            learning_rate=1e-4,
            checkpoint_identity=x0_identity,
        )
    elif registered_large:
        identity = _registered_large_checkpoint_identity(
            model,
            vqgan_sha256=vqgan_sha256,
            loaded=loaded,
        )
        system = RegisteredLargePaperDiffusionTrainingSystem(
            model,
            learning_rate=1e-4,
            checkpoint_identity=identity,
        )
    else:
        legacy_identity = _paper_checkpoint_identity(
            model,
            vqgan_sha256=vqgan_sha256,
            loaded=loaded,
        )
        system = PaperDiffusionTrainingSystem(
            model,
            learning_rate=1e-4,
            checkpoint_identity=legacy_identity,
        )
    train_dataset, val_dataset = _paper_transition_datasets(config, loaded)
    if registered_large and prediction_type == EPSILON_PREDICTION_TYPE:
        if paper_batch_size != REGISTERED_LARGE_PHYSICAL_BATCH_SIZE:
            raise ValueError(
                "registered-large diffusion physical batch size must be twelve"
            )
    train_loader, val_loader = _paper_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=paper_batch_size,
    )
    callbacks = [
        *_diffusion_callbacks(
            output,
            denoiser_architecture=runtime.denoiser_architecture,
            prediction_type=prediction_type,
        ),
        PaperDatasetEpochCallback(train_dataset, val_dataset),
    ]
    return system, train_loader, val_loader, callbacks


def _run_paper_diffusion_training(
    config: ExperimentConfig,
    args: Namespace,
    loaded: LoadedTransitions,
    output: Path,
) -> None:
    resume = getattr(args, "resume", None)
    warm_start_override = getattr(args, "warm_start_checkpoint", None)
    if resume and warm_start_override:
        raise ValueError("paper v4 warm-start cannot be combined with v5 resume")
    contract = config.diffusion.paper
    if contract is None:
        raise ValueError("paper training requires a paper runtime contract")
    registered_large = (
        config.diffusion.denoiser_architecture
        == REGISTERED_LARGE_PAPER_ARCHITECTURE
    )
    if registered_large and warm_start_override:
        raise ValueError(
            "registered-large paper training does not accept a warm-start checkpoint"
        )
    if registered_large and resume:
        if config.diffusion.prediction_type == X0_PREDICTION_TYPE:
            expected_x0_identity = _registered_x0_checkpoint_identity_from_runtime(
                config.diffusion,
                vqgan_sha256=contract.vqgan_sha256,
                loaded=loaded,
                physical_batch_size=config.diffusion_training.batch_size,
            )
            actual_x0_identity = peek_registered_x0_checkpoint_identity(resume)
            _validate_registered_x0_identity_match(
                actual_x0_identity,
                expected_x0_identity,
            )
        else:
            expected_resume_identity = _registered_large_checkpoint_identity_from_runtime(
                config.diffusion,
                vqgan_sha256=contract.vqgan_sha256,
                loaded=loaded,
            )
            actual_resume_identity = peek_registered_large_checkpoint_identity(resume)
            _validate_registered_large_identity_match(
                actual_resume_identity,
                expected_resume_identity,
            )

    system, train_loader, val_loader, callbacks = _build_paper_training_components(
        config, args, loaded, output
    )
    if not resume and not registered_large:
        warm_start_path = Path(warm_start_override or contract.warm_start_path)
        import_v4_paper_warm_start(
            system.model,
            warm_start_path,
            expected_sha256=contract.warm_start_sha256,
            expected_vqgan_sha256=contract.vqgan_sha256,
            expected_data_contract_sha256=loaded.data_contract_sha256,
            output_directory=output / "warm_start",
        )

    trainer = _trainer(
        args,
        output,
        callbacks,
        stage="diffusion",
        denoiser_architecture=config.diffusion.denoiser_architecture,
        accumulate_grad_batches=(
            REGISTERED_X0_ACCUMULATE_GRAD_BATCHES
            if getattr(config.diffusion, "prediction_type", EPSILON_PREDICTION_TYPE)
            == X0_PREDICTION_TYPE
            else (
                REGISTERED_LARGE_ACCUMULATE_GRAD_BATCHES
                if registered_large
                else 1
            )
        ),
        gradient_clip_val=1.0,
    )
    trainer.fit(
        system,
        train_loader,
        val_loader,
        ckpt_path=resume,
    )


def run_training(stage: str, args: Namespace) -> None:
    config = load_experiment_config(args.config)
    pl.seed_everything(config.runtime.seed, workers=True)
    loaded = _load_data(config)
    output = config.runtime.output_dir / stage
    output.mkdir(parents=True, exist_ok=True)
    if stage == "vqgan":
        vqgan_visits: LoadedTransitions | LoadedSplitVisits = loaded
        if _is_registered_strict_a(loaded):
            vqgan_visits = load_split_visit_records(
                config.data.bundle_json,
                config.data.phase_manifest_csv,
                backend=config.data.backend,
            )
        visit_loader = _visit_loader(config, vqgan_visits)
        try:
            perceptual = UncheckedLPIPSLoss.vgg()
        except Exception as exc:
            raise RuntimeError("slice LPIPS initialization failed") from exc
        system = _build_vqgan_training_system(
            config,
            args,
            loaded,
            perceptual_model=perceptual,
        )
        train = CachedVisitDataset(
            _vqgan_visit_ids(vqgan_visits, "train"),
            visit_loader,
        )
        val = CachedVisitDataset(
            _vqgan_visit_ids(vqgan_visits, "val"),
            visit_loader,
        )
        callbacks = _vqgan_callbacks(
            output,
            adversarial_full_step=system.adversarial_full_step,
            early_stopping_patience=config.vqgan_training.early_stopping_patience,
        )
        train_loader, val_loader = _vqgan_dataloaders(
            train,
            val,
            batch_size=config.vqgan_training.batch_size,
        )
        _trainer(
            args,
            output,
            callbacks,
            stage=stage,
            vqgan_precision=config.vqgan_training.precision,
        ).fit(
            system,
            train_loader,
            val_loader,
            ckpt_path=args.resume,
        )
        return
    if stage == "diffusion":
        if (
            getattr(getattr(config, "diffusion", None), "denoiser_architecture", None)
            in {
                PAPER_FAITHFUL_ARCHITECTURE,
                REGISTERED_LARGE_PAPER_ARCHITECTURE,
            }
        ):
            _run_paper_diffusion_training(config, args, loaded, output)
            return
        vqgan = load_mri_vqgan(
            args.vqgan_checkpoint,
            expected_data_contract_sha256=loaded.data_contract_sha256,
            expected_data_backend=loaded.backend,
            **_vqgan_load_contract_kwargs(loaded),
        )
        tower = SharedMedGemmaTower.from_pretrained(local_files_only=True)
        conditioner = ISPY2Conditioner(tower)
        model = _build_diffusion_model(vqgan, conditioner, config.diffusion)
        identity = _diffusion_checkpoint_identity(
            model,
            vqgan_sha256=sha256_file(Path(args.vqgan_checkpoint)),
            loaded=loaded,
        )
        is_ct = config.diffusion.denoiser_architecture == CT_DENOISER_ARCHITECTURE
        system = DiffusionTrainingSystem(
            model,
            learning_rate=1e-4,
            lora_learning_rate=1e-4 if is_ct else 2e-5,
            optimizer_name="adam" if is_ct else "adamw",
            checkpoint_identity=asdict(identity),
        )
        callbacks = _diffusion_callbacks(
            output,
            denoiser_architecture=config.diffusion.denoiser_architecture,
        )
        _trainer(
            args,
            output,
            callbacks,
            stage=stage,
            denoiser_architecture=config.diffusion.denoiser_architecture,
            accumulate_grad_batches=2 if is_ct else 1,
        ).fit(
            system,
            DataLoader(
                _transition_dataset(config, loaded, "train"),
                batch_size=1,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_transitions,
            ),
            DataLoader(
                _transition_dataset(config, loaded, "val"),
                batch_size=1,
                num_workers=0,
                collate_fn=collate_transitions,
            ),
            ckpt_path=args.resume,
        )
        return
    if stage == "segmenter":
        split_visits = load_split_visit_records(
            config.data.bundle_json,
            config.data.phase_manifest_csv,
            backend=config.data.backend,
        )
        if _is_registered_strict_a(split_visits):
            visit_loader = _visit_loader(config, split_visits)
        else:
            cache = DCE0Cache(
                config.data.cache_dir, output_shape_zyx=config.data.output_shape_zyx
            )
            visit_loader = lambda visit_id: cache.load_or_create(
                split_visits.visits[visit_id], backend=config.data.backend
            )
        system = DynUNetSegmentationSystem(build_dynunet())
        train = CachedVisitDataset(
            sorted(key for key, fold in split_visits.folds.items() if fold == "train"),
            visit_loader,
        )
        val = CachedVisitDataset(
            sorted(key for key, fold in split_visits.folds.items() if fold == "val"),
            visit_loader,
        )
        callback = ModelCheckpoint(
            dirpath=output / "checkpoints",
            monitor="val/dice",
            mode="max",
            save_last=True,
            save_top_k=1,
        )
        _trainer(args, output, [callback], stage=stage).fit(
            system,
            DataLoader(train, batch_size=1, shuffle=True, num_workers=2),
            DataLoader(val, batch_size=1, num_workers=2),
            ckpt_path=args.resume,
        )
        return
    raise ValueError(f"unknown training stage: {stage}")


def run_source_inference(args: Namespace) -> Path:
    config = load_experiment_config(args.config)
    pl.seed_everything(config.runtime.seed, workers=True)
    loaded = load_source_inference_records(
        config.data.bundle_json,
        config.data.phase_manifest_csv,
        backend=config.data.backend,
    )
    matches = [record for record in loaded.records if record.transition_id == args.transition_id]
    if len(matches) != 1:
        raise ValueError("source transition ID is not unique in the locked bundle")
    record = matches[0]
    if _is_registered_strict_a(loaded):
        source = _visit_loader(config, loaded)(record.source_visit_id)
    else:
        cache = DCE0Cache(
            config.data.cache_dir, output_shape_zyx=config.data.output_shape_zyx
        )
        source = cache.load_or_create(
            loaded.visits[record.source_visit_id], backend=config.data.backend
        )
    if config.diffusion.denoiser_architecture == PAPER_FAITHFUL_ARCHITECTURE:
        diffusion, identity, _ = load_paper_diffusion_for_inference(
            runtime=config.diffusion,
            loaded=loaded,
            vqgan_checkpoint_path=args.vqgan_checkpoint,
            diffusion_checkpoint_path=args.diffusion_checkpoint,
            vqgan_sha256=sha256_file(Path(args.vqgan_checkpoint)),
        )
    elif (
        config.diffusion.denoiser_architecture
        == REGISTERED_LARGE_PAPER_ARCHITECTURE
    ):
        diffusion, identity, _ = load_registered_paper_diffusion_for_inference(
            runtime=config.diffusion,
            loaded=loaded,
            vqgan_checkpoint_path=args.vqgan_checkpoint,
            diffusion_checkpoint_path=args.diffusion_checkpoint,
            physical_batch_size=config.diffusion_training.batch_size,
        )
    else:
        vqgan = load_mri_vqgan(
            args.vqgan_checkpoint,
            expected_data_contract_sha256=loaded.data_contract_sha256,
            expected_data_backend=loaded.backend,
            **_vqgan_load_contract_kwargs(loaded),
        )
        conditioner = ISPY2Conditioner(
            SharedMedGemmaTower.from_pretrained(local_files_only=True)
        )
        diffusion = _build_diffusion_model(vqgan, conditioner, config.diffusion)
        identity = _diffusion_checkpoint_identity(
            diffusion,
            vqgan_sha256=sha256_file(Path(args.vqgan_checkpoint)),
            loaded=loaded,
        )
        load_diffusion_checkpoint(diffusion, args.diffusion_checkpoint, identity)
    segmenter = _load_segmenter(args.segmenter_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion.to(device).eval()
    segmenter.to(device).eval()
    sample_count = 8
    source_image = source.image.unsqueeze(0).to(device).repeat(sample_count, 1, 1, 1, 1)
    source_mask = source.mask.unsqueeze(0).float().to(device).repeat(sample_count, 1, 1, 1, 1)
    with torch.inference_mode():
        generated = diffusion.sample(
            source_image,
            source_mask,
            [record.action_text] * sample_count,
            [record.clinical_text] * sample_count,
            torch.full((sample_count,), float(record.delta_days), device=device),
            torch.full((sample_count,), record.stage_id, dtype=torch.long, device=device),
        )
        result = segment_generated_samples(segmenter, generated)
    output = Path(args.output_dir) / record.transition_id.replace(":", "_").replace("->", "_to_")
    return publish_prediction(
        output,
        result,
        metadata={
            "transition_id": record.transition_id,
            "source_visit_id": source.visit_id,
            "source_phase_index": source.phase_index,
            "source_n_times": source.n_times,
            "source_image_sha256": source.image_sha256,
            "source_preprocessing": source.metadata,
            "data_backend": loaded.backend,
            "data_contract_sha256": loaded.data_contract_sha256,
            "vqgan_checkpoint_sha256": identity.vqgan_sha256,
            "diffusion_checkpoint_sha256": sha256_file(Path(args.diffusion_checkpoint)),
            "segmenter_checkpoint_sha256": sha256_file(Path(args.segmenter_checkpoint)),
            "seed": config.runtime.seed,
        },
    )


def _nifti_array(path: Path) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj)


def run_evaluation(args: Namespace) -> Path:
    prediction_directory = Path(args.prediction_dir)
    assert_prediction_complete(prediction_directory)
    prediction_manifest = json.loads((prediction_directory / "prediction.json").read_text())
    transition_id = prediction_manifest["transition_id"]
    config = load_experiment_config(args.config)
    if prediction_manifest.get("data_backend") != config.data.backend:
        raise ValueError("prediction data backend does not match evaluation config")
    loaded = _load_data(config)
    if prediction_manifest.get("data_contract_sha256") != loaded.data_contract_sha256:
        raise ValueError("prediction data contract does not match evaluation data")
    matches = [record for record in loaded.records if record.transition_id == transition_id]
    if len(matches) != 1:
        raise ValueError("prediction transition is not unique in the locked bundle")
    record = matches[0]
    if _is_registered_strict_a(loaded):
        identity_loader = _visit_loader(config, loaded)
        identity_source = identity_loader(record.source_visit_id)
    else:
        identity_cache = DCE0Cache(
            config.data.cache_dir, output_shape_zyx=config.data.output_shape_zyx
        )
        identity_source = identity_cache.load_or_create(
            loaded.visits[record.source_visit_id], backend=config.data.backend
        )
    validate_prediction_identity(
        prediction_manifest,
        expected_backend=config.data.backend,
        expected_data_contract_sha256=loaded.data_contract_sha256,
        expected_source=identity_source,
    )
    if (
        config.data.backend == "registered_t0"
        and not _is_registered_strict_a(loaded)
    ):
        cache = RegisteredPairCache(
            config.data.cache_dir, output_shape_zyx=config.data.output_shape_zyx
        )
        source, target = cache.load_or_create(
            loaded.visits[record.source_visit_id], loaded.visits[record.target_visit_id]
        )
    else:
        source = identity_source
        if _is_registered_strict_a(loaded):
            target = identity_loader(record.target_visit_id)
        else:
            target = identity_cache.load_or_create(
                loaded.visits[record.target_visit_id], backend=config.data.backend
            )
    dce_samples = np.stack(
        [_nifti_array(prediction_directory / f"sample_{index:02d}_dce.nii.gz") for index in range(8)]
    )
    mask_samples = np.stack(
        [_nifti_array(prediction_directory / f"sample_{index:02d}_mask.nii.gz") for index in range(8)]
    )
    result = evaluate_samples(
        dce_samples,
        mask_samples,
        target.image.numpy()[0],
        target.mask.numpy()[0],
        source.mask.numpy()[0],
        _nifti_array(prediction_directory / "entropy.nii.gz"),
        spacing_zyx=(2.0, 0.7032, 0.7032),
        surface_tolerance_mm=1.0,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.rows).to_csv(output / "samples.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True)
    )
    return output
