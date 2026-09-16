from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader, default_collate

from .cache import RegisteredStrictAROICache
from .ispy2_biflow_config import ISPY2BiFlowConfig, load_ispy2_biflow_config
from .ispy2_biflow_latent_contract import (
    ISPY2_BIFLOW_LATENT_NORMALIZATION,
    ISPY2BiFlowLatentCache,
)
from .ispy2_biflow_data import ISPY2BiFlowPairDataset
from .ispy2_biflow_preflight import audit_ispy2_biflow_training_assets
from .ispy2_biflow_training import (
    ISPY2_BIFLOW_CHECKPOINT_STATE_POLICY,
    ISPY2BiFlowTrainingSystem,
    build_ispy2_biflow_batch,
)
from .ispy2_biflow_world_model import (
    ISPY2BiFlowWorldModel,
    build_ispy2_biflow_world_model,
)
from .ispy2_dce0_world_data import (
    ISPY2DCE0WorldPair,
    build_ispy2_dce0_world_pairs,
)
from .ispy2_dce0_world_latents import (
    ISPY2_DCE0_CONTINUOUS_NORMALIZATION,
    ISPY2DCE0ContinuousLatentCache,
)
from .mu_glioma_world_model import MUMedicalTextTower


ISPY2_BIFLOW_PREFLIGHT_SCHEMA = "mewm_ispy2_dce0_biflow_preflight_v2"
ISPY2_BIFLOW_IDENTITY_SCHEMA = "mewm_ispy2_dce0_biflow_identity_v2"


class _DivergenceGuard(pl.Callback):
    def __init__(self, *, reference: float, multiplier: float, patience: int) -> None:
        super().__init__()
        self.threshold = reference * multiplier
        self.patience = patience
        self.consecutive_failures = 0

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get("val/velocity_mae")
        if metric is None:
            return
        value = float(metric.detach().cpu()) if torch.is_tensor(metric) else float(metric)
        self.consecutive_failures = (
            self.consecutive_failures + 1 if value > self.threshold else 0
        )
        if self.consecutive_failures >= self.patience:
            trainer.should_stop = True
            print(
                "BiFlow divergence guard stopped training: "
                f"val/velocity_mae={value:.9f} exceeded "
                f"{self.threshold:.9f} for {self.consecutive_failures} validations."
            )

    def state_dict(self) -> dict[str, Any]:
        return {"consecutive_failures": self.consecutive_failures}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.consecutive_failures = int(state_dict["consecutive_failures"])


class _ShapeOnlyTextTower(torch.nn.Module):
    hidden_size = 16

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        return torch.zeros(len(texts), self.hidden_size)


def _world_data(config: ISPY2BiFlowConfig):
    base = config.base
    pairs, audit = build_ispy2_dce0_world_pairs(
        base.data.bundle_json, base.data.phase_manifest_csv
    )
    if base.model.latent_normalization == ISPY2_BIFLOW_LATENT_NORMALIZATION:
        latent_cache = ISPY2BiFlowLatentCache(
            base.data.continuous_root,
            expected_normalization=base.model.latent_normalization,
        )
    elif base.model.latent_normalization == ISPY2_DCE0_CONTINUOUS_NORMALIZATION:
        latent_cache = ISPY2DCE0ContinuousLatentCache(
            base.data.continuous_root,
        )
    else:
        raise ValueError(
            "I-SPY2 BiFlowNet latent normalization is unsupported"
        )
    roi_cache = RegisteredStrictAROICache(
        base.data.roi_cache_dir,
        bundle_json=base.data.bundle_json,
        output_shape_zyx=base.data.output_shape_zyx,
    )
    return tuple(pairs), audit, latent_cache, roi_cache


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("I-SPY2 BiFlowNet collate requires samples")
    keys = set(samples[0])
    if any(set(sample) != keys for sample in samples):
        raise ValueError("I-SPY2 BiFlowNet sample fields differ")
    return {
        key: [sample[key] for sample in samples]
        if key == "metadata"
        else default_collate([sample[key] for sample in samples])
        for key in keys
    }


def _dataset(
    pairs: Sequence[ISPY2DCE0WorldPair],
    latent_cache: Any,
    roi_cache: RegisteredStrictAROICache,
    *,
    split: str,
) -> ISPY2BiFlowPairDataset:
    return ISPY2BiFlowPairDataset(
        pairs, latent_cache, roi_cache, split=split
    )


def _loader(
    dataset: ISPY2BiFlowPairDataset,
    *,
    config: ISPY2BiFlowConfig,
    shuffle: bool,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        collate_fn=_collate,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.training.num_workers > 0,
        generator=torch.Generator().manual_seed(config.runtime.seed),
    )


def _build_model(
    config: ISPY2BiFlowConfig,
    *,
    text_tower: Any | None = None,
) -> ISPY2BiFlowWorldModel:
    base = config.base
    tower = text_tower
    if tower is None:
        tower = MUMedicalTextTower.from_pretrained(
            model_id=base.text.model_id,
            revision=base.text.revision,
            max_length=base.text.max_length,
            lora_rank=base.text.lora_rank,
            lora_alpha=base.text.lora_alpha,
            lora_dropout=base.text.lora_dropout,
        )
    return build_ispy2_biflow_world_model(
        text_tower=tower,
        text_hidden_size=getattr(tower, "hidden_size", None),
        preset=config.preset,
        latent_channels=base.model.latent_channels,
        context_dim=base.model.context_dim,
    )


def _identity(
    config: ISPY2BiFlowConfig,
    *,
    model: ISPY2BiFlowWorldModel,
    latent_cache: Any,
) -> dict[str, Any]:
    identity = {
        "schema": ISPY2_BIFLOW_IDENTITY_SCHEMA,
        "config_identity_sha256": config.identity_sha256(),
        "conditioning_data": config.conditioning_data_payload(),
        "architecture": model.architecture_contract,
        "continuous_cache_identity": latent_cache.identity,
        "bundle_contract_sha256": config.base.data.bundle_contract_sha256,
        "vqgan_sha256": config.base.data.vqgan_sha256,
        "image_condition_path": "controlnet_only",
        "source_modalities": list(config.base.data.source_modalities),
        "text_model_id": config.base.text.model_id,
        "text_revision": config.base.text.revision,
        "prediction_target": "full_future_state",
        "velocity_loss": "l1",
        "checkpoint_state_policy": ISPY2_BIFLOW_CHECKPOINT_STATE_POLICY,
    }
    if config.runtime.warm_start_checkpoint is not None:
        identity["optimization"] = {
            "optimizer": "adamw",
            "layout": config.training.optimizer_layout,
            "backbone_learning_rate": config.training.backbone_learning_rate,
            "controlnet_learning_rate": config.training.controlnet_learning_rate,
            "conditioner_learning_rate": config.training.conditioner_learning_rate,
            "text_lora_learning_rate": config.training.text_lora_learning_rate,
            "minimum_learning_rate": config.training.min_learning_rate,
            "warmup_fraction": config.training.warmup_fraction,
            "schedule": "linear_warmup_cosine_decay_v1",
            "gradient_clip_val": config.training.gradient_clip_val,
        }
        identity["warm_start"] = {
            "checkpoint": str(config.runtime.warm_start_checkpoint),
            "sha256": config.runtime.warm_start_sha256,
            "state": "model_weights_only_fresh_optimizer",
        }
    return identity


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _warm_start_weights(
    system: ISPY2BiFlowTrainingSystem,
    *,
    checkpoint_path: Path,
    expected_sha256: str,
    target_identity: dict[str, Any],
) -> dict[str, Any]:
    actual_sha256 = _sha256_file(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("I-SPY2 BiFlowNet warm-start checkpoint SHA256 mismatch")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    source_identity = checkpoint.get("ispy2_biflow_identity")
    if not isinstance(source_identity, dict):
        raise ValueError("I-SPY2 BiFlowNet warm-start identity is missing")
    immutable_fields = (
        "architecture",
        "continuous_cache_identity",
        "bundle_contract_sha256",
        "vqgan_sha256",
        "source_modalities",
        "text_model_id",
        "text_revision",
        "prediction_target",
        "velocity_loss",
        "checkpoint_state_policy",
    )
    if any(source_identity.get(key) != target_identity.get(key) for key in immutable_fields):
        raise ValueError("I-SPY2 BiFlowNet warm-start model/data contract mismatch")
    state = checkpoint.get("state_dict")
    omitted = checkpoint.get("ispy2_biflow_omitted_state_keys")
    expected_omitted = system._omitted_frozen_text_state_keys()
    expected_saved = set(system.state_dict()) - set(expected_omitted)
    if (
        checkpoint.get("ispy2_biflow_state_policy")
        != ISPY2_BIFLOW_CHECKPOINT_STATE_POLICY
        or omitted != list(expected_omitted)
        or not isinstance(state, dict)
        or set(state) != expected_saved
    ):
        raise ValueError("I-SPY2 BiFlowNet warm-start state mismatch")
    incompatible = system.load_state_dict(state, strict=False)
    if (
        not set(incompatible.missing_keys).issubset(expected_omitted)
        or incompatible.unexpected_keys
    ):
        raise ValueError("I-SPY2 BiFlowNet warm-start load was incomplete")
    return {
        "checkpoint": str(checkpoint_path),
        "sha256": actual_sha256,
        "source_epoch": int(checkpoint["epoch"]),
        "source_global_step": int(checkpoint["global_step"]),
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
    }


def preflight_ispy2_biflow(config_path: str | Path) -> dict[str, Any]:
    config = load_ispy2_biflow_config(config_path)
    pairs, audit, latent_cache, roi_cache = _world_data(config)
    patient_sets = {
        split: {pair.patient_id for pair in pairs if pair.split == split}
        for split in ("train", "val")
    }
    leakage = sorted(patient_sets["train"] & patient_sets["val"])
    sample_pair = next(pair for pair in pairs if pair.split == "val")
    dataset = ISPY2BiFlowPairDataset(
        (sample_pair,), latent_cache, roi_cache, split="val"
    )
    sample = dataset[0]
    target = sample["target_latent"].unsqueeze(0)
    flow = build_ispy2_biflow_batch(
        target,
        noise=torch.zeros_like(target),
        flow_time=torch.tensor([0.5], dtype=target.dtype),
    )
    shape_model = build_ispy2_biflow_world_model(
        text_tower=_ShapeOnlyTextTower(),
        text_hidden_size=_ShapeOnlyTextTower.hidden_size,
        preset=config.preset,
        latent_channels=config.base.model.latent_channels,
        context_dim=config.base.model.context_dim,
    )
    architecture = shape_model.architecture_contract
    parameters = sum(value.numel() for value in shape_model.dynamics.parameters())
    assets = audit_ispy2_biflow_training_assets(
        config, pairs=pairs, latent_cache=latent_cache, roi_cache=roi_cache
    )
    ready = bool(
        assets["ready"]
        and not leakage
        and flow.flow_state.shape == target.shape
        and architecture["spatial_input"] == "flow_state_only"
        and architecture["image_condition_path"] == "controlnet_only"
    )
    result = {
        "schema": ISPY2_BIFLOW_PREFLIGHT_SCHEMA,
        "ready": ready,
        "config": {
            "path": str(config.path),
            "sha256": config.sha256,
            "identity_sha256": config.identity_sha256(),
        },
        "data": {
            "pair_count": audit.pair_count,
            "split_pair_counts": audit.split_pair_counts,
            "split_patient_counts": audit.split_patient_counts,
            "patient_leakage": leakage,
            "sample_target_shape": list(target.shape),
            "sample_flow_shape": list(flow.flow_state.shape),
            "sample_fields": sorted(sample),
            "source_latent_loaded": False,
        },
        "conditioning": {
            "source_modalities": list(config.base.data.source_modalities),
            "source_mri_shape": list(sample["source_mri"].shape),
            "image_condition_path": "controlnet_only",
            "context_tokens": config.base.model.context_tokens,
            "target_mri_access": False,
        },
        "dynamics": {
            "preset": config.preset,
            "spatial_input": "flow_state_only",
            "input_channels": config.base.model.latent_channels,
            "output_channels": config.base.model.latent_channels,
            "source_latent_input": False,
            "parameters": parameters,
            "architecture": architecture,
        },
        "training_assets": assets,
        "resources": {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        },
    }
    del shape_model, dataset, roi_cache, latent_cache
    gc.collect()
    return result


def profile_ispy2_biflow(
    config_path: str | Path,
    *,
    device: str,
) -> dict[str, Any]:
    config = load_ispy2_biflow_config(config_path)
    target_device = torch.device(device)
    model = build_ispy2_biflow_world_model(
        text_tower=_ShapeOnlyTextTower(),
        text_hidden_size=_ShapeOnlyTextTower.hidden_size,
        preset=config.preset,
        latent_channels=config.base.model.latent_channels,
        context_dim=config.base.model.context_dim,
    )
    model.dynamics.to(target_device).train()
    shape = config.base.model.latent_shape_czyx
    sample = torch.randn(
        config.training.batch_size,
        *shape,
        device=target_device,
        requires_grad=True,
    )
    flow_time = torch.rand(config.training.batch_size, device=target_device)
    context = torch.randn(
        config.training.batch_size,
        config.base.model.context_tokens,
        config.base.model.context_dim,
        device=target_device,
    )
    spatial_condition = torch.randn(
        config.training.batch_size,
        *config.base.model.source_mri_shape_czyx,
        device=target_device,
    )
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
        torch.cuda.synchronize(target_device)
    started = time.perf_counter()
    mixed_precision = target_device.type == "cuda" and (
        "bf16" in config.training.precision or "16" in config.training.precision
    )
    autocast_dtype = (
        torch.bfloat16
        if "bf16" in config.training.precision
        else torch.float16
    )
    with torch.autocast(
        device_type=target_device.type,
        dtype=autocast_dtype,
        enabled=mixed_precision,
    ):
        output = model.dynamics(
            sample,
            flow_time,
            context=context,
            spatial_condition=spatial_condition,
        )
        loss = output.square().mean()
    loss.backward()
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    elapsed = time.perf_counter() - started
    report = {
        "schema": "mewm_ispy2_dce0_biflow_profile_v1",
        "completed": True,
        "device": str(target_device),
        "preset": config.preset,
        "batch_size": config.training.batch_size,
        "precision": config.training.precision,
        "input_shape": list(sample.shape),
        "spatial_condition_shape": list(spatial_condition.shape),
        "image_condition_path": "controlnet_only",
        "output_shape": list(output.shape),
        "seconds": elapsed,
        "parameters": sum(value.numel() for value in model.dynamics.parameters()),
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(target_device)
            if target_device.type == "cuda"
            else None
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(target_device)
            if target_device.type == "cuda"
            else None
        ),
    }
    return report


def _callbacks(
    output: Path,
    *,
    patience: int,
    enable_early_stopping: bool,
    divergence_reference_metric: float | None = None,
    divergence_multiplier: float = 1.5,
    divergence_patience: int = 2,
) -> list[pl.Callback]:
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=output / "checkpoints",
            filename="best-{epoch:03d}",
            monitor="val/velocity_mae",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if enable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val/velocity_mae",
                mode="min",
                patience=patience,
            )
        )
    if divergence_reference_metric is not None:
        callbacks.append(
            _DivergenceGuard(
                reference=divergence_reference_metric,
                multiplier=divergence_multiplier,
                patience=divergence_patience,
            )
        )
    return callbacks


def train_ispy2_biflow(
    config_path: str | Path,
    *,
    devices: str | int = "auto",
    resume: str | Path | None = None,
    max_epochs: int | None = None,
    max_steps: int = -1,
    enable_early_stopping: bool = True,
) -> Path:
    config = load_ispy2_biflow_config(config_path)
    preflight = preflight_ispy2_biflow(config.path)
    if not preflight["ready"]:
        raise RuntimeError("I-SPY2 BiFlowNet preflight is not ready")
    if max_epochs is not None and (type(max_epochs) is not int or max_epochs <= 0):
        raise ValueError("I-SPY2 BiFlowNet max_epochs must be positive")
    if type(max_steps) is not int or max_steps == 0 or max_steps < -1:
        raise ValueError("I-SPY2 BiFlowNet max_steps must be -1 or positive")
    pl.seed_everything(config.runtime.seed, workers=True)
    pairs, _, latent_cache, roi_cache = _world_data(config)
    model = _build_model(config)
    identity = _identity(
        config,
        model=model,
        latent_cache=latent_cache,
    )
    system = ISPY2BiFlowTrainingSystem(
        model, config=config, checkpoint_identity=identity
    )
    output = config.runtime.output_root
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if resume is None and config.runtime.warm_start_checkpoint is not None:
        warm_start = _warm_start_weights(
            system,
            checkpoint_path=config.runtime.warm_start_checkpoint,
            expected_sha256=str(config.runtime.warm_start_sha256),
            target_identity=identity,
        )
        (output / "warm_start.json").write_text(
            json.dumps(warm_start, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    parsed_devices: str | int = devices
    if isinstance(parsed_devices, str) and parsed_devices.isdigit():
        parsed_devices = int(parsed_devices)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=parsed_devices,
        max_epochs=max_epochs or config.training.max_epochs,
        max_steps=max_steps,
        precision=(
            config.training.precision if torch.cuda.is_available() else "32-true"
        ),
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        gradient_clip_val=config.training.gradient_clip_val,
        default_root_dir=output,
        callbacks=_callbacks(
            output,
            patience=config.training.early_stopping_patience,
            enable_early_stopping=enable_early_stopping,
            divergence_reference_metric=(
                config.training.divergence_reference_metric
            ),
            divergence_multiplier=config.training.divergence_multiplier,
            divergence_patience=config.training.divergence_patience,
        ),
        deterministic=True,
        log_every_n_steps=10,
    )
    train_dataset = _dataset(
        pairs, latent_cache, roi_cache, split="train"
    )
    val_dataset = _dataset(pairs, latent_cache, roi_cache, split="val")
    trainer.fit(
        system,
        _loader(train_dataset, config=config, shuffle=True),
        _loader(val_dataset, config=config, shuffle=False),
        ckpt_path=str(Path(resume).expanduser().resolve()) if resume else None,
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ispy2-biflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    profile = subparsers.add_parser("profile")
    profile.add_argument("--config", required=True)
    profile.add_argument("--device", default="cuda")
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--devices", default="auto")
    train.add_argument("--resume")
    train.add_argument("--max-epochs", type=int)
    train.add_argument("--max-steps", type=int, default=-1)
    train.add_argument("--disable-early-stopping", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight_ispy2_biflow(args.config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ready"] else 2
    if args.command == "profile":
        report = profile_ispy2_biflow(args.config, device=args.device)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    output = train_ispy2_biflow(
        args.config,
        devices=args.devices,
        resume=args.resume,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        enable_early_stopping=not args.disable_early_stopping,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ISPY2_BIFLOW_IDENTITY_SCHEMA",
    "ISPY2_BIFLOW_PREFLIGHT_SCHEMA",
    "build_parser",
    "main",
    "preflight_ispy2_biflow",
    "profile_ispy2_biflow",
    "train_ispy2_biflow",
]
