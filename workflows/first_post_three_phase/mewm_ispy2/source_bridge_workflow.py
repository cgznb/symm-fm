from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import argparse
import contextlib
import copy
import fcntl
import gc
import json
import math
import random
from collections import defaultdict
from dataclasses import replace
from itertools import chain
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from .source_bridge import (
    BRIDGE_SCHEMA,
    BridgeSystem,
    EpochSampler,
    channel_statistics,
    flow_path,
    public_metadata,
    write_json,
)


def load_experiment(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = _release_yaml(path.read_text(encoding="utf-8"))
    defaults = {
        "schema": BRIDGE_SCHEMA,
        "noise_multiplier": 0.25,
        "seed": 2026,
        "max_steps": 40000,
        "batch_size": 1,
        "num_workers": 2,
        "validation_steps": 8,
        "validation_samples": 2,
        "validation_interval": 2000,
        "validation_pairs_per_patient": 1,
        "activation_checkpointing": False,
        "phase_manifest": None,
    }
    required = {
        "family",
        "base_config",
        "statistics_file",
        "distribution",
        "accumulate_grad_batches",
        "output_root",
    }
    optional = {"numerical_policy", "attention_subvolume_batch", "continuation_checkpoint"}
    if (
        not isinstance(raw, dict)
        or not required <= set(raw)
        or set(raw) - (set(defaults) | required | optional)
    ):
        raise ValueError("Source-bridge experiment fields are invalid")
    result = {**defaults, **raw}
    if result["schema"] != BRIDGE_SCHEMA or result["family"] not in ("ispy2", "mu"):
        raise ValueError("Unsupported source-bridge experiment")
    if result["distribution"] not in ("source_gaussian", "standard_normal"):
        raise ValueError("Unsupported source-bridge distribution")
    for key in ("base_config", "statistics_file", "output_root"):
        value = result[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid experiment path: {key}")
        result[key] = str((path.parent / value).resolve())
    if result["phase_manifest"] is not None:
        if result["family"] != "ispy2" or not isinstance(result["phase_manifest"], str):
            raise ValueError("A phase-manifest override is only supported for I-SPY2")
        result["phase_manifest"] = str(
            (path.parent / result["phase_manifest"]).resolve()
        )
    for key in (
        "max_steps",
        "batch_size",
        "accumulate_grad_batches",
        "validation_steps",
        "validation_samples",
        "validation_interval",
        "validation_pairs_per_patient",
    ):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"Experiment {key} must be a positive integer")
    for key in ("seed", "num_workers"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"Experiment {key} must be nonnegative")
    if type(result["activation_checkpointing"]) is not bool:
        raise ValueError("activation_checkpointing must be boolean")
    if result["family"] == "ispy2" and result["activation_checkpointing"]:
        raise ValueError("I-SPY2 preset does not expose activation checkpointing")
    scale = result["noise_multiplier"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Noise multiplier must be finite and positive")
    if result["distribution"] == "standard_normal" and scale != 1:
        raise ValueError("Standard-normal baseline requires noise_multiplier=1")
    result["noise_multiplier"] = float(scale)
    if optional & set(result):
        if result.get("numerical_policy") != "local_dit_fp32_v1":
            raise ValueError("Unsupported bridge numerical policy")
        chunk = result.get("attention_subvolume_batch")
        if type(chunk) is not int or chunk <= 0:
            raise ValueError("FP32 attention_subvolume_batch must be a positive integer")
        if "continuation_checkpoint" in result:
            checkpoint = result["continuation_checkpoint"]
            if not isinstance(checkpoint, str) or not checkpoint.strip():
                raise ValueError("Invalid continuation checkpoint path")
            result["continuation_checkpoint"] = str((path.parent / checkpoint).resolve())
    return result


class SourceDataset(Dataset):
    def __init__(self, dataset: Any, family: str):
        self.dataset, self.family = dataset, family
        self.pairs = dataset.pairs

    def __len__(self):
        return len(self.dataset)

    def source(self, index: int):
        pair = self.pairs[index]
        if self.family == "ispy2":
            return self.dataset._target_latent(pair.source_visit_id)
        return self.dataset._latent(pair.source_samples)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if self.family == "ispy2":
            sample = {**sample, "source_latent": self.source(index)}
        return sample

    def source_key(self, index: int) -> str:
        pair = self.pairs[index]
        return (
            pair.source_visit_id
            if self.family == "ispy2"
            else f"{pair.patient_id}/{pair.source_timepoint}"
        )


def pair_key(pair: Any) -> str:
    return (
        pair.pair_id
        if hasattr(pair, "pair_id")
        else f"{pair.patient_id}__{pair.source_timepoint}__{pair.target_timepoint}"
    )


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        key: [sample[key] for sample in samples]
        if key == "metadata"
        else default_collate([sample[key] for sample in samples])
        for key in samples[0]
    }


class Project:
    def __init__(self, experiment: dict[str, Any]):
        self.experiment = experiment
        family = experiment["family"]
        if family == "ispy2":
            from .ispy2_biflow_config import load_ispy2_biflow_config
            from .ispy2_biflow_data import ISPY2BiFlowPairDataset
            from .ispy2_biflow_workflow import _build_model, _world_data

            self.base = load_ispy2_biflow_config(experiment["base_config"])
            if experiment["phase_manifest"] is not None:
                self.base = replace(
                    self.base,
                    base=replace(
                        self.base.base,
                        data=replace(
                            self.base.base.data,
                            phase_manifest_csv=Path(experiment["phase_manifest"]),
                        ),
                    ),
                )
            if self.base.runtime.warm_start_checkpoint is not None:
                raise ValueError("Bridge pilots require scratch initialization")
            pairs, _, self.latents, self.conditions = _world_data(self.base)
            self.dataset_factory = lambda items: SourceDataset(
                ISPY2BiFlowPairDataset(items, self.latents, self.conditions), family
            )
            self.model_factory = lambda: _build_model(self.base)
        else:
            from .mu_glioma_biflow_config import load_mu_glioma_biflow_config
            from .mu_glioma_biflow_data import MUGliomaBiFlowPairDataset
            from .mu_glioma_biflow_workflow import (
                _fixed_data,
                build_configured_mu_glioma_biflow_model,
            )

            self.base = load_mu_glioma_biflow_config(experiment["base_config"])
            self.base = replace(
                self.base,
                training=replace(
                    self.base.training,
                    activation_checkpointing=experiment["activation_checkpointing"],
                ),
            )
            pairs, _, self.latents, self.conditions, _ = _fixed_data(
                self.base, require_condition_cache=True
            )
            self.dataset_factory = lambda items: SourceDataset(
                MUGliomaBiFlowPairDataset(items, self.latents, self.conditions), family
            )
            self.model_factory = lambda: build_configured_mu_glioma_biflow_model(
                self.base
            )
        self.pairs = tuple(sorted(pairs, key=pair_key))
        self.train_pairs = tuple(pair for pair in self.pairs if pair.split == "train")
        self.val_pairs = tuple(pair for pair in self.pairs if pair.split == "val")
        train_patients = {pair.patient_id for pair in self.train_pairs}
        val_patients = {pair.patient_id for pair in self.val_pairs}
        if not train_patients or not val_patients or train_patients & val_patients:
            raise ValueError("Bridge experiment patient split is invalid")
        self.train_dataset = self.dataset_factory(self.train_pairs)
        self.val_dataset = self.dataset_factory(self.val_pairs)
        self.reference = self._reference()

    def _reference(self):
        base = self.base
        data = base.base.data if self.experiment["family"] == "ispy2" else base.data
        paths = [
            Path(self.experiment["base_config"]),
            data.vqgan_checkpoint,
            data.continuous_root / "cache_identity.json",
        ]
        if hasattr(base, "base"):
            paths += [base.base.path, data.bundle_json, data.phase_manifest_csv]
        else:
            paths += [
                data.clinical_timeline,
                data.condition_root / "cache_identity.json",
            ]
        assets = [
            {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in paths
        ]
        return public_metadata(
            {
                "config": base.raw,
                "data_config": base.base.raw if hasattr(base, "base") else {},
                "assets": assets,
            }
        )

    def validation_pairs(self):
        grouped = defaultdict(list)
        for pair in self.val_pairs:
            grouped[pair.patient_id].append(pair)
        generator = random.Random(self.experiment["seed"])
        selected = []
        for patient in sorted(grouped):
            pairs = grouped[patient]
            selected.extend(
                generator.sample(
                    pairs,
                    min(len(pairs), self.experiment["validation_pairs_per_patient"]),
                )
            )
        return tuple(sorted(selected, key=pair_key))

    def source_inventory(self):
        unique = {}
        for index in range(len(self.train_dataset)):
            unique.setdefault(self.train_dataset.source_key(index), index)
        return dict(sorted(unique.items()))

    def statistics(self, *, create: bool = False):
        path = Path(self.experiment["statistics_file"])
        inventory = self.source_inventory()
        identity = {
            "schema": "biflow_training_source_statistics_v1",
            "family": self.experiment["family"],
            "reference": self.reference,
            "source_visits": list(inventory),
            "weighting": "unique_train_source_visits_all_latent_voxels",
            "normalization": "continuous_codebook_minmax_v1",
        }
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if any(result.get(key) != value for key, value in identity.items()):
                raise ValueError(
                    "Calibration does not match this training cohort/assets"
                )
            std = torch.tensor(result["channel_std"])
            if not torch.isfinite(std).all() or not (std > 0).all():
                raise ValueError("Saved training standard deviations are invalid")
            return result
        if not create:
            raise FileNotFoundError("Run prepare to compute training-only noise scales")

        def sources():
            for count, index in enumerate(inventory.values(), 1):
                yield self.train_dataset.source(index)
                if count % 25 == 0 or count == len(inventory):
                    print(
                        f"calibration: {count}/{len(inventory)} training source visits",
                        flush=True,
                    )

        result = {**identity, **channel_statistics(sources())}
        write_json(path, result)
        return result

    def contract(self, statistics, *, smoke=False):
        settings = dict(self.experiment)
        selected = self.validation_pairs()
        if smoke:
            settings.update(
                max_steps=settings["max_steps"] if "continuation_checkpoint" in settings else 2,
                validation_interval=2,
                num_workers=0,
                output_root=str(Path(settings["output_root"]) / "smoke"),
            )
            selected = selected[:2]
        return {
            "schema": BRIDGE_SCHEMA,
            "experiment": settings,
            "reference": self.reference,
            "statistics": statistics,
            "pair_order": [pair_key(pair) for pair in self.pairs],
            "train_pairs": [pair_key(pair) for pair in self.train_pairs],
            "validation_pairs": [pair_key(pair) for pair in selected],
            "initialization": "scratch",
            "velocity_loss": "l1_equal_channels",
            "checkpoint_metric": "patient_macro_endpoint_latent_mae",
            "precision": "bf16-mixed",
            "sampler": "epoch_seeded_permutation_v1",
        }


def optimizer_factory(system: BridgeSystem):
    if system.family == "ispy2":
        from .ispy2_biflow_training import ISPY2BiFlowTrainingSystem

        return ISPY2BiFlowTrainingSystem.configure_optimizers(system)
    from .mu_glioma_biflow_training import (
        MUGliomaBiFlowTrainingSystem,
        build_mu_glioma_biflow_scheduler,
    )

    groups = MUGliomaBiFlowTrainingSystem.optimizer_parameter_groups(system)
    training = system.experiment_config.training
    optimizer = torch.optim.AdamW(groups, betas=training.optimizer_betas)
    scheduler = build_mu_glioma_biflow_scheduler(
        optimizer,
        total_steps=system.contract["experiment"]["max_steps"],
        warmup_fraction=training.warmup_fraction,
        minimum_learning_rate=training.minimum_learning_rate,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }


def build_system(project: Project, contract: dict[str, Any]):
    model = project.model_factory()
    settings = contract["experiment"]
    if settings.get("numerical_policy") == "local_dit_fp32_v1":
        from .ispy2_biflow_backbone import enable_local_dit_fp32

        enable_local_dit_fp32(
            model.dynamics, subvolume_batch=settings["attention_subvolume_batch"]
        )
    return BridgeSystem(
        model,
        family=project.experiment["family"],
        experiment_config=project.base,
        contract=contract,
        optimizer_factory=optimizer_factory,
    )


def prepare(project: Project):
    if project.experiment["family"] == "ispy2":
        from .ispy2_biflow_preflight import audit_ispy2_biflow_training_assets

        assets = audit_ispy2_biflow_training_assets(
            project.base,
            pairs=project.pairs,
            latent_cache=project.latents,
            roi_cache=project.conditions,
        )
        if not assets["ready"]:
            raise RuntimeError(
                "BiFlow training assets are incomplete: "
                + json.dumps(public_metadata(assets))
            )
    else:
        from .ispy2_biflow_preflight import (
            _MEDGEMMA_REQUIRED_FILES,
            _huggingface_hub_root,
        )

        text = project.base.text
        snapshot = (
            _huggingface_hub_root()
            / f"models--{text.model_id.replace('/', '--')}"
            / "snapshots"
            / text.revision
        )
        if any(not (snapshot / name).is_file() for name in _MEDGEMMA_REQUIRED_FILES):
            raise FileNotFoundError("The frozen MedGemma snapshot is incomplete")
        samples = {
            sample.sample_id: sample
            for pair in project.pairs
            for sample in (*pair.source_samples, *pair.target_samples)
        }
        if any(
            not (
                project.latents.root
                / "volumes"
                / sample.split
                / f"{sample.sample_id}.pt"
            ).is_file()
            for sample in samples.values()
        ):
            raise FileNotFoundError(
                "Required MU source/target latent payload is missing"
            )
        assets = {
            "ready": True,
            "required_latent_payloads": len(samples),
            "source_conditions": project.conditions.validate_inventory(
                validate_payloads=False
            ),
            "medgemma_files_present": True,
        }
    statistics = project.statistics(create=True)
    probes = []
    for dataset in (project.train_dataset, project.val_dataset):
        batch = collate([dataset[0]])
        source, target = batch["source_latent"], batch["target_latent"]
        generator = torch.Generator().manual_seed(project.experiment["seed"])
        noise = torch.randn(source.shape, generator=generator)
        settings = {
            "distribution": project.experiment["distribution"],
            "multiplier": project.experiment["noise_multiplier"],
            "channel_std": torch.tensor(statistics["channel_std"]),
        }
        state0, velocity = flow_path(source, target, noise, torch.zeros(1), **settings)
        state1, _ = flow_path(source, target, noise, torch.ones(1), **settings)
        if not torch.equal(state1, target) or not torch.allclose(
            state0 + velocity, target, atol=2e-6, rtol=2e-6
        ):
            raise RuntimeError("Real-data bridge endpoint checks failed")
        probes.append(
            {
                "latent_shape": list(source.shape),
                "source_mri_shape": list(batch["source_mri"].shape),
                "initial_to_source_mae": (state0 - source).abs().mean().item(),
                "velocity_mae": velocity.abs().mean().item(),
            }
        )
    result = {
        "status": "prepared",
        "family": project.experiment["family"],
        "train_pairs": len(project.train_pairs),
        "val_pairs": len(project.val_pairs),
        "validation_selection_pairs": len(project.validation_pairs()),
        "calibration_visits": statistics["visits"],
        "channel_std_range": [
            min(statistics["channel_std"]),
            max(statistics["channel_std"]),
        ],
        "real_data_probes": probes,
        "training_assets": public_metadata(assets),
    }
    write_json(Path(project.experiment["output_root"]) / "preflight.json", result)
    return result


@contextlib.contextmanager
def run_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class FinalValidationCheckpoint(ModelCheckpoint):
    def save_final_validation(self, trainer: pl.Trainer):
        candidates = self._monitor_candidates(trainer)
        score = candidates.get(self.monitor)
        if score is None or not torch.isfinite(score).all():
            raise ValueError(
                "Final checkpoint selection requires a finite validation score"
            )
        # Lightning skips its normal checkpoint hook outside trainer.fit().
        self._save_topk_checkpoint(trainer, candidates)
        self._save_last_checkpoint(trainer, candidates)


class BridgeProgress(Callback):
    def on_exception(self, trainer, pl_module, exception):
        write_json(
            Path(pl_module.contract["experiment"]["output_root"]) / "progress.json",
            {
                "status": "interrupted" if isinstance(exception, KeyboardInterrupt) else "failed",
                "optimizer_step": trainer.global_step,
                "epoch": trainer.current_epoch,
                "error_type": type(exception).__name__,
                "gradient_norms": getattr(pl_module, "latest_gradient_norms", {}),
            },
        )


def validate_numerical_continuation(parent: dict, current: dict):
    expected = copy.deepcopy(current)
    settings = expected["experiment"]
    if (
        settings.pop("numerical_policy", None) != "local_dit_fp32_v1"
        or type(settings.pop("attention_subvolume_batch", None)) is not int
        or not settings.pop("continuation_checkpoint", None)
        or settings["output_root"] == parent["experiment"]["output_root"]
    ):
        raise ValueError("Continuation requires an explicit new FP32 run")
    settings["output_root"] = parent["experiment"]["output_root"]
    if expected != parent:
        raise ValueError("Numerical continuation changed the experiment contract")


def migrate_continuation_checkpoint(project: Project, contract: dict) -> tuple[str, int]:
    source = Path(project.experiment["continuation_checkpoint"])
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("source_bridge_schema") != BRIDGE_SCHEMA:
        raise ValueError("Continuation requires a source-bridge checkpoint")
    validate_numerical_continuation(
        payload["source_bridge_contract"],
        project.contract(contract["statistics"]),
    )
    step = payload["global_step"]
    if not 0 < step < project.experiment["max_steps"] or not payload.get("optimizer_states"):
        raise ValueError("Continuation requires unfinished training and optimizer state")
    provenance = {
        "parent_checkpoint": str(source.resolve()),
        "parent_optimizer_step": step,
        "parent_epoch": payload["epoch"],
        "numerical_policy": project.experiment["numerical_policy"],
        "attention_subvolume_batch": project.experiment["attention_subvolume_batch"],
        "total_optimizer_budget": project.experiment["max_steps"],
        "restored": ["model", "optimizer", "scheduler", "loops", "rng"],
        "checkpoint_selection": "reset_for_new_numerical_policy",
        "partial_accumulation": "unsaved pending gradients are not recoverable",
    }
    # Reset selection paths/scores before Lightning can remove any parent checkpoints.
    payload["callbacks"] = {}
    payload["source_bridge_contract"] = contract
    payload["source_bridge_continuation"] = provenance
    root = Path(contract["experiment"]["output_root"])
    destination = root / "checkpoints" / "last.ckpt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    write_json(root / "continuation.json", provenance)
    return str(destination), step


def train(project: Project, *, smoke=False, resume: str | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError("Full BiFlow training requires an available CUDA device")
    torch.cuda.reset_peak_memory_stats()
    contract = project.contract(project.statistics(), smoke=smoke)
    settings = contract["experiment"]
    root = Path(settings["output_root"])
    initial_step = 0
    with run_lock(root):
        identity_path = root / "run_contract.json"
        if resume is None:
            if identity_path.exists() or (root / "checkpoints").exists():
                raise FileExistsError("Fresh training requires an unused run directory")
        elif (
            not identity_path.is_file()
            or json.loads(identity_path.read_text()) != contract
        ):
            raise ValueError("Resume requires the exact saved experiment contract")
        if resume is None and "continuation_checkpoint" in settings:
            resume, initial_step = migrate_continuation_checkpoint(project, contract)
            write_json(identity_path, contract)
        elif resume is not None:
            payload = torch.load(resume, map_location="cpu", weights_only=True, mmap=True)
            initial_step = payload["global_step"]
            del payload
        pl.seed_everything(settings["seed"], workers=True)
        system = build_system(project, contract)
        train_loader = DataLoader(
            project.train_dataset,
            batch_size=settings["batch_size"],
            sampler=EpochSampler(len(project.train_dataset), settings["seed"]),
            collate_fn=collate,
            num_workers=settings["num_workers"],
            pin_memory=True,
            persistent_workers=settings["num_workers"] > 0,
        )
        keys = set(contract["validation_pairs"])
        selected = tuple(pair for pair in project.val_pairs if pair_key(pair) in keys)
        val_loader = DataLoader(
            project.dataset_factory(selected),
            batch_size=1,
            collate_fn=collate,
            num_workers=settings["num_workers"],
            pin_memory=True,
        )
        checkpoint = FinalValidationCheckpoint(
            dirpath=root / "checkpoints",
            filename="best",
            monitor="val/endpoint_mae",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
            enable_version_counter=False,
            save_on_train_epoch_end=False,
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=-1,
            max_steps=initial_step + 2 if smoke else settings["max_steps"],
            precision="bf16-mixed",
            accumulate_grad_batches=settings["accumulate_grad_batches"],
            gradient_clip_val=project.base.training.gradient_clip_val,
            callbacks=[BridgeProgress()] if smoke else [BridgeProgress(), checkpoint],
            enable_checkpointing=not smoke,
            logger=CSVLogger(root, name="metrics"),
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            val_check_interval=settings["validation_interval"]
            * settings["accumulate_grad_batches"],
            check_val_every_n_epoch=None,
            log_every_n_steps=1 if smoke else 25,
            default_root_dir=root,
        )
        write_json(identity_path, contract)
        write_json(root / "progress.json", {"status": "training", "optimizer_step": initial_step})
        trainer.fit(system, train_loader, val_loader, ckpt_path=resume)
        trainer.validate(system, val_loader, verbose=False)
        if smoke:
            (root / "checkpoints").mkdir(exist_ok=True)
            trainer.save_checkpoint(root / "checkpoints" / "last.ckpt")
        else:
            checkpoint.save_final_validation(trainer)
        reloaded = torch.load(
            root / "checkpoints" / "last.ckpt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        system.on_load_checkpoint(reloaded)
        if any(
            not torch.isfinite(value).all()
            for value in chain(
                reloaded["state_dict"].values(),
                (
                    value
                    for optimizer in reloaded["optimizer_states"]
                    for state in optimizer["state"].values()
                    for value in state.values()
                ),
            )
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ):
            raise FloatingPointError(
                "Saved bridge checkpoint contains non-finite state"
            )
        del reloaded
        result = {
            "status": "completed",
            "initial_optimizer_step": initial_step,
            "new_optimizer_steps": trainer.global_step - initial_step,
            "optimizer_steps": trainer.global_step,
            "epochs": trainer.current_epoch,
            "best_checkpoint": checkpoint.best_model_path,
            "best_validation_mae": float(checkpoint.best_model_score)
            if checkpoint.best_model_score is not None
            else None,
            "last_checkpoint": str(root / "checkpoints" / "last.ckpt"),
            "last_validation": system.latest_validation,
            "gradient_norms": getattr(system, "latest_gradient_norms", {}),
            "restored_training_state": system.restored_training_state,
            "checkpoint_readback": "passed",
            "optimizer_checkpoint_readback": "passed",
            "gpu_name": torch.cuda.get_device_name(),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
        if smoke:
            (root / "checkpoints" / "last.ckpt").unlink()
            result["last_checkpoint"] = None
            result["best_checkpoint"] = None
        write_json(root / "training_result.json", result)
        write_json(root / "progress.json", result)
        del trainer, system
        gc.collect()
        torch.cuda.empty_cache()
        return result


def evaluate(project: Project, checkpoint: str, *, samples=4, steps=8):
    contract = project.contract(project.statistics())
    root = (
        Path(project.experiment["output_root"]) / f"evaluation_euler{steps}_k{samples}"
    )
    if (root / "summary.json").exists():
        raise FileExistsError("Evaluation is already complete")
    pl.seed_everything(project.experiment["seed"], workers=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    system = build_system(project, contract)
    system.on_load_checkpoint(payload)
    incompatible = system.load_state_dict(payload["state_dict"], strict=False)
    if (
        not set(incompatible.missing_keys) <= set(system.omitted_keys())
        or incompatible.unexpected_keys
    ):
        raise ValueError("Bridge prediction weights are incomplete")
    system.to("cuda").eval()
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidates").mkdir(exist_ok=True)
    rows = []
    # Fixed representatives are selected using interval metadata, before any predictions.
    ordered = sorted(
        project.val_pairs, key=lambda pair: (pair.delta_days, pair_key(pair))
    )
    representatives = {
        pair_key(ordered[index]): slot
        for slot, index in enumerate((0, len(ordered) // 2, len(ordered) - 1))
    }
    for index in range(len(project.val_dataset)):
        batch = collate([project.val_dataset[index]])
        batch = {
            key: value.cuda() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = system.predict_batch(batch, steps=steps, samples=samples)
        mean = predictions.mean(0)
        target, source = batch["target_latent"], batch["source_latent"]
        metadata = batch["metadata"][0]
        torch.save(
            {"pair_id": metadata["pair_id"], "samples": predictions.cpu()},
            root / "candidates" / f"{index:05d}.pt",
        )
        row = {
            "pair_id": metadata["pair_id"],
            "patient_id": metadata["patient_id"],
            "delta_days": float(batch["delta_days"][0]),
            "endpoint_mae": (mean - target).abs().mean().item(),
            "endpoint_mse": (mean - target).square().mean().item(),
            "source_copy_mae": (source - target).abs().mean().item(),
            "sample_std": predictions.std(0, correction=0).mean().item(),
        }
        rows.append(row)
        if metadata["pair_id"] in representatives:
            torch.save(
                {
                    "pair_id": metadata["pair_id"],
                    "mean_latent": mean.cpu(),
                    "source_latent": source.cpu(),
                    "target_latent": target.cpu(),
                    "sample_std_latent": predictions.std(0, correction=0).cpu(),
                },
                root / f"representative_{representatives[metadata['pair_id']]}.pt",
            )
        if (index + 1) % 10 == 0:
            print(
                f"evaluated {index + 1}/{len(project.val_dataset)} validation pairs",
                flush=True,
            )
    patients = defaultdict(list)
    for row in rows:
        patients[row["patient_id"]].append(row)
    keys = ("endpoint_mae", "endpoint_mse", "source_copy_mae", "sample_std")
    metrics = {
        key: sum(
            sum(row[key] for row in values) / len(values)
            for values in patients.values()
        )
        / len(patients)
        for key in keys
    }
    write_json(root / "pair_metrics.json", rows)
    summary = {
        "status": "completed",
        "checkpoint": str(Path(checkpoint).resolve()),
        "pairs": len(rows),
        "patients": len(patients),
        "steps": steps,
        "samples": samples,
        "patient_macro": metrics,
        "metric_space": "normalized_latent",
        "distribution": system.settings["distribution"],
    }
    write_json(root / "generation_summary.json", summary)
    del system, payload, predictions, mean, batch, target, source
    gc.collect()
    torch.cuda.empty_cache()
    from .source_bridge_evaluation import decode_and_evaluate

    image_summary = decode_and_evaluate(project, root, representatives)
    summary["image_evaluation"] = "image_summary.json"
    summary["image_pairs"] = image_summary["pairs"]
    write_json(root / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Matched BiFlow source-bridge experiments"
    )
    parser.add_argument("command", choices=("prepare", "train", "smoke", "evaluate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    project = Project(load_experiment(args.config))
    if args.command == "prepare":
        result = prepare(project)
    elif args.command in ("train", "smoke"):
        result = train(project, smoke=args.command == "smoke", resume=args.resume)
    else:
        if not args.checkpoint:
            parser.error("evaluate requires --checkpoint")
        result = evaluate(project, args.checkpoint)
    print(json.dumps(public_metadata(result), indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
