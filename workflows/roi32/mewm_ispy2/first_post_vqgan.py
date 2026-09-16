from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from .first_post_data import (
    NUMERIC_CONTRACT,
    FirstPostDataset,
    prepare,
    prepare_cohort,
    read_config,
    timestamp,
    write_json,
)
from .perceptual import UncheckedLPIPSLoss
from .vqgan import (
    REGISTERED_VQGAN_NUMERIC_CONTRACT,
    MRILevelVQGAN,
    VQGANTrainingSystem,
    vqgan_config_from_checkpoint_identity,
)

REPO = Path(__file__).resolve().parents[1]
GIB = 1024**3


def resources(root: Path) -> dict[str, Any]:
    cgroup = Path("/sys/fs/cgroup")
    maximum = (cgroup / "memory.max").read_text().strip()
    current = int((cgroup / "memory.current").read_text())
    stats = dict(
        line.split() for line in (cgroup / "memory.stat").read_text().splitlines()
    )
    headroom = (
        None
        if maximum == "max"
        else int(maximum) - current + int(stats.get("inactive_file", 0))
    )
    full = next(
        line
        for line in (cgroup / "memory.pressure").read_text().splitlines()
        if line.startswith("full ")
    )
    pressure = float(dict(item.split("=") for item in full.split()[1:])["avg10"])
    return {
        "memory_headroom_bytes": headroom,
        "disk_free_bytes": shutil.disk_usage(root).free,
        "memory_pressure_full_avg10": pressure,
    }


def resource_reasons(
    snapshot: dict[str, Any], config: dict[str, Any], *, runtime: bool
) -> list[str]:
    prefix = "runtime" if runtime else "admission"
    reasons = []
    if (
        snapshot["memory_headroom_bytes"] is not None
        and snapshot["memory_headroom_bytes"]
        < config["queue"][f"{prefix}_memory_gib"] * GIB
    ):
        reasons.append("host_memory")
    if snapshot["disk_free_bytes"] < config["queue"][f"{prefix}_disk_gib"] * GIB:
        reasons.append("disk_space")
    if snapshot["memory_pressure_full_avg10"] > 1.0:
        reasons.append("host_memory_pressure")
    return reasons


class FirstPostSystem(VQGANTrainingSystem):
    def configure_optimizers(self):
        optimizers, schedules = super().configure_optimizers()
        milestones = (
            self.checkpoint_identity.get("configuration", {})
            .get("training", {})
            .get("scheduler_milestones")
        )
        if milestones is not None:
            schedules = [
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=milestones, gamma=gamma
                )
                for optimizer, gamma in zip(optimizers, (0.5, 0.1), strict=True)
            ]
        return optimizers, schedules

    def _perceptual_view(self, image_slice: torch.Tensor) -> torch.Tensor:
        return image_slice.repeat(1, 3, 1, 1).clamp(-1.0, 1.0)

    @property
    def training_contract(self) -> dict[str, Any]:
        return {
            **super().training_contract,
            "schema_version": "ispy2_first_post_vqgan_training_v1",
            "perceptual_input": "clamped_zscore_neg_one_one",
            "validation_slice_seed": 2026,
            "generator_update_discriminator_parameters": "frozen",
        }

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        image = batch["image"]
        generator, discriminator = self.optimizers()
        generator_schedule, discriminator_schedule = self.lr_schedulers()
        step = self.global_step
        self.toggle_optimizer(generator)
        generator.zero_grad(set_to_none=True)
        losses = self.compute_losses(image, global_step=step)
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError("Non-finite generator loss")
        self.manual_backward(losses["generator_total"])
        self.clip_gradients(
            generator,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )
        generator.step()
        generator_schedule.step()
        self.untoggle_optimizer(generator)
        self.toggle_optimizer(discriminator)
        discriminator.zero_grad(set_to_none=True)
        discriminator_loss = self.compute_discriminator_losses(
            image.detach(), global_step=step
        )["total"]
        if not torch.isfinite(discriminator_loss):
            raise FloatingPointError("Non-finite discriminator loss")
        self.manual_backward(discriminator_loss)
        self.clip_gradients(
            discriminator,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )
        discriminator.step()
        discriminator_schedule.step()
        self.untoggle_optimizer(discriminator)
        metrics = {f"train/{key}": value.detach() for key, value in losses.items()}
        metrics["train/discriminator_total"] = discriminator_loss.detach()
        self.log_dict(metrics, on_step=True, on_epoch=True, batch_size=image.shape[0])
        return losses["generator_total"].detach()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(2026 + batch_idx)
            if devices:
                torch.cuda.manual_seed(2026 + batch_idx)
            super().validation_step(batch, batch_idx)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["first_post_run_contract"] = self.checkpoint_identity

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("first_post_run_contract") != self.checkpoint_identity:
            raise ValueError("First-post resume contract changed")


def build_system(
    config: dict[str, Any], *, smoke: bool = False, cpu: bool = False
) -> FirstPostSystem:
    path = Path(config["initial_checkpoint"])
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    identity = payload["mewm_ispy2_vqgan_identity"]
    if (
        identity.get("mri_finetuned") is not True
        or identity.get("numeric_contract") != REGISTERED_VQGAN_NUMERIC_CONTRACT
    ):
        raise ValueError("Initialization is not the expected DCE0 MRI checkpoint")
    model = MRILevelVQGAN(vqgan_config_from_checkpoint_identity(identity))
    normalization = json.loads(
        (Path(config["output_dir"]) / "data/normalization.json").read_text()
    )
    stat = path.stat()
    contract = {
        "numeric_contract": NUMERIC_CONTRACT,
        "data_backend": "unregistered_first_post",
        "phase_index": 1,
        "registered": False,
        "normalization": normalization,
        "configuration": config,
        "smoke": smoke,
        "cpu_smoke": cpu,
        "initialization_method": "dce0_all_model_weights_fresh_optimizers",
        "discriminator_initialization": "dce0_checkpoint",
        "source_checkpoint": {
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "epoch": payload["epoch"],
            "global_step": payload["global_step"],
        },
    }
    training = config["training"]
    system = FirstPostSystem(
        model,
        learning_rate=training["generator_learning_rate"],
        discriminator_learning_rate=training["discriminator_learning_rate"],
        adversarial_always_on=True,
        batch_size=training["batch_size"],
        precision="32-true" if cpu else training["precision"],
        max_epochs=1 if smoke else training["max_epochs"],
        early_stopping_patience=training["early_stopping_patience"],
        perceptual_model=UncheckedLPIPSLoss.vgg(),
        checkpoint_identity=contract,
        **{
            key: training[key]
            for key in (
                "reconstruction_weight",
                "perceptual_weight",
                "image_gan_weight",
                "volume_gan_weight",
                "feature_matching_weight",
                "gradient_clip_val",
            )
        },
    )
    system.load_state_dict(payload["state_dict"], strict=True)
    for key, value in system.state_dict().items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite initialization: {key}")
    return system


class RunMonitor(pl.Callback):
    def __init__(self, config: dict[str, Any], output: Path, *, smoke: bool) -> None:
        self.config, self.output, self.smoke = config, output, smoke
        self.stop_reason: str | None = None
        self.violations = 0
        self.started = time.monotonic()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer, pl_module
        signal.signal(
            signal.SIGUSR1, lambda *_: setattr(self, "stop_reason", "requested_pause")
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch
        if batch_idx % 25 == 0 or self.stop_reason:
            snapshot = resources(self.output)
            reasons = resource_reasons(snapshot, self.config, runtime=True)
            self.violations = self.violations + 1 if reasons else 0
            if self.violations >= 2:
                self.stop_reason = ",".join(reasons)
            write_json(
                self.output / "progress.json",
                {
                    "status": "pausing" if self.stop_reason else "training",
                    "pid": os.getpid(),
                    "epoch": trainer.current_epoch,
                    "generator_updates": trainer.global_step // 2,
                    "optimizer_steps": trainer.global_step,
                    "batch_index": batch_idx,
                    "elapsed_seconds": round(time.monotonic() - self.started, 2),
                    "updated_at_utc": timestamp(),
                    "resources": snapshot,
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else 0,
                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()
                    if torch.cuda.is_available()
                    else 0,
                    "metrics": {
                        k: float(v)
                        for k, v in trainer.callback_metrics.items()
                        if torch.isfinite(v)
                    },
                },
            )
        if not self.smoke and (self.stop_reason or (batch_idx + 1) % 500 == 0):
            trainer.save_checkpoint(self.output / "checkpoints/last.ckpt")
        if self.stop_reason:
            trainer.should_stop = True

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: FirstPostSystem,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del outputs, dataloader_idx
        if (
            batch_idx
            or trainer.sanity_checking
            or (trainer.current_epoch % 5 and not self.smoke)
        ):
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        image = batch["image"]
        with torch.no_grad():
            reconstructed, _ = pl_module.autoencoder(image)
        real, fake = (
            image[0, 0].float().cpu().numpy(),
            reconstructed[0, 0].float().cpu().numpy(),
        )
        figures, axes = plt.subplots(3, 3, figsize=(10, 9))
        low, high = np.percentile(real[real != 0], [1, 99])
        spacing = list(reversed(self.config["data"]["target_spacing_xyz"]))
        for axis in range(3):
            index = real.shape[axis] // 2
            plane = [dimension for dimension in range(3) if dimension != axis]
            for column, volume in enumerate((real, fake, np.abs(real - fake))):
                view = np.take(volume, index, axis=axis)
                axes[axis, column].imshow(
                    view,
                    cmap="magma" if column == 2 else "gray",
                    origin="lower",
                    vmin=0 if column == 2 else low,
                    vmax=max(high - low, 1e-6) if column == 2 else high,
                    aspect=spacing[plane[0]] / spacing[plane[1]],
                )
                axes[axis, column].axis("off")
        for axis, title in zip(
            axes[0], ("First post", "Reconstruction", "Absolute error"), strict=True
        ):
            axis.set_title(title)
        figures.tight_layout()
        destination = self.output / "reconstructions"
        destination.mkdir(exist_ok=True)
        figures.savefig(destination / f"epoch_{trainer.current_epoch:03d}.png", dpi=100)
        plt.close(figures)


def require_training_enabled(config: dict[str, Any]) -> None:
    if config["training"].get("enabled", True) is not True:
        raise ValueError(
            "First-post training is not ready: " + config["training"]["disabled_reason"]
        )


def train(
    config: dict[str, Any], *, smoke: bool = False, cpu: bool = False
) -> dict[str, Any]:
    require_training_enabled(config)
    if config["data"].get("training_crop") == "tumor_union_adaptive":
        from .first_post_vqgan_performance import selected_config

        base = Path(config["output_dir"])
        review = read_status(base / "data/crop_review.json")
        if review.get("status") != "passed" or review.get(
            "preparation_config"
        ) != read_status(base / "data/manifest.json").get("preparation_config"):
            raise ValueError("Tumor crop technical review has not passed")
        config = selected_config(
            config, read_status(base / "performance/selection.json")
        )
    if cpu and not smoke:
        raise ValueError("Formal training requires a GPU")
    torch.set_num_threads(4)
    pl.seed_everything(config["seed"], workers=True)
    root = Path(config["output_dir"])
    output = root / ("cpu_smoke" if cpu else "gpu_smoke") if smoke else root
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    shape = [32, 32, 32] if cpu else None
    training_data = FirstPostDataset(config, "train", shape=shape)
    validation_data = FirstPostDataset(config, "val", shape=shape)
    workers = (
        0
        if cpu
        else config["training"].get("loader_workers", config["data"]["num_workers"])
    )
    options = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": workers,
        "pin_memory": not cpu,
    }
    if workers:
        options.update(
            prefetch_factor=config["training"].get("prefetch_factor", 1),
            persistent_workers=True,
        )
    training_loader = DataLoader(training_data, shuffle=True, **options)
    validation_loader = DataLoader(validation_data, shuffle=False, **options)
    system = build_system(config, smoke=smoke, cpu=cpu)
    monitor = RunMonitor(config, output, smoke=smoke)
    checkpoints = ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="best",
        monitor="val/composite",
        mode="min",
        save_top_k=1,
        save_last=True,
        enable_version_counter=False,
    )
    callbacks: list[pl.Callback] = [
        monitor,
        EarlyStopping(
            monitor="val/composite",
            mode="min",
            check_finite=True,
            patience=config["training"]["early_stopping_patience"],
            min_delta=config["training"]["early_stopping_min_delta"],
        ),
        checkpoints,
    ]
    trainer = pl.Trainer(
        default_root_dir=output,
        accelerator="cpu" if cpu else "gpu",
        devices=1,
        precision="32-true" if cpu else config["training"]["precision"],
        max_epochs=1 if smoke else config["training"]["max_epochs"],
        limit_train_batches=2 if smoke else 1.0,
        limit_val_batches=2 if smoke else 1.0,
        num_sanity_val_steps=0 if smoke else 2,
        log_every_n_steps=1 if smoke else 25,
        enable_progress_bar=False,
        logger=CSVLogger(output, name="metrics", version=0),
        callbacks=callbacks,
    )
    resume = output / "checkpoints/last.ckpt"
    contract_path = output / "run_contract.json"
    if (
        contract_path.exists()
        and json.loads(contract_path.read_text()) != system.checkpoint_identity
    ):
        raise ValueError("Refusing to overwrite a different run")
    write_json(contract_path, system.checkpoint_identity)
    tracked = (
        "autoencoder.encoder.input.weight",
        "image_discriminator.blocks.0.0.weight",
        "volume_discriminator.blocks.0.0.weight",
        "autoencoder.quantizer.embeddings",
    )
    before = {key: system.state_dict()[key].clone() for key in tracked} if smoke else {}
    if not cpu:
        torch.cuda.reset_peak_memory_stats()
    trainer.fit(
        system,
        training_loader,
        validation_loader,
        ckpt_path=str(resume) if resume.exists() and not smoke else None,
    )
    finite = all(
        not v.is_floating_point() or bool(torch.isfinite(v).all())
        for v in system.state_dict().values()
    )
    updates = {
        key: not torch.equal(value, system.state_dict()[key].cpu())
        for key, value in before.items()
    }
    if not finite or (
        smoke
        and (
            not all(updates.values()) or trainer.global_step != 4 or monitor.stop_reason
        )
    ):
        raise RuntimeError(
            "Training verification failed: non-finite state or missing optimizer updates"
        )
    trainer.save_checkpoint(resume)
    saved = torch.load(resume, map_location="cpu", weights_only=False, mmap=True)
    system.on_load_checkpoint(saved)
    if (
        saved["global_step"] != trainer.global_step
        or len(saved["optimizer_states"]) != 2
    ):
        raise RuntimeError("Saved training state is incomplete")
    if set(saved["state_dict"]) != set(system.state_dict()) or any(
        not torch.equal(value.cpu(), saved["state_dict"][key])
        for key, value in system.state_dict().items()
    ):
        raise RuntimeError("Checkpoint readback does not match the trained weights")
    result = {
        "status": "paused"
        if monitor.stop_reason
        else "passed"
        if smoke
        else "completed",
        "completed_at_utc": timestamp(),
        "pid": os.getpid(),
        "epoch": trainer.current_epoch,
        "optimizer_steps": trainer.global_step,
        "generator_updates": trainer.global_step // 2,
        "pause_reason": monitor.stop_reason,
        "finite_model_state": finite,
        "smoke": smoke,
        "input_shape_zyx": shape or config["data"]["output_shape_zyx"],
        "updated_parameter_groups": updates,
        "checkpoint_readback_matches": True,
        "checkpoint": str(resume),
        "best_checkpoint": checkpoints.best_model_path,
        "best_composite": float(checkpoints.best_model_score)
        if checkpoints.best_model_score is not None
        else None,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if not cpu else 0,
    }
    write_json(output / "training_result.json", result)
    write_json(output / "progress.json", result)
    return result


def read_status(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def queue_alive(manifest: Path, status: dict[str, Any]) -> bool:
    if status.get("status") in ("completed", "failed", "interrupted"):
        return False
    try:
        return (
            str(manifest).encode()
            in Path(f"/proc/{int(status['pid'])}/cmdline").read_bytes()
        )
    except (KeyError, FileNotFoundError, ValueError):
        return False


def gpu_reserved(gpu: int, config: dict[str, Any]) -> bool:
    queue = config["queue"]
    if gpu == 0:
        manifest = Path(queue["reserved_gpu0_queue"])
        if queue_alive(manifest, read_status(manifest.parent / "status.json")):
            return True
    for name in queue["prior_managed_queues"]:
        manifest = Path(name)
        status = read_status(manifest.parent / "status.json")
        if (
            queue_alive(manifest, status)
            and status.get("status") == "waiting_for_resources"
        ):
            return True
    return False


def gpu_idle(gpu: int) -> bool:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=20,
    )
    memory, utilization = (int(value) for value in output.strip().split(","))
    return memory < 1024 and utilization < 10


def run_child(
    stage: str, config_path: Path, root: Path, *, gpu: int | None = None
) -> None:
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
    )
    environment["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    command = [
        sys.executable,
        "-u",
        "-m",
        "mewm_ispy2.first_post_vqgan",
        stage,
        "--config",
        str(config_path),
    ]
    with (root / f"{stage}.log").open("a") as log:
        child = subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        write_json(
            root / "queue_status.json",
            {
                "status": "running",
                "stage": stage,
                "pid": os.getpid(),
                "child_pid": child.pid,
                "gpu": gpu,
                "updated_at_utc": timestamp(),
            },
        )
        try:
            returncode = child.wait()
        except BaseException:
            if child.poll() is None:
                child.send_signal(
                    signal.SIGUSR1 if stage == "train" else signal.SIGTERM
                )
                child.wait()
            raise
    if returncode:
        raise RuntimeError(
            f"{stage} exited with code {returncode}; inspect {root / (stage + '.log')}"
        )


def run_queue(config_path: Path) -> None:
    config = read_config(config_path)
    require_training_enabled(config)
    tumor = config["data"].get("training_crop") == "tumor_union_adaptive"
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)

    def interrupt(*_: Any) -> None:
        raise KeyboardInterrupt("Queue interrupted")

    signal.signal(signal.SIGTERM, interrupt)
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read_status(root / "training_result.json").get("status") == "completed":
            expected = config
            if tumor:
                from .first_post_vqgan_performance import selected_config

                expected = selected_config(
                    config, read_status(root / "performance/selection.json")
                )
            if read_status(root / "run_contract.json").get("configuration") != expected:
                raise ValueError("Completed run has a different configuration")
            write_json(
                root / "queue_status.json",
                {
                    "status": "completed",
                    "pid": os.getpid(),
                    "updated_at_utc": timestamp(),
                },
            )
            return
        try:
            run_child("prepare", config_path, root)
            if not tumor and (
                read_status(root / "cpu_smoke/training_result.json").get("status")
                != "passed"
            ):
                run_child("cpu-smoke", config_path, root)
            lock_root = Path(config["queue"]["lock_root"])
            lock_root.mkdir(parents=True, exist_ok=True)
            next_cache_release = 0.0
            while True:
                snapshot = resources(root)
                reasons = resource_reasons(snapshot, config, runtime=False)
                if (
                    tumor
                    and reasons == ["host_memory"]
                    and time.monotonic() >= next_cache_release
                ):
                    from .first_post_segmentation_parallel import (
                        release_completed_input_cache,
                    )

                    data = read_status(root / "data/manifest.json")
                    event = release_completed_input_cache(
                        Path(config["data"]["tumor_crop"]["segmentation_root"]),
                        Path(data["source_root"]),
                        [
                            {
                                **r,
                                "case_id": f"{r['patient_id']}_{r['visit']}_aqc1",
                                "source_image": str(
                                    Path(data["source_root"]) / r["relative_path"]
                                ),
                            }
                            for r in data["records"]
                        ],
                        target_headroom_bytes=(
                            config["queue"]["admission_memory_gib"] + 4
                        )
                        * GIB,
                    )
                    write_json(root / "cache_release_status.json", event)
                    from .first_post_tumor_crops import release_verified_crop_cache

                    crop_event = release_verified_crop_cache(config)
                    write_json(root / "crop_cache_release_status.json", crop_event)
                    next_cache_release = time.monotonic() + 300
                    snapshot = resources(root)
                    reasons = resource_reasons(snapshot, config, runtime=False)
                if not reasons:
                    for gpu in config["queue"]["gpus"]:
                        if gpu_reserved(gpu, config):
                            continue
                        with (lock_root / f"gpu{gpu}.lock").open("a") as gpu_lock:
                            try:
                                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                continue
                            if not gpu_idle(gpu):
                                continue
                            time.sleep(3)
                            if (
                                not gpu_idle(gpu)
                                or gpu_reserved(gpu, config)
                                or resource_reasons(
                                    resources(root), config, runtime=False
                                )
                            ):
                                continue
                            if tumor:
                                run_child("profile", config_path, root, gpu=gpu)
                                if (
                                    read_status(
                                        root / "cpu_smoke/training_result.json"
                                    ).get("status")
                                    != "passed"
                                ):
                                    run_child("cpu-smoke", config_path, root)
                            if (
                                read_status(
                                    root / "gpu_smoke/training_result.json"
                                ).get("status")
                                != "passed"
                            ):
                                run_child("smoke", config_path, root, gpu=gpu)
                            run_child("train", config_path, root, gpu=gpu)
                            result = read_status(root / "training_result.json")
                            if result.get("status") == "completed":
                                write_json(
                                    root / "queue_status.json",
                                    {
                                        "status": "completed",
                                        "pid": os.getpid(),
                                        "gpu": gpu,
                                        "updated_at_utc": timestamp(),
                                    },
                                )
                                return
                            if result.get("status") != "paused":
                                raise RuntimeError(
                                    "Training exited without a completion or pause record"
                                )
                write_json(
                    root / "queue_status.json",
                    {
                        "status": "waiting_for_resources",
                        "pid": os.getpid(),
                        "updated_at_utc": timestamp(),
                        "reasons": reasons or ["gpu_busy_or_reserved"],
                        "resources": snapshot,
                        "next_stage": "smoke_then_train",
                    },
                )
                time.sleep(config["queue"]["poll_seconds"])
        except BaseException as error:
            write_json(
                root / "queue_status.json",
                {
                    "status": "interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "failed",
                    "pid": os.getpid(),
                    "updated_at_utc": timestamp(),
                    "error": str(error),
                },
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "cohort",
            "prepare",
            "cpu-smoke",
            "smoke",
            "train",
            "queue",
            "profile",
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.stage == "cohort":
        print(json.dumps(prepare_cohort(config)), flush=True)
    elif args.stage == "prepare":
        print(json.dumps(prepare(config)), flush=True)
    elif args.stage == "queue":
        run_queue(args.config.resolve())
    elif args.stage == "profile":
        from .first_post_vqgan_performance import benchmark

        benchmark(args.config.resolve())
    else:
        print(
            json.dumps(
                train(
                    config, smoke=args.stage != "train", cpu=args.stage == "cpu-smoke"
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
