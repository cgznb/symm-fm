from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from .first_post_data import FirstPostDataset, read_config, timestamp, write_json


def selected_config(config, selection):
    if selection["source_configuration"] != config or selection["status"] != "passed":
        raise ValueError(
            "VQ-GAN performance selection belongs to another configuration"
        )
    result = copy.deepcopy(config)
    batch = selection["batch_size"]
    scale = batch / config["training"]["batch_size"]
    result["training"].update(
        batch_size=batch,
        precision=selection["precision"],
        generator_learning_rate=config["training"]["generator_learning_rate"] * scale,
        discriminator_learning_rate=config["training"]["discriminator_learning_rate"]
        * scale,
        scheduler_milestones=[max(1, round(n / scale)) for n in (60000, 120000)],
        loader_workers=selection["loader_workers"],
        prefetch_factor=selection["prefetch_factor"],
    )
    return result


class PerformanceMonitor(pl.Callback):
    def __init__(self):
        self.steps = []
        self.validation_seconds = []
        self.utilization = []
        self.gradient_checks = 0
        self.stop = threading.Event()
        self.sampler = None

    def on_fit_start(self, trainer, pl_module):
        del trainer, pl_module
        torch.cuda.reset_peak_memory_stats()
        gpu = os.environ["CUDA_VISIBLE_DEVICES"]

        def sample():
            while not self.stop.is_set():
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        gpu,
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=10,
                )
                self.utilization.append([int(v) for v in output.strip().split(",")])
                self.stop.wait(1.0)

        self.sampler = threading.Thread(target=sample, daemon=True)
        self.sampler.start()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del pl_module, batch, batch_idx
        torch.cuda.synchronize()
        self.started = time.monotonic()
        self.data_gap = (
            self.started - self.previous_end
            if self.steps and self.steps[-1]["epoch"] == trainer.current_epoch
            else 0.0
        )

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        del trainer, pl_module
        checks = [
            torch.isfinite(p.grad).all()
            for group in optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not checks or not torch.stack(checks).all():
            raise FloatingPointError("Non-finite or absent unscaled training gradients")
        self.gradient_checks += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del pl_module, outputs
        torch.cuda.synchronize()
        self.steps.append(
            {
                "epoch": trainer.current_epoch,
                "batch_index": batch_idx,
                "images": len(batch["image"]),
                "seconds": time.monotonic() - self.started,
                "data_and_transfer_gap_seconds": self.data_gap,
            }
        )
        self.previous_end = time.monotonic()

    def on_validation_start(self, trainer, pl_module):
        del trainer, pl_module
        torch.cuda.synchronize()
        self.validation_started = time.monotonic()

    def on_validation_end(self, trainer, pl_module):
        del trainer, pl_module
        torch.cuda.synchronize()
        self.validation_seconds.append(time.monotonic() - self.validation_started)

    def close(self):
        self.stop.set()
        if self.sampler is not None:
            self.sampler.join(timeout=15)


def trial(config, batch, precision, destination):
    from .first_post_vqgan import build_system

    torch.set_num_threads(4)
    pl.seed_everything(config["seed"], workers=True)
    config = selected_config(
        config,
        {
            "status": "passed",
            "source_configuration": config,
            "batch_size": batch,
            "precision": precision,
            "loader_workers": 2,
            "prefetch_factor": 2,
        },
    )
    loaders = []
    for fold in ("train", "val"):
        dataset = FirstPostDataset(config, fold)
        count = batch * (4 if fold == "train" else 2)
        indices = np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()
        loaders.append(
            DataLoader(
                Subset(dataset, indices),
                batch_size=batch,
                num_workers=2,
                prefetch_factor=2,
                pin_memory=True,
                persistent_workers=True,
            )
        )
    system = build_system(config)
    tracked = (
        "autoencoder.encoder.input.weight",
        "image_discriminator.blocks.0.0.weight",
        "volume_discriminator.blocks.0.0.weight",
        "autoencoder.quantizer.embeddings",
    )
    before = {k: system.state_dict()[k].clone() for k in tracked}
    monitor = PerformanceMonitor()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=precision,
        max_epochs=2,
        limit_train_batches=4,
        limit_val_batches=2,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=[monitor],
        enable_model_summary=False,
    )
    started = time.monotonic()
    try:
        trainer.fit(system, *loaders)
        finite_state = all(
            not v.is_floating_point() or bool(torch.isfinite(v).all())
            for v in system.state_dict().values()
        )
        optimizer_finite = all(
            not torch.is_tensor(v) or bool(torch.isfinite(v).all())
            for opt in trainer.optimizers
            for state in opt.state.values()
            for v in state.values()
        )
        updates = {
            k: not torch.equal(v, system.state_dict()[k].cpu())
            for k, v in before.items()
        }
        if (
            not finite_state
            or not optimizer_finite
            or not all(updates.values())
            or monitor.gradient_checks != 16
            or any(not opt.state for opt in trainer.optimizers)
        ):
            raise RuntimeError(
                "GAN profile lacks finite initialized optimizer states or required updates"
            )
        measured = monitor.steps[2:]
        seconds = sum(
            r["seconds"] + r["data_and_transfer_gap_seconds"] for r in measured
        )
        total_memory = torch.cuda.get_device_properties(0).total_memory
        peak = torch.cuda.max_memory_reserved()
        result = {
            "status": "passed",
            "batch_size": batch,
            "precision": precision,
            "loader_workers": 2,
            "prefetch_factor": 2,
            "generator_updates": trainer.global_step // 2,
            "finite_unscaled_gradient_checks": monitor.gradient_checks,
            "finite_model_and_optimizer_states": True,
            "updated_parameter_groups": updates,
            "training_steps": monitor.steps,
            "training_images_per_second": sum(r["images"] for r in measured) / seconds,
            "measured_data_and_transfer_gap_seconds": sum(
                r["data_and_transfer_gap_seconds"] for r in measured
            ),
            "validation_seconds": monitor.validation_seconds,
            "train_validation_train_transition_checked": len(monitor.validation_seconds)
            == 2,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": peak,
            "total_device_bytes": total_memory,
            "reserved_headroom_bytes": total_memory - peak,
            "stable_memory_margin": total_memory - peak >= 1.5 * 1024**3,
            "sampled_gpu_utilization_mean": float(
                np.mean([v[0] for v in monitor.utilization])
            )
            if monitor.utilization
            else None,
            "sampled_gpu_memory_mib_max": max(
                (v[1] for v in monitor.utilization), default=0
            ),
            "elapsed_seconds": time.monotonic() - started,
            "updated_at_utc": timestamp(),
        }
        write_json(destination, result)
    finally:
        monitor.close()


def benchmark(config_path):
    config = read_config(config_path)
    output = Path(config["output_dir"]) / "performance"
    output.mkdir(parents=True, exist_ok=True)
    selection_path = output / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        selected_config(config, selection)
        return selection
    signature_path = output / "source_configuration.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != config:
        raise ValueError("Performance trial configuration changed")
    write_json(signature_path, config)
    results = []
    for precision in ("bf16-mixed", "16-mixed"):
        for batch in (1, 2, 3, 4, 6, 8, 12, 16):
            path = output / f"{precision}_batch{batch}.json"
            if not path.exists():
                with path.with_suffix(".log").open("a") as log:
                    process = subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "mewm_ispy2.first_post_vqgan_performance",
                            "trial",
                            "--config",
                            str(config_path),
                            "--batch",
                            str(batch),
                            "--precision",
                            precision,
                            "--output",
                            str(path),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if process.returncode and not path.exists():
                    raise RuntimeError(
                        f"Performance trial failed without a result: {path.with_suffix('.log')}"
                    )
            result = json.loads(path.read_text())
            results.append(result)
            print(
                json.dumps(
                    {k: result[k] for k in ("status", "batch_size", "precision")}
                ),
                flush=True,
            )
            if result["status"] != "passed" or not result["stable_memory_margin"]:
                break
    passed = [
        r for r in results if r["status"] == "passed" and r["stable_memory_margin"]
    ]
    if not passed:
        raise RuntimeError("No numerically stable full GAN training batch fits the GPU")
    best = max(passed, key=lambda r: r["training_images_per_second"])
    selection = {
        **best,
        "source_configuration": config,
        "selection_rule": "highest_measured_full_GAN_images_per_second_with_1.5_GiB_reserved_margin",
        "sample_budget": "fixed_100_epochs_learning_rates_scaled_linearly_and_scheduler_milestones_inversely_with_batch",
        "trial_weights_and_rng": "isolated_subprocesses_formal_run_reloads_initializer",
    }
    write_json(selection_path, selection)
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("trial", "benchmark"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--precision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "benchmark":
        benchmark(args.config.resolve())
        return
    try:
        trial(read_config(args.config), args.batch, args.precision, args.output)
    except (torch.OutOfMemoryError, FloatingPointError) as error:
        write_json(
            args.output,
            {
                "status": "out_of_memory"
                if isinstance(error, torch.OutOfMemoryError)
                else "nonfinite",
                "batch_size": args.batch,
                "precision": args.precision,
                "error": str(error),
                "updated_at_utc": timestamp(),
            },
        )
        raise


if __name__ == "__main__":
    main()
