from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

BRIDGE_SCHEMA = "biflow_source_bridge_v1"
_FROZEN_TEXT_PREFIX = "model.conditioner.text_tower.model."


def public_metadata(value: Any) -> Any:
    """Keep structural provenance without copying legacy checksum metadata."""
    if isinstance(value, Mapping):
        return {
            str(key): public_metadata(item)
            for key, item in value.items()
            if not any(
                word in str(key).lower()
                for word in ("sha256", "checksum", "fingerprint", "hash", "revision")
            )
        }
    if isinstance(value, (tuple, list)):
        return [public_metadata(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return re.sub(
            r"(?<![a-zA-Z0-9])[a-fA-F0-9]{40,64}(?![a-zA-Z0-9])",
            "<legacy-digest>",
            value,
        )
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                public_metadata(payload), indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def latent_channels(value: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim not in (5, 6)
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
        or any(size < 1 for size in value.shape)
    ):
        raise ValueError("Expected a finite batched 3D latent")
    if value.ndim == 6 and value.shape[1] != 1:
        raise ValueError("Six-dimensional DCE0 latent must have one modality")
    return value.reshape(value.shape[0], -1, *value.shape[-3:])


def initial_state(
    source: torch.Tensor,
    noise: torch.Tensor,
    *,
    distribution: str,
    channel_std: torch.Tensor,
    multiplier: float,
) -> torch.Tensor:
    channels = latent_channels(source).shape[1]
    latent_channels(noise)
    if (
        source.shape != noise.shape
        or source.dtype != noise.dtype
        or source.device != noise.device
    ):
        raise ValueError("Source and noise must share shape, dtype and device")
    if (
        type(multiplier) not in (int, float)
        or not math.isfinite(multiplier)
        or multiplier <= 0
    ):
        raise ValueError("Noise multiplier must be positive and finite")
    if distribution == "standard_normal":
        if multiplier != 1.0:
            raise ValueError("The matched standard-normal baseline uses unit noise")
        return noise.clone()
    if distribution != "source_gaussian":
        raise ValueError("Unsupported base distribution")
    if (
        channel_std.shape != (channels,)
        or not torch.isfinite(channel_std).all()
        or not (channel_std > 0).all()
    ):
        raise ValueError("Training channel standard deviations are invalid")
    shape = (1,) * (source.ndim - 4) + (channels, 1, 1, 1)
    scale = channel_std.to(source).reshape(shape) * multiplier
    return source + scale * noise


def flow_path(
    source: torch.Tensor,
    target: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
    **settings: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_channels(target)
    if (
        source.shape != target.shape
        or source.dtype != target.dtype
        or source.device != target.device
    ):
        raise ValueError("Source and target must use the same latent coordinates")
    if (
        time.shape != (source.shape[0],)
        or time.device != source.device
        or not time.is_floating_point()
        or not torch.isfinite(time).all()
        or not ((time >= 0) & (time <= 1)).all()
    ):
        raise ValueError("Flow time must be finite [B] in [0,1]")
    initial = initial_state(source, noise, **settings)
    weight = time.to(source).reshape((-1,) + (1,) * (source.ndim - 1))
    return (1 - weight) * initial + weight * target, target - initial


@torch.no_grad()
def integrate(
    field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    source: torch.Tensor,
    noise: torch.Tensor,
    *,
    steps: int,
    **settings: Any,
) -> torch.Tensor:
    if type(steps) is not int or steps < 1:
        raise ValueError("Euler steps must be a positive integer")
    state = initial_state(source, noise, **settings)
    for index in range(steps):
        time = state.new_full((state.shape[0],), index / steps)
        velocity = field(state, time)
        if velocity.shape != state.shape or not torch.isfinite(velocity).all():
            raise FloatingPointError("Euler velocity is invalid")
        state = state + velocity / steps
    if not torch.isfinite(state).all():
        raise FloatingPointError("Euler endpoint is non-finite")
    return state


def channel_statistics(latents: Iterable[torch.Tensor]) -> dict[str, Any]:
    """Population moments over unique training source visits, using Chan updates."""
    count, visits = 0, 0
    mean = second = None
    for latent in latents:
        flat = latent.reshape(-1, math.prod(latent.shape[-3:])).double()
        if not torch.isfinite(flat).all():
            raise ValueError("Calibration latent is non-finite")
        variance, batch_mean = torch.var_mean(flat, dim=1, correction=0)
        batch_count = flat.shape[1]
        if mean is None:
            mean, second = batch_mean, variance * batch_count
        else:
            if batch_mean.shape != mean.shape:
                raise ValueError("Calibration channel count changed")
            delta = batch_mean - mean
            second = (
                second
                + variance * batch_count
                + delta.square() * (count * batch_count / (count + batch_count))
            )
            mean = mean + delta * (batch_count / (count + batch_count))
        count += batch_count
        visits += 1
    if visits == 0:
        raise ValueError("Calibration requires training source visits")
    std = (second / count).sqrt()
    if not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Calibration contains degenerate latent channels")
    return {
        "visits": visits,
        "voxel_count_per_channel": count,
        "channel_mean": mean.tolist(),
        "channel_std": std.tolist(),
    }


class EpochSampler(Sampler[int]):
    def __init__(self, length: int, seed: int) -> None:
        self.length, self.seed, self.epoch = length, seed, 0

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.length, generator=generator).tolist())


class BridgeSystem(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        family: str,
        experiment_config: Any,
        contract: dict[str, Any],
        optimizer_factory: Callable[[Any], Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.family = family
        self.experiment_config = experiment_config
        self.contract = contract
        self.optimizer_factory = optimizer_factory
        self.strict_loading = False
        self.register_buffer(
            "channel_std",
            torch.tensor(contract["statistics"]["channel_std"], dtype=torch.float32),
        )
        self.pair_indices = {
            key: index for index, key in enumerate(contract["pair_order"])
        }
        self.validation_rows: list[dict[str, Any]] = []
        self.latest_validation: dict[str, float] = {}

    @property
    def settings(self) -> dict[str, Any]:
        return {
            "distribution": self.contract["experiment"]["distribution"],
            "multiplier": self.contract["experiment"]["noise_multiplier"],
            "channel_std": self.channel_std,
        }

    def context(self, batch: Mapping[str, Any]) -> Any:
        kwargs = {
            key: batch[key] for key in ("clinical_text", "treatment_text", "delta_days")
        }
        if self.family == "ispy2":
            kwargs["target_stage"] = batch["target_stage"]
        else:
            kwargs["source_mask_onehot"] = batch["source_mask_onehot"]
        return self.model.prepare_context(batch["source_mri"], **kwargs)

    def velocity(self, state: torch.Tensor, time: torch.Tensor, prepared: Any):
        return self.model.velocity_from_context(
            flow_state=state, flow_time=time, prepared=prepared
        ).velocity

    def noise_and_time(
        self,
        source: torch.Tensor,
        metadata: list[dict[str, Any]],
        *,
        training: bool,
        sample: int = 0,
    ):
        noises, times = [], []
        seed = self.contract["experiment"]["seed"]
        for index, item in enumerate(metadata):
            pair_index = self.pair_indices[item["pair_id"]]
            offset = (
                self.current_epoch * len(self.pair_indices) + pair_index
                if training
                else 10_000_000 + pair_index * 1024 + sample
            )
            generator = torch.Generator(device=source.device).manual_seed(seed + offset)
            noises.append(
                torch.randn(
                    source[index].shape,
                    device=source.device,
                    dtype=source.dtype,
                    generator=generator,
                )
            )
            times.append(
                torch.rand(
                    (), device=source.device, dtype=source.dtype, generator=generator
                )
            )
        return torch.stack(noises), torch.stack(times)

    def training_step(self, batch: dict[str, Any], batch_idx: int):
        source, target = batch["source_latent"], batch["target_latent"]
        noise, time = self.noise_and_time(source, batch["metadata"], training=True)
        state, velocity = flow_path(source, target, noise, time, **self.settings)
        predicted = self.velocity(state, time, self.context(batch))
        loss = F.l1_loss(predicted, velocity)
        if not torch.isfinite(loss):
            raise FloatingPointError("Bridge training loss is non-finite")
        self.log(
            "train/velocity_mae",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=source.shape[0],
        )
        return loss

    @torch.no_grad()
    def predict_batch(self, batch: Mapping[str, Any], *, steps: int, samples: int):
        source = batch["source_latent"]
        prepared = self.context(batch)
        predictions = []
        for sample in range(samples):
            noise, _ = self.noise_and_time(
                source, batch["metadata"], training=False, sample=sample
            )
            predictions.append(
                integrate(
                    lambda state, time: self.velocity(state, time, prepared),
                    source,
                    noise,
                    steps=steps,
                    **self.settings,
                )
            )
        return torch.stack(predictions)

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.sampler
        if not isinstance(sampler, EpochSampler):
            raise TypeError("Bridge training requires the epoch-addressed sampler")
        sampler.epoch = self.current_epoch

    def on_train_batch_end(self, outputs, batch, batch_idx):
        step = self.global_step
        if (
            step
            and (
                step % 25 == 0
                or step - getattr(self, "_starting_step", 0) in (1, 2)
            )
            and getattr(self, "_last_progress_step", None) != step
        ):
            self._last_progress_step = step
            write_json(
                Path(self.contract["experiment"]["output_root"]) / "progress.json",
                {
                    "status": "training",
                    "epoch": self.current_epoch,
                    "optimizer_step": step,
                    "last_validation": self.latest_validation,
                    "gradient_norms": getattr(self, "latest_gradient_norms", {}),
                },
            )

    def on_validation_epoch_start(self):
        self.validation_rows = []

    def validation_step(self, batch: dict[str, Any], batch_idx: int):
        settings = self.contract["experiment"]
        predictions = self.predict_batch(
            batch,
            steps=settings["validation_steps"],
            samples=settings["validation_samples"],
        )
        mean = predictions.mean(0)
        target, source = batch["target_latent"], batch["source_latent"]
        error = (mean - target).abs().flatten(1).mean(1)
        copy_error = (source - target).abs().flatten(1).mean(1)
        spread = predictions.std(0, correction=0).flatten(1).mean(1)
        for index, metadata in enumerate(batch["metadata"]):
            self.validation_rows.append(
                {
                    "patient_id": metadata["patient_id"],
                    "endpoint_mae": error[index].item(),
                    "source_copy_mae": copy_error[index].item(),
                    "sample_std": spread[index].item(),
                }
            )

    def on_validation_epoch_end(self):
        if not self.validation_rows:
            raise RuntimeError("Bridge validation produced no pairs")
        by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.validation_rows:
            by_patient[row["patient_id"]].append(row)
        metrics = {}
        for key in ("endpoint_mae", "source_copy_mae", "sample_std"):
            metrics[key] = sum(
                sum(row[key] for row in rows) / len(rows)
                for rows in by_patient.values()
            ) / len(by_patient)
            self.log(f"val/{key}", metrics[key], prog_bar=key == "endpoint_mae")
        self.latest_validation = metrics
        if not self.trainer.sanity_checking:
            root = Path(self.contract["experiment"]["output_root"])
            write_json(
                root / "progress.json",
                {
                    "status": "training",
                    "epoch": self.current_epoch,
                    "optimizer_step": self.global_step,
                    "validation_patients": len(by_patient),
                    "validation_pairs": len(self.validation_rows),
                    **metrics,
                },
            )

    def configure_optimizers(self):
        return self.optimizer_factory(self)

    def on_before_optimizer_step(self, optimizer):
        if not hasattr(self, "_gradient_ownership"):
            self._gradient_ownership = {
                id(parameter): (
                    "backbone" if name.startswith("model.dynamics.backbone.")
                    else "controlnet" if name.startswith("model.dynamics.controlnet.")
                    else "text" if name.startswith("model.conditioner.text_tower.")
                    else "conditioner"
                )
                for name, parameter in self.named_parameters()
            }
        branches = defaultdict(list)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    branches[self._gradient_ownership[id(parameter)]].append(
                        torch.linalg.vector_norm(
                            parameter.grad.detach(), dtype=torch.float64
                        )
                    )
        norms = {
            name: torch.linalg.vector_norm(torch.stack(values))
            for name, values in branches.items()
        }
        if not norms:
            raise RuntimeError("Bridge model has no gradients")
        norm = torch.linalg.vector_norm(torch.stack(list(norms.values())))
        if not torch.isfinite(norm):
            raise FloatingPointError("Bridge gradients are non-finite")
        self._gradient_norm_for_clipping = norm
        self.latest_gradient_norms = {
            name: float(value) for name, value in {"total": norm, **norms}.items()
        }
        self.log("train/gradient_norm", norm, on_step=True, on_epoch=False)
        for name, value in norms.items():
            self.log(f"train/gradient_norm_{name}", value, on_step=True, on_epoch=False)

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        if not gradient_clip_val:
            return
        if gradient_clip_algorithm not in (None, "norm"):
            raise ValueError("Bridge training requires norm gradient clipping")
        # Reuse the FP64 reduction; FP32 sum-of-squares can overflow on finite gradients.
        coefficient = (
            gradient_clip_val / (self._gradient_norm_for_clipping + 1e-6)
        ).clamp(max=1.0)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.mul_(coefficient.to(parameter.grad))

    def on_train_start(self):
        self._starting_step = self.global_step
        self.restored_training_state = {
            "optimizer_step": self.global_step,
            "optimizer_state_entries": [len(item.state) for item in self.trainer.optimizers],
            "optimizer_group_learning_rates": [
                [group["lr"] for group in item.param_groups]
                for item in self.trainer.optimizers
            ],
            "scheduler_steps": [
                item.scheduler.last_epoch for item in self.trainer.lr_scheduler_configs
            ],
        }
        state = getattr(self, "_pending_rng_state", None)
        if state is not None:
            torch.set_rng_state(state["cpu"])
            if self.device.type == "cuda" and "cuda" in state:
                torch.cuda.set_rng_state(state["cuda"], self.device)
            del self._pending_rng_state

    def omitted_keys(self) -> tuple[str, ...]:
        trainable = {
            name for name, value in self.named_parameters() if value.requires_grad
        }
        return tuple(
            sorted(
                key
                for key in self.state_dict()
                if key.startswith(_FROZEN_TEXT_PREFIX) and key not in trainable
            )
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        omitted = self.omitted_keys()
        checkpoint["state_dict"] = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if key not in omitted
        }
        checkpoint["source_bridge_schema"] = BRIDGE_SCHEMA
        checkpoint["source_bridge_contract"] = self.contract
        checkpoint["source_bridge_omitted_keys"] = list(omitted)
        checkpoint["source_bridge_rng"] = {"cpu": torch.get_rng_state()}
        if self.device.type == "cuda":
            checkpoint["source_bridge_rng"]["cuda"] = torch.cuda.get_rng_state(
                self.device
            )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if (
            checkpoint.get("source_bridge_schema") != BRIDGE_SCHEMA
            or checkpoint.get("source_bridge_contract") != self.contract
        ):
            raise ValueError("Source-bridge checkpoint contract mismatch")
        state = checkpoint.get("state_dict", {})
        omitted = self.omitted_keys()
        if checkpoint.get("source_bridge_omitted_keys") != list(omitted) or set(
            state
        ) != set(self.state_dict()) - set(omitted):
            raise ValueError("Source-bridge checkpoint state is incomplete")
        if not torch.equal(state["channel_std"].cpu(), self.channel_std.cpu()):
            raise ValueError("Source-bridge checkpoint noise scale changed")
        self._pending_rng_state = checkpoint["source_bridge_rng"]
