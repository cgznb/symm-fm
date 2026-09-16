"""Single-GPU/DDP-compatible optimization steps for the two training stages."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ispy2_symmflow.flow.path import FlowLoss, SymmetricFlowObjective
from ispy2_symmflow.training.ema import ExponentialMovingAverage


@dataclass(frozen=True)
class AutoencoderLoss:
    total: Tensor
    reconstruction: Tensor
    kl: Tensor
    gradient: Tensor


def image_gradient_l1(prediction: Tensor, target: Tensor) -> Tensor:
    losses = []
    for dimension in (-3, -2, -1):
        pred_diff = torch.diff(prediction, dim=dimension)
        target_diff = torch.diff(target, dim=dimension)
        losses.append(F.l1_loss(pred_diff, target_diff))
    return torch.stack(losses).mean()


def autoencoder_objective(
    autoencoder: nn.Module,
    image: Tensor,
    *,
    kl_weight: float,
    gradient_weight: float = 0.0,
    sample_posterior: bool = True,
) -> AutoencoderLoss:
    """L1 reconstruction plus a weak Gaussian KL and optional 3D gradients."""

    reconstruction, mean, scale = autoencoder(image, sample_posterior=sample_posterior)
    if reconstruction.shape != image.shape:
        raise ValueError("autoencoder reconstruction shape differs from its input")
    if scale.shape != mean.shape or torch.any(scale < 0):
        raise ValueError("autoencoder posterior scale is invalid")
    reconstruction_loss = F.l1_loss(reconstruction, image)
    variance = scale.square().clamp_min(torch.finfo(scale.dtype).eps)
    kl = 0.5 * (mean.square() + variance - variance.log() - 1.0).flatten(1).mean(1).mean()
    gradient = (
        image_gradient_l1(reconstruction, image)
        if gradient_weight
        else reconstruction_loss.new_zeros(())
    )
    total = reconstruction_loss + float(kl_weight) * kl + float(gradient_weight) * gradient
    return AutoencoderLoss(total, reconstruction_loss, kl, gradient)


def _autocast(device: torch.device, precision: str):
    normalized = precision.lower()
    if normalized == "fp32":
        return nullcontext()
    if normalized not in {"bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16")
    if device.type not in {"cuda", "cpu"}:
        return nullcontext()
    dtype = torch.bfloat16 if normalized == "bf16" else torch.float16
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("fp16 autocast is unsupported on CPU")
    return torch.autocast(device_type=device.type, dtype=dtype)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("scheduler requires 0 <= warmup_steps < total_steps")

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class AutoencoderTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device,
        kl_weight: float,
        gradient_weight: float = 0.0,
        precision: str = "fp32",
        gradient_clip_norm: float = 1.0,
        scheduler: Any | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.kl_weight = float(kl_weight)
        self.gradient_weight = float(gradient_weight)
        self.precision = precision
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and precision.lower() == "fp16"
        )

    def train_batch(self, image: Tensor) -> dict[str, float]:
        self.model.train()
        image = image.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast(self.device, self.precision):
            losses = autoencoder_objective(
                self.model,
                image,
                kl_weight=self.kl_weight,
                gradient_weight=self.gradient_weight,
            )
        self.scaler.scale(losses.total).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite autoencoder gradient norm")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        return {
            "loss": float(losses.total.detach()),
            "reconstruction": float(losses.reconstruction.detach()),
            "kl": float(losses.kl.detach()),
            "gradient": float(losses.gradient.detach()),
            "gradient_norm": float(gradient_norm.detach()),
        }


class SymmFlowTrainer:
    """Train both velocity branches in a single network forward."""

    def __init__(
        self,
        velocity_model: nn.Module,
        condition_encoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        objective: SymmetricFlowObjective,
        *,
        device: torch.device,
        precision: str = "fp32",
        gradient_clip_norm: float = 1.0,
        gradient_accumulation: int = 1,
        scheduler: Any | None = None,
        ema: ExponentialMovingAverage | None = None,
    ) -> None:
        if gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be at least 1")
        self.velocity_model = velocity_model
        self.condition_encoder = condition_encoder
        self.optimizer = optimizer
        self.objective = objective
        self.device = device
        self.precision = precision
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.gradient_accumulation = int(gradient_accumulation)
        self.scheduler = scheduler
        self.ema = ema
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda" and precision.lower() == "fp16"
        )
        self.micro_step = 0
        self.optimizer.zero_grad(set_to_none=True)

    def train_batch(
        self,
        later_latent: Tensor,
        earlier_latent: Tensor,
        conditions: Mapping[str, Any],
    ) -> dict[str, float | bool]:
        self.velocity_model.train()
        self.condition_encoder.train()
        later = later_latent.to(self.device, non_blocking=True)
        earlier = earlier_latent.to(self.device, non_blocking=True)
        with _autocast(self.device, self.precision):
            tokens = self.condition_encoder(conditions, batch_size=later.shape[0])
            losses: FlowLoss = self.objective(
                self.velocity_model, later, earlier, tokens
            )
            scaled_loss = losses.total / self.gradient_accumulation
        self.scaler.scale(scaled_loss).backward()
        self.micro_step += 1
        updated = self.micro_step % self.gradient_accumulation == 0
        gradient_norm_value = float("nan")
        if updated:
            self.scaler.unscale_(self.optimizer)
            parameters = list(self.velocity_model.parameters()) + list(
                self.condition_encoder.parameters()
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, self.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("non-finite SymmFlow gradient norm")
            gradient_norm_value = float(gradient_norm.detach())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.ema is not None:
                self.ema.update(getattr(self.velocity_model, "module", self.velocity_model))
        return {
            "loss": float(losses.total.detach()),
            "loss_x": float(losses.x.detach()),
            "loss_y": float(losses.y.detach()),
            "gradient_norm": gradient_norm_value,
            "optimizer_updated": updated,
        }


@torch.no_grad()
def fit_latent_statistics(
    autoencoder: nn.Module,
    images: Iterable[Tensor],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, int]:
    """Fit shared per-channel latent moments from training-patient visits only."""

    autoencoder.eval()
    channel_sum: Tensor | None = None
    channel_square_sum: Tensor | None = None
    count = 0
    for image in images:
        latent = autoencoder.encode(image.to(device, non_blocking=True))
        reduce_dims = (0, 2, 3, 4)
        current_sum = latent.double().sum(dim=reduce_dims)
        current_square = latent.double().square().sum(dim=reduce_dims)
        current_count = latent.shape[0] * latent.shape[2] * latent.shape[3] * latent.shape[4]
        channel_sum = current_sum if channel_sum is None else channel_sum + current_sum
        channel_square_sum = (
            current_square if channel_square_sum is None else channel_square_sum + current_square
        )
        count += current_count
    if count < 2 or channel_sum is None or channel_square_sum is None:
        raise ValueError("at least two latent elements are required to fit statistics")
    mean = channel_sum / count
    variance = (channel_square_sum / count - mean.square()).clamp_min(1e-12)
    return mean.float(), variance.sqrt().float(), count
