"""Evaluation that keeps background, tumor masks, and stochastic summaries explicit."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


def _check_images(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if prediction.ndim != 5:
        raise ValueError("MRI tensors must use [B,C,D,H,W]")
    if any(size < 1 for size in prediction.shape):
        raise ValueError("MRI tensors cannot have empty dimensions")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("MRI tensors must use floating-point dtypes")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("metrics do not accept NaN or infinite image values")


def _check_data_range(data_range: float) -> float:
    value = float(data_range)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("data_range must be finite and positive")
    return value


def _prepare_mask(mask: Tensor, target: Tensor, *, name: str) -> Tensor:
    if mask.ndim == 4:
        mask = mask[:, None]
    if mask.ndim != 5:
        raise ValueError(f"{name} mask must use [B,D,H,W] or [B,C,D,H,W]")
    if mask.shape[0] != target.shape[0] or mask.shape[-3:] != target.shape[-3:]:
        raise ValueError(f"{name} mask geometry does not match target")
    if mask.shape[1] not in {1, target.shape[1]}:
        raise ValueError(f"{name} mask must have one channel or match target channels")
    if not torch.isfinite(mask).all():
        raise ValueError(f"{name} mask must contain only finite values")
    return mask


def _masked_mean(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    expanded = mask.expand_as(values).to(dtype=values.dtype)
    counts = expanded.flatten(1).sum(dim=1)
    totals = (values * expanded).flatten(1).sum(dim=1)
    valid = counts > 0
    result = torch.full_like(totals, float("nan"))
    result[valid] = totals[valid] / counts[valid]
    return result, valid


def ssim_3d(prediction: Tensor, target: Tensor, *, data_range: float) -> Tensor:
    """Channel-averaged local 3D SSIM using a uniform sliding window."""

    _check_images(prediction, target)
    data_range = _check_data_range(data_range)
    minimum_side = min(prediction.shape[-3:])
    window = min(7, minimum_side if minimum_side % 2 else minimum_side - 1)
    if window < 3:
        raise ValueError("SSIM requires each spatial dimension to be at least 3")
    padding = window // 2
    mu_x = F.avg_pool3d(prediction, window, stride=1, padding=padding)
    mu_y = F.avg_pool3d(target, window, stride=1, padding=padding)
    sigma_x = F.avg_pool3d(prediction.square(), window, stride=1, padding=padding) - mu_x.square()
    sigma_y = F.avg_pool3d(target.square(), window, stride=1, padding=padding) - mu_y.square()
    sigma_xy = F.avg_pool3d(prediction * target, window, stride=1, padding=padding) - mu_x * mu_y
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(torch.finfo(prediction.dtype).eps)).flatten(1).mean(1)


def mask_volume_ml(mask: Tensor, spacing_mm: tuple[float, float, float]) -> Tensor:
    """Return binary-mask volume in millilitres for each batch item."""

    if mask.ndim not in {4, 5}:
        raise ValueError("mask must use [B,D,H,W] or [B,1,D,H,W]")
    if any(size < 1 for size in mask.shape):
        raise ValueError("mask cannot have empty dimensions")
    if not torch.isfinite(mask).all():
        raise ValueError("mask must contain only finite values")
    if len(spacing_mm) != 3 or any((not math.isfinite(v) or v <= 0) for v in spacing_mm):
        raise ValueError("spacing_mm must contain three finite positive values")
    if mask.ndim == 5:
        if mask.shape[1] != 1:
            raise ValueError("volume masks may only have one channel")
        mask = mask[:, 0]
    voxel_ml = math.prod(spacing_mm) / 1000.0
    return (mask > 0).flatten(1).sum(1).to(torch.float64) * voxel_ml


def evaluate_prediction(
    prediction: Tensor,
    target: Tensor,
    *,
    data_range: float,
    foreground_mask: Tensor | None = None,
    tumor_mask: Tensor | None = None,
    spacing_mm: tuple[float, float, float] | None = None,
    mask_semantics_verified: bool = False,
) -> dict[str, Any]:
    """Compute per-case metrics without silently treating empty masks as zero error."""

    _check_images(prediction, target)
    data_range = _check_data_range(data_range)
    error = prediction - target
    mse = error.square().flatten(1).mean(1)
    mae = error.abs().flatten(1).mean(1)
    peak = torch.as_tensor(float(data_range), dtype=mse.dtype, device=mse.device)
    psnr = torch.where(mse == 0, torch.inf, 20 * torch.log10(peak) - 10 * torch.log10(mse))
    result: dict[str, Any] = {
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim_3d(prediction, target, data_range=data_range),
        "data_range": float(data_range),
    }
    if foreground_mask is not None:
        foreground_mask = _prepare_mask(foreground_mask, target, name="foreground")
        fg_mae, valid = _masked_mean(error.abs(), foreground_mask > 0)
        fg_mse, _ = _masked_mean(error.square(), foreground_mask > 0)
        result.update(foreground_mae=fg_mae, foreground_mse=fg_mse, foreground_valid=valid)
    if tumor_mask is not None:
        if not mask_semantics_verified:
            raise ValueError("tumor metrics require mask_semantics_verified=True")
        tumor_mask = _prepare_mask(tumor_mask, target, name="tumor")
        tumor_mae, valid = _masked_mean(error.abs(), tumor_mask > 0)
        tumor_mse, _ = _masked_mean(error.square(), tumor_mask > 0)
        result.update(tumor_mae=tumor_mae, tumor_mse=tumor_mse, tumor_valid=valid)
        if spacing_mm is not None:
            result["reference_mask_volume_ml"] = mask_volume_ml(tumor_mask, spacing_mm)
    return result


def evaluate_sample_set(
    samples: Tensor,
    target: Tensor,
    *,
    data_range: float,
    foreground_mask: Tensor | None = None,
) -> dict[str, Any]:
    """Separate candidate distribution, predictive-mean, and oracle summaries."""

    if samples.ndim != 6:
        raise ValueError("samples must use [K,B,C,D,H,W]")
    if samples.shape[0] < 1:
        raise ValueError("samples must contain at least one candidate")
    if samples.shape[1:] != target.shape:
        raise ValueError("sample and target geometry differs")
    per_sample = [
        evaluate_prediction(
            sample, target, data_range=data_range, foreground_mask=foreground_mask
        )
        for sample in samples
    ]
    candidate_mae = torch.stack([item["mae"] for item in per_sample], dim=0)
    candidate_mse = torch.stack([item["mse"] for item in per_sample], dim=0)
    candidate_metrics = {
        name: torch.stack([item[name] for item in per_sample], dim=0)
        for name in ("mae", "mse", "psnr", "ssim")
    }
    if foreground_mask is not None:
        candidate_metrics.update(
            {
                name: torch.stack([item[name] for item in per_sample], dim=0)
                for name in ("foreground_mae", "foreground_mse", "foreground_valid")
            }
        )
    mean_metrics = evaluate_prediction(
        samples.mean(dim=0), target, data_range=data_range, foreground_mask=foreground_mask
    )
    return {
        "candidate_mae": candidate_mae,
        "candidate_mse": candidate_mse,
        "candidate_metrics": candidate_metrics,
        "candidate_mae_mean": candidate_mae.mean(dim=0),
        "candidate_mae_std": candidate_mae.std(dim=0, unbiased=False),
        "predictive_mean": mean_metrics,
        "oracle_best_mae": candidate_mae.min(dim=0).values,
        "oracle_warning": "target-selected best-of-K; not a primary prospective metric",
    }
