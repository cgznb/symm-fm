from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity


@dataclass(frozen=True)
class EvaluationResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    denominator = int(prediction.sum() + target.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(prediction, target).sum() / denominator)


def _surface(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(bool)
    return np.logical_and(binary, np.logical_not(ndimage.binary_erosion(binary)))


def _surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_zyx: Sequence[float],
    tolerance_mm: float,
) -> tuple[float, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if not prediction.any() and not target.any():
        return 0.0, 1.0
    if not prediction.any() or not target.any():
        return math.inf, 0.0
    prediction_surface = _surface(prediction)
    target_surface = _surface(target)
    distance_to_target = ndimage.distance_transform_edt(
        ~target_surface, sampling=tuple(float(value) for value in spacing_zyx)
    )[prediction_surface]
    distance_to_prediction = ndimage.distance_transform_edt(
        ~prediction_surface, sampling=tuple(float(value) for value in spacing_zyx)
    )[target_surface]
    distances = np.concatenate((distance_to_target, distance_to_prediction))
    hd95 = float(np.percentile(distances, 95))
    overlap = int((distance_to_target <= tolerance_mm).sum()) + int(
        (distance_to_prediction <= tolerance_mm).sum()
    )
    surface_dice = float(overlap / max(distances.size, 1))
    return hd95, surface_dice


def _image_metrics(
    prediction: np.ndarray, target: np.ndarray, roi: np.ndarray
) -> dict[str, float]:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    values = difference[roi]
    mae = float(np.abs(values).mean())
    mse = float(np.square(values).mean())
    data_range = float(max(target.max() - target.min(), 1e-6))
    psnr = math.inf if mse == 0.0 else float(10.0 * math.log10(data_range**2 / mse))
    _, ssim_map = structural_similarity(
        target.astype(np.float32),
        prediction.astype(np.float32),
        data_range=data_range,
        full=True,
    )
    ssim = float(ssim_map[roi].mean())
    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim}


def _direction(new_volume: int, old_volume: int, tolerance_fraction: float = 0.01) -> int:
    tolerance = max(int(round(old_volume * tolerance_fraction)), 1)
    difference = new_volume - old_volume
    return 1 if difference > tolerance else -1 if difference < -tolerance else 0


def evaluate_samples(
    dce_samples: np.ndarray,
    mask_samples: np.ndarray,
    target_dce: np.ndarray,
    target_mask: np.ndarray,
    source_mask: np.ndarray,
    entropy: np.ndarray,
    *,
    spacing_zyx: Sequence[float],
    surface_tolerance_mm: float,
    neighborhood_iterations: int = 5,
) -> EvaluationResult:
    samples = np.asarray(dce_samples, dtype=np.float32)
    masks = np.asarray(mask_samples).astype(bool)
    target = np.asarray(target_dce, dtype=np.float32)
    target_binary = np.asarray(target_mask).astype(bool)
    source_binary = np.asarray(source_mask).astype(bool)
    uncertainty = np.asarray(entropy, dtype=np.float32)
    if samples.shape != (8, *target.shape) or masks.shape != (8, *target.shape):
        raise ValueError("evaluation expects eight DCE and Mask samples")
    if target.shape != target_binary.shape or source_binary.shape != target.shape:
        raise ValueError("evaluation image and Mask grids must match")
    full_roi = np.ones_like(target_binary, dtype=bool)
    neighborhood = ndimage.binary_dilation(
        np.logical_or(source_binary, target_binary), iterations=neighborhood_iterations
    )
    if not neighborhood.any():
        neighborhood = full_roi
    target_volume = int(target_binary.sum())
    source_volume = int(source_binary.sum())
    target_direction = _direction(target_volume, source_volume)
    rows: list[dict[str, Any]] = []
    for index, (sample, mask) in enumerate(zip(samples, masks, strict=True)):
        full = _image_metrics(sample, target, full_roi)
        local = _image_metrics(sample, target, neighborhood)
        hd95, surface_dice = _surface_metrics(
            mask, target_binary, spacing_zyx, surface_tolerance_mm
        )
        predicted_volume = int(mask.sum())
        predicted_direction = _direction(predicted_volume, source_volume)
        rows.append(
            {
                "sample_index": index,
                **full,
                "tumor_neighborhood_mae": local["mae"],
                "tumor_neighborhood_mse": local["mse"],
                "tumor_neighborhood_psnr": local["psnr"],
                "tumor_neighborhood_ssim": local["ssim"],
                "dice": _dice(mask, target_binary),
                "hd95_mm": hd95,
                "surface_dice": surface_dice,
                "predicted_volume_voxels": predicted_volume,
                "target_volume_voxels": target_volume,
                "absolute_volume_error_voxels": abs(predicted_volume - target_volume),
                "relative_volume_error": float(
                    abs(predicted_volume - target_volume) / max(target_volume, 1)
                ),
                "growth_direction": predicted_direction,
                "target_growth_direction": target_direction,
                "growth_direction_correct": predicted_direction == target_direction,
            }
        )
    mean_prediction = samples.mean(axis=0)
    absolute_error = np.abs(mean_prediction - target).reshape(-1)
    flattened_uncertainty = uncertainty.reshape(-1)
    if np.ptp(flattened_uncertainty) == 0 or np.ptp(absolute_error) == 0:
        correlation = math.nan
    else:
        correlation = spearmanr(flattened_uncertainty, absolute_error).statistic
    def mean_metric(key: str, *, finite_only: bool = False) -> float | None:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        if finite_only:
            values = values[np.isfinite(values)]
        return None if values.size == 0 else float(values.mean())

    hd95_values = np.asarray([row["hd95_mm"] for row in rows], dtype=np.float64)
    hd95_failure_count = int((~np.isfinite(hd95_values)).sum())
    best_mae = min(rows, key=lambda row: row["mae"])
    best_dice = max(rows, key=lambda row: row["dice"])
    summary = {
        "sample_count": 8,
        "best_mae_sample": int(best_mae["sample_index"]),
        "best_mae": float(best_mae["mae"]),
        "best_dice_sample": int(best_dice["sample_index"]),
        "best_dice": float(best_dice["dice"]),
        "mean_mae": mean_metric("mae"),
        "mean_mse": mean_metric("mse"),
        "mean_psnr": mean_metric("psnr", finite_only=True),
        "mean_ssim": mean_metric("ssim"),
        "mean_tumor_neighborhood_mae": mean_metric("tumor_neighborhood_mae"),
        "mean_tumor_neighborhood_mse": mean_metric("tumor_neighborhood_mse"),
        "mean_tumor_neighborhood_psnr": mean_metric(
            "tumor_neighborhood_psnr", finite_only=True
        ),
        "mean_tumor_neighborhood_ssim": mean_metric("tumor_neighborhood_ssim"),
        "mean_dice": mean_metric("dice"),
        "mean_hd95_mm": (
            None if hd95_failure_count else float(hd95_values.mean())
        ),
        "hd95_failure_count": hd95_failure_count,
        "hd95_failure_rate": float(hd95_failure_count / len(rows)),
        "mean_surface_dice": mean_metric("surface_dice"),
        "mean_absolute_volume_error_voxels": mean_metric(
            "absolute_volume_error_voxels"
        ),
        "mean_relative_volume_error": mean_metric("relative_volume_error"),
        "growth_direction_accuracy": mean_metric("growth_direction_correct"),
        "uncertainty_absolute_error_spearman": (
            None if not np.isfinite(correlation) else float(correlation)
        ),
    }
    return EvaluationResult(rows=rows, summary=summary)
