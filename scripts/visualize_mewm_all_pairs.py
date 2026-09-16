#!/usr/bin/env python3
"""Create provenance-bound qualitative figures for the MeWM all-pairs run."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ispy2_symmflow.cli import _load_sampling_models
from ispy2_symmflow.config import load_config, resolve_time_pairs
from ispy2_symmflow.data.mewm import (
    _cache_path,
    _validate_bundle,
    _validate_cache_payload,
)
from ispy2_symmflow.inference.sampler import SymmFlowSampler
from ispy2_symmflow.models.conditioning import ConditionSchema
from ispy2_symmflow.training.datasets import LatentPairDataset, read_jsonl
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    LATENT_STATISTICS_FINGERPRINT,
    validate_cached_latent_provenance,
    validate_cached_pair_manifest_binding,
)
from ispy2_symmflow.utils.hashing import sha256_file, stable_hash


SELECTION_NAMESPACE = "ispy2-symmflow-visual-v1"
EXPECTED_BRANCH_ORDER = ["later", "earlier"]
FIXED_CENTER_ZYX = (48, 128, 128)


@dataclass(frozen=True)
class ReferenceVolume:
    observed: np.ndarray
    reconstruction: np.ndarray
    analysis_mask: np.ndarray
    foreground: np.ndarray
    roi_sha256: str
    latent_sha256: str
    latent_parity_max_abs: float


@dataclass
class CaseView:
    pair_id: str
    patient_id: str
    earlier_stage: str
    later_stage: str
    delta_days: int
    forward_window: tuple[float, float]
    backward_window: tuple[float, float]
    forward: dict[str, np.ndarray]
    backward: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vqgan-checkpoint", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--roi-cache-dir", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    if args.num_samples < 1 or args.steps < 1:
        parser.error("--num-samples and --steps must be positive")
    return args


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def _selection_digest(pair_id: str) -> str:
    payload = f"{SELECTION_NAMESPACE}\0{pair_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select_validation_pairs(
    records: Sequence[Mapping[str, Any]],
    time_pairs: Sequence[Sequence[str]],
) -> list[dict[str, Any]]:
    by_interval: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("split") == "val":
            key = (str(record["earlier_stage"]), str(record["later_stage"]))
            by_interval[key].append(record)
    selected: list[dict[str, Any]] = []
    for raw_pair in time_pairs:
        interval = (str(raw_pair[0]), str(raw_pair[1]))
        candidates = by_interval.get(interval, [])
        if not candidates:
            raise ValueError(f"validation split has no pair for {interval}")
        choice = min(candidates, key=lambda row: _selection_digest(str(row["pair_id"])))
        selected.append(dict(choice))
    return selected


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width, height = right - left, bottom - top
    x = box[0] + (box[2] - box[0] - width) // 2
    y = box[1] + (box[3] - box[1] - height) // 2
    draw.text((x, y), text, font=font, fill=fill)


def _gray_image(
    values: np.ndarray, window: tuple[float, float], size: int
) -> Image.Image:
    low, high = window
    scaled = np.clip((np.asarray(values, dtype=np.float32) - low) / (high - low), 0, 1)
    image = Image.fromarray(np.rint(scaled * 255).astype(np.uint8), mode="L")
    return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def _heat_image(
    values: np.ndarray,
    high: float,
    size: int,
    *,
    uncertainty: bool,
) -> Image.Image:
    normalized = np.clip(np.asarray(values, dtype=np.float32) / max(high, 1e-8), 0, 1)
    if uncertainty:
        palette = np.asarray(
            ((7, 15, 24), (21, 91, 121), (50, 181, 151), (245, 224, 112)),
            dtype=np.float32,
        )
    else:
        palette = np.asarray(
            ((8, 8, 12), (75, 24, 95), (207, 55, 70), (255, 224, 92)),
            dtype=np.float32,
        )
    position = normalized * (len(palette) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(palette) - 1)
    fraction = (position - lower)[..., None]
    rgb = palette[lower] * (1 - fraction) + palette[upper] * fraction
    image = Image.fromarray(np.rint(rgb).astype(np.uint8), mode="RGB")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def _source_window(source: np.ndarray, foreground: np.ndarray) -> tuple[float, float]:
    selected = np.asarray(source, dtype=np.float32)[np.asarray(foreground) > 0]
    if not selected.size:
        selected = np.asarray(source, dtype=np.float32).reshape(-1)
    low, high = np.percentile(selected, (0.5, 99.5))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        raise ValueError("cannot derive a finite source-only display window")
    return float(low), float(high)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float32)[np.asarray(mask) > 0]
    if not selected.size or not np.isfinite(selected).all():
        raise ValueError("metric mask is empty or values are non-finite")
    return float(selected.mean())


def _metrics(
    samples: np.ndarray,
    prediction_mean: np.ndarray,
    prediction_std: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    target_vq: np.ndarray,
    foreground: np.ndarray,
) -> dict[str, float]:
    candidate_mae = np.asarray(
        [_masked_mean(np.abs(sample - target), foreground) for sample in samples]
    )
    return {
        "copy_source_foreground_mae": _masked_mean(np.abs(source - target), foreground),
        "target_vq_foreground_mae": _masked_mean(np.abs(target_vq - target), foreground),
        "candidate_foreground_mae_mean": float(candidate_mae.mean()),
        "candidate_foreground_mae_std": float(candidate_mae.std()),
        "sample_0_foreground_mae": float(candidate_mae[0]),
        "predictive_mean_foreground_mae": _masked_mean(
            np.abs(prediction_mean - target), foreground
        ),
        "predictive_mean_vs_target_vq_foreground_mae": _masked_mean(
            np.abs(prediction_mean - target_vq), foreground
        ),
        "predictive_std_foreground_mean": _masked_mean(prediction_std, foreground),
    }


def _reference_volume(
    visit_id: str,
    normalized_latent: torch.Tensor,
    *,
    codec: torch.nn.Module,
    device: torch.device,
    bundle: Any,
    bundle_rows: Mapping[str, Mapping[str, str]],
    roi_cache_dir: Path,
    latent_path: Path,
) -> ReferenceVolume:
    row = bundle_rows[visit_id]
    evidence = _validate_cache_payload(bundle, roi_cache_dir, row)
    roi_path, _ = _cache_path(bundle, roi_cache_dir, visit_id)
    payload = torch.load(roi_path, map_location="cpu", weights_only=True)
    observed = np.asarray(payload["mri"][0], dtype=np.float32)
    analysis_mask = np.asarray(payload["mask"][0], dtype=np.uint8)
    foreground = np.asarray(payload["valid_foreground"][0], dtype=np.uint8)
    source = torch.from_numpy(observed)[None, None].to(device)
    latent = normalized_latent[None].to(device)
    with torch.inference_mode():
        encoded = codec.encode(source, normalize=False)
        expected = codec.denormalize_latent(latent)
        reconstruction = codec.decode(latent, denormalize=True)
    difference = (encoded - expected).abs()
    if not torch.allclose(encoded, expected, rtol=2e-3, atol=2e-2):
        raise ValueError(f"ROI/continuous-latent parity failed for {visit_id}")
    result = ReferenceVolume(
        observed=observed,
        reconstruction=reconstruction[0, 0].detach().cpu().numpy(),
        analysis_mask=analysis_mask,
        foreground=foreground,
        roi_sha256=evidence.cache_sha256,
        latent_sha256=sha256_file(latent_path),
        latent_parity_max_abs=float(difference.max().item()),
    )
    del source, latent, encoded, expected, reconstruction, difference, payload
    return result


def _batch_arrays(batch: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_sample_shape = (len(batch.seeds), 1, 1, 96, 256, 256)
    expected_summary_shape = (1, 1, 96, 256, 256)
    if tuple(batch.samples.shape) != expected_sample_shape:
        raise ValueError(
            f"decoded samples have shape {tuple(batch.samples.shape)}, "
            f"expected {expected_sample_shape}"
        )
    if tuple(batch.mean.shape) != expected_summary_shape or tuple(batch.std.shape) != expected_summary_shape:
        raise ValueError("decoded sample summaries do not have shape [1,1,96,256,256]")
    if any(tensor.dtype != torch.float32 for tensor in (batch.samples, batch.mean, batch.std)):
        raise TypeError("decoded samples and summaries must use float32")
    samples = batch.samples[:, 0, 0].detach().cpu().numpy()
    mean = batch.mean[0, 0].detach().cpu().numpy()
    std = batch.std[0, 0].detach().cpu().numpy()
    if not np.isfinite(samples).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("sampling returned a non-finite decoded volume")
    return samples, mean, std


def _case_key(record: Mapping[str, Any]) -> str:
    return f"{record['earlier_stage']}_{record['later_stage']}__{record['patient_id']}"


def _draw_summary(
    views: Sequence[CaseView],
    path: Path,
    *,
    direction: str,
    error_high: float,
    std_high: float,
) -> None:
    cell = 220
    gap = 9
    left = 245
    top = 145
    row_stride = cell + 58
    columns = (
        "Observed source",
        "Source VQ recon",
        "Observed target",
        "Target VQ recon",
        "Candidate 0",
        "Predictive mean",
        "|Mean - target|",
        "Predictive std",
    )
    width = left + len(columns) * (cell + gap) + 28
    height = top + len(views) * row_stride + 105
    canvas = Image.new("RGB", (width, height), (246, 247, 244))
    draw = ImageDraw.Draw(canvas)
    title = (
        "Forward prediction: earlier to later"
        if direction == "forward"
        else "Backward: registration-assisted reconstruction"
    )
    draw.text((30, 22), title, font=_font(34, bold=True), fill=(22, 28, 34))
    draw.text(
        (30, 70),
        "Six deterministically selected validation cases | T0-mask crop center, axial z=48 | "
        "no target-selected candidates",
        font=_font(20),
        fill=(74, 82, 89),
    )
    for index, label in enumerate(columns):
        x = left + index * (cell + gap)
        _centered_text(draw, (x, 105, x + cell, top), label, _font(17, bold=True), (35, 42, 48))
    for row_index, view in enumerate(views):
        y = top + row_index * row_stride
        values = view.forward if direction == "forward" else view.backward
        window = view.forward_window if direction == "forward" else view.backward_window
        interval = f"{view.earlier_stage} -> {view.later_stage}"
        draw.text((24, y + 62), interval, font=_font(25, bold=True), fill=(28, 35, 40))
        draw.text((24, y + 100), view.patient_id, font=_font(18), fill=(66, 74, 82))
        draw.text((24, y + 130), f"{view.delta_days} days", font=_font(18), fill=(66, 74, 82))
        panels = (
            _gray_image(values["source_observed"], window, cell),
            _gray_image(values["source_vq"], window, cell),
            _gray_image(values["target_observed"], window, cell),
            _gray_image(values["target_vq"], window, cell),
            _gray_image(values["samples"][0], window, cell),
            _gray_image(values["mean"], window, cell),
            _heat_image(values["error"], error_high, cell, uncertainty=False),
            _heat_image(values["std"], std_high, cell, uncertainty=True),
        )
        for column_index, panel in enumerate(panels):
            x = left + column_index * (cell + gap)
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(210, 213, 210), width=1)
        draw.text(
            (left, y + cell + 10),
            f"source-only window [{window[0]:.3f}, {window[1]:.3f}]",
            font=_font(15),
            fill=(83, 89, 94),
        )
    footer_y = height - 73
    draw.text(
        (30, footer_y),
        f"Error scale 0..{error_high:.3f}; uncertainty scale 0..{std_high:.3f}. "
        "Observed ROI is registered/cropped standardized DCE0, not raw DICOM intensity.",
        font=_font(17),
        fill=(65, 72, 78),
    )
    canvas.save(path, format="PNG", optimize=True)


def _draw_candidates(
    view: CaseView,
    path: Path,
    *,
    error_high: float,
    std_high: float,
) -> None:
    cell = 150
    gap = 7
    left = 185
    top = 112
    row_stride = cell + 64
    sample_count = int(view.forward["samples"].shape[0])
    columns = ("Observed target", "Target VQ", "Mean") + tuple(
        f"Candidate {index}" for index in range(sample_count)
    ) + ("Std",)
    width = left + len(columns) * (cell + gap) + 25
    height = top + 2 * row_stride + 72
    canvas = Image.new("RGB", (width, height), (246, 247, 244))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 18),
        f"Decoded candidate set | {view.pair_id}",
        font=_font(31, bold=True),
        fill=(22, 28, 34),
    )
    draw.text(
        (28, 62),
        "T0-mask crop center, axial z=48; candidate seeds fixed before seeing target",
        font=_font(18),
        fill=(74, 82, 89),
    )
    for index, label in enumerate(columns):
        x = left + index * (cell + gap)
        _centered_text(draw, (x, 82, x + cell, top), label, _font(14, bold=True), (35, 42, 48))
    for row_index, (direction, label) in enumerate(
        (("forward", "Forward"), ("backward", "Backward"))
    ):
        y = top + row_index * row_stride
        values = view.forward if direction == "forward" else view.backward
        window = view.forward_window if direction == "forward" else view.backward_window
        draw.text((24, y + 56), label, font=_font(21, bold=True), fill=(28, 35, 40))
        if direction == "backward":
            draw.text(
                (24, y + 88),
                "registration-assisted",
                font=_font(13),
                fill=(66, 74, 82),
            )
        panels = [
            _gray_image(values["target_observed"], window, cell),
            _gray_image(values["target_vq"], window, cell),
            _gray_image(values["mean"], window, cell),
        ]
        panels.extend(
            _gray_image(sample, window, cell) for sample in values["samples"]
        )
        panels.append(_heat_image(values["std"], std_high, cell, uncertainty=True))
        for column_index, panel in enumerate(panels):
            x = left + column_index * (cell + gap)
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(210, 213, 210), width=1)
        metrics = view.metadata[direction]["metrics"]
        draw.text(
            (left, y + cell + 9),
            f"mean foreground MAE={metrics['predictive_mean_foreground_mae']:.4f}; "
            f"candidate MAE={metrics['candidate_foreground_mae_mean']:.4f} +/- "
            f"{metrics['candidate_foreground_mae_std']:.4f}",
            font=_font(15),
            fill=(76, 82, 87),
        )
    canvas.save(path, format="PNG", optimize=True)


def _draw_training_curve(metrics_path: Path, output_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    training = [row for row in rows if row.get("event") == "train"]
    validation = [row for row in rows if row.get("event") == "validation"]
    if not training or not validation:
        raise ValueError("training metric log has no train or validation records")
    train_steps = np.asarray([row["optimizer_step"] for row in training], dtype=np.float64)
    train_loss = np.asarray([row["loss"] for row in training], dtype=np.float64)
    smooth_window = min(50, len(train_loss))
    kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
    smooth_loss = np.convolve(train_loss, kernel, mode="valid")
    smooth_steps = train_steps[smooth_window - 1 :]
    val_steps = np.asarray([row["optimizer_step"] for row in validation], dtype=np.float64)
    val_loss = np.asarray([row["loss"] for row in validation], dtype=np.float64)
    width, height = 1500, 880
    left, right, top, bottom = 125, 55, 115, 105
    plot_width, plot_height = width - left - right, height - top - bottom
    all_loss = np.concatenate((smooth_loss, val_loss))
    y_low = max(0.0, float(np.percentile(all_loss, 0.5)) - 0.08)
    y_high = float(np.percentile(all_loss, 99.5)) + 0.12
    if y_high <= y_low:
        y_high = y_low + 1.0

    def xy(step: float, loss: float) -> tuple[int, int]:
        x_fraction = float(np.clip(step / 100000.0, 0.0, 1.0))
        y_fraction = float(np.clip((y_high - loss) / (y_high - y_low), 0.0, 1.0))
        x = left + int(round(x_fraction * plot_width))
        y = top + int(round(y_fraction * plot_height))
        return x, y

    canvas = Image.new("RGB", (width, height), (248, 249, 247))
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 25), "SymmFlow training history", font=_font(34, bold=True), fill=(22, 28, 34))
    draw.text(
        (left, 73),
        "Training loss: 500-step moving mean | Validation: all 538 held-out pairs every 1,000 steps",
        font=_font(18),
        fill=(74, 82, 89),
    )
    for index in range(6):
        step = index * 20000
        x, _ = xy(step, y_low)
        draw.line((x, top, x, top + plot_height), fill=(222, 225, 222), width=1)
        _centered_text(
            draw,
            (x - 55, top + plot_height + 16, x + 55, top + plot_height + 54),
            f"{step // 1000}k",
            _font(16),
            (74, 82, 89),
        )
    for index in range(6):
        loss = y_low + index * (y_high - y_low) / 5
        _, y = xy(0, loss)
        draw.line((left, y, left + plot_width, y), fill=(222, 225, 222), width=1)
        draw.text((35, y - 10), f"{loss:.2f}", font=_font(16), fill=(74, 82, 89))
    draw.line((left, top, left, top + plot_height), fill=(52, 58, 64), width=2)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill=(52, 58, 64),
        width=2,
    )
    train_points = [xy(step, loss) for step, loss in zip(smooth_steps, smooth_loss, strict=True)]
    val_points = [xy(step, loss) for step, loss in zip(val_steps, val_loss, strict=True)]
    draw.line(train_points, fill=(23, 118, 137), width=4, joint="curve")
    draw.line(val_points, fill=(202, 69, 63), width=4, joint="curve")
    for point in val_points[::10]:
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=(202, 69, 63))
    legend_y = height - 48
    draw.line((left, legend_y, left + 45, legend_y), fill=(23, 118, 137), width=5)
    draw.text((left + 58, legend_y - 12), "train moving mean", font=_font(17), fill=(45, 51, 57))
    draw.line((left + 310, legend_y, left + 355, legend_y), fill=(202, 69, 63), width=5)
    draw.text((left + 368, legend_y - 12), "validation", font=_font(17), fill=(45, 51, 57))
    draw.text(
        (left + 610, legend_y - 12),
        f"validation {val_loss[0]:.4f} -> {val_loss[-1]:.4f} "
        f"({(val_loss[0] - val_loss[-1]) / val_loss[0] * 100:.1f}% lower)",
        font=_font(17, bold=True),
        fill=(45, 51, 57),
    )
    canvas.save(output_path, format="PNG", optimize=True)
    return {
        "training_record_count": len(training),
        "validation_record_count": len(validation),
        "moving_average_optimizer_steps": 500,
        "final_training_optimizer_step": int(training[-1]["optimizer_step"]),
        "final_training_loss": float(training[-1]["loss"]),
        "final_validation_optimizer_step": int(validation[-1]["optimizer_step"]),
        "first_validation_loss": float(val_loss[0]),
        "final_validation_loss": float(val_loss[-1]),
        "best_validation_loss": float(val_loss.min()),
        "validation_history_fingerprint": stable_hash(validation),
        "validation_loss_reduction_percent": float(
            (val_loss[0] - val_loss[-1]) / val_loss[0] * 100
        ),
    }


def _report_markdown(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# MeWM all-pairs qualitative report",
        "",
        "This report is a deterministically case-selected qualitative audit, not a clinical result.",
        "Each row was selected before inference by a SHA-256 rule within one held-out",
        "time-pair stratum. No target metric was used to choose a case or candidate.",
        "",
        f"Sampling used {manifest['sampling']['num_samples']} independently decoded candidates per direction, "
        f"{manifest['sampling']['steps']}-step {manifest['sampling']['solver'].title()} integration, "
        f"and {manifest['sampling']['nfe_per_sample']} function evaluations per candidate.",
        f"Flow checkpoint step: {manifest['checkpoint_step']}; base seed: "
        f"{manifest['sampling']['base_seed']}.",
        f"Flow SHA-256: `{manifest['checkpoint_sha256']}`.",
        f"Frozen VQ-GAN SHA-256: `{manifest['vqgan_checkpoint_sha256']}`.",
        f"Pair manifest SHA-256: `{manifest['pair_manifest_sha256']}`.",
        f"Training metrics SHA-256: `{manifest['training_metrics_sha256']}`.",
        "[Full provenance, per-case seeds, and artifact hashes](manifest.json)",
        "",
        "![Training curves](training_curves.png)",
        "",
        "![Forward summary](forward_summary.png)",
        "",
        "![Backward summary](backward_summary.png)",
        "",
        "## Selected cases",
        "",
        "| Pair | Patient | Days | Forward mean MAE | Backward mean MAE | Candidates |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in manifest["cases"]:
        forward = case["forward"]["metrics"]["predictive_mean_foreground_mae"]
        backward = case["backward"]["metrics"]["predictive_mean_foreground_mae"]
        lines.append(
            f"| {case['earlier_stage']}->{case['later_stage']} | {case['patient_id']} | "
            f"{case['delta_days']} | {forward:.4f} | {backward:.4f} | "
            f"[view]({case['candidate_figure']}) |"
        )
    lines.extend(
        (
            "",
            "MAE is reported in the upstream standardized DCE0 intensity units over the",
            "valid foreground. Observed panels are registered/cropped cache arrays rather",
            "than raw DICOM intensity. VQ reconstruction panels expose the frozen codec",
            "ceiling. Masks are stored only as offline audit arrays and were not model input.",
            "The backward direction is registration-assisted reconstruction because every",
            "visit uses a T0-fixed registration and T0-mask-derived crop.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = _arguments()
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"visualization output already exists: {destination}")
    for path in (
        args.config,
        args.checkpoint,
        args.vqgan_checkpoint,
        args.pair_manifest,
        args.bundle_dir,
        args.roi_cache_dir,
        args.training_metrics,
    ):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)

    config = load_config(args.config)
    pair_manifest_path = args.pair_manifest.resolve()
    pair_manifest_sha256 = sha256_file(pair_manifest_path)
    training_metrics_path = args.training_metrics.resolve()
    training_metrics_sha256 = sha256_file(training_metrics_path)
    records = read_jsonl(pair_manifest_path)
    statistics_path = args.pair_manifest.resolve().with_name("latent_statistics.json")
    statistics = _json(statistics_path)
    statistics_fingerprint, pair_fingerprint, pair_count = (
        validate_cached_pair_manifest_binding(records, statistics)
    )
    time_pairs = resolve_time_pairs(config["data"])
    selected = _select_validation_pairs(records, time_pairs)
    validate_cached_latent_provenance(selected, statistics, validate_manifest=False)
    dataset = LatentPairDataset(selected)
    items = [dataset[index] for index in range(len(dataset))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("full-volume MeWM visualization requires CUDA")
    print("Loading final EMA SymmFlow and frozen MeWM VQ-GAN...", flush=True)
    codec, velocity, encoder, header = _load_sampling_models(
        config,
        str(args.checkpoint.resolve()),
        str(args.vqgan_checkpoint.resolve()),
        device,
    )
    if stable_hash(config) != stable_hash(header["config"]):
        raise ValueError("external config differs from the checkpoint training config")
    if sha256_file(pair_manifest_path) != pair_manifest_sha256:
        raise ValueError("pair manifest changed while the visualization inputs were loading")
    extra = header.get("extra")
    if not isinstance(extra, Mapping) or extra.get("branch_order") != EXPECTED_BRANCH_ORDER:
        raise ValueError("checkpoint branch order is not [later, earlier]")
    max_steps = int(header["config"]["flow"]["max_steps"])
    if int(header["step"]) != max_steps:
        raise ValueError(
            f"checkpoint step {header['step']} is not the configured final step {max_steps}"
        )
    condition_schema_fingerprint = ConditionSchema.from_dict(
        header["feature_schema"]
    ).fingerprint
    schema_provenance = extra.get("schema_provenance")
    if (
        not isinstance(schema_provenance, Mapping)
        or schema_provenance.get("schema_fingerprint") != condition_schema_fingerprint
    ):
        raise ValueError("checkpoint condition schema fingerprint is inconsistent")
    if float(header["config"]["flow"].get("sigma_min", 0.0)) != 0.0:
        raise ValueError("qualitative report requires clean sigma_min=0 endpoints")
    if stable_hash(header["latent_statistics"]) != stable_hash(statistics):
        raise ValueError("checkpoint and disk latent statistics differ")
    sampler = SymmFlowSampler(codec, velocity, sigma_min=0.0)

    bundle = _validate_bundle(args.bundle_dir)
    bundle_rows = {row["visit_id"]: row for row in bundle.visits}
    staging_parent = destination.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=staging_parent))
    views: list[CaseView] = []
    cases: list[dict[str, Any]] = []
    try:
        for case_index, (record, item) in enumerate(zip(selected, items, strict=True)):
            pair_id = str(record["pair_id"])
            print(f"[{case_index + 1}/{len(selected)}] {pair_id}: references", flush=True)
            earlier = _reference_volume(
                str(record["earlier_visit_id"]),
                item["earlier_latent"],
                codec=codec,
                device=device,
                bundle=bundle,
                bundle_rows=bundle_rows,
                roi_cache_dir=args.roi_cache_dir.resolve(),
                latent_path=Path(record["earlier_latent_path"]),
            )
            later = _reference_volume(
                str(record["later_visit_id"]),
                item["later_latent"],
                codec=codec,
                device=device,
                bundle=bundle,
                bundle_rows=bundle_rows,
                roi_cache_dir=args.roi_cache_dir.resolve(),
                latent_path=Path(record["later_latent_path"]),
            )
            conditions = item["conditions"]
            with torch.inference_mode():
                tokens = encoder(conditions, batch_size=1)
            forward_seed = int(args.seed + case_index * 100)
            backward_seed = int(args.seed + 10000 + case_index * 100)
            print(f"[{case_index + 1}/{len(selected)}] {pair_id}: forward K={args.num_samples}", flush=True)
            forward_batch = sampler.sample_forward_latent(
                item["earlier_latent"][None].to(device),
                tokens,
                num_samples=args.num_samples,
                seed=forward_seed,
                steps=args.steps,
                solver=args.solver,
                return_joint_state=False,
            )
            forward_samples, forward_mean, forward_std = _batch_arrays(forward_batch)
            forward_seeds = list(forward_batch.seeds)
            forward_nfe = int(forward_batch.nfe_per_sample)
            del forward_batch
            print(f"[{case_index + 1}/{len(selected)}] {pair_id}: backward K={args.num_samples}", flush=True)
            backward_batch = sampler.sample_backward_latent(
                item["later_latent"][None].to(device),
                tokens,
                num_samples=args.num_samples,
                seed=backward_seed,
                steps=args.steps,
                solver=args.solver,
                return_joint_state=False,
            )
            backward_samples, backward_mean, backward_std = _batch_arrays(backward_batch)
            backward_seeds = list(backward_batch.seeds)
            backward_nfe = int(backward_batch.nfe_per_sample)
            del backward_batch, tokens
            expected_nfe = args.steps * (2 if args.solver == "heun" else 1)
            if forward_nfe != expected_nfe or backward_nfe != expected_nfe:
                raise ValueError("ODE solver returned an unexpected number of evaluations")

            forward_metrics = _metrics(
                forward_samples,
                forward_mean,
                forward_std,
                earlier.observed,
                later.observed,
                later.reconstruction,
                later.foreground,
            )
            backward_metrics = _metrics(
                backward_samples,
                backward_mean,
                backward_std,
                later.observed,
                earlier.observed,
                earlier.reconstruction,
                earlier.foreground,
            )
            forward_window = _source_window(earlier.observed, earlier.foreground)
            backward_window = _source_window(later.observed, later.foreground)
            case_directory = staging / _case_key(record)
            case_directory.mkdir()
            arrays_path = case_directory / "decoded_samples.npz"
            np.savez_compressed(
                arrays_path,
                earlier_observed_dce0=earlier.observed,
                later_observed_dce0=later.observed,
                earlier_vq_reconstruction=earlier.reconstruction,
                later_vq_reconstruction=later.reconstruction,
                earlier_analysis_mask=earlier.analysis_mask,
                later_analysis_mask=later.analysis_mask,
                earlier_valid_foreground=earlier.foreground,
                later_valid_foreground=later.foreground,
                forward_samples=forward_samples,
                forward_mean=forward_mean,
                forward_std=forward_std,
                backward_samples=backward_samples,
                backward_mean=backward_mean,
                backward_std=backward_std,
                affine_lps=np.asarray(record["affine_lps"], dtype=np.float64),
                spacing_dhw=np.asarray(record["spacing_dhw"], dtype=np.float32),
            )
            arrays_sha256 = sha256_file(arrays_path)
            case_metadata = {
                "schema": "ispy2_symmflow_mewm_qualitative_case_v1",
                "pair_id": pair_id,
                "patient_id": record["patient_id"],
                "split": "val",
                "earlier_stage": record["earlier_stage"],
                "later_stage": record["later_stage"],
                "earlier_visit_id": record["earlier_visit_id"],
                "later_visit_id": record["later_visit_id"],
                "delta_days": int(record["delta_days"]),
                "selection_digest": _selection_digest(pair_id),
                "conditions": conditions,
                "coordinate_frame": record["coordinate_frame"],
                "crop_policy": record["crop_policy"],
                "affine_lps": record["affine_lps"],
                "spacing_dhw": record["spacing_dhw"],
                "fixed_display_center_zyx": list(FIXED_CENTER_ZYX),
                "fixed_display_center_policy": (
                    "T0-mask-derived crop center reused for all registered visits"
                ),
                "display_window_policy": "directional_observed_source_foreground_p0.5_p99.5",
                "display_windows": {
                    "forward": list(forward_window),
                    "backward": list(backward_window),
                },
                "decoded_arrays": arrays_path.name,
                "decoded_arrays_base": "case_directory",
                "decoded_arrays_report_relative": arrays_path.relative_to(staging).as_posix(),
                "decoded_arrays_sha256": arrays_sha256,
                "earlier_latent_path": str(Path(record["earlier_latent_path"]).resolve()),
                "earlier_latent_sha256": earlier.latent_sha256,
                "later_latent_path": str(Path(record["later_latent_path"]).resolve()),
                "later_latent_sha256": later.latent_sha256,
                "earlier_roi_cache_sha256": earlier.roi_sha256,
                "later_roi_cache_sha256": later.roi_sha256,
                "earlier_roi_latent_parity_max_abs": earlier.latent_parity_max_abs,
                "later_roi_latent_parity_max_abs": later.latent_parity_max_abs,
                "forward": {
                    "seeds": forward_seeds,
                    "nfe_per_sample": forward_nfe,
                    "metrics": forward_metrics,
                    "target": record["later_visit_id"],
                },
                "backward": {
                    "task_label": header["config"]["data"]["backward_task_label"],
                    "seeds": backward_seeds,
                    "nfe_per_sample": backward_nfe,
                    "metrics": backward_metrics,
                    "target": record["earlier_visit_id"],
                },
                "metric_scope": "valid_foreground_standardized_dce0_intensity",
                "analysis_mask_semantics": (
                    "upstream semi-automatic analysis mask; audit-only, not model input, "
                    "not asserted to be an expert tumor segmentation"
                ),
                "reference_semantics": (
                    "observed arrays are registered/cropped standardized DCE0 ROI cache values; "
                    "VQ reconstructions decode observed continuous latents"
                ),
            }
            _write_json(case_directory / "case.json", case_metadata)
            z = FIXED_CENTER_ZYX[0]
            forward_view = {
                "source_observed": earlier.observed[z].copy(),
                "source_vq": earlier.reconstruction[z].copy(),
                "target_observed": later.observed[z].copy(),
                "target_vq": later.reconstruction[z].copy(),
                "samples": forward_samples[:, z].copy(),
                "mean": forward_mean[z].copy(),
                "std": forward_std[z].copy(),
                "error": np.abs(forward_mean[z] - later.observed[z]),
            }
            backward_view = {
                "source_observed": later.observed[z].copy(),
                "source_vq": later.reconstruction[z].copy(),
                "target_observed": earlier.observed[z].copy(),
                "target_vq": earlier.reconstruction[z].copy(),
                "samples": backward_samples[:, z].copy(),
                "mean": backward_mean[z].copy(),
                "std": backward_std[z].copy(),
                "error": np.abs(backward_mean[z] - earlier.observed[z]),
            }
            views.append(
                CaseView(
                    pair_id=pair_id,
                    patient_id=str(record["patient_id"]),
                    earlier_stage=str(record["earlier_stage"]),
                    later_stage=str(record["later_stage"]),
                    delta_days=int(record["delta_days"]),
                    forward_window=forward_window,
                    backward_window=backward_window,
                    forward=forward_view,
                    backward=backward_view,
                    metadata=case_metadata,
                )
            )
            cases.append(case_metadata)
            del (
                forward_samples,
                forward_mean,
                forward_std,
                backward_samples,
                backward_mean,
                backward_std,
                earlier,
                later,
            )
            gc.collect()
            torch.cuda.empty_cache()

        all_errors = np.concatenate(
            [view.forward["error"].reshape(-1) for view in views]
            + [view.backward["error"].reshape(-1) for view in views]
        )
        all_std = np.concatenate(
            [view.forward["std"].reshape(-1) for view in views]
            + [view.backward["std"].reshape(-1) for view in views]
        )
        error_high = float(np.percentile(all_errors, 99.0))
        std_high = float(np.percentile(all_std, 99.0))
        _draw_summary(
            views,
            staging / "forward_summary.png",
            direction="forward",
            error_high=error_high,
            std_high=std_high,
        )
        _draw_summary(
            views,
            staging / "backward_summary.png",
            direction="backward",
            error_high=error_high,
            std_high=std_high,
        )
        for view, case in zip(views, cases, strict=True):
            relative = f"{_case_key(case)}/axial_candidates.png"
            _draw_candidates(
                view,
                staging / relative,
                error_high=error_high,
                std_high=std_high,
            )
            case["candidate_figure"] = relative
            _write_json(staging / _case_key(case) / "case.json", case)
        training_summary = _draw_training_curve(
            training_metrics_path, staging / "training_curves.png"
        )
        if sha256_file(training_metrics_path) != training_metrics_sha256:
            raise ValueError("training metrics changed while the report was being generated")
        checkpoint_history = extra.get("validation_history")
        if not isinstance(checkpoint_history, list):
            raise ValueError("checkpoint has no validation history")
        flattened_checkpoint_history = []
        for checkpoint_record in checkpoint_history:
            if not isinstance(checkpoint_record, Mapping) or not isinstance(
                checkpoint_record.get("metrics"), Mapping
            ):
                raise ValueError("checkpoint validation history is malformed")
            flattened = {
                key: value
                for key, value in checkpoint_record.items()
                if key != "metrics"
            }
            flattened.update(checkpoint_record["metrics"])
            flattened["event"] = "validation"
            flattened_checkpoint_history.append(flattened)
        if stable_hash(flattened_checkpoint_history) != training_summary[
            "validation_history_fingerprint"
        ]:
            raise ValueError("training metrics validation history differs from checkpoint")
        last_train_metrics = extra.get("last_train_metrics")
        if not isinstance(last_train_metrics, Mapping):
            raise ValueError("checkpoint has no final training metrics")
        best_validation_loss = float(extra["best_validation_loss"])
        if (
            training_summary["final_training_optimizer_step"] != int(header["step"])
            or training_summary["final_validation_optimizer_step"] != int(header["step"])
            or not math.isclose(
                training_summary["final_training_loss"],
                float(last_train_metrics["loss"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                training_summary["final_validation_loss"],
                best_validation_loss,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                training_summary["best_validation_loss"],
                best_validation_loss,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("training metrics terminal state differs from checkpoint")
        manifest: dict[str, Any] = {
            "schema": "ispy2_symmflow_mewm_qualitative_report_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection": {
                "namespace": SELECTION_NAMESPACE,
                "rule": "minimum SHA256(namespace + NUL + pair_id) within each val interval",
                "uses_prediction_or_target_quality": False,
            },
            "case_count": len(cases),
            "cases": cases,
            "sampling": {
                "num_samples": args.num_samples,
                "solver": args.solver,
                "steps": args.steps,
                "nfe_per_sample": args.steps * (2 if args.solver == "heun" else 1),
                "base_seed": args.seed,
                "branch_order": EXPECTED_BRANCH_ORDER,
                "sigma_min": 0.0,
                "candidate_selection": "sample index 0 only; never target-selected best-of-K",
            },
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_step": int(header["step"]),
            "resolved_config_fingerprint": stable_hash(config),
            "visualization_script": str(Path(__file__).resolve()),
            "visualization_script_sha256": sha256_file(Path(__file__)),
            "vqgan_checkpoint": str(args.vqgan_checkpoint.resolve()),
            "vqgan_checkpoint_sha256": sha256_file(args.vqgan_checkpoint),
            "pair_manifest": str(pair_manifest_path),
            "pair_manifest_sha256": pair_manifest_sha256,
            CACHED_PAIR_MANIFEST_FINGERPRINT: pair_fingerprint,
            CACHED_PAIR_MANIFEST_RECORD_COUNT: pair_count,
            LATENT_STATISTICS_FINGERPRINT: statistics_fingerprint,
            "condition_schema_fingerprint": condition_schema_fingerprint,
            "training_metrics": str(training_metrics_path),
            "training_metrics_sha256": training_metrics_sha256,
            "training": training_summary,
            "global_display_scales": {
                "display_axial_absolute_error_p99": error_high,
                "display_axial_predictive_std_p99": std_high,
            },
            "artifacts": {},
            "limitations": [
                "six deterministically selected validation cases are qualitative, not a cohort estimate",
                "observed ROI values are registered/cropped standardized DCE0, not raw DICOM intensity",
                "predictions include the frozen VQ-GAN decoding ceiling",
                "backward output is registration-assisted reconstruction",
                "analysis masks are audit-only and were not model inputs",
            ],
        }
        report_path = staging / "README.md"
        report_path.write_text(_report_markdown(manifest), encoding="utf-8")
        artifacts = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                artifacts[path.relative_to(staging).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        manifest["artifacts"] = artifacts
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        del sampler, codec, velocity, encoder
        torch.cuda.empty_cache()

    print(json.dumps({
        "output_dir": str(destination),
        "case_count": len(cases),
        "forward_summary": str(destination / "forward_summary.png"),
        "backward_summary": str(destination / "backward_summary.png"),
        "training_curves": str(destination / "training_curves.png"),
        "manifest": str(destination / "manifest.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
