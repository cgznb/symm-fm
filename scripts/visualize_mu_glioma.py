#!/usr/bin/env python3
"""Create provenance-bound MU-Glioma four-modality prediction figures."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F

from ispy2_symmflow.cli import (
    _load_baseline_sampling_models,
    _load_sampling_models,
)
from ispy2_symmflow.config import load_config
from ispy2_symmflow.inference.baselines import UnidirectionalCFMSampler
from ispy2_symmflow.inference.sampler import SymmFlowSampler
from ispy2_symmflow.models.conditioning import ConditionSchema
from ispy2_symmflow.training.datasets import LatentPairDataset, read_jsonl
from ispy2_symmflow.training.provenance import (
    validate_cached_latent_provenance,
    validate_cached_pair_manifest_binding,
)
from ispy2_symmflow.utils.hashing import sha256_file, stable_hash


MODALITIES = ("t1c", "t1n", "t2f", "t2w")
DISPLAY_INTERVALS = (("T1", "T2"), ("T2", "T4"), ("T3", "T6"))
SELECTION_NAMESPACE = "mu-glioma-symmflow-visual-v1"
EXPECTED_BRANCH_ORDER = ["later", "earlier"]
INPUT_SHAPE_XYZ = (240, 240, 155)
OUTPUT_SHAPE_ZYX = (160, 256, 256)
PAD_ZYX = ((2, 3), (8, 8), (8, 8))


@dataclass(frozen=True)
class RawVisit:
    image: np.ndarray
    brain_support: np.ndarray
    tumor_mask: np.ndarray
    reconstruction: np.ndarray
    image_sha256: dict[str, str]
    mask_sha256: str
    latent_parity_max_abs: float


@dataclass(frozen=True)
class CaseResult:
    record: dict[str, Any]
    source: RawVisit
    target: RawVisit
    samples: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    display_z: int
    crop_yxyx: tuple[int, int, int, int]
    windows: tuple[tuple[float, float], ...]
    metrics: dict[str, Any]
    seeds: tuple[int, ...]
    nfe_per_sample: int


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vqgan-checkpoint", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--solver", choices=("euler", "heun"), default="heun")
    parser.add_argument(
        "--model-family",
        choices=("symmflow", "unidirectional_cfm"),
        default="symmflow",
    )
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if args.num_samples < 2:
        parser.error("--num-samples must be at least 2 to estimate uncertainty")
    if args.steps < 1:
        parser.error("--steps must be positive")
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
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for earlier, later in DISPLAY_INTERVALS:
        candidates = [
            record
            for record in records
            if record.get("split") == "val"
            and record.get("earlier_stage") == earlier
            and record.get("later_stage") == later
        ]
        if not candidates:
            raise ValueError(f"validation split has no {earlier}->{later} pair")
        chosen = min(candidates, key=lambda row: _selection_digest(str(row["pair_id"])))
        selected.append(dict(chosen))
    if len({row["patient_id"] for row in selected}) != len(selected):
        raise ValueError("fixed qualitative selection unexpectedly repeats a patient")
    return selected


def _source_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    expected_columns = {
        "patient_id",
        "timepoint",
        "modality",
        "split",
        "image_path",
        "mask_path",
        "image_sha256",
        "mask_sha256",
        "image_shape_xyz",
        "image_mean",
        "image_std",
        "nonzero_voxels",
    }
    if not rows or not expected_columns.issubset(rows[0]):
        raise ValueError("MU-Glioma source manifest does not match the expected schema")
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["patient_id"], row["timepoint"], row["modality"])
        if key in indexed:
            raise ValueError(f"duplicate MU-Glioma source row: {key}")
        indexed[key] = row
    return indexed


def _raw_path(raw_root: Path, row: Mapping[str, str], *, mask: bool = False) -> Path:
    source_key = "mask_path" if mask else "image_path"
    source = str(row[source_key]).strip()
    if not source:
        raise ValueError(f"source manifest has no {source_key}")
    return (
        raw_root
        / str(row["patient_id"])
        / str(row["timepoint"])
        / Path(source).name
    ).resolve()


def _load_nifti(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.load(str(path))
    if tuple(image.shape) != INPUT_SHAPE_XYZ:
        raise ValueError(f"unexpected NIfTI shape for {path}: {image.shape}")
    value = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite NIfTI values: {path}")
    return value


def _pad_zyx(value: torch.Tensor) -> torch.Tensor:
    depth, height, width = PAD_ZYX
    return F.pad(
        value,
        (width[0], width[1], height[0], height[1], depth[0], depth[1]),
        mode="constant",
        value=0,
    )


def _preprocess_image(path: Path, row: Mapping[str, str]) -> tuple[np.ndarray, np.ndarray]:
    expected_sha256 = str(row["image_sha256"])
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"source image SHA-256 mismatch: {path}")
    if tuple(json.loads(row["image_shape_xyz"])) != INPUT_SHAPE_XYZ:
        raise ValueError(f"source manifest shape mismatch: {path}")
    xyz = _load_nifti(path)
    zyx = torch.from_numpy(np.ascontiguousarray(xyz.transpose(2, 1, 0)))
    padded = _pad_zyx(zyx)
    if tuple(padded.shape) != OUTPUT_SHAPE_ZYX:
        raise RuntimeError("MU-Glioma padding contract changed")
    support = padded != 0
    count = int(support.sum())
    foreground = padded[support]
    mean = foreground.mean()
    std = foreground.std(unbiased=False)
    if not math.isclose(float(mean), float(row["image_mean"]), rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError(f"source image mean mismatch: {path}")
    if not math.isclose(float(std), float(row["image_std"]), rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError(f"source image std mismatch: {path}")
    if count != int(row["nonzero_voxels"]):
        raise ValueError(f"source image nonzero count mismatch: {path}")
    normalized = padded.clone()
    normalized[support] = (foreground - mean) / (std if float(std) else 1.0)
    return normalized.numpy(), support.numpy()


def _load_mask(raw_root: Path, rows: Sequence[Mapping[str, str]]) -> tuple[np.ndarray, str]:
    mask_paths = {_raw_path(raw_root, row, mask=True) for row in rows}
    mask_hashes = {str(row["mask_sha256"]) for row in rows}
    if len(mask_paths) != 1 or len(mask_hashes) != 1 or "" in mask_hashes:
        raise ValueError("visit modalities do not bind one source tumor mask")
    path = next(iter(mask_paths))
    expected_sha256 = next(iter(mask_hashes))
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"source mask SHA-256 mismatch: {path}")
    xyz = _load_nifti(path)
    zyx = torch.from_numpy(np.ascontiguousarray(xyz.transpose(2, 1, 0)))
    mask = _pad_zyx(zyx).numpy() > 0
    return mask, actual_sha256


def _load_raw_visit(
    patient_id: str,
    visit_id: str,
    normalized_latent: torch.Tensor,
    *,
    rows: Mapping[tuple[str, str, str], Mapping[str, str]],
    raw_root: Path,
    codec: torch.nn.Module,
    device: torch.device,
) -> RawVisit:
    marker = "__Timepoint_"
    if marker not in visit_id:
        raise ValueError(f"invalid MU-Glioma visit id: {visit_id}")
    timepoint = f"Timepoint_{visit_id.rsplit(marker, 1)[1]}"
    visit_rows = []
    images = []
    supports = []
    image_hashes: dict[str, str] = {}
    for modality in MODALITIES:
        key = (patient_id, timepoint, modality)
        if key not in rows:
            raise ValueError(f"source manifest has no row for {key}")
        row = rows[key]
        if row["split"] != "val":
            raise ValueError(f"qualitative source row is not held out: {key}")
        path = _raw_path(raw_root, row)
        image, support = _preprocess_image(path, row)
        visit_rows.append(row)
        images.append(image)
        supports.append(support)
        image_hashes[modality] = sha256_file(path)
    tumor_mask, mask_sha256 = _load_mask(raw_root, visit_rows)
    image_array = np.stack(images).astype(np.float32, copy=False)
    support_array = np.stack(supports)
    image_tensor = torch.from_numpy(image_array)[None].to(device)
    latent_tensor = normalized_latent[None].to(device)
    with torch.inference_mode():
        encoded = codec.encode(image_tensor, normalize=False)
        expected = codec.denormalize_latent(latent_tensor)
        reconstruction = codec.decode(latent_tensor, denormalize=True)
    difference = (encoded - expected).abs()
    if not torch.allclose(encoded, expected, rtol=2e-3, atol=2e-2):
        raise ValueError(f"raw-image/continuous-latent parity failed for {visit_id}")
    reconstructed = reconstruction[0].float().cpu().numpy()
    parity = float(difference.max().item())
    del image_tensor, latent_tensor, encoded, expected, reconstruction, difference
    torch.cuda.empty_cache()
    return RawVisit(
        image=image_array,
        brain_support=support_array,
        tumor_mask=tumor_mask,
        reconstruction=reconstructed,
        image_sha256=image_hashes,
        mask_sha256=mask_sha256,
        latent_parity_max_abs=parity,
    )


def _display_slice(mask: np.ndarray, support: np.ndarray) -> int:
    tumor_counts = np.asarray(mask, dtype=np.int64).sum(axis=(1, 2))
    if int(tumor_counts.max()) > 0:
        return int(tumor_counts.argmax())
    brain_counts = np.asarray(support, dtype=np.int64).sum(axis=(1, 2))
    occupied = np.flatnonzero(brain_counts)
    if not occupied.size:
        raise ValueError("source brain support is empty")
    return int(occupied[len(occupied) // 2])


def _crop_from_mask(mask: np.ndarray, support: np.ndarray, z: int) -> tuple[int, int, int, int]:
    coordinates = np.argwhere(mask[z])
    if not coordinates.size:
        coordinates = np.argwhere(support[z])
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0) + 1
    center_y = int(round((int(y0) + int(y1)) / 2))
    center_x = int(round((int(x0) + int(x1)) / 2))
    side = max(96, int(y1 - y0) + 48, int(x1 - x0) + 48)
    side = min(side, OUTPUT_SHAPE_ZYX[1], OUTPUT_SHAPE_ZYX[2])
    top = min(max(center_y - side // 2, 0), OUTPUT_SHAPE_ZYX[1] - side)
    left = min(max(center_x - side // 2, 0), OUTPUT_SHAPE_ZYX[2] - side)
    return top, left, top + side, left + side


def _display_windows(source: RawVisit) -> tuple[tuple[float, float], ...]:
    windows = []
    for modality_index in range(len(MODALITIES)):
        values = source.image[modality_index][source.brain_support[modality_index]]
        low, high = np.percentile(values, (0.5, 99.5))
        if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
            raise ValueError("cannot derive a finite source-only display window")
        windows.append((float(low), float(high)))
    return tuple(windows)


def _masked_mean(value: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(value, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    if not selected.size or not np.isfinite(selected).all():
        raise ValueError("metric mask is empty or contains non-finite values")
    return float(selected.mean())


def _case_metrics(
    source: RawVisit,
    target: RawVisit,
    samples: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"per_modality": {}}
    for index, modality in enumerate(MODALITIES):
        mask = target.brain_support[index]
        candidate_mae = [
            _masked_mean(np.abs(sample[index] - target.image[index]), mask)
            for sample in samples
        ]
        result["per_modality"][modality] = {
            "copy_source_mae": _masked_mean(
                np.abs(source.image[index] - target.image[index]), mask
            ),
            "target_vq_reconstruction_mae": _masked_mean(
                np.abs(target.reconstruction[index] - target.image[index]), mask
            ),
            "candidate_mae_mean": float(np.mean(candidate_mae)),
            "candidate_mae_std": float(np.std(candidate_mae)),
            "candidate_0_mae": float(candidate_mae[0]),
            "predictive_mean_mae": _masked_mean(
                np.abs(mean[index] - target.image[index]), mask
            ),
            "predictive_std_mean": _masked_mean(std[index], mask),
        }
    for metric in (
        "copy_source_mae",
        "target_vq_reconstruction_mae",
        "candidate_mae_mean",
        "predictive_mean_mae",
        "predictive_std_mean",
    ):
        result[f"macro_{metric}"] = float(
            np.mean([result["per_modality"][name][metric] for name in MODALITIES])
        )
    return result


def _sample_forward(
    sampler: Any,
    source_latent: torch.Tensor,
    tokens: torch.Tensor,
    *,
    num_samples: int,
    seed: int,
    steps: int,
    solver: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], int]:
    outputs = []
    seeds = []
    nfe: int | None = None
    for index in range(num_samples):
        current_seed = seed + index
        print(f"    candidate {index + 1}/{num_samples} (seed {current_seed})", flush=True)
        batch = sampler.sample_forward_latent(
            source_latent,
            tokens,
            num_samples=1,
            seed=current_seed,
            steps=steps,
            solver=solver,
            return_joint_state=False,
        )
        value = batch.samples[0, 0].float().cpu().numpy()
        if value.shape != (len(MODALITIES), *OUTPUT_SHAPE_ZYX):
            raise ValueError(f"decoded candidate has unexpected shape {value.shape}")
        if not np.isfinite(value).all():
            raise FloatingPointError("sampling returned a non-finite decoded volume")
        outputs.append(value)
        seeds.append(current_seed)
        nfe = batch.nfe_per_sample
        del batch
        torch.cuda.empty_cache()
    samples = np.stack(outputs)
    return (
        samples,
        samples.mean(axis=0, dtype=np.float32),
        samples.std(axis=0, dtype=np.float32),
        tuple(seeds),
        int(nfe or 0),
    )


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
    bounds = draw.textbbox((0, 0), text, font=font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text(
        (
            box[0] + (box[2] - box[0] - width) // 2,
            box[1] + (box[3] - box[1] - height) // 2,
        ),
        text,
        font=font,
        fill=fill,
    )


def _crop(values: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    top, left, bottom, right = box
    return np.asarray(values)[top:bottom, left:right]


def _gray_image(
    values: np.ndarray,
    window: tuple[float, float],
    support: np.ndarray,
    size: int,
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    low, high = window
    scaled = np.clip((np.asarray(values, dtype=np.float32) - low) / (high - low), 0, 1)
    scaled = np.where(np.asarray(support, dtype=bool), scaled, 0.0)
    if crop is not None:
        scaled = _crop(scaled, crop)
    image = Image.fromarray(np.rint(scaled * 255).astype(np.uint8), mode="L")
    return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def _heat_image(
    values: np.ndarray,
    high: float,
    support: np.ndarray,
    size: int,
    *,
    uncertainty: bool,
    crop: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    normalized = np.clip(np.asarray(values, dtype=np.float32) / max(high, 1e-8), 0, 1)
    if crop is not None:
        normalized = _crop(normalized, crop)
        support = _crop(support, crop)
    palette = np.asarray(
        (
            ((7, 15, 24), (21, 91, 121), (50, 181, 151), (245, 224, 112))
            if uncertainty
            else ((8, 8, 12), (75, 24, 95), (207, 55, 70), (255, 224, 92))
        ),
        dtype=np.float32,
    )
    position = normalized * (len(palette) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(palette) - 1)
    fraction = (position - lower)[..., None]
    rgb = palette[lower] * (1 - fraction) + palette[upper] * fraction
    rgb[~np.asarray(support, dtype=bool)] = 0
    image = Image.fromarray(np.rint(rgb).astype(np.uint8), mode="RGB")
    return image.resize((size, size), Image.Resampling.LANCZOS)


def _outline_mask(image: Image.Image, mask: np.ndarray, size: int) -> Image.Image:
    from scipy.ndimage import binary_erosion

    boundary = np.asarray(mask, dtype=bool) & ~binary_erosion(mask)
    resized = Image.fromarray(boundary.astype(np.uint8) * 255, mode="L").resize(
        (size, size), Image.Resampling.NEAREST
    )
    value = np.asarray(image).copy()
    edge = np.asarray(resized) > 0
    value[edge] = (40, 220, 205)
    return Image.fromarray(value, mode="RGB")


def _case_panels(
    case: CaseResult,
    *,
    modality_index: int,
    error_high: float,
    std_high: float,
    size: int,
    crop: tuple[int, int, int, int] | None,
) -> list[Image.Image]:
    z = case.display_z
    source_support = case.source.brain_support[modality_index, z]
    target_support = case.target.brain_support[modality_index, z]
    common_support = source_support | target_support
    window = case.windows[modality_index]
    source_panel = _gray_image(
        case.source.image[modality_index, z], window, source_support, size, crop=crop
    )
    source_mask = case.source.tumor_mask[z]
    if crop is not None:
        source_mask = _crop(source_mask, crop)
    source_panel = _outline_mask(source_panel, source_mask, size)
    return [
        source_panel,
        _gray_image(
            case.source.reconstruction[modality_index, z],
            window,
            source_support,
            size,
            crop=crop,
        ),
        _gray_image(
            case.target.image[modality_index, z], window, target_support, size, crop=crop
        ),
        _gray_image(
            case.target.reconstruction[modality_index, z],
            window,
            target_support,
            size,
            crop=crop,
        ),
        _gray_image(
            case.samples[0, modality_index, z],
            window,
            common_support,
            size,
            crop=crop,
        ),
        _gray_image(
            case.mean[modality_index, z], window, common_support, size, crop=crop
        ),
        _heat_image(
            np.abs(case.mean[modality_index, z] - case.target.image[modality_index, z]),
            error_high,
            common_support,
            size,
            uncertainty=False,
            crop=crop,
        ),
        _heat_image(
            case.std[modality_index, z],
            std_high,
            common_support,
            size,
            uncertainty=True,
            crop=crop,
        ),
    ]


def _draw_case(
    case: CaseResult,
    path: Path,
    *,
    model_label: str,
    error_high: float,
    std_high: float,
    cropped: bool,
) -> None:
    cell = 188 if cropped else 180
    gap = 8
    left = 130
    top = 155
    row_stride = cell + 40
    columns = (
        "Source MRI",
        "Source VQ",
        "Target MRI",
        "Target VQ",
        "Candidate 0",
        "Pred. mean",
        "|Mean-target|",
        "Uncertainty",
    )
    width = left + len(columns) * (cell + gap) + 25
    height = top + len(MODALITIES) * row_stride + 95
    canvas = Image.new("RGB", (width, height), (246, 247, 244))
    draw = ImageDraw.Draw(canvas)
    record = case.record
    title_suffix = "source-mask ROI" if cropped else "full axial slice"
    draw.text(
        (28, 20),
        f"MU-Glioma {model_label} forward prediction | {record['earlier_stage']} -> "
        f"{record['later_stage']} | {title_suffix}",
        font=_font(31, bold=True),
        fill=(22, 28, 34),
    )
    draw.text(
        (28, 66),
        f"{record['pair_id']} | {record['delta_days']} days | held-out patient | "
        f"axial z={case.display_z}",
        font=_font(18),
        fill=(65, 73, 80),
    )
    draw.text(
        (28, 99),
        "Cyan contour: source tumor mask used only to choose/display the slice; candidate 0 "
        "and cases were fixed before target inspection.",
        font=_font(16),
        fill=(65, 73, 80),
    )
    for column_index, label in enumerate(columns):
        x = left + column_index * (cell + gap)
        _centered_text(
            draw, (x, 124, x + cell, top), label, _font(14, bold=True), (34, 41, 47)
        )
    crop = case.crop_yxyx if cropped else None
    for modality_index, modality in enumerate(MODALITIES):
        y = top + modality_index * row_stride
        draw.text((25, y + cell // 2 - 15), modality.upper(), font=_font(22, bold=True), fill=(29, 37, 43))
        panels = _case_panels(
            case,
            modality_index=modality_index,
            error_high=error_high,
            std_high=std_high,
            size=cell,
            crop=crop,
        )
        for column_index, panel in enumerate(panels):
            x = left + column_index * (cell + gap)
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(205, 209, 207))
        modality_metrics = case.metrics["per_modality"][modality]
        draw.text(
            (left, y + cell + 7),
            f"MAE: copy={modality_metrics['copy_source_mae']:.3f} | "
            f"VQ floor={modality_metrics['target_vq_reconstruction_mae']:.3f} | "
            f"mean={modality_metrics['predictive_mean_mae']:.3f}",
            font=_font(14),
            fill=(76, 83, 88),
        )
    draw.text(
        (28, height - 62),
        f"Heat scales pooled over the three fixed cases: error 0..{error_high:.3f}; "
        f"uncertainty 0..{std_high:.3f}. Intensities are per-volume nonzero z-scores.",
        font=_font(16),
        fill=(65, 73, 80),
    )
    canvas.save(path, format="PNG", optimize=True)


def _draw_candidates(
    case: CaseResult,
    path: Path,
    *,
    std_high: float,
) -> None:
    cell = 178
    gap = 8
    left = 130
    top = 145
    row_stride = cell + 34
    columns = ("Source MRI", "Target MRI") + tuple(
        f"Candidate {index}" for index in range(case.samples.shape[0])
    ) + ("Pred. mean", "Uncertainty")
    width = left + len(columns) * (cell + gap) + 25
    height = top + len(MODALITIES) * row_stride + 70
    canvas = Image.new("RGB", (width, height), (246, 247, 244))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 20),
        f"Decoded candidate set | {case.record['pair_id']}",
        font=_font(31, bold=True),
        fill=(22, 28, 34),
    )
    draw.text(
        (28, 66),
        f"K={case.samples.shape[0]} | {case.nfe_per_sample} velocity evaluations per candidate | "
        "fixed seeds, no target-based candidate selection",
        font=_font(17),
        fill=(65, 73, 80),
    )
    for column_index, label in enumerate(columns):
        x = left + column_index * (cell + gap)
        _centered_text(
            draw, (x, 112, x + cell, top), label, _font(14, bold=True), (34, 41, 47)
        )
    z = case.display_z
    crop = case.crop_yxyx
    for modality_index, modality in enumerate(MODALITIES):
        y = top + modality_index * row_stride
        source_support = case.source.brain_support[modality_index, z]
        target_support = case.target.brain_support[modality_index, z]
        common_support = source_support | target_support
        window = case.windows[modality_index]
        draw.text((25, y + cell // 2 - 15), modality.upper(), font=_font(22, bold=True), fill=(29, 37, 43))
        panels = [
            _gray_image(
                case.source.image[modality_index, z], window, source_support, cell, crop=crop
            ),
            _gray_image(
                case.target.image[modality_index, z], window, target_support, cell, crop=crop
            ),
        ]
        panels.extend(
            _gray_image(sample[modality_index, z], window, common_support, cell, crop=crop)
            for sample in case.samples
        )
        panels.extend(
            (
                _gray_image(
                    case.mean[modality_index, z], window, common_support, cell, crop=crop
                ),
                _heat_image(
                    case.std[modality_index, z],
                    std_high,
                    common_support,
                    cell,
                    uncertainty=True,
                    crop=crop,
                ),
            )
        )
        for column_index, panel in enumerate(panels):
            x = left + column_index * (cell + gap)
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(205, 209, 207))
    canvas.save(path, format="PNG", optimize=True)


def _draw_summary(
    cases: Sequence[CaseResult],
    path: Path,
    *,
    model_label: str,
    error_high: float,
    std_high: float,
) -> None:
    cell = 155
    gap = 7
    left = 220
    top = 155
    row_stride = cell + 7
    case_gap = 48
    columns = (
        "Source MRI",
        "Target MRI",
        "Target VQ",
        "Candidate 0",
        "Pred. mean",
        "|Mean-target|",
        "Uncertainty",
    )
    content_height = len(cases) * (len(MODALITIES) * row_stride + case_gap)
    width = left + len(columns) * (cell + gap) + 25
    height = top + content_height + 78
    canvas = Image.new("RGB", (width, height), (246, 247, 244))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 20),
        f"MU-Glioma {model_label} | held-out forward predictions",
        font=_font(33, bold=True),
        fill=(22, 28, 34),
    )
    draw.text(
        (28, 70),
        "Three SHA-256-selected patients spanning 84-288 days | four jointly generated MRI modalities",
        font=_font(18),
        fill=(65, 73, 80),
    )
    draw.text(
        (28, 101),
        "Source-mask ROI crops; cyan source contour; no target-based case, slice, or candidate selection",
        font=_font(17),
        fill=(65, 73, 80),
    )
    for column_index, label in enumerate(columns):
        x = left + column_index * (cell + gap)
        _centered_text(
            draw, (x, 126, x + cell, top), label, _font(13, bold=True), (34, 41, 47)
        )
    cursor = top
    for case in cases:
        record = case.record
        draw.text(
            (22, cursor + 42),
            f"{record['earlier_stage']} -> {record['later_stage']}",
            font=_font(21, bold=True),
            fill=(29, 37, 43),
        )
        draw.text(
            (22, cursor + 76),
            f"{record['patient_id']}",
            font=_font(15),
            fill=(65, 73, 80),
        )
        draw.text(
            (22, cursor + 101),
            f"{record['delta_days']} days | z={case.display_z}",
            font=_font(15),
            fill=(65, 73, 80),
        )
        for modality_index, modality in enumerate(MODALITIES):
            y = cursor + modality_index * row_stride
            draw.text(
                (155, y + cell // 2 - 12),
                modality.upper(),
                font=_font(17, bold=True),
                fill=(42, 49, 55),
            )
            panels = _case_panels(
                case,
                modality_index=modality_index,
                error_high=error_high,
                std_high=std_high,
                size=cell,
                crop=case.crop_yxyx,
            )
            selected_panels = (
                panels[0],
                panels[2],
                panels[3],
                panels[4],
                panels[5],
                panels[6],
                panels[7],
            )
            for column_index, panel in enumerate(selected_panels):
                x = left + column_index * (cell + gap)
                canvas.paste(panel, (x, y))
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(205, 209, 207))
        macro = case.metrics
        footer_y = cursor + len(MODALITIES) * row_stride + 8
        draw.text(
            (left, footer_y),
            f"Macro MAE: copy {macro['macro_copy_source_mae']:.3f} | "
            f"VQ floor {macro['macro_target_vq_reconstruction_mae']:.3f} | "
            f"predictive mean {macro['macro_predictive_mean_mae']:.3f}",
            font=_font(15),
            fill=(65, 73, 80),
        )
        cursor += len(MODALITIES) * row_stride + case_gap
    draw.text(
        (28, height - 52),
        "Feasibility visualization only: visits share a standardized index grid but explicit longitudinal "
        "registration was not established.",
        font=_font(16),
        fill=(65, 73, 80),
    )
    canvas.save(path, format="PNG", optimize=True)


def _report(manifest: Mapping[str, Any]) -> str:
    lines = [
        f"# MU-Glioma {manifest['model_label']} four-modality qualitative report",
        "",
        "This is a held-out feasibility visualization, not a clinical result. Cases were",
        "selected before inference by a fixed SHA-256 rule in three predeclared interval",
        "strata. The source tumor mask selected the axial slice and ROI only; it was not a",
        "model input. The target did not select cases, slices, or candidates.",
        "",
        "![Summary](summary.png)",
        "",
        "## Cases",
        "",
        "| Pair | Days | Copy MAE | VQ floor MAE | Predictive mean MAE | Figure |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in manifest["cases"]:
        metrics = case["metrics"]
        lines.append(
            f"| `{case['pair_id']}` | {case['delta_days']} | "
            f"{metrics['macro_copy_source_mae']:.4f} | "
            f"{metrics['macro_target_vq_reconstruction_mae']:.4f} | "
            f"{metrics['macro_predictive_mean_mae']:.4f} | "
            f"[view]({case['case_directory']}/comparison_roi.png) |"
        )
    lines.extend(
        (
            "",
            "## Interpretation limits",
            "",
            "MAE is measured on per-volume nonzero-z-scored MRI. Source-copy and voxel-wise",
            "prediction comparisons are descriptive because explicit cross-visit physical",
            "registration was not established. Predictive uncertainty is the voxel-wise",
            f"standard deviation of K={manifest['sampling']['num_samples']} decoded candidates,",
            "not calibrated clinical uncertainty. Target VQ reconstruction shows the codec",
            "distortion floor separately from the SymmFlow prediction error.",
            "",
            "See [manifest.json](manifest.json) for hashes, seeds, parity checks, and per-modality metrics.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = _arguments()
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"visualization output already exists: {destination}")
    inputs = (
        args.config,
        args.checkpoint,
        args.vqgan_checkpoint,
        args.pair_manifest,
        args.source_manifest,
        args.raw_root,
        args.training_metrics,
    )
    for path in inputs:
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)

    config = load_config(args.config)
    pair_manifest = args.pair_manifest.resolve()
    records = read_jsonl(pair_manifest)
    statistics_path = pair_manifest.with_name("latent_statistics.json")
    statistics = _json(statistics_path)
    statistics_fingerprint, pair_fingerprint, pair_count = (
        validate_cached_pair_manifest_binding(records, statistics)
    )
    selected = _select_validation_pairs(records)
    validate_cached_latent_provenance(selected, statistics, validate_manifest=False)
    dataset = LatentPairDataset(selected)
    items = [dataset[index] for index in range(len(dataset))]
    source_manifest = args.source_manifest.resolve()
    if sha256_file(source_manifest) != statistics["source_manifest_sha256"]:
        raise ValueError("source manifest differs from the latent-cache contract")
    source_rows = _source_rows(source_manifest)

    if not torch.cuda.is_available():
        raise RuntimeError("full-volume MU-Glioma visualization requires CUDA")
    device = torch.device("cuda")
    print(
        f"Loading EMA {args.model_family} and frozen MU-Glioma VQ-GAN...",
        flush=True,
    )
    if args.model_family == "symmflow":
        codec, velocity, condition_encoder, header = _load_sampling_models(
            config,
            str(args.checkpoint.resolve()),
            str(args.vqgan_checkpoint.resolve()),
            device,
        )
        sampler: Any = SymmFlowSampler(codec, velocity, sigma_min=0.0)
        model_label = "SymmFlow"
        planned_steps = int(header["config"]["flow"]["max_steps"])
        validation_step_key = "optimizer_step"
    else:
        codec, velocity, condition_encoder, header, baseline_kind = (
            _load_baseline_sampling_models(
                str(args.checkpoint.resolve()),
                str(args.vqgan_checkpoint.resolve()),
                device,
            )
        )
        if baseline_kind != "unidirectional_cfm":
            raise ValueError(
                "--model-family=unidirectional_cfm requires a matching checkpoint"
            )
        baseline_settings = header["config"].get("baseline_training", {})
        cfm_base_distribution = str(
            baseline_settings.get("cfm_base_distribution", "standard_normal")
        )
        cfm_noise_scale = float(baseline_settings.get("cfm_noise_scale", 1.0))
        sampler = UnidirectionalCFMSampler(
            codec,
            velocity,
            condition_encoder,
            sigma_min=0.0,
            base_distribution=cfm_base_distribution,
            noise_scale=cfm_noise_scale,
        )
        model_label = (
            "source-bridge CFM"
            if cfm_base_distribution == "source_gaussian"
            else "source-conditioned CFM"
        )
        planned_steps = int(
            header["config"].get("baseline_training", {}).get(
                "max_steps", header["config"]["flow"]["max_steps"]
            )
        )
        validation_step_key = "step"
    if stable_hash(config) != stable_hash(header["config"]):
        raise ValueError("external config differs from the checkpoint training config")
    extra = header.get("extra")
    if not isinstance(extra, Mapping):
        raise ValueError("checkpoint has no training metadata")
    if args.model_family == "symmflow":
        if extra.get("branch_order") != EXPECTED_BRANCH_ORDER:
            raise ValueError("checkpoint branch order is not [later, earlier]")
    elif extra.get("direction") != "forward":
        raise ValueError("CFM checkpoint is not a forward model")
    checkpoint_step = int(header["step"])
    validation_history = extra.get("validation_history")
    if (
        checkpoint_step < 1
        or checkpoint_step > planned_steps
        or not isinstance(validation_history, list)
        or not validation_history
        or int(validation_history[-1].get(validation_step_key, -1)) != checkpoint_step
    ):
        raise ValueError(
            "qualitative inference requires a completed validation checkpoint"
        )
    if float(header["config"]["flow"].get("sigma_min", 0.0)) != 0.0:
        raise ValueError("qualitative inference requires clean sigma_min=0 endpoints")
    if stable_hash(header["latent_statistics"]) != stable_hash(statistics):
        raise ValueError("checkpoint and disk latent statistics differ")
    condition_schema_fingerprint = ConditionSchema.from_dict(
        header["feature_schema"]
    ).fingerprint
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    cases: list[CaseResult] = []
    case_records: list[dict[str, Any]] = []
    try:
        for case_index, (record, item) in enumerate(zip(selected, items, strict=True)):
            pair_id = str(record["pair_id"])
            print(f"[{case_index + 1}/{len(selected)}] {pair_id}: validating references", flush=True)
            source = _load_raw_visit(
                str(record["patient_id"]),
                str(record["earlier_visit_id"]),
                item["earlier_latent"],
                rows=source_rows,
                raw_root=args.raw_root.resolve(),
                codec=codec,
                device=device,
            )
            target = _load_raw_visit(
                str(record["patient_id"]),
                str(record["later_visit_id"]),
                item["later_latent"],
                rows=source_rows,
                raw_root=args.raw_root.resolve(),
                codec=codec,
                device=device,
            )
            with torch.inference_mode():
                tokens = condition_encoder(item["conditions"], batch_size=1)
            source_latent = item["earlier_latent"][None].to(device)
            case_seed = int(args.seed + case_index * 100)
            print(f"[{case_index + 1}/{len(selected)}] {pair_id}: forward sampling", flush=True)
            samples, mean, std, seeds, nfe = _sample_forward(
                sampler,
                source_latent,
                tokens,
                num_samples=args.num_samples,
                seed=case_seed,
                steps=args.steps,
                solver=args.solver,
            )
            expected_nfe = args.steps * (2 if args.solver == "heun" else 1)
            if nfe != expected_nfe:
                raise ValueError(f"unexpected ODE function-evaluation count: {nfe}")
            z = _display_slice(source.tumor_mask, source.brain_support[0])
            crop = _crop_from_mask(source.tumor_mask, source.brain_support[0], z)
            metrics = _case_metrics(source, target, samples, mean, std)
            case = CaseResult(
                record=record,
                source=source,
                target=target,
                samples=samples,
                mean=mean,
                std=std,
                display_z=z,
                crop_yxyx=crop,
                windows=_display_windows(source),
                metrics=metrics,
                seeds=seeds,
                nfe_per_sample=nfe,
            )
            cases.append(case)
            case_directory_name = pair_id
            case_directory = staging / case_directory_name
            case_directory.mkdir()
            np.savez_compressed(
                case_directory / "display_slices.npz",
                source_mri=source.image[:, z],
                source_vq=source.reconstruction[:, z],
                target_mri=target.image[:, z],
                target_vq=target.reconstruction[:, z],
                candidates=samples[:, :, z],
                predictive_mean=mean[:, z],
                predictive_std=std[:, z],
                source_tumor_mask=source.tumor_mask[z],
                source_brain_support=source.brain_support[:, z],
                target_brain_support=target.brain_support[:, z],
            )
            case_record = {
                "pair_id": pair_id,
                "patient_id": record["patient_id"],
                "split": "val",
                "earlier_stage": record["earlier_stage"],
                "later_stage": record["later_stage"],
                "earlier_visit_id": record["earlier_visit_id"],
                "later_visit_id": record["later_visit_id"],
                "delta_days": int(record["delta_days"]),
                "selection_digest": _selection_digest(pair_id),
                "selection_interval": [record["earlier_stage"], record["later_stage"]],
                "display_z": z,
                "display_slice_policy": "source_tumor_mask_max_axial_area",
                "crop_yxyx": list(crop),
                "crop_policy": "source_tumor_bbox_plus_24_voxels_minimum_96_square",
                "display_windows": [list(window) for window in case.windows],
                "display_window_policy": "source_brain_p0.5_p99.5_per_modality",
                "modalities": list(MODALITIES),
                "conditions": item["conditions"],
                "seeds": list(seeds),
                "nfe_per_sample": nfe,
                "metrics": metrics,
                "metric_scope": "target_nonzero_brain_per_volume_zscore_intensity",
                "source_image_sha256": source.image_sha256,
                "target_image_sha256": target.image_sha256,
                "source_mask_sha256": source.mask_sha256,
                "target_mask_sha256": target.mask_sha256,
                "source_latent_sha256": sha256_file(Path(record["earlier_latent_path"])),
                "target_latent_sha256": sha256_file(Path(record["later_latent_path"])),
                "source_latent_parity_max_abs": source.latent_parity_max_abs,
                "target_latent_parity_max_abs": target.latent_parity_max_abs,
                "case_directory": case_directory_name,
                "slice_archive": "display_slices.npz",
            }
            case_records.append(case_record)
            _write_json(case_directory / "case.json", case_record)
            del source_latent, tokens
            gc.collect()
            torch.cuda.empty_cache()

        error_values = []
        std_values = []
        for case in cases:
            for modality_index in range(len(MODALITIES)):
                z = case.display_z
                support = (
                    case.source.brain_support[modality_index, z]
                    | case.target.brain_support[modality_index, z]
                )
                error_values.append(
                    np.abs(
                        case.mean[modality_index, z]
                        - case.target.image[modality_index, z]
                    )[support]
                )
                std_values.append(case.std[modality_index, z][support])
        error_high = float(np.percentile(np.concatenate(error_values), 99.0))
        std_high = float(np.percentile(np.concatenate(std_values), 99.0))
        for case in cases:
            case_directory = staging / str(case.record["pair_id"])
            _draw_case(
                case,
                case_directory / "comparison_full.png",
                model_label=model_label,
                error_high=error_high,
                std_high=std_high,
                cropped=False,
            )
            _draw_case(
                case,
                case_directory / "comparison_roi.png",
                model_label=model_label,
                error_high=error_high,
                std_high=std_high,
                cropped=True,
            )
            _draw_candidates(case, case_directory / "candidates_roi.png", std_high=std_high)
        _draw_summary(
            cases,
            staging / "summary.png",
            model_label=model_label,
            error_high=error_high,
            std_high=std_high,
        )

        manifest = {
            "schema": "ispy2_symmflow_mu_glioma_qualitative_report_v2",
            "model_family": args.model_family,
            "model_label": model_label,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_namespace": SELECTION_NAMESPACE,
            "selection_policy": "minimum_sha256_per_predeclared_interval",
            "predeclared_intervals": [list(pair) for pair in DISPLAY_INTERVALS],
            "checkpoint_step": int(header["step"]),
            "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
            "vqgan_checkpoint_sha256": sha256_file(args.vqgan_checkpoint.resolve()),
            "pair_manifest_sha256": sha256_file(pair_manifest),
            "pair_manifest_fingerprint": pair_fingerprint,
            "pair_manifest_record_count": pair_count,
            "latent_statistics_fingerprint": statistics_fingerprint,
            "source_manifest_sha256": sha256_file(source_manifest),
            "training_metrics_sha256": sha256_file(args.training_metrics.resolve()),
            "condition_schema_fingerprint": condition_schema_fingerprint,
            "modality_order": list(MODALITIES),
            "sampling": {
                "direction": "forward",
                "num_samples": args.num_samples,
                "steps": args.steps,
                "solver": args.solver,
                "base_seed": args.seed,
                "nfe_per_sample": cases[0].nfe_per_sample,
                "cfm_base_distribution": getattr(
                    sampler, "base_distribution", None
                ),
                "cfm_noise_scale": getattr(sampler, "noise_scale", None),
            },
            "heat_scales": {
                "absolute_error_p99": error_high,
                "predictive_std_p99": std_high,
            },
            "longitudinal_registration_verified": False,
            "cases": case_records,
        }
        _write_json(staging / "manifest.json", manifest)
        (staging / "REPORT.md").write_text(_report(manifest), encoding="utf-8")
        shutil.move(str(staging), str(destination))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Wrote qualitative report to {destination}", flush=True)


if __name__ == "__main__":
    main()
