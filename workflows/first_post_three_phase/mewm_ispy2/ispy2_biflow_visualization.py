from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .backend import load_transition_records
from .cache import RegisteredStrictAROICache
from .ispy2_biflow_config import ISPY2BiFlowConfig, load_ispy2_biflow_config
from .ispy2_biflow_latent_contract import decode_ispy2_biflow_continuous
from .ispy2_biflow_training import (
    ISPY2_BIFLOW_CHECKPOINT_SCHEMA,
    ISPY2BiFlowTrainingSystem,
    integrate_ispy2_biflow_euler,
)
from .ispy2_biflow_workflow import _build_model, _identity, _world_data
from .ispy2_dce0_world_data import ISPY2DCE0WorldPair
from .latent_statistics import sha256_file
from .vqgan import REGISTERED_VQGAN_NUMERIC_CONTRACT
from .workflows import load_mri_vqgan


VISUALIZATION_SCHEMA = "mewm_ispy2_dce0_biflow_visualization_v2"
TRANSITION_ORDER = (
    "T0->T1",
    "T0->T2",
    "T0->T3",
    "T1->T2",
    "T2->T3",
    "T1->T3",
)
PLANE_NAMES = ("axial", "coronal", "sagittal")
METRIC_FIELDS = (
    "pair_id",
    "patient_id",
    "transition_type",
    "delta_days",
    "seed",
    "common_foreground_voxels",
    "target_tumor_voxels",
    "prediction_vs_target_mae",
    "prediction_vs_target_rmse",
    "prediction_vs_target_psnr_windowed_db",
    "prediction_vs_target_tumor_mae",
    "prediction_vs_vq_target_mae",
    "vq_target_vs_target_mae",
)


@dataclass(frozen=True)
class BiFlowVisualizationCase:
    pair: ISPY2DCE0WorldPair
    seed: int
    source: torch.Tensor
    target: torch.Tensor
    target_reconstruction: torch.Tensor
    prediction: torch.Tensor
    source_mask: torch.Tensor
    target_mask: torch.Tensor
    common_foreground: torch.Tensor
    source_max_zyx: tuple[int, int, int]
    target_max_zyx: tuple[int, int, int]
    target_centroid_zyx: tuple[int, int, int]
    metrics: dict[str, float | int | str]


def select_visualization_pairs(
    pairs: Sequence[ISPY2DCE0WorldPair], count: int
) -> tuple[ISPY2DCE0WorldPair, ...]:
    if type(count) is not int or count <= 0:
        raise ValueError("visualization case count must be positive")
    candidates = sorted(
        (pair for pair in pairs if pair.split == "val"),
        key=lambda pair: (pair.patient_id, pair.source_stage, pair.target_stage),
    )
    if count > len({pair.patient_id for pair in candidates}):
        raise ValueError("not enough distinct validation patients")
    chosen: list[ISPY2DCE0WorldPair] = []
    used_patients: set[str] = set()
    while len(chosen) < count:
        made_progress = False
        for transition in TRANSITION_ORDER:
            pair = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.transition_type == transition
                    and candidate.patient_id not in used_patients
                    and candidate not in chosen
                ),
                None,
            )
            if pair is None:
                continue
            chosen.append(pair)
            used_patients.add(pair.patient_id)
            made_progress = True
            if len(chosen) == count:
                break
        if not made_progress:
            raise ValueError("unable to select distinct validation patients")
    return tuple(chosen)


def _mask_centroid(mask: torch.Tensor) -> tuple[int, int, int]:
    coordinates = mask[0].bool().nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        return tuple(int(value // 2) for value in mask.shape[1:])
    center = coordinates.float().mean(dim=0).round().long()
    return tuple(int(value) for value in center)


def _maximum_axial_centroid(mask: torch.Tensor) -> tuple[int, int, int]:
    foreground = mask[0].bool()
    areas = foreground.sum(dim=(1, 2))
    if int(areas.max()) == 0:
        return _mask_centroid(mask)
    z_index = int(areas.argmax())
    center = foreground[z_index].nonzero(as_tuple=False).float().mean(dim=0).round()
    return z_index, int(center[0]), int(center[1])


def _plane(
    value: torch.Tensor, center: tuple[int, int, int], plane: str
) -> np.ndarray:
    volume = value[0].float().numpy()
    if plane == "axial":
        return volume[center[0]]
    if plane == "coronal":
        return volume[:, center[1], :]
    if plane == "sagittal":
        return volume[:, :, center[2]]
    raise ValueError(f"unsupported plane: {plane}")


def _sampled_quantile(
    arrays: Sequence[np.ndarray], quantile: float, maximum_total: int = 2_000_000
) -> float:
    if not arrays:
        raise ValueError("quantile arrays are empty")
    maximum_per_array = max(1, maximum_total // len(arrays))
    samples = []
    for array in arrays:
        flattened = np.asarray(array, dtype=np.float32).reshape(-1)
        if flattened.size > maximum_per_array:
            stride = math.ceil(flattened.size / maximum_per_array)
            flattened = flattened[::stride][:maximum_per_array]
        samples.append(flattened)
    return float(np.quantile(np.concatenate(samples), quantile))


def _region_mae(
    prediction: torch.Tensor, target: torch.Tensor, region: torch.Tensor
) -> float:
    selected = region.bool()
    if not bool(selected.any()):
        return math.nan
    return float((prediction.float() - target.float()).abs()[selected].mean())


def _case_metrics(
    *,
    pair: ISPY2DCE0WorldPair,
    seed: int,
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_reconstruction: torch.Tensor,
    common: torch.Tensor,
    target_mask: torch.Tensor,
    image_window: tuple[float, float],
) -> dict[str, float | int | str]:
    selected = common.bool()
    if not bool(selected.any()):
        raise ValueError("case has no common valid foreground")
    difference = prediction.float()[selected] - target.float()[selected]
    low, high = image_window
    prediction_windowed = prediction.float().clamp(low, high)
    target_windowed = target.float().clamp(low, high)
    windowed_mse = float(
        ((prediction_windowed[selected] - target_windowed[selected]) / (high - low))
        .square()
        .mean()
    )
    tumor = selected & target_mask.bool()
    return {
        "pair_id": pair.pair_id,
        "patient_id": pair.patient_id,
        "transition_type": pair.transition_type,
        "delta_days": pair.delta_days,
        "seed": seed,
        "common_foreground_voxels": int(selected.sum()),
        "target_tumor_voxels": int(tumor.sum()),
        "prediction_vs_target_mae": float(difference.abs().mean()),
        "prediction_vs_target_rmse": float(difference.square().mean().sqrt()),
        "prediction_vs_target_psnr_windowed_db": (
            math.inf if windowed_mse == 0.0 else 10.0 * math.log10(1.0 / windowed_mse)
        ),
        "prediction_vs_target_tumor_mae": _region_mae(
            prediction, target, tumor
        ),
        "prediction_vs_vq_target_mae": _region_mae(
            prediction, target_reconstruction, selected
        ),
        "vq_target_vs_target_mae": _region_mae(
            target_reconstruction, target, selected
        ),
    }


def _load_system(
    config: ISPY2BiFlowConfig,
    checkpoint_path: Path,
) -> tuple[ISPY2BiFlowTrainingSystem, dict[str, Any]]:
    _, _, latent_cache, _ = _world_data(config)
    model = _build_model(config)
    identity = _identity(config, model=model, latent_cache=latent_cache)
    system = ISPY2BiFlowTrainingSystem(
        model, config=config, checkpoint_identity=identity
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if checkpoint.get("ispy2_biflow_schema") != ISPY2_BIFLOW_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not an I-SPY2 BiFlowNet schema-v3 checkpoint")
    system.on_load_checkpoint(checkpoint)
    incompatible = system.load_state_dict(checkpoint["state_dict"], strict=False)
    expected_missing = set(checkpoint["ispy2_biflow_omitted_state_keys"])
    actual_missing = set(incompatible.missing_keys)
    if not actual_missing <= expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "checkpoint parameters did not restore exactly: "
            f"expected_missing={len(expected_missing)}, "
            f"actual_missing={len(actual_missing)}, "
            f"unexpected={len(incompatible.unexpected_keys)}, "
            f"missing_only={sorted(actual_missing - expected_missing)[:8]}, "
            f"expected_only={sorted(expected_missing - actual_missing)[:8]}, "
            f"unexpected_examples={incompatible.unexpected_keys[:8]}"
        )
    del latent_cache
    return system.eval(), checkpoint


def _generate_latents(
    *,
    config: ISPY2BiFlowConfig,
    checkpoint_path: Path,
    pairs: Sequence[ISPY2DCE0WorldPair],
    roi_cache: RegisteredStrictAROICache,
    latent_cache: Any,
    solver_steps: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system, checkpoint = _load_system(config, checkpoint_path)
    system.to(device)
    model = system.model.eval()
    generated: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        case_seed = seed + index
        source_mri = roi_cache.load_source_mri(pair.source_visit).unsqueeze(0).to(device)
        target = latent_cache.load(pair.target_visit_id).float().unsqueeze(0).unsqueeze(0)
        generator = torch.Generator(device=device).manual_seed(case_seed)
        noise = torch.randn(
            target.shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = integrate_ispy2_biflow_euler(
                model,
                source_mri=source_mri,
                clinical_text=[pair.clinical_text],
                treatment_text=[pair.treatment_text],
                delta_days=torch.tensor([pair.delta_days], device=device),
                target_stage=torch.tensor([pair.target_stage], device=device),
                solver_steps=solver_steps,
                noise=noise,
            )
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError(f"non-finite generated latent: {pair.pair_id}")
        generated.append(
            {
                "pair": pair,
                "seed": case_seed,
                "prediction": prediction.cpu().float(),
                "target": target.float(),
            }
        )
        print(f"generated {index + 1}/{len(pairs)}: {pair.pair_id}", flush=True)
    metadata = {
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
    }
    del checkpoint, model, system
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return generated, metadata


def _decode_cases(
    *,
    config: ISPY2BiFlowConfig,
    generated: Sequence[dict[str, Any]],
    latent_cache: Any,
    roi_cache: RegisteredStrictAROICache,
    visits: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], tuple[float, float]]:
    vqgan = load_mri_vqgan(
        config.base.data.vqgan_checkpoint,
        expected_numeric_contract=REGISTERED_VQGAN_NUMERIC_CONTRACT,
    ).to(device).eval().requires_grad_(False)
    decoded: list[dict[str, Any]] = []
    factual_samples: list[np.ndarray] = []
    for index, raw in enumerate(generated):
        pair = raw["pair"]
        source = roi_cache.load(pair.source_visit)
        target = roi_cache.load(visits[pair.target_visit_id])
        images = []
        for latent in (raw["target"], raw["prediction"]):
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                image = decode_ispy2_biflow_continuous(
                    vqgan,
                    latent_cache.denormalize(latent.to(device)),
                )
            images.append(image[0, 0].cpu().float())
        target_reconstruction, prediction = images
        common = source.valid_foreground.bool() & target.valid_foreground.bool()
        factual_samples.extend(
            (source.image[common].numpy(), target.image[common].numpy())
        )
        decoded.append(
            {
                **raw,
                "source": source,
                "target_visit": target,
                "target_reconstruction": target_reconstruction,
                "prediction_image": prediction,
                "common": common,
            }
        )
        print(f"decoded {index + 1}/{len(generated)}: {pair.pair_id}", flush=True)
    image_window = (
        _sampled_quantile(factual_samples, 0.005),
        _sampled_quantile(factual_samples, 0.995),
    )
    if not image_window[1] > image_window[0]:
        raise ValueError("display window is degenerate")
    del vqgan
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return decoded, image_window


def _finalize_cases(
    decoded: Sequence[dict[str, Any]], image_window: tuple[float, float]
) -> tuple[BiFlowVisualizationCase, ...]:
    result = []
    for raw in decoded:
        pair = raw["pair"]
        source = raw["source"]
        target = raw["target_visit"]
        metrics = _case_metrics(
            pair=pair,
            seed=raw["seed"],
            prediction=raw["prediction_image"],
            target=target.image,
            target_reconstruction=raw["target_reconstruction"],
            common=raw["common"],
            target_mask=target.mask,
            image_window=image_window,
        )
        result.append(
            BiFlowVisualizationCase(
                pair=pair,
                seed=raw["seed"],
                source=source.image,
                target=target.image,
                target_reconstruction=raw["target_reconstruction"],
                prediction=raw["prediction_image"],
                source_mask=source.mask,
                target_mask=target.mask,
                common_foreground=raw["common"],
                source_max_zyx=_maximum_axial_centroid(source.mask),
                target_max_zyx=_maximum_axial_centroid(target.mask),
                target_centroid_zyx=_mask_centroid(target.mask),
                metrics=metrics,
            )
        )
    return tuple(result)


def _draw_view(
    axes: Sequence[Any],
    case: BiFlowVisualizationCase,
    *,
    center: tuple[int, int, int],
    plane: str,
    row_label: str,
    image_window: tuple[float, float],
    error_max: float,
    show_titles: bool,
    show_contours: bool = True,
) -> None:
    volumes = (
        case.source,
        case.target,
        case.target_reconstruction,
        case.prediction,
        (case.prediction - case.target).abs(),
    )
    titles = (
        "Source DCE0",
        "Target DCE0",
        "Target VQ recon",
        "BiFlow prediction",
        "Absolute error",
    )
    valid = _plane(case.common_foreground.float(), center, plane) > 0.5
    source_mask = (
        _plane(case.source_mask.float(), center, plane) > 0.5
        if show_contours
        else None
    )
    target_mask = (
        _plane(case.target_mask.float(), center, plane) > 0.5
        if show_contours
        else None
    )
    grayscale = plt.get_cmap("gray").copy()
    grayscale.set_bad("#303030")
    error_map = plt.get_cmap("magma").copy()
    error_map.set_bad("#303030")
    for index, (axis, volume) in enumerate(zip(axes, volumes, strict=True)):
        values = np.ma.masked_where(~valid, _plane(volume, center, plane))
        if index < 4:
            axis.imshow(
                values,
                cmap=grayscale,
                vmin=image_window[0],
                vmax=image_window[1],
                origin="lower",
            )
            outline = source_mask if index == 0 else target_mask
            if outline is not None and np.any(outline & valid):
                axis.contour(
                    outline & valid,
                    levels=[0.5],
                    colors=["#00e5ff" if index == 0 else "#ffd54f"],
                    linewidths=0.8,
                )
        else:
            axis.imshow(
                values,
                cmap=error_map,
                vmin=0.0,
                vmax=error_max,
                origin="lower",
            )
            if target_mask is not None and np.any(target_mask & valid):
                axis.contour(
                    target_mask & valid,
                    levels=[0.5],
                    colors=["#ffd54f"],
                    linewidths=0.8,
                )
        if show_titles:
            axis.set_title(titles[index], fontsize=10)
        axis.axis("off")
    axes[0].text(
        -0.08,
        0.5,
        row_label,
        rotation=90,
        va="center",
        ha="center",
        transform=axes[0].transAxes,
        fontsize=9,
    )


def _save_figure(figure: plt.Figure, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        figure.savefig(temporary, format="png", dpi=150)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


def render_case(
    case: BiFlowVisualizationCase,
    output: Path,
    *,
    image_window: tuple[float, float],
    error_max: float,
    show_contours: bool = True,
) -> Path:
    views = (
        (case.source_max_zyx, "axial", "Source-max axial"),
        (case.target_max_zyx, "axial", "Target-max axial"),
        (case.target_centroid_zyx, "axial", "Centroid axial"),
        (case.target_centroid_zyx, "coronal", "Centroid coronal"),
        (case.target_centroid_zyx, "sagittal", "Centroid sagittal"),
    )
    figure, axes = plt.subplots(
        len(views), 5, figsize=(16, 15), constrained_layout=True, squeeze=False
    )
    for row, (center, plane, label) in enumerate(views):
        _draw_view(
            axes[row],
            case,
            center=center,
            plane=plane,
            row_label=label,
            image_window=image_window,
            error_max=error_max,
            show_titles=row == 0,
            show_contours=show_contours,
        )
    figure.suptitle(
        f"{case.pair.pair_id} | days={case.pair.delta_days} | "
        f"MAE={case.metrics['prediction_vs_target_mae']:.4f} | "
        f"tumor MAE={case.metrics['prediction_vs_target_tumor_mae']:.4f}",
        fontsize=13,
    )
    _save_figure(figure, output)
    return output


def render_overview(
    cases: Sequence[BiFlowVisualizationCase],
    output: Path,
    *,
    image_window: tuple[float, float],
    error_max: float,
    show_contours: bool = True,
) -> Path:
    figure, axes = plt.subplots(
        len(cases), 5, figsize=(16, 3.1 * len(cases)), constrained_layout=True, squeeze=False
    )
    for row, case in enumerate(cases):
        _draw_view(
            axes[row],
            case,
            center=case.target_max_zyx,
            plane="axial",
            row_label=f"{case.pair.patient_id}\n{case.pair.transition_type}",
            image_window=image_window,
            error_max=error_max,
            show_titles=row == 0,
            show_contours=show_contours,
        )
    figure.suptitle("I-SPY2 BiFlowNet validation samples | target-max axial slices", fontsize=14)
    _save_figure(figure, output)
    return output


def _safe_pair_id(pair_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", pair_id)


def visualize_ispy2_biflow(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    case_count: int = 5,
    solver_steps: int | None = None,
    seed: int | None = None,
    device: str = "cuda",
) -> Path:
    config = load_ispy2_biflow_config(config_path)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("BiFlowNet checkpoint is missing")
    steps = config.training.evaluation_solver_steps if solver_steps is None else solver_steps
    if type(steps) is not int or steps <= 0:
        raise ValueError("Euler solver steps must be positive")
    sample_seed = config.runtime.seed if seed is None else seed
    if type(sample_seed) is not int or sample_seed < 0:
        raise ValueError("visualization seed must be a nonnegative integer")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else checkpoint.parent.parent
        / "visualizations"
        / f"{checkpoint.stem}-{case_count}cases-seed{sample_seed}"
    )
    output.mkdir(parents=True, exist_ok=True)

    all_pairs, _, latent_cache, roi_cache = _world_data(config)
    selected = select_visualization_pairs(all_pairs, case_count)
    loaded = load_transition_records(
        config.base.data.bundle_json,
        config.base.data.phase_manifest_csv,
        backend=config.base.data.backend,
    )
    generated, checkpoint_metadata = _generate_latents(
        config=config,
        checkpoint_path=checkpoint,
        pairs=selected,
        roi_cache=roi_cache,
        latent_cache=latent_cache,
        solver_steps=steps,
        seed=sample_seed,
        device=target_device,
    )
    decoded, image_window = _decode_cases(
        config=config,
        generated=generated,
        latent_cache=latent_cache,
        roi_cache=roi_cache,
        visits=loaded.visits,
        device=target_device,
    )
    cases = _finalize_cases(decoded, image_window)
    error_samples = [
        (case.prediction - case.target).abs()[case.common_foreground.bool()].numpy()
        for case in cases
    ]
    error_max = max(_sampled_quantile(error_samples, 0.99), 1e-6)

    files = []
    overview = output / "overview-target-max-axial.png"
    render_overview(
        cases, overview, image_window=image_window, error_max=error_max
    )
    files.append(overview.name)
    latent_dir = output / "latents"
    latent_dir.mkdir(exist_ok=True)
    for case, raw in zip(cases, generated, strict=True):
        stem = _safe_pair_id(case.pair.pair_id)
        detail = output / f"{stem}-five-view.png"
        render_case(case, detail, image_window=image_window, error_max=error_max)
        files.append(detail.name)
        torch.save(
            {
                "schema": VISUALIZATION_SCHEMA,
                "pair_id": case.pair.pair_id,
                "seed": case.seed,
                "solver_steps": steps,
                "predicted_normalized_latent": raw["prediction"].half(),
                "target_normalized_latent": raw["target"].half(),
            },
            latent_dir / f"{stem}.pt",
        )

    metrics_path = output / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(case.metrics for case in cases)
    manifest = {
        "schema": VISUALIZATION_SCHEMA,
        "config": str(config.path),
        "config_sha256": config.sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": checkpoint_metadata["epoch"],
        "checkpoint_global_step": checkpoint_metadata["global_step"],
        "vqgan_checkpoint": str(config.base.data.vqgan_checkpoint),
        "vqgan_sha256": config.base.data.vqgan_sha256,
        "solver_steps": steps,
        "base_seed": sample_seed,
        "case_count": len(cases),
        "selection": "transition-stratified distinct validation patients",
        "display_window": list(image_window),
        "absolute_error_vmax": error_max,
        "outside_fov_rendering": "masked dark gray on common valid foreground",
        "contours": {"source_tumor": "cyan", "target_tumor": "yellow"},
        "files": files,
        "metrics_csv": metrics_path.name,
        "cases": [
            {
                "pair_id": case.pair.pair_id,
                "patient_id": case.pair.patient_id,
                "transition_type": case.pair.transition_type,
                "source_visit_id": case.pair.source_visit_id,
                "target_visit_id": case.pair.target_visit_id,
                "delta_days": case.pair.delta_days,
                "seed": case.seed,
                "source_max_zyx": list(case.source_max_zyx),
                "target_max_zyx": list(case.target_max_zyx),
                "target_centroid_zyx": list(case.target_centroid_zyx),
                "metrics": case.metrics,
            }
            for case in cases
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved visualization gallery: {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render I-SPY2 BiFlowNet validation predictions"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--solver-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    visualize_ispy2_biflow(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        case_count=args.cases,
        solver_steps=args.solver_steps,
        seed=args.seed,
        device=args.device,
    )
    return 0


__all__ = [
    "BiFlowVisualizationCase",
    "main",
    "render_case",
    "render_overview",
    "select_visualization_pairs",
    "visualize_ispy2_biflow",
]
