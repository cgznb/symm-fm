from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from skimage.metrics import structural_similarity

from .backend import load_transition_records
from .ispy2_biflow_config import load_ispy2_biflow_config
from .ispy2_biflow_latent_contract import decode_ispy2_biflow_continuous
from .ispy2_biflow_training import integrate_ispy2_biflow_euler
from .ispy2_biflow_visualization import (
    BiFlowVisualizationCase,
    _load_system,
    _mask_centroid,
    _maximum_axial_centroid,
    _safe_pair_id,
    render_case,
    render_overview,
)
from .ispy2_biflow_workflow import _world_data
from .ispy2_dce0_world_data import ISPY2DCE0WorldPair
from .latent_statistics import sha256_file
from .vqgan import REGISTERED_VQGAN_NUMERIC_CONTRACT
from .workflows import load_mri_vqgan


COHORT_SCHEMA = "mewm_ispy2_dce0_biflow_cohort_evaluation_v1"
LATENT_SCHEMA = "mewm_ispy2_dce0_biflow_cohort_latent_v1"
DECODED_SCHEMA = "mewm_ispy2_dce0_biflow_cohort_decoded_v1"
MRI_WINDOW = (-0.61669921875, 4.609375)
MRI_DATA_RANGE = MRI_WINDOW[1] - MRI_WINDOW[0]
SSIM_WIN_SIZE = 11
SSIM_SIGMA = 1.5
TUMOR_NEIGHBORHOOD_ITERATIONS = 5
COMPARISONS = (
    "biflow_prediction",
    "source_copy",
    "vq_target_reconstruction",
)
REGION_NAMES = (
    "common_foreground",
    "target_tumor",
    "tumor_union_neighborhood",
)
METRIC_NAMES = (
    "mae_unclipped",
    "rmse_unclipped",
    "psnr_windowed_db",
    "ssim_3d_windowed",
)
TRANSITION_ORDER = (
    "T0->T1",
    "T0->T2",
    "T0->T3",
    "T1->T2",
    "T1->T3",
    "T2->T3",
)
ADJACENT_TRANSITIONS = ("T0->T1", "T1->T2", "T2->T3")
ENDPOINT_FIELDS = (
    "patient_id",
    "transition_id",
    "transition_type",
    "delta_days",
    "seed",
    "comparison",
    "region",
    "voxel_count",
    "ssim_center_count",
    *METRIC_NAMES,
)
PATIENT_FIELDS = (
    "patient_id",
    "comparison",
    "region",
    "case_count",
    "voxel_count",
    "ssim_center_count",
    *METRIC_NAMES,
)
SUMMARY_FIELDS = (
    "scope",
    "transition_type",
    "comparison",
    "region",
    "metric",
    "count",
    "mean",
    "sample_sd",
    "sample_variance",
    "median",
    "q25",
    "q75",
    "minimum",
    "maximum",
)


def all_validation_pairs(
    pairs: Sequence[ISPY2DCE0WorldPair], *, limit: int | None = None
) -> tuple[ISPY2DCE0WorldPair, ...]:
    selected = tuple(
        sorted(
            (pair for pair in pairs if pair.split == "val"),
            key=lambda pair: (
                pair.patient_id,
                pair.source_stage,
                pair.target_stage,
                pair.pair_id,
            ),
        )
    )
    if not selected:
        raise ValueError("validation cohort is empty")
    if limit is not None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("cohort limit must be a positive integer")
        selected = selected[:limit]
    return selected


def select_validation_pairs(
    pairs: Sequence[ISPY2DCE0WorldPair],
    *,
    pair_mode: str,
    limit: int | None = None,
) -> tuple[ISPY2DCE0WorldPair, ...]:
    full = all_validation_pairs(pairs)
    if pair_mode == "all_connected":
        selected = full
    elif pair_mode == "adjacent":
        selected = tuple(
            pair for pair in full if pair.transition_type in ADJACENT_TRANSITIONS
        )
    else:
        raise ValueError("pair mode must be all_connected or adjacent")
    if limit is not None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("cohort limit must be a positive integer")
        selected = selected[:limit]
    if not selected:
        raise ValueError("selected validation cohort is empty")
    return selected


def _as_volume(value: torch.Tensor | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D volume")
    return array


def _window_to_unit(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    return (np.clip(array, *MRI_WINDOW) - MRI_WINDOW[0]) / MRI_DATA_RANGE


def region_masks(
    source_foreground: np.ndarray,
    target_foreground: np.ndarray,
    source_tumor: np.ndarray,
    target_tumor: np.ndarray,
) -> dict[str, np.ndarray]:
    common = np.asarray(source_foreground, dtype=bool) & np.asarray(
        target_foreground, dtype=bool
    )
    source = np.asarray(source_tumor, dtype=bool)
    target = np.asarray(target_tumor, dtype=bool)
    if any(array.shape != common.shape for array in (source, target)):
        raise ValueError("foreground and tumor masks must share one grid")
    neighborhood = ndimage.binary_dilation(
        source | target, iterations=TUMOR_NEIGHBORHOOD_ITERATIONS
    )
    return {
        "common_foreground": common,
        "target_tumor": common & target,
        "tumor_union_neighborhood": common & neighborhood,
    }


def valid_ssim_centers(common_foreground: np.ndarray) -> np.ndarray:
    return ndimage.binary_erosion(
        np.asarray(common_foreground, dtype=bool),
        structure=np.ones((SSIM_WIN_SIZE,) * 3, dtype=bool),
        border_value=0,
    )


def comparison_metrics(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    regions: Mapping[str, np.ndarray],
    centers_in_common_foreground: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    predicted = _as_volume(prediction, "prediction").astype(np.float32, copy=False)
    actual = _as_volume(target, "target").astype(np.float32, copy=False)
    if predicted.shape != actual.shape or not np.isfinite(predicted).all():
        raise ValueError("prediction and target must be finite volumes on one grid")
    if not np.isfinite(actual).all():
        raise ValueError("target contains non-finite values")

    predicted_windowed = _window_to_unit(predicted)
    actual_windowed = _window_to_unit(actual)
    _, ssim_map = structural_similarity(
        actual_windowed,
        predicted_windowed,
        data_range=1.0,
        gaussian_weights=True,
        sigma=SSIM_SIGMA,
        use_sample_covariance=False,
        win_size=SSIM_WIN_SIZE,
        full=True,
    )
    result: dict[str, dict[str, float | int]] = {}
    for region_name in REGION_NAMES:
        region = np.asarray(regions[region_name], dtype=bool)
        voxel_count = int(region.sum())
        centers = np.asarray(centers_in_common_foreground, dtype=bool) & region
        center_count = int(centers.sum())
        if voxel_count:
            raw_difference = predicted[region].astype(np.float64) - actual[
                region
            ].astype(np.float64)
            windowed_difference = predicted_windowed[region].astype(
                np.float64
            ) - actual_windowed[region].astype(np.float64)
            raw_mse = float(np.square(raw_difference).mean())
            windowed_mse = float(np.square(windowed_difference).mean())
            values = {
                "mae_unclipped": float(np.abs(raw_difference).mean()),
                "rmse_unclipped": float(math.sqrt(raw_mse)),
                "psnr_windowed_db": (
                    math.inf
                    if windowed_mse == 0.0
                    else float(10.0 * math.log10(1.0 / windowed_mse))
                ),
            }
        else:
            values = {name: math.nan for name in METRIC_NAMES[:-1]}
        result[region_name] = {
            "voxel_count": voxel_count,
            "ssim_center_count": center_count,
            **values,
            "ssim_3d_windowed": (
                float(np.asarray(ssim_map, dtype=np.float32)[centers].mean())
                if center_count
                else math.nan
            ),
        }
    return result


def describe_values(values: Iterable[Any]) -> dict[str, float | int | None]:
    finite = np.asarray([float(value) for value in values], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            "count": 0,
            "mean": None,
            "sample_sd": None,
            "sample_variance": None,
            "median": None,
            "q25": None,
            "q75": None,
            "minimum": None,
            "maximum": None,
        }
    variance = float(finite.var(ddof=1)) if finite.size > 1 else None
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "sample_sd": None if variance is None else float(math.sqrt(variance)),
        "sample_variance": variance,
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }


def _finite_mean(values: Iterable[Any]) -> float:
    finite = np.asarray([float(value) for value in values], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else math.nan


def aggregate_patient_rows(
    endpoint_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in endpoint_rows:
        grouped[
            (str(row["patient_id"]), str(row["comparison"]), str(row["region"]))
        ].append(row)
    result = []
    for (patient_id, comparison, region), rows in sorted(grouped.items()):
        result.append(
            {
                "patient_id": patient_id,
                "comparison": comparison,
                "region": region,
                "case_count": len(rows),
                "voxel_count": sum(int(row["voxel_count"]) for row in rows),
                "ssim_center_count": sum(
                    int(row["ssim_center_count"]) for row in rows
                ),
                **{
                    metric: _finite_mean(row[metric] for row in rows)
                    for metric in METRIC_NAMES
                },
            }
        )
    return result


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact is not a mapping: {path}")
    return payload


def _valid_tensor(value: Any, *, ndim: int) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.ndim == ndim
        and bool(torch.isfinite(value).all())
    )


def _validate_latent_artifact(
    path: Path, pair: ISPY2DCE0WorldPair, seed: int, solver_steps: int
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load_payload(path)
        return (
            payload.get("schema") == LATENT_SCHEMA
            and payload.get("pair_id") == pair.pair_id
            and payload.get("seed") == seed
            and payload.get("solver_steps") == solver_steps
            and _valid_tensor(payload.get("predicted_normalized_latent"), ndim=6)
            and _valid_tensor(payload.get("target_normalized_latent"), ndim=6)
        )
    except Exception:
        return False


def _validate_decoded_artifact(path: Path, pair: ISPY2DCE0WorldPair) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _load_payload(path)
        return (
            payload.get("schema") == DECODED_SCHEMA
            and payload.get("pair_id") == pair.pair_id
            and _valid_tensor(payload.get("prediction"), ndim=4)
            and _valid_tensor(payload.get("target_reconstruction"), ndim=4)
        )
    except Exception:
        return False


def _artifact_path(directory: Path, pair: ISPY2DCE0WorldPair) -> Path:
    return directory / pair.patient_id / f"{_safe_pair_id(pair.pair_id)}.pt"


def _generate_latents(
    *,
    config: Any,
    checkpoint: Path,
    pairs: Sequence[ISPY2DCE0WorldPair],
    latent_cache: Any,
    roi_cache: Any,
    latent_dir: Path,
    solver_steps: int,
    case_seeds: Mapping[str, int],
    device: torch.device,
) -> None:
    pending = [
        (index, pair)
        for index, pair in enumerate(pairs)
        if not _validate_latent_artifact(
            _artifact_path(latent_dir, pair),
            pair,
            case_seeds[pair.pair_id],
            solver_steps,
        )
    ]
    if not pending:
        print("all latent predictions already exist", flush=True)
        return
    system, _ = _load_system(config, checkpoint)
    system.to(device)
    model = system.model.eval()
    for completed, (index, pair) in enumerate(pending, start=1):
        case_seed = case_seeds[pair.pair_id]
        source_mri = roi_cache.load_source_mri(pair.source_visit).unsqueeze(0).to(device)
        target = latent_cache.load(pair.target_visit_id).float().unsqueeze(0).unsqueeze(0)
        generator = torch.Generator(device=device).manual_seed(case_seed)
        noise = torch.randn(
            target.shape, generator=generator, device=device, dtype=torch.float32
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
        _atomic_torch_save(
            _artifact_path(latent_dir, pair),
            {
                "schema": LATENT_SCHEMA,
                "pair_id": pair.pair_id,
                "patient_id": pair.patient_id,
                "seed": case_seed,
                "solver_steps": solver_steps,
                "predicted_normalized_latent": prediction.cpu().half(),
                "target_normalized_latent": target.half(),
            },
        )
        print(
            f"generated {completed}/{len(pending)} pending "
            f"({index + 1}/{len(pairs)} cohort): {pair.pair_id}",
            flush=True,
        )
        del source_mri, target, noise, prediction
    del model, system
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _decode_predictions(
    *,
    config: Any,
    pairs: Sequence[ISPY2DCE0WorldPair],
    latent_cache: Any,
    latent_dir: Path,
    decoded_dir: Path,
    device: torch.device,
) -> None:
    pending = [
        pair
        for pair in pairs
        if not _validate_decoded_artifact(_artifact_path(decoded_dir, pair), pair)
    ]
    if not pending:
        print("all decoded predictions already exist", flush=True)
        return
    vqgan = (
        load_mri_vqgan(
            config.base.data.vqgan_checkpoint,
            expected_numeric_contract=REGISTERED_VQGAN_NUMERIC_CONTRACT,
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    for completed, pair in enumerate(pending, start=1):
        latent_payload = _load_payload(_artifact_path(latent_dir, pair))
        images = []
        for key in ("target_normalized_latent", "predicted_normalized_latent"):
            normalized = latent_payload[key].float().to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                decoded = decode_ispy2_biflow_continuous(
                    vqgan, latent_cache.denormalize(normalized)
                )
            images.append(decoded[0, 0].cpu().half())
            del normalized, decoded
        target_reconstruction, prediction = images
        _atomic_torch_save(
            _artifact_path(decoded_dir, pair),
            {
                "schema": DECODED_SCHEMA,
                "pair_id": pair.pair_id,
                "patient_id": pair.patient_id,
                "prediction": prediction,
                "target_reconstruction": target_reconstruction,
            },
        )
        print(
            f"decoded {completed}/{len(pending)} pending: {pair.pair_id}", flush=True
        )
        del latent_payload, images, target_reconstruction, prediction
    del vqgan
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _case_from_decoded(
    *,
    pair: ISPY2DCE0WorldPair,
    seed: int,
    source: Any,
    target: Any,
    prediction: torch.Tensor,
    target_reconstruction: torch.Tensor,
    common: torch.Tensor,
    prediction_metrics: Mapping[str, Mapping[str, Any]],
) -> BiFlowVisualizationCase:
    return BiFlowVisualizationCase(
        pair=pair,
        seed=seed,
        source=source.image,
        target=target.image,
        target_reconstruction=target_reconstruction,
        prediction=prediction,
        source_mask=source.mask,
        target_mask=target.mask,
        common_foreground=common,
        source_max_zyx=_maximum_axial_centroid(source.mask),
        target_max_zyx=_maximum_axial_centroid(target.mask),
        target_centroid_zyx=_mask_centroid(target.mask),
        metrics={
            "prediction_vs_target_mae": prediction_metrics["common_foreground"][
                "mae_unclipped"
            ],
            "prediction_vs_target_tumor_mae": prediction_metrics["target_tumor"][
                "mae_unclipped"
            ],
        },
    )


def _flush_patient_overview(
    patient_cases: list[BiFlowVisualizationCase], patient_dir: Path
) -> None:
    if not patient_cases:
        return
    output = patient_dir / f"{patient_cases[0].pair.patient_id}-overview.png"
    if not output.is_file():
        render_overview(
            patient_cases,
            output,
            image_window=MRI_WINDOW,
            error_max=MRI_DATA_RANGE,
        )


def _evaluate_and_render(
    *,
    pairs: Sequence[ISPY2DCE0WorldPair],
    roi_cache: Any,
    visits: Mapping[str, Any],
    decoded_dir: Path,
    output: Path,
    case_seeds: Mapping[str, int],
    render_visuals: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint_rows: list[dict[str, Any]] = []
    gallery_rows: list[dict[str, Any]] = []
    patient_cases: list[BiFlowVisualizationCase] = []
    current_patient: str | None = None
    case_dir = output / "cases"
    patient_dir = output / "patients"
    patient_dir.mkdir(parents=True, exist_ok=True)
    for index, pair in enumerate(pairs):
        if current_patient is not None and pair.patient_id != current_patient:
            if render_visuals:
                _flush_patient_overview(patient_cases, patient_dir)
            patient_cases = []
        current_patient = pair.patient_id
        source = roi_cache.load(pair.source_visit)
        target = roi_cache.load(visits[pair.target_visit_id])
        decoded = _load_payload(_artifact_path(decoded_dir, pair))
        prediction = decoded["prediction"].float()
        target_reconstruction = decoded["target_reconstruction"].float()
        regions = region_masks(
            _as_volume(source.valid_foreground, "source foreground"),
            _as_volume(target.valid_foreground, "target foreground"),
            _as_volume(source.mask, "source tumor"),
            _as_volume(target.mask, "target tumor"),
        )
        if not regions["common_foreground"].any():
            raise ValueError(f"common foreground is empty: {pair.pair_id}")
        centers = valid_ssim_centers(regions["common_foreground"])
        comparisons = {
            "biflow_prediction": prediction,
            "source_copy": source.image,
            "vq_target_reconstruction": target_reconstruction,
        }
        case_metrics = {}
        for comparison, predicted in comparisons.items():
            metrics = comparison_metrics(predicted, target.image, regions, centers)
            case_metrics[comparison] = metrics
            for region_name in REGION_NAMES:
                endpoint_rows.append(
                    {
                        "patient_id": pair.patient_id,
                        "transition_id": pair.pair_id,
                        "transition_type": pair.transition_type,
                        "delta_days": pair.delta_days,
                        "seed": case_seeds[pair.pair_id],
                        "comparison": comparison,
                        "region": region_name,
                        **metrics[region_name],
                    }
                )
        case = _case_from_decoded(
            pair=pair,
            seed=case_seeds[pair.pair_id],
            source=source,
            target=target,
            prediction=prediction,
            target_reconstruction=target_reconstruction,
            common=source.valid_foreground.bool() & target.valid_foreground.bool(),
            prediction_metrics=case_metrics["biflow_prediction"],
        )
        patient_cases.append(case)
        case_path = (
            case_dir
            / pair.patient_id
            / f"{_safe_pair_id(pair.pair_id)}-five-view.png"
        )
        case_path.parent.mkdir(parents=True, exist_ok=True)
        if render_visuals and not case_path.is_file():
            render_case(
                case,
                case_path,
                image_window=MRI_WINDOW,
                error_max=MRI_DATA_RANGE,
            )
        gallery_rows.append(
            {
                "patient_id": pair.patient_id,
                "transition_id": pair.pair_id,
                "transition_type": pair.transition_type,
                "delta_days": pair.delta_days,
                "seed": case_seeds[pair.pair_id],
                "five_view_png": str(case_path.relative_to(output)),
                "patient_overview_png": str(
                    (patient_dir / f"{pair.patient_id}-overview.png").relative_to(
                        output
                    )
                ),
                "latent_artifact": str(
                    _artifact_path(output / "latents", pair).relative_to(output)
                ),
                "decoded_artifact": str(
                    _artifact_path(decoded_dir, pair).relative_to(output)
                ),
            }
        )
        print(f"evaluated {index + 1}/{len(pairs)}: {pair.pair_id}", flush=True)
        del decoded, prediction, target_reconstruction, comparisons, case_metrics
        if (index + 1) % 10 == 0:
            gc.collect()
    if render_visuals:
        _flush_patient_overview(patient_cases, patient_dir)
    return endpoint_rows, gallery_rows


def _summary_rows(
    endpoint_rows: Sequence[Mapping[str, Any]],
    patient_rows: Sequence[Mapping[str, Any]],
    *,
    transitions: Sequence[str] = TRANSITION_ORDER,
) -> list[dict[str, Any]]:
    output = []
    scopes: list[tuple[str, str, Sequence[Mapping[str, Any]]]] = [
        ("endpoint_macro", "", endpoint_rows),
        ("patient_macro", "", patient_rows),
    ]
    scopes.extend(
        (
            "transition_endpoint_macro",
            transition,
            [row for row in endpoint_rows if row["transition_type"] == transition],
        )
        for transition in transitions
    )
    for scope, transition, rows in scopes:
        for comparison in COMPARISONS:
            for region in REGION_NAMES:
                group = [
                    row
                    for row in rows
                    if row["comparison"] == comparison and row["region"] == region
                ]
                for metric in METRIC_NAMES:
                    output.append(
                        {
                            "scope": scope,
                            "transition_type": transition,
                            "comparison": comparison,
                            "region": region,
                            "metric": metric,
                            **describe_values(row[metric] for row in group),
                        }
                    )
    return output


def _nested_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        scope = str(row["scope"])
        transition = str(row["transition_type"] or "all")
        comparison = str(row["comparison"])
        region = str(row["region"])
        metric = str(row["metric"])
        values = {key: row[key] for key in SUMMARY_FIELDS[5:]}
        result.setdefault(scope, {}).setdefault(transition, {}).setdefault(
            comparison, {}
        ).setdefault(region, {})[metric] = values
    return result


def _metric_distribution_plot(
    endpoint_rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    transitions: Sequence[str] = TRANSITION_ORDER,
    title: str = "BiFlowNet full validation cohort | common-foreground endpoint metrics",
) -> None:
    labels = {
        "mae_unclipped": "MAE",
        "rmse_unclipped": "RMSE",
        "psnr_windowed_db": "PSNR (dB)",
        "ssim_3d_windowed": "3D SSIM",
    }
    selected = [
        row
        for row in endpoint_rows
        if row["comparison"] == "biflow_prediction"
        and row["region"] == "common_foreground"
    ]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    rng = np.random.default_rng(2026)
    for axis, metric in zip(axes.flat, METRIC_NAMES, strict=True):
        groups = []
        for transition in transitions:
            values = np.asarray(
                [
                    float(row[metric])
                    for row in selected
                    if row["transition_type"] == transition
                    and math.isfinite(float(row[metric]))
                ],
                dtype=np.float64,
            )
            groups.append(values)
        axis.boxplot(groups, showfliers=False, widths=0.55)
        for position, values in enumerate(groups, start=1):
            if values.size:
                jitter = rng.uniform(-0.16, 0.16, values.size)
                axis.scatter(
                    position + jitter,
                    values,
                    s=8,
                    alpha=0.28,
                    color="#2a6f97",
                    edgecolors="none",
                )
                axis.scatter(
                    [position], [values.mean()], marker="D", s=35, color="#c23b22"
                )
        axis.set_xticks(range(1, len(transitions) + 1), transitions)
        axis.set_ylabel(labels[metric])
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def export_adjacent_metric_subset(
    *, source_dir: str | Path, output_dir: str | Path
) -> Path:
    started = time.perf_counter()
    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not (source / "summary.json").is_file():
        raise FileNotFoundError("source cohort summary is missing")
    if output.exists():
        raise FileExistsError(f"adjacent-only output already exists: {output}")
    output.mkdir(parents=True)

    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    endpoint_rows = [
        row
        for row in _read_csv(source / "endpoint_metrics.csv")
        if row["transition_type"] in ADJACENT_TRANSITIONS
    ]
    gallery_rows = [
        row
        for row in _read_csv(source / "gallery_index.csv")
        if row["transition_type"] in ADJACENT_TRANSITIONS
    ]
    pair_ids = {row["transition_id"] for row in gallery_rows}
    if len(gallery_rows) != 278 or len(pair_ids) != 278:
        raise RuntimeError("adjacent validation gallery does not contain 278 pairs")
    if len(endpoint_rows) != 278 * len(COMPARISONS) * len(REGION_NAMES):
        raise RuntimeError("adjacent endpoint metric row count is incomplete")

    patient_rows = aggregate_patient_rows(endpoint_rows)
    summary_rows = _summary_rows(
        endpoint_rows, patient_rows, transitions=ADJACENT_TRANSITIONS
    )
    relative_gallery_rows = []
    for row in gallery_rows:
        updated = dict(row)
        for field in ("five_view_png", "latent_artifact", "decoded_artifact"):
            updated[field] = os.path.relpath(source / row[field], output)
        updated.pop("patient_overview_png", None)
        relative_gallery_rows.append(updated)

    _write_csv(output / "endpoint_metrics.csv", endpoint_rows, ENDPOINT_FIELDS)
    _write_csv(output / "patient_metrics.csv", patient_rows, PATIENT_FIELDS)
    _write_csv(output / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(
        output / "gallery_index.csv",
        relative_gallery_rows,
        (
            "patient_id",
            "transition_id",
            "transition_type",
            "delta_days",
            "seed",
            "five_view_png",
            "latent_artifact",
            "decoded_artifact",
        ),
    )
    _metric_distribution_plot(
        endpoint_rows,
        output / "metrics-by-transition.png",
        transitions=ADJACENT_TRANSITIONS,
        title=(
            "BiFlowNet adjacent-transition validation cohort | "
            "common-foreground endpoint metrics"
        ),
    )
    _comparison_plot(
        patient_rows,
        output / "metrics-by-comparison.png",
        title="Adjacent-transition validation cohort | patient-macro comparison",
    )

    transition_counts = {
        transition: sum(row["transition_type"] == transition for row in gallery_rows)
        for transition in ADJACENT_TRANSITIONS
    }
    manifest = {
        "schema": "mewm_ispy2_dce0_biflow_adjacent_metric_subset_v1",
        "source_cohort": str(source),
        "source_summary_sha256": sha256_file(source / "summary.json"),
        "config": source_summary["config"],
        "config_sha256": source_summary["config_sha256"],
        "checkpoint": source_summary["checkpoint"],
        "checkpoint_sha256": source_summary["checkpoint_sha256"],
        "checkpoint_epoch": source_summary["checkpoint_epoch"],
        "checkpoint_global_step": source_summary["checkpoint_global_step"],
        "solver_steps": source_summary["solver_steps"],
        "base_seed": source_summary["base_seed"],
        "cohort": {
            "patients": len({row["patient_id"] for row in gallery_rows}),
            "endpoints": len(gallery_rows),
            "selection": "adjacent validation transitions only",
            "included_transitions": list(ADJACENT_TRANSITIONS),
            "excluded_transitions": [
                transition
                for transition in TRANSITION_ORDER
                if transition not in ADJACENT_TRANSITIONS
            ],
            "transition_counts": transition_counts,
        },
        "metric_contract": {
            **source_summary["metric_contract"],
            "aggregation": (
                "adjacent-only endpoint macro, patient macro, and transition "
                "endpoint macro; mean, sample SD/variance, median, quartiles, "
                "minimum, maximum"
            ),
            "confidence_intervals": "not computed by user request",
        },
        "visualization": {
            "five_view_case_count": len(gallery_rows),
            "storage": "reuses exact per-case PNGs and tensors from source_cohort",
        },
        "files": {
            "endpoint_metrics": "endpoint_metrics.csv",
            "patient_metrics": "patient_metrics.csv",
            "summary_metrics": "summary_metrics.csv",
            "gallery_index": "gallery_index.csv",
            "metrics_by_transition": "metrics-by-transition.png",
            "metrics_by_comparison": "metrics-by-comparison.png",
        },
        "summary": _nested_summary(summary_rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(output / "summary.json", manifest)
    print(f"saved adjacent-only metric subset: {output}", flush=True)
    return output


def _comparison_plot(
    patient_rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    title: str = "Full validation cohort | patient-macro comparison",
) -> None:
    labels = {
        "mae_unclipped": "MAE",
        "rmse_unclipped": "RMSE",
        "psnr_windowed_db": "PSNR (dB)",
        "ssim_3d_windowed": "3D SSIM",
    }
    comparison_labels = {
        "biflow_prediction": "BiFlow",
        "source_copy": "Source copy",
        "vq_target_reconstruction": "VQ target recon",
    }
    colors = ("#2a6f97", "#6c757d", "#c23b22")
    x = np.arange(len(REGION_NAMES), dtype=np.float64)
    width = 0.24
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRIC_NAMES, strict=True):
        for offset_index, comparison in enumerate(COMPARISONS):
            means = []
            deviations = []
            for region in REGION_NAMES:
                values = [
                    row[metric]
                    for row in patient_rows
                    if row["comparison"] == comparison and row["region"] == region
                ]
                description = describe_values(values)
                means.append(description["mean"])
                deviations.append(description["sample_sd"] or 0.0)
            axis.bar(
                x + (offset_index - 1) * width,
                means,
                width,
                yerr=deviations,
                capsize=2,
                color=colors[offset_index],
                label=comparison_labels[comparison],
                alpha=0.9,
            )
        axis.set_xticks(
            x,
            ("Common foreground", "Target tumor", "Tumor neighborhood"),
        )
        axis.set_ylabel(f"{labels[metric]} (mean +/- patient SD)")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def evaluate_ispy2_biflow_cohort(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    solver_steps: int | None = None,
    seed: int | None = None,
    device: str = "cuda",
    limit: int | None = None,
    render_visuals: bool = True,
    pair_mode: str = "all_connected",
) -> Path:
    started = time.perf_counter()
    config = load_ispy2_biflow_config(config_path)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("BiFlowNet checkpoint is missing")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    steps = config.training.evaluation_solver_steps if solver_steps is None else solver_steps
    if type(steps) is not int or steps <= 0:
        raise ValueError("Euler solver steps must be positive")
    base_seed = config.runtime.seed if seed is None else seed
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("cohort seed must be a nonnegative integer")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    all_pairs, _, latent_cache, roi_cache = _world_data(config)
    full_pairs = all_validation_pairs(all_pairs)
    pairs = select_validation_pairs(all_pairs, pair_mode=pair_mode, limit=limit)
    full_pair_positions = {pair.pair_id: index for index, pair in enumerate(full_pairs)}
    case_seeds = {
        pair.pair_id: base_seed + full_pair_positions[pair.pair_id] for pair in pairs
    }
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    checkpoint_metadata = {
        "epoch": int(checkpoint_payload["epoch"]),
        "global_step": int(checkpoint_payload["global_step"]),
    }
    del checkpoint_payload
    run_identity = {
        "schema": COHORT_SCHEMA,
        "config": str(config.path),
        "config_sha256": config.sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": checkpoint_metadata["epoch"],
        "checkpoint_global_step": checkpoint_metadata["global_step"],
        "vqgan_checkpoint": str(config.base.data.vqgan_checkpoint),
        "vqgan_sha256": config.base.data.vqgan_sha256,
        "solver_steps": steps,
        "base_seed": base_seed,
        "seed_schedule": "base seed plus index in full all-connected cohort",
        "pair_mode": pair_mode,
        "pair_ids": [pair.pair_id for pair in pairs],
        "case_seeds": [case_seeds[pair.pair_id] for pair in pairs],
    }
    identity_path = output / "run_identity.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != run_identity:
            raise ValueError("existing output uses a different cohort run identity")
    else:
        _atomic_json(identity_path, run_identity)

    latent_dir = output / "latents"
    decoded_dir = output / "decoded"
    _generate_latents(
        config=config,
        checkpoint=checkpoint,
        pairs=pairs,
        latent_cache=latent_cache,
        roi_cache=roi_cache,
        latent_dir=latent_dir,
        solver_steps=steps,
        case_seeds=case_seeds,
        device=target_device,
    )
    _decode_predictions(
        config=config,
        pairs=pairs,
        latent_cache=latent_cache,
        latent_dir=latent_dir,
        decoded_dir=decoded_dir,
        device=target_device,
    )
    loaded = load_transition_records(
        config.base.data.bundle_json,
        config.base.data.phase_manifest_csv,
        backend=config.base.data.backend,
    )
    endpoint_rows, gallery_rows = _evaluate_and_render(
        pairs=pairs,
        roi_cache=roi_cache,
        visits=loaded.visits,
        decoded_dir=decoded_dir,
        output=output,
        case_seeds=case_seeds,
        render_visuals=render_visuals,
    )
    patient_rows = aggregate_patient_rows(endpoint_rows)
    selected_transitions = (
        ADJACENT_TRANSITIONS if pair_mode == "adjacent" else TRANSITION_ORDER
    )
    summary_rows = _summary_rows(
        endpoint_rows, patient_rows, transitions=selected_transitions
    )
    _write_csv(output / "endpoint_metrics.csv", endpoint_rows, ENDPOINT_FIELDS)
    _write_csv(output / "patient_metrics.csv", patient_rows, PATIENT_FIELDS)
    _write_csv(output / "summary_metrics.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(
        output / "gallery_index.csv",
        gallery_rows,
        (
            "patient_id",
            "transition_id",
            "transition_type",
            "delta_days",
            "seed",
            "five_view_png",
            "patient_overview_png",
            "latent_artifact",
            "decoded_artifact",
        ),
    )
    if render_visuals:
        _metric_distribution_plot(
            endpoint_rows,
            output / "metrics-by-transition.png",
            transitions=selected_transitions,
            title=(
                "BiFlowNet adjacent-transition validation cohort | "
                "common-foreground endpoint metrics"
                if pair_mode == "adjacent"
                else "BiFlowNet full validation cohort | common-foreground endpoint metrics"
            ),
        )
        _comparison_plot(patient_rows, output / "metrics-by-comparison.png")

    expected_endpoint_rows = len(pairs) * len(COMPARISONS) * len(REGION_NAMES)
    expected_patient_rows = (
        len({pair.patient_id for pair in pairs})
        * len(COMPARISONS)
        * len(REGION_NAMES)
    )
    if len(endpoint_rows) != expected_endpoint_rows:
        raise RuntimeError("endpoint metric row count is incomplete")
    if len(patient_rows) != expected_patient_rows:
        raise RuntimeError("patient metric row count is incomplete")
    transition_counts = {
        transition: sum(pair.transition_type == transition for pair in pairs)
        for transition in selected_transitions
    }
    manifest = {
        **run_identity,
        "cohort": {
            "patients": len({pair.patient_id for pair in pairs}),
            "endpoints": len(pairs),
            "selection": (
                "adjacent validation transitions sorted by patient and stage"
                if pair_mode == "adjacent"
                else "all validation transitions sorted by patient and stage"
            ),
            "transition_counts": transition_counts,
        },
        "metric_contract": {
            "decoded_input": "registered normalized DCE0; unclipped for MAE/RMSE",
            "fixed_window": list(MRI_WINDOW),
            "window_use": "PSNR and SSIM only",
            "comparisons": list(COMPARISONS),
            "regions": {
                "common_foreground": "source/target valid_foreground intersection",
                "target_tumor": "target tumor intersected with common foreground",
                "tumor_union_neighborhood": (
                    "source/target tumor union dilated 5 voxels and intersected "
                    "with common foreground"
                ),
            },
            "ssim": {
                "implementation": "skimage structural_similarity",
                "data_range": 1.0,
                "gaussian_weights": True,
                "sigma": SSIM_SIGMA,
                "win_size": SSIM_WIN_SIZE,
                "use_sample_covariance": False,
            },
            "aggregation": (
                "endpoint macro, patient macro, and transition endpoint macro; "
                "mean, sample SD/variance, median, quartiles, minimum, maximum"
            ),
            "confidence_intervals": "not computed by user request",
        },
        "visualization": {
            "five_view_case_count": len(pairs) if render_visuals else 0,
            "patient_overview_count": (
                len({pair.patient_id for pair in pairs}) if render_visuals else 0
            ),
            "image_window": list(MRI_WINDOW),
            "absolute_error_vmax": MRI_DATA_RANGE,
            "outside_fov_rendering": "masked dark gray on common valid foreground",
            "contours": {"source_tumor": "cyan", "target_tumor": "yellow"},
        },
        "files": {
            "endpoint_metrics": "endpoint_metrics.csv",
            "patient_metrics": "patient_metrics.csv",
            "summary_metrics": "summary_metrics.csv",
            "gallery_index": "gallery_index.csv",
            "metrics_by_transition": (
                "metrics-by-transition.png" if render_visuals else None
            ),
            "metrics_by_comparison": (
                "metrics-by-comparison.png" if render_visuals else None
            ),
        },
        "summary": _nested_summary(summary_rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(output / "summary.json", manifest)
    print(f"saved full cohort evaluation: {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize every I-SPY2 BiFlowNet validation pair"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--solver-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-visuals", action="store_true")
    parser.add_argument(
        "--pair-mode",
        choices=("all_connected", "adjacent"),
        default="all_connected",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate_ispy2_biflow_cohort(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        solver_steps=args.solver_steps,
        seed=args.seed,
        device=args.device,
        limit=args.limit,
        render_visuals=not args.no_visuals,
        pair_mode=args.pair_mode,
    )
    return 0


__all__ = [
    "aggregate_patient_rows",
    "all_validation_pairs",
    "comparison_metrics",
    "describe_values",
    "evaluate_ispy2_biflow_cohort",
    "export_adjacent_metric_subset",
    "main",
    "region_masks",
    "select_validation_pairs",
    "valid_ssim_centers",
]
