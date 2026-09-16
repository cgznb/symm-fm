#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mewm_ispy2.backend import load_transition_records
from mewm_ispy2.ispy2_biflow_cohort_evaluation import (
    ADJACENT_TRANSITIONS,
    METRIC_NAMES,
    MRI_DATA_RANGE,
    MRI_WINDOW,
    describe_values,
)
from mewm_ispy2.ispy2_biflow_config import load_ispy2_biflow_config
from mewm_ispy2.ispy2_biflow_visualization import (
    _mask_centroid,
    _maximum_axial_centroid,
    _plane,
)
from mewm_ispy2.ispy2_biflow_workflow import _world_data


STEP_ORDER = (2, 8, 20)
REGIONS = ("common_foreground", "target_tumor")
REGION_LABELS = {
    "common_foreground": "Common foreground",
    "target_tumor": "Target tumor",
}
METRIC_LABELS = {
    "mae_unclipped": "MAE",
    "rmse_unclipped": "RMSE",
    "psnr_windowed_db": "PSNR (dB)",
    "ssim_3d_windowed": "3D SSIM",
}
EXPECTED_ENDPOINT_ROWS = 278 * 3 * 3
EXPECTED_PATIENT_ROWS = 102 * 3 * 3


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_evaluation(path: Path, expected_steps: int) -> dict[str, Any]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    endpoint = _read_csv(path / "endpoint_metrics.csv")
    patient = _read_csv(path / "patient_metrics.csv")
    gallery = _read_csv(path / "gallery_index.csv")
    identity_path = path / "run_identity.json"
    raw_identity = (
        json.loads(identity_path.read_text(encoding="utf-8"))
        if identity_path.is_file()
        else summary
    )
    identity = {
        key: raw_identity[key]
        for key in (
            "config_sha256",
            "checkpoint_sha256",
            "checkpoint_epoch",
            "checkpoint_global_step",
            "solver_steps",
            "base_seed",
        )
    }
    identity.update(
        {
            "pair_mode": "adjacent",
            "pair_ids": [row["transition_id"] for row in gallery],
            "case_seeds": [int(row["seed"]) for row in gallery],
        }
    )
    if int(identity["solver_steps"]) != expected_steps:
        raise ValueError(f"expected Euler {expected_steps}: {path}")
    if raw_identity.get("pair_mode", "adjacent") != "adjacent":
        raise ValueError(f"evaluation is not adjacent-only: {path}")
    if len(endpoint) != EXPECTED_ENDPOINT_ROWS or len(patient) != EXPECTED_PATIENT_ROWS:
        raise ValueError(f"incomplete metric tables: {path}")
    if len(gallery) != 278 or len({row["transition_id"] for row in gallery}) != 278:
        raise ValueError(f"incomplete gallery index: {path}")
    if summary["cohort"]["transition_counts"] != {
        "T0->T1": 100,
        "T1->T2": 92,
        "T2->T3": 86,
    }:
        raise ValueError(f"unexpected transition counts: {path}")
    return {
        "path": path,
        "summary": summary,
        "identity": identity,
        "endpoint": endpoint,
        "patient": patient,
        "gallery": gallery,
    }


def _validate_matched(evaluations: Mapping[int, Mapping[str, Any]]) -> None:
    reference = evaluations[STEP_ORDER[0]]["identity"]
    stable_fields = (
        "config_sha256",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "checkpoint_global_step",
        "base_seed",
        "pair_mode",
        "pair_ids",
        "case_seeds",
    )
    for steps in STEP_ORDER[1:]:
        candidate = evaluations[steps]["identity"]
        for field in stable_fields:
            if candidate[field] != reference[field]:
                raise ValueError(f"Euler {steps} differs in identity field {field}")

    reference_static = {
        (row["transition_id"], row["comparison"], row["region"]): tuple(
            row[field] for field in ("voxel_count", "ssim_center_count", *METRIC_NAMES)
        )
        for row in evaluations[STEP_ORDER[0]]["endpoint"]
        if row["comparison"] != "biflow_prediction"
    }
    for steps in STEP_ORDER[1:]:
        candidate_static = {
            (row["transition_id"], row["comparison"], row["region"]): tuple(
                row[field]
                for field in ("voxel_count", "ssim_center_count", *METRIC_NAMES)
            )
            for row in evaluations[steps]["endpoint"]
            if row["comparison"] != "biflow_prediction"
        }
        if candidate_static != reference_static:
            raise ValueError(f"static baseline metrics differ for Euler {steps}")


def _summary_record(
    *,
    steps: int | str,
    scope: str,
    transition: str,
    region: str,
    metric: str,
    values: Iterable[Any],
) -> dict[str, Any]:
    return {
        "solver_steps": steps,
        "scope": scope,
        "transition_type": transition,
        "comparison": "biflow_prediction" if steps != "source_copy" else "source_copy",
        "region": region,
        "metric": metric,
        **describe_values(values),
    }


def _metric_summaries(
    evaluations: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    patient_rows = []
    transition_rows = []
    for steps in STEP_ORDER:
        patient = evaluations[steps]["patient"]
        endpoint = evaluations[steps]["endpoint"]
        for region in REGIONS:
            for metric in METRIC_NAMES:
                patient_rows.append(
                    _summary_record(
                        steps=steps,
                        scope="patient_macro",
                        transition="",
                        region=region,
                        metric=metric,
                        values=(
                            row[metric]
                            for row in patient
                            if row["comparison"] == "biflow_prediction"
                            and row["region"] == region
                        ),
                    )
                )
                for transition in ADJACENT_TRANSITIONS:
                    transition_rows.append(
                        _summary_record(
                            steps=steps,
                            scope="transition_endpoint_macro",
                            transition=transition,
                            region=region,
                            metric=metric,
                            values=(
                                row[metric]
                                for row in endpoint
                                if row["comparison"] == "biflow_prediction"
                                and row["region"] == region
                                and row["transition_type"] == transition
                            ),
                        )
                    )

    reference_patient = evaluations[STEP_ORDER[0]]["patient"]
    for region in REGIONS:
        for metric in METRIC_NAMES:
            patient_rows.append(
                _summary_record(
                    steps="source_copy",
                    scope="patient_macro",
                    transition="",
                    region=region,
                    metric=metric,
                    values=(
                        row[metric]
                        for row in reference_patient
                        if row["comparison"] == "source_copy"
                        and row["region"] == region
                    ),
                )
            )
    return patient_rows, transition_rows


def _plot_patient_metrics(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    lookup = {
        (str(row["solver_steps"]), row["region"], row["metric"]): row for row in rows
    }
    figure, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    x = np.arange(len(STEP_ORDER), dtype=np.float64)
    for row_index, region in enumerate(REGIONS):
        for column, metric in enumerate(METRIC_NAMES):
            axis = axes[row_index, column]
            means = np.asarray(
                [lookup[(str(step), region, metric)]["mean"] for step in STEP_ORDER],
                dtype=np.float64,
            )
            deviations = np.asarray(
                [lookup[(str(step), region, metric)]["sample_sd"] for step in STEP_ORDER],
                dtype=np.float64,
            )
            baseline = float(lookup[("source_copy", region, metric)]["mean"])
            axis.errorbar(
                x,
                means,
                yerr=deviations,
                marker="o",
                capsize=3,
                linewidth=1.8,
                color="#2166ac",
                label="BiFlow mean +/- patient SD",
            )
            axis.axhline(
                baseline,
                color="#b2182b",
                linestyle="--",
                linewidth=1.4,
                label="Source-copy mean",
            )
            for position, value in zip(x, means, strict=True):
                y_span = max(float(np.ptp(means)), float(deviations.max()), 1e-6)
                label_offset = (
                    (-14 if value < baseline else 7)
                    if abs(value - baseline) < 0.12 * y_span
                    else 7
                )
                axis.annotate(
                    f"{value:.3f}",
                    (position, value),
                    xytext=(0, label_offset),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            axis.set_xticks(x, [f"Euler {step}" for step in STEP_ORDER])
            axis.set_title(METRIC_LABELS[metric])
            if column == 0:
                axis.set_ylabel(REGION_LABELS[region])
            axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("BiFlowNet best-083 | adjacent validation | 102 patients / 278 pairs")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _prediction_rows(evaluation: Mapping[str, Any], region: str) -> dict[str, dict[str, str]]:
    return {
        row["transition_id"]: row
        for row in evaluation["endpoint"]
        if row["comparison"] == "biflow_prediction" and row["region"] == region
    }


def _select_cases(
    evaluations: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    e2 = _prediction_rows(evaluations[2], "common_foreground")
    e8 = _prediction_rows(evaluations[8], "common_foreground")
    e20 = _prediction_rows(evaluations[20], "common_foreground")
    selected = []
    for transition in ADJACENT_TRANSITIONS:
        candidates = []
        for pair_id, row2 in e2.items():
            if row2["transition_type"] != transition:
                continue
            difference = float(e20[pair_id]["mae_unclipped"]) - float(
                row2["mae_unclipped"]
            )
            candidates.append((difference, pair_id))
        candidates.sort()
        positions = {
            "best": 0,
            "median": len(candidates) // 2,
            "worst": len(candidates) - 1,
        }
        for rank, position in positions.items():
            difference, pair_id = candidates[position]
            selected.append(
                {
                    "transition_type": transition,
                    "selection": rank,
                    "transition_id": pair_id,
                    "patient_id": e2[pair_id]["patient_id"],
                    "seed": int(e2[pair_id]["seed"]),
                    "euler2_mae": float(e2[pair_id]["mae_unclipped"]),
                    "euler8_mae": float(e8[pair_id]["mae_unclipped"]),
                    "euler20_mae": float(e20[pair_id]["mae_unclipped"]),
                    "euler20_minus_euler2_mae": difference,
                }
            )
    return selected


def _gallery_paths(evaluation: Mapping[str, Any]) -> dict[str, Path]:
    root = Path(evaluation["path"])
    return {
        row["transition_id"]: (root / row["decoded_artifact"]).resolve()
        for row in evaluation["gallery"]
    }


def _draw_volume(
    axis: Any,
    volume: torch.Tensor,
    valid: torch.Tensor,
    mask: torch.Tensor,
    center: tuple[int, int, int],
    plane: str,
    *,
    error: bool,
) -> None:
    valid_plane = _plane(valid.float(), center, plane) > 0.5
    values = np.ma.masked_where(~valid_plane, _plane(volume.float(), center, plane))
    cmap = plt.get_cmap("magma" if error else "gray").copy()
    cmap.set_bad("#303030")
    axis.imshow(
        values,
        cmap=cmap,
        vmin=0.0 if error else MRI_WINDOW[0],
        vmax=MRI_DATA_RANGE if error else MRI_WINDOW[1],
        origin="lower",
    )
    mask_plane = _plane(mask.float(), center, plane) > 0.5
    if np.any(mask_plane & valid_plane):
        axis.contour(
            mask_plane & valid_plane,
            levels=[0.5],
            colors=["#ffd54f"],
            linewidths=0.8,
        )
    axis.axis("off")


def _render_case_comparison(
    *,
    record: Mapping[str, Any],
    pair: Any,
    source: Any,
    target: Any,
    predictions: Mapping[int, torch.Tensor],
    output: Path,
) -> None:
    common = source.valid_foreground.bool() & target.valid_foreground.bool()
    target_max = _maximum_axial_centroid(target.mask)
    centroid = _mask_centroid(target.mask)
    views = (
        (target_max, "axial", "Target-max axial"),
        (centroid, "coronal", "Centroid coronal"),
        (centroid, "sagittal", "Centroid sagittal"),
    )
    titles = (
        "Source",
        "Target",
        f"Euler 2\nMAE {record['euler2_mae']:.3f}",
        f"Euler 8\nMAE {record['euler8_mae']:.3f}",
        f"Euler 20\nMAE {record['euler20_mae']:.3f}",
        "|E2 - target|",
        "|E8 - target|",
        "|E20 - target|",
    )
    volumes = (
        source.image,
        target.image,
        predictions[2],
        predictions[8],
        predictions[20],
        (predictions[2] - target.image).abs(),
        (predictions[8] - target.image).abs(),
        (predictions[20] - target.image).abs(),
    )
    figure, axes = plt.subplots(3, 8, figsize=(24, 9), constrained_layout=True)
    for row, (center, plane, label) in enumerate(views):
        for column, (axis, volume) in enumerate(zip(axes[row], volumes, strict=True)):
            _draw_volume(
                axis,
                volume,
                common,
                target.mask,
                center,
                plane,
                error=column >= 5,
            )
            if row == 0:
                axis.set_title(titles[column], fontsize=10)
        axes[row, 0].text(
            -0.08,
            0.5,
            label,
            rotation=90,
            va="center",
            ha="center",
            transform=axes[row, 0].transAxes,
            fontsize=9,
        )
    figure.suptitle(
        f"{pair.pair_id} | {record['selection']} E20-vs-E2 common-foreground MAE "
        f"| delta={record['euler20_minus_euler2_mae']:+.4f} | seed={record['seed']}",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _render_selected_cases(
    *,
    config_path: Path,
    evaluations: Mapping[int, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    config = load_ispy2_biflow_config(config_path)
    all_pairs, _, _, roi_cache = _world_data(config)
    pairs = {pair.pair_id: pair for pair in all_pairs if pair.split == "val"}
    visits = load_transition_records(
        config.base.data.bundle_json,
        config.base.data.phase_manifest_csv,
        backend=config.base.data.backend,
    ).visits
    paths = {steps: _gallery_paths(evaluations[steps]) for steps in STEP_ORDER}
    for record in records:
        pair_id = str(record["transition_id"])
        pair = pairs[pair_id]
        source = roi_cache.load(pair.source_visit)
        target = roi_cache.load(visits[pair.target_visit_id])
        predictions = {
            steps: torch.load(
                paths[steps][pair_id],
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )["prediction"].float()
            for steps in STEP_ORDER
        }
        filename = (
            f"{record['transition_type'].replace('->', '-to-')}-"
            f"{record['selection']}-{record['patient_id']}.png"
        )
        _render_case_comparison(
            record=record,
            pair=pair,
            source=source,
            target=target,
            predictions=predictions,
            output=output / "cases" / filename,
        )


def compare(
    *,
    config_path: Path,
    evaluation_dirs: Mapping[int, Path],
    output_dir: Path,
) -> Path:
    evaluations = {
        steps: _load_evaluation(evaluation_dirs[steps].resolve(), steps)
        for steps in STEP_ORDER
    }
    _validate_matched(evaluations)
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_rows, transition_rows = _metric_summaries(evaluations)
    selected = _select_cases(evaluations)
    _write_csv(output_dir / "patient-macro-metrics.csv", patient_rows)
    _write_csv(output_dir / "transition-endpoint-macro-metrics.csv", transition_rows)
    _write_csv(output_dir / "selected-cases.csv", selected)
    _plot_patient_metrics(patient_rows, output_dir / "patient-macro-comparison.png")
    _render_selected_cases(
        config_path=config_path,
        evaluations=evaluations,
        records=selected,
        output=output_dir,
    )
    manifest = {
        "checkpoint_sha256": evaluations[2]["identity"]["checkpoint_sha256"],
        "checkpoint_epoch": evaluations[2]["identity"]["checkpoint_epoch"],
        "checkpoint_global_step": evaluations[2]["identity"][
            "checkpoint_global_step"
        ],
        "solver_steps": list(STEP_ORDER),
        "base_seed": evaluations[2]["identity"]["base_seed"],
        "seed_schedule": "base seed plus index in full all-connected cohort",
        "cohort": {
            "patients": 102,
            "endpoints": 278,
            "transition_counts": {"T0->T1": 100, "T1->T2": 92, "T2->T3": 86},
        },
        "aggregation": "patient macro and transition endpoint macro",
        "confidence_intervals": "not computed by user request",
        "case_selection": (
            "within each transition, best/median/worst Euler-20 minus Euler-2 "
            "common-foreground endpoint MAE"
        ),
        "files": {
            "patient_metrics": "patient-macro-metrics.csv",
            "transition_metrics": "transition-endpoint-macro-metrics.csv",
            "metric_plot": "patient-macro-comparison.png",
            "selected_cases": "selected-cases.csv",
            "case_visualizations": "cases/*.png",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare matched Euler 2/8/20 BiFlowNet cohort evaluations"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--euler2-dir", type=Path, required=True)
    parser.add_argument("--euler8-dir", type=Path, required=True)
    parser.add_argument("--euler20-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        config_path=args.config,
        evaluation_dirs={
            2: args.euler2_dir,
            8: args.euler8_dir,
            20: args.euler20_dir,
        },
        output_dir=args.output_dir,
    )
    print(result.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
