#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mewm_ispy2.backend import load_transition_records
from mewm_ispy2.ispy2_biflow_cohort_evaluation import (
    MRI_DATA_RANGE,
    MRI_WINDOW,
    _atomic_json,
    _case_from_decoded,
    _load_payload,
    _read_csv,
    _write_csv,
)
from mewm_ispy2.ispy2_biflow_config import load_ispy2_biflow_config
from mewm_ispy2.ispy2_biflow_visualization import render_case, render_overview
from mewm_ispy2.ispy2_biflow_workflow import _world_data


CLEAN_VISUALIZATION_SCHEMA = "mewm_ispy2_dce0_biflow_clean_cohort_visualization_v1"
CLEAN_GALLERY_FIELDS = (
    "patient_id",
    "transition_id",
    "transition_type",
    "delta_days",
    "seed",
    "clean_five_view_png",
    "clean_patient_overview_png",
    "decoded_artifact",
)


def _prediction_metrics(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        if row["comparison"] != "biflow_prediction" or row["region"] not in (
            "common_foreground",
            "target_tumor",
        ):
            continue
        result.setdefault(row["transition_id"], {})[row["region"]] = {
            "mae_unclipped": float(row["mae_unclipped"]),
        }
    return result


def _flush_patient(
    cases: Sequence[Any], output: Path, patient_id: str, *, overwrite: bool
) -> Path:
    destination = output / "patients-clean" / f"{patient_id}-overview.png"
    if overwrite or not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        render_overview(
            cases,
            destination,
            image_window=MRI_WINDOW,
            error_max=MRI_DATA_RANGE,
            show_contours=False,
        )
    return destination


def render_clean_cohort(
    *,
    config_path: Path,
    evaluation_dir: Path,
    overwrite: bool = False,
    limit: int | None = None,
) -> Path:
    output = evaluation_dir.expanduser().resolve()
    summary_path = output / "summary.json"
    identity_path = output / "run_identity.json"
    if not summary_path.is_file() or not identity_path.is_file():
        raise FileNotFoundError("evaluation summary or run identity is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    gallery = _read_csv(output / "gallery_index.csv")
    endpoint_rows = _read_csv(output / "endpoint_metrics.csv")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        gallery = gallery[:limit]
    if not gallery:
        raise ValueError("evaluation gallery is empty")
    if len({row["transition_id"] for row in gallery}) != len(gallery):
        raise ValueError("evaluation gallery contains duplicate transition ids")
    if limit is None and [row["transition_id"] for row in gallery] != identity[
        "pair_ids"
    ]:
        raise ValueError("gallery order differs from run identity")

    config = load_ispy2_biflow_config(config_path)
    all_pairs, _, _, roi_cache = _world_data(config)
    pairs = {pair.pair_id: pair for pair in all_pairs if pair.split == "val"}
    visits = load_transition_records(
        config.base.data.bundle_json,
        config.base.data.phase_manifest_csv,
        backend=config.base.data.backend,
    ).visits
    metrics = _prediction_metrics(endpoint_rows)

    clean_rows = []
    patient_cases = []
    current_patient: str | None = None
    completed_patients = 0
    for index, row in enumerate(gallery, start=1):
        pair_id = row["transition_id"]
        if pair_id not in pairs or pair_id not in metrics:
            raise ValueError(f"pair or prediction metrics are missing: {pair_id}")
        pair = pairs[pair_id]
        if current_patient is not None and pair.patient_id != current_patient:
            _flush_patient(
                patient_cases, output, current_patient, overwrite=overwrite
            )
            completed_patients += 1
            patient_cases = []
        current_patient = pair.patient_id

        source = roi_cache.load(pair.source_visit)
        target = roi_cache.load(visits[pair.target_visit_id])
        decoded_path = (output / row["decoded_artifact"]).resolve()
        decoded = _load_payload(decoded_path)
        case = _case_from_decoded(
            pair=pair,
            seed=int(row["seed"]),
            source=source,
            target=target,
            prediction=decoded["prediction"].float(),
            target_reconstruction=decoded["target_reconstruction"].float(),
            common=source.valid_foreground.bool() & target.valid_foreground.bool(),
            prediction_metrics=metrics[pair_id],
        )
        patient_cases.append(case)
        case_path = (
            output
            / "cases-clean"
            / pair.patient_id
            / Path(row["five_view_png"]).name
        )
        if overwrite or not case_path.is_file():
            case_path.parent.mkdir(parents=True, exist_ok=True)
            render_case(
                case,
                case_path,
                image_window=MRI_WINDOW,
                error_max=MRI_DATA_RANGE,
                show_contours=False,
            )
        patient_path = output / "patients-clean" / f"{pair.patient_id}-overview.png"
        clean_rows.append(
            {
                "patient_id": pair.patient_id,
                "transition_id": pair.pair_id,
                "transition_type": pair.transition_type,
                "delta_days": pair.delta_days,
                "seed": int(row["seed"]),
                "clean_five_view_png": str(case_path.relative_to(output)),
                "clean_patient_overview_png": str(patient_path.relative_to(output)),
                "decoded_artifact": row["decoded_artifact"],
            }
        )
        print(f"rendered clean {index}/{len(gallery)}: {pair_id}", flush=True)
    if current_patient is not None:
        _flush_patient(patient_cases, output, current_patient, overwrite=overwrite)
        completed_patients += 1

    _write_csv(output / "gallery_index_clean.csv", clean_rows, CLEAN_GALLERY_FIELDS)
    manifest = {
        "schema": CLEAN_VISUALIZATION_SCHEMA,
        "source_evaluation": str(output),
        "source_checkpoint_sha256": identity["checkpoint_sha256"],
        "solver_steps": identity["solver_steps"],
        "cases": len(clean_rows),
        "patients": completed_patients,
        "complete_cohort": limit is None,
        "visible_tumor_contours": False,
        "slice_selection_uses_tumor_mask": True,
        "outside_fov_masking": summary["visualization"]["outside_fov_rendering"],
        "image_window": list(MRI_WINDOW),
        "absolute_error_vmax": MRI_DATA_RANGE,
        "files": {
            "gallery_index": "gallery_index_clean.csv",
            "case_visualizations": "cases-clean/*/*.png",
            "patient_overviews": "patients-clean/*.png",
        },
    }
    _atomic_json(output / "clean_visualization.json", manifest)
    print(f"saved clean cohort visualization: {output}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-render a saved BiFlow cohort without visible tumor contours"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    render_clean_cohort(
        config_path=args.config,
        evaluation_dir=args.evaluation_dir,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
