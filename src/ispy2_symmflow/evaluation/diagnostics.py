"""Held-out perturbation diagnostics for source and condition usage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .aggregate import aggregate_by_patient, patient_bootstrap_mean_ci


@dataclass(frozen=True)
class DiagnosticCase:
    patient_id: str
    source: Tensor
    target: Tensor
    conditions: Mapping[str, Any]


PredictionFunction = Callable[[Tensor, Mapping[str, Any], int], Tensor]


DEFAULT_ABLATIONS: Mapping[str, tuple[str, ...]] = {
    "treatment": ("treatment_arm", "drug", "dose"),
    "clinical": ("age", "hr_status", "her2_status", "mammaprint"),
    "time": ("stage_i", "stage_j", "delta_days", "interval_missing", "interval_source"),
}


def patient_derangement(patient_ids: Sequence[str], *, seed: int) -> tuple[int, ...]:
    """Return a reproducible permutation with no patient mapped to itself."""

    if len(patient_ids) < 2 or len(set(patient_ids)) != len(patient_ids):
        raise ValueError("diagnostics require at least two unique held-out patients")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patient_ids))
    mapping = np.empty(len(patient_ids), dtype=np.int64)
    mapping[order] = np.roll(order, -1)
    if any(patient_ids[index] == patient_ids[int(other)] for index, other in enumerate(mapping)):
        raise RuntimeError("failed to construct a patient derangement")
    return tuple(int(value) for value in mapping)


def _case_image(value: Tensor, *, name: str) -> Tensor:
    if value.ndim == 4:
        value = value[None]
    if value.ndim != 5 or value.shape[0] != 1:
        raise ValueError(f"diagnostic {name} must represent one [C,D,H,W] MRI")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"diagnostic {name} must be a finite floating-point tensor")
    return value


def _mae(left: Tensor, right: Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError("diagnostic prediction and reference geometries differ")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise FloatingPointError("diagnostic prediction contains non-finite values")
    return float((left - right).abs().mean())


def run_usage_diagnostics(
    cases: Sequence[DiagnosticCase],
    predictor: PredictionFunction,
    *,
    seed: int = 0,
    ablations: Mapping[str, Sequence[str]] = DEFAULT_ABLATIONS,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    """Compare reference predictions with controlled held-out perturbations.

    Source shuffling keeps each target/condition fixed but supplies another
    patient's source. Pair shuffling keeps predictions fixed and rotates targets.
    Ablations replace selected known-condition fields with their missing value.
    """

    patient_ids = [str(case.patient_id) for case in cases]
    mapping = patient_derangement(patient_ids, seed=seed)
    sources = [_case_image(case.source, name="source") for case in cases]
    targets = [_case_image(case.target, name="target") for case in cases]
    if len({tuple(value.shape) for value in sources + targets}) != 1:
        raise ValueError("all diagnostic cases must share one voxel geometry")

    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_seed = int(seed) + 1009 * index
        reference = predictor(sources[index], dict(case.conditions), case_seed)
        shuffled_source = predictor(
            sources[mapping[index]], dict(case.conditions), case_seed
        )
        record: dict[str, Any] = {
            "patient_id": patient_ids[index],
            "shuffled_source_patient_id": patient_ids[mapping[index]],
            "shuffled_target_patient_id": patient_ids[mapping[index]],
            "reference_mae": _mae(reference, targets[index]),
            "source_shuffled_mae": _mae(shuffled_source, targets[index]),
            "source_output_change_mae": _mae(shuffled_source, reference),
            "pair_shuffled_target_mae": _mae(reference, targets[mapping[index]]),
        }
        for name, fields in ablations.items():
            altered = dict(case.conditions)
            ablated_fields = [field for field in fields if field in altered]
            for field in ablated_fields:
                altered[field] = None
            prediction = predictor(sources[index], altered, case_seed)
            record[f"{name}_ablated_mae"] = _mae(prediction, targets[index])
            record[f"{name}_output_change_mae"] = _mae(prediction, reference)
            record[f"{name}_ablated_fields"] = ablated_fields
        records.append(record)

    metric_names = sorted(
        key
        for key in records[0]
        if key.endswith("_mae") and isinstance(records[0][key], float)
    )
    summaries = {}
    for metric in metric_names:
        patient_values = aggregate_by_patient(records, metric=metric)
        summaries[metric] = patient_bootstrap_mean_ci(
            patient_values,
            samples=bootstrap_samples,
            seed=int(seed),
        )
    return {
        "records": records,
        "patient_level": summaries,
        "patient_count": len(cases),
        "seed": int(seed),
        "warnings": [
            "source/pair shuffling intentionally breaks patient correspondence and is diagnostic only",
            "condition ablation measures model sensitivity, not a causal treatment effect",
        ],
    }


__all__ = [
    "DEFAULT_ABLATIONS",
    "DiagnosticCase",
    "patient_derangement",
    "run_usage_diagnostics",
]
