"""Patient-level aggregation and nonparametric confidence intervals."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable, Mapping

import numpy as np


def aggregate_by_patient(
    records: Iterable[Mapping[str, object]],
    *,
    metric: str,
) -> dict[str, float]:
    """Average repeated time-pair values within each patient first."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        patient = str(record["patient_id"])
        value = float(record[metric])
        if math.isfinite(value):
            grouped[patient].append(value)
    if not grouped:
        raise ValueError(f"no finite patient values for metric {metric!r}")
    return {patient: float(np.mean(values)) for patient, values in sorted(grouped.items())}


def patient_bootstrap_mean_ci(
    patient_values: Mapping[str, float],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int | None | str]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    values = np.asarray(list(patient_values.values()), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("patient_values has no finite values")
    result: dict[str, float | int | None | str] = {
        "mean": float(values.mean()),
        "patient_count": int(values.size),
        "confidence": float(confidence),
    }
    if values.size < 2:
        result.update(
            lower=None,
            upper=None,
            status="not_computed: at least two held-out patients are required",
        )
        return result
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    result.update(
        lower=float(np.quantile(means, alpha)),
        upper=float(np.quantile(means, 1.0 - alpha)),
        status="computed_by_patient_bootstrap",
    )
    return result
