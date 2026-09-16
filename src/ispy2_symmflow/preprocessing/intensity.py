"""Training-patient-only, cross-phase shared MRI intensity scaling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IntensityStats:
    lower: float
    upper: float
    lower_percentile: float
    upper_percentile: float
    fit_split: str
    fit_patient_count: int
    fit_patient_hash: str
    sampled_voxel_count: int
    channels_shared: bool = True
    method: str = "global_train_foreground_percentile"

    def to_dict(self) -> dict[str, object]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "fit_split": self.fit_split,
            "fit_patient_count": self.fit_patient_count,
            "fit_patient_hash": self.fit_patient_hash,
            "sampled_voxel_count": self.sampled_voxel_count,
            "channels_shared": self.channels_shared,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "IntensityStats":
        return cls(
            lower=float(value["lower"]),
            upper=float(value["upper"]),
            lower_percentile=float(value["lower_percentile"]),
            upper_percentile=float(value["upper_percentile"]),
            fit_split=str(value["fit_split"]),
            fit_patient_count=int(value["fit_patient_count"]),
            fit_patient_hash=str(value["fit_patient_hash"]),
            sampled_voxel_count=int(value["sampled_voxel_count"]),
            channels_shared=bool(value.get("channels_shared", True)),
            method=str(value.get("method", "global_train_foreground_percentile")),
        )


def _patient_hash(patient_ids: set[str]) -> str:
    payload = json.dumps(sorted(patient_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_values(
    volume: NDArray[np.floating],
    *,
    max_voxels: int,
    exclude_zero_background: bool,
) -> NDArray[np.float64]:
    values = np.asarray(volume, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values)
    if exclude_zero_background:
        valid &= values != 0
    values = values[valid]
    if values.size > max_voxels:
        indices = np.linspace(0, values.size - 1, max_voxels, dtype=np.int64)
        values = values[indices]
    return values


def fit_intensity_stats(
    samples: Iterable[tuple[str, NDArray[np.floating]]],
    split_by_patient: Mapping[str, str],
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    max_voxels_per_visit: int = 250_000,
    exclude_zero_background: bool = True,
) -> IntensityStats:
    """Fit one scalar range across all channels from training patients only."""

    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    if max_voxels_per_visit <= 0:
        raise ValueError("max_voxels_per_visit must be positive")
    chunks: list[NDArray[np.float64]] = []
    patients: set[str] = set()
    for patient_id, volume in samples:
        if patient_id not in split_by_patient:
            raise ValueError(f"no split assignment for patient {patient_id}")
        if split_by_patient[patient_id] != "train":
            continue
        array = np.asarray(volume)
        if array.ndim not in (3, 4):
            raise ValueError(f"MRI volume must be [D,H,W] or [C,D,H,W], got {array.shape}")
        values = _sample_values(
            array,
            max_voxels=max_voxels_per_visit,
            exclude_zero_background=exclude_zero_background,
        )
        if values.size:
            chunks.append(values)
            patients.add(patient_id)
    if not chunks:
        raise ValueError("no finite non-background training voxels were supplied")
    combined = np.concatenate(chunks)
    lower, upper = np.percentile(combined, [lower_percentile, upper_percentile])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("training intensity range is empty or degenerate")
    return IntensityStats(
        lower=float(lower),
        upper=float(upper),
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        fit_split="train",
        fit_patient_count=len(patients),
        fit_patient_hash=_patient_hash(patients),
        sampled_voxel_count=int(combined.size),
    )


def apply_intensity_stats(
    volume: NDArray[np.floating],
    stats: IntensityStats,
    *,
    output_range: tuple[float, float] = (-1.0, 1.0),
) -> NDArray[np.float32]:
    """Clip to fixed training statistics and map to a documented range."""

    output_low, output_high = output_range
    if output_high <= output_low:
        raise ValueError("output_range must be increasing")
    array = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("MRI volume contains non-finite values")
    clipped = np.clip(array, stats.lower, stats.upper)
    scaled = (clipped - stats.lower) / (stats.upper - stats.lower)
    scaled = scaled * (output_high - output_low) + output_low
    return np.asarray(scaled, dtype=np.float32)
