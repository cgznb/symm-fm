"""Machine-checkable evidence that preprocessing did not use target content."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class LeakageError(ValueError):
    pass


@dataclass(frozen=True)
class PreprocessingProvenance:
    version: str
    source_only: bool
    target_content_used: bool
    crop_basis: str
    normalization_scope: str
    intensity_stats_hash: str
    input_shape_cdhw: tuple[int, int, int, int]
    oriented_shape_cdhw: tuple[int, int, int, int]
    resampled_shape_cdhw: tuple[int, int, int, int]
    output_shape_cdhw: tuple[int, int, int, int]
    source_spacing_dhw: tuple[float, float, float]
    output_spacing_dhw: tuple[float, float, float]
    input_affine_lps: tuple[tuple[float, float, float, float], ...]
    output_affine_lps: tuple[tuple[float, float, float, float], ...]
    crop_plan: Mapping[str, Any]
    extra: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.source_only or self.target_content_used:
            raise LeakageError("preprocessing provenance is not source-only")
        serialized_keys = _all_keys(self.extra)
        forbidden = {
            key
            for key in serialized_keys
            if key.startswith("target_")
            or key in {"target_mask", "target_bbox", "target_statistics"}
        }
        if forbidden:
            raise LeakageError(f"provenance contains target-derived fields: {sorted(forbidden)}")
        if self.normalization_scope != "global_train_patients_shared_phases":
            raise LeakageError("normalization was not fixed from training patients")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": self.version,
            "source_only": self.source_only,
            "target_content_used": self.target_content_used,
            "crop_basis": self.crop_basis,
            "normalization_scope": self.normalization_scope,
            "intensity_stats_hash": self.intensity_stats_hash,
            "input_shape_cdhw": list(self.input_shape_cdhw),
            "oriented_shape_cdhw": list(self.oriented_shape_cdhw),
            "resampled_shape_cdhw": list(self.resampled_shape_cdhw),
            "output_shape_cdhw": list(self.output_shape_cdhw),
            "source_spacing_dhw": list(self.source_spacing_dhw),
            "output_spacing_dhw": list(self.output_spacing_dhw),
            "input_affine_lps": [list(row) for row in self.input_affine_lps],
            "output_affine_lps": [list(row) for row in self.output_affine_lps],
            "crop_plan": dict(self.crop_plan),
            "extra": dict(self.extra),
        }


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result = {str(key).lower() for key in value}
        for child in value.values():
            result.update(_all_keys(child))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for child in value:
            result.update(_all_keys(child))
        return result
    return set()
