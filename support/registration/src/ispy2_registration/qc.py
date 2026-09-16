from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .models import Geometry


@dataclass(frozen=True)
class QCThresholds:
    max_ftv_volume_change: float = 0.10
    similarity_tolerance: float = 1e-6
    protected_margin_mm: float = 10.0
    max_outside_fold_fraction: float = 0.02

    def __post_init__(self) -> None:
        if self.max_ftv_volume_change < 0:
            raise ValueError("max_ftv_volume_change must be nonnegative")
        if self.similarity_tolerance < 0:
            raise ValueError("similarity_tolerance must be nonnegative")
        if self.protected_margin_mm < 0:
            raise ValueError("protected_margin_mm must be nonnegative")
        if not 0.0 <= self.max_outside_fold_fraction <= 1.0:
            raise ValueError("max_outside_fold_fraction must be between zero and one")


@dataclass(frozen=True)
class QCInputs:
    fixed_image: np.ndarray
    rigid_image: np.ndarray
    deformable_image: np.ndarray
    fixed_ftv: np.ndarray
    rigid_ftv: np.ndarray
    deformable_ftv: np.ndarray
    source_moving_ftv: np.ndarray
    source_geometry: Geometry
    target_geometry: Geometry
    jacobian: np.ndarray


@dataclass(frozen=True)
class QCResult:
    accepted: bool
    status: str
    qc_status: str
    reasons: tuple[str, ...]
    qc_warning_codes: tuple[str, ...]
    rigid_non_tumor_ncc: float
    deformable_non_tumor_ncc: float
    rigid_protected_ncc: float | None
    deformable_protected_ncc: float | None
    jacobian_min: float
    jacobian_max: float
    protected_roi_fold_fraction: float
    outside_protected_roi_fold_fraction: float
    global_fold_fraction: float
    candidate_outside_protected_roi_folding: bool
    outside_protected_roi_folding_accepted: bool
    ftv_volume_change: float | None


def masked_ncc(
    fixed: np.ndarray,
    moving: np.ndarray,
    excluded: np.ndarray | None = None,
    *,
    included: np.ndarray | None = None,
) -> float:
    first = np.asarray(fixed, dtype=np.float64)
    second = np.asarray(moving, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("NCC arrays must have the same shape")
    valid = np.isfinite(first) & np.isfinite(second)
    if excluded is not None:
        excluded_array = np.asarray(excluded, dtype=bool)
        if excluded_array.shape != first.shape:
            raise ValueError("NCC exclusion mask shape differs from images")
        valid &= ~excluded_array
    if included is not None:
        included_array = np.asarray(included, dtype=bool)
        if included_array.shape != first.shape:
            raise ValueError("NCC inclusion mask shape differs from images")
        valid &= included_array
    first_values = first[valid]
    second_values = second[valid]
    if first_values.size < 2:
        return float("-inf")
    first_values -= np.mean(first_values)
    second_values -= np.mean(second_values)
    denominator = np.linalg.norm(first_values) * np.linalg.norm(second_values)
    if denominator == 0:
        return 1.0 if np.allclose(first_values, second_values) else float("-inf")
    return float(np.dot(first_values, second_values) / denominator)


def _physical_volume(mask: np.ndarray, geometry: Geometry) -> float:
    return float(np.count_nonzero(np.asarray(mask) > 0) * geometry.voxel_volume_mm3)


def protected_roi_mask(inputs: QCInputs, margin_mm: float) -> np.ndarray:
    if margin_mm < 0:
        raise ValueError("protected margin must be nonnegative")
    tumor = (
        (np.asarray(inputs.fixed_ftv) > 0)
        | (np.asarray(inputs.rigid_ftv) > 0)
        | (np.asarray(inputs.deformable_ftv) > 0)
    )
    if not np.any(tumor) or margin_mm == 0:
        return tumor
    spacing_zyx = tuple(reversed(inputs.target_geometry.spacing_xyz))
    distance = ndimage.distance_transform_edt(~tumor, sampling=spacing_zyx)
    return tumor | (distance <= margin_mm)


def _fraction(mask: np.ndarray, region: np.ndarray) -> float:
    denominator = int(np.count_nonzero(region))
    if denominator == 0:
        return 0.0
    return float(np.count_nonzero(mask & region) / denominator)


def evaluate_deformable_qc(
    inputs: QCInputs,
    thresholds: QCThresholds,
) -> QCResult:
    target_shape = inputs.target_geometry.shape_zyx
    target_arrays = (
        inputs.fixed_image,
        inputs.rigid_image,
        inputs.deformable_image,
        inputs.fixed_ftv,
        inputs.rigid_ftv,
        inputs.deformable_ftv,
        inputs.jacobian,
    )
    if any(np.asarray(array).shape != target_shape for array in target_arrays):
        raise ValueError("QC target arrays must match target geometry")
    if np.asarray(inputs.source_moving_ftv).shape != inputs.source_geometry.shape_zyx:
        raise ValueError("source moving FTV does not match source geometry")

    reasons: list[str] = []
    deformable = np.asarray(inputs.deformable_image)
    jacobian = np.asarray(inputs.jacobian, dtype=np.float64)
    if not np.isfinite(deformable).all():
        reasons.append("deformable_nonfinite")
    if not np.isfinite(jacobian).all():
        reasons.append("jacobian_nonfinite")
    finite_jacobian = jacobian[np.isfinite(jacobian)]
    jacobian_min = float(np.min(finite_jacobian)) if finite_jacobian.size else float("nan")
    jacobian_max = float(np.max(finite_jacobian)) if finite_jacobian.size else float("nan")
    protected = protected_roi_mask(inputs, thresholds.protected_margin_mm)
    outside = ~protected
    folding = np.isfinite(jacobian) & (jacobian <= 0)
    protected_fold_fraction = _fraction(folding, protected)
    outside_fold_fraction = _fraction(folding, outside)
    global_fold_fraction = float(np.count_nonzero(folding) / folding.size)
    if protected_fold_fraction > 0:
        reasons.append("jacobian_nonpositive_protected_roi")
    if outside_fold_fraction > thresholds.max_outside_fold_fraction:
        reasons.append("outside_protected_roi_fold_fraction")

    excluded = (
        (np.asarray(inputs.fixed_ftv) > 0)
        | (np.asarray(inputs.rigid_ftv) > 0)
        | (np.asarray(inputs.deformable_ftv) > 0)
    )
    rigid_ncc = masked_ncc(inputs.fixed_image, inputs.rigid_image, excluded)
    deformable_ncc = masked_ncc(inputs.fixed_image, inputs.deformable_image, excluded)
    if deformable_ncc + thresholds.similarity_tolerance < rigid_ncc:
        reasons.append("non_tumor_similarity_regressed")

    protected_shell = protected & ~excluded
    rigid_protected_ncc: float | None = None
    deformable_protected_ncc: float | None = None
    if np.count_nonzero(protected_shell) >= 2:
        rigid_protected_ncc = masked_ncc(
            inputs.fixed_image,
            inputs.rigid_image,
            included=protected_shell,
        )
        deformable_protected_ncc = masked_ncc(
            inputs.fixed_image,
            inputs.deformable_image,
            included=protected_shell,
        )
        if (
            deformable_protected_ncc + thresholds.similarity_tolerance
            < rigid_protected_ncc
        ):
            reasons.append("protected_roi_similarity_regressed")

    source_volume = _physical_volume(inputs.source_moving_ftv, inputs.source_geometry)
    ftv_volume_change: float | None = None
    if source_volume > 0:
        registered_volume = _physical_volume(inputs.deformable_ftv, inputs.target_geometry)
        ftv_volume_change = abs(registered_volume / source_volume - 1.0)
        if ftv_volume_change > thresholds.max_ftv_volume_change:
            reasons.append("ftv_volume_change")

    unique_reasons = tuple(dict.fromkeys(reasons))
    accepted = not unique_reasons
    candidate_outside_folding = outside_fold_fraction > 0
    outside_folding_accepted = accepted and candidate_outside_folding
    warnings: list[str] = []
    if candidate_outside_folding:
        warnings.append("candidate_outside_protected_roi_folding")
    if outside_folding_accepted:
        warnings.append("outside_protected_roi_folding_accepted")
    status = "deformable" if accepted else "rigid_fallback"
    qc_status = (
        "warning_outside_protected_roi_folding"
        if outside_folding_accepted
        else status
    )
    return QCResult(
        accepted=accepted,
        status=status,
        qc_status=qc_status,
        reasons=unique_reasons,
        qc_warning_codes=tuple(warnings),
        rigid_non_tumor_ncc=rigid_ncc,
        deformable_non_tumor_ncc=deformable_ncc,
        rigid_protected_ncc=rigid_protected_ncc,
        deformable_protected_ncc=deformable_protected_ncc,
        jacobian_min=jacobian_min,
        jacobian_max=jacobian_max,
        protected_roi_fold_fraction=protected_fold_fraction,
        outside_protected_roi_fold_fraction=outside_fold_fraction,
        global_fold_fraction=global_fold_fraction,
        candidate_outside_protected_roi_folding=candidate_outside_folding,
        outside_protected_roi_folding_accepted=outside_folding_accepted,
        ftv_volume_change=ftv_volume_change,
    )
