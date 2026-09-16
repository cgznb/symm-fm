from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import exposure, morphology


@dataclass(frozen=True)
class PairDrivers:
    fixed_anatomical: np.ndarray
    moving_anatomical: np.ndarray
    fixed_mask: np.ndarray
    moving_mask: np.ndarray
    fixed_rigid: np.ndarray
    moving_rigid: np.ndarray
    fixed_bspline: np.ndarray
    moving_bspline: np.ndarray
    moving_rigidity: np.ndarray
    rigidity_applied: bool


def average_dce(phases: Sequence[np.ndarray]) -> np.ndarray:
    if not phases:
        raise ValueError("at least one DCE phase is required")
    shape = np.asarray(phases[0]).shape
    if len(shape) != 3:
        raise ValueError("DCE phases must be 3D z,y,x arrays")
    accumulator = np.zeros(shape, dtype=np.float64)
    for phase in phases:
        array = np.asarray(phase)
        if array.shape != shape:
            raise ValueError("all DCE phases must have the same shape")
        accumulator += array
    with np.errstate(invalid="ignore", over="ignore"):
        accumulator /= len(phases)
    accumulator[~np.isfinite(accumulator)] = 0.0
    return accumulator.astype(np.float32)


def slice_clahe(volume_zyx: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume_zyx, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("CLAHE input must be a 3D z,y,x array")
    result = np.zeros(volume.shape, dtype=np.float32)
    for z_index, source_slice in enumerate(volume):
        clean = np.array(source_slice, dtype=np.float32, copy=True)
        clean[~np.isfinite(clean)] = 0.0
        maximum = float(np.max(clean))
        if maximum <= 0.0:
            continue
        normalized = np.clip(clean / maximum, 0.0, 1.0)
        enhanced = exposure.equalize_adapthist(normalized, clip_limit=0.01, nbins=256)
        result[z_index] = np.asarray(enhanced * 350.0, dtype=np.float32)
    result[~np.isfinite(result)] = 0.0
    return result


def prepare_ftv_mask(mask_zyx: np.ndarray) -> np.ndarray:
    source = np.asarray(mask_zyx)
    if source.ndim != 3:
        raise ValueError("FTV mask must be a 3D z,y,x array")
    finite = np.isfinite(source)
    binary = finite & (source > 0)
    footprint = morphology.disk(2).astype(bool)
    result = np.zeros(binary.shape, dtype=np.uint8)
    for z_index, source_slice in enumerate(binary):
        dilated = ndimage.binary_dilation(source_slice, structure=footprint)
        result[z_index] = ndimage.binary_fill_holes(dilated).astype(np.uint8)
    return result


def pad_z(array_zyx: np.ndarray) -> np.ndarray:
    array = np.asarray(array_zyx)
    if array.ndim != 3:
        raise ValueError("z padding requires a 3D z,y,x array")
    return np.pad(array, ((1, 1), (0, 0), (0, 0)), mode="constant")


def _normalize_255(array: np.ndarray) -> np.ndarray:
    clean = np.asarray(array, dtype=np.float32).copy()
    clean[~np.isfinite(clean)] = 0.0
    maximum = float(np.max(clean))
    if maximum <= 0.0:
        return np.zeros(clean.shape, dtype=np.float32)
    return np.asarray(clean * (255.0 / maximum), dtype=np.float32)


def rigid_driver(
    anatomical_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    *,
    use_tumor: bool,
) -> np.ndarray:
    image = np.asarray(anatomical_zyx, dtype=np.float32)
    mask = np.asarray(mask_zyx) > 0
    if image.shape != mask.shape:
        raise ValueError("rigid image and mask must have the same shape")
    enhanced = image.copy()
    if use_tumor and np.any(mask):
        enhanced += 5.0 * float(np.max(image)) * mask
    return _normalize_255(enhanced)


def bspline_driver(
    anatomical_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    *,
    weight: float,
    use_tumor: bool,
) -> np.ndarray:
    if weight < 0 or not np.isfinite(weight):
        raise ValueError("tumor weight must be finite and nonnegative")
    image = np.asarray(anatomical_zyx, dtype=np.float32)
    mask = np.asarray(mask_zyx) > 0
    if image.shape != mask.shape:
        raise ValueError("B-spline image and mask must have the same shape")
    enhanced = image.copy()
    if use_tumor and np.any(mask):
        neighborhood = ndimage.gaussian_filter(mask.astype(np.float32), sigma=2.0)
        enhanced *= 1.0 + weight * neighborhood
    return _normalize_255(enhanced)


def prepare_pair_drivers(
    fixed_phases: Sequence[np.ndarray],
    moving_phases: Sequence[np.ndarray],
    fixed_ftv: np.ndarray,
    moving_ftv: np.ndarray,
    *,
    weight: float = 0.5,
) -> PairDrivers:
    fixed_avg = average_dce(fixed_phases)
    moving_avg = average_dce(moving_phases)
    if fixed_avg.shape != np.asarray(fixed_ftv).shape:
        raise ValueError("fixed DCE and FTV mask shapes differ")
    if moving_avg.shape != np.asarray(moving_ftv).shape:
        raise ValueError("moving DCE and FTV mask shapes differ")
    fixed_anatomical = pad_z(slice_clahe(fixed_avg))
    moving_anatomical = pad_z(slice_clahe(moving_avg))
    fixed_mask = pad_z(prepare_ftv_mask(fixed_ftv))
    moving_mask = pad_z(prepare_ftv_mask(moving_ftv))
    rigidity_applied = bool(np.any(moving_mask))
    use_tumor = rigidity_applied
    return PairDrivers(
        fixed_anatomical=fixed_anatomical,
        moving_anatomical=moving_anatomical,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        fixed_rigid=rigid_driver(fixed_anatomical, fixed_mask, use_tumor=use_tumor),
        moving_rigid=rigid_driver(moving_anatomical, moving_mask, use_tumor=use_tumor),
        fixed_bspline=bspline_driver(
            fixed_anatomical, fixed_mask, weight=weight, use_tumor=use_tumor
        ),
        moving_bspline=bspline_driver(
            moving_anatomical, moving_mask, weight=weight, use_tumor=use_tumor
        ),
        moving_rigidity=moving_mask.copy(),
        rigidity_applied=rigidity_applied,
    )
