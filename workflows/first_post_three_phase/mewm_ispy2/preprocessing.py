from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import ndimage


NORMALIZATION_PERCENTILES = (0.5, 99.5)
OUTPUT_SHAPE_ZYX = (128, 128, 128)
TARGET_SPACING_ZYX = (2.0, 0.7032, 0.7032)


def resample_image_and_mask(
    image_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    *,
    source_spacing_zyx: Sequence[float],
    target_spacing_zyx: Sequence[float] = TARGET_SPACING_ZYX,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image = np.asarray(image_zyx, dtype=np.float32)
    mask = np.asarray(mask_zyx)
    source_spacing = tuple(float(value) for value in source_spacing_zyx)
    target_spacing = tuple(float(value) for value in target_spacing_zyx)
    if image.ndim != 3 or mask.shape != image.shape:
        raise ValueError("resampling expects matching Z/Y/X image and mask")
    if (
        len(source_spacing) != 3
        or len(target_spacing) != 3
        or not np.isfinite(source_spacing + target_spacing).all()
        or any(value <= 0 for value in source_spacing + target_spacing)
    ):
        raise ValueError("resampling spacing must contain positive finite Z/Y/X values")
    zoom = tuple(
        source / target
        for source, target in zip(source_spacing, target_spacing, strict=True)
    )
    output_image = ndimage.zoom(
        image, zoom=zoom, order=1, mode="constant", cval=0.0, prefilter=False
    ).astype(np.float32, copy=False)
    output_mask = (
        ndimage.zoom(
            (mask > 0).astype(np.uint8),
            zoom=zoom,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        > 0
    ).astype(np.uint8)
    if output_image.shape != output_mask.shape:
        raise RuntimeError("resampled image and mask shapes diverged")
    return output_image, output_mask, {
        "source_spacing_zyx": list(source_spacing),
        "target_spacing_zyx": list(target_spacing),
        "zoom_zyx": list(zoom),
        "shape_before_zyx": list(image.shape),
        "shape_after_zyx": list(output_image.shape),
    }


def robust_nonzero_scale(
    array: np.ndarray,
    *,
    percentiles: Sequence[float] = NORMALIZATION_PERCENTILES,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(array, dtype=np.float32)
    if not np.isfinite(source).all():
        raise ValueError("MRI volume contains non-finite values")
    if len(percentiles) != 2:
        raise ValueError("percentiles must contain two values")
    low_percentile, high_percentile = (float(value) for value in percentiles)
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")

    nonzero = source != 0.0
    foreground = source[nonzero]
    if foreground.size == 0:
        raise ValueError("MRI volume has no nonzero voxels")
    low, high = np.percentile(
        foreground.astype(np.float64), (low_percentile, high_percentile)
    )
    if not np.isfinite((low, high)).all() or float(high - low) <= eps:
        raise ValueError("MRI nonzero percentile range has no usable dynamic range")

    output = np.full_like(source, -1.0, dtype=np.float32)
    unit_scaled = (np.clip(source[nonzero], low, high) - low) / (high - low)
    output[nonzero] = (2.0 * unit_scaled - 1.0).astype(np.float32)
    return output, {
        "low": float(low),
        "high": float(high),
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "nonzero_count": int(foreground.size),
        "output_range": [-1.0, 1.0],
    }


def _validate_arrays(
    image: np.ndarray, mask: np.ndarray, output_shape_zyx: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    image_array = np.asarray(image)
    mask_array = np.asarray(mask)
    output_shape = tuple(int(value) for value in output_shape_zyx)
    if len(output_shape) != 3 or any(value <= 0 for value in output_shape):
        raise ValueError("output_shape_zyx must contain three positive integers")
    if image_array.ndim != 4 or mask_array.ndim != 3:
        raise ValueError("expected C/Z/Y/X image and Z/Y/X mask")
    if image_array.shape[-3:] != mask_array.shape:
        raise ValueError("image and mask spatial shapes must match")
    return image_array, mask_array, output_shape


def _crop_or_pad(
    array: np.ndarray,
    starts_zyx: Sequence[int],
    output_shape_zyx: Sequence[int],
    *,
    fill_value: float | int = 0,
) -> np.ndarray:
    output_shape = tuple(int(value) for value in output_shape_zyx)
    output = np.full(
        (*array.shape[:-3], *output_shape), fill_value, dtype=array.dtype
    )
    source_slices: list[slice] = []
    target_slices: list[slice] = []
    for start, length, available in zip(
        starts_zyx, output_shape, array.shape[-3:], strict=True
    ):
        source_start = max(int(start), 0)
        source_stop = min(int(start) + length, int(available))
        target_start = source_start - int(start)
        target_stop = target_start + max(source_stop - source_start, 0)
        source_slices.append(slice(source_start, source_stop))
        target_slices.append(slice(target_start, target_stop))
    output[(Ellipsis, *target_slices)] = array[(Ellipsis, *source_slices)]
    return output


def _mask_centered_starts(
    mask: np.ndarray, output_shape: tuple[int, int, int]
) -> tuple[list[int], np.ndarray, np.ndarray]:
    positive = np.argwhere(mask > 0)
    if positive.size == 0:
        raise ValueError("cannot center a crop on an empty source mask")
    minimum = positive.min(axis=0)
    maximum = positive.max(axis=0)
    center = (minimum + maximum) // 2
    starts: list[int] = []
    for axis, requested in enumerate(output_shape):
        desired = int(center[axis]) - requested // 2
        if int(maximum[axis] - minimum[axis] + 1) <= requested:
            desired = min(desired, int(minimum[axis]))
            desired = max(desired, int(maximum[axis]) - requested + 1)
        lower = min(0, int(mask.shape[axis]) - requested)
        upper = max(0, int(mask.shape[axis]) - requested)
        starts.append(min(max(desired, lower), upper))
    return starts, minimum, maximum


def tumor_center_crop(
    image_czyx: np.ndarray,
    mask_zyx: np.ndarray,
    output_shape_zyx: Sequence[int] = OUTPUT_SHAPE_ZYX,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image, mask, output_shape = _validate_arrays(
        image_czyx, mask_zyx, output_shape_zyx
    )
    starts, minimum, maximum = _mask_centered_starts(mask, output_shape)
    binary = (mask > 0).astype(np.uint8)
    cropped_image = _crop_or_pad(image, starts, output_shape, fill_value=-1.0)
    cropped_mask = _crop_or_pad(binary, starts, output_shape)[None]
    before = int(binary.sum())
    after = int(cropped_mask.sum())
    return cropped_image, cropped_mask, {
        "coordinate_frame": "tumor_centered_per_visit",
        "bbox_min_zyx": minimum.astype(int).tolist(),
        "bbox_max_zyx": maximum.astype(int).tolist(),
        "crop_start_zyx": starts,
        "mask_voxel_count_before_crop": before,
        "mask_voxel_count_after_crop": after,
        "mask_fully_retained": before == after,
    }


@dataclass(frozen=True)
class RegisteredPairCrop:
    source_image: np.ndarray
    source_mask: np.ndarray
    target_image: np.ndarray
    target_mask: np.ndarray
    metadata: dict[str, Any]


def source_centered_pair_crop(
    source_image_czyx: np.ndarray,
    source_mask_zyx: np.ndarray,
    target_image_czyx: np.ndarray,
    target_mask_zyx: np.ndarray,
    output_shape_zyx: Sequence[int] = OUTPUT_SHAPE_ZYX,
) -> RegisteredPairCrop:
    source_image, source_mask, output_shape = _validate_arrays(
        source_image_czyx, source_mask_zyx, output_shape_zyx
    )
    target_image, target_mask, target_shape = _validate_arrays(
        target_image_czyx, target_mask_zyx, output_shape_zyx
    )
    if target_shape != output_shape or target_image.shape[-3:] != source_image.shape[-3:]:
        raise ValueError("registered source and target grids must match")
    starts, minimum, maximum = _mask_centered_starts(source_mask, output_shape)
    source_binary = (source_mask > 0).astype(np.uint8)
    target_binary = (target_mask > 0).astype(np.uint8)
    return RegisteredPairCrop(
        source_image=_crop_or_pad(
            source_image, starts, output_shape, fill_value=-1.0
        ),
        source_mask=_crop_or_pad(source_binary, starts, output_shape)[None],
        target_image=_crop_or_pad(
            target_image, starts, output_shape, fill_value=-1.0
        ),
        target_mask=_crop_or_pad(target_binary, starts, output_shape)[None],
        metadata={
            "coordinate_frame": "source_mask_centered_registered_t0",
            "source_bbox_min_zyx": minimum.astype(int).tolist(),
            "source_bbox_max_zyx": maximum.astype(int).tolist(),
            "crop_start_zyx": starts,
        },
    )
