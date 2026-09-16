"""Fixed-grid and explicitly source-derived 3D crop operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


LPS_TO_RAS = np.diag((-1.0, -1.0, 1.0, 1.0))


@dataclass(frozen=True)
class CropPlan:
    source_shape_dhw: tuple[int, int, int]
    output_shape_dhw: tuple[int, int, int]
    input_start_dhw: tuple[int, int, int]
    input_stop_dhw: tuple[int, int, int]
    output_start_dhw: tuple[int, int, int]
    output_stop_dhw: tuple[int, int, int]
    source_offset_dhw: tuple[int, int, int]
    basis: str
    foreground_empty: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "source_shape_dhw": list(self.source_shape_dhw),
            "output_shape_dhw": list(self.output_shape_dhw),
            "input_start_dhw": list(self.input_start_dhw),
            "input_stop_dhw": list(self.input_stop_dhw),
            "output_start_dhw": list(self.output_start_dhw),
            "output_stop_dhw": list(self.output_stop_dhw),
            "source_offset_dhw": list(self.source_offset_dhw),
            "basis": self.basis,
            "foreground_empty": self.foreground_empty,
        }


def _shape3(value: Sequence[int], name: str) -> tuple[int, int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 3 or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain three positive integers")
    return result  # type: ignore[return-value]


def _foreground_mask(volume: NDArray[np.floating]) -> NDArray[np.bool_]:
    array = np.asarray(volume)
    if array.ndim == 4:
        return np.any(np.isfinite(array) & (array != 0), axis=0)
    if array.ndim == 3:
        return np.isfinite(array) & (array != 0)
    raise ValueError(f"volume must be [D,H,W] or [C,D,H,W], got {array.shape}")


def make_crop_plan(
    volume: NDArray[np.floating],
    output_shape_dhw: Sequence[int],
    *,
    mode: str = "fixed_center",
    source_mask: NDArray[np.bool_] | None = None,
) -> CropPlan:
    """Create a crop using a fixed rule or this source image only.

    There is intentionally no target volume or target mask argument.
    """

    array = np.asarray(volume)
    if array.ndim not in (3, 4):
        raise ValueError(f"volume must be [D,H,W] or [C,D,H,W], got {array.shape}")
    source_shape = _shape3(array.shape[-3:], "source shape")
    output_shape = _shape3(output_shape_dhw, "output_shape_dhw")
    foreground_empty = False
    if mode == "fixed_center":
        center = tuple(size // 2 for size in source_shape)
        basis = "fixed_grid_center"
    elif mode in {"source_foreground", "source_mask"}:
        if mode == "source_mask":
            if source_mask is None:
                raise ValueError("source_mask mode requires a source-side mask")
            foreground = np.asarray(source_mask, dtype=bool)
            if foreground.shape != source_shape:
                raise ValueError("source_mask shape must match the source spatial shape")
            basis = "source_mask"
        else:
            if source_mask is not None:
                raise ValueError("source_mask is only accepted when mode='source_mask'")
            foreground = _foreground_mask(array)
            basis = "source_image_foreground"
        coordinates = np.argwhere(foreground)
        if coordinates.size:
            low = coordinates.min(axis=0)
            high = coordinates.max(axis=0) + 1
            center = tuple(int((left + right - 1) // 2) for left, right in zip(low, high))
        else:
            center = tuple(size // 2 for size in source_shape)
            foreground_empty = True
            basis += "_empty_fallback_center"
    else:
        raise ValueError(f"unsupported crop mode: {mode}")

    desired_start = tuple(c - out // 2 for c, out in zip(center, output_shape))
    desired_stop = tuple(start + out for start, out in zip(desired_start, output_shape))
    input_start = tuple(max(0, start) for start in desired_start)
    input_stop = tuple(min(size, stop) for size, stop in zip(source_shape, desired_stop))
    output_start = tuple(source - desired for source, desired in zip(input_start, desired_start))
    output_stop = tuple(
        start + (stop - source_start)
        for start, stop, source_start in zip(output_start, input_stop, input_start)
    )
    return CropPlan(
        source_shape_dhw=source_shape,
        output_shape_dhw=output_shape,
        input_start_dhw=input_start,
        input_stop_dhw=input_stop,
        output_start_dhw=output_start,
        output_stop_dhw=output_stop,
        source_offset_dhw=desired_start,
        basis=basis,
        foreground_empty=foreground_empty,
    )


def apply_crop_plan(
    volume: NDArray[np.floating],
    plan: CropPlan,
    *,
    padding_value: float = -1.0,
) -> NDArray[np.float32]:
    array = np.asarray(volume)
    if tuple(array.shape[-3:]) != plan.source_shape_dhw:
        raise ValueError("crop plan source shape does not match the volume")
    output_shape = (*array.shape[:-3], *plan.output_shape_dhw)
    output = np.full(output_shape, padding_value, dtype=np.float32)
    source_slices = tuple(
        slice(start, stop)
        for start, stop in zip(plan.input_start_dhw, plan.input_stop_dhw)
    )
    output_slices = tuple(
        slice(start, stop)
        for start, stop in zip(plan.output_start_dhw, plan.output_stop_dhw)
    )
    output[(..., *output_slices)] = array[(..., *source_slices)]
    return output


def update_affine_for_crop(
    affine_lps: NDArray[np.floating], plan: CropPlan
) -> NDArray[np.float64]:
    affine = np.asarray(affine_lps, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError("affine_lps must have shape [4,4]")
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = np.asarray(plan.source_offset_dhw, dtype=np.float64)
    return affine @ translation


def reorient_volume(
    volume: NDArray[np.floating],
    affine_lps: NDArray[np.floating],
    target_axis_codes: str | None,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float64],
    tuple[float, float, float],
    tuple[str, str, str] | None,
]:
    """Reorient spatial axes while retaining an LPS world-coordinate affine.

    Axis codes are interpreted by nibabel in RAS world coordinates. `SAR` is
    the canonical target for this project's array order: slice/depth, row, and
    column. The returned affine still maps array indices to DICOM LPS.
    """

    array = np.asarray(volume, dtype=np.float32)
    if array.ndim not in (3, 4):
        raise ValueError(f"volume must be [D,H,W] or [C,D,H,W], got {array.shape}")
    affine = np.asarray(affine_lps, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError("affine_lps must have shape [4,4]")
    if target_axis_codes is None:
        spacing = tuple(float(np.linalg.norm(affine[:3, axis])) for axis in range(3))
        return array, affine, spacing, None  # type: ignore[return-value]
    codes = tuple(target_axis_codes.upper())
    if len(codes) != 3 or any(code not in "LRPAIS" for code in codes):
        raise ValueError("target_axis_codes must be three anatomical axis codes")
    anatomical_axes = [{"L", "R"}, {"P", "A"}, {"I", "S"}]
    if any(sum(code in axis for code in codes) != 1 for axis in anatomical_axes):
        raise ValueError("target_axis_codes must contain one code from each anatomical axis")
    try:
        from nibabel.orientations import (
            aff2axcodes,
            apply_orientation,
            axcodes2ornt,
            inv_ornt_aff,
            io_orientation,
            ornt_transform,
        )
    except ImportError as exc:  # pragma: no cover - optional train dependency
        raise RuntimeError("nibabel is required for anatomical reorientation") from exc

    affine_ras = LPS_TO_RAS @ affine
    start = io_orientation(affine_ras)
    end = axcodes2ornt(codes)
    transform = ornt_transform(start, end)
    spatial_shape = array.shape[-3:]
    if array.ndim == 4:
        oriented = np.stack(
            [apply_orientation(channel, transform) for channel in array], axis=0
        )
    else:
        oriented = apply_orientation(array, transform)
    oriented_ras = affine_ras @ inv_ornt_aff(transform, spatial_shape)
    actual_codes = tuple(str(code) for code in aff2axcodes(oriented_ras))
    if actual_codes != codes:
        raise ValueError(
            f"could not reorient affine to {codes}; resulting codes are {actual_codes}"
        )
    oriented_lps = LPS_TO_RAS @ oriented_ras
    spacing = tuple(
        float(np.linalg.norm(oriented_lps[:3, axis])) for axis in range(3)
    )
    return (
        np.asarray(oriented, dtype=np.float32),
        np.asarray(oriented_lps, dtype=np.float64),
        spacing,  # type: ignore[arg-type]
        actual_codes,  # type: ignore[arg-type]
    )


def resample_volume(
    volume: NDArray[np.floating],
    affine_lps: NDArray[np.floating],
    source_spacing_dhw: Sequence[float],
    target_spacing_dhw: Sequence[float] | None,
) -> tuple[NDArray[np.float32], NDArray[np.float64], tuple[float, float, float]]:
    """Linear 3D resampling with the physical origin and directions retained."""

    array = np.asarray(volume, dtype=np.float32)
    source_spacing = tuple(float(value) for value in source_spacing_dhw)
    if len(source_spacing) != 3 or any(value <= 0 for value in source_spacing):
        raise ValueError("source_spacing_dhw must contain three positive values")
    if target_spacing_dhw is None:
        return array, np.asarray(affine_lps, dtype=np.float64), source_spacing  # type: ignore[return-value]
    target_spacing = tuple(float(value) for value in target_spacing_dhw)
    if len(target_spacing) != 3 or any(value <= 0 for value in target_spacing):
        raise ValueError("target_spacing_dhw must contain three positive values")
    factors = tuple(source / target for source, target in zip(source_spacing, target_spacing))
    if all(abs(factor - 1.0) < 1e-6 for factor in factors):
        return array, np.asarray(affine_lps, dtype=np.float64), target_spacing  # type: ignore[return-value]
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:  # pragma: no cover - depends on optional train extra
        raise RuntimeError("scipy is required when target spacing changes") from exc
    zoom_factors = factors if array.ndim == 3 else (1.0, *factors)
    resampled = zoom(array, zoom_factors, order=1, mode="nearest", prefilter=False)
    affine = np.asarray(affine_lps, dtype=np.float64).copy()
    for axis, spacing in enumerate(target_spacing):
        direction = affine[:3, axis]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            raise ValueError("affine_lps contains a degenerate spatial direction")
        affine[:3, axis] = direction / norm * spacing
    return np.asarray(resampled, dtype=np.float32), affine, target_spacing  # type: ignore[return-value]
