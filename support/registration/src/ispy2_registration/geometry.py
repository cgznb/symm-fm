from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk

from .models import Geometry


def direction_from_iop(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != 6:
        raise ValueError("ImageOrientationPatient must contain six values")
    first = np.asarray(values[:3], dtype=np.float64)
    second = np.asarray(values[3:], dtype=np.float64)
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("ImageOrientationPatient must be finite")
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm == 0 or second_norm == 0:
        raise ValueError("ImageOrientationPatient axes must be nonzero")
    first /= first_norm
    second /= second_norm
    if not np.isclose(np.dot(first, second), 0.0, atol=1e-5):
        raise ValueError("ImageOrientationPatient axes must be orthogonal")
    third = np.cross(first, second)
    third_norm = np.linalg.norm(third)
    if third_norm == 0:
        raise ValueError("ImageOrientationPatient axes must define a slice direction")
    third /= third_norm
    direction = np.column_stack((first, second, third))
    return tuple(float(value) for value in direction.ravel())


def centered_origin(geometry: Geometry) -> tuple[float, float, float]:
    direction = np.asarray(geometry.direction, dtype=np.float64).reshape(3, 3)
    size = np.asarray(geometry.size_xyz, dtype=np.float64)
    spacing = np.asarray(geometry.spacing_xyz, dtype=np.float64)
    half_extent = (size - 1.0) * spacing / 2.0
    origin = -(direction @ half_extent)
    return tuple(float(value) for value in origin)


def sitk_image_from_array(array_zyx: np.ndarray, geometry: Geometry) -> sitk.Image:
    array = np.asarray(array_zyx)
    if array.shape != geometry.shape_zyx:
        raise ValueError(
            f"array shape {array.shape} does not match geometry {geometry.shape_zyx}"
        )
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(geometry.spacing_xyz)
    image.SetDirection(geometry.direction)
    image.SetOrigin(centered_origin(geometry))
    return image


def array_from_sitk_image(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image)


def itk_image_from_array(array_zyx: np.ndarray, geometry: Geometry):
    import itk

    array = np.asarray(array_zyx)
    if array.shape != geometry.shape_zyx:
        raise ValueError(
            f"array shape {array.shape} does not match geometry {geometry.shape_zyx}"
        )
    image = itk.image_from_array(array)
    image.SetSpacing(geometry.spacing_xyz)
    image.SetDirection(
        itk.matrix_from_array(np.asarray(geometry.direction, dtype=np.float64).reshape(3, 3))
    )
    image.SetOrigin(centered_origin(geometry))
    return image


def array_from_itk_image(image) -> np.ndarray:
    import itk

    return itk.array_from_image(image)


def geometry_from_sitk_image(image: sitk.Image) -> Geometry:
    return Geometry(
        shape_zyx=tuple(reversed(image.GetSize())),
        spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
        direction=tuple(float(value) for value in image.GetDirection()),
    )
