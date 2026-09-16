from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from typing import Literal

import itk
import numpy as np
import SimpleITK as sitk

from .geometry import centered_origin, itk_image_from_array, sitk_image_from_array
from .models import Geometry


Interpolation = Literal["linear", "nearest"]


class TransformApplicationError(RuntimeError):
    pass


def _strings(values) -> list[str]:
    return [format(float(value), ".15g") for value in values]


def configure_transform_parameters(
    transform_parameters,
    target_geometry: Geometry,
    *,
    interpolation: Interpolation,
    default_value: float = 0.0,
):
    if interpolation not in {"linear", "nearest"}:
        raise ValueError(f"unsupported interpolation: {interpolation}")
    configured = itk.ParameterObject.New()
    configured.SetParameterMaps(transform_parameters.GetParameterMaps())
    index = configured.GetNumberOfParameterMaps() - 1
    if index < 0:
        raise TransformApplicationError("transform parameter object is empty")
    configured.SetParameter(index, "Size", [str(value) for value in target_geometry.size_xyz])
    configured.SetParameter(index, "Index", ["0", "0", "0"])
    configured.SetParameter(index, "Spacing", _strings(target_geometry.spacing_xyz))
    configured.SetParameter(index, "Origin", _strings(centered_origin(target_geometry)))
    configured.SetParameter(index, "Direction", _strings(target_geometry.direction))
    configured.SetParameter(index, "UseDirectionCosines", ["true"])
    configured.SetParameter(index, "ResampleInterpolator", ["FinalBSplineInterpolator"])
    configured.SetParameter(
        index,
        "FinalBSplineInterpolationOrder",
        ["1" if interpolation == "linear" else "0"],
    )
    configured.SetParameter(
        index,
        "DefaultPixelValue",
        [format(float(default_value), ".15g")],
    )
    configured.SetParameter(index, "ResultImagePixelType", ["float"])
    configured.SetParameter(index, "CompressResultImage", ["true"])
    return configured


def _clean_float(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32).copy()
    result[~np.isfinite(result)] = 0.0
    return result


def displacement_transform(
    field_zyx_xyz: np.ndarray,
    geometry: Geometry,
) -> sitk.DisplacementFieldTransform:
    field = np.asarray(field_zyx_xyz, dtype=np.float64)
    expected = (*geometry.shape_zyx, 3)
    if field.shape != expected or not np.isfinite(field).all():
        raise TransformApplicationError(
            f"invalid displacement field shape or values: {field.shape}, expected {expected}"
        )
    image = sitk.GetImageFromArray(field, isVector=True)
    image.SetSpacing(geometry.spacing_xyz)
    image.SetDirection(geometry.direction)
    image.SetOrigin(centered_origin(geometry))
    return sitk.DisplacementFieldTransform(image)


def compose_phase_with_longitudinal(
    phase_transform: sitk.Transform | None,
    longitudinal_field: np.ndarray,
    target_geometry: Geometry,
) -> sitk.CompositeTransform:
    longitudinal = displacement_transform(longitudinal_field, target_geometry)
    combined = sitk.CompositeTransform(3)
    if phase_transform is not None:
        if phase_transform.GetDimension() != 3:
            raise ValueError("phase transform must be 3D")
        combined.AddTransform(phase_transform)
    combined.AddTransform(longitudinal)
    return combined


def dicom_grid_transform(
    dce_geometry: Geometry,
    dce_origin_xyz: Sequence[float],
    dwi_geometry: Geometry,
    dwi_origin_xyz: Sequence[float],
) -> sitk.TranslationTransform:
    """Map centered DCE coordinates to centered DWI coordinates in one DICOM frame."""
    dce_origin = np.asarray(tuple(dce_origin_xyz), dtype=np.float64)
    dwi_origin = np.asarray(tuple(dwi_origin_xyz), dtype=np.float64)
    if dce_origin.shape != (3,) or dwi_origin.shape != (3,):
        raise ValueError("DICOM origins must contain three values")
    if not np.isfinite(dce_origin).all() or not np.isfinite(dwi_origin).all():
        raise ValueError("DICOM origins must be finite")
    offset = (
        dce_origin
        - np.asarray(centered_origin(dce_geometry))
        - dwi_origin
        + np.asarray(centered_origin(dwi_geometry))
    )
    return sitk.TranslationTransform(3, tuple(float(value) for value in offset))


def compose_source_grid_with_longitudinal(
    source_grid_transform: sitk.Transform,
    longitudinal_field: np.ndarray,
    target_geometry: Geometry,
) -> sitk.CompositeTransform:
    if source_grid_transform.GetDimension() != 3:
        raise ValueError("source grid transform must be 3D")
    longitudinal = displacement_transform(longitudinal_field, target_geometry)
    combined = sitk.CompositeTransform(3)
    combined.AddTransform(source_grid_transform)
    combined.AddTransform(longitudinal)
    return combined


def resample_multimodal_once(
    array_zyx: np.ndarray,
    source_geometry: Geometry,
    target_geometry: Geometry,
    source_grid_transform: sitk.Transform,
    longitudinal_field: np.ndarray,
    *,
    interpolation: Interpolation,
    default_value: float = 0.0,
) -> np.ndarray:
    combined = compose_source_grid_with_longitudinal(
        source_grid_transform, longitudinal_field, target_geometry
    )
    return resample_rigid(
        array_zyx,
        source_geometry,
        target_geometry,
        combined,
        interpolation=interpolation,
        default_value=default_value,
    )


def apply_transform(
    array_zyx: np.ndarray,
    source_geometry: Geometry,
    target_geometry: Geometry,
    transform_parameters,
    *,
    interpolation: Interpolation,
    default_value: float = 0.0,
) -> np.ndarray:
    source = np.asarray(array_zyx)
    if source.shape != source_geometry.shape_zyx:
        raise ValueError("source array shape does not match source geometry")
    moving_image = itk_image_from_array(_clean_float(source), source_geometry)
    configured = configure_transform_parameters(
        transform_parameters,
        target_geometry,
        interpolation=interpolation,
        default_value=default_value,
    )
    try:
        output_image = itk.transformix_filter(
            moving_image=moving_image,
            transform_parameter_object=configured,
            log_to_console=False,
            log_to_file=False,
        )
    except RuntimeError as exc:
        raise TransformApplicationError(f"Transformix failed: {exc}") from exc
    output = np.asarray(itk.array_from_image(output_image), dtype=np.float32)
    if output.shape != target_geometry.shape_zyx:
        raise TransformApplicationError(
            f"Transformix output shape {output.shape} does not match {target_geometry.shape_zyx}"
        )
    if not np.isfinite(output).all():
        raise TransformApplicationError("Transformix output contains nonfinite values")
    if interpolation == "nearest":
        rounded = np.rint(output)
        if np.any((rounded < 0) | (rounded > 255)):
            raise TransformApplicationError("nearest-neighbor labels exceed uint8 range")
        return rounded.astype(np.uint8)
    return output


def resample_rigid(
    array_zyx: np.ndarray,
    source_geometry: Geometry,
    target_geometry: Geometry,
    transform: sitk.Transform,
    *,
    interpolation: Interpolation,
    default_value: float = 0.0,
) -> np.ndarray:
    source = np.asarray(array_zyx)
    if source.shape != source_geometry.shape_zyx:
        raise ValueError("source array shape does not match source geometry")
    moving = sitk_image_from_array(_clean_float(source), source_geometry)
    reference = sitk_image_from_array(
        np.zeros(target_geometry.shape_zyx, dtype=np.float32), target_geometry
    )
    interpolator = sitk.sitkLinear if interpolation == "linear" else sitk.sitkNearestNeighbor
    try:
        registered = sitk.Resample(
            moving,
            reference,
            transform,
            interpolator,
            float(default_value),
            sitk.sitkFloat32,
        )
    except RuntimeError as exc:
        raise TransformApplicationError(f"rigid resampling failed: {exc}") from exc
    output = sitk.GetArrayFromImage(registered).astype(np.float32, copy=False)
    if not np.isfinite(output).all():
        raise TransformApplicationError("rigid output contains nonfinite values")
    if interpolation == "nearest":
        rounded = np.rint(output)
        if np.any((rounded < 0) | (rounded > 255)):
            raise TransformApplicationError("nearest-neighbor labels exceed uint8 range")
        return rounded.astype(np.uint8)
    return output


def resample_dce_phases_once(
    phases: Sequence[np.ndarray],
    phase_ids: Sequence[int],
    phase_transforms: Mapping[int, sitk.Transform],
    source_geometry: Geometry,
    target_geometry: Geometry,
    longitudinal_field: np.ndarray,
) -> tuple[np.ndarray, ...]:
    ids = tuple(int(value) for value in phase_ids)
    if len(ids) != len(phases) or not ids or ids[0] != 0 or len(set(ids)) != len(ids):
        raise ValueError("phase IDs must be unique, start at zero, and match DCE phases")
    expected = set(ids[1:])
    if set(phase_transforms) != expected:
        raise ValueError("phase transforms must match every non-reference DCE phase")
    outputs: list[np.ndarray] = []
    for phase_id, phase in zip(ids, phases, strict=True):
        combined = compose_phase_with_longitudinal(
            phase_transforms.get(phase_id),
            longitudinal_field,
            target_geometry,
        )
        outputs.append(
            resample_rigid(
                phase,
                source_geometry,
                target_geometry,
                combined,
                interpolation="linear",
            )
        )
    return tuple(outputs)


def displacement_and_jacobian(
    source_geometry: Geometry,
    target_geometry: Geometry,
    transform_parameters,
) -> tuple[np.ndarray, np.ndarray]:
    moving_image = itk_image_from_array(
        np.zeros(source_geometry.shape_zyx, dtype=np.float32), source_geometry
    )
    configured = configure_transform_parameters(
        transform_parameters,
        target_geometry,
        interpolation="linear",
    )
    try:
        with tempfile.TemporaryDirectory(prefix="ispy2-transformix-") as directory:
            field_image = itk.transformix_deformation_field(
                moving_image=moving_image,
                transform_parameter_object=configured,
                output_directory=directory,
                log_to_console=False,
                log_to_file=False,
            )
            field = np.asarray(itk.array_from_image(field_image), dtype=np.float64).copy()
    except RuntimeError as exc:
        raise TransformApplicationError(f"deformation field generation failed: {exc}") from exc
    expected_shape = (*target_geometry.shape_zyx, 3)
    if field.shape != expected_shape or not np.isfinite(field).all():
        raise TransformApplicationError(
            f"invalid deformation field shape or values: {field.shape}, expected {expected_shape}"
        )
    sitk_field = sitk.GetImageFromArray(field, isVector=True)
    sitk_field.SetSpacing(target_geometry.spacing_xyz)
    sitk_field.SetDirection(target_geometry.direction)
    sitk_field.SetOrigin(centered_origin(target_geometry))
    jacobian_image = sitk.DisplacementFieldJacobianDeterminant(sitk_field)
    jacobian = sitk.GetArrayFromImage(jacobian_image).astype(np.float32)
    return field.astype(np.float32), jacobian


def rigid_displacement_and_jacobian(
    target_geometry: Geometry,
    transform: sitk.Transform,
) -> tuple[np.ndarray, np.ndarray]:
    field_image = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        target_geometry.size_xyz,
        centered_origin(target_geometry),
        target_geometry.spacing_xyz,
        target_geometry.direction,
    )
    field = sitk.GetArrayFromImage(field_image).astype(np.float32)
    jacobian = sitk.GetArrayFromImage(
        sitk.DisplacementFieldJacobianDeterminant(field_image)
    ).astype(np.float32)
    return field, jacobian
