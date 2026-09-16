"""Pixel-free DICOM discovery and geometry extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from statistics import median
from typing import Iterable, Iterator, Sequence

from .schema import Geometry, QCEvent


MR_IMAGE_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.4"
SEGMENTATION_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.66.4"

HEADER_TAGS = [
    "SOPClassUID",
    "SOPInstanceUID",
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "StudyDescription",
    "SeriesDescription",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "AcquisitionDateTime",
    "AcquisitionTime",
    "ContentTime",
    "ContrastBolusStartTime",
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "SeriesNumber",
    "InstanceNumber",
    "AcquisitionNumber",
    "TemporalPositionIdentifier",
    "NumberOfTemporalPositions",
    "TemporalPositionTimeOffset",
    "ImageType",
    "Rows",
    "Columns",
    "NumberOfFrames",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "BitsAllocated",
    "BitsStored",
    "PixelRepresentation",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "RescaleSlope",
    "RescaleIntercept",
]


class DicomDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class DicomHeader:
    path: Path
    patient_id: str
    study_uid: str
    series_uid: str
    sop_instance_uid: str
    sop_class_uid: str
    modality: str
    study_description: str
    series_description: str
    study_date: str | None
    series_number: str | None
    instance_number: int | None
    acquisition_number: int | None
    temporal_position: int | None
    number_of_temporal_positions: int | None
    acquisition_time: str | None
    acquisition_datetime: str | None
    contrast_bolus_start_time: str | None
    rows: int | None
    columns: int | None
    number_of_frames: int | None
    pixel_spacing: tuple[float, float] | None
    slice_thickness: float | None
    spacing_between_slices: float | None
    orientation_lps: tuple[float, float, float, float, float, float] | None
    position_lps: tuple[float, float, float] | None
    transfer_syntax_uid: str
    transfer_syntax_compressed: bool
    image_type: tuple[str, ...]
    rescale_slope: float | None
    rescale_intercept: float | None
    pixels_read: bool = False


@dataclass(frozen=True)
class SeriesSummary:
    path: Path
    collection: str
    patient_directory: str
    study_directory: str
    header: DicomHeader
    instance_count: int
    qc: tuple[QCEvent, ...] = ()


def _optional_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_tuple(value: object, length: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        result = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if len(result) == length else None


def read_dicom_header(path: str | Path) -> DicomHeader:
    """Read selected metadata and stop before Pixel Data.

    This function is the only DICOM reader used by the indexing layer. Keeping
    ``stop_before_pixels=True`` here makes metadata audits independent of pixel
    codecs and prevents accidental 132 GB pixel scans.
    """

    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - exercised without optional dep
        raise DicomDependencyError("pydicom is required for DICOM metadata audit") from exc

    source = Path(path)
    dataset = pydicom.dcmread(
        source,
        stop_before_pixels=True,
        specific_tags=HEADER_TAGS,
    )
    transfer_syntax = getattr(dataset.file_meta, "TransferSyntaxUID", "")
    image_type_value = getattr(dataset, "ImageType", ())
    if isinstance(image_type_value, str):
        image_type = (image_type_value,)
    else:
        image_type = tuple(str(item) for item in image_type_value)

    pixel_spacing = _float_tuple(getattr(dataset, "PixelSpacing", None), 2)
    orientation = _float_tuple(getattr(dataset, "ImageOrientationPatient", None), 6)
    position = _float_tuple(getattr(dataset, "ImagePositionPatient", None), 3)
    return DicomHeader(
        path=source,
        patient_id=str(getattr(dataset, "PatientID", "")),
        study_uid=str(getattr(dataset, "StudyInstanceUID", "")),
        series_uid=str(getattr(dataset, "SeriesInstanceUID", "")),
        sop_instance_uid=str(getattr(dataset, "SOPInstanceUID", "")),
        sop_class_uid=str(getattr(dataset, "SOPClassUID", "")),
        modality=str(getattr(dataset, "Modality", "")),
        study_description=str(getattr(dataset, "StudyDescription", "")),
        series_description=str(getattr(dataset, "SeriesDescription", "")),
        study_date=str(getattr(dataset, "StudyDate", "")) or None,
        series_number=str(getattr(dataset, "SeriesNumber", "")) or None,
        instance_number=_optional_int(getattr(dataset, "InstanceNumber", None)),
        acquisition_number=_optional_int(getattr(dataset, "AcquisitionNumber", None)),
        temporal_position=_optional_int(
            getattr(dataset, "TemporalPositionIdentifier", None)
        ),
        number_of_temporal_positions=_optional_int(
            getattr(dataset, "NumberOfTemporalPositions", None)
        ),
        acquisition_time=str(getattr(dataset, "AcquisitionTime", "")) or None,
        acquisition_datetime=str(getattr(dataset, "AcquisitionDateTime", "")) or None,
        contrast_bolus_start_time=str(
            getattr(dataset, "ContrastBolusStartTime", "")
        )
        or None,
        rows=_optional_int(getattr(dataset, "Rows", None)),
        columns=_optional_int(getattr(dataset, "Columns", None)),
        number_of_frames=_optional_int(getattr(dataset, "NumberOfFrames", None)),
        pixel_spacing=pixel_spacing,  # type: ignore[arg-type]
        slice_thickness=_optional_float(getattr(dataset, "SliceThickness", None)),
        spacing_between_slices=_optional_float(
            getattr(dataset, "SpacingBetweenSlices", None)
        ),
        orientation_lps=orientation,  # type: ignore[arg-type]
        position_lps=position,  # type: ignore[arg-type]
        transfer_syntax_uid=str(transfer_syntax),
        transfer_syntax_compressed=bool(
            getattr(transfer_syntax, "is_compressed", False)
        ),
        image_type=image_type,
        rescale_slope=_optional_float(getattr(dataset, "RescaleSlope", None)),
        rescale_intercept=_optional_float(
            getattr(dataset, "RescaleIntercept", None)
        ),
    )


def iter_dicom_files(series_path: str | Path) -> Iterator[Path]:
    with os.scandir(series_path) as entries:
        paths = [
            Path(entry.path)
            for entry in entries
            if entry.is_file() and entry.name.lower().endswith(".dcm")
        ]
    yield from sorted(paths)


def iter_series_directories(data_root: str | Path) -> Iterator[tuple[str, str, str, Path]]:
    """Yield collection, patient directory, study directory, and series path.

    TCIA downloads use ``collection/patient/study/series``. Administrative
    directories prefixed with an underscore are excluded explicitly.
    """

    root = Path(data_root).resolve()
    collection_dirs: list[Path]
    if root.name.upper() == "ISPY2":
        collection_dirs = [root]
    elif (root / "ISPY2").is_dir():
        collection_dirs = [root / "ISPY2"]
    else:
        collection_dirs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith("_")
        )

    for collection_dir in collection_dirs:
        for patient_dir in sorted(path for path in collection_dir.iterdir() if path.is_dir()):
            for study_dir in sorted(path for path in patient_dir.iterdir() if path.is_dir()):
                for series_dir in sorted(path for path in study_dir.iterdir() if path.is_dir()):
                    yield (
                        collection_dir.name,
                        patient_dir.name,
                        study_dir.name,
                        series_dir,
                    )


def summarize_series(
    collection: str,
    patient_directory: str,
    study_directory: str,
    series_path: str | Path,
) -> SeriesSummary:
    files = list(iter_dicom_files(series_path))
    if not files:
        raise ValueError(f"DICOM series is empty: {series_path}")
    header = read_dicom_header(files[0])
    qc: list[QCEvent] = []
    if header.patient_id and header.patient_id != patient_directory:
        qc.append(
            QCEvent(
                code="patient_directory_mismatch",
                severity="error",
                message="DICOM PatientID differs from the patient directory",
                details={
                    "directory": patient_directory,
                    "dicom_patient_id": header.patient_id,
                },
            )
        )
    return SeriesSummary(
        path=Path(series_path),
        collection=collection,
        patient_directory=patient_directory,
        study_directory=study_directory,
        header=header,
        instance_count=len(files),
        qc=tuple(qc),
    )


def read_series_headers(series_path: str | Path) -> list[DicomHeader]:
    return [read_dicom_header(path) for path in iter_dicom_files(series_path)]


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(left * right for left, right in zip(a, b))


def geometry_from_headers(headers: Iterable[DicomHeader]) -> Geometry | None:
    """Build an LPS affine after verifying one spatial phase's geometry."""

    items = list(headers)
    if not items:
        return None
    first = items[0]
    if (
        first.rows is None
        or first.columns is None
        or first.pixel_spacing is None
        or first.orientation_lps is None
        or first.position_lps is None
    ):
        return None

    def close(left: Sequence[float], right: Sequence[float], tolerance: float = 1e-5) -> bool:
        return len(left) == len(right) and all(
            abs(a - b) <= tolerance for a, b in zip(left, right)
        )

    for item in items:
        if (
            item.rows != first.rows
            or item.columns != first.columns
            or item.pixel_spacing is None
            or not close(item.pixel_spacing, first.pixel_spacing)
            or item.orientation_lps is None
            or not close(item.orientation_lps, first.orientation_lps)
            or item.position_lps is None
        ):
            return None

    orientation = first.orientation_lps
    column_step_direction = orientation[:3]
    row_step_direction = orientation[3:]
    normal = _cross(column_step_direction, row_step_direction)
    normal_norm = math.sqrt(_dot(normal, normal))
    if normal_norm <= 1e-8:
        return None
    normal = tuple(value / normal_norm for value in normal)

    positioned = items
    projections = sorted({_dot(item.position_lps, normal) for item in positioned})
    if len(projections) > 1:
        gaps = [
            abs(right - left)
            for left, right in zip(projections, projections[1:])
            if abs(right - left) > 1e-6
        ]
        slice_spacing = median(gaps) if gaps else None
        if slice_spacing is not None and any(
            abs(gap - slice_spacing) > max(1e-3, 0.05 * slice_spacing)
            for gap in gaps
        ):
            return None
    else:
        slice_spacing = None
    slice_spacing = (
        slice_spacing
        or first.spacing_between_slices
        or first.slice_thickness
    )
    if slice_spacing is None or slice_spacing <= 0:
        return None

    origin_header = min(
        positioned,
        key=lambda item: _dot(item.position_lps, normal),  # type: ignore[arg-type]
    )
    origin = origin_header.position_lps
    assert origin is not None
    row_spacing, column_spacing = first.pixel_spacing
    d_step = tuple(value * slice_spacing for value in normal)
    h_step = tuple(value * row_spacing for value in row_step_direction)
    w_step = tuple(value * column_spacing for value in column_step_direction)
    affine = (
        (d_step[0], h_step[0], w_step[0], origin[0]),
        (d_step[1], h_step[1], w_step[1], origin[1]),
        (d_step[2], h_step[2], w_step[2], origin[2]),
        (0.0, 0.0, 0.0, 1.0),
    )
    return Geometry(
        shape_dhw=(len(projections) or len(items), first.rows, first.columns),
        spacing_dhw=(slice_spacing, row_spacing, column_spacing),
        orientation_lps=orientation,
        origin_lps=origin,
        affine_lps=affine,
    )


def geometries_match(
    left: Geometry | None,
    right: Geometry | None,
    *,
    tolerance: float = 1e-4,
) -> bool:
    if left is None or right is None:
        return False
    if left.shape_dhw != right.shape_dhw:
        return False
    numeric_pairs = zip(
        (*left.spacing_dhw, *left.orientation_lps, *left.origin_lps),
        (*right.spacing_dhw, *right.orientation_lps, *right.origin_lps),
    )
    return all(abs(a - b) <= tolerance for a, b in numeric_pairs)
