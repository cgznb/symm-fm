from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import nibabel as nib

from .geometry import direction_from_iop
from .models import Geometry, PatientVisits, VisitRecord


class DataValidationError(ValueError):
    pass


def _integer(mapping: Mapping[str, Any], key: str, context: str) -> int:
    try:
        return int(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(f"{context}: invalid {key}") from exc


def _require_equal(context: str, field: str, csv_value: Any, json_value: Any) -> None:
    if csv_value != json_value:
        raise DataValidationError(
            f"{context}: {field} differs between CSV ({csv_value!r}) and JSON "
            f"({json_value!r})"
        )


def _path(value: Any, context: str, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"{context}: missing {field}")
    return Path(value)


def _path_map(value: Any, context: str, field: str) -> dict[str, Path]:
    if not isinstance(value, Mapping) or not value:
        raise DataValidationError(f"{context}: {field} must be a nonempty object")
    return {str(key): _path(path, context, f"{field}.{key}") for key, path in value.items()}


def _validate_path_agreement(
    context: str,
    field: str,
    csv_value: Path | tuple[Path, ...] | Mapping[str, Path],
    json_value: Path | tuple[Path, ...] | Mapping[str, Path],
) -> None:
    if csv_value != json_value:
        raise DataValidationError(f"{context}: {field} differs between CSV and JSON")


def _phase_ids(metadata: Mapping[str, Any], count: int, context: str) -> tuple[int, ...]:
    values = metadata.get("dce_phase_ids")
    if not isinstance(values, list):
        raise DataValidationError(f"{context}: dce_phase_ids must be a JSON list")
    try:
        phase_ids = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{context}: invalid dce_phase_ids") from exc
    if (
        len(phase_ids) != count
        or not phase_ids
        or phase_ids[0] != 0
        or len(set(phase_ids)) != len(phase_ids)
        or any(value < 0 for value in phase_ids)
    ):
        raise DataValidationError(
            f"{context}: dce_phase_ids must be unique, start at zero, and match n_times"
        )
    return phase_ids


def _validate_image_headers(record: VisitRecord) -> None:
    expected_shape = record.geometry.shape_zyx
    dce_grid_paths = (
        *record.dce_paths,
        record.ftv_mask_path,
        record.raw_mask_path,
        record.ser_path,
        *record.pe_paths.values(),
    )
    for path in dce_grid_paths:
        if not path.is_file():
            raise DataValidationError(f"{record.patient_id}/{record.visit}: {path} does not exist")
        try:
            shape = tuple(int(value) for value in nib.load(path).shape)
        except Exception as exc:
            raise DataValidationError(
                f"{record.patient_id}/{record.visit}: cannot read NIfTI header {path}"
            ) from exc
        if shape != expected_shape:
            raise DataValidationError(
                f"{record.patient_id}/{record.visit}: {path} shape {shape} does not match "
                f"{expected_shape}"
            )
    if record.dwi_paths:
        if record.dwi_geometry is None:
            raise DataValidationError(
                f"{record.patient_id}/{record.visit}: DWI geometry is missing"
            )
        dwi_shape = record.dwi_geometry.shape_zyx
        dwi_grid_paths = (
            *record.dwi_paths,
            *((record.adc_path,) if record.adc_path is not None else ()),
            *((record.dwi_mask_path,) if record.dwi_mask_path is not None else ()),
        )
        for path in dwi_grid_paths:
            try:
                shape = tuple(int(value) for value in nib.load(path).shape)
            except Exception as exc:
                raise DataValidationError(
                    f"{record.patient_id}/{record.visit}: cannot read NIfTI header {path}"
                ) from exc
            if shape != dwi_shape:
                raise DataValidationError(
                    f"{record.patient_id}/{record.visit}: {path} shape {shape} does not match "
                    f"DWI geometry {dwi_shape}"
                )


def _optional_path(value: Any, context: str, field: str) -> Path | None:
    if value in (None, ""):
        return None
    return _path(value, context, field)


def _optional_dwi(
    row: Mapping[str, str], metadata: Mapping[str, Any], context: str
) -> dict[str, Any]:
    csv_paths = tuple(
        _path(value, context, "dwi_paths")
        for value in row.get("dwi_paths", "").split(";")
        if value
    )
    meta_values = metadata.get("dwi_paths", ())
    if not isinstance(meta_values, (list, tuple)):
        raise DataValidationError(f"{context}: dwi_paths must be a JSON list")
    meta_paths = tuple(_path(value, context, "dwi_paths") for value in meta_values)
    _validate_path_agreement(context, "dwi_paths", csv_paths, meta_paths)
    raw_b_values = row.get("dwi_b_values_json", "")
    try:
        csv_b_values = tuple(float(value) for value in json.loads(raw_b_values or "[]"))
        meta_b_values = tuple(float(value) for value in metadata.get("dwi_b_values", ()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"{context}: invalid DWI b-values") from exc
    if csv_b_values != meta_b_values or len(csv_paths) != len(csv_b_values):
        raise DataValidationError(f"{context}: DWI paths and b-values differ or have different counts")
    if csv_b_values and (
        tuple(sorted(csv_b_values)) != csv_b_values or len(set(csv_b_values)) != len(csv_b_values)
    ):
        raise DataValidationError(f"{context}: DWI b-values must be unique and sorted")
    adc_path = _optional_path(row.get("adc_path", ""), context, "adc_path")
    mask_path = _optional_path(row.get("dwi_mask_path", ""), context, "dwi_mask_path")
    _validate_path_agreement(
        context,
        "adc_path",
        adc_path,
        _optional_path(metadata.get("adc_path", ""), context, "adc_path"),
    )
    _validate_path_agreement(
        context,
        "dwi_mask_path",
        mask_path,
        _optional_path(metadata.get("dwi_mask_path", ""), context, "dwi_mask_path"),
    )
    dwi_payload = metadata.get("dwi")
    dwi_geometry = None
    dwi_origin = None
    dwi_frame = None
    if csv_paths:
        if not isinstance(dwi_payload, Mapping):
            raise DataValidationError(f"{context}: DWI metadata object is missing")
        registered_status = str(
            metadata.get(
                "dwi_registration_status", row.get("dwi_registration_status", "")
            )
        )
        native = (
            dwi_payload.get("registered_geometry")
            if registered_status.startswith("registered_")
            else dwi_payload.get("native_geometry")
        )
        if not isinstance(native, Mapping):
            raise DataValidationError(f"{context}: DWI native geometry is missing")
        try:
            dwi_geometry = Geometry(
                shape_zyx=tuple(int(value) for value in native["shape_zyx"]),
                spacing_xyz=tuple(float(value) for value in native["spacing_xyz"]),
                direction=tuple(float(value) for value in native["direction"]),
            )
            dwi_origin = tuple(float(value) for value in native["origin_xyz"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"{context}: invalid DWI native geometry") from exc
        if len(dwi_origin) != 3 or not all(math.isfinite(value) for value in dwi_origin):
            raise DataValidationError(f"{context}: invalid DWI origin")
        dwi_frame = str(
            dwi_payload.get("registered_frame_of_reference_uid", "")
            if registered_status.startswith("registered_")
            else dwi_payload.get("frame_of_reference_uid", "")
        ) or None
    source_group_value = row.get("dwi_source_group", "")
    try:
        source_group: int | str | None = int(source_group_value) if source_group_value else None
    except ValueError:
        source_group = source_group_value or None
    return {
        "dwi_paths": csv_paths,
        "dwi_b_values": csv_b_values,
        "adc_path": adc_path,
        "dwi_mask_path": mask_path,
        "dwi_source_group": source_group,
        "dwi_registration_status": str(
            metadata.get(
                "dwi_registration_status",
                row.get("dwi_registration_status", "not_available") or "not_available",
            )
        ),
        "dwi_geometry": dwi_geometry,
        "dwi_origin_xyz": dwi_origin,
        "dwi_frame_of_reference_uid": dwi_frame,
    }


def _record_from_row(
    row: Mapping[str, str], *, validate_files: bool, validate_nifti_shapes: bool
) -> VisitRecord:
    patient_id = row.get("patient_id", "")
    visit = row.get("visit", "")
    context = f"{patient_id or '<missing patient>'}/{visit or '<missing visit>'}"
    if not patient_id or not visit:
        raise DataValidationError(f"{context}: patient_id and visit are required")
    meta_path = _path(row.get("meta_path"), context, "meta_path")
    if not meta_path.is_file():
        raise DataValidationError(f"{context}: {meta_path} does not exist")
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"{context}: cannot read metadata {meta_path}") from exc
    if not isinstance(metadata, dict):
        raise DataValidationError(f"{context}: metadata must be a JSON object")

    _require_equal(context, "patient_id", patient_id, metadata.get("patient_id"))
    _require_equal(context, "visit", visit, metadata.get("visit"))
    for field in ("n_times", "n_slices", "rows", "cols", "ftv_voxel_count"):
        _require_equal(context, field, _integer(row, field, context), _integer(metadata, field, context))

    dce_paths = tuple(_path(value, context, "dce_paths") for value in row["dce_paths"].split(";") if value)
    meta_dce_paths = tuple(
        _path(value, context, "dce_paths") for value in metadata.get("dce_paths", ())
    )
    _validate_path_agreement(context, "dce_paths", dce_paths, meta_dce_paths)
    if len(dce_paths) != _integer(row, "n_times", context):
        raise DataValidationError(f"{context}: dce_paths count does not match n_times")
    _phase_ids(metadata, len(dce_paths), context)

    ftv_mask_path = _path(row.get("mask_path"), context, "mask_path")
    raw_mask_path = _path(row.get("raw_mask_path"), context, "raw_mask_path")
    ser_path = _path(row.get("ser_path"), context, "ser_path")
    _validate_path_agreement(
        context, "mask_path", ftv_mask_path, _path(metadata.get("mask_path"), context, "mask_path")
    )
    _validate_path_agreement(
        context,
        "raw_mask_path",
        raw_mask_path,
        _path(metadata.get("raw_mask_path"), context, "raw_mask_path"),
    )
    _validate_path_agreement(
        context, "ser_path", ser_path, _path(metadata.get("ser_path"), context, "ser_path")
    )

    try:
        csv_pe = json.loads(row.get("pe_paths_json", ""))
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"{context}: invalid pe_paths_json") from exc
    pe_paths = _path_map(csv_pe, context, "pe_paths")
    meta_pe_paths = _path_map(metadata.get("pe_paths"), context, "pe_paths")
    _validate_path_agreement(context, "pe_paths", pe_paths, meta_pe_paths)

    try:
        pixel_spacing = tuple(float(value) for value in metadata["pixel_spacing"])
        slice_spacing_value = metadata.get("spacing_between_slices")
        if slice_spacing_value is None:
            slice_spacing_value = metadata["slice_thickness"]
        slice_spacing = float(slice_spacing_value)
        iop = tuple(float(value) for value in metadata["image_orientation_patient"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(f"{context}: invalid geometry metadata") from exc
    if len(pixel_spacing) != 2:
        raise DataValidationError(f"{context}: pixel_spacing must contain row and column spacing")
    geometry = Geometry(
        shape_zyx=(
            _integer(row, "n_slices", context),
            _integer(row, "rows", context),
            _integer(row, "cols", context),
        ),
        spacing_xyz=(pixel_spacing[1], pixel_spacing[0], slice_spacing),
        direction=direction_from_iop(iop),
    )
    warnings = metadata.get("qc_warnings", [])
    if not isinstance(warnings, list):
        raise DataValidationError(f"{context}: qc_warnings must be a list")
    dwi = _optional_dwi(row, metadata, context)
    collection = str(metadata.get("collection", row.get("collection", "")))
    if row.get("collection", "") and collection != row.get("collection"):
        raise DataValidationError(f"{context}: collection differs between CSV and JSON")
    record = VisitRecord(
        patient_id=patient_id,
        visit=visit,
        n_times=_integer(row, "n_times", context),
        geometry=geometry,
        dce_paths=dce_paths,
        ftv_mask_path=ftv_mask_path,
        raw_mask_path=raw_mask_path,
        ser_path=ser_path,
        pe_paths=pe_paths,
        meta_path=meta_path,
        ftv_voxel_count=_integer(row, "ftv_voxel_count", context),
        qc_status=str(metadata.get("qc_status", row.get("qc_status", ""))),
        qc_warnings=tuple(str(value) for value in warnings),
        metadata=metadata,
        collection=collection,
        **dwi,
    )
    if validate_files:
        paths: Iterable[Path] = record.all_image_paths
        for path in paths:
            if not path.is_file():
                raise DataValidationError(f"{context}: {path} does not exist")
    if validate_files and validate_nifti_shapes:
        _validate_image_headers(record)
    return record


def load_manifest(
    manifest_path: Path | str,
    *,
    validate_files: bool = True,
    validate_nifti_shapes: bool = True,
) -> list[VisitRecord]:
    path = Path(manifest_path)
    if not path.is_file():
        raise DataValidationError(f"manifest does not exist: {path}")
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise DataValidationError(f"cannot read manifest: {path}") from exc
    if not rows:
        raise DataValidationError(f"manifest has no data rows: {path}")
    records = [
        _record_from_row(
            row,
            validate_files=validate_files,
            validate_nifti_shapes=validate_nifti_shapes,
        )
        for row in rows
    ]
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record.patient_id, record.visit)
        if key in seen:
            raise DataValidationError(f"duplicate manifest row: {record.patient_id}/{record.visit}")
        seen.add(key)
    return records


def group_by_patient(records: Iterable[VisitRecord]) -> dict[str, PatientVisits]:
    grouped: defaultdict[str, dict[str, VisitRecord]] = defaultdict(dict)
    for record in records:
        if record.visit in grouped[record.patient_id]:
            raise DataValidationError(f"duplicate visit: {record.patient_id}/{record.visit}")
        grouped[record.patient_id][record.visit] = record
    return {
        patient_id: PatientVisits(patient_id=patient_id, visits=dict(visits))
        for patient_id, visits in grouped.items()
    }
