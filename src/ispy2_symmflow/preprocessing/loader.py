"""Load selected DCE phases into a channel-first 3D array."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ispy2_symmflow.data.dicom import (
    DicomHeader,
    geometries_match,
    geometry_from_headers,
    read_series_headers,
)
from ispy2_symmflow.data.schema import Geometry, PhaseRef, VisitRecord


@dataclass(frozen=True)
class LoadedVisit:
    image: NDArray[np.float32]
    affine_lps: NDArray[np.float64]
    spacing_dhw: tuple[float, float, float]
    phase_roles: tuple[str, ...]
    source_series_uids: tuple[str, ...]
    source_temporal_positions: tuple[int, ...]


def _normal(header: DicomHeader) -> NDArray[np.float64]:
    if header.orientation_lps is None:
        raise ValueError(f"missing ImageOrientationPatient: {header.path}")
    orientation = np.asarray(header.orientation_lps, dtype=np.float64)
    normal = np.cross(orientation[:3], orientation[3:])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-8:
        raise ValueError(f"degenerate ImageOrientationPatient: {header.path}")
    return normal / norm


def _ordered_headers(headers: Sequence[DicomHeader]) -> list[DicomHeader]:
    if not headers:
        raise ValueError("phase contains no DICOM instances")
    normal = _normal(headers[0])
    if any(header.position_lps is None for header in headers):
        raise ValueError("phase instances need ImagePositionPatient")
    ordered = sorted(
        headers,
        key=lambda header: float(np.dot(np.asarray(header.position_lps), normal)),
    )
    positions = [tuple(header.position_lps or ()) for header in ordered]
    if len(set(positions)) != len(positions):
        raise ValueError("phase contains duplicate spatial positions")
    return ordered


def _load_phase(headers: Sequence[DicomHeader]) -> tuple[NDArray[np.float32], Geometry]:
    ordered = _ordered_headers(headers)
    geometry = geometry_from_headers(ordered)
    if geometry is None:
        raise ValueError("cannot construct complete DICOM patient geometry")
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - base dependency in package
        raise RuntimeError("pydicom is required to prepare DICOM pixels") from exc

    slices: list[NDArray[np.float32]] = []
    for header in ordered:
        dataset = pydicom.dcmread(header.path)
        pixels = np.asarray(dataset.pixel_array)
        if pixels.ndim != 2:
            raise ValueError(
                f"classic MR phase expects one 2D frame per instance, got {pixels.shape}"
            )
        slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
        slices.append(np.asarray(pixels, dtype=np.float32) * slope + intercept)
    return np.stack(slices), geometry


def load_visit_phases(
    visit: VisitRecord,
    phase_roles: Sequence[str] = ("pre", "early", "late"),
) -> LoadedVisit:
    """Decode only explicitly selected phases after metadata QC has passed."""

    roles = tuple(str(role) for role in phase_roles)
    missing = [role for role in roles if role not in visit.phase_paths]
    if missing:
        raise ValueError(f"visit {visit.visit_id} lacks phase roles {missing}")
    headers_by_path: dict[str, list[DicomHeader]] = {}
    volumes: list[NDArray[np.float32]] = []
    geometries: list[Geometry] = []
    for role in roles:
        reference = visit.phase_paths[role]
        if reference.series_path not in headers_by_path:
            headers_by_path[reference.series_path] = read_series_headers(
                reference.series_path
            )
        headers = headers_by_path[reference.series_path]
        selected = [
            header
            for header in headers
            if reference.temporal_position is None
            or header.temporal_position == reference.temporal_position
        ]
        if len(selected) != reference.instance_count:
            raise ValueError(
                f"phase {role} instance count changed: manifest={reference.instance_count}, disk={len(selected)}"
            )
        volume, geometry = _load_phase(selected)
        if reference.geometry is not None and not geometries_match(
            reference.geometry, geometry
        ):
            raise ValueError(f"phase {role} geometry changed since manifest creation")
        volumes.append(volume)
        geometries.append(geometry)
    reference_geometry = geometries[0]
    if any(
        not geometries_match(reference_geometry, geometry)
        for geometry in geometries[1:]
    ):
        raise ValueError("selected DCE phase geometries are not compatible")
    image = np.stack(volumes).astype(np.float32, copy=False)
    return LoadedVisit(
        image=image,
        affine_lps=np.asarray(reference_geometry.affine_lps, dtype=np.float64),
        spacing_dhw=reference_geometry.spacing_dhw,
        phase_roles=roles,
        source_series_uids=tuple(visit.phase_paths[role].series_uid for role in roles),
        source_temporal_positions=tuple(
            visit.phase_paths[role].temporal_position
            if visit.phase_paths[role].temporal_position is not None
            else -1
            for role in roles
        ),
    )
