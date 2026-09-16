from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Geometry:
    shape_zyx: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    direction: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.shape_zyx) != 3 or any(int(value) <= 0 for value in self.shape_zyx):
            raise ValueError("shape_zyx must contain three positive dimensions")
        if len(self.spacing_xyz) != 3 or any(
            not np.isfinite(value) or float(value) <= 0 for value in self.spacing_xyz
        ):
            raise ValueError("spacing_xyz must contain three positive finite values")
        if len(self.direction) != 9:
            raise ValueError("direction must contain nine values")
        matrix = np.asarray(self.direction, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(matrix)) or not np.allclose(
            matrix.T @ matrix, np.eye(3), atol=1e-5
        ):
            raise ValueError("direction must be an orthonormal 3x3 matrix")

    @property
    def size_xyz(self) -> tuple[int, int, int]:
        return tuple(reversed(self.shape_zyx))

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing_xyz))

    def with_shape(self, shape_zyx: tuple[int, int, int]) -> Geometry:
        return Geometry(
            shape_zyx=shape_zyx,
            spacing_xyz=self.spacing_xyz,
            direction=self.direction,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "shape_zyx": list(self.shape_zyx),
            "spacing_xyz": list(self.spacing_xyz),
            "direction": list(self.direction),
        }


@dataclass(frozen=True)
class VisitRecord:
    patient_id: str
    visit: str
    n_times: int
    geometry: Geometry
    dce_paths: tuple[Path, ...]
    ftv_mask_path: Path
    raw_mask_path: Path
    ser_path: Path
    pe_paths: Mapping[str, Path]
    meta_path: Path
    ftv_voxel_count: int
    qc_status: str
    qc_warnings: tuple[str, ...]
    metadata: Mapping[str, Any] = field(repr=False)
    collection: str = ""
    dwi_paths: tuple[Path, ...] = ()
    dwi_b_values: tuple[float, ...] = ()
    adc_path: Path | None = None
    dwi_mask_path: Path | None = None
    dwi_source_group: int | str | None = None
    dwi_registration_status: str = "not_available"
    dwi_geometry: Geometry | None = None
    dwi_origin_xyz: tuple[float, float, float] | None = None
    dwi_frame_of_reference_uid: str | None = None

    @property
    def dce_phase_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.metadata["dce_phase_ids"])

    @property
    def all_image_paths(self) -> tuple[Path, ...]:
        return (
            *self.dce_paths,
            self.ftv_mask_path,
            self.raw_mask_path,
            self.ser_path,
            *self.pe_paths.values(),
            *self.dwi_paths,
            *((self.adc_path,) if self.adc_path is not None else ()),
            *((self.dwi_mask_path,) if self.dwi_mask_path is not None else ()),
        )


@dataclass(frozen=True)
class PatientVisits:
    patient_id: str
    visits: Mapping[str, VisitRecord]

    def get(self, visit: str) -> VisitRecord | None:
        return self.visits.get(visit)
