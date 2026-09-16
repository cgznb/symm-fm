from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_release import settings as _release_settings

import matplotlib

matplotlib.use("Agg")

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from .config import RegistrationConfig
from .io import atomic_output_directory, read_nifti
from .manifest import group_by_patient
from .models import Geometry, VisitRecord
from .preprocess import prepare_pair_drivers
from .qc import QCInputs, protected_roi_mask
from .transforms import apply_transform, displacement_and_jacobian, resample_rigid


FIGURE_SIZE = (16.0, 12.0)
FIGURE_DPI = 150
DISPLAY_AXES = ((0, "Axial"), (1, "Coronal"), (2, "Sagittal"))


class Category(str, Enum):
    PROTECTED_FOLD = "protected_roi_folding"
    OUTSIDE_FOLD = "outside_roi_folding"
    PROTECTED_NCC = "protected_roi_similarity_regressed"
    NON_TUMOR_NCC = "non_tumor_similarity_regressed"
    BSPLINE_FAILED = "bspline_failed"
    JACOBIAN_NONFINITE = "jacobian_nonfinite"


@dataclass(frozen=True)
class StatusRecord:
    patient_id: str
    visit: str
    status_path: Path
    reasons: tuple[str, ...]
    qc: Mapping[str, Any]


@dataclass(frozen=True)
class ExampleSlot:
    ordinal: int
    category: Category
    patient_id: str
    visit: str
    status_path: Path
    reasons: tuple[str, ...]
    secondary_view: bool = False

    @property
    def filename(self) -> str:
        suffix = "_secondary_view" if self.secondary_view else ""
        return (
            f"{self.ordinal:02d}_{self.category.value}_"
            f"{self.patient_id}_{self.visit}{suffix}.png"
        )


@dataclass(frozen=True)
class ElastixMetricFailure:
    resolution: int
    outside_samples: int
    total_samples: int


@dataclass(frozen=True)
class DiagnosticBundle:
    fixed: np.ndarray
    rigid: np.ndarray
    candidate: np.ndarray | None
    fixed_ftv: np.ndarray
    rigid_ftv: np.ndarray
    candidate_ftv: np.ndarray | None
    candidate_jacobian: np.ndarray | None
    protected_roi: np.ndarray
    valid_fov: np.ndarray
    fixed_geometry: Geometry


@dataclass(frozen=True)
class GenerationSummary:
    destination: Path
    files: tuple[Path, ...]
    categories: Mapping[str, int]
    pairs_by_category: Mapping[str, tuple[str, ...]]


_CATEGORY_REASON = {
    Category.PROTECTED_FOLD: "jacobian_nonpositive_protected_roi",
    Category.OUTSIDE_FOLD: "outside_protected_roi_fold_fraction",
    Category.PROTECTED_NCC: "protected_roi_similarity_regressed",
    Category.NON_TUMOR_NCC: "non_tumor_similarity_regressed",
    Category.BSPLINE_FAILED: "bspline_failed",
    Category.JACOBIAN_NONFINITE: "jacobian_nonfinite",
}


def parse_elastix_metric_failure(text: str) -> ElastixMetricFailure:
    resolutions = re.findall(r"resolution\s+(\d+)", text, flags=re.IGNORECASE)
    counts = re.findall(
        r"Too many samples map outside moving image buffer:\s*(\d+)\s*/\s*(\d+)",
        text,
    )
    if not counts:
        raise ValueError("elastix log lacks moving-buffer metric error")
    outside, total = counts[-1]
    return ElastixMetricFailure(
        resolution=int(resolutions[-1]) if resolutions else 0,
        outside_samples=int(outside),
        total_samples=int(total),
    )


def load_status_records(output_root: Path | str) -> tuple[StatusRecord, ...]:
    root = Path(output_root)
    records: list[StatusRecord] = []
    for status_path in sorted(root.glob("ISPY2-*/*/registration.json")):
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        if payload.get("registration_status") != "rigid_fallback":
            continue
        qc = payload.get("qc")
        if not isinstance(qc, dict):
            raise ValueError(f"missing QC payload: {status_path}")
        reason_values = qc.get("reasons")
        if not isinstance(reason_values, list) or not all(
            isinstance(value, str) for value in reason_values
        ):
            raise ValueError(f"invalid QC reasons: {status_path}")
        records.append(
            StatusRecord(
                patient_id=str(payload["patient_id"]),
                visit=str(payload["visit"]),
                status_path=status_path,
                reasons=tuple(reason_values),
                qc=qc,
            )
        )
    return tuple(records)


def _metric(record: StatusRecord, category: Category) -> float | None:
    qc = record.qc
    if category is Category.PROTECTED_FOLD:
        value = qc.get("protected_roi_fold_fraction")
    elif category is Category.OUTSIDE_FOLD:
        value = qc.get("outside_protected_roi_fold_fraction")
    elif category is Category.PROTECTED_NCC:
        rigid = qc.get("rigid_protected_ncc")
        deformable = qc.get("deformable_protected_ncc")
        value = None if rigid is None or deformable is None else float(rigid) - float(deformable)
    elif category is Category.NON_TUMOR_NCC:
        rigid = qc.get("rigid_non_tumor_ncc")
        deformable = qc.get("deformable_non_tumor_ncc")
        value = None if rigid is None or deformable is None else float(rigid) - float(deformable)
    else:
        return None
    if value is None:
        return None
    metric = float(value)
    return metric if math.isfinite(metric) else None


def rank_category_records(
    records: Sequence[StatusRecord],
    category: Category,
    *,
    count: int,
) -> tuple[StatusRecord, ...]:
    reason = _CATEGORY_REASON[category]
    eligible = [
        record
        for record in records
        if reason in record.reasons and _metric(record, category) is not None
    ]
    eligible.sort(
        key=lambda record: (
            -float(_metric(record, category)),
            len([value for value in record.reasons if value != reason]),
            record.patient_id,
            record.visit,
        )
    )
    if len(eligible) < count:
        raise ValueError(
            f"{category.value} has {len(eligible)} eligible pairs; expected {count}"
        )
    return tuple(eligible[:count])


def _required_record(
    records: Sequence[StatusRecord],
    category: Category,
    patient_id: str,
    visit: str,
) -> StatusRecord:
    reason = _CATEGORY_REASON[category]
    for record in records:
        if record.patient_id == patient_id and record.visit == visit:
            if reason not in record.reasons:
                raise ValueError(f"{patient_id}/{visit} lacks {reason}")
            return record
    raise ValueError(f"missing required pair: {patient_id}/{visit}")


def _slots(
    category: Category,
    records: Sequence[StatusRecord],
    start: int,
) -> list[ExampleSlot]:
    return [
        ExampleSlot(
            ordinal=start + index,
            category=category,
            patient_id=record.patient_id,
            visit=record.visit,
            status_path=record.status_path,
            reasons=record.reasons,
        )
        for index, record in enumerate(records)
    ]


def select_example_slots(output_root: Path | str) -> tuple[ExampleSlot, ...]:
    examples = _release_settings().get("registration_examples", {})
    for name, count in (("bspline_failed", 3), ("jacobian_nonfinite", 2)):
        pairs = examples.get(name, [])
        if not isinstance(pairs, list) or len(pairs) != count or any(
            not isinstance(pair, (list, tuple)) or len(pair) != 2
            or not all(isinstance(value, str) and value.strip() for value in pair)
            for pair in pairs
        ):
            raise ValueError(
                f"Set registration_examples.{name} in paths.local.yaml "
                f"to {count} private [patient_id, visit] pairs"
            )
    records = load_status_records(output_root)
    slots: list[ExampleSlot] = []
    for category in (
        Category.PROTECTED_FOLD,
        Category.OUTSIDE_FOLD,
        Category.PROTECTED_NCC,
        Category.NON_TUMOR_NCC,
    ):
        slots.extend(
            _slots(category, rank_category_records(records, category, count=3), len(slots) + 1)
        )

    bspline_records = tuple(
        _required_record(records, Category.BSPLINE_FAILED, patient_id, visit)
        for patient_id, visit in examples["bspline_failed"]
    )
    slots.extend(_slots(Category.BSPLINE_FAILED, bspline_records, len(slots) + 1))

    nonfinite_records = tuple(
        _required_record(records, Category.JACOBIAN_NONFINITE, patient_id, visit)
        for patient_id, visit in examples["jacobian_nonfinite"]
    )
    slots.extend(_slots(Category.JACOBIAN_NONFINITE, nonfinite_records, len(slots) + 1))
    primary = nonfinite_records[0]
    slots.append(
        ExampleSlot(
            ordinal=len(slots) + 1,
            category=Category.JACOBIAN_NONFINITE,
            patient_id=primary.patient_id,
            visit=primary.visit,
            status_path=primary.status_path,
            reasons=primary.reasons,
            secondary_view=True,
        )
    )
    if len(slots) != 18:
        raise AssertionError(f"expected 18 example slots, found {len(slots)}")
    return tuple(slots)


def validate_bundle_shapes(bundle: DiagnosticBundle) -> None:
    expected = bundle.fixed_geometry.shape_zyx
    required = {
        "fixed": bundle.fixed,
        "rigid": bundle.rigid,
        "fixed FTV": bundle.fixed_ftv,
        "rigid FTV": bundle.rigid_ftv,
        "protected ROI": bundle.protected_roi,
        "valid FOV": bundle.valid_fov,
    }
    optional = {
        "candidate": bundle.candidate,
        "candidate FTV": bundle.candidate_ftv,
        "candidate Jacobian": bundle.candidate_jacobian,
    }
    for name, array in required.items():
        if np.asarray(array).shape != expected:
            raise ValueError(f"{name} shape differs from T0: {np.asarray(array).shape}")
    for name, array in optional.items():
        if array is not None and np.asarray(array).shape != expected:
            raise ValueError(f"{name} shape differs from T0: {np.asarray(array).shape}")
    present = tuple(value is not None for value in optional.values())
    if len(set(present)) != 1:
        raise ValueError("candidate image, FTV, and Jacobian must be present together")


def abnormal_slice_indices(
    region: np.ndarray,
    *,
    exclude: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    mask = np.asarray(region, dtype=bool)
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError("abnormal region must be a nonempty 3D mask")
    coordinates = np.argwhere(mask)
    if exclude is None:
        from scipy import ndimage

        distance = ndimage.distance_transform_edt(mask)
        selected = np.asarray(np.unravel_index(int(np.argmax(distance)), mask.shape))
    else:
        center = np.asarray(exclude, dtype=np.float64)
        distances = np.sum((coordinates - center) ** 2, axis=1)
        selected = coordinates[int(np.argmax(distances))]
    return tuple(int(value) for value in selected)


def similarity_slice_indices(
    fixed: np.ndarray,
    rigid: np.ndarray,
    candidate: np.ndarray,
    region: np.ndarray,
) -> tuple[int, int, int]:
    fixed_array = np.asarray(fixed, dtype=np.float64)
    rigid_array = np.asarray(rigid, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    mask = np.asarray(region, dtype=bool)
    if (
        fixed_array.shape != rigid_array.shape
        or fixed_array.shape != candidate_array.shape
        or fixed_array.shape != mask.shape
    ):
        raise ValueError("similarity arrays and region must share one shape")
    if fixed_array.ndim != 3 or not np.any(mask):
        raise ValueError("similarity region must be a nonempty 3D mask")
    increase = np.abs(fixed_array - candidate_array) - np.abs(fixed_array - rigid_array)
    increase[~np.isfinite(increase) | ~mask] = -np.inf
    if not np.isfinite(increase).any():
        raise ValueError("similarity region contains no finite residuals")
    return tuple(
        int(value)
        for value in np.unravel_index(int(np.argmax(increase)), increase.shape)
    )


def load_parameter_object(pair_dir: Path | str):
    import itk

    pair = Path(pair_dir)
    initial = pair / "transforms" / "elastix" / "InitialTransformParameters.0.txt"
    bspline = pair / "transforms" / "elastix" / "TransformParameters.0.txt"
    if not initial.is_file() or not bspline.is_file():
        raise FileNotFoundError(f"missing retained transform files: {pair}")
    parameters = itk.ParameterObject.New()
    parameters.ReadParameterFiles([str(initial), str(bspline)])
    if parameters.GetNumberOfParameterMaps() != 2:
        raise ValueError(f"expected rigid plus B-spline maps: {pair}")
    return parameters


def load_aligned_phases(
    record: VisitRecord,
    pair_dir: Path | str,
) -> tuple[np.ndarray, ...]:
    pair = Path(pair_dir)
    phases = tuple(read_nifti(path, dtype=np.float32) for path in record.dce_paths)
    alignment_path = pair / "transforms" / "intravisit" / "phase_alignment.json"
    payload = json.loads(alignment_path.read_text(encoding="utf-8"))
    entries = {
        int(item["phase_id"]): item
        for item in payload.get("registrations", [])
        if isinstance(item, dict) and "phase_id" in item
    }
    aligned: list[np.ndarray] = [phases[0]]
    for phase_id, phase in zip(record.dce_phase_ids[1:], phases[1:], strict=True):
        if phase_id not in entries:
            raise ValueError(f"missing phase {phase_id} alignment: {alignment_path}")
        transform_path = Path(entries[phase_id]["transform_path"])
        transform = sitk.ReadTransform(str(transform_path))
        aligned.append(
            resample_rigid(
                phase,
                record.geometry,
                record.geometry,
                transform,
                interpolation="linear",
            )
        )
    return tuple(aligned)


def reconstruct_candidate_bundle(
    fixed: VisitRecord,
    moving: VisitRecord,
    output_root: Path | str,
    config: RegistrationConfig,
    *,
    include_candidate: bool,
) -> DiagnosticBundle:
    if fixed.patient_id != moving.patient_id:
        raise ValueError("fixed and moving records must belong to one patient")
    root = Path(output_root)
    fixed_dir = root / fixed.patient_id / fixed.visit
    moving_dir = root / moving.patient_id / moving.visit
    fixed_ftv = (read_nifti(fixed.ftv_mask_path) > 0).astype(np.uint8)
    moving_ftv = (read_nifti(moving.ftv_mask_path) > 0).astype(np.uint8)
    drivers = prepare_pair_drivers(
        load_aligned_phases(fixed, fixed_dir),
        load_aligned_phases(moving, moving_dir),
        fixed_ftv,
        moving_ftv,
        weight=config.tumor_weight,
    )
    fixed_anatomical = drivers.fixed_anatomical[1:-1]
    moving_anatomical = drivers.moving_anatomical[1:-1]
    rigid_transform = sitk.ReadTransform(str(moving_dir / "transforms" / "rigid.tfm"))
    rigid_anatomical = resample_rigid(
        moving_anatomical,
        moving.geometry,
        fixed.geometry,
        rigid_transform,
        interpolation="linear",
    )
    rigid_ftv = resample_rigid(
        moving_ftv,
        moving.geometry,
        fixed.geometry,
        rigid_transform,
        interpolation="nearest",
    )
    valid_fov = resample_rigid(
        np.ones(moving.geometry.shape_zyx, dtype=np.uint8),
        moving.geometry,
        fixed.geometry,
        rigid_transform,
        interpolation="nearest",
    ).astype(bool)

    candidate_anatomical: np.ndarray | None = None
    candidate_ftv: np.ndarray | None = None
    candidate_jacobian: np.ndarray | None = None
    if include_candidate:
        parameters = load_parameter_object(moving_dir)
        candidate_anatomical = apply_transform(
            moving_anatomical,
            moving.geometry,
            fixed.geometry,
            parameters,
            interpolation="linear",
        )
        candidate_ftv = apply_transform(
            moving_ftv,
            moving.geometry,
            fixed.geometry,
            parameters,
            interpolation="nearest",
        )
        _, candidate_jacobian = displacement_and_jacobian(
            moving.geometry,
            fixed.geometry,
            parameters,
        )

    qc_candidate = candidate_anatomical if candidate_anatomical is not None else rigid_anatomical
    qc_candidate_ftv = candidate_ftv if candidate_ftv is not None else rigid_ftv
    qc_jacobian = (
        candidate_jacobian
        if candidate_jacobian is not None
        else np.ones(fixed.geometry.shape_zyx, dtype=np.float32)
    )
    inputs = QCInputs(
        fixed_image=fixed_anatomical,
        rigid_image=rigid_anatomical,
        deformable_image=qc_candidate,
        fixed_ftv=fixed_ftv,
        rigid_ftv=rigid_ftv,
        deformable_ftv=qc_candidate_ftv,
        source_moving_ftv=moving_ftv,
        source_geometry=moving.geometry,
        target_geometry=fixed.geometry,
        jacobian=qc_jacobian,
    )
    protected = protected_roi_mask(inputs, config.protected_margin_mm)
    bundle = DiagnosticBundle(
        fixed=fixed_anatomical,
        rigid=rigid_anatomical,
        candidate=candidate_anatomical,
        fixed_ftv=fixed_ftv,
        rigid_ftv=rigid_ftv,
        candidate_ftv=candidate_ftv,
        candidate_jacobian=candidate_jacobian,
        protected_roi=protected,
        valid_fov=valid_fov,
        fixed_geometry=fixed.geometry,
    )
    validate_bundle_shapes(bundle)
    return bundle


def display_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    values = np.asarray(array)
    if axis == 0:
        return values[index]
    if axis == 1:
        return values[:, index, :]
    if axis == 2:
        return values[:, :, index]
    raise ValueError(f"invalid display axis: {axis}")


def physical_aspect(geometry: Geometry, axis: int) -> float:
    spacing_x, spacing_y, spacing_z = geometry.spacing_xyz
    if axis == 0:
        return float(spacing_y / spacing_x)
    if axis == 1:
        return float(spacing_z / spacing_x)
    if axis == 2:
        return float(spacing_z / spacing_y)
    raise ValueError(f"invalid display axis: {axis}")


def robust_window(*images: np.ndarray) -> tuple[float, float]:
    finite = [
        np.asarray(image)[np.isfinite(image)]
        for image in images
        if np.asarray(image)[np.isfinite(image)].size
    ]
    if not finite:
        return 0.0, 1.0
    values = np.concatenate(finite)
    low, high = np.percentile(values, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _normalized(array: np.ndarray, low: float, high: float) -> np.ndarray:
    clean = np.asarray(array, dtype=np.float32).copy()
    clean[~np.isfinite(clean)] = low
    return np.clip((clean - low) / (high - low), 0.0, 1.0)


def _red_cyan_overlay(
    reference: np.ndarray,
    comparison: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    red = _normalized(reference, low, high)
    cyan = _normalized(comparison, low, high)
    return np.stack((red, cyan, cyan), axis=-1)


def _contour(
    panel,
    mask: np.ndarray,
    *,
    color: str,
    linewidth: float = 0.8,
) -> None:
    values = np.asarray(mask, dtype=bool)
    if np.any(values) and np.any(~values):
        panel.contour(
            values.astype(np.uint8),
            levels=[0.5],
            colors=[color],
            linewidths=linewidth,
            origin="lower",
        )


def _header(slot: ExampleSlot, detail: str) -> str:
    reasons = ", ".join(slot.reasons)
    view = " | secondary view" if slot.secondary_view else ""
    return (
        f"{slot.patient_id}/{slot.visit} | {slot.category.value}{view}\n"
        f"QC reasons: {reasons}\n{detail}"
    )


def _require_candidate(bundle: DiagnosticBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        bundle.candidate is None
        or bundle.candidate_ftv is None
        or bundle.candidate_jacobian is None
    ):
        raise ValueError("category requires a reconstructed B-spline candidate")
    return bundle.candidate, bundle.candidate_ftv, bundle.candidate_jacobian


def _image_panel(
    panel,
    image: np.ndarray,
    *,
    low: float,
    high: float,
    title: str,
    row: int,
    region: np.ndarray | None = None,
) -> None:
    panel.imshow(image, cmap="gray", vmin=low, vmax=high, origin="lower")
    if region is not None:
        _contour(panel, region, color="#ffd60a")
    if row == 0:
        panel.set_title(title, fontsize=10)
    panel.set_axis_off()


def _render_folding(
    slot: ExampleSlot,
    bundle: DiagnosticBundle,
    destination: Path,
    qc: Mapping[str, Any],
) -> None:
    candidate, _, jacobian = _require_candidate(bundle)
    finite = np.isfinite(jacobian)
    folding = finite & (jacobian <= 0)
    if slot.category is Category.PROTECTED_FOLD:
        abnormal = folding & bundle.protected_roi
        metric = float(qc["protected_roi_fold_fraction"])
        detail = f"Protected fold fraction={metric:.6f} | threshold=0"
    elif slot.category is Category.OUTSIDE_FOLD:
        abnormal = folding & ~bundle.protected_roi
        metric = float(qc["outside_protected_roi_fold_fraction"])
        detail = f"Outside fold fraction={metric:.6f} | threshold=0.020000"
    elif slot.category is Category.JACOBIAN_NONFINITE:
        abnormal = ~finite
        finite_values = jacobian[finite]
        minimum = float(np.min(finite_values)) if finite_values.size else float("nan")
        maximum = float(np.max(finite_values)) if finite_values.size else float("nan")
        detail = (
            f"Nonfinite voxels={int(np.count_nonzero(abnormal))} | "
            f"finite range=[{minimum:.4g}, {maximum:.4g}]"
        )
    else:
        raise ValueError(f"unsupported folding category: {slot.category.value}")
    primary = abnormal_slice_indices(abnormal)
    indices = abnormal_slice_indices(abnormal, exclude=primary) if slot.secondary_view else primary
    low, high = robust_window(bundle.fixed, bundle.rigid, candidate)
    figure, axes = plt.subplots(
        3,
        5,
        figsize=FIGURE_SIZE,
        dpi=FIGURE_DPI,
        constrained_layout=True,
    )
    figure.suptitle(_header(slot, detail), fontsize=11)
    for row, (axis, axis_name) in enumerate(DISPLAY_AXES):
        index = indices[axis]
        fixed_slice = display_slice(bundle.fixed, axis, index)
        rigid_slice = display_slice(bundle.rigid, axis, index)
        candidate_slice = display_slice(candidate, axis, index)
        protected_slice = display_slice(bundle.protected_roi, axis, index)
        abnormal_slice = display_slice(abnormal, axis, index)
        jacobian_slice = display_slice(jacobian, axis, index)
        _image_panel(
            axes[row, 0], fixed_slice, low=low, high=high, title="T0", row=row
        )
        _image_panel(
            axes[row, 1], rigid_slice, low=low, high=high, title="Rigid", row=row
        )
        _image_panel(
            axes[row, 2],
            candidate_slice,
            low=low,
            high=high,
            title="B-spline candidate",
            row=row,
        )
        jacobian_panel = axes[row, 3]
        jacobian_panel.imshow(
            np.clip(jacobian_slice, 0.0, 2.0),
            cmap="coolwarm",
            vmin=0.0,
            vmax=2.0,
            origin="lower",
        )
        _contour(jacobian_panel, protected_slice, color="#ffd60a")
        if np.any(abnormal_slice):
            color = np.array([1.0, 0.0, 1.0, 0.8]) if slot.category is Category.JACOBIAN_NONFINITE else np.array([1.0, 0.0, 0.0, 0.8])
            overlay = np.zeros((*abnormal_slice.shape, 4), dtype=np.float32)
            overlay[abnormal_slice] = color
            jacobian_panel.imshow(overlay, origin="lower")
        if row == 0:
            jacobian_panel.set_title("Candidate Jacobian [0, 2]", fontsize=10)
        jacobian_panel.set_axis_off()
        overlay_panel = axes[row, 4]
        overlay_panel.imshow(fixed_slice, cmap="gray", vmin=low, vmax=high, origin="lower")
        _contour(overlay_panel, protected_slice, color="#ffd60a", linewidth=1.0)
        if np.any(abnormal_slice):
            color = np.array([1.0, 0.0, 1.0, 0.85]) if slot.category is Category.JACOBIAN_NONFINITE else np.array([1.0, 0.0, 0.0, 0.85])
            overlay = np.zeros((*abnormal_slice.shape, 4), dtype=np.float32)
            overlay[abnormal_slice] = color
            overlay_panel.imshow(overlay, origin="lower")
        if row == 0:
            overlay_panel.set_title("Abnormality + protected ROI", fontsize=10)
        overlay_panel.set_axis_off()
        axes[row, 0].text(
            -0.05,
            0.5,
            f"{axis_name}\nindex {index}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
        for panel in axes[row]:
            panel.set_aspect(physical_aspect(bundle.fixed_geometry, axis))
    figure.savefig(destination, dpi=FIGURE_DPI)
    plt.close(figure)


def _similarity_values(
    slot: ExampleSlot,
    qc: Mapping[str, Any],
) -> tuple[float, float, str]:
    if slot.category is Category.PROTECTED_NCC:
        rigid = qc.get("rigid_protected_ncc")
        candidate = qc.get("deformable_protected_ncc")
        label = "Protected ROI"
    else:
        rigid = qc.get("rigid_non_tumor_ncc")
        candidate = qc.get("deformable_non_tumor_ncc")
        label = "Non-tumor"
    if rigid is None or candidate is None:
        raise ValueError("similarity renderer requires finite rigid and candidate NCC")
    rigid_value = float(rigid)
    candidate_value = float(candidate)
    if not math.isfinite(rigid_value) or not math.isfinite(candidate_value):
        raise ValueError("similarity renderer requires finite rigid and candidate NCC")
    return rigid_value, candidate_value, label


def _render_similarity(
    slot: ExampleSlot,
    bundle: DiagnosticBundle,
    destination: Path,
    qc: Mapping[str, Any],
) -> None:
    candidate, candidate_ftv, _ = _require_candidate(bundle)
    rigid_ncc, candidate_ncc, label = _similarity_values(slot, qc)
    excluded = (
        (bundle.fixed_ftv > 0)
        | (bundle.rigid_ftv > 0)
        | (candidate_ftv > 0)
    )
    if slot.category is Category.PROTECTED_NCC:
        region = bundle.protected_roi & ~excluded
    else:
        region = ~excluded
    indices = similarity_slice_indices(
        bundle.fixed,
        bundle.rigid,
        candidate,
        region,
    )
    low, high = robust_window(bundle.fixed, bundle.rigid, candidate)
    rigid_residual = np.abs(bundle.fixed - bundle.rigid)
    candidate_residual = np.abs(bundle.fixed - candidate)
    residual_high = float(
        np.percentile(
            np.concatenate((rigid_residual[region], candidate_residual[region])),
            99.0,
        )
    )
    if residual_high <= 0:
        residual_high = 1.0
    detail = (
        f"{label} NCC: rigid={rigid_ncc:.6f}, candidate={candidate_ncc:.6f}, "
        f"drop={rigid_ncc - candidate_ncc:.6f}"
    )
    figure, axes = plt.subplots(
        3,
        5,
        figsize=FIGURE_SIZE,
        dpi=FIGURE_DPI,
        constrained_layout=True,
    )
    figure.suptitle(_header(slot, detail), fontsize=11)
    for row, (axis, axis_name) in enumerate(DISPLAY_AXES):
        index = indices[axis]
        region_slice = display_slice(region, axis, index)
        for column, (image, title) in enumerate(
            (
                (bundle.fixed, "T0"),
                (bundle.rigid, "Rigid"),
                (candidate, "B-spline candidate"),
            )
        ):
            _image_panel(
                axes[row, column],
                display_slice(image, axis, index),
                low=low,
                high=high,
                title=title,
                row=row,
                region=region_slice,
            )
        for column, (residual, title) in enumerate(
            (
                (rigid_residual, "|T0 - rigid|"),
                (candidate_residual, "|T0 - candidate|"),
            ),
            start=3,
        ):
            panel = axes[row, column]
            panel.imshow(
                display_slice(residual, axis, index),
                cmap="magma",
                vmin=0.0,
                vmax=residual_high,
                origin="lower",
            )
            _contour(panel, region_slice, color="#00e5ff")
            if row == 0:
                panel.set_title(title, fontsize=10)
            panel.set_axis_off()
        axes[row, 0].text(
            -0.05,
            0.5,
            f"{axis_name}\nindex {index}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
        for panel in axes[row]:
            panel.set_aspect(physical_aspect(bundle.fixed_geometry, axis))
    figure.savefig(destination, dpi=FIGURE_DPI)
    plt.close(figure)


def _render_bspline_failure(
    slot: ExampleSlot,
    bundle: DiagnosticBundle,
    destination: Path,
    failure: ElastixMetricFailure | None,
) -> None:
    if failure is None:
        raise ValueError("B-spline failure renderer requires elastix failure diagnostics")
    tumor = (bundle.fixed_ftv > 0) | (bundle.rigid_ftv > 0)
    indices = (
        abnormal_slice_indices(tumor)
        if np.any(tumor)
        else tuple(int(value // 2) for value in bundle.fixed.shape)
    )
    low, high = robust_window(bundle.fixed, bundle.rigid)
    detail = (
        f"Elastix metric failure at resolution {failure.resolution}: "
        f"outside moving buffer={failure.outside_samples}/{failure.total_samples}"
    )
    figure, axes = plt.subplots(
        3,
        5,
        figsize=FIGURE_SIZE,
        dpi=FIGURE_DPI,
        constrained_layout=True,
    )
    figure.suptitle(_header(slot, detail), fontsize=11)
    for row, (axis, axis_name) in enumerate(DISPLAY_AXES):
        index = indices[axis]
        fixed_slice = display_slice(bundle.fixed, axis, index)
        rigid_slice = display_slice(bundle.rigid, axis, index)
        _image_panel(
            axes[row, 0], fixed_slice, low=low, high=high, title="T0", row=row
        )
        _image_panel(
            axes[row, 1], rigid_slice, low=low, high=high, title="Rigid", row=row
        )
        overlay = _red_cyan_overlay(fixed_slice, rigid_slice, low, high)
        axes[row, 2].imshow(overlay, origin="lower")
        if row == 0:
            axes[row, 2].set_title("T0 (red) / rigid (cyan)", fontsize=10)
        axes[row, 2].set_axis_off()
        fov_slice = display_slice(bundle.valid_fov, axis, index)
        axes[row, 3].imshow(fixed_slice, cmap="gray", vmin=low, vmax=high, origin="lower")
        coverage = np.zeros((*fov_slice.shape, 4), dtype=np.float32)
        coverage[..., 1] = 1.0
        coverage[..., 3] = fov_slice.astype(np.float32) * 0.35
        axes[row, 3].imshow(coverage, origin="lower")
        _contour(axes[row, 3], fov_slice, color="#00ff66")
        if row == 0:
            axes[row, 3].set_title("Rigid moving valid FOV", fontsize=10)
        axes[row, 3].set_axis_off()
        diagnostic = axes[row, 4]
        diagnostic.set_facecolor("#f4f4f4")
        diagnostic.text(
            0.05,
            0.75,
            f"Resolution: {failure.resolution}\n"
            f"Outside samples: {failure.outside_samples}\n"
            f"Total samples: {failure.total_samples}\n\n"
            "AdvancedMattesMutualInformationMetric\n"
            "Too many samples map outside\n"
            "moving image buffer",
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
        )
        diagnostic.set_xticks([])
        diagnostic.set_yticks([])
        if row == 0:
            diagnostic.set_title("Elastix diagnostic", fontsize=10)
        axes[row, 0].text(
            -0.05,
            0.5,
            f"{axis_name}\nindex {index}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
        for panel in axes[row, :4]:
            panel.set_aspect(physical_aspect(bundle.fixed_geometry, axis))
    figure.savefig(destination, dpi=FIGURE_DPI)
    plt.close(figure)


def render_example(
    slot: ExampleSlot,
    bundle: DiagnosticBundle,
    destination: Path | str,
    *,
    qc: Mapping[str, Any],
    elastix_failure: ElastixMetricFailure | None,
) -> Path:
    validate_bundle_shapes(bundle)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if slot.category in {
            Category.PROTECTED_FOLD,
            Category.OUTSIDE_FOLD,
            Category.JACOBIAN_NONFINITE,
        }:
            _render_folding(slot, bundle, output, qc)
        elif slot.category in {Category.PROTECTED_NCC, Category.NON_TUMOR_NCC}:
            _render_similarity(slot, bundle, output, qc)
        elif slot.category is Category.BSPLINE_FAILED:
            _render_bspline_failure(slot, bundle, output, elastix_failure)
        else:
            raise ValueError(f"unsupported category: {slot.category}")
    finally:
        plt.close("all")
    image = matplotlib.image.imread(output)
    if image.shape[:2] != (1800, 2400):
        raise ValueError(f"rendered PNG has wrong dimensions: {output} {image.shape}")
    if float(np.std(image)) <= 0.01:
        raise ValueError(f"rendered PNG is blank: {output}")
    return output


def validate_output_directory(directory: Path | str) -> tuple[Path, ...]:
    root = Path(directory)
    files = tuple(sorted(root.glob("*.png")))
    if len(files) != 18:
        raise ValueError(f"expected 18 PNG files, found {len(files)}")
    counts = {
        category.value: sum(
            f"_{category.value}_ISPY2-" in path.name for path in files
        )
        for category in Category
    }
    expected = {category.value: 3 for category in Category}
    if counts != expected:
        raise ValueError(f"category counts differ: {counts}")
    for path in files:
        image = matplotlib.image.imread(path)
        if image.shape[:2] != (1800, 2400):
            raise ValueError(f"rendered PNG has wrong dimensions: {path} {image.shape}")
        if float(np.std(image)) <= 0.01:
            raise ValueError(f"rendered PNG is blank: {path}")
    return files


def _nonfinite_severity(bundle: DiagnosticBundle) -> tuple[int, float]:
    if bundle.candidate_jacobian is None:
        raise ValueError("nonfinite severity requires a candidate Jacobian")
    jacobian = bundle.candidate_jacobian
    finite = jacobian[np.isfinite(jacobian)]
    extreme = float(np.max(np.abs(finite))) if finite.size else float("inf")
    return int(np.count_nonzero(~np.isfinite(jacobian))), extreme


def _assign_secondary_nonfinite_slot(
    slots: Sequence[ExampleSlot],
    bundles: Mapping[tuple[str, str], DiagnosticBundle],
) -> tuple[ExampleSlot, ...]:
    primary = [
        slot
        for slot in slots
        if slot.category is Category.JACOBIAN_NONFINITE and not slot.secondary_view
    ]
    if len(primary) != 2:
        raise ValueError(f"expected two primary nonfinite slots, found {len(primary)}")
    ranked = sorted(
        primary,
        key=lambda slot: (
            -_nonfinite_severity(bundles[(slot.patient_id, slot.visit)])[0],
            -_nonfinite_severity(bundles[(slot.patient_id, slot.visit)])[1],
            slot.patient_id,
            slot.visit,
        ),
    )
    selected = ranked[0]
    updated: list[ExampleSlot] = []
    for slot in slots:
        if slot.category is Category.JACOBIAN_NONFINITE and slot.secondary_view:
            updated.append(
                replace(
                    slot,
                    patient_id=selected.patient_id,
                    visit=selected.visit,
                    status_path=selected.status_path,
                    reasons=selected.reasons,
                )
            )
        else:
            updated.append(slot)
    return tuple(updated)


def generate_rigid_fallback_examples(
    records: Sequence[VisitRecord],
    output_root: Path | str,
    config: RegistrationConfig,
    destination: Path | str,
) -> GenerationSummary:
    root = Path(output_root)
    final = Path(destination)
    if final.exists() and any(final.iterdir()):
        raise FileExistsError(f"destination is nonempty: {final}")
    patients = group_by_patient(records)
    slots = select_example_slots(root)
    bundles: dict[tuple[str, str], DiagnosticBundle] = {}

    def bundle_for(slot: ExampleSlot) -> DiagnosticBundle:
        key = (slot.patient_id, slot.visit)
        if key not in bundles:
            patient = patients.get(slot.patient_id)
            if patient is None:
                raise ValueError(f"manifest lacks selected patient: {slot.patient_id}")
            fixed = patient.get(config.fixed_visit)
            moving = patient.get(slot.visit)
            if fixed is None or moving is None:
                raise ValueError(f"manifest lacks selected pair: {slot.patient_id}/{slot.visit}")
            bundles[key] = reconstruct_candidate_bundle(
                fixed,
                moving,
                root,
                config,
                include_candidate=slot.category is not Category.BSPLINE_FAILED,
            )
        return bundles[key]

    for slot in slots:
        if not slot.secondary_view:
            bundle_for(slot)
    slots = _assign_secondary_nonfinite_slot(slots, bundles)

    with atomic_output_directory(final) as temporary:
        for slot in slots:
            status = json.loads(slot.status_path.read_text(encoding="utf-8"))
            qc = status.get("qc")
            if not isinstance(qc, dict) or tuple(qc.get("reasons", ())) != slot.reasons:
                raise ValueError(f"QC reasons changed for {slot.patient_id}/{slot.visit}")
            failure: ElastixMetricFailure | None = None
            if slot.category is Category.BSPLINE_FAILED:
                log_path = (
                    root
                    / slot.patient_id
                    / slot.visit
                    / "transforms"
                    / "elastix"
                    / "elastix.log"
                )
                failure = parse_elastix_metric_failure(
                    log_path.read_text(encoding="utf-8", errors="replace")
                )
            render_example(
                slot,
                bundle_for(slot),
                temporary / slot.filename,
                qc=qc,
                elastix_failure=failure,
            )
        validate_output_directory(temporary)

    files = validate_output_directory(final)
    categories = Counter(slot.category.value for slot in slots)
    pairs: dict[str, list[str]] = {category.value: [] for category in Category}
    for slot in slots:
        suffix = " secondary_view" if slot.secondary_view else ""
        pairs[slot.category.value].append(f"{slot.patient_id}/{slot.visit}{suffix}")
    return GenerationSummary(
        destination=final,
        files=files,
        categories=dict(categories),
        pairs_by_category={key: tuple(value) for key, value in pairs.items()},
    )
