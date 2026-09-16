"""End-to-end prepared-volume writer with an explicit NPZ contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ispy2_symmflow.data.manifest import (
    build_pair_manifest,
    read_visit_manifest,
    write_jsonl,
)
from ispy2_symmflow.data.schema import PairRecord, QCEvent, VisitRecord
from ispy2_symmflow.data.split import assign_patient_splits, split_hash

from .intensity import IntensityStats, apply_intensity_stats, fit_intensity_stats
from .loader import LoadedVisit, load_visit_phases
from .provenance import PreprocessingProvenance
from .spatial import (
    apply_crop_plan,
    make_crop_plan,
    reorient_volume,
    resample_volume,
    update_affine_for_crop,
)


NPZ_KEYS = (
    "image",
    "affine_lps",
    "spacing_dhw",
    "phase_roles",
    "patient_id",
    "visit_id",
    "study_uid",
    "visit_stage",
    "split",
    "source_series_uids",
    "source_temporal_positions",
    "provenance_json",
)


@dataclass(frozen=True)
class PreprocessConfig:
    phase_roles: tuple[str, ...] = ("pre", "early", "late")
    target_spacing_dhw: tuple[float, float, float] | None = (2.5, 1.25, 1.25)
    output_shape_dhw: tuple[int, int, int] = (64, 128, 128)
    target_axis_codes: str | None = "SAR"
    crop_mode: str = "fixed_center"
    lower_percentile: float = 0.5
    upper_percentile: float = 99.5
    max_voxels_per_visit: int = 250_000
    padding_value: float = -1.0


@dataclass(frozen=True)
class PreprocessedVisit:
    image: NDArray[np.float32]
    affine_lps: NDArray[np.float64]
    spacing_dhw: tuple[float, float, float]
    provenance: PreprocessingProvenance


@dataclass(frozen=True)
class PrepareResult:
    visits: tuple[VisitRecord, ...]
    pairs: tuple[PairRecord, ...]
    intensity_stats: IntensityStats
    visit_manifest_path: str
    pair_manifest_path: str
    intensity_stats_path: str
    skipped_visit_ids: tuple[str, ...]
    invalid_pair_ids: tuple[str, ...]


VolumeLoader = Callable[[VisitRecord, Sequence[str]], LoadedVisit]


def _stats_hash(stats: IntensityStats) -> str:
    payload = json.dumps(stats.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preprocess_source(
    source: LoadedVisit,
    stats: IntensityStats,
    config: PreprocessConfig,
) -> PreprocessedVisit:
    """Preprocess one supplied source; this API has no target argument."""

    raw = np.asarray(source.image, dtype=np.float32)
    if raw.ndim != 4:
        raise ValueError(f"source image must be [C,D,H,W], got {raw.shape}")
    oriented, oriented_affine, oriented_spacing, actual_axis_codes = reorient_volume(
        raw, source.affine_lps, config.target_axis_codes
    )
    resampled, resampled_affine, spacing = resample_volume(
        oriented,
        oriented_affine,
        oriented_spacing,
        config.target_spacing_dhw,
    )
    plan = make_crop_plan(
        resampled,
        config.output_shape_dhw,
        mode=config.crop_mode,
    )
    normalized = apply_intensity_stats(resampled, stats)
    output = apply_crop_plan(normalized, plan, padding_value=config.padding_value)
    output_affine = update_affine_for_crop(resampled_affine, plan)
    provenance = PreprocessingProvenance(
        version="1.0",
        source_only=True,
        target_content_used=False,
        crop_basis=plan.basis,
        normalization_scope="global_train_patients_shared_phases",
        intensity_stats_hash=_stats_hash(stats),
        input_shape_cdhw=tuple(int(value) for value in raw.shape),
        oriented_shape_cdhw=tuple(int(value) for value in oriented.shape),
        resampled_shape_cdhw=tuple(int(value) for value in resampled.shape),
        output_shape_cdhw=tuple(int(value) for value in output.shape),
        source_spacing_dhw=tuple(float(value) for value in source.spacing_dhw),
        output_spacing_dhw=tuple(float(value) for value in spacing),
        input_affine_lps=tuple(
            tuple(float(value) for value in row) for row in source.affine_lps
        ),
        output_affine_lps=tuple(
            tuple(float(value) for value in row) for row in output_affine
        ),
        crop_plan=plan.to_dict(),
        extra={
            "phase_roles": list(source.phase_roles),
            "resampling": "scipy_ndimage_linear" if oriented.shape != resampled.shape else "none",
            "affine_world_coordinate_system": "DICOM_LPS",
            "array_axis_codes": list(actual_axis_codes) if actual_axis_codes else "native",
            "orientation_transform": (
                "nibabel_axis_permutation_and_flip"
                if config.target_axis_codes is not None
                else "none"
            ),
        },
    )
    provenance.validate()
    return PreprocessedVisit(output, output_affine, spacing, provenance)


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "unknown"


def _eligible(visit: VisitRecord, roles: Sequence[str]) -> bool:
    return set(roles).issubset(visit.phase_paths) and not any(
        event.severity == "error" for event in visit.qc
    )


def _archive_path(output_dir: Path, visit: VisitRecord, split: str) -> Path:
    study_token = hashlib.sha256(visit.study_uid.encode("utf-8")).hexdigest()[:12]
    return (
        output_dir
        / "volumes"
        / split
        / _safe_component(visit.patient_id)
        / f"{_safe_component(visit.visit_stage)}-{study_token}.npz"
    )


def _save_archive(
    destination: Path,
    prepared: PreprocessedVisit,
    loaded: LoadedVisit,
    visit: VisitRecord,
    split: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    provenance_json = json.dumps(
        prepared.provenance.to_dict(), sort_keys=True, separators=(",", ":")
    )
    np.savez_compressed(
        temporary,
        image=np.asarray(prepared.image, dtype=np.float32),
        affine_lps=np.asarray(prepared.affine_lps, dtype=np.float64),
        spacing_dhw=np.asarray(prepared.spacing_dhw, dtype=np.float32),
        phase_roles=np.asarray(loaded.phase_roles),
        patient_id=np.asarray(visit.patient_id),
        visit_id=np.asarray(visit.visit_id),
        study_uid=np.asarray(visit.study_uid),
        visit_stage=np.asarray(visit.visit_stage),
        split=np.asarray(split),
        source_series_uids=np.asarray(loaded.source_series_uids),
        source_temporal_positions=np.asarray(
            loaded.source_temporal_positions, dtype=np.int32
        ),
        provenance_json=np.asarray(provenance_json),
    )
    os.replace(temporary, destination)


def _with_pair_grid_qc(
    pairs: Iterable[PairRecord],
    prepared_by_visit: Mapping[str, PreprocessedVisit],
) -> list[PairRecord]:
    checked: list[PairRecord] = []
    for pair in pairs:
        earlier = prepared_by_visit[pair.earlier_visit_id]
        later = prepared_by_visit[pair.later_visit_id]
        affine_match = np.allclose(
            earlier.affine_lps, later.affine_lps, rtol=1e-5, atol=1e-4
        )
        spacing_match = np.allclose(
            earlier.spacing_dhw, later.spacing_dhw, rtol=1e-6, atol=1e-6
        )
        shape_match = earlier.image.shape == later.image.shape
        qc = list(pair.qc)
        if not (affine_match and spacing_match and shape_match):
            origin_distance = float(
                np.linalg.norm(earlier.affine_lps[:3, 3] - later.affine_lps[:3, 3])
            )
            qc.append(
                QCEvent(
                    code="pair_physical_grid_mismatch",
                    severity="error",
                    message=(
                        "Prepared longitudinal visits do not share one physical voxel grid; "
                        "validated alignment is required before paired latent training"
                    ),
                    details={
                        "affine_match": bool(affine_match),
                        "spacing_match": bool(spacing_match),
                        "shape_match": bool(shape_match),
                        "origin_distance_mm": origin_distance,
                        "earlier_affine_lps": earlier.affine_lps.tolist(),
                        "later_affine_lps": later.affine_lps.tolist(),
                    },
                )
            )
        checked.append(replace(pair, qc=tuple(qc)))
    return checked


def prepare_dataset(
    visits: Iterable[VisitRecord] | str | Path,
    output_dir: str | Path,
    *,
    split_by_patient: Mapping[str, str] | None = None,
    split_ratios: Mapping[str, float] | None = None,
    split_seed: int = 20260908,
    config: PreprocessConfig | None = None,
    volume_loader: VolumeLoader | None = None,
    earlier_stage: str = "T0",
    later_stage: str = "T1",
) -> PrepareResult:
    """Prepare eligible visits and write directly resolvable pair JSONL.

    ``volume_loader`` is injectable for tests or pre-existing trusted volumes.
    The default loader decodes only the phase references selected by metadata QC.
    """

    records = (
        read_visit_manifest(visits)
        if isinstance(visits, (str, Path))
        else list(visits)
    )
    if not records:
        raise ValueError("visit manifest is empty")
    settings = config or PreprocessConfig()
    loader = volume_loader or load_visit_phases
    patient_ids = sorted({visit.patient_id for visit in records})
    embedded_splits = {visit.patient_id: visit.split for visit in records if visit.split}
    patients_with_embedded_split = {visit.patient_id for visit in records if visit.split}
    if patients_with_embedded_split and patients_with_embedded_split != set(patient_ids):
        raise ValueError("visit manifest has partial patient split assignments")
    for patient_id in patients_with_embedded_split:
        observed = {visit.split for visit in records if visit.patient_id == patient_id}
        if len(observed) != 1:
            raise ValueError(f"patient {patient_id} has conflicting visit split assignments")
    if split_by_patient is None:
        if embedded_splits:
            split_by_patient = {key: str(value) for key, value in embedded_splits.items()}
        else:
            ratios = split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
            split_by_patient = assign_patient_splits(
                patient_ids, ratios=ratios, seed=split_seed
            )
    elif set(split_by_patient) != set(patient_ids):
        raise ValueError("split assignments must cover exactly the visit patients")
    elif embedded_splits and dict(split_by_patient) != embedded_splits:
        raise ValueError("explicit split assignments disagree with the audited visit manifest")

    eligible = [visit for visit in records if _eligible(visit, settings.phase_roles)]
    skipped = tuple(visit.visit_id for visit in records if visit not in eligible)
    training = [
        visit
        for visit in eligible
        if split_by_patient[visit.patient_id] == "train"
    ]
    if not training:
        raise ValueError("no QC-passing training visits are available")

    stats = fit_intensity_stats(
        (
            (visit.patient_id, loader(visit, settings.phase_roles).image)
            for visit in training
        ),
        split_by_patient,
        lower_percentile=settings.lower_percentile,
        upper_percentile=settings.upper_percentile,
        max_voxels_per_visit=settings.max_voxels_per_visit,
    )

    destination = Path(output_dir).resolve()
    prepared_records: list[VisitRecord] = []
    prepared_by_visit: dict[str, PreprocessedVisit] = {}
    for visit in eligible:
        split = split_by_patient[visit.patient_id]
        loaded = loader(visit, settings.phase_roles)
        prepared = preprocess_source(loaded, stats, settings)
        archive_path = _archive_path(destination, visit, split)
        _save_archive(archive_path, prepared, loaded, visit, split)
        prepared_by_visit[visit.visit_id] = prepared
        prepared_records.append(
            replace(visit, split=split, prepared_path=str(archive_path))
        )

    pairs = _with_pair_grid_qc(
        build_pair_manifest(
            prepared_records,
            split_by_patient,
            earlier_stage=earlier_stage,
            later_stage=later_stage,
            require_three_phase=set(("pre", "early", "late")).issubset(
                settings.phase_roles
            ),
        ),
        prepared_by_visit,
    )
    visit_manifest_path = destination / "visits.prepared.jsonl"
    pair_manifest_path = destination / f"pairs.{earlier_stage}-{later_stage}.jsonl"
    stats_path = destination / "intensity_stats.json"
    write_jsonl(prepared_records, visit_manifest_path)
    write_jsonl(pairs, pair_manifest_path)
    stats_path.write_text(
        json.dumps(
            {
                **stats.to_dict(),
                "split_hash": split_hash(split_by_patient),
                "phase_roles": list(settings.phase_roles),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return PrepareResult(
        visits=tuple(prepared_records),
        pairs=tuple(pairs),
        intensity_stats=stats,
        visit_manifest_path=str(visit_manifest_path),
        pair_manifest_path=str(pair_manifest_path),
        intensity_stats_path=str(stats_path),
        skipped_visit_ids=skipped,
        invalid_pair_ids=tuple(
            pair.pair_id
            for pair in pairs
            if any(event.severity == "error" for event in pair.qc)
        ),
    )
