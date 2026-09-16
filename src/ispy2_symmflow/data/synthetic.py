"""Clearly labeled synthetic volumes for software smoke tests only."""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any

import numpy as np

from ispy2_symmflow.data.manifest import build_pair_manifest, visit_from_dict
from ispy2_symmflow.training.datasets import write_jsonl


def _synthetic_volume(
    rng: np.random.Generator,
    shape: tuple[int, int, int, int],
    *,
    lesion_scale: float,
) -> np.ndarray:
    channels, depth, height, width = shape
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, depth),
        np.linspace(-1, 1, height),
        np.linspace(-1, 1, width),
        indexing="ij",
    )
    anatomy = np.exp(-((z / 0.75) ** 2 + (y / 0.85) ** 2 + (x / 0.9) ** 2) * 2.0)
    lesion = np.exp(-(((z - 0.1) / 0.25) ** 2 + ((y + 0.15) / 0.2) ** 2 + ((x - 0.2) / 0.2) ** 2) * 2.0)
    volume = []
    enhancements = np.linspace(0.15, 0.55, channels)
    for enhancement in enhancements:
        noise = rng.normal(0, 0.015, size=anatomy.shape)
        phase = -1.0 + 1.4 * anatomy + lesion_scale * enhancement * lesion + noise
        volume.append(np.clip(phase, -1.0, 1.0))
    return np.asarray(volume, dtype=np.float32)


def create_synthetic_dataset(
    output_dir: str | Path,
    *,
    patient_count: int = 6,
    image_shape: tuple[int, int, int, int] = (3, 16, 32, 32),
    seed: int = 1234,
) -> dict[str, Any]:
    """Create paired T0/T1 NPZ files and manifests, never presented as I-SPY2."""

    if patient_count < 3:
        raise ValueError("synthetic smoke data needs at least three patients")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    visit_records: list[dict[str, Any]] = []
    split_counts = {
        "train": max(1, patient_count - 2),
        "val": 1,
        "test": 1,
    }
    boundaries = (split_counts["train"], split_counts["train"] + 1)
    for index in range(patient_count):
        patient = f"SYNTH-{index:03d}"
        split = "train" if index < boundaries[0] else "val" if index < boundaries[1] else "test"
        arm = "synthetic_arm_a" if index % 2 == 0 else "synthetic_arm_b"
        interval_days = 35 + index
        t0_date = date(2020, 1, 1) + timedelta(days=90 * index)
        stage_dates = {
            "T0": t0_date,
            "T1": t0_date + timedelta(days=interval_days),
        }
        for stage, lesion_scale in (("T0", 1.0), ("T1", 0.75 + 0.1 * (index % 3))):
            visit_id = f"{patient}-{stage}"
            study_date = stage_dates[stage].isoformat()
            volume = _synthetic_volume(rng, image_shape, lesion_scale=lesion_scale)
            path = destination / f"{visit_id}.npz"
            provenance = {
                "synthetic": True,
                "generator": "ispy2_symmflow.data.synthetic",
                "seed": seed,
                "not_patient_data": True,
            }
            np.savez_compressed(
                path,
                image=volume,
                affine_lps=np.eye(4, dtype=np.float64),
                spacing_dhw=np.asarray((4.0, 4.0, 4.0)),
                phase_roles=np.asarray(("pre", "early", "late")[: image_shape[0]]),
                patient_id=patient,
                visit_id=visit_id,
                study_uid=f"synthetic-{visit_id}",
                visit_stage=stage,
                study_date=study_date,
                date_source="synthetic_generator_known_date",
                relative_date_verified=True,
                split=split,
                source_series_uids=np.asarray([], dtype="U1"),
                source_temporal_positions=np.asarray([], dtype=np.int64),
                provenance_json=json.dumps(provenance, sort_keys=True),
            )
            visit_records.append(
                {
                    "schema_version": "synthetic-1.0",
                    "visit_id": visit_id,
                    "patient_id": patient,
                    "collection": "SYNTHETIC_SMOKE_ONLY",
                    "study_uid": f"synthetic-{visit_id}",
                    "visit_stage": stage,
                    "study_date": study_date,
                    "date_source": "synthetic_generator_known_date",
                    "relative_date_verified": True,
                    "split": split,
                    "prepared_path": str(path),
                    "baseline_clinical": {"age": 40 + index},
                    "treatment": {"treatment_arm": arm},
                    "evaluation_metadata": {},
                    "synthetic": True,
                }
            )
    typed_visits = [visit_from_dict(record) for record in visit_records]
    split_by_patient = {
        str(record["patient_id"]): str(record["split"]) for record in visit_records
    }
    pair_records = [
        {
            **pair.to_dict(),
            "evaluation_metadata": {},
            "synthetic": True,
        }
        for pair in build_pair_manifest(
            typed_visits,
            split_by_patient,
            earlier_stage="T0",
            later_stage="T1",
            require_three_phase=False,
        )
    ]
    visits = write_jsonl(destination / "visits.jsonl", visit_records)
    pairs = write_jsonl(destination / "pairs.jsonl", pair_records)
    record = {
        "synthetic": True,
        "patient_count": patient_count,
        "image_shape": list(image_shape),
        "seed": seed,
        "split_counts": split_counts,
        "visit_manifest": str(visits),
        "pair_manifest": str(pairs),
    }
    (destination / "SYNTHETIC_DATASET.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record
