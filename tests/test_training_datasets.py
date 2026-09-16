from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.training.datasets import (
    LatentPairDataset,
    PreparedVisitDataset,
    pair_conditions,
    validate_prepared_image,
)


def _preprocessing_provenance(shape, roles) -> str:
    return json.dumps(
        {
            "version": "1.0",
            "source_only": True,
            "target_content_used": False,
            "crop_basis": "fixed_center",
            "normalization_scope": "global_train_patients_shared_phases",
            "intensity_stats_hash": "unit-test-training-statistics",
            "output_shape_cdhw": list(shape),
            "output_spacing_dhw": [1.0, 1.0, 1.0],
            "output_affine_lps": np.eye(4, dtype=np.float64).tolist(),
            "extra": {"phase_roles": list(roles)},
        },
        sort_keys=True,
    )


def _write_prepared_archive(
    path,
    *,
    patient_id="p",
    visit_id="p-T0",
    split="train",
    shape=(3, 4, 8, 8),
) -> None:
    roles = ("pre", "early", "late")[: shape[0]]
    np.savez_compressed(
        path,
        image=np.zeros(shape, np.float32),
        affine_lps=np.eye(4, dtype=np.float64),
        spacing_dhw=np.ones(3, dtype=np.float32),
        phase_roles=np.asarray(roles),
        patient_id=np.asarray(patient_id),
        visit_id=np.asarray(visit_id),
        study_uid=np.asarray(f"study-{visit_id}"),
        visit_stage=np.asarray(visit_id.rsplit("-", 1)[-1]),
        split=np.asarray(split),
        source_series_uids=np.asarray([f"series-{role}" for role in roles]),
        source_temporal_positions=np.arange(len(roles), dtype=np.int32),
        provenance_json=np.asarray(_preprocessing_provenance(shape, roles)),
    )


def test_prepared_visit_contract(tmp_path) -> None:
    volume = tmp_path / "visit.npz"
    _write_prepared_archive(volume)
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "patient_id": "p",
                "visit_id": "p-T0",
                "split": "train",
                "prepared_path": str(volume),
            }
        )
        + "\n"
    )
    item = PreparedVisitDataset(manifest, split="train")[0]
    assert item["image"].shape == (3, 4, 8, 8)
    signature = validate_prepared_image(
        item["image"], item["archive_metadata"], label="standalone sampling source"
    )
    assert signature["kind"] == "prepared_patient_volume"
    assert signature["phase_roles"] == ["pre", "early", "late"]


def test_prepared_visit_rejects_train_record_redirected_to_val_archive(tmp_path) -> None:
    val_volume = tmp_path / "val-visit.npz"
    _write_prepared_archive(val_volume, split="val")
    record = {
        "patient_id": "p",
        "visit_id": "p-T0",
        "split": "train",
        "prepared_path": str(val_volume),
    }

    with pytest.raises(ValueError, match="split mismatch"):
        PreparedVisitDataset([record], split="train")


@pytest.mark.parametrize(
    ("archive_field", "archive_value"),
    (("patient_id", "other-patient"), ("visit_id", "p-T1")),
)
def test_prepared_visit_rejects_manifest_archive_identity_mismatch(
    tmp_path, archive_field, archive_value
) -> None:
    identity = {"patient_id": "p", "visit_id": "p-T0", "split": "train"}
    identity[archive_field] = archive_value
    volume = tmp_path / f"bad-{archive_field}.npz"
    _write_prepared_archive(volume, **identity)
    record = {
        "patient_id": "p",
        "visit_id": "p-T0",
        "split": "train",
        "prepared_path": str(volume),
    }

    with pytest.raises(ValueError, match=rf"{archive_field} mismatch"):
        PreparedVisitDataset([record], split="train")


def test_pair_condition_rejects_outcome_leakage() -> None:
    record = {
        "earlier_stage": "T0",
        "later_stage": "T1",
        "delta_days": 35,
        "baseline_clinical": {"age": 52, "pcr": 1},
        "treatment": {},
    }
    with pytest.raises(ValueError, match="forbidden"):
        pair_conditions(record)


def test_pair_condition_rejects_outcome_name_variants() -> None:
    record = {
        "earlier_stage": "T0",
        "later_stage": "T1",
        "delta_days": 35,
        "baseline_clinical": {"pcr_status": 1},
        "treatment": {},
    }
    with pytest.raises(ValueError, match="forbidden"):
        pair_conditions(record)


def test_pair_condition_uses_non_reserved_interval_missing_categories() -> None:
    base = {
        "earlier_stage": "T0",
        "later_stage": "T1",
        "delta_days": None,
        "baseline_clinical": {},
        "treatment": {},
    }

    assert pair_conditions({**base, "interval_missing": True})[
        "interval_missing"
    ] == "yes"
    assert pair_conditions({**base, "interval_missing": False})[
        "interval_missing"
    ] == "no"


def test_pair_latent_semantics_are_fixed(tmp_path) -> None:
    earlier = tmp_path / "early.npz"
    later = tmp_path / "late.npz"
    np.savez_compressed(
        earlier,
        latent=np.zeros((2, 2, 2, 2), np.float32),
        visit_id="p-T0",
        patient_id="p",
        split="train",
        autoencoder_id="ae",
        affine_lps=np.eye(4, dtype=np.float64),
        spacing_dhw=np.ones(3, dtype=np.float32),
    )
    np.savez_compressed(
        later,
        latent=np.ones((2, 2, 2, 2), np.float32),
        visit_id="p-T1",
        patient_id="p",
        split="train",
        autoencoder_id="ae",
        affine_lps=np.eye(4, dtype=np.float64),
        spacing_dhw=np.ones(3, dtype=np.float32),
    )
    record = {
        "pair_id": "p-T0-T1",
        "patient_id": "p",
        "split": "train",
        "earlier_visit_id": "p-T0",
        "later_visit_id": "p-T1",
        "earlier_stage": "T0",
        "later_stage": "T1",
        "delta_days": 30,
        "earlier_latent_path": str(earlier),
        "later_latent_path": str(later),
        "autoencoder_id": "ae",
    }
    item = LatentPairDataset([record])[0]
    assert item["earlier_latent"].sum() == 0
    assert item["later_latent"].sum() == 16


def test_pair_dataset_rejects_unresolved_error_qc() -> None:
    record = {
        "pair_id": "bad",
        "split": "train",
        "earlier_latent_path": "/unused/early.npz",
        "later_latent_path": "/unused/late.npz",
        "qc": [{"severity": "error", "code": "pair_physical_grid_mismatch"}],
    }
    with pytest.raises(ValueError, match="unresolved error-level QC"):
        LatentPairDataset([record])
