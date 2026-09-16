from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import ispy2_symmflow.training.mu_glioma_import as mu_import
from ispy2_symmflow.training.datasets import LatentPairDataset, pair_conditions, read_jsonl
from ispy2_symmflow.training.provenance import validate_cached_latent_provenance
from ispy2_symmflow.utils.hashing import sha256_file


def _clinical_patient() -> dict[str, object]:
    return {
        "context_static": {
            "age_at_diagnosis_years": 51,
            "sex_at_birth": "Female",
            "race": "Unknown",
            "primary_diagnosis": "GBM",
            "who_grade": 4,
            "previous_brain_tumor": "No",
            "stereotactic_biopsy_before_resection": 0,
            "genomics": {"idh1": 1, "mgmt": 0},
            "progression": True,
            "survival": 300,
        },
        "timeline": [
            {
                "tp_id": "T1",
                "mri_day": 10,
                "survival": {"status": "alive"},
                "actions": {
                    "chemotherapy": [
                        {"agent": "Temozolomide", "start_day": 10, "end_day": 20}
                    ],
                    "radiotherapy": [
                        {"agent": "external", "start_day": 30, "end_day": 40}
                    ],
                },
            },
            {
                "tp_id": "T2",
                "mri_day": 30,
                "survival": {"status": "alive"},
                "actions": {},
            },
        ],
    }


def test_conditions_are_source_available_and_target_day_is_exclusive() -> None:
    patient = _clinical_patient()
    baseline = mu_import._baseline_conditions(patient["context_static"])
    timeline = patient["timeline"]
    treatment = mu_import._treatment_conditions(timeline, 0, 1)

    assert baseline["age_at_diagnosis_years"] == 51.0
    assert baseline["race"] is None
    assert baseline["genomic_idh1"] == "positive_or_altered"
    assert baseline["genomic_mgmt"] == "negative_or_wild_type"
    assert "survival" not in baseline
    assert "progression" not in baseline
    assert treatment["chemotherapy_received"] == "yes"
    assert treatment["radiotherapy_received"] == "no"


def test_tiny_joint4_import_is_bound_and_loadable(
    tmp_path: Path, monkeypatch
) -> None:
    per_modality_shape = (2, 2, 2, 2)
    joint_shape = (8, 2, 2, 2)
    monkeypatch.setattr(mu_import, "MU_PER_MODALITY_LATENT_SHAPE", per_modality_shape)
    monkeypatch.setattr(mu_import, "MU_JOINT_LATENT_SHAPE", joint_shape)
    monkeypatch.setattr(mu_import, "MU_IMAGE_SHAPE", (8, 8, 8))
    monkeypatch.setattr(
        mu_import, "MU_EXPECTED_SAMPLE_SPLITS", {"train": 8, "val": 8}
    )
    monkeypatch.setattr(
        mu_import, "MU_EXPECTED_VISIT_SPLITS", {"train": 2, "val": 2}
    )
    monkeypatch.setattr(
        mu_import, "MU_EXPECTED_PAIR_SPLITS", {"train": 1, "val": 1}
    )

    root = tmp_path / "cache"
    checkpoint = tmp_path / "vqgan.ckpt"
    checkpoint.write_bytes(b"mu-vqgan")
    checkpoint_sha256 = sha256_file(checkpoint)
    codebook_sha256 = "2" * 64
    data_contract_sha256 = "3" * 64
    manifest_sha256 = "4" * 64
    identity = {
        "schema": mu_import.MU_CACHE_SCHEMA,
        "sample_count": 16,
        "split_counts": {"train": 8, "val": 8},
        "input_shape_zyx": [8, 8, 8],
        "latent_shape_czyx": list(per_modality_shape),
        "latent_dtype": "float16",
        "vqgan_checkpoint": "/upstream/vqgan.ckpt",
        "vqgan_sha256": checkpoint_sha256,
        "codebook_sha256": codebook_sha256,
        "codebook_min": -2.0,
        "codebook_max": 2.0,
        "data_contract_sha256": data_contract_sha256,
        "manifest_sha256": manifest_sha256,
        "numeric_contract": mu_import.MU_NUMERIC_CONTRACT,
        "normalization": mu_import.MU_NORMALIZATION,
    }
    root.mkdir()
    (root / "cache_identity.json").write_text(
        json.dumps(identity), encoding="utf-8"
    )
    modality_values = {name: index - 1.5 for index, name in enumerate(mu_import.MU_MODALITIES)}
    for patient_id, split in (("PatientID_0001", "train"), ("PatientID_0002", "val")):
        split_dir = root / "volumes" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for timepoint in ("Timepoint_1", "Timepoint_2"):
            for modality, value in modality_values.items():
                sample_id = f"{patient_id}__{timepoint}__{modality}"
                torch.save(
                    {
                        "schema": mu_import.MU_PAYLOAD_SCHEMA,
                        "sample_id": sample_id,
                        "patient_id": patient_id,
                        "timepoint": timepoint,
                        "modality": modality,
                        "split": split,
                        "continuous_latent": torch.full(
                            per_modality_shape,
                            value
                            + (1.0 if timepoint == "Timepoint_2" else 0.0)
                            + (10.0 if split == "val" else 0.0),
                            dtype=torch.float16,
                        ),
                        "vqgan_sha256": checkpoint_sha256,
                        "codebook_sha256": codebook_sha256,
                        "data_contract_sha256": data_contract_sha256,
                        "manifest_sha256": manifest_sha256,
                        "numeric_contract": mu_import.MU_NUMERIC_CONTRACT,
                        "normalization": mu_import.MU_NORMALIZATION,
                    },
                    split_dir / f"{sample_id}.pt",
                )

    clinical = tmp_path / "clinical.json"
    clinical.write_text(
        json.dumps(
            {
                "schema_version": mu_import.MU_CLINICAL_SCHEMA,
                "patients": {
                    patient_id: _clinical_patient()
                    for patient_id in ("PatientID_0001", "PatientID_0002")
                },
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "imported"

    result = mu_import.import_mu_glioma_continuous_latents(
        root / "cache_identity.json",
        root,
        clinical,
        checkpoint,
        destination,
    )

    assert result.pair_count == 2
    assert result.visit_count == 4
    records = read_jsonl(result.pair_manifest_path)
    statistics = json.loads(Path(result.statistics_path).read_text(encoding="utf-8"))
    validate_cached_latent_provenance(records, statistics)
    conditions = pair_conditions(records[0])
    assert not {"survival", "progression", "target_mask"}.intersection(conditions)
    item = LatentPairDataset(records, split="train")[0]
    assert item["earlier_latent"].shape == joint_shape
    expected = np.repeat(
        np.asarray([(value / 2.0) for value in modality_values.values()]), 2
    )
    np.testing.assert_allclose(
        item["earlier_latent"][:, 0, 0, 0].numpy(), expected
    )

    zscore_destination = tmp_path / "imported-zscore"
    zscore_result = mu_import.import_mu_glioma_continuous_latents(
        root / "cache_identity.json",
        root,
        clinical,
        checkpoint,
        zscore_destination,
        flow_normalization=mu_import.MU_FLOW_NORMALIZATION_TRAIN_ZSCORE,
    )
    zscore_statistics = json.loads(
        Path(zscore_result.statistics_path).read_text(encoding="utf-8")
    )
    assert zscore_statistics["normalization"] == (
        mu_import.MU_FLOW_NORMALIZATION_TRAIN_ZSCORE
    )
    assert zscore_statistics["fit_split"] == "train"
    assert zscore_statistics["fit_visit_count"] == 2
    np.testing.assert_allclose(
        zscore_statistics["mean"],
        np.repeat(np.asarray(list(modality_values.values())) + 0.5, 2),
    )
    np.testing.assert_allclose(zscore_statistics["std"], 0.5)
    zscore_records = read_jsonl(zscore_result.pair_manifest_path)
    train_item = LatentPairDataset(zscore_records, split="train")[0]
    validation_item = LatentPairDataset(zscore_records, split="val")[0]
    np.testing.assert_allclose(train_item["earlier_latent"].numpy(), -1.0)
    np.testing.assert_allclose(validation_item["earlier_latent"].numpy(), 19.0)
