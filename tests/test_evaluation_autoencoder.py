from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import ispy2_symmflow.evaluation.autoencoder as autoencoder_evaluation
from ispy2_symmflow.evaluation.autoencoder import evaluate_autoencoder_reconstruction
from ispy2_symmflow.utils.hashing import stable_hash


class _PosteriorMeanAutoencoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encode_normalize: list[bool] = []
        self.decode_denormalize: list[bool] = []

    def encode(self, image, *, normalize: bool = False):
        self.encode_normalize.append(normalize)
        return image

    def decode(self, latent, *, denormalize: bool = False):
        self.decode_denormalize.append(denormalize)
        reconstruction = latent.clone()
        reconstruction[:, 1] += 0.1
        return reconstruction

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self


def _write_visit(path: Path, *, patient: str, visit: str, split: str) -> dict[str, object]:
    roles = ["pre", "early", "late"]
    provenance = json.dumps(
        {
            "generator": "autoencoder-evaluation-test",
            "not_patient_data": True,
            "seed": 11,
            "synthetic": True,
        },
        sort_keys=True,
    )
    image = np.stack(
        (
            np.full((4, 4, 4), 0.0, dtype=np.float32),
            np.full((4, 4, 4), 0.4, dtype=np.float32),
            np.full((4, 4, 4), 0.8, dtype=np.float32),
        )
    )
    np.savez_compressed(
        path,
        image=image,
        affine_lps=np.eye(4, dtype=np.float64),
        spacing_dhw=np.ones(3, dtype=np.float32),
        phase_roles=np.asarray(roles),
        patient_id=np.asarray(patient),
        visit_id=np.asarray(visit),
        visit_stage=np.asarray(visit.rsplit("-", 1)[-1]),
        split=np.asarray(split),
        provenance_json=np.asarray(provenance),
    )
    return {
        "patient_id": patient,
        "visit_id": visit,
        "visit_stage": visit.rsplit("-", 1)[-1],
        "split": split,
        "prepared_path": str(path),
    }


def _write_checkpoint(path: Path, records: list[dict[str, object]]) -> Path:
    config = {
        "data": {"phase_channels": ["pre", "early", "late"]},
        "autoencoder": {"in_channels": 3},
    }
    assignments = {
        str(record["patient_id"]): str(record["split"]) for record in records
    }
    torch.save(
        {
            "format_version": 1,
            "config": config,
            "split_hash": stable_hash(assignments),
            "upstream_commits": {"monai": "test"},
            "extra": {
                "training_signature": {
                    "stage": "autoencoder",
                    "ordered_manifest_fingerprint": stable_hash(records),
                    "manifest_record_count": len(records),
                    "config_fingerprint": stable_hash(config),
                }
            },
        },
        path,
    )
    return path


def test_autoencoder_reconstruction_is_posterior_mean_and_patient_aggregated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "autoencoder.pt"
    records = [
        _write_visit(tmp_path / "P1-T0.npz", patient="P1", visit="P1-T0", split="val"),
        _write_visit(tmp_path / "P1-T1.npz", patient="P1", visit="P1-T1", split="val"),
        _write_visit(tmp_path / "P2-T0.npz", patient="P2", visit="P2-T0", split="val"),
        _write_visit(tmp_path / "P3-T0.npz", patient="P3", visit="P3-T0", split="train"),
    ]
    _write_checkpoint(checkpoint, records)
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    model = _PosteriorMeanAutoencoder()
    monkeypatch.setattr(
        autoencoder_evaluation, "build_autoencoder_from_config", lambda config: model
    )
    monkeypatch.setattr(
        autoencoder_evaluation,
        "load_checkpoint",
        lambda *args, **kwargs: {"format_version": 1},
    )

    report = evaluate_autoencoder_reconstruction(
        checkpoint,
        manifest,
        split="val",
        data_range=2.0,
        bootstrap_samples=100,
        device="cpu",
    )

    assert report["case_count"] == 3
    assert report["patient_count"] == 2
    assert report["posterior_representation"] == "mean"
    assert model.encode_normalize == [False, False, False]
    assert model.decode_denormalize == [False, False, False]
    assert report["cases"][0]["phase_mae"]["early"]["mae"] == pytest.approx(0.1)
    assert report["cases"][0]["enhancement_difference"]["phases"]["early"][
        "mae"
    ] == pytest.approx(0.1)
    assert report["aggregate"]["phase_mae"]["early"]["mae"]["patient_count"] == 2
    assert report["tumor_metrics"]["status"] == "unavailable"
    assert "FTV" in report["tumor_metrics"]["reason"]
    json.dumps(report, allow_nan=False)


def test_autoencoder_reconstruction_refuses_training_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="val or test"):
        evaluate_autoencoder_reconstruction(
            tmp_path / "missing.pt",
            tmp_path / "missing.jsonl",
            split="train",
            data_range=2.0,
        )


def test_autoencoder_rejects_changed_patient_split(tmp_path: Path) -> None:
    records = [
        _write_visit(tmp_path / "P1-T0.npz", patient="P1", visit="P1-T0", split="val"),
        _write_visit(tmp_path / "P2-T0.npz", patient="P2", visit="P2-T0", split="train"),
    ]
    checkpoint = _write_checkpoint(tmp_path / "autoencoder.pt", records)
    records[1] = {**records[1], "split": "val"}
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="split_hash differs"):
        evaluate_autoencoder_reconstruction(
            checkpoint, manifest, split="val", data_range=2.0, device="cpu"
        )


def test_autoencoder_rejects_reordered_complete_manifest(tmp_path: Path) -> None:
    records = [
        _write_visit(tmp_path / "P1-T0.npz", patient="P1", visit="P1-T0", split="val"),
        _write_visit(tmp_path / "P2-T0.npz", patient="P2", visit="P2-T0", split="train"),
    ]
    checkpoint = _write_checkpoint(tmp_path / "autoencoder.pt", records)
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in reversed(records)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ordered manifest fingerprint differs"):
        evaluate_autoencoder_reconstruction(
            checkpoint, manifest, split="val", data_range=2.0, device="cpu"
        )
