from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.training import baselines
from ispy2_symmflow.training.datasets import write_jsonl
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    LATENT_STATISTICS_FINGERPRINT,
    bind_cached_pair_manifest_provenance,
)


STATISTICS_BASE = {"autoencoder_id": "ae-id", "mean": [0.0], "std": [1.0]}


class TinyDeterministicPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, source, tokens):
        condition = tokens.mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)
        return source + self.weight + 0.01 * condition


def _save_latent(path, value: float, *, patient: str, visit: str, split: str) -> None:
    np.savez_compressed(
        path,
        latent=np.full((1, 2, 2, 2), value, dtype=np.float32),
        patient_id=np.asarray(patient),
        visit_id=np.asarray(visit),
        split=np.asarray(split),
        autoencoder_id=np.asarray("ae-id"),
        affine_lps=np.eye(4),
        spacing_dhw=np.ones(3),
    )


def _pair(tmp_path, patient: str, split: str, value: float) -> dict[str, object]:
    earlier = f"{patient}-T0"
    later = f"{patient}-T1"
    earlier_path = tmp_path / f"{earlier}.npz"
    later_path = tmp_path / f"{later}.npz"
    _save_latent(earlier_path, value, patient=patient, visit=earlier, split=split)
    _save_latent(later_path, value + 1.0, patient=patient, visit=later, split=split)
    return {
        "pair_id": f"{patient}-pair",
        "patient_id": patient,
        "split": split,
        "earlier_visit_id": earlier,
        "later_visit_id": later,
        "earlier_stage": "T0",
        "later_stage": "T1",
        "earlier_latent_path": str(earlier_path),
        "later_latent_path": str(later_path),
        "autoencoder_id": "ae-id",
        "interval_missing": True,
        "interval_source": "stage_only",
    }


def _bind_pair_artifacts(records):
    bound, statistics = bind_cached_pair_manifest_provenance(records, STATISTICS_BASE)
    for record in bound:
        if any(
            str(event.get("severity", "")).lower() == "error"
            for event in record.get("qc", [])
        ):
            continue
        for branch in ("earlier", "later"):
            path = Path(str(record[f"{branch}_latent_path"]))
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: np.asarray(archive[key]) for key in archive.files}
            arrays.update(
                {
                    LATENT_STATISTICS_FINGERPRINT: np.asarray(
                        statistics[LATENT_STATISTICS_FINGERPRINT]
                    ),
                    CACHED_PAIR_MANIFEST_FINGERPRINT: np.asarray(
                        statistics[CACHED_PAIR_MANIFEST_FINGERPRINT]
                    ),
                    CACHED_PAIR_MANIFEST_RECORD_COUNT: np.asarray(
                        statistics[CACHED_PAIR_MANIFEST_RECORD_COUNT]
                    ),
                }
            )
            np.savez_compressed(path, **arrays)
    return bound, statistics


@pytest.mark.parametrize("kind", baselines.BASELINE_KINDS)
def test_baseline_rejects_cached_pair_tampering_before_schema_fit(
    tmp_path, monkeypatch, kind
) -> None:
    records, statistics = _bind_pair_artifacts(
        [_pair(tmp_path, "train", "train", 0.0)]
    )
    records[0]["baseline_clinical"] = {"age": 999}
    manifest = tmp_path / f"{kind}-tampered-pairs.jsonl"
    write_jsonl(manifest, records)
    (tmp_path / "latent_statistics.json").write_text(
        json.dumps(statistics), encoding="utf-8"
    )
    schema_called = False

    def unexpected_schema_fit(*args, **kwargs):
        nonlocal schema_called
        schema_called = True
        raise AssertionError("condition schema fitting must not run")

    monkeypatch.setattr(baselines, "fit_condition_schema", unexpected_schema_fit)
    with pytest.raises(ValueError, match="pair manifest fingerprint differs"):
        baselines.train_baseline(
            {"project": {"seed": 1}},
            manifest,
            kind=kind,
            output_dir=tmp_path / kind,
            max_steps=1,
            device=torch.device("cpu"),
        )
    assert schema_called is False


def test_baseline_staged_resume_and_error_qc_filtering(tmp_path, monkeypatch) -> None:
    records = [
        _pair(tmp_path, "train", "train", 0.0),
        {
            "pair_id": "rejected",
            "patient_id": "rejected",
            "split": "train",
            "qc": [{"severity": "ERROR", "code": "bad_grid"}],
        },
        _pair(tmp_path, "validation", "val", 2.0),
    ]
    records, statistics = _bind_pair_artifacts(records)
    manifest = tmp_path / "pairs.jsonl"
    write_jsonl(manifest, records)
    (tmp_path / "latent_statistics.json").write_text(
        json.dumps(statistics),
        encoding="utf-8",
    )
    config = {
        "project": {"seed": 11, "num_workers": 0},
        "data": {"time_pair": ["T0", "T1"], "phase_channels": ["pre"]},
        "conditions": {
            "token_dim": 2,
            "categorical": {
                "stage_i": ["T0"],
                "stage_j": ["T1"],
                "interval_missing": ["no", "yes"],
                "interval_source": ["stage_only"],
            },
            "numeric": ["delta_days"],
        },
        "velocity": {"batch_size": 1},
        "flow": {
            "sigma_min": 0.0,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "max_steps": 2,
            "warmup_steps": 0,
            "gradient_clip_norm": 10.0,
            "precision": "fp32",
        },
        "baseline_training": {
            "max_steps": 2,
            "warmup_steps": 0,
            "gradient_accumulation": 1,
            "validation_interval_steps": 1,
            "validation_seed": 7,
            "precision": "fp32",
            "ema_decay": 0.9,
        },
    }
    monkeypatch.setattr(
        baselines,
        "build_deterministic_baseline_from_config",
        lambda _: TinyDeterministicPredictor(),
    )
    output = tmp_path / "baseline"

    first = baselines.train_baseline(
        config,
        manifest,
        kind="deterministic",
        output_dir=output,
        expected_autoencoder_id="ae-id",
        max_steps=1,
        device=torch.device("cpu"),
    )
    assert first["optimizer_steps"] == 1
    assert first["rejected_error_qc_count"] == 1
    assert first["batch_in_epoch"] == 0
    first_payload = torch.load(first["checkpoint"], weights_only=False)
    assert first_payload["extra"]["batch_in_epoch"] == 0
    assert first_payload["extra"]["last_train_metrics"]

    second = baselines.train_baseline(
        config,
        manifest,
        kind="deterministic",
        output_dir=output,
        expected_autoencoder_id="ae-id",
        resume=first["checkpoint"],
        max_steps=2,
        device=torch.device("cpu"),
    )
    assert second["optimizer_steps"] == 2
    assert second["batch_in_epoch"] == 0
    assert len(torch.load(second["checkpoint"], weights_only=False)["extra"]["validation_history"]) == 2


def test_baseline_training_rejects_torchrun(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="single-process"):
        baselines.train_baseline(
            {},
            tmp_path / "unused.jsonl",
            kind="deterministic",
            output_dir=tmp_path,
            device=torch.device("cpu"),
        )
