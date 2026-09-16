from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.training.distributed import DistributedContext
from ispy2_symmflow.training import run
from ispy2_symmflow.training.datasets import write_jsonl
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    LATENT_STATISTICS_FINGERPRINT,
    bind_cached_pair_manifest_provenance,
)


_LATENT_STATISTICS_BASE = {"autoencoder_id": "ae-id", "mean": [0.0], "std": [1.0]}


class TinyAutoencoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, image, *, sample_posterior=True):
        mean = image * self.weight
        scale = torch.full_like(mean, 0.5)
        return mean, mean, scale


class TinyVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, state, tau, tokens):
        condition = tokens.mean(dim=(1, 2)).reshape(state.shape[0], 1, 1, 1, 1)
        return self.scale * state + condition


@pytest.fixture
def cpu_training(monkeypatch):
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        initialized_here=False,
    )
    monkeypatch.setattr(run, "initialize_distributed", lambda: context)
    monkeypatch.setattr(run, "finalize_distributed", lambda _: None)


def _save_image(
    path: Path,
    value: float,
    *,
    patient_id: str,
    visit_id: str,
    split: str,
) -> None:
    shape = (1, 2, 2, 2)
    affine = np.eye(4, dtype=np.float64)
    spacing = np.ones(3, dtype=np.float32)
    roles = ("pre",)
    provenance = {
        "version": "1.0",
        "source_only": True,
        "target_content_used": False,
        "crop_basis": "fixed_center",
        "normalization_scope": "global_train_patients_shared_phases",
        "intensity_stats_hash": "training-run-test-statistics",
        "output_shape_cdhw": list(shape),
        "output_spacing_dhw": spacing.tolist(),
        "output_affine_lps": affine.tolist(),
        "extra": {"phase_roles": list(roles)},
    }
    np.savez_compressed(
        path,
        image=np.full(shape, value, dtype=np.float32),
        affine_lps=affine,
        spacing_dhw=spacing,
        phase_roles=np.asarray(roles),
        patient_id=np.asarray(patient_id),
        visit_id=np.asarray(visit_id),
        study_uid=np.asarray(f"study-{visit_id}"),
        visit_stage=np.asarray(visit_id.rsplit("-", 1)[-1]),
        split=np.asarray(split),
        source_series_uids=np.asarray(["series-pre"]),
        source_temporal_positions=np.asarray([0], dtype=np.int32),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )


def _save_latent(
    path: Path,
    value: float,
    *,
    patient_id: str,
    visit_id: str,
    split: str,
) -> None:
    np.savez_compressed(
        path,
        latent=np.full((1, 2, 2, 2), value, dtype=np.float32),
        patient_id=np.asarray(patient_id),
        visit_id=np.asarray(visit_id),
        split=np.asarray(split),
        autoencoder_id=np.asarray("ae-id"),
        affine_lps=np.eye(4),
        spacing_dhw=np.ones(3),
    )


def test_ordered_training_signature_covers_every_manifest_field() -> None:
    records = [
        {"patient_id": "a", "split": "train", "nested": {"value": 1}},
        {"patient_id": "b", "split": "val", "nested": {"value": 2}},
    ]
    kwargs = {
        "batch_size": 1,
        "world_size": 1,
        "planned_steps": 2,
        "batches_per_epoch": 1,
    }
    signature = run._training_signature("autoencoder", records, {"seed": 1}, **kwargs)
    reordered = run._training_signature(
        "autoencoder", list(reversed(records)), {"seed": 1}, **kwargs
    )
    changed = [dict(records[0], nested={"value": 99}), records[1]]
    modified = run._training_signature("autoencoder", changed, {"seed": 1}, **kwargs)

    assert signature["ordered_manifest_fingerprint"] != reordered[
        "ordered_manifest_fingerprint"
    ]
    assert signature["ordered_manifest_fingerprint"] != modified[
        "ordered_manifest_fingerprint"
    ]


def test_autoencoder_partial_resume_validates_and_updates_best(
    tmp_path, monkeypatch, cpu_training
) -> None:
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _save_image(
        train_path,
        1.0,
        patient_id="a",
        visit_id="a-T0",
        split="train",
    )
    _save_image(
        val_path,
        7.0,
        patient_id="b",
        visit_id="b-T0",
        split="val",
    )
    manifest = tmp_path / "visits.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "visit_id": "a-T0",
                "patient_id": "a",
                "split": "train",
                "prepared_path": str(train_path),
            },
            {
                "visit_id": "b-T0",
                "patient_id": "b",
                "split": "val",
                "prepared_path": str(val_path),
            },
        ],
    )
    config = {
        "project": {"seed": 13, "num_workers": 0},
        "autoencoder": {
            "batch_size": 1,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "kl_weight": 0.0,
            "gradient_weight": 0.0,
            "gradient_clip_norm": 1.0,
            "warmup_steps": 0,
            "max_epochs": 2,
            "validation_interval_epochs": 99,
        },
        "flow": {"precision": "fp32"},
    }
    validation_losses = iter((3.0, 2.0))

    def fake_validation(model, images, **kwargs):
        batches = list(images)
        assert len(batches) == 1
        assert torch.all(batches[0] == 7.0)
        loss = next(validation_losses)
        return {"loss": loss, "reconstruction": loss, "kl": 0.0, "gradient": 0.0,
                "sample_count": 1}

    monkeypatch.setattr(run, "build_autoencoder_from_config", lambda _: TinyAutoencoder())
    monkeypatch.setattr(run, "validate_autoencoder", fake_validation)
    output = tmp_path / "ae"

    first = run.train_autoencoder(config, manifest, output_dir=output, max_steps=1)
    assert first["steps"] == 1
    assert first["best_validation_loss"] == 3.0
    first_payload = torch.load(first["checkpoint"], weights_only=False)
    assert first_payload["extra"]["scaler_state"] == {}
    assert len(first_payload["extra"]["validation_history"]) == 1

    second = run.train_autoencoder(
        config,
        manifest,
        output_dir=output,
        resume=first["checkpoint"],
        max_steps=2,
    )
    assert second["steps"] == 2
    assert second["best_validation_loss"] == 2.0
    assert len(second["validation_history"]) == 2
    best_payload = torch.load(second["best_checkpoint"], weights_only=False)
    latest_payload = torch.load(second["checkpoint"], weights_only=False)
    assert best_payload["step"] == latest_payload["step"] == 2
    assert best_payload["extra"]["training_signature"] == second["training_signature"]

    with pytest.raises(ValueError, match="resume step 2 is beyond requested"):
        run.train_autoencoder(
            config,
            manifest,
            output_dir=tmp_path / "ae-rewind",
            resume=second["checkpoint"],
            max_steps=1,
        )


def _pair_record(
    root: Path,
    patient_id: str,
    split: str,
    *,
    value: float,
) -> dict[str, object]:
    earlier_id = f"{patient_id}-T0"
    later_id = f"{patient_id}-T1"
    earlier_path = root / f"{earlier_id}.npz"
    later_path = root / f"{later_id}.npz"
    _save_latent(
        earlier_path,
        value,
        patient_id=patient_id,
        visit_id=earlier_id,
        split=split,
    )
    _save_latent(
        later_path,
        value + 1.0,
        patient_id=patient_id,
        visit_id=later_id,
        split=split,
    )
    return {
        "pair_id": f"{patient_id}-T0-T1",
        "patient_id": patient_id,
        "split": split,
        "earlier_visit_id": earlier_id,
        "later_visit_id": later_id,
        "earlier_stage": "T0",
        "later_stage": "T1",
        "earlier_latent_path": str(earlier_path),
        "later_latent_path": str(later_path),
        "autoencoder_id": "ae-id",
        "interval_missing": True,
        "interval_source": "stage_only",
    }


def _bind_pair_artifacts(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    bound, statistics = bind_cached_pair_manifest_provenance(
        records, _LATENT_STATISTICS_BASE
    )
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


@pytest.mark.parametrize("tamper", ("age", "endpoints"))
def test_symmflow_rejects_cached_pair_tampering_before_schema_fit(
    tmp_path, monkeypatch, cpu_training, tamper
) -> None:
    records, statistics = _bind_pair_artifacts(
        [_pair_record(tmp_path, "train", "train", value=0.0)]
    )
    if tamper == "age":
        records[0]["baseline_clinical"] = {"age": 999}
    else:
        records[0]["earlier_visit_id"], records[0]["later_visit_id"] = (
            records[0]["later_visit_id"],
            records[0]["earlier_visit_id"],
        )
        records[0]["earlier_latent_path"], records[0]["later_latent_path"] = (
            records[0]["later_latent_path"],
            records[0]["earlier_latent_path"],
        )
    manifest = tmp_path / "tampered-pairs.jsonl"
    write_jsonl(manifest, records)
    (tmp_path / "latent_statistics.json").write_text(
        json.dumps(statistics), encoding="utf-8"
    )
    schema_called = False

    def unexpected_schema_fit(*args, **kwargs):
        nonlocal schema_called
        schema_called = True
        raise AssertionError("condition schema fitting must not run")

    monkeypatch.setattr(run, "fit_condition_schema", unexpected_schema_fit)
    with pytest.raises(ValueError, match="pair manifest fingerprint differs"):
        run.train_symmflow(
            {"project": {"seed": 1}},
            manifest,
            output_dir=tmp_path / "flow",
            max_steps=1,
        )
    assert schema_called is False


def test_symmflow_resume_restores_micro_step_and_validates_ema(
    tmp_path, monkeypatch, cpu_training
) -> None:
    train = _pair_record(tmp_path, "train", "train", value=0.0)
    validation = _pair_record(tmp_path, "validation", "val", value=2.0)
    rejected = {
        "pair_id": "rejected",
        "patient_id": "rejected",
        "split": "train",
        "autoencoder_id": "wrong-id",
        "qc": [{"code": "bad_grid", "severity": "error"}],
    }
    records, statistics = _bind_pair_artifacts([train, rejected, validation])
    manifest = tmp_path / "pairs.jsonl"
    write_jsonl(manifest, records)
    (tmp_path / "latent_statistics.json").write_text(
        json.dumps(statistics),
        encoding="utf-8",
    )
    config = {
        "project": {"seed": 17, "num_workers": 0},
        "data": {"time_pair": ["T0", "T1"]},
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
            "loss_weight_x": 1.0,
            "loss_weight_y": 1.0,
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "max_steps": 2,
            "warmup_steps": 0,
            "gradient_clip_norm": 10.0,
            "gradient_accumulation": 2,
            "ema_decay": 0.9,
            "precision": "fp32",
            "validation_interval_steps": 1,
            "validation_repeats": 1,
            "validation_seed": 23,
        },
    }
    validation_losses = iter((4.0, 5.0))
    validated_scales: list[torch.Tensor] = []

    def fake_validation(velocity, conditions, objective, batches, **kwargs):
        items = list(batches)
        assert len(items) == 1
        assert items[0]["record"][0]["patient_id"] == "validation"
        validated_scales.append(velocity.scale.detach().clone())
        loss = next(validation_losses)
        return {"loss": loss, "loss_x": loss / 2, "loss_y": loss / 2,
                "sample_count": 1, "stochastic_repeats": 1, "seed": 23}

    monkeypatch.setattr(run, "build_velocity_model_from_config", lambda _: TinyVelocity())
    monkeypatch.setattr(run, "validate_symmflow", fake_validation)
    output = tmp_path / "flow"

    first = run.train_symmflow(config, manifest, output_dir=output, max_steps=1)
    first_payload = torch.load(first["checkpoint"], weights_only=False)
    assert first["micro_step"] == 2
    assert first["rejected_error_qc_count"] == 1
    assert torch.equal(validated_scales[0], first_payload["ema"]["shadow"]["scale"])
    assert "velocity.scale" in first_payload["model"]

    resumed_output = tmp_path / "flow-resumed"
    second = run.train_symmflow(
        config,
        manifest,
        output_dir=resumed_output,
        resume=first["checkpoint"],
        max_steps=2,
    )
    latest_payload = torch.load(second["checkpoint"], weights_only=False)
    best_payload = torch.load(second["best_checkpoint"], weights_only=False)
    assert second["micro_step"] == 4
    assert len(second["validation_history"]) == 2
    assert latest_payload["extra"]["micro_step"] == 4
    assert latest_payload["extra"]["rejected_error_qc_count"] == 1
    assert torch.equal(validated_scales[-1], latest_payload["ema"]["shadow"]["scale"])
    assert second["best_validation_loss"] == 4.0
    assert best_payload["step"] == 1
    assert latest_payload["step"] == 2
    assert torch.equal(validated_scales[0], best_payload["ema"]["shadow"]["scale"])
    assert Path(second["best_checkpoint"]).parent == output
    assert not (resumed_output / "symmflow_best.pt").exists()

    with pytest.raises(ValueError, match="resume step 2 is beyond requested"):
        run.train_symmflow(
            config,
            manifest,
            output_dir=tmp_path / "flow-rewind",
            resume=second["checkpoint"],
            max_steps=1,
        )
