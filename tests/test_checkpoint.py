from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.training.checkpoint import (
    CheckpointCompatibilityError,
    load_checkpoint,
    save_checkpoint,
)
from ispy2_symmflow.training.ema import ExponentialMovingAverage


def test_checkpoint_round_trip_and_schema_guard(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    ema = ExponentialMovingAverage(model, decay=0.9)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        epoch=2,
        step=17,
        config={"flow": {"sigma_min": 0.0}},
        autoencoder_id="ae-sha256",
        latent_statistics={"mean": [0.0], "std": [1.0]},
        feature_schema={"categorical": ["arm"]},
        split_hash="split-sha256",
        upstream_commits={"symmetricflow": "cb14c609"},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    payload = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        expected_autoencoder_id="ae-sha256",
        expected_feature_schema={"categorical": ["arm"]},
        expected_split_hash="split-sha256",
        expected_latent_statistics={"mean": [0.0], "std": [1.0]},
        expected_sigma_min=0.0,
    )
    assert payload["step"] == 17
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
    with pytest.raises(CheckpointCompatibilityError, match="schema"):
        load_checkpoint(path, model=model, expected_feature_schema={"categorical": ["pcr"]})
    with pytest.raises(CheckpointCompatibilityError, match="normalization"):
        load_checkpoint(
            path,
            model=model,
            expected_latent_statistics={"mean": [1.0], "std": [1.0]},
        )


def test_ema_context_restores_training_parameters() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    live = model.weight.detach().clone()
    with ema.average_parameters(model):
        assert not torch.equal(model.weight, live)
    assert torch.equal(model.weight, live)


def test_missing_optimizer_state_is_rejected_before_model_mutation(tmp_path) -> None:
    source = torch.nn.Linear(2, 1)
    path = save_checkpoint(
        tmp_path / "model-only.pt",
        model=source,
        config={},
        autoencoder_id=None,
        latent_statistics=None,
        feature_schema=None,
        split_hash="split",
        upstream_commits={},
    )
    target = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(target.parameters())
    before = {name: value.clone() for name, value in target.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="optimizer"):
        load_checkpoint(path, model=target, optimizer=optimizer)

    for name, value in target.state_dict().items():
        assert torch.equal(value, before[name])


def test_training_signature_and_full_latent_provenance_are_checked(tmp_path) -> None:
    model = torch.nn.Linear(1, 1)
    stats = {
        "mean": [0.0],
        "std": [1.0],
        "element_count_per_channel": 8,
        "fit_split": "train",
        "fit_patient_ids": ["a"],
        "fit_visit_ids": ["a-T0"],
        "autoencoder_id": "ae",
        "source_split_hash": "split",
        "source_preprocessing_signature": {
            "kind": "prepared_patient_volume",
            "version": "1.0",
            "intensity_stats_hash": "intensity",
            "phase_roles": ["pre", "early", "late"],
        },
    }
    path = save_checkpoint(
        tmp_path / "resume.pt",
        model=model,
        config={},
        autoencoder_id="ae",
        latent_statistics=stats,
        feature_schema={},
        split_hash="split",
        upstream_commits={},
        extra={"training_signature": {"batch_size": 1}, "scaler_state": {}},
    )
    with pytest.raises(CheckpointCompatibilityError, match="training plan"):
        load_checkpoint(
            path,
            model=model,
            expected_training_signature={"batch_size": 2},
        )
    changed = dict(stats, fit_visit_ids=["a-T1"])
    with pytest.raises(CheckpointCompatibilityError, match="normalization"):
        load_checkpoint(path, model=model, expected_latent_statistics=changed)
    changed = dict(
        stats,
        source_preprocessing_signature={
            **stats["source_preprocessing_signature"],
            "intensity_stats_hash": "different",
        },
    )
    with pytest.raises(CheckpointCompatibilityError, match="normalization"):
        load_checkpoint(path, model=model, expected_latent_statistics=changed)
    with pytest.raises(CheckpointCompatibilityError, match="resume state"):
        load_checkpoint(path, model=model, required_extra_keys=("micro_step",))
