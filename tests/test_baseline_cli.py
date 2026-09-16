from __future__ import annotations

import inspect
import json
from pathlib import Path

import click
import numpy as np
import pytest
from click.testing import CliRunner

torch = pytest.importorskip("torch")

from ispy2_symmflow import cli
from ispy2_symmflow.cli import (
    _load_baseline_sampling_models,
    main,
    sample_baseline_command,
)
from ispy2_symmflow.models import ConditionSchema, StructuredConditionEncoder
from ispy2_symmflow.models.conditioning import CategoricalField
from ispy2_symmflow.training.checkpoint import save_checkpoint
from ispy2_symmflow.training.ema import ExponentialMovingAverage
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    bind_cached_pair_manifest_provenance,
    bind_latent_statistics_fingerprint,
)
from ispy2_symmflow.utils.hashing import sha256_file


class TinyAutoencoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("latent_mean", torch.zeros(1, 1, 1, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, 1, 1, 1, 1))

    def set_latent_statistics(self, mean, std) -> None:
        self.latent_mean.copy_(torch.as_tensor(mean).reshape_as(self.latent_mean))
        self.latent_std.copy_(torch.as_tensor(std).reshape_as(self.latent_std))

    def freeze(self):
        return self.requires_grad_(False).eval()

    def encode(self, image, *, normalize=True):
        return image

    def decode(self, latent, *, denormalize=True):
        return latent


class TinyPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, source, tokens):
        return source * self.weight + 0.0 * tokens.mean()


def _schema() -> ConditionSchema:
    return ConditionSchema(
        categorical_fields=(
            CategoricalField("stage_i", ("T0",)),
            CategoricalField("stage_j", ("T1",)),
        ),
        numeric_fields=(),
        token_dim=2,
    )


def test_baseline_commands_and_forward_only_interface_are_registered() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for command in ("train-baseline", "train_baseline", "sample-baseline", "sample_baseline"):
        assert command in result.output
    parameters = inspect.signature(sample_baseline_command.callback).parameters
    assert "source" in parameters and "conditions" in parameters
    assert "direction" not in parameters and "target" not in parameters


def test_baseline_loader_applies_ema_and_checks_autoencoder_hash(
    tmp_path, monkeypatch
) -> None:
    autoencoder_path = tmp_path / "autoencoder.pt"
    save_checkpoint(
        autoencoder_path,
        model=TinyAutoencoder(),
        config={"autoencoder": {"latent_channels": 1}},
        autoencoder_id=None,
        latent_statistics=None,
        feature_schema=None,
        split_hash="split",
        upstream_commits={},
    )
    autoencoder_id = sha256_file(autoencoder_path)
    schema = _schema()
    statistics = bind_latent_statistics_fingerprint(
        {
            "mean": [2.0],
            "std": [3.0],
            "autoencoder_id": autoencoder_id,
        }
    )
    predictor = TinyPredictor()
    encoder = StructuredConditionEncoder(schema)
    system = torch.nn.ModuleDict({"model": predictor, "conditions": encoder})
    ema = ExponentialMovingAverage(predictor)
    ema.shadow["weight"].fill_(7.0)
    baseline_path = tmp_path / "deterministic.pt"
    training_config = {
        "data": {"time_pair": ["T0", "T1"], "phase_channels": ["pre"]},
        "flow": {"sigma_min": 0.0},
    }
    save_checkpoint(
        baseline_path,
        model=system,
        ema=ema,
        config=training_config,
        autoencoder_id=autoencoder_id,
        latent_statistics=statistics,
        feature_schema=schema.to_dict(),
        split_hash="split",
        upstream_commits={},
        extra={"baseline_kind": "deterministic", "direction": "forward"},
    )
    codec_calls = []

    def build_codec(config, *, latent_statistics=None):
        codec_calls.append((config, latent_statistics))
        return TinyAutoencoder()

    monkeypatch.setattr("ispy2_symmflow.models.build_codec_from_config", build_codec)
    monkeypatch.setattr(
        "ispy2_symmflow.models.build_deterministic_baseline_from_config",
        lambda _: TinyPredictor(),
    )

    loaded_ae, loaded_model, _, header, kind = _load_baseline_sampling_models(
        str(baseline_path), str(autoencoder_path), torch.device("cpu")
    )
    assert kind == "deterministic"
    assert header["latent_statistics"] == statistics
    assert loaded_ae.latent_mean.item() == pytest.approx(2.0)
    assert loaded_ae.latent_std.item() == pytest.approx(3.0)
    assert loaded_model.weight.item() == pytest.approx(7.0)
    assert codec_calls == [
        ({"autoencoder": {"latent_channels": 1}}, statistics)
    ]

    tampered = torch.load(baseline_path, map_location="cpu", weights_only=False)
    tampered["latent_statistics"]["mean"] = [99.0]
    torch.save(tampered, baseline_path)
    with pytest.raises(click.ClickException, match="fingerprint does not match"):
        _load_baseline_sampling_models(
            str(baseline_path), str(autoencoder_path), torch.device("cpu")
        )


def test_sampling_codec_factory_loads_mewm_checkpoint_without_local_loader(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "relocated-vqgan.ckpt"
    checkpoint.write_bytes(b"external checkpoint fixture")
    statistics = {"mean": [1.0], "std": [2.0]}
    training_config = {
        "_config_path": str(tmp_path / "experiment.yaml"),
        "codec": {
            "backend": "mewm_vqgan",
            "config_path": "upstream.yaml",
            "checkpoint_path": "/stale/location/vqgan.ckpt",
            "checkpoint_sha256": "a" * 64,
        },
    }
    calls = []

    def build_codec(config, *, latent_statistics=None):
        calls.append((config, latent_statistics))
        return TinyAutoencoder()

    def reject_local_loader(*args, **kwargs):
        raise AssertionError("MeWM codec must load its own upstream checkpoint")

    monkeypatch.setattr("ispy2_symmflow.models.build_codec_from_config", build_codec)
    monkeypatch.setattr(
        "ispy2_symmflow.training.checkpoint.load_checkpoint", reject_local_loader
    )

    loaded = cli._load_sampling_codec(
        training_config,
        statistics,
        str(checkpoint),
        torch.device("cpu"),
    )

    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert len(calls) == 1
    codec_config, observed_statistics = calls[0]
    assert observed_statistics == statistics
    assert codec_config["codec"]["checkpoint_path"] == str(checkpoint.resolve())
    assert codec_config["codec"]["config_path"] == "upstream.yaml"
    assert training_config["codec"]["checkpoint_path"] == "/stale/location/vqgan.ckpt"


def test_deterministic_baseline_cli_rejects_multiple_candidates(tmp_path, monkeypatch) -> None:
    paths = {
        name: tmp_path / name
        for name in ("baseline.pt", "autoencoder.pt", "source.npz", "conditions.json")
    }
    for path in paths.values():
        path.touch()
    monkeypatch.setattr(
        cli,
        "_load_baseline_sampling_models",
        lambda *args: (None, None, None, {}, "deterministic"),
    )
    result = CliRunner().invoke(
        main,
        [
            "sample-baseline",
            "--config",
            str(Path("configs/smoke.yaml").resolve()),
            "--checkpoint",
            str(paths["baseline.pt"]),
            "--autoencoder-checkpoint",
            str(paths["autoencoder.pt"]),
            "--source",
            str(paths["source.npz"]),
            "--conditions",
            str(paths["conditions.json"]),
            "--output",
            str(tmp_path / "prediction.npz"),
            "--num-samples",
            "2",
        ],
    )
    assert result.exit_code != 0
    assert "exactly one candidate" in result.output


def test_formal_sample_rejects_condition_pair_that_differs_from_checkpoint(
    tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "flow.pt"
    autoencoder_checkpoint = tmp_path / "autoencoder.pt"
    checkpoint.touch()
    autoencoder_checkpoint.touch()
    source = tmp_path / "source.npz"
    np.savez_compressed(
        source,
        image=np.ones((1, 2, 2, 2), dtype=np.float32),
        visit_stage=np.asarray("T0"),
        phase_roles=np.asarray(["pre"]),
    )
    conditions = tmp_path / "conditions.json"
    conditions.write_text(
        json.dumps({"stage_i": "T0", "stage_j": "T3"}), encoding="utf-8"
    )
    header = {
        "config": {
            "data": {"time_pair": ["T0", "T1"], "phase_channels": ["pre"]},
            "flow": {"sigma_min": 0.0},
        }
    }
    monkeypatch.setattr(
        cli,
        "_load_sampling_models",
        lambda *args: (None, None, None, header),
    )
    result = CliRunner().invoke(
        main,
        [
            "sample",
            "--config",
            str(Path("configs/smoke.yaml").resolve()),
            "--checkpoint",
            str(checkpoint),
            "--autoencoder-checkpoint",
            str(autoencoder_checkpoint),
            "--source",
            str(source),
            "--conditions",
            str(conditions),
            "--direction",
            "forward",
            "--output",
            str(tmp_path / "prediction.npz"),
        ],
    )
    assert result.exit_code != 0
    assert "must match checkpoint time_pair ['T0', 'T1']" in result.output


def test_deterministic_baseline_cli_saves_forward_provenance(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "baseline.pt"
    autoencoder_checkpoint = tmp_path / "autoencoder.pt"
    checkpoint.write_bytes(b"baseline")
    autoencoder_checkpoint.write_bytes(b"autoencoder")
    source = tmp_path / "source.npz"
    source_provenance = {
        "synthetic": True,
        "generator": "baseline-cli-test",
        "seed": 7,
        "not_patient_data": True,
    }
    source_signature = {
        "kind": "synthetic",
        "provenance": source_provenance,
        "phase_roles": ["pre"],
        "output_shape_cdhw": [1, 2, 2, 2],
        "spacing_dhw": [1.0, 1.0, 1.0],
    }
    np.savez_compressed(
        source,
        image=np.ones((1, 2, 2, 2), dtype=np.float32),
        affine_lps=np.eye(4, dtype=np.float64),
        spacing_dhw=np.ones(3, dtype=np.float32),
        patient_id=np.asarray("patient"),
        visit_id=np.asarray("patient-T0"),
        study_uid=np.asarray("study-patient-T0"),
        visit_stage=np.asarray("T0"),
        split=np.asarray("val"),
        phase_roles=np.asarray(["pre"]),
        provenance_json=np.asarray(json.dumps(source_provenance, sort_keys=True)),
    )
    conditions = tmp_path / "conditions.json"
    conditions.write_text(json.dumps({"stage_i": "T0", "stage_j": "T1"}), encoding="utf-8")
    schema = _schema()
    _, sampling_statistics = bind_cached_pair_manifest_provenance(
        [{"pair_id": "fixture-pair"}],
        {
            "autoencoder_id": "ae-hash",
            "mean": [0.0],
            "std": [1.0],
            "source_preprocessing_signature": source_signature,
        },
    )
    header = {
        "config": {
            "data": {"time_pair": ["T0", "T1"], "phase_channels": ["pre"]},
            "flow": {"sigma_min": 0.0},
        },
        "feature_schema": schema.to_dict(),
        "latent_statistics": sampling_statistics,
        "split_hash": "split-hash",
        "extra": {
            "schema_provenance": {
                "unavailable_fields": [],
                "train_observed_coverage": {
                    "stage_i": {
                        "kind": "categorical",
                        "observation_count": 1,
                        "observed_categories": ["T0"],
                    },
                    "stage_j": {
                        "kind": "categorical",
                        "observation_count": 1,
                        "observed_categories": ["T1"],
                    },
                },
            },
            "training_signature": {
                "ordered_manifest_hash": "manifest-hash",
                "manifest_record_count": 1,
                CACHED_PAIR_MANIFEST_FINGERPRINT: sampling_statistics[
                    CACHED_PAIR_MANIFEST_FINGERPRINT
                ],
                CACHED_PAIR_MANIFEST_RECORD_COUNT: 1,
            }
        },
    }
    monkeypatch.setattr(
        cli,
        "_load_baseline_sampling_models",
        lambda *args: (
            TinyAutoencoder().freeze(),
            TinyPredictor().eval(),
            StructuredConditionEncoder(schema).eval(),
            header,
            "deterministic",
        ),
    )
    output = tmp_path / "prediction.npz"
    result = CliRunner().invoke(
        main,
        [
            "sample-baseline",
            "--config",
            str(Path("configs/smoke.yaml").resolve()),
            "--checkpoint",
            str(checkpoint),
            "--autoencoder-checkpoint",
            str(autoencoder_checkpoint),
            "--source",
            str(source),
            "--conditions",
            str(conditions),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    with np.load(output, allow_pickle=False) as archive:
        assert archive["samples"].shape == (1, 1, 1, 2, 2, 2)
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["direction"] == "forward"
    assert sidecar["seeds"] == []
    assert sidecar["metadata"]["baseline_kind"] == "deterministic"
    assert sidecar["metadata"]["ema_applied"] is True
    assert sidecar["metadata"]["steps"] == 0
    assert sidecar["metadata"]["baseline_checkpoint_sha256"] == sha256_file(checkpoint)
    assert sidecar["metadata"]["source_sha256"] == sha256_file(source)
    assert sidecar["metadata"]["source_preprocessing_signature"] == source_signature
    assert sidecar["metadata"]["endpoint_semantics"] == "paper_clean_endpoints"
    assert sidecar["metadata"]["training_split_hash"] == "split-hash"
    assert sidecar["metadata"]["training_signature"]["ordered_manifest_hash"] == (
        "manifest-hash"
    )
    assert sidecar["metadata"]["condition_schema_provenance"][
        "unavailable_fields"
    ] == []
