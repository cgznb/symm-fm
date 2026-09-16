from __future__ import annotations

import inspect
import json
from pathlib import Path

import click
import numpy as np
import pytest
from click.testing import CliRunner

from ispy2_symmflow.cli import (
    _endpoint_provenance,
    _validate_sampling_source,
    main,
    sample_command,
)
from ispy2_symmflow.cli import _write_json
from ispy2_symmflow.config import ConfigError, load_config, resolve_time_pairs
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    bind_cached_pair_manifest_provenance,
)
from ispy2_symmflow.utils.hashing import sha256_file


def test_required_pipeline_commands_are_registered() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for command in (
        "audit-data",
        "prepare-data",
        "train-autoencoder",
        "cache-latents",
        "import-mewm-latents",
        "import-mu-glioma-latents",
        "train-symmflow",
        "sample",
        "evaluate",
    ):
        assert command in result.output


def test_formal_sample_command_has_no_target_or_mask_argument() -> None:
    parameters = inspect.signature(sample_command.callback).parameters
    assert "source" in parameters
    assert "conditions" in parameters
    assert "target" not in parameters
    assert "target_mask" not in parameters
    result = CliRunner().invoke(main, ["sample", "--help"])
    assert "--source" in result.output
    assert "--target" not in result.output


def test_flow_sampling_sidecar_binds_source_archive(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from ispy2_symmflow import cli
    from ispy2_symmflow.models import ConditionSchema, StructuredConditionEncoder
    from ispy2_symmflow.models.conditioning import CategoricalField

    class TinyAutoencoder(torch.nn.Module):
        def encode(self, image, *, normalize=True):
            return image

        def decode(self, latent, *, denormalize=True):
            return latent

    class ZeroVelocity(torch.nn.Module):
        def forward(self, state, tau, condition_tokens):
            return torch.zeros_like(state)

    checkpoint = tmp_path / "flow.pt"
    autoencoder_checkpoint = tmp_path / "autoencoder.pt"
    checkpoint.write_bytes(b"flow")
    autoencoder_checkpoint.write_bytes(b"autoencoder")
    source = tmp_path / "source.npz"
    source_provenance = {
        "synthetic": True,
        "generator": "flow-cli-test",
        "seed": 11,
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
    conditions.write_text(
        json.dumps({"stage_i": "T0", "stage_j": "T1"}), encoding="utf-8"
    )
    schema = ConditionSchema(
        categorical_fields=(
            CategoricalField("stage_i", ("T0",)),
            CategoricalField("stage_j", ("T1",)),
        ),
        numeric_fields=(),
        token_dim=2,
    )
    coverage = {
        field: {
            "kind": "categorical",
            "observation_count": 1,
            "observed_categories": [stage],
        }
        for field, stage in (("stage_i", "T0"), ("stage_j", "T1"))
    }
    _, latent_statistics = bind_cached_pair_manifest_provenance(
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
        "latent_statistics": latent_statistics,
        "split_hash": "split-hash",
        "extra": {
            "schema_provenance": {
                "unavailable_fields": [],
                "train_observed_coverage": coverage,
            },
                "training_signature": {
                    "ordered_manifest_fingerprint": "manifest-hash",
                    "manifest_record_count": 1,
                    CACHED_PAIR_MANIFEST_FINGERPRINT: latent_statistics[
                        CACHED_PAIR_MANIFEST_FINGERPRINT
                    ],
                    CACHED_PAIR_MANIFEST_RECORD_COUNT: 1,
                },
        },
    }
    monkeypatch.setattr(
        cli,
        "_load_sampling_models",
        lambda *args: (
            TinyAutoencoder().eval(),
            ZeroVelocity().eval(),
            StructuredConditionEncoder(schema).eval(),
            header,
        ),
    )
    output = tmp_path / "prediction.npz"

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
            str(output),
            "--num-samples",
            "1",
            "--steps",
            "1",
            "--solver",
            "euler",
        ],
    )

    assert result.exit_code == 0, result.output
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["metadata"]["source_sha256"] == sha256_file(source)


def test_make_synthetic_cli_and_evaluate_cli(tmp_path) -> None:
    runner = CliRunner()
    generated = runner.invoke(
        main,
        ["make-synthetic", "--output-dir", str(tmp_path / "synthetic"), "--patients", "3"],
    )
    assert generated.exit_code == 0, generated.output
    target_path = tmp_path / "synthetic" / "SYNTH-002-T1.npz"
    with np.load(target_path, allow_pickle=False) as archive:
        target = np.asarray(archive["image"], dtype=np.float32)
    prediction = tmp_path / "prediction.npz"
    np.savez_compressed(prediction, samples=np.stack((target[None], target[None]), axis=0))
    output = tmp_path / "metrics.json"
    evaluated = runner.invoke(
        main,
        [
            "evaluate",
            "--prediction",
            str(prediction),
            "--target",
            str(target_path),
            "--data-range",
            "2",
            "--output",
            str(output),
            "--allow-unverified-geometry",
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output
    assert output.is_file()


def test_single_case_evaluation_rejects_tampered_prediction_unless_exploratory(
    tmp_path,
) -> None:
    runner = CliRunner()
    generated = runner.invoke(
        main,
        ["make-synthetic", "--output-dir", str(tmp_path / "synthetic"), "--patients", "3"],
    )
    assert generated.exit_code == 0, generated.output
    target_path = tmp_path / "synthetic" / "SYNTH-002-T1.npz"
    with np.load(target_path, allow_pickle=False) as archive:
        target = np.asarray(archive["image"], dtype=np.float32)
    prediction = tmp_path / "prediction.npz"
    np.savez_compressed(prediction, samples=target[None, None])
    prediction.with_suffix(".json").write_text(
        json.dumps(
            {
                "array_file": prediction.name,
                "array_sha256": sha256_file(prediction),
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(prediction, samples=np.zeros_like(target)[None, None])
    common = [
        "evaluate",
        "--prediction",
        str(prediction),
        "--target",
        str(target_path),
        "--data-range",
        "2",
        "--output",
        str(tmp_path / "metrics.json"),
    ]

    strict = runner.invoke(main, common)
    assert strict.exit_code != 0
    assert "array SHA-256 differs" in strict.output

    exploratory = runner.invoke(main, [*common, "--allow-unverified-geometry"])
    assert exploratory.exit_code == 0, exploratory.output
    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["prediction_integrity_verified"] is False
    assert "SHA-256 differs" in report["prediction_integrity_status"]


def test_single_case_evaluation_rejects_replaced_source_with_unchanged_metadata(
    tmp_path,
) -> None:
    runner = CliRunner()
    generated = runner.invoke(
        main,
        ["make-synthetic", "--output-dir", str(tmp_path / "synthetic"), "--patients", "3"],
    )
    assert generated.exit_code == 0, generated.output
    source = tmp_path / "synthetic" / "SYNTH-002-T0.npz"
    target_path = tmp_path / "synthetic" / "SYNTH-002-T1.npz"
    with np.load(source, allow_pickle=False) as archive:
        source_payload = {key: np.asarray(archive[key]) for key in archive.files}
        source_metadata = {
            key: archive[key].item() if archive[key].ndim == 0 else archive[key].tolist()
            for key in archive.files
            if key != "image"
        }
    with np.load(target_path, allow_pickle=False) as archive:
        target = np.asarray(archive["image"], dtype=np.float32)

    prediction = tmp_path / "prediction.npz"
    np.savez_compressed(prediction, samples=target[None, None])
    prediction.with_suffix(".json").write_text(
        json.dumps(
            {
                "array_file": prediction.name,
                "array_sha256": sha256_file(prediction),
                "direction": "forward",
                "metadata": {
                    "source_path": str(source.resolve()),
                    "source_sha256": sha256_file(source),
                    "source_metadata": source_metadata,
                    "conditions": {"stage_i": "T0", "stage_j": "T1"},
                },
            }
        ),
        encoding="utf-8",
    )
    source_payload["image"] = np.zeros_like(source_payload["image"])
    np.savez_compressed(source, **source_payload)
    common = [
        "evaluate",
        "--prediction",
        str(prediction),
        "--target",
        str(target_path),
        "--source",
        str(source),
        "--data-range",
        "2",
        "--output",
        str(tmp_path / "metrics.json"),
    ]

    strict = runner.invoke(main, common)
    assert strict.exit_code != 0
    assert "source archive SHA-256 differs" in strict.output

    exploratory = runner.invoke(main, [*common, "--allow-unverified-geometry"])
    assert exploratory.exit_code == 0, exploratory.output
    report = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert report["prediction_integrity_verified"] is False
    assert report["source_integrity_verified"] is False
    assert "source archive SHA-256 differs" in report["source_integrity_status"]


def test_json_reports_replace_nonfinite_metrics_with_null(tmp_path) -> None:
    output = _write_json(tmp_path / "metrics.json", {"empty": float("nan"), "perfect": np.inf})
    text = output.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert __import__("json").loads(text) == {"empty": None, "perfect": None}


def test_compatibility_config_and_unsupported_registration_are_explicit() -> None:
    compatible = load_config("configs/symmflow_compat.yaml")
    assert compatible["flow"]["sigma_min"] == pytest.approx(1e-4)
    with pytest.raises(ConfigError, match="registration"):
        load_config("configs/smoke.yaml", ("data.registration=rigid",))


def test_multi_interval_config_contract_is_explicit() -> None:
    assert resolve_time_pairs(
        {
            "time_pair": None,
            "time_pairs": [["T0", "T1"], ["T0", "T3"], ["T2", "T3"]],
        }
    ) == (("T0", "T1"), ("T0", "T3"), ("T2", "T3"))
    with pytest.raises(ConfigError, match="only one"):
        resolve_time_pairs(
            {"time_pair": ["T0", "T1"], "time_pairs": [["T1", "T2"]]}
        )


def test_external_registered_t0_config_requires_full_disclosure() -> None:
    with pytest.raises(ConfigError, match="registration_assisted"):
        load_config(
            "configs/smoke.yaml",
            ("data.registration=external_registered_t0",),
        )
    config = load_config(
        "configs/smoke.yaml",
        (
            "data.registration=external_registered_t0",
            "data.registration_assisted=true",
            "data.backward_task_label=registration-assisted reconstruction",
            "data.crop_policy=single_T0_mask_bbox_center_reused_for_all_visits",
        ),
    )
    assert config["data"]["registration"] == "external_registered_t0"


def test_formal_sampling_source_requires_identity_and_training_preprocessing() -> None:
    torch = pytest.importorskip("torch")
    image = torch.ones(1, 2, 2, 2)
    provenance = {
        "synthetic": True,
        "generator": "cli-test",
        "seed": 3,
        "not_patient_data": True,
    }
    metadata = {
        "patient_id": "P1",
        "visit_id": "P1-T0",
        "study_uid": "study-P1-T0",
        "visit_stage": "T0",
        "split": "val",
        "phase_roles": ["pre"],
        "affine_lps": np.eye(4).tolist(),
        "spacing_dhw": [1.0, 1.0, 1.0],
        "provenance_json": json.dumps(provenance, sort_keys=True),
    }
    signature = {
        "kind": "synthetic",
        "provenance": provenance,
        "phase_roles": ["pre"],
        "output_shape_cdhw": [1, 2, 2, 2],
        "spacing_dhw": [1.0, 1.0, 1.0],
    }
    header = {
        "config": {"data": {"phase_channels": ["pre"]}},
        "latent_statistics": {"source_preprocessing_signature": signature},
    }

    observed, roles, identity = _validate_sampling_source(
        image, metadata, header, expected_stage="T0"
    )
    assert observed == signature
    assert roles == ["pre"]
    assert identity["split"] == "val"

    with pytest.raises(click.ClickException, match="missing 'visit_id'"):
        _validate_sampling_source(
            image, {key: value for key, value in metadata.items() if key != "visit_id"},
            header,
            expected_stage="T0",
        )
    incompatible = {
        **header,
        "latent_statistics": {
            "source_preprocessing_signature": {
                **signature,
                "provenance": {**provenance, "seed": 99},
            }
        },
    }
    with pytest.raises(click.ClickException, match="preprocessing contract differs"):
        _validate_sampling_source(image, metadata, incompatible, expected_stage="T0")


def test_sigma_compatibility_sidecar_discloses_backward_approximation() -> None:
    clean = _endpoint_provenance(
        sigma_min=0.0, direction="backward", residual_consent=False
    )
    assert clean["source_endpoint_approximation"] is False
    compatible = _endpoint_provenance(
        sigma_min=1e-4, direction="backward", residual_consent=True
    )
    assert compatible["source_endpoint_approximation"] is True
    assert compatible["decoded_endpoint_contains_sigma_residual"] is True
    assert "not clean-endpoint" in compatible["endpoint_note"]
