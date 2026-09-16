from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

torch = pytest.importorskip("torch")

from ispy2_symmflow.cli import main
from ispy2_symmflow.evaluation.cohort import evaluate_cohort
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    bind_cached_pair_manifest_provenance,
)
from ispy2_symmflow.utils.hashing import sha256_file, stable_hash


ROLES = ("pre", "early", "late")
PROVENANCE = json.dumps(
    {
        "generator": "cohort-test",
        "not_patient_data": True,
        "seed": 7,
        "synthetic": True,
    },
    sort_keys=True,
)
PREPROCESSING_SIGNATURE = {
    "kind": "synthetic",
    "provenance": json.loads(PROVENANCE),
    "phase_roles": list(ROLES),
    "output_shape_cdhw": [3, 4, 4, 4],
    "spacing_dhw": [1.0, 1.0, 1.0],
}
CONDITION_SCHEMA_PROVENANCE = {
    "fit_split": "train",
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
}


def _prepared(
    path: Path,
    *,
    patient: str,
    stage: str,
    split: str,
    value: float,
    affine_offset: float = 0.0,
) -> Path:
    affine = np.eye(4, dtype=np.float64)
    affine[0, 3] = affine_offset
    np.savez_compressed(
        path,
        image=np.full((3, 4, 4, 4), value, dtype=np.float32),
        affine_lps=affine,
        spacing_dhw=np.ones(3, dtype=np.float32),
        phase_roles=np.asarray(ROLES),
        patient_id=np.asarray(patient),
        visit_id=np.asarray(f"{patient}-{stage}"),
        visit_stage=np.asarray(stage),
        study_uid=np.asarray(f"study-{patient}-{stage}"),
        split=np.asarray(split),
        provenance_json=np.asarray(PROVENANCE),
    )
    return path


def _metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: archive[key].item() if archive[key].ndim == 0 else archive[key].tolist()
            for key in archive.files
            if key != "image"
        }


def _prediction(
    path: Path,
    *,
    source: Path,
    target: Path,
    direction: str,
    flow_hash: str = "flow-hash",
    baseline_kind: str | None = None,
    sigma_min: float = 0.0,
) -> Path:
    with np.load(target, allow_pickle=False) as archive:
        target_image = np.asarray(archive["image"], dtype=np.float32)
    samples = (
        target_image[None, None]
        if baseline_kind == "deterministic"
        else np.stack((np.zeros_like(target_image), target_image), axis=0)[:, None]
    )
    np.savez_compressed(path, samples=samples)
    source_metadata = _metadata(source)
    metadata = {
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "source_metadata": source_metadata,
        "source_phase_roles": list(ROLES),
        "source_preprocessing_provenance": source_metadata["provenance_json"],
        "source_preprocessing_signature": PREPROCESSING_SIGNATURE,
        "conditions": {"stage_i": "T0", "stage_j": "T1"},
        "autoencoder_checkpoint_sha256": "ae-hash",
        "training_config_fingerprint": "config-hash",
        "training_split_hash": "pending",
        "training_signature": {},
        "condition_schema": {"categorical": ["stage_i", "stage_j"]},
        "condition_schema_provenance": json.loads(
            json.dumps(CONDITION_SCHEMA_PROVENANCE)
        ),
        "solver": "none" if baseline_kind == "deterministic" else "heun",
        "steps": 0 if baseline_kind == "deterministic" else 4,
        "sigma_min": 0.0 if baseline_kind == "deterministic" else sigma_min,
        "endpoint_semantics": (
            "paper_clean_endpoints"
            if sigma_min == 0 or baseline_kind == "deterministic"
            else "upstream_sigma_compatibility_experiment"
        ),
        "residual_endpoint_consent": bool(
            sigma_min > 0 and baseline_kind != "deterministic"
        ),
        "source_endpoint_approximation": bool(
            sigma_min > 0 and baseline_kind != "deterministic" and direction == "backward"
        ),
        "decoded_endpoint_contains_sigma_residual": bool(
            sigma_min > 0 and baseline_kind != "deterministic"
        ),
    }
    if baseline_kind is None:
        metadata.update(
            flow_checkpoint_sha256=flow_hash,
            checkpoint_variant=flow_hash,
            branch_order=["later", "earlier"],
        )
    else:
        metadata.update(
            model_family="baseline",
            baseline_kind=baseline_kind,
            baseline_checkpoint_sha256=f"{baseline_kind}-hash",
            checkpoint_variant=baseline_kind,
            direction="forward",
            time_pair=["T0", "T1"],
            branch_order=["later"],
        )
    if sigma_min > 0 and baseline_kind != "deterministic":
        metadata["endpoint_note"] = "residual endpoint compatibility test"
    sidecar = {
        "array_file": path.name,
        "array_sha256": sha256_file(path),
        "direction": direction,
        "nfe_per_sample": 0 if baseline_kind == "deterministic" else 8,
        "seeds": [] if baseline_kind == "deterministic" else [1, 2],
        "metadata": metadata,
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return path


def _case(
    tmp_path: Path,
    *,
    patient: str,
    split: str = "val",
    direction: str = "forward",
    flow_hash: str = "flow-hash",
    affine_offset: float = 0.0,
    baseline_kind: str | None = None,
    sigma_min: float = 0.0,
) -> dict[str, object]:
    earlier = _prepared(
        tmp_path / f"{patient}-T0.npz",
        patient=patient,
        stage="T0",
        split=split,
        value=0.25,
        affine_offset=affine_offset,
    )
    later = _prepared(
        tmp_path / f"{patient}-T1.npz",
        patient=patient,
        stage="T1",
        split=split,
        value=0.75,
        affine_offset=affine_offset,
    )
    source, target = (earlier, later) if direction == "forward" else (later, earlier)
    prediction = _prediction(
        tmp_path / f"{patient}-{direction}.npz",
        source=source,
        target=target,
        direction=direction,
        flow_hash=flow_hash,
        baseline_kind=baseline_kind,
        sigma_min=sigma_min,
    )
    return {
        "prediction": str(prediction),
        "target": str(target),
        "source": str(source),
        "direction": direction,
        "time_pair": ["T0", "T1"],
        "split": split,
        "patient": patient,
    }


def _bind_trusted_pairs(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    pair_records: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        source = Path(str(record["source"]))
        target = Path(str(record["target"]))
        source_metadata = _metadata(source)
        target_metadata = _metadata(target)
        earlier, later = (
            (source_metadata, target_metadata)
            if source_metadata["visit_stage"] == "T0"
            else (target_metadata, source_metadata)
        )
        key = (
            str(earlier["patient_id"]),
            str(earlier["visit_id"]),
            str(later["visit_id"]),
        )
        if key in seen:
            continue
        seen.add(key)
        pair_records.append(
            {
                "pair_id": f"pair-{key[0]}",
                "patient_id": key[0],
                "split": str(earlier["split"]),
                "earlier_visit_id": key[1],
                "later_visit_id": key[2],
                "earlier_stage": str(earlier["visit_stage"]),
                "later_stage": str(later["visit_stage"]),
                "earlier_latent_path": f"{key[1]}.latent.npz",
                "later_latent_path": f"{key[2]}.latent.npz",
                "autoencoder_id": "ae-hash",
                "baseline_clinical": dict(
                    record.get("trusted_baseline_clinical", {})
                ),
                "treatment": dict(record.get("trusted_treatment", {})),
                "qc": [],
            }
        )
    pair_records, latent_statistics = bind_cached_pair_manifest_provenance(
        pair_records,
        {
            "autoencoder_id": "ae-hash",
            "mean": [0.0],
            "std": [1.0],
            "source_preprocessing_signature": PREPROCESSING_SIGNATURE,
        },
    )
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in pair_records), encoding="utf-8"
    )
    assignments = {str(record["patient_id"]): str(record["split"]) for record in pair_records}
    split_hash = stable_hash(assignments)
    ordered_hash = stable_hash(pair_records)

    checkpoint_paths: dict[tuple[str, str, float], Path] = {}
    for record in records:
        prediction = Path(str(record["prediction"]))
        sidecar_path = prediction.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        metadata = sidecar["metadata"]
        family = str(metadata.get("model_family", "symmflow"))
        variant = str(metadata.pop("checkpoint_variant"))
        sigma_min = float(metadata["sigma_min"])
        key = (family, variant, sigma_min)
        checkpoint_path = checkpoint_paths.get(key)
        config = {
            "data": {"time_pair": ["T0", "T1"]},
            "flow": {"sigma_min": sigma_min},
        }
        if family == "symmflow":
            signature = {
                "stage": "symmflow",
                "ordered_manifest_fingerprint": ordered_hash,
                "manifest_record_count": len(pair_records),
                "config_fingerprint": stable_hash(config),
                CACHED_PAIR_MANIFEST_FINGERPRINT: latent_statistics[
                    CACHED_PAIR_MANIFEST_FINGERPRINT
                ],
                CACHED_PAIR_MANIFEST_RECORD_COUNT: len(pair_records),
            }
            path_field = "flow_checkpoint"
            hash_field = "flow_checkpoint_sha256"
        else:
            signature = {
                "kind": str(metadata["baseline_kind"]),
                "ordered_manifest_hash": ordered_hash,
                "manifest_record_count": len(pair_records),
                "config_fingerprint": stable_hash(config),
                CACHED_PAIR_MANIFEST_FINGERPRINT: latent_statistics[
                    CACHED_PAIR_MANIFEST_FINGERPRINT
                ],
                CACHED_PAIR_MANIFEST_RECORD_COUNT: len(pair_records),
            }
            path_field = "baseline_checkpoint"
            hash_field = "baseline_checkpoint_sha256"
        if checkpoint_path is None:
            sigma_tag = str(sigma_min).replace(".", "p")
            checkpoint_path = tmp_path / f"{family}-{variant}-sigma-{sigma_tag}.pt"
            torch.save(
                {
                    "format_version": 1,
                    "config": config,
                    "feature_schema": metadata["condition_schema"],
                    "autoencoder_id": "ae-hash",
                    "latent_statistics": latent_statistics,
                    "split_hash": split_hash,
                    "extra": {
                        "training_signature": signature,
                        "schema_provenance": metadata[
                            "condition_schema_provenance"
                        ],
                        "fixture_variant": variant,
                    },
                },
                checkpoint_path,
            )
            checkpoint_paths[key] = checkpoint_path
        metadata.update(
            {
                path_field: str(checkpoint_path.resolve()),
                hash_field: sha256_file(checkpoint_path),
                "training_config_fingerprint": stable_hash(config),
                "training_split_hash": split_hash,
                "training_signature": signature,
            }
        )
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return pair_manifest


def test_cohort_separates_groups_and_stochastic_summary_types(tmp_path: Path) -> None:
    records = [
        _case(tmp_path, patient="P1"),
        _case(tmp_path, patient="P2"),
        _case(tmp_path, patient="P3", direction="backward"),
    ]
    pair_manifest = _bind_trusted_pairs(tmp_path, records)

    report = evaluate_cohort(
        records,
        pair_manifest=pair_manifest,
        data_range=2.0,
        bootstrap_samples=100,
        bootstrap_seed=3,
    )

    assert report["split"] == "val"
    assert report["patient_count"] == 3
    assert [(group["direction"], group["time_pair"]) for group in report["groups"]] == [
        ("backward", ["T0", "T1"]),
        ("forward", ["T0", "T1"]),
    ]
    forward = report["groups"][1]
    assert forward["predictive_mean"]["mae"]["patient_count"] == 2
    assert forward["candidate_distribution"]["mae"][
        "case_candidate_mean_patient_bootstrap"
    ]["patient_count"] == 2
    assert forward["oracle"]["mae"]["patient_count"] == 2
    assert forward["copy_source_no_change"]["case_count"] == 2
    assert "target-selected" in forward["oracle"]["warning"]
    assert report["endpoint_provenance"]["source_endpoint_approximation_case_count"] == 0
    assert report["sampling_provenance"]["source_sha256"] == sha256_file(
        Path(str(records[0]["source"]))
    )
    assert {
        binding["source_sha256"] for binding in report["source_archive_bindings"]
    } == {sha256_file(Path(str(record["source"]))) for record in records}
    json.dumps(report, allow_nan=False)


def test_cohort_rejects_training_data(tmp_path: Path) -> None:
    record = _case(tmp_path, patient="P1", split="train")
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])
    with pytest.raises(ValueError, match="held-out"):
        evaluate_cohort([record], pair_manifest=pair_manifest, data_range=2.0)


def test_cohort_requires_one_model_provenance(tmp_path: Path) -> None:
    records = [
        _case(tmp_path, patient="P1", flow_hash="first"),
        _case(tmp_path, patient="P2", flow_hash="second"),
    ]
    pair_manifest = _bind_trusted_pairs(tmp_path, records)
    with pytest.raises(ValueError, match="one sampling/model provenance"):
        evaluate_cohort(records, pair_manifest=pair_manifest, data_range=2.0)


def test_cohort_rejects_source_target_grid_mismatch(tmp_path: Path) -> None:
    record = _case(tmp_path, patient="P1")
    target = Path(str(record["target"]))
    _prepared(
        target,
        patient="P1",
        stage="T1",
        split="val",
        value=0.75,
        affine_offset=10.0,
    )
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])
    with pytest.raises(ValueError, match="physical grids differ"):
        evaluate_cohort([record], pair_manifest=pair_manifest, data_range=2.0)


def test_cohort_loads_sidecar_source_when_record_omits_it(tmp_path: Path) -> None:
    record = _case(tmp_path, patient="P1")
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])
    expected_source = str(Path(str(record.pop("source"))).resolve())

    report = evaluate_cohort([record], pair_manifest=pair_manifest, data_range=2.0)

    assert report["cases"][0]["source"] == expected_source
    assert report["groups"][0]["copy_source_no_change"]["case_count"] == 1


def test_cohort_rejects_tampered_sidecar_source_archive(tmp_path: Path) -> None:
    record = _case(tmp_path, patient="P1")
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])
    source = Path(str(record.pop("source")))
    _prepared(source, patient="P9", stage="T0", split="val", value=0.25)
    sidecar_path = Path(str(record["prediction"])).with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["metadata"]["source_sha256"] = sha256_file(source)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="recorded source patient_id differs"):
        evaluate_cohort([record], pair_manifest=pair_manifest, data_range=2.0)


def test_cohort_rejects_replaced_source_image_with_unchanged_metadata(
    tmp_path: Path,
) -> None:
    record = _case(tmp_path, patient="P1")
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])
    source = Path(str(record["source"]))
    _prepared(source, patient="P1", stage="T0", split="val", value=0.5)

    with pytest.raises(ValueError, match="source archive SHA-256 differs"):
        evaluate_cohort([record], pair_manifest=pair_manifest, data_range=2.0)


def test_cohort_rejects_prediction_changed_after_sampling(tmp_path: Path) -> None:
    record = _case(tmp_path, patient="P1")
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])
    prediction = Path(str(record["prediction"]))
    np.savez_compressed(
        prediction,
        samples=np.full((2, 1, 3, 4, 4, 4), 0.125, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="array SHA-256 differs"):
        evaluate_cohort([record], pair_manifest=pair_manifest, data_range=2.0)


def test_cohort_rejects_conditions_that_differ_from_trusted_pair(tmp_path: Path) -> None:
    record = _case(tmp_path, patient="P1")
    record["trusted_baseline_clinical"] = {"age": 50}
    sidecar_path = Path(str(record["prediction"])).with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["metadata"]["conditions"]["age"] = 99
    sidecar["metadata"]["condition_schema"] = {
        "version": 1,
        "token_dim": 8,
        "categorical_fields": [
            {"name": "stage_i", "categories": ["T0"]},
            {"name": "stage_j", "categories": ["T1"]},
        ],
        "numeric_fields": [{"name": "age", "mean": 50.0, "std": 1.0}],
    }
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])

    with pytest.raises(ValueError, match="conditions differ.*age"):
        evaluate_cohort([record], pair_manifest=pair_manifest, data_range=2.0)


def test_cohort_treats_trusted_values_unavailable_in_training_as_missing(
    tmp_path: Path,
) -> None:
    record = _case(tmp_path, patient="P1")
    record["trusted_baseline_clinical"] = {"hr_status": "positive"}
    sidecar_path = Path(str(record["prediction"])).with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["metadata"]["conditions"]["hr_status"] = None
    sidecar["metadata"]["condition_schema"] = {
        "version": 1,
        "token_dim": 8,
        "categorical_fields": [
            {"name": "stage_i", "categories": ["T0"]},
            {"name": "stage_j", "categories": ["T1"]},
            {"name": "hr_status", "categories": ["negative", "positive"]},
        ],
        "numeric_fields": [],
    }
    provenance = json.loads(json.dumps(CONDITION_SCHEMA_PROVENANCE))
    provenance["unavailable_fields"] = ["hr_status"]
    provenance["train_observed_coverage"]["hr_status"] = {
        "kind": "categorical",
        "observation_count": 0,
        "observed_categories": [],
    }
    sidecar["metadata"]["condition_schema_provenance"] = provenance
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])

    report = evaluate_cohort([record], pair_manifest=pair_manifest, data_range=2.0)

    assert report["cases"][0]["conditions_verified_against_trusted_pair"] is True


def test_cohort_rejects_condition_schema_provenance_changed_after_sampling(
    tmp_path: Path,
) -> None:
    record = _case(tmp_path, patient="P1")
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])
    sidecar_path = Path(str(record["prediction"])).with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["metadata"]["condition_schema_provenance"]["unavailable_fields"] = [
        "hr_status"
    ]
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="schema provenance differs"):
        evaluate_cohort([record], pair_manifest=pair_manifest, data_range=2.0)


def test_cohort_rejects_pair_manifest_that_differs_from_checkpoint(tmp_path: Path) -> None:
    record = _case(tmp_path, patient="P1")
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])
    pair = json.loads(pair_manifest.read_text(encoding="utf-8"))
    pair["note"] = "changed after training"
    pair_manifest.write_text(json.dumps(pair) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ordered manifest fingerprint"):
        evaluate_cohort([record], pair_manifest=pair_manifest, data_range=2.0)


def test_cohort_rejects_source_signature_that_differs_from_sidecar(tmp_path: Path) -> None:
    record = _case(tmp_path, patient="P1")
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])
    sidecar_path = Path(str(record["prediction"])).with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["metadata"]["source_preprocessing_signature"]["spacing_dhw"] = [2.0, 1.0, 1.0]
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from the source archive"):
        evaluate_cohort([record], pair_manifest=pair_manifest, data_range=2.0)


def test_cohort_rejects_invalid_solver_nfe(tmp_path: Path) -> None:
    record = _case(tmp_path, patient="P1")
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])
    sidecar_path = Path(str(record["prediction"])).with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["nfe_per_sample"] = 7
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="must record nfe_per_sample=8"):
        evaluate_cohort([record], pair_manifest=pair_manifest, data_range=2.0)


def test_cohort_preserves_residual_endpoint_disclosure(tmp_path: Path) -> None:
    record = _case(
        tmp_path, patient="P1", direction="backward", sigma_min=0.05
    )
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])

    report = evaluate_cohort([record], pair_manifest=pair_manifest, data_range=2.0)

    assert report["endpoint_provenance"] == {
        "endpoint_semantics": "upstream_sigma_compatibility_experiment",
        "residual_endpoint_consent": True,
        "decoded_endpoint_contains_sigma_residual": True,
        "source_endpoint_approximation_case_count": 1,
    }
    assert report["cases"][0]["endpoint_provenance"][
        "source_endpoint_approximation"
    ] is True


def test_evaluation_commands_are_registered() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "evaluate-cohort" in result.output
    assert "evaluate-autoencoder" in result.output


def test_cohort_cli_writes_strict_json(tmp_path: Path) -> None:
    record = _case(tmp_path, patient="P1")
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])
    manifest = tmp_path / "cohort.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output = tmp_path / "report.json"
    config = Path(__file__).parents[1] / "configs" / "smoke.yaml"

    result = CliRunner().invoke(
        main,
        [
            "evaluate-cohort",
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--pair-manifest",
            str(pair_manifest),
            "--output",
            str(output),
            "--bootstrap-samples",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8"))["case_count"] == 1
    assert "NaN" not in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("baseline_kind", ["deterministic", "unidirectional_cfm"])
def test_cohort_accepts_formal_forward_baseline_sidecars(
    tmp_path: Path, baseline_kind: str
) -> None:
    record = _case(tmp_path, patient="P1", baseline_kind=baseline_kind)
    pair_manifest = _bind_trusted_pairs(tmp_path, [record])

    report = evaluate_cohort(
        [record], pair_manifest=pair_manifest, data_range=2.0, bootstrap_samples=10
    )

    assert report["sampling_provenance"]["model_family"] == "baseline"
    assert report["sampling_provenance"]["baseline_kind"] == baseline_kind
    assert len(report["sampling_provenance"]["baseline_checkpoint_sha256"]) == 64
    assert report["groups"][0]["direction"] == "forward"


def test_cohort_does_not_mix_symmflow_and_baseline_provenance(tmp_path: Path) -> None:
    records = [
        _case(tmp_path, patient="P1"),
        _case(tmp_path, patient="P2", baseline_kind="deterministic"),
    ]
    pair_manifest = _bind_trusted_pairs(tmp_path, records)

    with pytest.raises(ValueError, match="one sampling/model provenance"):
        evaluate_cohort(records, pair_manifest=pair_manifest, data_range=2.0)
