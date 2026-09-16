from __future__ import annotations

import csv
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.training.datasets import LatentPairDataset, pair_conditions, read_jsonl
from ispy2_symmflow.data.mewm import (
    MewmConnectedPairGeometry,
    MewmMetadataValidationResult,
    MewmVisitGeometry,
)
import ispy2_symmflow.training.mewm_import as mewm_import_module
from ispy2_symmflow.training.mewm_import import (
    CACHE_SCHEMA,
    COORDINATE_FRAME,
    CROP_POLICY,
    LATENT_SHAPE,
    NORMALIZATION,
    NUMERIC_CONTRACT,
    PAYLOAD_SCHEMA,
    STORED_REPRESENTATION,
    import_mewm_continuous_latents,
)
from ispy2_symmflow.training.provenance import validate_cached_latent_provenance
from ispy2_symmflow.utils.hashing import sha256_file, stable_hash


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(autouse=True)
def _physical_geometry_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(
        bundle_dir,
        upstream_metadata_dir,
        *,
        time_pairs=None,
        earlier_stage=None,
        later_stage=None,
        selected_pair_ids=None,
    ):
        root = Path(bundle_dir)
        assert Path(upstream_metadata_dir).is_dir()
        _, visits = mewm_import_module._source_visits(root / "visits.csv")
        rows = mewm_import_module._pair_rows(
            root / "transitions.csv",
            visits,
            time_pairs=tuple(time_pairs),
            selected_pair_ids=selected_pair_ids,
        )
        affine = (
            (0.0, 0.0, 0.7, 10.0),
            (0.0, 0.7, 0.0, 20.0),
            (2.0, 0.0, 0.0, 30.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        by_visit = {}
        for row in rows:
            for visit_id in (row["source_visit_id"], row["target_visit_id"]):
                visit = visits[visit_id]
                by_visit[visit_id] = MewmVisitGeometry(
                    visit_id=visit_id,
                    patient_id=visit.patient_id,
                    visit_stage=visit.stage,
                    split=visit.split,
                    study_uid=visit.study_uid,
                    affine_lps=affine,
                    base_affine_lps=affine,
                    spacing_dhw=(2.0, 0.7, 0.7),
                    crop_start_zyx=(1, 2, 3),
                    output_shape_zyx=(96, 256, 256),
                    meta_sha256="3" * 64,
                    registration_sha256="4" * 64,
                    transform_artifacts=(("registration.json", "5" * 64),),
                    source_geometry_sha256="6" * 64,
                    registered_grid_geometry_sha256="7" * 64,
                )
        pairs = tuple(
            MewmConnectedPairGeometry(
                pair_id=row["transition_id"],
                patient_id=row["patient_id"],
                split=row["fold"],
                earlier_stage=row["source_visit"],
                later_stage=row["target_visit"],
                earlier_visit_id=row["source_visit_id"],
                later_visit_id=row["target_visit_id"],
                delta_days=int(row["delta_days"]),
                action_text=row["action_text"],
                edge_transition_ids=(row["transition_id"],),
                earlier=by_visit[row["source_visit_id"]],
                later=by_visit[row["target_visit_id"]],
            )
            for row in rows
        )
        document = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
        artifacts = {
            name: value["sha256"] for name, value in document["artifacts"].items()
        }
        return MewmMetadataValidationResult(
            bundle_contract_sha256=document["bundle_contract_sha256"],
            bundle_json_sha256=sha256_file(root / "bundle.json"),
            artifact_sha256=MappingProxyType(artifacts),
            time_pairs=tuple(tuple(pair) for pair in time_pairs),
            output_shape_zyx=(96, 256, 256),
            spacing_dhw=(2.0, 0.7, 0.7),
            visits=tuple(by_visit.values()),
            pairs=pairs,
            visit_geometry_by_id=MappingProxyType(by_visit),
            pair_geometry_by_endpoints=MappingProxyType(
                {
                    (pair.earlier_visit_id, pair.later_visit_id): pair
                    for pair in pairs
                }
            ),
        )

    monkeypatch.setattr(mewm_import_module, "validate_mewm_bundle_metadata", validate)


def _fixture(
    tmp_path: Path,
    *,
    target_is_condition: bool = False,
    all_stages: bool = False,
) -> dict[str, Path]:
    bundle_dir = tmp_path / "bundle"
    latent_dir = tmp_path / "source" / "visits"
    bundle_dir.mkdir()
    latent_dir.mkdir(parents=True)
    metadata_dir = tmp_path / "upstream_metadata"
    metadata_dir.mkdir()
    checkpoint = tmp_path / "vqgan.ckpt"
    checkpoint.write_bytes(b"audited-vqgan-checkpoint")
    vqgan_sha = sha256_file(checkpoint)

    visits: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    stage_dates = (
        (("T0", "2020-01-01"), ("T1", "2020-02-01"), ("T2", "2020-03-01"), ("T3", "2020-04-01"))
        if all_stages
        else (("T0", "2020-01-01"), ("T1", "2020-02-01"))
    )
    values: dict[str, float] = {}
    for patient, split in (("train-p", "train"), ("val-p", "val")):
        for stage_index, (stage, day) in enumerate(stage_dates):
            visit_id = f"{patient}:{stage}"
            values[visit_id] = (
                1.0 + 2.0 * stage_index
                if split == "train"
                else -1.0 + 2.0 * stage_index
            )
            visits.append(
                {
                    "patient_id": patient,
                    "visit_id": visit_id,
                    "visit": stage,
                    "study_instance_uid": f"study-{visit_id}",
                    "visit_date": day,
                    "visit_date_source": "verified_fixture_dates",
                    "fold": split,
                    "HR": "1",
                    "HER2": "0",
                    "MP": "1",
                    "Age_at_Screening": "42",
                    "menopausal_status": "pre",
                    "trial_arm": "Arm A",
                    "registration_status": "fixed_reference" if stage == "T0" else "rigid_fallback",
                    "quality_pass": "True",
                }
            )
        for edge_index, ((source_stage, source_date), (target_stage, target_date)) in enumerate(
            zip(stage_dates, stage_dates[1:], strict=False)
        ):
            delta_days = (np.datetime64(target_date) - np.datetime64(source_date)).astype(int)
            transitions.append({
                "transition_id": f"{patient}:{source_stage}->{target_stage}",
                "patient_id": patient,
                "fold": split,
                "transition_type": f"{source_stage}->{target_stage}",
                "source_visit_id": f"{patient}:{source_stage}",
                "target_visit_id": f"{patient}:{target_stage}",
                "source_visit": source_stage,
                "target_visit": target_stage,
                "source_study_instance_uid": f"study-{patient}:{source_stage}",
                "target_study_instance_uid": f"study-{patient}:{target_stage}",
                "delta_days": str(delta_days),
                "source_ftv_volume_cc": str(2.0 - 0.25 * edge_index),
                "target_ftv_volume_cc": str(1.0 - 0.25 * edge_index),
                "source_ftv_is_condition": "True",
                "target_ftv_is_condition": str(target_is_condition),
                "target_ftv_is_audit_only": "True",
                "action_text": "treatment arm Arm A",
            })
    visits_path = bundle_dir / "visits.csv"
    transitions_path = bundle_dir / "transitions.csv"
    preprocess_path = bundle_dir / "preprocess.json"
    normalization_path = bundle_dir / "normalization.json"
    _write_csv(visits_path, visits)
    _write_csv(transitions_path, transitions)
    preprocess_path.write_text(
        json.dumps(
            {
                "coordinate_frame": COORDINATE_FRAME,
                "crop_policy": CROP_POLICY,
                "output_shape_zyx": [96, 256, 256],
            }
        ),
        encoding="utf-8",
    )
    normalization_path.write_text(json.dumps({"schema": "fixture"}), encoding="utf-8")
    artifacts = {}
    for name, path in (
        ("visits", visits_path),
        ("transitions", transitions_path),
        ("preprocess", preprocess_path),
        ("normalization", normalization_path),
    ):
        artifacts[name] = {"path": path.name, "sha256": sha256_file(path)}
    bundle = {"schema_version": "fixture-bundle", "artifacts": artifacts}
    bundle["bundle_contract_sha256"] = stable_hash(bundle)
    bundle_path = bundle_dir / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")

    source_stats = {
        "mean": [1.0] * 8,
        "std": [2.0] * 8,
        "source_split": "train",
        "variance_estimator": "population_ddof0",
        "element_count_per_channel": len(stage_dates) * 24 * 64 * 64,
        "visit_count": len(stage_dates),
    }
    source_stats["sha256"] = stable_hash(source_stats)
    data_sha = "1" * 64
    codebook_sha = "2" * 64
    visit_ids = [str(row["visit_id"]) for row in visits]
    identity = {
        "schema": CACHE_SCHEMA,
        "payload_schema": PAYLOAD_SCHEMA,
        "stored_representation": STORED_REPRESENTATION,
        "normalization": NORMALIZATION,
        "numeric_contract": NUMERIC_CONTRACT,
        "latent_shape_czyx": list(LATENT_SHAPE),
        "input_shape_zyx": [96, 256, 256],
        "latent_dtype": "float16",
        "latent_statistics": source_stats,
        "vqgan_sha256": vqgan_sha,
        "codebook_sha256": codebook_sha,
        "data_contract_sha256": data_sha,
        "bundle_json_sha256": sha256_file(bundle_path),
        "bundle_contract_sha256": bundle["bundle_contract_sha256"],
        "roi_cache_contract": {"schema": "fixture-roi"},
        "visit_ids": visit_ids,
        "visit_count": len(visit_ids),
    }
    identity_path = tmp_path / "cache_identity.json"
    identity_path.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
    for visit_id, value in values.items():
        patient, stage = visit_id.rsplit(":", 1)
        split = "train" if patient == "train-p" else "val"
        torch.save(
            {
                "schema": PAYLOAD_SCHEMA,
                "visit_id": visit_id,
                "split": split,
                "continuous_latent": torch.full(LATENT_SHAPE, value, dtype=torch.float16),
                "vqgan_sha256": vqgan_sha,
                "codebook_sha256": codebook_sha,
                "data_contract_sha256": data_sha,
                "normalization": NORMALIZATION,
                "latent_statistics_sha256": source_stats["sha256"],
            },
            latent_dir / f"{patient}__{stage}.pt",
        )
    return {
        "identity": identity_path,
        "bundle": bundle_dir,
        "latents": latent_dir,
        "checkpoint": checkpoint,
        "metadata": metadata_dir,
    }


def test_import_mewm_latents_normalizes_both_paired_branches(tmp_path: Path) -> None:
    source = _fixture(tmp_path)
    output = tmp_path / "imported"
    result = import_mewm_continuous_latents(
        source["identity"],
        source["bundle"],
        source["latents"],
        source["checkpoint"],
        output,
        upstream_metadata_dir=source["metadata"],
    )

    assert result.pair_count == 2
    assert result.visit_count == 4
    assert result.split_pair_counts == {"train": 1, "val": 1}
    pairs = read_jsonl(result.pair_manifest_path)
    statistics = json.loads(Path(result.statistics_path).read_text(encoding="utf-8"))
    validate_cached_latent_provenance(pairs, statistics)
    train = LatentPairDataset(pairs, split="train")[0]
    assert torch.all(train["earlier_latent"] == 0)
    assert torch.all(train["later_latent"] == 1)
    assert train["conditions"]["mammaprint"] == "1"
    assert train["conditions"]["hr_status"] == "1"
    assert "source_ftv_volume_cc" not in train["conditions"]
    assert pairs[0]["evaluation_metadata"]["target_ftv_volume_cc"] == 1.0
    assert statistics["decoder_contract"] == "denormalize_then_quantize_then_vqgan_decode"
    assert statistics["source_preprocessing_signature"]["image_shape_czyx"] == [
        1,
        96,
        256,
        256,
    ]
    with np.load(pairs[0]["earlier_latent_path"], allow_pickle=False) as archive:
        assert not np.array_equal(archive["affine_lps"], np.eye(4))
        assert archive["spacing_dhw"].tolist() == pytest.approx([2.0, 0.7, 0.7])
        provenance = json.loads(str(archive["provenance_json"].item()))
    assert provenance["header_affine_semantics"].startswith("physical_lps")


def test_import_mewm_latents_rejects_target_ftv_condition_contract(tmp_path: Path) -> None:
    source = _fixture(tmp_path, target_is_condition=True)
    with pytest.raises(ValueError, match="leaks target FTV"):
        import_mewm_continuous_latents(
            source["identity"],
            source["bundle"],
            source["latents"],
            source["checkpoint"],
            tmp_path / "imported",
            upstream_metadata_dir=source["metadata"],
        )


def test_imported_pair_conditions_do_not_expose_evaluation_metadata(tmp_path: Path) -> None:
    source = _fixture(tmp_path)
    result = import_mewm_continuous_latents(
        source["identity"],
        source["bundle"],
        source["latents"],
        source["checkpoint"],
        tmp_path / "imported",
        upstream_metadata_dir=source["metadata"],
    )
    pair = read_jsonl(result.pair_manifest_path)[0]
    conditions = pair_conditions(pair)
    assert not ({"source_ftv_volume_cc", "target_ftv_volume_cc", "pCR"} & conditions.keys())


def test_import_mewm_latents_derives_all_connected_pairs_from_adjacent_edges(
    tmp_path: Path,
) -> None:
    source = _fixture(tmp_path, all_stages=True)
    result = import_mewm_continuous_latents(
        source["identity"],
        source["bundle"],
        source["latents"],
        source["checkpoint"],
        tmp_path / "imported",
        upstream_metadata_dir=source["metadata"],
        time_pairs=(
            ("T0", "T1"),
            ("T0", "T2"),
            ("T0", "T3"),
            ("T1", "T2"),
            ("T1", "T3"),
            ("T2", "T3"),
        ),
    )
    assert result.pair_count == 12
    pairs = read_jsonl(result.pair_manifest_path)
    by_id = {pair["pair_id"]: pair for pair in pairs}
    composed = by_id["train-p:T0->T3"]
    assert composed["delta_days"] == 91
    assert composed["earlier_stage"] == "T0"
    assert composed["later_stage"] == "T3"
    assert composed["evaluation_metadata"]["source_ftv_volume_cc"] == 2.0
    assert composed["evaluation_metadata"]["target_ftv_volume_cc"] == 0.5
