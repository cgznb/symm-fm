from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.data.manifest import build_pair_manifest, visit_from_dict
from ispy2_symmflow.training.cache import cache_latents
from ispy2_symmflow.training.datasets import read_jsonl
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    LATENT_STATISTICS_FINGERPRINT,
    bind_latent_statistics_fingerprint,
    require_latent_statistics_fingerprint,
    validate_cached_latent_provenance,
)
from ispy2_symmflow.utils.hashing import stable_hash


class CacheAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encode_calls = 0
        self.register_buffer("latent_mean", torch.zeros(1, 1, 1, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, 1, 1, 1, 1))

    def set_latent_statistics(self, mean, std):
        self.latent_mean.copy_(mean.reshape_as(self.latent_mean))
        self.latent_std.copy_(std.reshape_as(self.latent_std))

    def encode(self, image, *, normalize=False):
        self.encode_calls += 1
        latent = image[:, :1]
        return (latent - self.latent_mean) / self.latent_std if normalize else latent


def _preprocessing_provenance(shape, roles, spacing, affine) -> str:
    return json.dumps(
        {
            "version": "1.0",
            "source_only": True,
            "target_content_used": False,
            "crop_basis": "fixed_center",
            "normalization_scope": "global_train_patients_shared_phases",
            "intensity_stats_hash": "unit-test-training-statistics",
            "output_shape_cdhw": list(shape),
            "output_spacing_dhw": list(spacing),
            "output_affine_lps": affine.tolist(),
            "extra": {"phase_roles": list(roles)},
        },
        sort_keys=True,
    )


def _prepared(
    tmp_path,
    visit,
    patient,
    split,
    value,
    *,
    archive_visit=None,
    archive_patient=None,
    archive_split=None,
    shape=(1, 2, 2, 2),
    spacing=(1.0, 1.0, 1.0),
    affine_offset=0.0,
    study_date=None,
):
    path = tmp_path / f"{visit}.npz"
    roles = ("pre",)
    stage = visit.rsplit("-", 1)[-1]
    dates = {"T0": "2020-01-01", "T1": "2020-02-01"}
    stored_visit = archive_visit or visit
    stored_patient = archive_patient or patient
    stored_split = archive_split or split
    affine = np.eye(4, dtype=np.float64)
    affine[0, 3] = affine_offset
    np.savez_compressed(
        path,
        image=np.full(shape, value, np.float32),
        affine_lps=affine,
        spacing_dhw=np.asarray(spacing, dtype=np.float32),
        phase_roles=np.asarray(roles),
        patient_id=np.asarray(stored_patient),
        visit_id=np.asarray(stored_visit),
        study_uid=np.asarray(f"study-{stored_visit}"),
        visit_stage=np.asarray(stored_visit.rsplit("-", 1)[-1]),
        split=np.asarray(stored_split),
        source_series_uids=np.asarray(["series-pre"]),
        source_temporal_positions=np.asarray([0], dtype=np.int32),
        provenance_json=np.asarray(
            _preprocessing_provenance(shape, roles, spacing, affine)
        ),
    )
    return {
        "schema_version": "1.0",
        "visit_id": visit,
        "patient_id": patient,
        "collection": "ISPY2",
        "study_uid": f"study-{visit}",
        "visit_stage": stage,
        "study_date": study_date or dates.get(stage),
        "date_source": "test",
        "relative_date_verified": False,
        "phase_paths": {},
        "baseline_clinical": {"age": 50},
        "treatment": {"treatment_arm": "A"},
        "evaluation_metadata": {},
        "qc": [],
        "series_count": 1,
        "split": split,
        "prepared_path": str(path),
    }


def _pair(records, **changes):
    typed = [visit_from_dict(record) for record in records]
    assignments = {
        str(record["patient_id"]): str(record["split"]) for record in records
    }
    pairs = build_pair_manifest(typed, assignments, require_three_phase=False)
    assert len(pairs) == 1
    value = pairs[0].to_dict()
    value.update(changes)
    return value


def _checkpoint_header(records, *, roles=("pre",)):
    config = {
        "data": {"phase_channels": list(roles)},
        "autoencoder": {"in_channels": len(roles)},
    }
    assignments = {
        str(record["patient_id"]): str(record["split"]) for record in records
    }
    return {
        "config": config,
        "split_hash": stable_hash(assignments),
        "extra": {
            "training_signature": {
                "stage": "autoencoder",
                "ordered_manifest_fingerprint": stable_hash(records),
                "manifest_record_count": len(records),
                "config_fingerprint": stable_hash(config),
            }
        },
    }


def test_cache_statistics_use_train_visits_only(tmp_path) -> None:
    records = [
        _prepared(tmp_path, "a-T0", "a", "train", 0),
        _prepared(tmp_path, "b-T0", "b", "train", 2),
        _prepared(tmp_path, "c-T0", "c", "test", 100),
    ]
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = cache_latents(
        CacheAutoencoder(),
        manifest,
        tmp_path / "latents",
        device=torch.device("cpu"),
        autoencoder_id="ae",
        autoencoder_checkpoint_header=_checkpoint_header(records),
    )
    assert result["statistics"]["mean"] == pytest.approx([1.0])
    assert result["statistics"]["std"] == pytest.approx([1.0])
    assert result["statistics"]["fit_patient_ids"] == ["a", "b"]
    assert result["statistics"]["source_preprocessing_signature"] == {
        "kind": "prepared_patient_volume",
        "version": "1.0",
        "crop_basis": "fixed_center",
        "normalization_scope": "global_train_patients_shared_phases",
        "intensity_stats_hash": "unit-test-training-statistics",
        "phase_roles": ["pre"],
        "output_shape_cdhw": [1, 2, 2, 2],
        "spacing_dhw": [1.0, 1.0, 1.0],
    }


def test_cache_rejects_patient_across_splits(tmp_path) -> None:
    records = [
        _prepared(tmp_path, "a-T0", "a", "train", 0),
        _prepared(tmp_path, "a-T1", "a", "test", 1),
    ]
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="leakage"):
        cache_latents(
            CacheAutoencoder(),
            manifest,
            tmp_path / "latents",
            device=torch.device("cpu"),
            autoencoder_id="ae",
            autoencoder_checkpoint_header=_checkpoint_header(records),
        )


def test_cache_rejects_train_record_redirected_to_val_before_statistics_or_writes(
    tmp_path,
) -> None:
    records = [
        _prepared(
            tmp_path,
            "a-T0",
            "a",
            "train",
            0,
            archive_split="val",
        )
    ]
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    output_dir = tmp_path / "latents"
    autoencoder = CacheAutoencoder()

    with pytest.raises(ValueError, match="split mismatch"):
        cache_latents(
            autoencoder,
            manifest,
            output_dir,
            device=torch.device("cpu"),
            autoencoder_id="ae",
            autoencoder_checkpoint_header=_checkpoint_header(records),
        )

    assert autoencoder.encode_calls == 0
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("archive_keyword", "archive_value", "message"),
    (
        ("archive_patient", "other-patient", "patient_id mismatch"),
        ("archive_visit", "a-T1", "visit_id mismatch"),
    ),
)
def test_cache_rejects_prepared_identity_mismatch_before_encoding(
    tmp_path, archive_keyword, archive_value, message
) -> None:
    record = _prepared(
        tmp_path,
        "a-T0",
        "a",
        "train",
        0,
        **{archive_keyword: archive_value},
    )
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    autoencoder = CacheAutoencoder()

    with pytest.raises(ValueError, match=message):
        cache_latents(
            autoencoder,
            manifest,
            tmp_path / "latents",
            device=torch.device("cpu"),
            autoencoder_id="ae",
            autoencoder_checkpoint_header=_checkpoint_header([record]),
        )

    assert autoencoder.encode_calls == 0


def test_cache_fails_closed_when_all_pairs_have_error_qc(tmp_path) -> None:
    records = [
        _prepared(tmp_path, "a-T0", "a", "train", 0),
        _prepared(tmp_path, "a-T1", "a", "train", 1),
    ]
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(
        json.dumps(
            {
                "pair_id": "a-T0-T1",
                "earlier_visit_id": "a-T0",
                "later_visit_id": "a-T1",
                "qc": [{"severity": "error", "code": "pair_physical_grid_mismatch"}],
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="no QC-passing"):
        cache_latents(
            CacheAutoencoder(),
            manifest,
            tmp_path / "latents",
            device=torch.device("cpu"),
            autoencoder_id="ae",
            autoencoder_checkpoint_header=_checkpoint_header(records),
            pair_manifest=pair_manifest,
        )


def test_cache_reports_rejected_pair_ids_once_in_manifest_order(tmp_path) -> None:
    records = [
        _prepared(tmp_path, "b-T0", "b", "train", 0),
        _prepared(tmp_path, "b-T1", "b", "train", 1),
        _prepared(tmp_path, "a-T0", "a", "train", 2),
        _prepared(tmp_path, "a-T1", "a", "train", 3),
        _prepared(tmp_path, "c-T0", "c", "train", 4),
        _prepared(tmp_path, "c-T1", "c", "train", 5),
    ]
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    error_qc = [{"severity": "ERROR", "code": "rejected_for_test"}]
    rejected_b = _pair(records[0:2], qc=error_qc)
    rejected_a = _pair(records[2:4], qc=error_qc)
    accepted_c = _pair(records[4:6])
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(
        "".join(
            json.dumps(pair) + "\n"
            for pair in (rejected_b, rejected_a, rejected_b, accepted_c)
        ),
        encoding="utf-8",
    )

    result = cache_latents(
        CacheAutoencoder(),
        manifest,
        tmp_path / "latents",
        device=torch.device("cpu"),
        autoencoder_id="ae",
        autoencoder_checkpoint_header=_checkpoint_header(records),
        pair_manifest=pair_manifest,
    )

    assert result["rejected_pair_ids"] == [
        rejected_b["pair_id"],
        rejected_a["pair_id"],
    ]
    assert result["rejected_pair_count"] == 2
    assert result["pair_count"] == 1


def test_cache_rejects_pair_when_endpoint_visit_has_error_qc(tmp_path) -> None:
    records = [
        _prepared(tmp_path, "a-T0", "a", "train", 0),
        _prepared(tmp_path, "a-T1", "a", "train", 1),
    ]
    records[1]["qc"] = [
        {
            "severity": "error",
            "code": "tampered_endpoint",
            "message": "endpoint must not enter paired training",
            "details": {},
        }
    ]
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(json.dumps(_pair(records)) + "\n", encoding="utf-8")
    autoencoder = CacheAutoencoder()
    output = tmp_path / "latents"

    with pytest.raises(ValueError, match="endpoint-QC"):
        cache_latents(
            autoencoder,
            manifest,
            output,
            device=torch.device("cpu"),
            autoencoder_id="ae",
            autoencoder_checkpoint_header=_checkpoint_header(records),
            pair_manifest=pair_manifest,
        )

    assert autoencoder.encode_calls == 0
    assert not output.exists()


def test_cache_rejects_pair_when_rebuilt_pair_has_error_qc(tmp_path) -> None:
    records = [
        _prepared(tmp_path, "a-T0", "a", "train", 0),
        _prepared(
            tmp_path,
            "a-T1",
            "a",
            "train",
            1,
            study_date="2019-12-01",
        ),
    ]
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(
        json.dumps(_pair(records, qc=[])) + "\n", encoding="utf-8"
    )
    autoencoder = CacheAutoencoder()

    with pytest.raises(ValueError, match="endpoint-QC"):
        cache_latents(
            autoencoder,
            manifest,
            tmp_path / "latents",
            device=torch.device("cpu"),
            autoencoder_id="ae",
            autoencoder_checkpoint_header=_checkpoint_header(records),
            pair_manifest=pair_manifest,
        )

    assert autoencoder.encode_calls == 0


@pytest.mark.parametrize(
    "later_grid",
    (
        {"shape": (1, 3, 2, 2)},
        {"spacing": (2.0, 1.0, 1.0)},
        {"affine_offset": 5.0},
    ),
)
def test_cache_rechecks_pair_grid_from_prepared_archives(tmp_path, later_grid) -> None:
    records = [
        _prepared(tmp_path, "a-T0", "a", "train", 0),
        _prepared(tmp_path, "a-T1", "a", "train", 1, **later_grid),
    ]
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    pair = _pair(records)
    assert pair["qc"] == []
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(json.dumps(pair) + "\n", encoding="utf-8")
    autoencoder = CacheAutoencoder()
    output = tmp_path / "latents"

    with pytest.raises(ValueError, match="physical-grid validation"):
        cache_latents(
            autoencoder,
            manifest,
            output,
            device=torch.device("cpu"),
            autoencoder_id="ae",
            autoencoder_checkpoint_header=_checkpoint_header(records),
            pair_manifest=pair_manifest,
        )

    assert autoencoder.encode_calls == 0
    assert not output.exists()


@pytest.mark.parametrize(
    ("header_change", "message"),
    (
        ("split", "split_hash differs"),
        ("order", "ordered manifest fingerprint differs"),
        ("roles", "phase roles differ"),
    ),
)
def test_cache_rejects_autoencoder_checkpoint_from_different_training_data(
    tmp_path, header_change, message
) -> None:
    records = [
        _prepared(tmp_path, "a-T0", "a", "train", 0),
        _prepared(tmp_path, "b-T0", "b", "val", 1),
    ]
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    if header_change == "order":
        header = _checkpoint_header(list(reversed(records)))
    elif header_change == "roles":
        header = _checkpoint_header(records, roles=("early",))
    else:
        header = _checkpoint_header(records)
        header["split_hash"] = "different-split"
    autoencoder = CacheAutoencoder()
    output = tmp_path / "latents"

    with pytest.raises(ValueError, match=message):
        cache_latents(
            autoencoder,
            manifest,
            output,
            device=torch.device("cpu"),
            autoencoder_id="ae",
            autoencoder_checkpoint_header=header,
        )

    assert autoencoder.encode_calls == 0
    assert not output.exists()


def test_cache_binds_statistics_fingerprint_to_every_artifact(tmp_path) -> None:
    records = [
        _prepared(tmp_path, "a-T0", "a", "train", 0),
        _prepared(tmp_path, "a-T1", "a", "train", 2),
    ]
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(
        json.dumps(_pair(records)) + "\n",
        encoding="utf-8",
    )
    result = cache_latents(
        CacheAutoencoder(),
        manifest,
        tmp_path / "latents",
        device=torch.device("cpu"),
        autoencoder_id="ae",
        autoencoder_checkpoint_header=_checkpoint_header(records),
        pair_manifest=pair_manifest,
    )

    fingerprint = require_latent_statistics_fingerprint(result["statistics"])
    stored_statistics = json.loads(
        (tmp_path / "latents" / "latent_statistics.json").read_text(
            encoding="utf-8"
        )
    )
    assert require_latent_statistics_fingerprint(stored_statistics) == fingerprint
    pair_fingerprint = stored_statistics[CACHED_PAIR_MANIFEST_FINGERPRINT]
    pair_record_count = stored_statistics[CACHED_PAIR_MANIFEST_RECORD_COUNT]
    assert pair_record_count == 1
    visits = read_jsonl(result["visit_manifest"])
    pairs = read_jsonl(result["pair_manifest"])
    assert all(
        record[LATENT_STATISTICS_FINGERPRINT] == fingerprint for record in visits
    )
    assert pairs[0][LATENT_STATISTICS_FINGERPRINT] == fingerprint
    assert pairs[0][CACHED_PAIR_MANIFEST_FINGERPRINT] == pair_fingerprint
    assert pairs[0][CACHED_PAIR_MANIFEST_RECORD_COUNT] == pair_record_count
    for record in visits:
        assert record[CACHED_PAIR_MANIFEST_FINGERPRINT] == pair_fingerprint
        with np.load(record["latent_path"], allow_pickle=False) as archive:
            assert str(archive[LATENT_STATISTICS_FINGERPRINT].item()) == fingerprint
            assert (
                str(archive[CACHED_PAIR_MANIFEST_FINGERPRINT].item())
                == pair_fingerprint
            )
            assert int(archive[CACHED_PAIR_MANIFEST_RECORD_COUNT].item()) == 1
    assert validate_cached_latent_provenance(pairs, stored_statistics) == fingerprint

    foreign_statistics = bind_latent_statistics_fingerprint(
        {**stored_statistics, "mean": [99.0]}
    )
    with pytest.raises(ValueError, match="differs from latent_statistics.json"):
        validate_cached_latent_provenance(pairs, foreign_statistics)


@pytest.mark.parametrize("tamper", ("conditions", "endpoints"))
def test_cached_pair_binding_rejects_post_cache_contract_tampering(
    tmp_path, tamper
) -> None:
    records = [
        _prepared(tmp_path, "a-T0", "a", "train", 0),
        _prepared(tmp_path, "a-T1", "a", "train", 2),
    ]
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(json.dumps(_pair(records)) + "\n", encoding="utf-8")
    result = cache_latents(
        CacheAutoencoder(),
        manifest,
        tmp_path / "latents",
        device=torch.device("cpu"),
        autoencoder_id="ae",
        autoencoder_checkpoint_header=_checkpoint_header(records),
        pair_manifest=pair_manifest,
    )
    pairs = read_jsonl(result["pair_manifest"])
    if tamper == "conditions":
        pairs[0]["baseline_clinical"]["age"] = 999
    else:
        pairs[0]["earlier_visit_id"], pairs[0]["later_visit_id"] = (
            pairs[0]["later_visit_id"],
            pairs[0]["earlier_visit_id"],
        )
        pairs[0]["earlier_latent_path"], pairs[0]["later_latent_path"] = (
            pairs[0]["later_latent_path"],
            pairs[0]["earlier_latent_path"],
        )

    with pytest.raises(ValueError, match="pair manifest fingerprint differs"):
        validate_cached_latent_provenance(pairs, result["statistics"])


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("patient_id", "other-patient"),
        ("pair_id", "tampered-pair-id"),
        ("split", "val"),
        ("later_stage", "T2"),
        ("earlier_prepared_path", "/wrong/earlier.npz"),
        ("later_prepared_path", "/wrong/later.npz"),
        ("baseline_clinical", {"age": 99}),
        ("treatment", {"treatment_arm": "B"}),
        ("delta_days", 31),
        ("observed_delta_days", 99),
        ("interval_missing", False),
        ("interval_source", "verified_relative_dicom_study_date"),
    ),
)
def test_cache_rejects_pair_fields_relabelled_against_prepared_visits(
    tmp_path, field, changed
) -> None:
    records = [
        _prepared(tmp_path, "a-T0", "a", "train", 0),
        _prepared(tmp_path, "a-T1", "a", "train", 2),
    ]
    manifest = tmp_path / "visits.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    pair_manifest = tmp_path / "pairs.jsonl"
    pair_manifest.write_text(
        json.dumps(_pair(records, **{field: changed})) + "\n", encoding="utf-8"
    )
    autoencoder = CacheAutoencoder()
    output = tmp_path / "latents"

    with pytest.raises(ValueError, match="pair"):
        cache_latents(
            autoencoder,
            manifest,
            output,
            device=torch.device("cpu"),
            autoencoder_id="ae",
            autoencoder_checkpoint_header=_checkpoint_header(records),
            pair_manifest=pair_manifest,
        )

    assert autoencoder.encode_calls == 0
    assert not output.exists()
