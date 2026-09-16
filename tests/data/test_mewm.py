from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import ispy2_symmflow.data.mewm as mewm
from ispy2_symmflow.training.datasets import read_jsonl
from ispy2_symmflow.utils.hashing import sha256_file, stable_hash


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: set[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _visit(patient: str, stage: str, split: str, index: int) -> dict[str, str]:
    visit_id = f"{patient}:{stage}"
    date_value = "2020-01-01" if stage == "T0" else "2020-02-01"
    arm = f"arm {patient}"
    result = {key: "" for key in mewm._VISIT_COLUMNS}
    result.update(
        {
            "patient_id": patient,
            "visit_id": visit_id,
            "visit": stage,
            "study_instance_uid": f"1.2.840.{index}.{stage[-1]}",
            "visit_date": date_value,
            "visit_date_source": "verified_test_date",
            "qc_status": "ok",
            "dce0_path": f"/remote/{patient}/{stage}/dce0.nii.gz",
            "ser_path": f"/remote/{patient}/{stage}/ser.nii.gz",
            "mask_path": f"/remote/{patient}/{stage}/mask.nii.gz",
            "meta_path": f"/remote/{patient}/{stage}/meta.json",
            "row_spacing_mm": "1.0",
            "column_spacing_mm": "1.0",
            "slice_spacing_mm": "2.0",
            "ftv_volume_cc": str(float(index + 1)),
            "HR": str(index % 2),
            "HER2": str((index + 1) % 2),
            "MP": str(index % 2),
            "Age_at_Screening": str(40 + index),
            "menopausal_status": (
                "Premenopausal(<6 months since LMP AND no prior bilateral "
                "ovariectomy AND not on estrogen replacement)"
            ),
            "trial_arm": arm,
            "clinical_text": f"clinical {patient}",
            "fold": split,
            "orientation_lps_json": json.dumps(np.eye(3).tolist()),
            "resampled_shape_zyx_json": "[4,5,6]",
            "crop_start_zyx_json": f"[{index},0,0]",
            "registration_status": "fixed_reference" if stage == "T0" else "rigid_fallback",
            "quality_pass": "True",
        }
    )
    return result


def _transition(earlier: dict[str, str], later: dict[str, str]) -> dict[str, str]:
    patient = earlier["patient_id"]
    result = {key: "" for key in mewm._TRANSITION_COLUMNS}
    result.update(
        {
            "transition_id": f"{patient}:T0->T1",
            "patient_id": patient,
            "fold": earlier["fold"],
            "transition_type": "T0->T1",
            "source_visit_id": earlier["visit_id"],
            "target_visit_id": later["visit_id"],
            "source_visit": "T0",
            "target_visit": "T1",
            "source_study_instance_uid": earlier["study_instance_uid"],
            "target_study_instance_uid": later["study_instance_uid"],
            "delta_days": "31",
            "source_dce0_path": earlier["dce0_path"],
            "source_ser_path": earlier["ser_path"],
            "source_mask_path": earlier["mask_path"],
            "source_meta_path": earlier["meta_path"],
            "target_dce0_path": later["dce0_path"],
            "target_mask_path": later["mask_path"],
            "target_meta_path": later["meta_path"],
            "source_ftv_volume_cc": earlier["ftv_volume_cc"],
            "target_ftv_volume_cc": later["ftv_volume_cc"],
            "source_ftv_is_condition": "True",
            "target_ftv_is_condition": "False",
            "target_ftv_is_audit_only": "True",
            "clinical_text": earlier["clinical_text"],
            "action_text": f"treatment arm {earlier['trial_arm']}",
        }
    )
    return result


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    patient_splits = {"P1": "train", "P2": "train", "P3": "val"}
    visits: list[dict[str, str]] = []
    transitions: list[dict[str, str]] = []
    crop_plans: dict[str, dict[str, Any]] = {}
    for index, (patient, split) in enumerate(patient_splits.items()):
        earlier = _visit(patient, "T0", split, index)
        later = _visit(patient, "T1", split, index)
        visits.extend((earlier, later))
        transitions.append(_transition(earlier, later))
        crop_plans[patient] = {
            "coordinate_frame": mewm.COORDINATE_FRAME,
            "input_shape_zyx": [4, 5, 6],
            "output_shape_zyx": [96, 256, 256],
            "bbox_min_zyx": [1, 1, 1],
            "bbox_max_zyx": [2, 3, 4],
            "center_zyx": [1, 2, 2],
            "crop_start_zyx": [index, 0, 0],
            "t0_mask_voxel_count": 8,
        }
    exclusions = [
        {
            "transition_id": "EXCLUDED:T0->T1",
            "patient_id": "EXCLUDED",
            "source_visit_id": "EXCLUDED:T0",
            "target_visit_id": "EXCLUDED:T1",
            "transition_type": "T0->T1",
            "reason": "quality_failed",
            "detail": "{}",
        }
    ]
    _write_csv(root / "visits.csv", visits, set(mewm._VISIT_COLUMNS))
    _write_csv(root / "transitions.csv", transitions, set(mewm._TRANSITION_COLUMNS))
    _write_csv(root / "exclusions.csv", exclusions, set(mewm._EXCLUSION_COLUMNS))
    (root / "crop_plans.json").write_text(
        json.dumps(crop_plans, sort_keys=True), encoding="utf-8"
    )
    (root / "split.json").write_text(
        json.dumps({"train": ["P1", "P2"], "val": ["P3"]}, sort_keys=True),
        encoding="utf-8",
    )
    normalization = {
        "schema": mewm.NORMALIZATION_SCHEMA,
        "channels": {
            "dce0": {"count": 100, "mean": 1.0, "std": 2.0},
            "ser": {"count": 100, "mean": 3.0, "std": 4.0},
        },
        "visit_ids": ["P1:T0", "P1:T1", "P2:T0", "P2:T1"],
    }
    (root / "normalization.json").write_text(
        json.dumps(normalization, sort_keys=True), encoding="utf-8"
    )
    normalization_sha = sha256_file(root / "normalization.json")
    preprocess = {
        "schema": mewm.PREPROCESS_SCHEMA,
        "coordinate_frame": mewm.COORDINATE_FRAME,
        "crop_policy": mewm.CROP_POLICY,
        "image_interpolation": "linear",
        "mask_interpolation": "nearest",
        "output_shape_zyx": [96, 256, 256],
        "target_spacing_xyz": [1.0, 1.0, 2.0],
        "normalization": {
            "schema": mewm.NORMALIZATION_SCHEMA,
            "scope": "unique_train_transition_visits",
            "foreground": "finite_nonzero_DCE0",
            "SER_foreground": "same_as_DCE0",
            "outside_fov_and_padding": 0,
            "normalization_sha256": normalization_sha,
        },
    }
    (root / "preprocess.json").write_text(
        json.dumps(preprocess, sort_keys=True), encoding="utf-8"
    )
    artifact_files = {
        "crop_plans": "crop_plans.json",
        "exclusions": "exclusions.csv",
        "normalization": "normalization.json",
        "preprocess": "preprocess.json",
        "split": "split.json",
        "transitions": "transitions.csv",
        "visits": "visits.csv",
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, filename in artifact_files.items():
        entry: dict[str, Any] = {
            "path": filename,
            "sha256": sha256_file(root / filename),
        }
        if name == "visits":
            entry["row_count"] = len(visits)
        elif name == "transitions":
            entry["row_count"] = len(transitions)
        elif name == "exclusions":
            entry["row_count"] = len(exclusions)
        artifacts[name] = entry
    bundle: dict[str, Any] = {
        "schema_version": mewm.BUNDLE_SCHEMA,
        "artifacts": artifacts,
        "base_cache_contract": {
            "schema": mewm.CACHE_SCHEMA,
            "bundle_contract_sha256": "b" * 64,
            "preprocess_sha256": artifacts["preprocess"]["sha256"],
            "normalization_sha256": artifacts["normalization"]["sha256"],
        },
        "counts": {
            "visit_count": len(visits),
            "transition_count": len(transitions),
            "exclusion_count": len(exclusions),
            "patient_count": len(patient_splits),
        },
    }
    bundle["bundle_contract_sha256"] = stable_hash(bundle)
    (root / "bundle.json").write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")
    return root


def _fake_bundle_for_cache(tmp_path: Path) -> mewm._BundleData:
    contract = {
        "schema": mewm.CACHE_SCHEMA,
        "bundle_contract_sha256": "b" * 64,
        "preprocess_sha256": "p" * 64,
        "normalization_sha256": "n" * 64,
    }
    return mewm._BundleData(
        root=tmp_path,
        document={"base_cache_contract": contract},
        artifact_paths={},
        artifact_hashes={},
        visits=(),
        transitions=(),
        exclusions=(),
        crop_plans={
            "P1": {
                "coordinate_frame": mewm.COORDINATE_FRAME,
                "input_shape_zyx": [2, 3, 4],
                "output_shape_zyx": [2, 3, 4],
                "crop_start_zyx": [0, 0, 0],
            }
        },
        normalization={},
        preprocess={"output_shape_zyx": [2, 3, 4]},
        split_by_patient={"P1": "train"},
    )


def _connected_bundle_data(tmp_path: Path) -> mewm._BundleData:
    root = tmp_path / "connected-bundle"
    root.mkdir()
    (root / "bundle.json").write_text("{}", encoding="utf-8")
    dates = ("2020-01-01", "2020-01-11", "2020-01-31", "2020-03-01")
    visits: list[dict[str, str]] = []
    for number, date_value in enumerate(dates):
        row = _visit("P1", f"T{number}", "train", 0)
        row["visit_date"] = date_value
        row["crop_start_zyx_json"] = "[0,0,0]"
        visits.append(row)
    transitions: list[dict[str, str]] = []
    for number, delta_days in enumerate((10, 20, 30)):
        source, target = visits[number], visits[number + 1]
        transitions.append(
            {
                "transition_id": f"P1:T{number}->T{number + 1}",
                "patient_id": "P1",
                "fold": "train",
                "source_visit": f"T{number}",
                "target_visit": f"T{number + 1}",
                "source_visit_id": source["visit_id"],
                "target_visit_id": target["visit_id"],
                "delta_days": str(delta_days),
                "action_text": "treatment arm arm P1",
            }
        )
    return mewm._BundleData(
        root=root,
        document={"bundle_contract_sha256": "b" * 64},
        artifact_paths={},
        artifact_hashes={"visits": "v" * 64, "transitions": "t" * 64},
        visits=tuple(visits),
        transitions=tuple(transitions),
        exclusions=(),
        crop_plans={},
        normalization={},
        preprocess={
            "target_spacing_xyz": [0.5, 1.0, 2.0],
            "output_shape_zyx": [96, 256, 256],
        },
        split_by_patient={"P1": "train"},
    )


def _shared_registration_evidence(row: dict[str, str]) -> mewm._RegistrationEvidence:
    affine = np.eye(4, dtype=np.float64)
    return mewm._RegistrationEvidence(
        base_affine_lps=affine,
        cropped_affine_lps=affine,
        meta_sha256="m" * 64,
        registration_sha256="r" * 64,
        transform_artifacts=(
            ()
            if row["visit"] == "T0"
            else ({"path": "transform.json", "sha256": "f" * 64},)
        ),
        source_geometry_sha256="s" * 64,
        registered_grid_geometry_sha256="g" * 64,
    )


def test_bundle_validation_binds_every_artifact_sha(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    validated = mewm._validate_bundle(root)
    assert len(validated.visits) == 6
    assert len(validated.transitions) == 3
    assert set(validated.artifact_hashes) == mewm._ARTIFACT_NAMES

    with (root / "preprocess.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="preprocess.*SHA-256 mismatch"):
        mewm._validate_bundle(root)


def test_cache_payload_checks_deterministic_identity_schema_and_dtype(
    tmp_path: Path,
) -> None:
    bundle = _fake_bundle_for_cache(tmp_path)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    row = {"visit_id": "P1:T0", "patient_id": "P1"}
    contract = mewm._cache_contract(bundle, row["visit_id"])
    key = stable_hash(contract)
    crop_plan = dict(bundle.crop_plans["P1"])
    payload = {
        **contract,
        "cache_key": key,
        "mri": torch.zeros((2, 2, 3, 4), dtype=torch.float16),
        "mask": torch.zeros((1, 2, 3, 4), dtype=torch.uint8),
        "valid_foreground": torch.ones((1, 2, 3, 4), dtype=torch.uint8),
        "crop_plan": crop_plan,
    }
    path = cache_root / f"{key}.pt"
    torch.save(payload, path)

    loaded = mewm._validate_cache_payload(bundle, cache_root, row)
    assert loaded.image.dtype == np.float32
    assert loaded.image.shape == (2, 2, 3, 4)
    assert loaded.cache_key == key
    assert loaded.cache_sha256 == sha256_file(path)

    payload["mri"] = payload["mri"].float()
    torch.save(payload, path)
    with pytest.raises(ValueError, match="mri must be"):
        mewm._validate_cache_payload(bundle, cache_root, row)


def test_metadata_validation_builds_six_connected_pair_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _connected_bundle_data(tmp_path)
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    monkeypatch.setattr(mewm, "_validate_bundle", lambda _: bundle)
    monkeypatch.setattr(
        mewm,
        "_geometry_from_metadata",
        lambda source, root, row: _shared_registration_evidence(row),
    )

    result = mewm.validate_mewm_bundle_metadata(bundle.root, metadata_root)

    assert result.time_pairs == (
        ("T0", "T1"),
        ("T0", "T2"),
        ("T0", "T3"),
        ("T1", "T2"),
        ("T1", "T3"),
        ("T2", "T3"),
    )
    assert len(result.pairs) == 6
    longest = result.pair_geometry_by_endpoints[("P1:T0", "P1:T3")]
    assert longest.delta_days == 60
    assert longest.edge_transition_ids == (
        "P1:T0->T1",
        "P1:T1->T2",
        "P1:T2->T3",
    )
    assert longest.earlier is result.visit_geometry_by_id["P1:T0"]
    assert longest.later is result.visit_geometry_by_id["P1:T3"]
    assert longest.earlier.spacing_dhw == (2.0, 1.0, 0.5)
    assert longest.earlier.affine_lps == tuple(tuple(row) for row in np.eye(4))
    with pytest.raises(TypeError):
        result.visit_geometry_by_id["P1:T4"] = longest.later  # type: ignore[index]


def test_metadata_validation_rejects_geometry_break_inside_connected_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _connected_bundle_data(tmp_path)
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    monkeypatch.setattr(mewm, "_validate_bundle", lambda _: bundle)

    def geometry(source, root, row):
        item = _shared_registration_evidence(row)
        if row["visit"] == "T2":
            affine = item.cropped_affine_lps.copy()
            affine[0, 3] = 1.0
            return mewm._RegistrationEvidence(
                **{**item.__dict__, "cropped_affine_lps": affine}
            )
        return item

    monkeypatch.setattr(mewm, "_geometry_from_metadata", geometry)
    with pytest.raises(ValueError, match="registered affine"):
        mewm.validate_mewm_bundle_metadata(
            bundle.root,
            metadata_root,
            time_pairs=(("T0", "T3"),),
        )


def test_metadata_validation_allows_distinct_native_visit_grids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _connected_bundle_data(tmp_path)
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    monkeypatch.setattr(mewm, "_validate_bundle", lambda _: bundle)

    def geometry(source, root, row):
        item = _shared_registration_evidence(row)
        return mewm._RegistrationEvidence(
            **{
                **item.__dict__,
                "source_geometry_sha256": f"{int(row['visit'][1:]) + 1:064x}",
            }
        )

    monkeypatch.setattr(mewm, "_geometry_from_metadata", geometry)
    result = mewm.validate_mewm_bundle_metadata(
        bundle.root,
        metadata_root,
        time_pairs=(("T0", "T3"),),
    )

    assert len(result.pairs) == 1
    assert len({visit.source_geometry_sha256 for visit in result.visits}) == 4


def test_geometry_reconstructs_resampling_affine_and_crop_translation(tmp_path: Path) -> None:
    bundle = _fake_bundle_for_cache(tmp_path)
    bundle = mewm._BundleData(
        **{
            **bundle.__dict__,
            "preprocess": {"target_spacing_xyz": [0.5, 1.0, 2.0]},
        }
    )
    metadata_root = tmp_path / "metadata"
    visit_dir = metadata_root / "ISPY2-123" / "T1"
    visit_dir.mkdir(parents=True)
    source_geometry = {
        "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "shape_zyx": [4, 5, 6],
        "spacing_xyz": [0.8, 1.2, 2.5],
    }
    registered_geometry = {
        "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "shape_zyx": [4, 5, 6],
        "spacing_xyz": [1.0, 1.0, 2.0],
    }
    meta = {
        "patient_id": "ACRIN-6698-123",
        "visit": "T1",
        "qc_status": "ok",
        "registered_to_visit": "T0",
        "registration_status": "rigid_fallback",
        "image_orientation_patient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "image_position_patient_first": [10.0, 20.0, 30.0],
        "n_slices": 4,
        "rows": 5,
        "cols": 6,
        "pixel_spacing": [1.0, 1.0],
        "spacing_between_slices": 2.0,
        "source_geometry": source_geometry,
        "target_geometry": registered_geometry,
    }
    registration = {
        "patient_id": "ACRIN-6698-123",
        "visit": "T1",
        "registered_to_visit": "T0",
        "registration_status": "rigid_fallback",
        "qc": {"status": "rigid_fallback"},
    }
    transform = {
        "patient_id": "ISPY2-123",
        "fixed_visit": "T0",
        "moving_visit": "T1",
        "selected_transform": "rigid_fallback",
        "rigidity_applied": True,
    }
    (visit_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (visit_dir / "registration.json").write_text(
        json.dumps(registration), encoding="utf-8"
    )
    (visit_dir / "transform.json").write_text(json.dumps(transform), encoding="utf-8")
    row = {
        "patient_id": "ISPY2-123",
        "visit_id": "ISPY2-123:T1",
        "visit": "T1",
        "study_instance_uid": "1.2.3",
        "qc_status": "ok",
        "registration_status": "rigid_fallback",
        "orientation_lps_json": json.dumps(np.diag([-1.0, -1.0, 1.0]).tolist()),
        "row_spacing_mm": "1.2",
        "column_spacing_mm": "0.8",
        "slice_spacing_mm": "2.5",
        "resampled_shape_zyx_json": "[4,5,11]",
        "crop_start_zyx_json": "[1,2,3]",
    }

    evidence = mewm._geometry_from_metadata(bundle, metadata_root, row)
    np.testing.assert_allclose(
        evidence.base_affine_lps,
        np.asarray(
            [
                [0.0, 0.0, -0.5, 15.0],
                [0.0, -1.0, 0.0, 24.0],
                [2.0, 0.0, 0.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    np.testing.assert_allclose(evidence.cropped_affine_lps[:3, 3], [13.5, 22.0, 32.0])
    assert evidence.transform_artifacts[0]["path"] == "transform.json"

    meta["study_uid"] = "9.9.9"
    (visit_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="study UID differs"):
        mewm._geometry_from_metadata(bundle, metadata_root, row)


def test_import_writes_fixed_semantics_manifests_and_truthful_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root = _bundle(tmp_path)
    bundle = mewm._validate_bundle(bundle_root)
    cache_root = tmp_path / "roi_cache"
    metadata_root = tmp_path / "metadata"
    cache_root.mkdir()
    metadata_root.mkdir()
    for row in bundle.visits:
        path, _ = mewm._cache_path(bundle, cache_root, row["visit_id"])
        path.write_bytes(b"test-cache-placeholder")

    shared_image = np.broadcast_to(
        np.zeros((), dtype=np.float32), (2, 96, 256, 256)
    )

    def fake_cache(
        source: mewm._BundleData, cache_dir: Path, row: dict[str, str]
    ) -> mewm._CachePayload:
        _, key = mewm._cache_path(source, cache_dir, row["visit_id"])
        return mewm._CachePayload(
            image=shared_image,
            cache_key=key,
            cache_sha256="c" * 64,
            crop_plan=source.crop_plans[row["patient_id"]],
        )

    def fake_geometry(
        source: mewm._BundleData, root: Path, row: dict[str, str]
    ) -> mewm._RegistrationEvidence:
        patient_index = int(row["patient_id"][-1])
        affine = np.eye(4, dtype=np.float64)
        affine[0, 3] = patient_index
        return mewm._RegistrationEvidence(
            base_affine_lps=affine,
            cropped_affine_lps=affine,
            meta_sha256="m" * 64,
            registration_sha256="r" * 64,
            transform_artifacts=(),
            source_geometry_sha256=f"{patient_index:064x}",
            registered_grid_geometry_sha256=f"{patient_index:064x}",
        )

    monkeypatch.setattr(mewm, "_validate_cache_payload", fake_cache)
    monkeypatch.setattr(mewm, "_geometry_from_metadata", fake_geometry)
    output = tmp_path / "prepared"
    result = mewm.import_mewm_roi_cache(
        bundle_root, cache_root, metadata_root, output
    )

    assert len(result.visits) == 6
    assert len(result.pairs) == 3
    assert result.audit["imported_pair_counts_by_split"] == {"train": 2, "val": 1}
    pairs = read_jsonl(result.pair_manifest_path)
    assert {(row["earlier_stage"], row["later_stage"]) for row in pairs} == {
        ("T0", "T1")
    }
    assert all(row["delta_days"] == 31 for row in pairs)
    visits = read_jsonl(result.visit_manifest_path)
    assert {row["split"] for row in visits} == {"train", "val"}
    first = visits[0]
    assert first["baseline_clinical"]["hr_status"] in {"0", "1"}
    assert first["baseline_clinical"]["her2_status"] in {"0", "1"}
    assert first["baseline_clinical"]["mammaprint"] in {"0", "1"}
    assert first["baseline_clinical"]["menopausal_status"] == "premenopausal"
    assert "source_ftv_volume_cc" not in first["baseline_clinical"]
    assert first["treatment"]["treatment_arm"].startswith("treatment arm ")
    assert not any("ftv" in key.lower() for key in first["treatment"])
    with np.load(first["prepared_path"], allow_pickle=False) as archive:
        assert archive["image"].shape == (1, 96, 256, 256)
        assert archive["image"].dtype == np.float32
        assert archive["phase_roles"].tolist() == ["dce0"]
        provenance = json.loads(str(archive["provenance_json"].item()))
    assert provenance["source_only"] is False
    assert provenance["target_content_used"] is True
    assert provenance["extra"]["registration_assisted"] is True
    assert provenance["extra"]["deployment_scope"] == "paired_registered_research_only"
    assert not any(key.startswith("target_") for key in provenance["extra"])


def test_import_rejects_pair_affine_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root = _bundle(tmp_path)
    bundle = mewm._validate_bundle(bundle_root)
    cache_root = tmp_path / "roi_cache"
    metadata_root = tmp_path / "metadata"
    cache_root.mkdir()
    metadata_root.mkdir()
    for row in bundle.visits:
        path, _ = mewm._cache_path(bundle, cache_root, row["visit_id"])
        path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        mewm,
        "_validate_cache_payload",
        lambda source, cache_dir, row: mewm._CachePayload(
            image=np.zeros((2, 1, 1, 1), dtype=np.float32),
            cache_key="k",
            cache_sha256="c" * 64,
            crop_plan=source.crop_plans[row["patient_id"]],
        ),
    )

    def mismatched_geometry(source, root, row):
        affine = np.eye(4)
        affine[0, 3] = 1.0 if row["visit"] == "T1" else 0.0
        return mewm._RegistrationEvidence(
            base_affine_lps=affine,
            cropped_affine_lps=affine,
            meta_sha256="m" * 64,
            registration_sha256="r" * 64,
            transform_artifacts=(),
            source_geometry_sha256="s" * 64,
            registered_grid_geometry_sha256="g" * 64,
        )

    monkeypatch.setattr(mewm, "_geometry_from_metadata", mismatched_geometry)
    with pytest.raises(ValueError, match="registered affine"):
        mewm.import_mewm_roi_cache(
            bundle_root, cache_root, metadata_root, tmp_path / "output"
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("roles", [("ser",), ("ser", "dce0"), ("dce0", "dce0")])
def test_import_rejects_unsupported_phase_roles(tmp_path: Path, roles) -> None:
    with pytest.raises(ValueError, match="phase_roles"):
        mewm.import_mewm_roi_cache(
            tmp_path, tmp_path, tmp_path, tmp_path / "output", phase_roles=roles
        )
