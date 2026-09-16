from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
import torch

from mewm_ispy2.first_post_data import (
    ALL_VISITS_POLICY,
    FirstPostDataset,
    combine_moments,
    image_patch,
    nonzero_moments,
    prepare,
    prepare_cohort,
    resample_native,
    select_records,
)
from mewm_ispy2.first_post_tumor_crops import (
    advise_verified_cache,
    case_id,
    crop_reference,
    normalize_cache,
    prepare_tumor_crops,
)
from mewm_ispy2.first_post_vqgan import (
    FirstPostSystem,
    gpu_reserved,
    queue_alive,
    resource_reasons,
    train,
)
from mewm_ispy2.first_post_vqgan_performance import selected_config


def geometry(shape=(8, 9, 10)):
    return {
        "shape_zyx": list(shape),
        "pixel_spacing_yx_mm": [1.0, 1.0],
        "slice_spacing_mm": 1.0,
        "image_orientation_patient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "image_position_patient_first": [4.0, 5.0, 6.0],
    }


@pytest.fixture
def configuration(tmp_path):
    source = tmp_path / "source"
    bundle = tmp_path / "baseline"
    source.mkdir()
    bundle.mkdir()
    records = []
    for index, fold in enumerate(("train", "val")):
        relative = f"{fold}/T0/dce/{fold}_dce_aqc_1.nii.gz"
        path = source / relative
        path.parent.mkdir(parents=True)
        array = np.arange(8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10) + 1
        if index:
            array += 10000
        nib.save(nib.Nifti1Image(array, np.eye(4)), path)
        records.append(
            {
                **geometry(),
                "patient_id": fold,
                "visit": "T0",
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "registered": False,
        "phase_index": 1,
        "destination_root": str(source),
        "records": records,
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "verification.json").write_text(
        json.dumps(
            {
                "registered": False,
                "phase_index": 1,
                "destination_root": str(source),
                "stage": "complete",
                "content_differences": 0,
            }
        )
    )
    (bundle / "split.json").write_text(json.dumps({"train": ["train"], "val": ["val"]}))
    with (bundle / "visits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "visit", "fold"])
        writer.writeheader()
        writer.writerows(
            {"patient_id": fold, "visit": "T0", "fold": fold}
            for fold in ("train", "val")
        )
    return {
        "schema": "ispy2_first_post_unregistered_training_v1",
        "source_manifest": str(source / "manifest.json"),
        "source_verification": str(source / "verification.json"),
        "baseline_bundle": str(bundle),
        "output_dir": str(tmp_path / "run"),
        "data": {
            "expected_train_visits": 1,
            "expected_val_visits": 1,
            "target_spacing_xyz": [1.0, 1.0, 1.0],
            "output_shape_zyx": [32, 32, 32],
        },
    }


def test_preparation_fits_only_training_and_preserves_zero_padding(configuration):
    result = prepare(configuration)
    assert result["normalization"]["mean"] == pytest.approx(360.5)
    assert result["normalization"]["count"] == 720
    assert result["fully_decoded_visits"] == 2
    dataset = FirstPostDataset(configuration, "val")
    one, two = dataset[0]["image"], dataset[0]["image"]
    assert torch.equal(one, two)
    assert tuple(one.shape) == (1, 32, 32, 32)
    assert torch.count_nonzero(one) == 720
    assert one.max() > 40
    assert prepare(configuration) == result


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("angle", [0.0, 0.35])
def test_tumor_crop_keeps_separated_regions_and_oblique_physical_bounds(empty, angle):
    values = np.zeros((25, 40, 100), dtype=np.uint8)
    if not empty:
        values[1:3, 4:7, 2:5] = 1
        values[20:24, 30:36, 90:99] = 1
    mask = sitk.GetImageFromArray(values)
    mask.SetSpacing((1.7, 1.5, 3.0))
    mask.SetOrigin((-101.0, 24.5, 46.0))
    c, s = np.cos(angle), np.sin(angle)
    mask.SetDirection(
        (float(c), -float(s), 0.0, float(s), float(c), 0.0, 0.0, 0.0, 1.0)
    )
    image = sitk.Cast(mask, sitk.sitkFloat32)
    data = {
        "output_shape_zyx": [32, 32, 32],
        "target_spacing_xyz": [1.0, 1.0, 2.0],
        "tumor_crop": {"margin_mm": 20.0},
    }
    reference, report = crop_reference(image, mask, data)
    assert reference.GetSize() == (32, 32, 32)
    assert reference.GetDirection() == mask.GetDirection()
    assert report["spacing_scale"] > 1
    if empty:
        points = np.array([[0, 0, 0], [99, 39, 24]])
        assert report["localization"] == "empty_mask_full_image_fallback"
    else:
        points = np.argwhere(values)[..., ::-1]
        assert np.min(report["minimum_margin_xyz_mm"]) >= 20 - 1e-5
    for point in points:
        physical = mask.TransformContinuousIndexToPhysicalPoint(
            tuple(float(v) for v in point)
        )
        index = reference.TransformPhysicalPointToContinuousIndex(physical)
        assert np.all(np.array(index) >= 0)
        assert np.all(np.array(index) <= 31)


def test_tumor_crop_rejects_mask_on_other_grid_and_nonbinary_labels():
    image = sitk.Image([32, 32, 32], sitk.sitkFloat32)
    mask = sitk.Image([32, 32, 32], sitk.sitkUInt8)
    data = {
        "output_shape_zyx": [32, 32, 32],
        "target_spacing_xyz": [1.0, 1.0, 2.0],
        "tumor_crop": {"margin_mm": 20.0},
    }
    mask.SetOrigin((1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="grids differ"):
        crop_reference(image, mask, data)
    mask.SetOrigin(image.GetOrigin())
    mask[3, 3, 3] = 2
    with pytest.raises(ValueError, match="not binary"):
        crop_reference(image, mask, data)


def test_float16_tumor_cache_preserves_padding_and_normalization():
    values = np.array([0, 100, 200, 1000], dtype=np.float32)
    cached = normalize_cache(values.copy(), {"mean": 100.0, "std": 50.0})
    assert cached.dtype == np.float16
    np.testing.assert_array_equal(cached, [0, 0, 2, 18])
    with pytest.raises(ValueError, match="non-finite"):
        normalize_cache(np.array([np.nan, 1.0]), {"mean": 0, "std": 1})


def test_tumor_preparation_keeps_all_cases_and_rejects_stale_caches(
    configuration, monkeypatch
):
    from types import SimpleNamespace

    from mewm_ispy2.first_post_segmentation import mask_qc, native_image
    from scripts.verify_first_post_tumor_crops import verify_one

    monkeypatch.setattr(
        "mewm_ispy2.first_post_tumor_crops.shutil.disk_usage",
        lambda _: SimpleNamespace(free=100 * 1024**3),
    )
    config = configuration
    config["data"].update(
        registered=False,
        phase_index=1,
        orientation="LPS",
        normalization="train_resampled_nonzero_global_zscore",
        training_crop="random_spatial_patch",
        validation_crop="center",
    )
    base_summary = prepare(config)
    base = Path(config["output_dir"])
    root, records = select_records(config)
    segmentation = base.parent / "segmentation"
    for name in ("masks", "case_reports", "overlays"):
        (segmentation / name).mkdir(parents=True)
    (segmentation / "queue_status.json").write_text(
        json.dumps({"status": "completed", "total_cases": 2, "completed_cases": 2})
    )
    for record in records:
        source = root / record["relative_path"]
        array = np.asarray(nib.load(source).dataobj, dtype=np.float32)
        image = native_image(array, record)
        labels = np.zeros_like(array, dtype=np.uint8)
        if record["fold"] == "train":
            labels[1:4, 2:5, 3:6] = 1
        mask = sitk.GetImageFromArray(labels)
        mask.CopyInformation(image)
        case = case_id(record)
        mask_path = segmentation / "masks" / f"{case}.nii.gz"
        sitk.WriteImage(mask, str(mask_path))
        report = {
            "status": "completed",
            "case_id": case,
            "source_image": str(source),
            "source_mtime_ns": record["local_mtime_ns"],
            "mask_size_bytes": mask_path.stat().st_size,
            "registered": False,
            "folds": [0, 1, 2, 3, 4],
            "mirror": True,
            "qc": mask_qc(mask, image),
        }
        (segmentation / "case_reports" / f"{case}.json").write_text(json.dumps(report))
        (segmentation / "overlays" / f"{case}.png").touch()
    config = copy.deepcopy(config)
    config["output_dir"] = str(base.parent / "tumor")
    config["data"].update(
        training_crop="tumor_union_adaptive",
        validation_crop="tumor_union_adaptive",
        tumor_crop={
            "prepared_data": str(base / "data"),
            "segmentation_root": str(segmentation),
            "margin_mm": 20.0,
            "empty_mask": "full_image_fov",
            "components": "keep_all",
        },
    )
    summary = prepare_tumor_crops(config)
    assert summary["total_visits"] == 2
    assert summary["normalization"]["mean"] == base_summary["normalization"]["mean"]
    assert summary["localization_counts"] == {
        "all_predicted_components_union": 1,
        "empty_mask_full_image_fallback": 1,
    }
    assert summary["all_source_mask_voxel_faces_in_fov"]
    rows = json.loads((Path(config["output_dir"]) / "data/manifest.json").read_text())[
        "records"
    ]
    assert verify_one(rows[0], config, root)["empty"] is False
    assert verify_one(rows[1], config, root)["empty"] is True
    moved = copy.deepcopy(rows[0])
    moved["crop"]["geometry"]["origin_lps_mm"][0] += 100.0
    with pytest.raises(ValueError, match="outside the crop"):
        verify_one(moved, config, root)
    dataset = FirstPostDataset(config, "val")
    assert len(dataset) == 1
    assert dataset[0]["image"].shape == (1, 32, 32, 32)
    assert dataset[0]["image"].dtype == torch.float32
    assert torch.equal(dataset[0]["image"], dataset[0]["image"])
    assert prepare_tumor_crops(config)["total_visits"] == 2
    path = Path(config["output_dir"]) / "data/crops" / f"{case_id(records[1])}.npy"
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="cache changed"):
        dataset[0]
    with pytest.raises(ValueError, match="crop or dependency changed"):
        prepare_tumor_crops(config)


def test_profile_selection_preserves_data_and_sample_based_schedule():
    config = {
        "data": {"output_shape_zyx": [96, 256, 256]},
        "training": {
            "batch_size": 1,
            "generator_learning_rate": 1e-5,
            "discriminator_learning_rate": 5e-6,
            "max_epochs": 100,
        },
    }
    selection = {
        "source_configuration": copy.deepcopy(config),
        "status": "passed",
        "batch_size": 3,
        "precision": "bf16-mixed",
        "loader_workers": 2,
        "prefetch_factor": 2,
    }
    selected = selected_config(config, selection)
    assert selected["data"] == config["data"]
    assert selected["training"]["generator_learning_rate"] == pytest.approx(3e-5)
    assert selected["training"]["scheduler_milestones"] == [20000, 40000]
    assert selected["training"]["max_epochs"] == 100
    assert config["training"]["batch_size"] == 1
    config["data"]["output_shape_zyx"][0] = 80
    with pytest.raises(ValueError, match="another configuration"):
        selected_config(config, selection)


def test_crop_cache_memory_advice_is_bounded_and_preserves_data(tmp_path, monkeypatch):
    path = tmp_path / "crop.npy"
    content = b"immutable prepared data"
    path.write_bytes(content)
    original = path.stat()
    identity = {"size_bytes": original.st_size, "mtime_ns": original.st_mtime_ns}
    advice = []
    monkeypatch.setattr(
        "mewm_ispy2.first_post_tumor_crops.os.posix_fadvise",
        lambda fd, start, length, policy: advice.append((start, length, policy)),
    )
    assert advise_verified_cache(path, identity, 8) == 8
    assert advice[0][0:2] == (0, 8)
    assert path.read_bytes() == content
    assert path.stat().st_mtime_ns == original.st_mtime_ns
    path.write_bytes(content + b" changed")
    with pytest.raises(ValueError, match="changed before memory advice"):
        advise_verified_cache(path, identity, 8)
    assert len(advice) == 1


@pytest.fixture
def full_configuration(configuration):
    config = copy.deepcopy(configuration)
    source = Path(config["source_manifest"]).parent
    manifest = json.loads(Path(config["source_manifest"]).read_text())
    records = manifest["records"]
    for index, patient in enumerate(["train", "val", *[f"new{i}" for i in range(18)]]):
        for visit in ("T0", "T1"):
            existing = next(
                (
                    r
                    for r in records
                    if (r["patient_id"], r["visit"]) == (patient, visit)
                ),
                None,
            )
            if existing is None:
                relative = f"{patient}/{visit}/dce/{patient}_{visit}_dce_aqc_1.nii.gz"
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                array = np.arange(720, dtype=np.float32).reshape(8, 9, 10) + 1 + index
                nib.save(nib.Nifti1Image(array, np.eye(4)), path)
                existing = {
                    **geometry(),
                    "patient_id": patient,
                    "visit": visit,
                    "relative_path": relative,
                    "size_bytes": path.stat().st_size,
                }
                records.append(existing)
            metadata = source / patient / visit / "meta.json"
            metadata.write_text(
                json.dumps(
                    {
                        "patient_id": patient,
                        "visit": visit,
                        "dce_phase_ids": [0, 1, 2],
                        "dce_paths": [
                            existing["relative_path"].replace("aqc_1", f"aqc_{i}")
                            for i in range(3)
                        ],
                        "n_times": 3,
                        "n_slices": 8,
                        "rows": 9,
                        "cols": 10,
                        "qc_status": "ok",
                    }
                )
            )
            existing["metadata_source"] = str(metadata)
    Path(config["source_manifest"]).write_text(json.dumps(manifest))
    config["seed"] = 2026
    config["cohort"] = {
        "policy": ALL_VISITS_POLICY,
        "expected_visits": 40,
        "expected_patients": 20,
        "validation_fraction": 0.1,
        "require_phase_metadata": True,
    }
    config["data"].update(expected_train_visits=36, expected_val_visits=4)
    return config


def test_full_cohort_includes_extra_visits_and_preserves_patients(full_configuration):
    config = full_configuration
    (Path(config["baseline_bundle"]) / "visits.csv").unlink()
    _, selected = select_records(config)
    assert len(selected) == 40
    summary = prepare_cohort(config)
    assert summary["patient_counts"] == {"train": 18, "val": 2}
    assert summary["visit_counts"] == {"train": 36, "val": 4}
    assert summary["patient_overlap"] == 0
    assert summary["phase_verification"]["verified_visits"] == 40
    split = json.loads(Path(summary["split_file"]).read_text())
    assert "train" in split["train"] and "val" in split["val"]
    for patient in split["train"] + split["val"]:
        assert len({r["fold"] for r in selected if r["patient_id"] == patient}) == 1
    manifest = Path(config["source_manifest"])
    data = json.loads(manifest.read_text())
    data["records"].reverse()
    manifest.write_text(json.dumps(data))
    assert select_records(config)[1] == selected
    assert prepare_cohort(config) == summary


def test_full_preparation_refits_training_only_and_rejects_old_stats(
    full_configuration,
):
    config = full_configuration
    root, selected = select_records(config)
    expected = []
    for record in selected:
        if record["fold"] == "train":
            expected.append(
                np.asarray(nib.load(root / record["relative_path"]).dataobj).ravel()
            )
    values = np.concatenate(expected)
    result = prepare(config)
    assert result["fully_decoded_visits"] == 40
    assert result["normalization"]["mean"] == pytest.approx(values.mean())
    assert result["normalization"]["std"] == pytest.approx(values.std())
    assert result["normalization"]["fit_patients"] == 18
    assert len(FirstPostDataset(config, "train")) == 36
    assert len(FirstPostDataset(config, "val")) == 4
    normalized = Path(config["output_dir"]) / "data/normalization.json"
    old = json.loads(normalized.read_text())
    old.pop("preparation_config")
    normalized.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="normalization differs"):
        FirstPostDataset(config, "train")


@pytest.mark.parametrize("change", ["seed", "ratio", "saved_membership"])
def test_frozen_full_cohort_rejects_changes(full_configuration, change):
    config = full_configuration
    prepare_cohort(config)
    if change == "seed":
        config["seed"] += 1
    elif change == "ratio":
        config["cohort"]["validation_fraction"] = 0.2
        config["data"].update(expected_train_visits=32, expected_val_visits=8)
    else:
        path = Path(config["output_dir"]) / "data/split.json"
        split = json.loads(path.read_text())
        split["train"].append(split["val"].pop())
        path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="Saved first-post cohort changed"):
        prepare_cohort(config)


@pytest.mark.parametrize(
    "change", ["pre_only", "phase_order", "wrong_phase_path", "patient"]
)
def test_full_cohort_rejects_inconsistent_phase_metadata(full_configuration, change):
    config = full_configuration
    record = json.loads(Path(config["source_manifest"]).read_text())["records"][0]
    path = Path(record["metadata_source"])
    metadata = json.loads(path.read_text())
    if change == "pre_only":
        metadata.update(
            dce_phase_ids=[0], dce_paths=metadata["dce_paths"][:1], n_times=1
        )
    elif change == "phase_order":
        metadata["dce_phase_ids"] = [1, 0, 2]
    elif change == "wrong_phase_path":
        metadata["dce_paths"][1] = metadata["dce_paths"][0]
    else:
        metadata["patient_id"] = "wrong_patient"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="First-post phase metadata"):
        prepare_cohort(config)


def test_tumor_crop_hold_prevents_training_before_loading_data():
    with pytest.raises(ValueError, match="pending_reviewed_tumor_centered_crop"):
        train(
            {
                "training": {
                    "enabled": False,
                    "disabled_reason": "pending_reviewed_tumor_centered_crop",
                }
            }
        )


def test_patient_overlap_is_rejected(configuration):
    split = Path(configuration["baseline_bundle"]) / "split.json"
    split.write_text(json.dumps({"train": ["train", "val"], "val": ["val"]}))
    with pytest.raises(ValueError, match="Patient leakage"):
        select_records(configuration)


def test_interrupted_preparation_resumes_without_refitting_partial_data(
    configuration, monkeypatch
):
    import mewm_ispy2.first_post_data as data

    original = data.scan_record

    def interrupted(root, record, config):
        if record["fold"] == "val":
            raise RuntimeError("Simulated read interruption")
        return original(root, record, config)

    monkeypatch.setattr(data, "scan_record", interrupted)
    with pytest.raises(RuntimeError, match="Simulated"):
        prepare(configuration)
    output = Path(configuration["output_dir"]) / "data"
    assert not (output / "normalization.json").exists()

    def resumed(root, record, config):
        assert record["fold"] == "val"
        return original(root, record, config)

    monkeypatch.setattr(data, "scan_record", resumed)
    assert prepare(configuration)["normalization"]["mean"] == pytest.approx(360.5)


def test_training_cannot_change_preprocessing_after_preparation(configuration):
    prepare(configuration)
    configuration["data"]["target_spacing_xyz"] = [2.0, 2.0, 2.0]
    with pytest.raises(ValueError, match="preprocessing differs"):
        FirstPostDataset(configuration, "train")


@pytest.mark.parametrize(
    "change", ("phase", "registered", "duplicate", "wrong_filename", "size")
)
def test_invalid_source_is_rejected(configuration, change):
    path = Path(configuration["source_manifest"])
    value = json.loads(path.read_text())
    if change == "phase":
        value["phase_index"] = 0
    elif change == "registered":
        value["registered"] = True
    elif change == "duplicate":
        value["records"].append(value["records"][0])
    elif change == "wrong_filename":
        value["records"][0]["relative_path"] = "different_aqc_0.nii.gz"
    else:
        value["records"][0]["size_bytes"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        select_records(configuration)


def test_geometry_restores_physical_flip_and_retains_obliquity():
    array = np.arange(720, dtype=np.float32).reshape(8, 9, 10)
    native = geometry()
    expected, expected_geometry = resample_native(array, native, [1.0, 1.0, 1.0])
    flipped = geometry()
    flipped["image_orientation_patient"] = [-1, 0, 0, 0, -1, 0]
    flipped["image_position_patient_first"] = [13, 13, 6]
    actual, actual_geometry = resample_native(
        array[:, ::-1, ::-1].copy(), flipped, [1.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual_geometry == expected_geometry
    oblique = geometry()
    angle = 0.2
    oblique["image_orientation_patient"] = [
        np.cos(angle),
        np.sin(angle),
        0,
        -np.sin(angle),
        np.cos(angle),
        0,
    ]
    result, metadata = resample_native(array, oblique, [0.5, 0.5, 1.0])
    assert result.shape == (8, 17, 19)
    assert metadata["origin_lps"] == oblique["image_position_patient_first"]
    assert metadata["direction_lps"][1] == pytest.approx(-np.sin(angle))


def test_invalid_geometry_and_nonfinite_voxels_fail():
    native = geometry()
    native["image_orientation_patient"] = [1, 0, 0, 1, 0, 0]
    with pytest.raises(ValueError, match="Non-orthonormal"):
        resample_native(np.ones((8, 9, 10), np.float32), native, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="Non-finite"):
        nonzero_moments(np.array([1.0, float("nan")]))


def test_pooled_moments_are_voxel_weighted():
    one = np.array([0, 1, 2, 3], dtype=np.float32)
    two = np.array([10, 20], dtype=np.float32)
    result = combine_moments([nonzero_moments(one), nonzero_moments(two)])
    expected = np.array([1, 2, 3, 10, 20], dtype=np.float64)
    assert result["count"] == 5
    assert result["mean"] == pytest.approx(expected.mean())
    assert result["std"] == pytest.approx(expected.std())


def test_random_training_crop_can_cover_both_edges():
    array = np.arange(16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16) + 1
    torch.manual_seed(2026)
    patches = [image_patch(array, [8, 8, 8], random_crop=True) for _ in range(100)]
    assert min(p.min() for p in patches) < 256
    assert max(p.max() for p in patches) > 3840
    np.testing.assert_array_equal(
        image_patch(array, [8, 8, 8], random_crop=False), array[4:12, 4:12, 4:12]
    )


def test_resume_requires_same_first_post_contract():
    from mewm_ispy2.vqgan import MRILevelVQGAN, VQGANConfig

    model = FirstPostSystem(
        MRILevelVQGAN(VQGANConfig(n_codes=8)),
        discriminator_channels=4,
        checkpoint_identity={"phase_index": 1, "registered": False},
    )
    saved = {"first_post_run_contract": copy.deepcopy(model.checkpoint_identity)}
    model.on_load_checkpoint(saved)
    saved["first_post_run_contract"]["phase_index"] = 0
    with pytest.raises(ValueError, match="contract changed"):
        model.on_load_checkpoint(saved)


def test_queued_existing_jobs_have_priority(monkeypatch, tmp_path):
    manifest = tmp_path / "queue.json"
    config = {
        "queue": {
            "reserved_gpu0_queue": str(manifest),
            "prior_managed_queues": [str(manifest)],
        }
    }
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "waiting_for_resources"})
    )
    monkeypatch.setattr("mewm_ispy2.first_post_vqgan.queue_alive", lambda *args: True)
    assert gpu_reserved(0, config)
    assert gpu_reserved(1, config)
    assert not queue_alive(manifest, {"status": "completed"})


def test_resource_admission_and_runtime_reserves_differ():
    config = {
        "queue": {
            "runtime_memory_gib": 8,
            "runtime_disk_gib": 8,
            "admission_memory_gib": 24,
            "admission_disk_gib": 15,
        }
    }
    state = {
        "memory_headroom_bytes": 16 * 1024**3,
        "disk_free_bytes": 20 * 1024**3,
        "memory_pressure_full_avg10": 0,
    }
    assert resource_reasons(state, config, runtime=False) == ["host_memory"]
    assert resource_reasons(state, config, runtime=True) == []
