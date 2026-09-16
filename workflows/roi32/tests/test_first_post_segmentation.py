from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
import torch

from mewm_ispy2.first_post_segmentation import (
    execution_configuration,
    finished,
    geometry,
    infer_logits,
    mask_qc,
    native_image,
    next_case,
    same_grid,
    standard_input,
    sustained_runtime_reasons,
    write_image,
)


def record(shape=(7, 9, 11), ras=False, oblique=False):
    angle = 0.1 if oblique else 0.0
    direction = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    if ras:
        direction[:, :2] *= -1
    return {
        "shape_zyx": list(shape),
        "pixel_spacing_yx_mm": [0.8, 0.6],
        "slice_spacing_mm": 2.5,
        "image_position_patient_first": [12.0, -31.0, 7.0],
        "image_orientation_patient": [*direction[:, 0], *direction[:, 1]],
    }


@pytest.mark.parametrize("ras,oblique", [(False, False), (True, False), (True, True)])
def test_restore_physical_points_and_nifti_values(tmp_path, ras, oblique):
    info = record(ras=ras, oblique=oblique)
    data = np.arange(np.prod(info["shape_zyx"]), dtype=np.float32).reshape(
        info["shape_zyx"]
    )
    path = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    info.update(
        source_image=str(path),
        size_bytes=path.stat().st_size,
        local_mtime_ns=path.stat().st_mtime_ns,
    )
    result = standard_input(info, tmp_path / "CASE_T0_0000.nii.gz")
    native = native_image(data, info)
    for z, y, x in [(0, 0, 0), (2, 3, 4), (6, 8, 10)]:
        physical = native.TransformIndexToPhysicalPoint((x, y, z))
        index = result.TransformPhysicalPointToIndex(physical)
        assert result[index] == data[z, y, x]
    assert geometry(result)["orientation"] == "LPS"
    reread = sitk.ReadImage(str(tmp_path / "CASE_T0_0000.nii.gz"))
    assert same_grid(result, reread)
    assert np.array_equal(
        sitk.GetArrayFromImage(result), sitk.GetArrayFromImage(reread)
    )
    assert np.allclose(nib.load(path).affine, np.eye(4))


def test_nonunit_spacing_and_axis_order():
    info = record()
    data = np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11)
    image = native_image(data, info)
    assert image.GetSize() == (11, 9, 7)
    assert image.GetSpacing() == (0.6, 0.8, 2.5)
    assert np.allclose(
        image.TransformIndexToPhysicalPoint((2, 3, 4)), [13.2, -28.6, 17.0]
    )


@pytest.mark.parametrize("change", ["nonfinite", "shape", "spacing", "direction"])
def test_reject_invalid_source(change):
    info = record()
    data = np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11)
    if change == "nonfinite":
        data[0, 0, 0] = np.nan
    elif change == "shape":
        info["shape_zyx"] = [1, 2, 3]
    elif change == "spacing":
        info["slice_spacing_mm"] = 0
    else:
        info["image_orientation_patient"] = [1, 1, 1, 1, 1, 1]
    with pytest.raises(ValueError):
        native_image(data, info)


def test_mask_extent_in_millimeters_and_multifocal_preservation():
    info = record()
    image = native_image(
        np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11), info
    )
    values = np.zeros((7, 9, 11), dtype=np.uint8)
    values[1:3, 2:4, 3:5] = 1
    values[5, 6, 8] = 1
    mask = sitk.GetImageFromArray(values)
    mask.CopyInformation(image)
    qc = mask_qc(mask, image)
    assert qc["tumor_voxels"] == 9
    assert qc["tumor_volume_mm3"] == pytest.approx(9 * 0.6 * 0.8 * 2.5)
    assert qc["bbox_extent_xyz_mm"] == pytest.approx([3.6, 4.0, 12.5])
    assert qc["connected_components_26"] == 2
    assert "multiple_components_review_required" in qc["flags"]
    assert np.array_equal(sitk.GetArrayFromImage(mask), values)


def test_empty_mask_kept_and_geometry_mismatch_rejected():
    image = native_image(
        np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11), record()
    )
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(image)
    qc = mask_qc(mask, image)
    assert qc["bbox_zyx_exclusive"] is None
    assert "empty_mask_not_proof_of_complete_response" in qc["flags"]
    mask.SetOrigin((0, 0, 0))
    with pytest.raises(ValueError, match="physical grids"):
        mask_qc(mask, image)


def test_resume_rejects_missing_committed_mask(tmp_path):
    (tmp_path / "case_reports").mkdir()
    report = {"status": "completed", "source_mtime_ns": 123, "mask_size_bytes": 100}
    (tmp_path / "case_reports/CASE.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="missing or changed"):
        finished(tmp_path, {"case_id": "CASE", "local_mtime_ns": 123})


def test_mask_file_round_trip(tmp_path):
    image = native_image(
        np.arange(7 * 9 * 11, dtype=np.float32).reshape(7, 9, 11), record(ras=True)
    )
    mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(image)
    mask[2, 3, 4] = 1
    path = tmp_path / "mask.nii.gz"
    write_image(path, mask)
    restored = sitk.ReadImage(str(path))
    assert same_grid(mask, restored)
    assert restored[2, 3, 4] == 1


def test_transient_pressure_does_not_restart_long_inference():
    samples = 0
    for _ in range(5):
        reasons, samples = sustained_runtime_reasons(["host_memory_pressure"], samples)
        assert reasons == []
    reasons, samples = sustained_runtime_reasons(["host_memory_pressure"], samples)
    assert reasons == ["sustained_host_memory_pressure"]
    assert sustained_runtime_reasons([], samples) == ([], 0)
    assert sustained_runtime_reasons(["host_memory", "disk_space"], 0) == (
        ["host_memory", "disk_space"],
        0,
    )


def test_cpu_order_retains_pilot_and_cohort_priority():
    def case(name, size, pilot, fold):
        return dict(
            record((size, size, size)), case_id=name, pilot=pilot, cohort_fold=fold
        )

    large = case("large_pilot", 100, True, "train")
    small = case("small_pilot", 50, True, "train")
    training = case("train", 30, False, "train")
    validation = case("val", 10, False, "val")
    pending = [large, small, training, validation]
    assert next_case(pending, cpu=True) is small
    assert next_case(pending, cpu=False) is large
    assert next_case([training, validation], cpu=True) is training


def test_official_multifold_cpu_ensemble_keeps_inference_context():
    module = pytest.importorskip("nnunetv2.inference.predict_from_raw_data")
    instance = module.nnUNetPredictor(device=torch.device("cpu"), allow_tqdm=False)
    instance.network = torch.nn.Identity()
    instance.list_of_parameters = [{}, {}]
    values = iter([1.0, 2.0])

    @torch.inference_mode()
    def prediction(_):
        return torch.full((2, 2, 2, 2), next(values))

    instance.predict_sliding_window_return_logits = prediction
    logits = infer_logits(instance, np.zeros((1, 2, 2, 2), dtype=np.float32))
    torch.testing.assert_close(logits, torch.full((2, 2, 2, 2), 1.5))


def test_gpu1_override_preserves_model_contract_and_reservations(tmp_path):
    config = {
        "output_dir": str(tmp_path),
        "folds": [0, 1, 2, 3, 4],
        "queue": {
            "gpus": [0, 1],
            "cpu_fallback": True,
            "lock_root": "locks",
            "prior_managed_queues": ["prior"],
        },
    }
    assert execution_configuration(config) is config
    profile = {
        "schema": "first_post_mamamia_execution_v1",
        "gpus": [1],
        "cpu_fallback": False,
    }
    (tmp_path / "execution_profile.json").write_text(json.dumps(profile))
    result = execution_configuration(config)
    assert result["queue"]["gpus"] == [1]
    assert result["queue"]["cpu_fallback"] is False
    assert result["queue"]["prior_managed_queues"] == ["prior"]
    assert result["folds"] == config["folds"]
    assert config["queue"]["gpus"] == [0, 1]


@pytest.mark.parametrize("gpus,workers", [([1, 0], 2), ([1], 2), ([0], 1)])
def test_parallel_profile_limits(tmp_path, gpus, workers):
    config = {"output_dir": str(tmp_path), "queue": {"poll_seconds": 10}}
    profile = {
        "schema": "first_post_mamamia_execution_v1",
        "gpus": gpus,
        "cpu_fallback": False,
        "max_workers": workers,
        "cases_per_worker": 8,
        "poll_seconds": 1,
    }
    (tmp_path / "execution_profile.json").write_text(json.dumps(profile))
    if gpus == [1, 0]:
        queue = execution_configuration(config)["queue"]
        assert queue["max_workers"] == 2
        assert queue["resource_poll_seconds"] == 10
        assert queue["cases_per_worker"] == 8
    else:
        with pytest.raises(ValueError):
            execution_configuration(config)
