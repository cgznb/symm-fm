import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from mewm_ispy2.first_post_data import read_config
from mewm_ispy2.first_post_segmentation import mask_qc, native_image
from mewm_ispy2.first_post_tumor_crops import (
    admit_crop_records,
    case_id,
    crop_reference,
    file_identity,
    prepare_one,
)
from scripts.verify_first_post_tumor_crops import verify_one


def fixture():
    labels = np.zeros((30, 40, 50), np.uint8)
    labels[5:9, 12:17, 4:8] = 1
    labels[18:22, 25:30, 32:39] = 1
    baseline = sitk.GetImageFromArray(labels)
    baseline.SetSpacing((0.8, 0.9, 2.0))
    baseline.SetOrigin((-60.0, -70.0, -30.0))
    image = sitk.Image((80, 90, 70), sitk.sitkFloat32)
    image.SetOrigin((-80, -100, -70))
    image.SetSpacing((1.1, 1.2, 1.5))
    empty = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    empty.CopyInformation(image)
    data = {
        "output_shape_zyx": [32, 64, 64],
        "target_spacing_xyz": [1.0, 1.0, 2.0],
        "tumor_crop": {"margin_mm": 20.0, "empty_mask": "t0_mask"},
    }
    return baseline, image, empty, data


@pytest.mark.parametrize("angle", [0.0, 0.35])
@pytest.mark.parametrize("translation", [(0.0, 0.0, 0.0), (14.0, -19.0, 8.0)])
def test_t0_mask_faces_and_margin_survive_physical_transfer(angle, translation):
    baseline, image, empty, data = fixture()
    transform = sitk.Euler3DTransform()
    transform.SetRotation(0.0, 0.0, angle)
    transform.SetTranslation(translation)
    reference, report = crop_reference(
        image, empty, data, t0_mask=baseline, t0_to_visit=transform
    )
    assert report["localization"] == "empty_mask_t0_mask_fallback"
    assert report["source_mask_voxels"] == 0
    assert report["localization_mask_voxels"] == np.count_nonzero(
        sitk.GetArrayFromImage(baseline)
    )
    assert report["margin_mm"] == 20.0
    assert not sitk.GetArrayFromImage(empty).any()
    voxel_indices = np.argwhere(sitk.GetArrayFromImage(baseline))[:, ::-1]
    spacing = np.asarray(reference.GetSpacing())
    limits = (np.asarray(reference.GetSize()) - 1) * spacing
    for voxel in voxel_indices:
        for offset in itertools.product((-0.5, 0.5), repeat=3):
            point = baseline.TransformContinuousIndexToPhysicalPoint(
                tuple(voxel + offset)
            )
            point = transform.TransformPoint(point)
            index = np.asarray(reference.TransformPhysicalPointToContinuousIndex(point))
            assert np.all(index * spacing >= 20.0 - 1e-5)
            assert np.all(index * spacing <= limits - 20.0 + 1e-5)


def test_explicit_identity_uses_t0_physical_location_not_followup_image_center():
    baseline, image, empty, data = fixture()
    transform = sitk.Transform(3, sitk.sitkIdentity)
    first, _ = crop_reference(
        image, empty, data, t0_mask=baseline, t0_to_visit=transform
    )
    image.SetOrigin((500.0, 600.0, 700.0))
    empty.CopyInformation(image)
    second, report = crop_reference(
        image, empty, data, t0_mask=baseline, t0_to_visit=transform
    )
    assert first.GetOrigin() == second.GetOrigin()
    assert first.GetSpacing() == second.GetSpacing()
    assert report["localization_center_in_acquisition"] is False


def test_nonempty_followup_keeps_its_own_mask():
    _, image, mask, data = fixture()
    mask[30, 40, 20] = 1
    reference, report = crop_reference(image, mask, data)
    assert report["localization"] == "all_predicted_components_union"
    assert report["source_mask_voxels"] == 1
    assert report["margin_mm"] == 20.0
    center = reference.TransformContinuousIndexToPhysicalPoint(
        tuple((np.asarray(reference.GetSize()) - 1) / 2)
    )
    np.testing.assert_allclose(center, mask.TransformIndexToPhysicalPoint((30, 40, 20)))


@pytest.mark.parametrize("missing", ["mask", "mapping"])
def test_t0_policy_cannot_silently_return_to_full_image(missing):
    baseline, image, empty, data = fixture()
    with pytest.raises(ValueError, match="explicit T0-to-visit mapping"):
        crop_reference(
            image,
            empty,
            data,
            t0_mask=None if missing == "mask" else baseline,
            t0_to_visit=None
            if missing == "mapping"
            else sitk.Transform(3, sitk.sitkIdentity),
        )


@pytest.mark.parametrize("value", [0, 2])
def test_empty_or_nonbinary_t0_is_unresolved(value):
    baseline, image, empty, data = fixture()
    baseline = baseline * 0 + value
    with pytest.raises(ValueError, match="nonempty and binary"):
        crop_reference(
            image,
            empty,
            data,
            t0_mask=baseline,
            t0_to_visit=sitk.Transform(3, sitk.sitkIdentity),
        )


def test_non_linear_transfer_requires_full_mask_warp_instead_of_corner_approximation():
    baseline, image, empty, data = fixture()
    transform = sitk.BSplineTransformInitializer(baseline, [2, 2, 2])
    with pytest.raises(ValueError, match="linear physical transform"):
        crop_reference(image, empty, data, t0_mask=baseline, t0_to_visit=transform)


@pytest.fixture
def prepared_sources(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "mewm_ispy2.first_post_tumor_crops.shutil.disk_usage",
        lambda _: SimpleNamespace(free=100 * 1024**3),
    )
    root, seg, output = (
        tmp_path / "source",
        tmp_path / "segmentation",
        tmp_path / "output",
    )
    for name in ("masks", "case_reports", "overlays"):
        (seg / name).mkdir(parents=True)
    for name in ("crops", "crop_reports", "crop_review"):
        (output / "data" / name).mkdir(parents=True)
    records = {}
    for stage in ("T0", "T1", "T2", "T3"):
        record = {
            "patient_id": "patient",
            "visit": stage,
            "fold": "train",
            "shape_zyx": [16, 40, 40],
            "pixel_spacing_yx_mm": [1.0, 1.0],
            "slice_spacing_mm": 2.0,
            "image_orientation_patient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "image_position_patient_first": [-20.0, -20.0, -16.0],
            "relative_path": f"patient/{stage}/image_dce_aqc_1.nii.gz",
        }
        array = (np.arange(16 * 40 * 40).reshape(16, 40, 40) % 200 + 1).astype(
            np.float32
        )
        source = root / record["relative_path"]
        source.parent.mkdir(parents=True)
        nib.save(nib.Nifti1Image(array, np.eye(4)), source)
        record.update(
            size_bytes=source.stat().st_size, local_mtime_ns=source.stat().st_mtime_ns
        )
        image = native_image(array, record)
        labels = np.zeros(array.shape, np.uint8)
        if stage == "T0":
            labels[6:9, 15:20, 13:18] = 1
        mask = sitk.GetImageFromArray(labels)
        mask.CopyInformation(image)
        path = seg / "masks" / f"{case_id(record)}.nii.gz"
        sitk.WriteImage(mask, str(path))
        report = {
            "status": "completed",
            "case_id": case_id(record),
            "source_image": str(source),
            "source_mtime_ns": record["local_mtime_ns"],
            "mask_size_bytes": path.stat().st_size,
            "registered": False,
            "folds": list(range(5)),
            "mirror": True,
            "qc": mask_qc(mask, image),
        }
        (seg / "case_reports" / f"{case_id(record)}.json").write_text(
            json.dumps(report)
        )
        (seg / "overlays" / f"{case_id(record)}.png").touch()
        records[("patient", stage)] = record
    config = {
        "output_dir": str(output),
        "data": {
            "output_shape_zyx": [32, 32, 32],
            "target_spacing_xyz": [1.0, 1.0, 2.0],
            "tumor_crop": {
                "margin_mm": 20.0,
                "empty_mask": "t0_mask",
                "segmentation_root": str(seg),
                "reference_mapping": "physical_lps",
            },
        },
    }
    return root, seg, config, records


@pytest.mark.parametrize("stage", ["T1", "T2", "T3"])
@pytest.mark.parametrize("mapping", ["physical_lps", "transform_manifest"])
def test_followup_preparation_uses_t0_and_preserves_current_empty_mask(
    prepared_sources, stage, mapping
):
    root, seg, config, records = prepared_sources
    record, baseline = records[("patient", stage)], records[("patient", "T0")]
    shift = (3.0, -2.0, 1.0) if mapping == "transform_manifest" else (0.0, 0.0, 0.0)
    if mapping == "transform_manifest":
        transform = sitk.TranslationTransform(3, shift)
        transform_path = root / f"T0_to_{stage}.tfm"
        sitk.WriteTransform(transform, str(transform_path))
        manifest_path = root / "transforms.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": "t0_mask_physical_transforms_v1",
                    "records": {
                        case_id(record): {
                            "status": "passed",
                            "case_id": case_id(record),
                            "reference_case_id": case_id(baseline),
                            "mapping_direction": "T0_LPS_to_visit_LPS",
                            "transform_path": transform_path.name,
                            "transform_identity": file_identity(transform_path),
                            "reference_image_identity": file_identity(
                                root / baseline["relative_path"]
                            ),
                            "visit_image_identity": file_identity(
                                root / record["relative_path"]
                            ),
                        }
                    },
                }
            )
        )
        config["data"]["tumor_crop"].update(
            reference_mapping=mapping, reference_transform_manifest=str(manifest_path)
        )
    normalization = {"mean": 100.0, "std": 50.0}
    result = prepare_one(root, record, config, normalization, set(), records)
    crop = result["crop"]
    assert crop["source_mask_voxels"] == crop["resampled_mask_voxels"] == 0
    assert crop["localization_mask_voxels"] == 75
    assert crop["t0_reference_case_id"] == "patient_T0_aqc1"
    assert crop["reference_mapping"] == mapping
    baseline_mask = sitk.ReadImage(str(seg / "masks" / "patient_T0_aqc1.nii.gz"))
    desired = (
        np.asarray(
            baseline_mask.TransformContinuousIndexToPhysicalPoint((15.0, 17.0, 7.0))
        )
        + shift
    )
    grid = crop["geometry"]
    center = (
        np.asarray(grid["origin_lps_mm"])
        + (np.asarray(grid["shape_zyx"])[::-1] - 1) * grid["spacing_xyz_mm"] / 2
    )
    np.testing.assert_allclose(center, desired)
    cached_record = {
        **record,
        "crop": crop,
        "cache_identity": result["cache_identity"],
        "dependencies": result["dependencies"],
    }
    assert verify_one(cached_record, config, root, records)["empty"]
    assert prepare_one(root, record, config, normalization, set(), records) == result
    report_path = seg / "case_reports" / "patient_T0_aqc1.json"
    with report_path.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="crop or dependency changed"):
        prepare_one(root, record, config, normalization, set(), records)


def test_missing_t0_stops_before_writing_a_crop(prepared_sources):
    root, _, config, records = prepared_sources
    records.pop(("patient", "T0"))
    with pytest.raises(ValueError, match="manual localization"):
        prepare_one(
            root,
            records[("patient", "T3")],
            config,
            {"mean": 100.0, "std": 50.0},
            set(),
            records,
        )
    assert not list((Path(config["output_dir"]) / "data/crops").glob("*.npy"))


def test_t0_from_other_patient_is_rejected(prepared_sources):
    root, _, config, records = prepared_sources
    records[("patient", "T0")] = {
        **records[("patient", "T0")],
        "patient_id": "different",
    }
    with pytest.raises(ValueError, match="different patient"):
        prepare_one(
            root,
            records[("patient", "T1")],
            config,
            {"mean": 100.0, "std": 50.0},
            set(),
            records,
        )


def test_new_config_requires_explicit_mapping(tmp_path):
    data = {
        "registered": False,
        "phase_index": 1,
        "normalization": "train_resampled_nonzero_global_zscore",
        "orientation": "LPS",
        "training_crop": "tumor_union_adaptive",
        "validation_crop": "tumor_union_adaptive",
        "output_shape_zyx": [32, 32, 32],
        "target_spacing_xyz": [1.0, 1.0, 2.0],
        "tumor_crop": {
            "margin_mm": 20.0,
            "empty_mask": "t0_mask",
            "components": "keep_all",
        },
    }
    config = {"schema": "ispy2_first_post_unregistered_training_v1", "data": data}
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="explicit physical mapping"):
        read_config(path)
    data["tumor_crop"]["reference_mapping"] = "physical_lps"
    path.write_text(json.dumps(config))
    assert read_config(path)["data"]["tumor_crop"]["empty_mask"] == "t0_mask"
    data["tumor_crop"]["reference_mapping"] = "transform_manifest"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="transform manifest"):
        read_config(path)


def test_admission_excludes_only_outside_examination(prepared_sources):
    root, seg, config, records = prepared_sources
    path = seg / "masks" / "patient_T2_aqc1.nii.gz"
    mask = sitk.ReadImage(str(path))
    mask.SetOrigin((500.0, 500.0, 500.0))
    sitk.WriteImage(mask, str(path))
    report_path = seg / "case_reports" / "patient_T2_aqc1.json"
    report = json.loads(report_path.read_text())
    report["mask_size_bytes"] = path.stat().st_size
    report_path.write_text(json.dumps(report))
    admitted, excluded = admit_crop_records(root, list(records.values()), config)
    assert [r["visit"] for r in admitted] == ["T0", "T1", "T3"]
    assert len(excluded) == 1
    assert excluded[0]["reason"] == "t0_center_outside_acquisition"
    assert excluded[0]["visit"] == "T2" and excluded[0]["fold"] == "train"
    assert not list((Path(config["output_dir"]) / "data/crops").glob("*.npy"))


def test_admission_missing_t0_does_not_exclude_nonempty_followup(prepared_sources):
    root, seg, config, records = prepared_sources
    records.pop(("patient", "T0"))
    report_path = seg / "case_reports" / "patient_T1_aqc1.json"
    report = json.loads(report_path.read_text())
    report["qc"]["tumor_voxels"] = 1
    report_path.write_text(json.dumps(report))
    admitted, excluded = admit_crop_records(root, list(records.values()), config)
    assert [r["visit"] for r in admitted] == ["T1"]
    assert {r["visit"] for r in excluded} == {"T2", "T3"}
    assert all(r["reason"] == "missing_t0_reference" for r in excluded)


def test_admission_does_not_hide_corrupt_segmentation_provenance(prepared_sources):
    root, seg, config, records = prepared_sources
    path = seg / "case_reports" / "patient_T0_aqc1.json"
    report = json.loads(path.read_text())
    report["source_mtime_ns"] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="provenance"):
        admit_crop_records(root, list(records.values()), config)
