from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset

from mewm_ispy2.registered_roi32_data import (
    acquisition_affine,
    connected_pairs,
    file_identity,
    largest_region,
    read_config,
    resample_array,
    save_npz,
    write_json,
)
from mewm_ispy2.registered_roi32_fm import bridge
from mewm_ispy2.registered_roi32_followup import affine_transform, cached_record, compose_followup_transform, reference_image
from mewm_ispy2.registered_roi32_latents import decode_normalized, fit_channel_statistics
from mewm_ispy2.registered_roi32_runtime import TrainingBatches, output_lock, restore_rng, rng_state, seed_all, verify_contract_files
from mewm_ispy2.registered_roi32_smoke import assert_replay, snapshot
from mewm_ispy2.registered_roi32_vq import FP32Quantizer, ROI32VQ, adversarial_factor, optimizers, update
from mewm_ispy2.vqgan import PatchDiscriminator, VQGANConfig


@pytest.fixture
def config():
    return read_config(Path(__file__).parents[1] / "configs/registered_dce0_roi32_firstpostmask_v1.yaml")


def test_largest_component_uses_voxel_cell_bbox_and_keeps_other_regions():
    mask = np.zeros((35, 150, 150), dtype=np.uint8)
    mask[3:9, 5:11, 7:13] = 1
    mask[30:33, 141:145, 139:143] = 1
    original = mask.copy()
    result = largest_region(mask, np.eye(4))
    assert result["components"] == 2
    assert np.array_equal(result["center"], [5.5, 7.5, 9.5])
    assert result["largest_voxels"] == 216
    assert result["largest_clipped"] is False
    assert result["all_clipped"] is True
    assert np.array_equal(mask, original)
    assert largest_region(np.zeros_like(mask), np.eye(4)) is None


def test_largest_component_uses_26_connectivity_and_deterministic_ties():
    mask = np.zeros((40, 40, 40), dtype=np.uint8)
    mask[1, 1, 1] = mask[2, 2, 2] = 1
    mask[31, 31, 31] = mask[32, 32, 32] = 1
    result = largest_region(mask, np.eye(4))
    assert result["components"] == 2
    assert result["largest_label"] == 1
    assert np.array_equal(result["center"], [1.5, 1.5, 1.5])


def test_saved_geometry_and_fractional_zyx_sampling():
    meta = {"image_orientation_patient": [-1, 0, 0, 0, -1, 0], "pixel_spacing": [0.7, 0.8],
            "spacing_between_slices": 2, "image_position_patient_first": [21, -8, 31]}
    affine = acquisition_affine(meta)
    assert np.allclose(affine @ [2, 3, 4, 1], [19.4, -10.1, 39, 1])
    z, y, x = np.indices((40, 140, 140), dtype=np.float32)
    array = z * 10000 + y * 100 + x
    mapping = np.eye(4)
    mapping[:3, 3] = [1.5, 2.5, 3.5]
    cropped = resample_array(array, mapping, 1)
    assert cropped.shape == (32, 128, 128)
    assert cropped[0, 0, 0] == pytest.approx(15253.5)
    assert cropped[8, 9, 10] == pytest.approx(96163.5)


def test_connected_pairs_require_complete_original_edges():
    visits = []
    for stage in range(4):
        visits.append({"visit_id": f"case:T{stage}", "patient_id": "case", "fold": "train", "visit_date_source": "saved",
                       "trial_arm": "arm", "Age_at_Screening": "48", "HR": "1", "HER2": "0", "MP": "1", "menopausal_status": "pre"})
    edges = [{"source_visit_id": f"case:T{i}", "target_visit_id": f"case:T{i + 1}", "source_visit": f"T{i}", "target_visit": f"T{i + 1}",
              "patient_id": "case", "transition_id": f"case:T{i}->T{i + 1}", "delta_days": str(10 * (i + 1))} for i in range(3)]
    pairs = connected_pairs(edges, visits)
    assert len(pairs) == 6
    assert next(p for p in pairs if p["pair_id"] == "case:T0->T3")["delta_days"] == 60
    assert len(connected_pairs([edges[0], edges[2]], visits)) == 2
    visits[1]["fold"] = "val"
    with pytest.raises(ValueError, match="crosses"):
        connected_pairs(edges, visits)


class Indices(Dataset):
    records = [{"patient_id": "many"}] * 8 + [{"patient_id": "few"}]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return index


@pytest.mark.parametrize("balanced", [False, True])
def test_sampler_recovery_and_global_rng_isolation(balanced):
    seed_all(25)
    expected_rng = torch.get_rng_state().clone()
    one = TrainingBatches(Indices(), batch_size=2, effective_batch=4, seed=72, balanced=balanced)
    next(one)
    next(one)
    state = one.state_dict()
    expected = [next(one).tolist() for _ in range(9)]
    two = TrainingBatches(Indices(), batch_size=2, effective_batch=4, seed=72, balanced=balanced)
    two.load_state_dict(state)
    assert expected == [next(two).tolist() for _ in range(9)]
    assert torch.equal(expected_rng, torch.get_rng_state())
    one.close()
    two.close()


def test_channel_statistics_use_each_training_visit_once():
    one = np.arange(8 * 4, dtype=np.float32).reshape(8, 1, 2, 2)
    two = one + 100
    stats = fit_channel_statistics([("a:T0", one), ("a:T1", two)])
    assert stats["fit_split"] == "train"
    assert stats["element_count_per_channel"] == 8
    assert np.allclose(stats["mean"], one.reshape(8, 4).mean(1) + 50)
    with pytest.raises(ValueError, match="unique"):
        fit_channel_statistics([("a:T0", one), ("a:T0", two)])
    with pytest.raises(FloatingPointError, match="Zero-variance"):
        fit_channel_statistics([("a:T0", np.ones_like(one))])


def test_quantization_is_fp32_and_evaluation_does_not_update():
    quantizer = FP32Quantizer(VQGANConfig(n_codes=16))
    latent = torch.randn(2, 8, 4, 4, 4, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, info = quantizer(latent)
    assert output.dtype == torch.float32
    (output.mean() + info["commitment_loss"]).backward()
    assert torch.isfinite(latent.grad).all()
    quantizer.eval()
    before = snapshot(quantizer)
    quantizer(latent.detach())
    assert_replay(before, snapshot(quantizer))


def test_gan_schedule_counts_generator_updates():
    assert [adversarial_factor(step, 10000, 10000) for step in [1, 10000, 15000, 20000, 80000]] == [0, 0, 0.5, 1, 1]


class PerceptualFixture(torch.nn.Module):
    def forward(self, fake, real):
        return (fake - real).abs().mean()


def test_optimizer_replay_and_gan_warmup(config):
    torch.set_num_threads(2)
    seed_all(19)
    config = copy.deepcopy(config)
    config["vq"]["model"].update(hidden_channels=2, num_groups=1, n_codes=32)
    model = ROI32VQ(config, PerceptualFixture())
    model.image_discriminator = PatchDiscriminator(2, 4, 2)
    model.volume_discriminator = PatchDiscriminator(3, 4, 2)
    generator, discriminator = optimizers(model, config)
    batches = [{"image": torch.randn(2, 1, 32, 32, 32)}]
    before = snapshot(model.image_discriminator)
    before3 = snapshot(model.volume_discriminator)
    update(model, batches, generator, discriminator, step=1, device=torch.device("cpu"), config=config)
    assert not discriminator.state
    assert_replay(before, snapshot(model.image_discriminator))
    assert_replay(before3, snapshot(model.volume_discriminator))
    saved = copy.deepcopy({"model": model.state_dict(), "g": generator.state_dict(), "d": discriminator.state_dict(), "rng": rng_state()})
    update(model, batches, generator, discriminator, step=20000, device=torch.device("cpu"), config=config)
    expected = snapshot(model)
    assert discriminator.state
    assert not torch.equal(before["blocks.0.0.weight"], expected["image_discriminator.blocks.0.0.weight"])
    assert not torch.equal(before3["blocks.0.0.weight"], expected["volume_discriminator.blocks.0.0.weight"])
    model.load_state_dict(saved["model"])
    generator.load_state_dict(saved["g"])
    discriminator.load_state_dict(saved["d"])
    restore_rng(saved["rng"])
    update(model, batches, generator, discriminator, step=20000, device=torch.device("cpu"), config=config)
    assert_replay(expected, snapshot(model))
    before_codebook = snapshot(model.autoencoder.quantizer)
    model.discriminator_loss(batches[0]["image"], step=20000).backward()
    assert_replay(before_codebook, snapshot(model.autoencoder.quantizer))
    before = snapshot(model.image_discriminator)
    before3 = snapshot(model.volume_discriminator)
    model.generator_loss(batches[0]["image"], step=20000)
    assert_replay(before, snapshot(model.image_discriminator))
    assert_replay(before3, snapshot(model.volume_discriminator))
    assert not any(p.requires_grad for p in model.image_discriminator.parameters())
    assert not any(p.requires_grad for p in model.volume_discriminator.parameters())


def test_decoder_applies_inverse_normalization_then_quantization():
    class Codec(torch.nn.Module):
        def quantizer(self, value):
            assert torch.all(value == 7)
            return value + 2, {}

        def decode(self, value):
            assert torch.all(value == 9)
            return value
    result = decode_normalized(Codec().eval(), torch.ones(1, 8, 1, 1, 1), {"mean": [3] * 8, "std": [4] * 8})
    assert torch.all(result == 9)


def test_output_lock_survives_detach_and_rejects_duplicates(tmp_path):
    child = None
    try:
        with output_lock(tmp_path) as lock:
            child = subprocess.Popen([sys.executable, "-B", "-c", "import sys; print('ready', flush=True); sys.stdin.read(1)"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, pass_fds=(lock.fileno(),), start_new_session=True)
            assert child.stdout.readline().strip() == "ready"
            with pytest.raises(RuntimeError, match="already owns"):
                with output_lock(tmp_path):
                    pass
        with pytest.raises(RuntimeError, match="already owns"):
            with output_lock(tmp_path):
                pass
        child.communicate("x", timeout=10)
        assert child.returncode == 0
        with output_lock(tmp_path):
            pass
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.communicate(timeout=10)


def test_contract_checks_nested_codec_and_statistics(tmp_path):
    source = tmp_path / "codec"
    source.write_bytes(b"version one")
    contract = {"codec": file_identity(source), "configuration": {"name": "test"}}
    verify_contract_files(contract)
    source.write_bytes(b"different version two")
    with pytest.raises(ValueError, match="Source file changed"):
        verify_contract_files(contract)


def test_followup_transform_order_with_noncommuting_physical_transforms():
    native = np.array([[0, -1, 0, 30], [1, 0, 0, -5], [0, 0, 1, 8], [0, 0, 0, 1]], dtype=float)
    to_native = affine_transform(native)
    phase = sitk.TranslationTransform(3, [1, 2, 3])
    longitudinal = sitk.TranslationTransform(3, [-2, 1, 0])
    transform = compose_followup_transform(to_native, phase, longitudinal)
    assert transform.TransformPoint([2, 3, 4]) == pytest.approx([24, -4, 15])
    image = reference_image((3, 4, 5), native)
    assert image.GetSize() == (5, 4, 3)
    assert image.TransformIndexToPhysicalPoint([1, 2, 0]) == pytest.approx([28, -4, 8])


def test_followup_cache_invalidates_changed_masks_or_transforms(tmp_path):
    source = tmp_path / "transform.txt"
    source.write_text("first transform")
    mask_path = tmp_path / "mask.npz"
    mask = np.ones((32, 128, 128), dtype=bool)
    save_npz(mask_path, mask=mask)
    marker = tmp_path / "visit.json"
    write_json(marker, {"status": "passed", "actual_cropped_voxels": int(mask.sum()), "transform_sources": [file_identity(source)]})
    cached = cached_record(marker, [source], mask_path)
    assert cached["input_sources"] == [file_identity(source)]
    assert cached_record(marker, [source], mask_path) == cached
    save_npz(mask_path, mask=np.zeros_like(mask))
    assert cached_record(marker, [source], mask_path) is None
    source.write_text("changed transform")
    assert cached_record(marker, [source], mask_path) is None


def test_fm_conditions_fit_train_only_and_reject_future_outcomes(config):
    module = bridge(config)
    from ispy2_symmflow.training.schema import fit_condition_schema
    one = {"pair_id": "train:T0->T1", "split": "train", "earlier_stage": "T0", "later_stage": "T1", "delta_days": 10,
           "interval_missing": False, "interval_source": "saved", "baseline_clinical": {"age": 40}, "treatment": {"treatment_arm": "training_arm"}}
    two = {**one, "split": "val", "pair_id": "val:T0->T1", "baseline_clinical": {"age": 900}, "treatment": {"treatment_arm": "validation_only_arm"}}
    schema, provenance = fit_condition_schema([one, two], config["fm"]["conditions"])
    assert provenance["fit_pair_ids"] == [one["pair_id"]]
    assert next(field for field in schema.numeric_fields if field.name == "age").mean == 40
    assert "validation_only_arm" not in next(field for field in schema.categorical_fields if field.name == "treatment_arm").categories
    latent = torch.zeros(8, 8, 32, 32)
    item = {"record": {**one, "future_mask": "evaluation only", "pcr": 1}, "earlier_latent": latent, "later_latent": latent + 1}
    batch = module.collate_pairs([item])
    assert "future_mask" not in batch["conditions"] and "pcr" not in batch["conditions"]
    item["record"]["baseline_clinical"] = {"pcr": 1}
    with pytest.raises(ValueError, match="forbidden"):
        module.collate_pairs([item])


@pytest.mark.parametrize("vq_passed", [True, False])
def test_controller_orders_stages_and_stops_at_failed_gate(config, monkeypatch, vq_passed):
    from mewm_ispy2 import registered_roi32_evaluation as evaluation
    from mewm_ispy2 import registered_roi32_fm as fm
    from mewm_ispy2 import registered_roi32_followup as followup
    from mewm_ispy2 import registered_roi32_latents as latents
    from mewm_ispy2 import registered_roi32_vq as vq
    path = Path(__file__).parents[1] / "scripts/run_registered_roi32.py"
    spec = importlib.util.spec_from_file_location("roi32_controller_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    events = []

    def event(name, value=True):
        def call(*args, **kwargs):
            events.append(name)
            return value
        return call

    class Reservation:
        def close(self):
            events.append("release_gpu")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(runner, "ensure_data", event("data"))
    monkeypatch.setattr(runner, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "check_disk", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "write_json", event("complete"))
    monkeypatch.setattr(runner, "reserve_gpu", event("gpu", (0, Reservation())))
    monkeypatch.setattr(runner, "select_batch", lambda config, stage, device: 2)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(evaluation, "preview_crops", event("preview"))
    monkeypatch.setattr(followup, "run_audit", event("audit", {"statuses": {"passed": 3}, "visits": 3}))
    monkeypatch.setattr(vq, "train", event("vq", vq_passed))
    monkeypatch.setattr(evaluation, "evaluate_vq", event("vq_evaluation"))
    monkeypatch.setattr(latents, "extract", event("latents"))
    monkeypatch.setattr(fm, "train", event("fm"))
    monkeypatch.setattr(evaluation, "evaluate_fm", event("fm_evaluation"))
    runner.execute(config, "all")
    expected = ["data", "preview", "audit", "gpu", "vq"]
    if vq_passed:
        expected += ["vq_evaluation", "latents", "fm", "fm_evaluation", "complete"]
    assert events == expected + ["release_gpu"]
