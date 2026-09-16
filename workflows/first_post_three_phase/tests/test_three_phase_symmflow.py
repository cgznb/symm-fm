
from research_release import path as _release_path, load_yaml as _release_yaml
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mewm_ispy2.three_phase_pilot_data import select_patients
from mewm_ispy2.three_phase_symmflow import (
    SharedThreePhaseCodec, ThreePhaseSymmFlow, join_phase_latents, split_phase_latents,
)


class ToyCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def encode_continuous(self, image):
        return image.repeat(1, 8, 1, 1, 1) * self.scale

    def quantizer(self, latent):
        return latent, {}

    def decode(self, latent):
        return latent[:, :1]


def test_shared_codec_preserves_phase_order_and_remains_frozen():
    codec = SharedThreePhaseCodec(ToyCodec())
    images = torch.stack([torch.full((2, 3, 4), float(i)) for i in (1, 4, 9)])[None]
    latent = codec.encode(images)
    assert latent.shape == (1, 24, 2, 3, 4)
    assert [float(part.mean()) for part in split_phase_latents(latent)] == [1, 4, 9]
    torch.testing.assert_close(codec.decode(latent), images)
    codec.train()
    assert not codec.codec.training
    assert all(not parameter.requires_grad for parameter in codec.parameters())


@pytest.mark.parametrize("channels", [8, 16, 32])
def test_other_experiment_latents_are_not_silently_accepted(channels):
    with pytest.raises(ValueError, match="24"):
        split_phase_latents(torch.zeros(1, channels, 2, 3, 4))


def test_phase_grid_mismatch_is_rejected():
    values = [torch.zeros(1, 8, 2, 3, 4) for _ in range(3)]
    values[2] = torch.zeros(1, 8, 3, 3, 4)
    with pytest.raises(ValueError, match="share shape"):
        join_phase_latents(values)


def model_record():
    return {"patient_id": "example", "pair_id": "example:T0-T1", "split": "train",
            "earlier_stage": "T0", "later_stage": "T1", "delta_days": 21,
            "interval_missing": False, "interval_source": "verified_relative_dicom_study_date",
            "baseline_clinical": {"age": 50}, "treatment": {"treatment_arm": "Paclitaxel"}}


class CoupledVelocity(nn.Module):
    def forward(self, x, timesteps, context):
        return torch.cat((x[:, 24:], torch.ones_like(x[:, 24:])), dim=1)


def test_sampling_uses_source_only_and_evolves_both_branches():
    model = ThreePhaseSymmFlow([model_record()], source=_release_path('@repo/.'), backbone=CoupledVelocity())
    source = torch.full((1, 24, 2, 2, 2), 2.0)
    noise = torch.zeros_like(source)
    result = model.sample(source, [model_record()], noise, steps=2)
    # Euler reads y=2 then y=2.5; holding the source branch fixed gives only 2.
    torch.testing.assert_close(result, torch.full_like(result, 2.25))
    assert model.tokens([model_record()]).shape == (1, 11, 256)


def cohort_fixture():
    cohort = {"visits": [], "split": {"train": ["a", "b", "c"], "val": ["heldout"]}}
    world = {"pairs": []}
    for pid in ("a", "b", "c", "heldout"):
        for stage in range(4):
            row = {"canonical_patient_id": pid, "patient_id": pid, "visit": f"T{stage}",
                   "visit_id": f"{pid}:T{stage}", "split": "val" if pid == "heldout" else "train",
                   "late_index": 3, "remote_phases": {"pre": "/example_aqc_0.nii.gz"},
                   "crop_geometry": {"origin": stage}}
            cohort["visits"].append(row)
            if stage:
                world["pairs"].append({**model_record(), "patient_id": pid, "later_stage": f"T{stage}",
                                       "earlier_visit_id": f"{pid}:T0", "later_visit_id": f"{pid}:T{stage}"})
    cfg = {"seed": 2026, "selection": {"train_patients": 2, "validation_patients": 1}}
    return cohort, world, cfg


def test_selection_excludes_original_holdout_and_uses_only_t0_grid():
    cohort, world, cfg = cohort_fixture()
    visits, pairs = select_patients(cohort, world, cfg)
    assert len(visits) == 12 and len(pairs) == 9
    assert "heldout" not in {row["patient_id"] for row in visits}
    assert all(row["source_geometry"] == {"origin": 0} for row in visits)
    assert all(row["grid_visit_id"].endswith(":T0") for row in pairs)
    train = {row["patient_id"] for row in pairs if row["split"] == "train"}
    val = {row["patient_id"] for row in pairs if row["split"] == "val"}
    assert train.isdisjoint(val)
    assert select_patients(copy.deepcopy(cohort), copy.deepcopy(world), cfg) == (visits, pairs)


def test_generated_pillar_adapter_needs_no_future_images_or_masks(monkeypatch):
    from mewm_ispy2 import three_phase_pilot_data as data
    monkeypatch.setattr(data, "IMAGE_SHAPE", (2, 3, 4))
    calls = []
    def channel(array, spacing):
        calls.append((array.copy(), spacing))
        return torch.from_numpy(array).permute(1, 2, 0)
    helper = SimpleNamespace(pillar_channel=channel, PILLAR_SHAPE=(3, 3, 4, 2))
    monkeypatch.setattr(data, "pcr_helpers", lambda cfg: (helper, {}))
    prediction = np.stack([np.full((2, 3, 4), i, np.float32) for i in (1, 2, 3)])
    support = np.ones((2, 3, 4), dtype=bool)
    support[0, 0, 0] = False
    volume = data.build_generated_pillar_input({}, prediction, {"spacing_xyz_mm": [1, 2, 3]}, support,
                                               {"mean": 10, "std": 2})
    assert volume.shape == (3, 3, 4, 2)
    assert [float(array[1, 1, 1]) for array, _ in calls] == [12, 14, 16]
    assert all(array[0, 0, 0] == 0 and spacing == [3, 2, 1] for array, spacing in calls)


def test_centering_retains_baseline_scale_and_uses_target_center_only_for_supervision():
    from mewm_ispy2.three_phase_pilot_data import CENTERED_GRID, FIXED_GRID, training_geometry
    source = {"shape_zyx": [6, 8, 10], "spacing_xyz_mm": [1, 2, 3],
              "origin_lps_mm": [0, 0, 0], "direction_lps": np.eye(3).ravel().tolist()}
    target = {**source, "origin_lps_mm": [100, -50, 200], "spacing_xyz_mm": [2, 4, 6]}
    visit = {"visit": "T3", "source_geometry": source, "crop_geometry": target}
    actual = training_geometry(visit, CENTERED_GRID)
    assert actual["spacing_xyz_mm"] == source["spacing_xyz_mm"]
    assert actual["shape_zyx"] == source["shape_zyx"]
    np.testing.assert_allclose(actual["origin_lps_mm"], [104.5, -43, 207.5])
    assert training_geometry(visit, FIXED_GRID) == source
    assert training_geometry({**visit, "visit": "T0"}, CENTERED_GRID) == source
    assert source["origin_lps_mm"] == [0, 0, 0]


def test_latent_statistics_do_not_fit_validation_values():
    from mewm_ispy2.three_phase_pilot_training import fit_latent_statistics
    visits = [{"visit_id": "a", "pilot_split": "train", "canonical_patient_id": "train"},
              {"visit_id": "b", "pilot_split": "val", "canonical_patient_id": "validation"}]
    latents = {"a": torch.tensor([-1.0, 1.0]).repeat(24, 1).view(24, 1, 1, 2),
               "b": torch.full((24, 1, 1, 2), float("nan"))}
    actual = fit_latent_statistics({"visits": visits}, latents)
    assert actual["mean"] == [0.0] * 24 and actual["std"] == [1.0] * 24
    assert actual["patients"] == actual["unique_visits"] == 1
    latents["a"][0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="Invalid training"):
        fit_latent_statistics({"visits": visits}, latents)


def test_source_only_latent_loading_never_opens_future_or_training_caches(monkeypatch):
    from mewm_ispy2.three_phase_pilot_training import load_latents
    from contextlib import contextmanager
    visits = [{"visit_id": "source", "pilot_split": "val", "visit": "T0", "cache_file": "source.npz"},
              {"visit_id": "future", "pilot_split": "val", "visit": "T3", "cache_file": "forbidden.npz"},
              {"visit_id": "train", "pilot_split": "train", "visit": "T0", "cache_file": "forbidden.npz"}]
    opened = []
    @contextmanager
    def guarded_load(path, **kwargs):
        assert path.name == "source.npz"
        opened.append(path.name)
        yield {"latent": np.zeros((24, 24, 64, 64), np.float32)}
    monkeypatch.setattr(np, "load", guarded_load)
    values = load_latents({"output_root": "/example"}, {"visits": visits}, source_only=True)
    assert list(values) == ["source"] and opened == ["source.npz"]


def test_metric_rejects_nonfinite_predictions_even_outside_foreground(monkeypatch):
    from mewm_ispy2 import three_phase_pilot_data as data
    monkeypatch.setattr(data, "IMAGE_SHAPE", (2, 3, 4))
    reference = np.ones((3, 2, 3, 4), np.float32)
    prediction = reference.copy()
    mask = np.ones_like(reference, dtype=bool)
    mask[:, 0, 0, 0] = False
    prediction[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        data.phase_errors(prediction, reference, mask)
    with pytest.raises(ValueError, match="no foreground"):
        data.phase_errors(reference, reference, np.zeros_like(mask))


def test_validation_exception_restores_training_mode_and_rng(monkeypatch):
    from mewm_ispy2 import three_phase_pilot_training as training
    model = nn.Linear(2, 2).train()
    original = torch.get_rng_state()
    monkeypatch.setattr(training, "rng_state", torch.get_rng_state)
    monkeypatch.setattr(training, "restore_rng", torch.set_rng_state)
    def fail(*args, **kwargs):
        torch.rand(10)
        assert not model.training
        raise ValueError("intentional validation failure")
    monkeypatch.setattr(training, "_evaluate", fail)
    with pytest.raises(ValueError, match="intentional"):
        training.evaluate({}, {}, model, None, step=1)
    assert model.training
    assert torch.equal(original, torch.get_rng_state())


def test_quality_admission_excludes_entire_training_patient_and_never_filters_validation(tmp_path):
    from mewm_ispy2.three_phase_pilot_data import CENTERED_GRID, admit_inventory
    cohort, world, selection = cohort_fixture()
    visits, pairs = select_patients(cohort, world, selection)
    cfg = {"output_root": str(tmp_path), "spatial_policy": CENTERED_GRID,
           "quality": {"minimum_source_support_coverage": 0.5, "exclude_failing_training_patients": True}}
    audit = {"spatial_policy": CENTERED_GRID,
             "records": [{"case_index": i, "source_support_coverage": 0.2 if i == 0 else 1.0} for i in range(3)]}
    inventory, report = admit_inventory(cfg, {"visits": visits, "pairs": pairs}, audit)
    assert report["patients"] == {"train": 1, "val": 1, "excluded_geometry": 1}
    excluded = [v for v in inventory["visits"] if v["pilot_split"] == "excluded_geometry"]
    assert len(excluded) == 4
    assert sum(p["split"] == "excluded_geometry" for p in inventory["pairs"]) == 3
    audit["records"][2]["source_support_coverage"] = 0.2
    with pytest.raises(ValueError, match="Validation.*cannot be filtered"):
        admit_inventory(cfg, {"visits": visits, "pairs": pairs}, audit)


def test_original_symm_objective_optimizer_and_ema_resume_match_uninterrupted(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from mewm_ispy2.first_post_world_training import atomic_checkpoint, train_step
    from ispy2_symmflow.training.ema import ExponentialMovingAverage
    from ispy2_symmflow.training.engine import build_warmup_cosine_scheduler
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    class TinyVelocity(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(48, 48, 1)
            self.condition = nn.Linear(256, 48)
        def forward(self, x, timesteps, context):
            return self.conv(x) + self.condition(context.mean(1))[:, :, None, None, None]
    def build():
        model = ThreePhaseSymmFlow([model_record()], source=_release_path('@repo/.'), backbone=TinyVelocity())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=1, total_steps=4)
        ema = ExponentialMovingAverage(model)
        return model, optimizer, scheduler, ema
    torch.manual_seed(71)
    source, target = torch.randn(1, 24, 2, 2, 2), torch.randn(1, 24, 2, 2, 2)
    batch = {"source": source, "target": target, "records": [model_record()]}
    model, optimizer, scheduler, ema = build()
    before = model.velocity_model.state_dict()
    before = {k: v.clone() for k, v in before.items()}
    metrics = train_step(model, optimizer, scheduler, ema, iter([batch]), 1, "cpu")
    assert set(metrics) == {"loss", "gradient_norm", "x", "y"}
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(before[k], v) for k, v in model.velocity_model.state_dict().items())
    path = tmp_path / "recovery.ckpt"
    atomic_checkpoint(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "ema": ema.state_dict(),
                            "rng": torch.get_rng_state()}, tmp_path)
    expected = train_step(model, optimizer, scheduler, ema, iter([batch]), 1, "cpu")
    restored, restored_optimizer, restored_scheduler, restored_ema = build()
    saved = torch.load(path, weights_only=False)
    restored.load_state_dict(saved["model"])
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_scheduler.load_state_dict(saved["scheduler"])
    restored_ema.load_state_dict(saved["ema"])
    torch.set_rng_state(saved["rng"])
    actual = train_step(restored, restored_optimizer, restored_scheduler, restored_ema, iter([batch]), 1, "cpu")
    assert actual == expected
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    for name, value in ema.shadow.items():
        torch.testing.assert_close(restored_ema.shadow[name], value, rtol=0, atol=0)
    assert scheduler.state_dict() == restored_scheduler.state_dict()
