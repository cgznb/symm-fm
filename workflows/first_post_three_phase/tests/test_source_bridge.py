from __future__ import annotations

import contextlib
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from mewm_ispy2.source_bridge import (
    BRIDGE_SCHEMA,
    BridgeSystem,
    EpochSampler,
    channel_statistics,
    flow_path,
    initial_state,
    integrate,
    public_metadata,
    write_json,
)
from mewm_ispy2.source_bridge_evaluation import (
    change_metrics,
    decode_and_evaluate,
    patient_macro,
)
from mewm_ispy2.source_bridge_workflow import (
    FinalValidationCheckpoint,
    collate,
    load_experiment,
)


@pytest.mark.parametrize("shape", [(2, 1, 8, 2, 3, 4), (2, 32, 2, 3, 4)])
def test_bridge_path_and_exact_constant_velocity_integration(shape):
    generator = torch.Generator().manual_seed(9)
    source = torch.randn(shape, generator=generator)
    target = torch.randn(shape, generator=generator)
    noise = torch.randn(shape, generator=generator)
    settings = {
        "distribution": "source_gaussian",
        "multiplier": 0.25,
        "channel_std": torch.linspace(0.1, 0.9, shape[-4]),
    }
    initial = initial_state(source, noise, **settings)
    for time in (0.0, 0.35, 1.0):
        state, velocity = flow_path(
            source, target, noise, torch.full((shape[0],), time), **settings
        )
        torch.testing.assert_close(state, (1 - time) * initial + time * target)
        torch.testing.assert_close(velocity, target - initial)
        result = integrate(
            lambda state, time, velocity=velocity: velocity,
            source,
            noise,
            steps=8,
            **settings,
        )
        torch.testing.assert_close(result, target, atol=2e-6, rtol=2e-6)
    assert torch.equal(
        initial_state(
            source,
            noise,
            distribution="standard_normal",
            multiplier=1.0,
            channel_std=settings["channel_std"],
        ),
        noise,
    )


def test_noise_is_scaled_per_channel_and_source_is_not_mutated():
    source = torch.full((1, 1, 2, 2, 2, 2), 3.0)
    original = source.clone()
    result = initial_state(
        source,
        torch.ones_like(source),
        distribution="source_gaussian",
        channel_std=torch.tensor([2.0, 4.0]),
        multiplier=0.5,
    )
    assert torch.all(result[:, :, 0] == 4)
    assert torch.all(result[:, :, 1] == 5)
    assert torch.equal(source, original)


@pytest.mark.parametrize("change", ["dtype", "shape", "nonfinite", "scale", "mode"])
def test_invalid_bridge_inputs_fail(change):
    source = torch.zeros(1, 2, 2, 2, 2)
    noise = torch.ones_like(source)
    settings = {
        "distribution": "source_gaussian",
        "channel_std": torch.ones(2),
        "multiplier": 0.5,
    }
    if change == "dtype":
        noise = noise.double()
    elif change == "shape":
        noise = noise[:, :1]
    elif change == "nonfinite":
        noise[0, 0, 0, 0, 0] = float("nan")
    elif change == "scale":
        settings["channel_std"][0] = 0
    else:
        settings["distribution"] = "source_copy"
    with pytest.raises(ValueError):
        initial_state(source, noise, **settings)


def test_streaming_statistics_match_population_moments():
    values = [
        torch.arange(16).reshape(2, 2, 2, 2).float(),
        torch.arange(16, 32).reshape(2, 2, 2, 2).float(),
    ]
    result = channel_statistics(iter(values))
    expected = torch.cat([value.reshape(2, -1) for value in values], dim=1).double()
    torch.testing.assert_close(
        torch.tensor(result["channel_std"], dtype=torch.float64),
        expected.std(dim=1, correction=0),
    )
    assert result["visits"] == 2
    assert result["voxel_count_per_channel"] == 16


class TinyVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def prepare_context(self, source_mri, **kwargs):
        return source_mri

    def velocity_from_context(self, *, flow_state, flow_time, prepared):
        return SimpleNamespace(velocity=flow_state * self.weight + prepared)


def contract(root):
    return {
        "schema": BRIDGE_SCHEMA,
        "experiment": {
            "distribution": "source_gaussian",
            "noise_multiplier": 0.25,
            "seed": 5,
            "validation_steps": 2,
            "validation_samples": 2,
            "output_root": str(root),
        },
        "statistics": {"channel_std": [0.4, 0.6]},
        "pair_order": ["p0", "p1"],
    }


def sample(index):
    return {
        "source_latent": torch.ones(2, 2, 2, 2) * index,
        "target_latent": torch.ones(2, 2, 2, 2) * (index + 1),
        "source_mri": torch.ones(2, 2, 2, 2) * 0.1,
        "source_mask_onehot": torch.zeros(4, 2, 2, 2),
        "clinical_text": "",
        "treatment_text": "",
        "delta_days": torch.tensor(1.0),
        "metadata": {"pair_id": f"p{index}", "patient_id": f"patient{index}"},
    }


def system(root):
    return BridgeSystem(
        TinyVelocity(),
        family="mu",
        experiment_config=None,
        contract=contract(root),
        optimizer_factory=lambda system: torch.optim.AdamW(
            system.parameters(), lr=0.01
        ),
    )


def test_generation_requires_no_target_and_is_pair_order_independent(tmp_path):
    model = system(tmp_path).eval()
    batch = collate([sample(0), sample(1)])
    batch.pop("target_latent")
    result = model.predict_batch(batch, steps=2, samples=2)
    reversed_batch = collate([sample(1), sample(0)])
    reversed_batch.pop("target_latent")
    reversed_result = model.predict_batch(reversed_batch, steps=2, samples=2)
    torch.testing.assert_close(result[:, [1, 0]], reversed_result)


def test_cpu_training_checkpoint_roundtrip_and_contract_rejection(tmp_path):
    model = system(tmp_path)
    samples = [sample(0), sample(1)]
    train_loader = DataLoader(
        samples, batch_size=1, collate_fn=collate, sampler=EpochSampler(2, 5)
    )
    val_loader = DataLoader(samples, batch_size=1, collate_fn=collate)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=1,
        max_steps=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    original = model.model.weight.detach().clone()
    trainer.fit(model, train_loader, val_loader)
    assert trainer.global_step == 2
    assert not torch.equal(model.model.weight, original)
    path = tmp_path / "bridge.ckpt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    restored = system(tmp_path)
    restored.on_load_checkpoint(payload)
    restored.load_state_dict(payload["state_dict"])
    torch.testing.assert_close(restored.model.weight, model.model.weight)
    changed = copy.deepcopy(payload)
    changed["source_bridge_contract"]["experiment"]["noise_multiplier"] = 0.5
    with pytest.raises(ValueError, match="contract mismatch"):
        restored.on_load_checkpoint(changed)
    changed = copy.deepcopy(payload)
    changed["state_dict"].pop("model.weight")
    with pytest.raises(ValueError, match="incomplete"):
        restored.on_load_checkpoint(changed)


@pytest.mark.parametrize("final_score", [0.05, 5.0])
def test_final_validation_selects_best_and_persists_callback_state(
    tmp_path, final_score
):
    class ScoredBridge(BridgeSystem):
        validation_score = 1.0

        def on_validation_epoch_end(self):
            self.log("val/endpoint_mae", self.validation_score)

    model = ScoredBridge(
        TinyVelocity(),
        family="mu",
        experiment_config=None,
        contract=contract(tmp_path),
        optimizer_factory=lambda model: torch.optim.AdamW(model.parameters(), lr=0.01),
    )
    samples = [sample(0), sample(1), sample(0)]
    train_loader = DataLoader(
        samples, batch_size=1, collate_fn=collate, sampler=EpochSampler(3, 5)
    )
    val_loader = DataLoader(samples[:2], batch_size=1, collate_fn=collate)
    checkpoint = FinalValidationCheckpoint(
        dirpath=tmp_path / "checkpoints",
        filename="best",
        monitor="val/endpoint_mae",
        mode="min",
        save_top_k=1,
        save_last=True,
        enable_version_counter=False,
        save_on_train_epoch_end=False,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=-1,
        max_steps=4,
        accumulate_grad_batches=2,
        val_check_interval=4,
        check_val_every_n_epoch=None,
        callbacks=[checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)
    previous = torch.load(checkpoint.best_model_path, weights_only=True)
    assert previous["global_step"] < trainer.global_step
    model.validation_score = final_score
    trainer.validate(model, val_loader, verbose=False)
    assert float(checkpoint.best_model_score) == pytest.approx(1.0)
    checkpoint.save_final_validation(trainer)
    expected = min(1.0, final_score)
    assert float(checkpoint.best_model_score) == pytest.approx(expected)
    best = torch.load(checkpoint.best_model_path, weights_only=True)
    last = torch.load(checkpoint.last_model_path, weights_only=True)
    assert best["global_step"] == (
        trainer.global_step if final_score < 1.0 else previous["global_step"]
    )
    assert last["global_step"] == trainer.global_step
    restored = FinalValidationCheckpoint(dirpath=tmp_path / "checkpoints")
    restored.load_state_dict(last["callbacks"][checkpoint.state_key])
    assert float(restored.best_model_score) == pytest.approx(expected)
    assert restored.best_model_path == checkpoint.best_model_path
    torch.testing.assert_close(last["state_dict"]["model.weight"], model.model.weight)


def test_artifact_metadata_excludes_legacy_digests(tmp_path):
    payload = {
        "path": "/data",
        "sha256": "legacy",
        "nested": {"revision": "old", "count": 8, "cache_hash": "old"},
    }
    write_json(tmp_path / "artifact.json", payload)
    assert json.loads((tmp_path / "artifact.json").read_text()) == {
        "path": "/data",
        "nested": {"count": 8},
    }
    assert public_metadata({"count": 1}) == {"count": 1}


def test_large_finite_gradients_are_measured_and_clipped_without_overflow(tmp_path):
    model = system(tmp_path)
    model.log = lambda *args, **kwargs: None
    optimizer = model.configure_optimizers()
    model.model.weight.grad = torch.full_like(model.model.weight, 1e30)
    model.on_before_optimizer_step(optimizer)
    assert model.latest_gradient_norms["total"] == pytest.approx(1e30)
    model.configure_gradient_clipping(optimizer, 1.0, "norm")
    assert float(model.model.weight.grad) == pytest.approx(1.0)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_actual_nonfinite_gradients_are_rejected(tmp_path, value):
    model = system(tmp_path)
    model.model.weight.grad = torch.full_like(model.model.weight, value)
    with pytest.raises(FloatingPointError, match="gradients are non-finite"):
        model.on_before_optimizer_step(model.configure_optimizers())


def test_numerical_continuation_retains_optimizer_scheduler_and_total_steps(tmp_path):
    from mewm_ispy2.source_bridge_workflow import (
        migrate_continuation_checkpoint,
        validate_numerical_continuation,
    )

    def optimizer_factory(model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    loader = DataLoader(
        [sample(0), sample(1)], collate_fn=collate, sampler=EpochSampler(2, 5)
    )
    trainer_args = dict(
        accelerator="cpu", max_epochs=-1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
        gradient_clip_val=1.0,
    )
    parent = system(tmp_path / "parent")
    parent.contract["experiment"]["max_steps"] = 40
    parent.optimizer_factory = optimizer_factory
    trainer = pl.Trainer(max_steps=2, **trainer_args)
    trainer.fit(parent, loader)
    source = tmp_path / "parent.ckpt"
    trainer.save_checkpoint(source)
    current = copy.deepcopy(parent.contract)
    current["experiment"].update(
        output_root=str(tmp_path / "continued"), numerical_policy="local_dit_fp32_v1",
        attention_subvolume_batch=16, continuation_checkpoint=str(source),
    )
    for key, value in (("max_steps", 80), ("noise_multiplier", 0.5)):
        changed = copy.deepcopy(current)
        changed["experiment"][key] = value
        with pytest.raises(ValueError, match="experiment contract"):
            validate_numerical_continuation(parent.contract, changed)
    changed = copy.deepcopy(current)
    changed["statistics"]["channel_std"][0] = 1.0
    with pytest.raises(ValueError, match="experiment contract"):
        validate_numerical_continuation(parent.contract, changed)
    project = SimpleNamespace(experiment=current["experiment"], contract=lambda stats: current)
    path, step = migrate_continuation_checkpoint(project, current)
    migrated = torch.load(path, weights_only=True)
    original = torch.load(source, weights_only=True)
    assert step == 2 and migrated["callbacks"] == {}
    assert migrated["loops"] == original["loops"]
    assert migrated["lr_schedulers"] == original["lr_schedulers"]
    assert "numerical_policy" not in original["source_bridge_contract"]["experiment"]
    torch.testing.assert_close(
        migrated["optimizer_states"][0]["state"][0]["exp_avg"],
        original["optimizer_states"][0]["state"][0]["exp_avg"],
    )
    resumed = system(tmp_path / "continued")
    resumed.contract = current
    resumed.optimizer_factory = optimizer_factory
    seen = {}

    class ObserveResume(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            seen.update(
                step=trainer.global_step,
                lr=trainer.optimizers[0].param_groups[0]["lr"],
                scheduler_step=trainer.lr_scheduler_configs[0].scheduler.last_epoch,
            )

    resumed_trainer = pl.Trainer(max_steps=4, callbacks=[ObserveResume()], **trainer_args)
    resumed_trainer.fit(resumed, loader, ckpt_path=path)
    assert seen == {"step": 2, "lr": 0.005, "scheduler_step": 2}
    assert resumed_trainer.global_step == 4
    assert resumed_trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.0025)
    assert float(resumed_trainer.optimizers[0].state[resumed.model.weight]["step"]) == 4


def test_experiment_rejects_unknown_fields(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("schema: unknown\n")
    with pytest.raises(ValueError):
        load_experiment(path)


def test_image_aggregation_weights_patients_equally():
    rows = [
        {
            "patient_id": patient,
            "method": "generation",
            "modality": "dce0",
            "region": "foreground",
            "mae": value,
        }
        for patient, value in (("a", 0.0), ("a", 2.0), ("b", 5.0))
    ]
    result = patient_macro(rows)
    assert result[0]["mean"] == 3.0
    assert result[0]["patient_count"] == 2


def test_change_metrics_distinguish_copy_from_future_change():
    source = torch.ones(2, 2, 2)
    target = source * 2
    region = torch.ones_like(source, dtype=torch.bool)
    copied = change_metrics(source, source, target, region)
    perfect = change_metrics(target, source, target, region)
    assert copied["predicted_change_mae"] == 0
    assert copied["change_cosine"] is None
    assert perfect["change_cosine"] == pytest.approx(1)


def test_image_evaluation_decodes_each_candidate_before_averaging(
    tmp_path, monkeypatch
):
    if not (
        Path(__file__).parents[1] / "mewm_ispy2/ispy2_biflow_cohort_evaluation.py"
    ).exists():
        pytest.skip("I-SPY2 image adapter is exercised in the I-SPY2 project")
    from mewm_ispy2 import backend, ispy2_biflow_latent_contract, workflows
    from mewm_ispy2 import source_bridge_evaluation as evaluation

    shape = (1, 1, 1, 12, 12, 12)
    latents = torch.stack([torch.zeros(shape), torch.ones(shape)])
    (tmp_path / "candidates").mkdir()
    shard = tmp_path / "candidates" / "00000.pt"
    torch.save({"pair_id": "p0", "samples": latents}, shard)
    source = SimpleNamespace(
        image=torch.full((1, 12, 12, 12), 0.2),
        mask=torch.ones(1, 12, 12, 12),
        valid_foreground=torch.ones(1, 12, 12, 12, dtype=torch.bool),
    )
    target = copy.deepcopy(source)
    target.image.fill_(0.5)
    pair = SimpleNamespace(
        pair_id="p0",
        patient_id="patient",
        source_visit="source",
        target_visit_id="target",
    )
    project = SimpleNamespace(
        experiment={"family": "ispy2"},
        val_pairs=[pair],
        base=SimpleNamespace(
            base=SimpleNamespace(
                data=SimpleNamespace(
                    vqgan_checkpoint="unused",
                    bundle_json="unused",
                    phase_manifest_csv="unused",
                )
            )
        ),
        conditions=SimpleNamespace(
            load=lambda visit: source if visit == "source" else target
        ),
        latents=SimpleNamespace(denormalize=lambda value: value),
        val_dataset=SimpleNamespace(
            dataset=SimpleNamespace(
                _target_latent=lambda key: torch.full(shape[1:], math.sqrt(0.5))
            )
        ),
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(
        torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext()
    )
    codec = torch.nn.Identity()
    monkeypatch.setattr(codec, "cuda", lambda: codec)
    monkeypatch.setattr(workflows, "load_mri_vqgan", lambda *args, **kwargs: codec)
    monkeypatch.setattr(
        backend,
        "load_transition_records",
        lambda *args, **kwargs: SimpleNamespace(visits={"target": "target"}),
    )
    monkeypatch.setattr(
        ispy2_biflow_latent_contract,
        "decode_ispy2_biflow_continuous",
        lambda codec, latent: latent.square(),
    )
    rendered = {}

    def capture(path, source, target, reconstruction, mean, std, mask, modalities):
        rendered.update(mean=mean.clone(), std=std.clone())

    monkeypatch.setattr(evaluation, "render_comparison", capture)
    result = decode_and_evaluate(project, tmp_path, {"p0": 0})
    assert result["pairs"] == 1
    torch.testing.assert_close(rendered["mean"], torch.full_like(target.image, 0.5))
    torch.testing.assert_close(rendered["std"], torch.full_like(target.image, 0.5))
    assert not shard.exists()


@pytest.mark.parametrize("family", ["ispy2", "mu"])
def test_optimizer_adapter_retains_complete_disjoint_parameter_groups(family):
    from mewm_ispy2.source_bridge_workflow import optimizer_factory

    if (
        family == "mu"
        and not (
            Path(__file__).parents[1] / "mewm_ispy2/mu_glioma_biflow_training.py"
        ).exists()
    ):
        pytest.skip("MU adapter is exercised in the MU project")

    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.ones(2))
            self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)

        def lora_parameters(self):
            return (self.adapter,)

    model = torch.nn.Module()
    model.dynamics = torch.nn.Module()
    model.dynamics.backbone = torch.nn.Linear(2, 2)
    model.dynamics.controlnet = torch.nn.Linear(2, 2)
    model.conditioner = torch.nn.Module()
    model.conditioner.projection = torch.nn.Linear(2, 2)
    model.conditioner.text_tower = Text()
    training = SimpleNamespace(
        optimizer_layout="legacy_joint_v1",
        dynamics_learning_rate=1e-4,
        backbone_learning_rate=1e-4,
        controlnet_learning_rate=1e-4,
        conditioner_learning_rate=1e-4,
        text_lora_learning_rate=1e-5,
        weight_decay=0.05,
        optimizer_betas=(0.9, 0.999),
        warmup_fraction=0.05,
        minimum_learning_rate=1e-6,
    )
    holder = SimpleNamespace(
        family=family,
        model=model,
        experiment_config=SimpleNamespace(training=training),
        contract={"experiment": {"max_steps": 2000}},
    )
    configured = optimizer_factory(holder)
    optimizer = configured["optimizer"] if isinstance(configured, dict) else configured
    parameters = [
        id(value) for group in optimizer.param_groups for value in group["params"]
    ]
    assert len(parameters) == len(set(parameters))
    assert set(parameters) == {
        id(value) for value in model.parameters() if value.requires_grad
    }


def test_registered_arm_configs_have_matched_budgets_and_separate_outputs():
    root = Path(__file__).parents[1]
    arms = [
        load_experiment(root / "configs" / f"source_bridge_pilot_{arm}.yaml")
        for arm in ("pure_noise", "bridge025", "bridge050")
    ]
    for key in (
        "base_config",
        "max_steps",
        "seed",
        "accumulate_grad_batches",
        "batch_size",
        "statistics_file",
        "validation_steps",
        "validation_samples",
    ):
        assert len({arm[key] for arm in arms}) == 1
    assert len({arm["output_root"] for arm in arms}) == 3
    assert [arm["noise_multiplier"] for arm in arms] == [1.0, 0.25, 0.5]


def test_mu_image_adapter_decodes_each_candidate_before_averaging(
    tmp_path, monkeypatch
):
    if not (
        Path(__file__).parents[1] / "mewm_ispy2/mu_glioma_biflow_cohort_evaluation.py"
    ).exists():
        pytest.skip("MU image adapter is exercised in the MU project")
    from mewm_ispy2 import mu_glioma_biflow_cohort_evaluation as cohort
    from mewm_ispy2 import mu_glioma_biflow_workflow as workflow
    from mewm_ispy2 import source_bridge_evaluation as evaluation

    shape = (1, 32, 12, 12, 12)
    (tmp_path / "candidates").mkdir()
    torch.save(
        {
            "pair_id": "p0",
            "samples": torch.stack([torch.zeros(shape), torch.ones(shape)]),
        },
        tmp_path / "candidates/00000.pt",
    )
    source = torch.full((4, 12, 12, 12), 0.2)
    target = torch.full_like(source, 0.5)
    foreground = torch.ones_like(source, dtype=torch.bool)
    mask = torch.ones(1, 12, 12, 12)
    pair = SimpleNamespace(
        pair_id="p0",
        patient_id="patient",
        source_samples="source",
        target_samples="target",
    )
    project = SimpleNamespace(
        experiment={"family": "mu"},
        val_pairs=[pair],
        base=SimpleNamespace(
            data=SimpleNamespace(modalities=("t1c", "t1n", "t2f", "t2w"))
        ),
        conditions=SimpleNamespace(load_visit=lambda visit: (source, mask)),
        latents=SimpleNamespace(codebook_min=-1, codebook_max=1),
        val_dataset=SimpleNamespace(
            dataset=SimpleNamespace(
                _latent=lambda samples: torch.full(shape[1:], math.sqrt(0.5))
            )
        ),
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    monkeypatch.setattr(
        torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        workflow, "_load_vqgan", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(
        workflow,
        "decode_mu_glioma_biflow_prediction",
        lambda codec, latent, **kwargs: latent.reshape(1, 4, 8, 12, 12, 12)[
            :, :, :1
        ].square(),
    )
    monkeypatch.setattr(
        cohort, "_decoded_modalities", lambda value, name: value[0, :, 0].float()
    )
    monkeypatch.setattr(
        cohort,
        "_load_formal_visit",
        lambda samples, **kwargs: (
            source if samples == "source" else target,
            foreground,
            mask,
        ),
    )
    rendered = {}

    def capture(path, source, target, reconstruction, mean, std, mask, modalities):
        rendered.update(mean=mean.clone(), std=std.clone())

    monkeypatch.setattr(evaluation, "render_comparison", capture)
    result = decode_and_evaluate(project, tmp_path, {"p0": 0})
    assert result["pairs"] == 1
    torch.testing.assert_close(rendered["mean"], torch.full_like(source, 0.5))
    torch.testing.assert_close(rendered["std"], torch.full_like(source, 0.5))
