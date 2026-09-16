from __future__ import annotations

import inspect
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mewm_ispy2.contracts import DCE0Phase
from mewm_ispy2.ispy2_biflow_backbone import (
    ISPY2_BIFLOW_PRESET,
    ISPY2BiFlowControlNet,
    ISPY2BiFlowPreset,
    ISPY2ConditionalBiFlowNet,
    ISPY2ControlledBiFlowNet,
    ORIGINAL_ISPY2_BIFLOW_PRESET,
)
from mewm_ispy2.ispy2_biflow_data import ISPY2BiFlowPairDataset
from mewm_ispy2.ispy2_biflow_training import (
    ISPY2_BIFLOW_CHECKPOINT_SCHEMA,
    ISPY2_BIFLOW_CHECKPOINT_STATE_POLICY,
    ISPY2_BIFLOW_LEGACY_CHECKPOINT_SCHEMA,
    ISPY2BiFlowTrainingSystem,
    build_ispy2_biflow_batch,
    integrate_ispy2_biflow_euler,
)
from mewm_ispy2.ispy2_biflow_world_model import (
    ISPY2BiFlowNonImageConditioner,
    ISPY2BiFlowWorldModel,
)
from mewm_ispy2.ispy2_biflow_workflow import (
    _DivergenceGuard,
    _warm_start_weights,
)
from mewm_ispy2.ispy2_dce0_world_data import ISPY2DCE0WorldPair
from mewm_ispy2.manifest import VisitRecord


def _latent(value: float, shape: tuple[int, ...] = (1, 1, 8, 3, 5, 6)) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float32)


def _small_preset() -> ISPY2BiFlowPreset:
    return replace(
        ORIGINAL_ISPY2_BIFLOW_PRESET,
        dim=24,
        dim_mults=(1, 1, 2),
        sub_volume_size=(2, 2, 2),
        dit_heads=4,
        attention_heads=2,
        norm_groups=8,
        attention_levels=(False, False, False),
        downsample_after=(False, True, False),
        upsample_after=(True, False, False),
        context_pool_heads=4,
    )


def test_original_preset_is_versioned_and_not_run_configurable() -> None:
    preset = ORIGINAL_ISPY2_BIFLOW_PRESET

    assert preset.name == ISPY2_BIFLOW_PRESET
    assert preset.dim == 72
    assert preset.dim_mults == (1, 1, 2, 4, 8)
    assert preset.sub_volume_size == (8, 8, 8)
    assert preset.patch_size == (1, 1, 1)
    assert (
        preset.local_encoder_blocks,
        preset.local_mid_blocks,
        preset.local_decoder_blocks,
    ) == (2, 1, 2)
    assert preset.attention_levels == (False, False, False, True, True)
    assert preset.downsample_after == (False, True, True, True, False)
    assert preset.upsample_after == (True, True, True, False, False)


def test_flow_state_only_backbone_restores_nondivisible_shape_and_gradients() -> None:
    model = ISPY2ConditionalBiFlowNet(
        input_channels=2,
        output_channels=2,
        context_dim=12,
        preset=_small_preset(),
    )
    sample = torch.randn(1, 2, 3, 5, 6, requires_grad=True)
    output = model(
        sample,
        torch.tensor([0.4]),
        context=torch.randn(1, 5, 12),
    )

    assert output.shape == sample.shape
    assert model.architecture_contract["spatial_input"] == "flow_state_only"
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()



def test_controlnet_condition_encoder_matches_latent_spatial_shape() -> None:
    backbone = ISPY2ConditionalBiFlowNet(
        input_channels=8,
        output_channels=8,
        context_dim=12,
        preset=_small_preset(),
    )
    controlnet = ISPY2BiFlowControlNet(backbone)

    encoded = controlnet.encode_spatial_condition(
        torch.randn(1, 2, 12, 20, 24)
    )

    assert encoded.shape == (1, 24, 3, 5, 6)
    assert torch.count_nonzero(encoded) == 0


def test_zero_initialized_controlnet_preserves_backbone_output() -> None:
    torch.manual_seed(7)
    backbone = ISPY2ConditionalBiFlowNet(
        input_channels=8,
        output_channels=8,
        context_dim=12,
        preset=_small_preset(),
    )
    controlled = ISPY2ControlledBiFlowNet(
        backbone,
        ISPY2BiFlowControlNet(backbone),
    ).eval()
    sample = torch.randn(1, 8, 3, 5, 6)
    flow_time = torch.tensor([0.4])
    context = torch.randn(1, 4, 12)

    with torch.no_grad():
        expected = backbone(sample, flow_time, context=context)
        actual = controlled(
            sample,
            flow_time,
            context=context,
            spatial_condition=torch.randn(1, 2, 12, 20, 24),
        )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_trained_control_residual_makes_velocity_depend_on_dce0_ser() -> None:
    torch.manual_seed(11)
    backbone = ISPY2ConditionalBiFlowNet(
        input_channels=8,
        output_channels=8,
        context_dim=12,
        preset=_small_preset(),
    )
    controlnet = ISPY2BiFlowControlNet(backbone)
    nn.init.normal_(controlnet.condition_encoder[-1].weight, std=0.02)
    nn.init.normal_(controlnet.zero_middle.weight, std=0.02)
    controlled = ISPY2ControlledBiFlowNet(backbone, controlnet).eval()
    sample = torch.randn(1, 8, 3, 5, 6)
    flow_time = torch.tensor([0.4])
    context = torch.randn(1, 4, 12)
    first_condition = torch.randn(1, 2, 12, 20, 24)
    second_condition = first_condition.clone()
    second_condition[:, 1].add_(1.0)

    with torch.no_grad():
        first = controlled(
            sample,
            flow_time,
            context=context,
            spatial_condition=first_condition,
        )
        second = controlled(
            sample,
            flow_time,
            context=context,
            spatial_condition=second_condition,
        )

    assert first.shape == sample.shape
    assert not torch.equal(first, second)


class _TargetLatentSpy:
    def __init__(self, target_visit_id: str) -> None:
        self.target_visit_id = target_visit_id
        self.calls: list[str] = []

    def load(self, visit_id: str) -> torch.Tensor:
        if visit_id != self.target_visit_id:
            raise AssertionError("source latent was accessed")
        self.calls.append(visit_id)
        return torch.zeros((8, 24, 64, 64), dtype=torch.float16)


class _ROISpy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def load_source_mri(self, visit: VisitRecord) -> torch.Tensor:
        self.calls.append(visit.visit_id)
        return torch.zeros((2, 96, 256, 256), dtype=torch.float16)


def _pair() -> ISPY2DCE0WorldPair:
    patient_id = "ISPY2-TEST"
    path = Path("source.nii.gz")
    visit = VisitRecord(
        visit_id=f"{patient_id}:T0",
        patient_id=patient_id,
        visit="T0",
        dce_paths=(path,),
        dce0=DCE0Phase(path, 0, 1),
        mask_path=Path("source-mask.nii.gz"),
        meta_path=Path("source-meta.json"),
        qc_status="ok",
        registration_status="fixed_reference",
    )
    return ISPY2DCE0WorldPair(
        pair_id=f"{patient_id}:T0->T1",
        patient_id=patient_id,
        split="train",
        transition_type="T0->T1",
        source_visit_id=visit.visit_id,
        target_visit_id=f"{patient_id}:T1",
        source_stage=0,
        target_stage=1,
        delta_days=14,
        source_visit=visit,
        clinical_text="clinical condition",
        treatment_text="treatment condition",
        adjacent_transition_ids=(f"{patient_id}:T0->T1",),
    )


def test_dataset_loads_only_target_latent_and_keeps_source_mri_conditions() -> None:
    pair = _pair()
    latent_loader = _TargetLatentSpy(pair.target_visit_id)
    roi_cache = _ROISpy()
    sample = ISPY2BiFlowPairDataset(
        (pair,), latent_loader, roi_cache, split="train"
    )[0]

    assert "source_latent" not in sample
    assert sample["target_latent"].shape == (1, 8, 24, 64, 64)
    assert sample["target_latent"].dtype == torch.float32
    assert sample["source_mri"].shape == (2, 96, 256, 256)
    assert latent_loader.calls == [pair.target_visit_id]
    assert roi_cache.calls == [pair.source_visit_id]


def test_rectified_flow_uses_full_target_and_euler_has_no_source_latent() -> None:
    target = _latent(3.0)
    noise = _latent(-1.0)
    flow = build_ispy2_biflow_batch(
        target, noise=noise, flow_time=torch.tensor([0.25])
    )

    torch.testing.assert_close(flow.endpoint, target)
    torch.testing.assert_close(flow.flow_state, _latent(0.0))
    torch.testing.assert_close(flow.target_velocity, _latent(4.0))
    signature = inspect.signature(integrate_ispy2_biflow_euler)
    assert "source_latent" not in signature.parameters


class _LoRATextTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.register_parameter(
            "base_weight", nn.Parameter(torch.ones(()), requires_grad=False)
        )
        self.model.register_parameter(
            "lora_adapter", nn.Parameter(torch.zeros(()))
        )

    def lora_parameters(self) -> tuple[nn.Parameter, ...]:
        return (self.model.lora_adapter,)


class _Conditioner(nn.Module):
    token_count = 4

    def __init__(self) -> None:
        super().__init__()
        self.text_tower = _LoRATextTower()
        self.condition_scale = nn.Parameter(torch.ones(()))

    def forward(self, **conditions: object) -> torch.Tensor:
        delta_days = conditions["delta_days"]
        assert isinstance(delta_days, torch.Tensor)
        return torch.ones(delta_days.shape[0], 4, 6) * self.condition_scale


class _Dynamics(nn.Module):
    architecture_contract = {
        "spatial_input": "flow_state_only",
        "input_channels": 8,
        "output_channels": 8,
    }

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))
        self.last_input_channels: int | None = None
        self.last_spatial_condition: torch.Tensor | None = None

    def forward(
        self,
        sample: torch.Tensor,
        flow_time: torch.Tensor,
        *,
        context: torch.Tensor,
        spatial_condition: torch.Tensor,
    ) -> torch.Tensor:
        del flow_time, context
        self.last_input_channels = sample.shape[1]
        self.last_spatial_condition = spatial_condition
        return torch.ones_like(sample) + self.offset


class _EmbeddingTextTower(nn.Module):
    hidden_size = 5

    def encode(self, texts: list[str]) -> torch.Tensor:
        return torch.arange(
            len(texts) * self.hidden_size, dtype=torch.float32
        ).reshape(len(texts), self.hidden_size)


def test_non_image_conditioner_produces_exactly_four_tokens() -> None:
    conditioner = ISPY2BiFlowNonImageConditioner(
        _EmbeddingTextTower(), context_dim=6
    )

    tokens = conditioner(
        clinical_text=["clinical", ""],
        treatment_text=["drug A", "drug B"],
        delta_days=torch.tensor([14.0, 28.0]),
        target_stage=torch.tensor([1, 3]),
    )

    assert tokens.shape == (2, 4, 6)
    assert torch.isfinite(tokens).all()


def test_world_model_euler_reuses_context_and_integrates_noise_only() -> None:
    model = ISPY2BiFlowWorldModel(
        conditioner=_Conditioner(),
        dynamics=_Dynamics(),
        latent_channels=8,
        context_dim=6,
    )
    noise = _latent(0.0, (1, 1, 8, 2, 2, 2))
    endpoint = integrate_ispy2_biflow_euler(
        model,
        source_mri=torch.zeros(1, 2, 96, 256, 256),
        clinical_text=["clinical"],
        treatment_text=["treatment"],
        delta_days=torch.tensor([10.0]),
        target_stage=torch.tensor([1]),
        solver_steps=4,
        noise=noise,
    )

    torch.testing.assert_close(endpoint, torch.ones_like(noise))
    assert model.dynamics.last_input_channels == 8
    assert model.dynamics.last_spatial_condition is not None
    assert model.dynamics.last_spatial_condition.shape == (1, 2, 96, 256, 256)


def test_optimizer_contains_only_dynamics_and_text_groups() -> None:
    config = SimpleNamespace(
        raw={"model": {"preset": ISPY2_BIFLOW_PRESET}},
        training=SimpleNamespace(
            optimizer_layout="legacy_joint_v1",
            dynamics_learning_rate=1e-4,
            backbone_learning_rate=1e-4,
            controlnet_learning_rate=1e-4,
            conditioner_learning_rate=1e-4,
            text_lora_learning_rate=2e-4,
            weight_decay=0.05,
        ),
    )
    model = ISPY2BiFlowWorldModel(
        conditioner=_Conditioner(),
        dynamics=_Dynamics(),
        latent_channels=8,
        context_dim=6,
    )
    optimizer = ISPY2BiFlowTrainingSystem(
        model, config=config, checkpoint_identity={}
    ).configure_optimizers()

    assert [group["name"] for group in optimizer.param_groups] == [
        "dynamics",
        "text",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 2e-4]


class _SplitDynamics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.controlnet = nn.Linear(2, 2)


def test_stable_optimizer_splits_modules_without_parameter_overlap() -> None:
    training = SimpleNamespace(
        optimizer_layout="split_backbone_v1",
        dynamics_learning_rate=1e-4,
        backbone_learning_rate=1e-5,
        controlnet_learning_rate=1e-4,
        conditioner_learning_rate=1e-4,
        text_lora_learning_rate=1e-5,
        min_learning_rate=1e-6,
        warmup_fraction=0.05,
        weight_decay=0.05,
    )
    config = SimpleNamespace(raw={}, training=training)
    model = ISPY2BiFlowWorldModel(
        conditioner=_Conditioner(),
        dynamics=_SplitDynamics(),
        latent_channels=8,
        context_dim=6,
    )
    system = ISPY2BiFlowTrainingSystem(
        model, config=config, checkpoint_identity={}
    )
    system._trainer = SimpleNamespace(estimated_stepping_batches=100)

    configured = system.configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]["scheduler"]

    assert [group["name"] for group in optimizer.param_groups] == [
        "backbone",
        "controlnet",
        "conditioner",
        "text",
    ]
    grouped_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert [group["initial_lr"] for group in optimizer.param_groups] == [
        1e-5,
        1e-4,
        1e-4,
        1e-5,
    ]
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [2e-6, 2e-5, 2e-5, 2e-6]
    )
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-5, 1e-4, 1e-4, 1e-5]
    )
    for _ in range(95):
        optimizer.step()
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-6] * 4
    )


def test_warm_start_loads_only_model_state_with_locked_contract(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(raw={})
    source_model = ISPY2BiFlowWorldModel(
        conditioner=_Conditioner(),
        dynamics=_Dynamics(),
        latent_channels=8,
        context_dim=6,
    )
    target_model = ISPY2BiFlowWorldModel(
        conditioner=_Conditioner(),
        dynamics=_Dynamics(),
        latent_channels=8,
        context_dim=6,
    )
    identity = {
        key: {"value": key}
        for key in (
            "architecture",
            "continuous_cache_identity",
            "bundle_contract_sha256",
            "vqgan_sha256",
            "source_modalities",
            "text_model_id",
            "text_revision",
            "prediction_target",
            "velocity_loss",
            "checkpoint_state_policy",
        )
    }
    source = ISPY2BiFlowTrainingSystem(
        source_model, config=config, checkpoint_identity=identity
    )
    target = ISPY2BiFlowTrainingSystem(
        target_model, config=config, checkpoint_identity=identity
    )
    with torch.no_grad():
        source.model.dynamics.offset.fill_(3.0)
        target.model.dynamics.offset.fill_(-2.0)
    checkpoint = {
        "state_dict": source.state_dict(),
        "epoch": 14,
        "global_step": 17340,
    }
    source.on_save_checkpoint(checkpoint)
    path = tmp_path / "best.ckpt"
    torch.save(checkpoint, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    result = _warm_start_weights(
        target,
        checkpoint_path=path,
        expected_sha256=digest,
        target_identity=identity,
    )

    torch.testing.assert_close(
        target.model.dynamics.offset, source.model.dynamics.offset
    )
    assert result["source_epoch"] == 14
    assert result["source_global_step"] == 17340
    assert result["optimizer_state_loaded"] is False
    assert result["scheduler_state_loaded"] is False


def test_warm_start_allows_loader_to_report_subset_of_omitted_quantization_state(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(raw={})
    source = ISPY2BiFlowTrainingSystem(
        ISPY2BiFlowWorldModel(
            conditioner=_Conditioner(),
            dynamics=_Dynamics(),
            latent_channels=8,
            context_dim=6,
        ),
        config=config,
        checkpoint_identity={},
    )
    target = ISPY2BiFlowTrainingSystem(
        ISPY2BiFlowWorldModel(
            conditioner=_Conditioner(),
            dynamics=_Dynamics(),
            latent_channels=8,
            context_dim=6,
        ),
        config=config,
        checkpoint_identity={},
    )
    checkpoint = {
        "state_dict": source.state_dict(),
        "epoch": 1,
        "global_step": 2,
    }
    source.on_save_checkpoint(checkpoint)
    extra_omitted = "model.conditioner.text_tower.model.quantization_metadata"
    checkpoint["ispy2_biflow_omitted_state_keys"].append(extra_omitted)
    original = target._omitted_frozen_text_state_keys
    target._omitted_frozen_text_state_keys = lambda: (*original(), extra_omitted)
    path = tmp_path / "quantized.ckpt"
    torch.save(checkpoint, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    result = _warm_start_weights(
        target,
        checkpoint_path=path,
        expected_sha256=digest,
        target_identity={},
    )

    assert result["optimizer_state_loaded"] is False


def test_divergence_guard_requires_consecutive_threshold_failures() -> None:
    guard = _DivergenceGuard(reference=0.5, multiplier=1.5, patience=2)
    trainer = SimpleNamespace(
        sanity_checking=False,
        callback_metrics={"val/velocity_mae": torch.tensor(0.76)},
        should_stop=False,
    )

    guard.on_validation_end(trainer, nn.Identity())
    assert trainer.should_stop is False
    trainer.callback_metrics["val/velocity_mae"] = torch.tensor(0.70)
    guard.on_validation_end(trainer, nn.Identity())
    assert guard.consecutive_failures == 0
    trainer.callback_metrics["val/velocity_mae"] = torch.tensor(0.80)
    guard.on_validation_end(trainer, nn.Identity())
    guard.on_validation_end(trainer, nn.Identity())
    assert trainer.should_stop is True


def test_checkpoint_omits_only_frozen_text_base_and_keeps_lora() -> None:
    config = SimpleNamespace(raw={"model": {"preset": ISPY2_BIFLOW_PRESET}})
    model = ISPY2BiFlowWorldModel(
        conditioner=_Conditioner(),
        dynamics=_Dynamics(),
        latent_channels=8,
        context_dim=6,
    )
    system = ISPY2BiFlowTrainingSystem(
        model, config=config, checkpoint_identity={"id": "test"}
    )
    checkpoint: dict[str, object] = {"state_dict": system.state_dict()}

    system.on_save_checkpoint(checkpoint)

    state = checkpoint["state_dict"]
    assert isinstance(state, dict)
    assert (
        "model.conditioner.text_tower.model.base_weight"
        not in state
    )
    assert (
        "model.conditioner.text_tower.model.lora_adapter"
        in state
    )
    assert "model.dynamics.offset" in state
    system.on_load_checkpoint(checkpoint)

    del state["model.dynamics.offset"]
    with pytest.raises(ValueError, match="state mismatch"):
        system.on_load_checkpoint(checkpoint)


def test_checkpoint_schema_isolated_from_legacy_world_model() -> None:
    config = SimpleNamespace(raw={"model": {"preset": ISPY2_BIFLOW_PRESET}})
    system = ISPY2BiFlowTrainingSystem(
        nn.Identity(), config=config, checkpoint_identity={"preset": ISPY2_BIFLOW_PRESET}
    )
    checkpoint: dict[str, object] = {"state_dict": system.state_dict()}
    system.on_save_checkpoint(checkpoint)

    assert checkpoint["ispy2_biflow_schema"] == ISPY2_BIFLOW_CHECKPOINT_SCHEMA
    assert checkpoint["ispy2_biflow_state_policy"] == (
        ISPY2_BIFLOW_CHECKPOINT_STATE_POLICY
    )
    system.on_load_checkpoint(checkpoint)
    checkpoint["ispy2_biflow_schema"] = "mewm_ispy2_dce0_world_checkpoint_v1"
    with pytest.raises(ValueError, match="identity mismatch"):
        system.on_load_checkpoint(checkpoint)


def test_checkpoint_accepts_v2_only_for_exact_legacy_minmax_identity() -> None:
    current_identity = {
        "schema": "mewm_ispy2_dce0_biflow_identity_v2",
        "config_identity_sha256": "locked-config",
        "continuous_cache_identity": {"normalization": "continuous_codebook_minmax_v1"},
    }
    config = SimpleNamespace(
        raw={},
        base=SimpleNamespace(
            model=SimpleNamespace(
                latent_normalization="continuous_codebook_minmax_v1"
            )
        ),
    )
    system = ISPY2BiFlowTrainingSystem(
        nn.Identity(), config=config, checkpoint_identity=current_identity
    )
    checkpoint: dict[str, object] = {"state_dict": system.state_dict()}
    system.on_save_checkpoint(checkpoint)
    legacy_identity = dict(current_identity)
    legacy_identity["schema"] = "mewm_ispy2_dce0_biflow_identity_v1"
    checkpoint["ispy2_biflow_schema"] = ISPY2_BIFLOW_LEGACY_CHECKPOINT_SCHEMA
    checkpoint["ispy2_biflow_identity"] = legacy_identity

    system.on_load_checkpoint(checkpoint)

    checkpoint["ispy2_biflow_identity"] = {
        **legacy_identity,
        "config_identity_sha256": "different-config",
    }
    with pytest.raises(ValueError, match="identity mismatch"):
        system.on_load_checkpoint(checkpoint)

    config.base.model.latent_normalization = (
        "continuous_train_unique_visit_channel_zscore_v1"
    )
    checkpoint["ispy2_biflow_identity"] = legacy_identity
    with pytest.raises(ValueError, match="identity mismatch"):
        system.on_load_checkpoint(checkpoint)
