from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import copy
from pathlib import Path

import pytest
import yaml

from mewm_ispy2.ispy2_biflow_backbone import ISPY2_BIFLOW_PRESET
from mewm_ispy2.ispy2_biflow_config import load_ispy2_biflow_config


CONFIG = Path("configs/ispy2_dce0_biflow_original_v1.yaml").resolve()
STABLE_CONFIG = Path(
    "configs/ispy2_dce0_biflow_channel_zscore_stable_v3.yaml"
).resolve()


def test_config_exposes_preset_but_not_internal_architecture_fields() -> None:
    config = load_ispy2_biflow_config(CONFIG)

    assert config.preset == ISPY2_BIFLOW_PRESET
    assert set(config.raw["model"]) == {"preset"}
    forbidden = {
        "dim",
        "dim_mults",
        "patch_size",
        "sub_volume_size",
        "dit_heads",
        "attention_heads",
        "norm_groups",
        "attention_levels",
    }
    assert forbidden.isdisjoint(config.raw["model"])
    assert config.identity_payload()["spatial_input"] == "flow_state_only"
    assert config.identity_payload()["image_condition_path"] == "controlnet_only"
    assert config.identity_payload()["prediction_target"] == "full_future_state"
    assert "base_config_identity_sha256" not in config.identity_payload()


def test_config_rejects_ad_hoc_architecture_override(tmp_path: Path) -> None:
    raw = copy.deepcopy(_release_yaml(CONFIG.read_text(encoding="utf-8")))
    raw["model"]["dim"] = 96
    candidate = tmp_path / "override.yaml"
    candidate.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="model fields are invalid"):
        load_ispy2_biflow_config(candidate)


def test_config_has_no_hardware_or_acceptance_thresholds() -> None:
    raw = _release_yaml(CONFIG.read_text(encoding="utf-8"))
    base = _release_yaml(Path(raw["base_config"]).read_text(encoding="utf-8"))
    serialized = yaml.safe_dump({"run": raw, "base": base}).lower()

    assert "32gb" not in serialized
    assert "memory_threshold" not in serialized
    assert "throughput_threshold" not in serialized
    assert "acceptance" not in serialized
    assert "metric_threshold" not in serialized
    assert "fm-bcmri" not in serialized
    assert "fmbcmri" not in serialized
    assert "image_tower" not in serialized


def test_stable_config_splits_backbone_and_locks_warm_start() -> None:
    config = load_ispy2_biflow_config(STABLE_CONFIG)

    assert config.training.optimizer_layout == "split_backbone_v1"
    assert config.training.backbone_learning_rate == pytest.approx(1e-5)
    assert config.training.controlnet_learning_rate == pytest.approx(1e-4)
    assert config.training.conditioner_learning_rate == pytest.approx(1e-4)
    assert config.training.text_lora_learning_rate == pytest.approx(1e-5)
    assert config.training.min_learning_rate == pytest.approx(1e-6)
    assert config.training.warmup_fraction == pytest.approx(0.05)
    assert config.training.divergence_reference_metric == pytest.approx(
        0.504343808
    )
    assert config.training.divergence_multiplier == pytest.approx(1.5)
    assert config.training.divergence_patience == 2
    assert config.runtime.warm_start_checkpoint is not None
    assert config.runtime.warm_start_checkpoint.name == "best-014.ckpt"
    assert config.runtime.warm_start_sha256 == (
        "9897d14f48ffd66bf4be9989625c7d2aa8c3cc5e391d00213ad779e7dc362886"
    )


def test_stable_config_rejects_minimum_lr_above_backbone_peak(
    tmp_path: Path,
) -> None:
    raw = copy.deepcopy(_release_yaml(STABLE_CONFIG.read_text(encoding="utf-8")))
    raw["training"]["min_learning_rate"] = 2e-5
    candidate = tmp_path / "invalid-stable.yaml"
    candidate.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="exceeds a peak learning rate"):
        load_ispy2_biflow_config(candidate)
