from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Sequence

from .cache import RegisteredStrictAROICache
from .ispy2_biflow_config import ISPY2BiFlowConfig
from .ispy2_biflow_latent_contract import ISPY2BiFlowLatentCache
from .ispy2_dce0_world_data import ISPY2DCE0WorldPair


_MEDGEMMA_REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)
_TRAINING_PACKAGES = ("bitsandbytes", "peft", "transformers")


def _huggingface_hub_root() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    hf_home = Path(
        os.environ.get("HF_HOME", "~/.cache/huggingface")
    ).expanduser()
    return (hf_home / "hub").resolve()


def _latent_payload_path(root: Path, visit_id: str) -> Path:
    if not visit_id or "/" in visit_id or "\\" in visit_id:
        raise ValueError("I-SPY2 BiFlowNet latent visit ID is unsafe")
    return root / "visits" / f"{visit_id.replace(':', '__')}.pt"


def audit_ispy2_biflow_training_assets(
    config: ISPY2BiFlowConfig,
    *,
    pairs: Sequence[ISPY2DCE0WorldPair],
    latent_cache: ISPY2BiFlowLatentCache,
    roi_cache: RegisteredStrictAROICache,
) -> dict[str, Any]:
    target_ids = tuple(sorted({pair.target_visit_id for pair in pairs}))
    source_visits = {
        pair.source_visit_id: pair.source_visit for pair in pairs
    }

    unregistered_targets = sorted(set(target_ids) - latent_cache.visit_ids)
    missing_target_files = [
        visit_id
        for visit_id in target_ids
        if not _latent_payload_path(latent_cache.root, visit_id).is_file()
    ]
    missing_source_files: list[str] = []
    for visit_id in sorted(source_visits):
        try:
            cache_key = roi_cache.cache_key(visit_id)
        except ValueError:
            missing_source_files.append(visit_id)
            continue
        if not (roi_cache.root / f"{cache_key}.pt").is_file():
            missing_source_files.append(visit_id)

    hub_root = _huggingface_hub_root()
    medgemma_snapshot = (
        hub_root
        / f"models--{config.base.text.model_id.replace('/', '--')}"
        / "snapshots"
        / config.base.text.revision
    )
    missing_medgemma = [
        name
        for name in _MEDGEMMA_REQUIRED_FILES
        if not (medgemma_snapshot / name).is_file()
    ]
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in _TRAINING_PACKAGES
    }

    ready = not any(
        (
            unregistered_targets,
            missing_target_files,
            missing_source_files,
            missing_medgemma,
        )
    ) and all(packages.values())
    return {
        "ready": ready,
        "target_latents": {
            "required_visits": len(target_ids),
            "registered_visits": len(set(target_ids) & latent_cache.visit_ids),
            "missing_files": len(missing_target_files),
            "missing_examples": missing_target_files[:10],
            "unregistered_visits": len(unregistered_targets),
            "unregistered_examples": unregistered_targets[:10],
        },
        "source_roi_cache": {
            "required_visits": len(source_visits),
            "missing_files": len(missing_source_files),
            "missing_examples": missing_source_files[:10],
        },
        "controlnet": {
            "type": "biflownet_encoder_clone_zero_conv",
            "condition": "raw_dce0_ser",
            "external_checkpoint_required": False,
        },
        "medgemma": {
            "model_id": config.base.text.model_id,
            "revision": config.base.text.revision,
            "snapshot": str(medgemma_snapshot),
            "missing_files": missing_medgemma,
        },
        "packages": packages,
    }


__all__ = ["audit_ispy2_biflow_training_assets"]
