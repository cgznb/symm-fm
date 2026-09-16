from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from .cache import RegisteredStrictAROICache
from .ispy2_dce0_world_config import (
    ISPY2_DCE0_WORLD_LATENT_SHAPE,
    ISPY2_DCE0_WORLD_SOURCE_MODALITIES,
    ISPY2_DCE0_WORLD_SOURCE_MRI_SHAPE,
)
from .ispy2_dce0_world_data import ContinuousLatentLoader, ISPY2DCE0WorldPair


class ISPY2BiFlowPairDataset(Dataset[dict[str, Any]]):
    """I-SPY2 RF samples with target latent only; source MRI remains context."""

    def __init__(
        self,
        pairs: Sequence[ISPY2DCE0WorldPair],
        latent_loader: ContinuousLatentLoader,
        roi_cache: RegisteredStrictAROICache,
        *,
        split: str | None = None,
    ) -> None:
        if split not in (None, "train", "val"):
            raise ValueError("I-SPY2 BiFlowNet dataset split is invalid")
        if not hasattr(latent_loader, "load"):
            raise TypeError("I-SPY2 BiFlowNet latent loader is invalid")
        if not hasattr(roi_cache, "load_source_mri"):
            raise TypeError("I-SPY2 BiFlowNet ROI cache is invalid")
        self.pairs = tuple(
            pair for pair in pairs if split is None or pair.split == split
        )
        if any(not isinstance(pair, ISPY2DCE0WorldPair) for pair in self.pairs):
            raise TypeError("I-SPY2 BiFlowNet dataset pairs are invalid")
        self.latent_loader = latent_loader
        self.roi_cache = roi_cache

    def __len__(self) -> int:
        return len(self.pairs)

    def _target_latent(self, visit_id: str) -> torch.Tensor:
        latent = self.latent_loader.load(visit_id)
        if (
            not isinstance(latent, torch.Tensor)
            or not latent.is_floating_point()
            or tuple(latent.shape) != ISPY2_DCE0_WORLD_LATENT_SHAPE
            or not bool(torch.isfinite(latent).all())
        ):
            raise ValueError(f"I-SPY2 BiFlowNet target latent is invalid: {visit_id}")
        return latent.to(dtype=torch.float32).contiguous().unsqueeze(0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        source_mri = self.roi_cache.load_source_mri(pair.source_visit)
        if (
            not isinstance(source_mri, torch.Tensor)
            or source_mri.dtype != torch.float16
            or tuple(source_mri.shape) != ISPY2_DCE0_WORLD_SOURCE_MRI_SHAPE
            or not bool(torch.isfinite(source_mri).all())
        ):
            raise ValueError("I-SPY2 BiFlowNet source MRI is invalid")
        return {
            "source_mri": source_mri.contiguous(),
            "target_latent": self._target_latent(pair.target_visit_id),
            "clinical_text": pair.clinical_text,
            "treatment_text": pair.treatment_text,
            "delta_days": torch.tensor(pair.delta_days, dtype=torch.float32),
            "target_stage": torch.tensor(pair.target_stage, dtype=torch.long),
            "metadata": {
                "pair_id": pair.pair_id,
                "patient_id": pair.patient_id,
                "split": pair.split,
                "transition_type": pair.transition_type,
                "source_visit_id": pair.source_visit_id,
                "target_visit_id": pair.target_visit_id,
                "source_stage": pair.source_stage,
                "target_stage": pair.target_stage,
                "source_modalities": ISPY2_DCE0_WORLD_SOURCE_MODALITIES,
                "adjacent_transition_ids": pair.adjacent_transition_ids,
            },
        }


__all__ = ["ISPY2BiFlowPairDataset"]
