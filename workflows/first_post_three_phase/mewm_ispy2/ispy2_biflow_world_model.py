from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .ispy2_biflow_backbone import (
    ISPY2_BIFLOW_PRESET,
    build_original_ispy2_controlled_biflownet,
)
from .mu_glioma_world_model import (
    MU_EMPTY_CLINICAL_TEXT,
    MUFourierDays,
    MUTextTower,
)

class ISPY2BiFlowNonImageConditioner(nn.Module):
    """Build text, interval, and target-stage tokens without image features."""

    token_count = 4

    def __init__(
        self,
        text_tower: MUTextTower,
        *,
        text_hidden_size: int | None = None,
        context_dim: int = 768,
        time_fourier_dim: int = 128,
    ) -> None:
        super().__init__()
        if not callable(getattr(text_tower, "encode", None)):
            raise TypeError("I-SPY2 conditioner text tower must implement encode")
        hidden_size = text_hidden_size or getattr(text_tower, "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("I-SPY2 conditioner requires a positive text hidden size")
        self.text_tower = text_tower
        self.text_hidden_size = hidden_size
        self.context_dim = int(context_dim)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, context_dim)
        )
        self.text_role_embedding = nn.Embedding(2, context_dim)
        self.days_encoder = MUFourierDays(time_fourier_dim)
        self.days_projection = nn.Sequential(
            nn.LayerNorm(time_fourier_dim), nn.Linear(time_fourier_dim, context_dim)
        )
        self.target_stage_embedding = nn.Embedding(4, context_dim)

    def forward(
        self,
        *,
        clinical_text: Sequence[str],
        treatment_text: Sequence[str],
        delta_days: torch.Tensor,
        target_stage: torch.Tensor,
    ) -> torch.Tensor:
        batch = delta_days.shape[0]
        if (
            isinstance(clinical_text, (str, bytes))
            or isinstance(treatment_text, (str, bytes))
            or len(clinical_text) != batch
            or len(treatment_text) != batch
        ):
            raise ValueError("I-SPY2 text conditions require one value per patient")
        clinical = [
            value
            if isinstance(value, str) and value.strip()
            else MU_EMPTY_CLINICAL_TEXT
            for value in clinical_text
        ]
        if any(
            not isinstance(value, str) or not value.strip()
            for value in treatment_text
        ):
            raise ValueError("I-SPY2 treatment text values must be nonempty")
        if delta_days.shape != (batch,) or not bool(torch.isfinite(delta_days).all()):
            raise ValueError("I-SPY2 delta days must be finite [B]")
        if (
            target_stage.shape != (batch,)
            or not bool(((target_stage >= 1) & (target_stage <= 3)).all())
        ):
            raise ValueError("I-SPY2 target stage must be T1, T2, or T3")

        hidden = self.text_tower.encode([*clinical, *list(treatment_text)])
        if hidden.shape != (2 * batch, self.text_hidden_size):
            raise RuntimeError("I-SPY2 text tower output shape changed")
        projection = self.text_projection[1]
        hidden = hidden.to(
            device=projection.weight.device, dtype=projection.weight.dtype
        )
        text = F.normalize(self.text_projection(hidden), p=2, dim=-1).reshape(
            2, batch, self.context_dim
        )
        roles = self.text_role_embedding(
            torch.arange(2, device=text.device)
        ).to(dtype=text.dtype)
        days = self.days_projection(
            self.days_encoder(
                delta_days.to(self.days_projection[1].weight.device)
            )
        ).to(device=text.device, dtype=text.dtype)
        stage = self.target_stage_embedding(
            target_stage.to(self.target_stage_embedding.weight.device).long()
        ).to(device=text.device, dtype=text.dtype)
        return torch.stack(
            (text[0] + roles[0], text[1] + roles[1], days, stage), dim=1
        )



@dataclass(frozen=True)
class ISPY2BiFlowPreparedContext:
    tokens: torch.Tensor
    spatial_condition: torch.Tensor


@dataclass(frozen=True)
class ISPY2BiFlowOutput:
    velocity: torch.Tensor


class ISPY2BiFlowWorldModel(nn.Module):
    """Future-DCE0 RF model whose spatial dynamics input is only flow state."""

    def __init__(
        self,
        *,
        conditioner: ISPY2BiFlowNonImageConditioner,
        dynamics: nn.Module,
        latent_channels: int,
        context_dim: int,
    ) -> None:
        super().__init__()
        if latent_channels <= 0 or context_dim <= 0:
            raise ValueError("I-SPY2 BiFlowNet dimensions must be positive")
        self.conditioner = conditioner
        self.dynamics = dynamics
        self.latent_channels = latent_channels
        self.context_dim = context_dim

    @property
    def architecture_contract(self) -> dict[str, object]:
        dynamics_contract = getattr(self.dynamics, "architecture_contract", None)
        if not isinstance(dynamics_contract, dict):
            raise RuntimeError("I-SPY2 dynamics architecture contract is unavailable")
        return {
            "spatial_input": "flow_state_only",
            "latent_channels": self.latent_channels,
            "context_dim": self.context_dim,
            "context_tokens": self.conditioner.token_count,
            "image_condition_path": "controlnet_only",
            "dynamics": dynamics_contract,
        }

    def prepare_context(
        self,
        source_mri: torch.Tensor,
        *,
        clinical_text: Sequence[str],
        treatment_text: Sequence[str],
        delta_days: torch.Tensor,
        target_stage: torch.Tensor,
    ) -> ISPY2BiFlowPreparedContext:
        if source_mri.ndim != 5 or source_mri.shape[1] != 2 or not source_mri.is_floating_point() or not bool(torch.isfinite(source_mri).all()):
            raise ValueError("I-SPY2 source DCE0+SER must be finite [B,2,D,H,W]")
        tokens = self.conditioner(
            clinical_text=clinical_text,
            treatment_text=treatment_text,
            delta_days=delta_days,
            target_stage=target_stage,
        )
        return ISPY2BiFlowPreparedContext(tokens=tokens, spatial_condition=source_mri)

    def velocity_from_context(
        self,
        *,
        flow_state: torch.Tensor,
        flow_time: torch.Tensor,
        prepared: ISPY2BiFlowPreparedContext,
    ) -> ISPY2BiFlowOutput:
        if (
            flow_state.ndim != 6
            or flow_state.shape[1] != 1
            or flow_state.shape[2] != self.latent_channels
            or not bool(torch.isfinite(flow_state).all())
        ):
            raise ValueError("I-SPY2 flow state must be finite [B,1,C,D,H,W]")
        batch = flow_state.shape[0]
        if flow_time.shape != (batch,) or not bool(torch.isfinite(flow_time).all()):
            raise ValueError("I-SPY2 flow time must be finite [B]")
        if (
            prepared.tokens.ndim != 3
            or prepared.tokens.shape[0] != batch
            or prepared.tokens.shape[2] != self.context_dim
        ):
            raise ValueError("I-SPY2 prepared context shape is invalid")
        if prepared.spatial_condition.ndim != 5 or prepared.spatial_condition.shape[:2] != (batch, 2):
            raise ValueError("I-SPY2 prepared ControlNet condition shape is invalid")
        flat_state = flow_state.reshape(
            batch, self.latent_channels, *flow_state.shape[3:]
        )
        velocity = self.dynamics(
            flat_state,
            flow_time.to(device=flat_state.device, dtype=flat_state.dtype),
            context=prepared.tokens,
            spatial_condition=prepared.spatial_condition,
        )
        if velocity.shape != flat_state.shape:
            raise RuntimeError("I-SPY2 BiFlowNet velocity shape changed")
        return ISPY2BiFlowOutput(velocity=velocity.unsqueeze(1))

    def forward(
        self,
        *,
        source_mri: torch.Tensor,
        flow_state: torch.Tensor,
        flow_time: torch.Tensor,
        clinical_text: Sequence[str],
        treatment_text: Sequence[str],
        delta_days: torch.Tensor,
        target_stage: torch.Tensor,
    ) -> ISPY2BiFlowOutput:
        prepared = self.prepare_context(
            source_mri,
            clinical_text=clinical_text,
            treatment_text=treatment_text,
            delta_days=delta_days,
            target_stage=target_stage,
        )
        return self.velocity_from_context(
            flow_state=flow_state,
            flow_time=flow_time,
            prepared=prepared,
        )


def build_ispy2_biflow_world_model(
    *,
    text_tower: MUTextTower,
    text_hidden_size: int | None = None,
    preset: str = ISPY2_BIFLOW_PRESET,
    latent_channels: int = 8,
    context_dim: int = 768,
) -> ISPY2BiFlowWorldModel:
    if preset != ISPY2_BIFLOW_PRESET:
        raise ValueError(f"unsupported I-SPY2 BiFlowNet preset: {preset}")
    conditioner = ISPY2BiFlowNonImageConditioner(
        text_tower,
        text_hidden_size=text_hidden_size,
        context_dim=context_dim,
    )
    dynamics = build_original_ispy2_controlled_biflownet(
        input_channels=latent_channels,
        output_channels=latent_channels,
        context_dim=context_dim,
    )
    return ISPY2BiFlowWorldModel(
        conditioner=conditioner,
        dynamics=dynamics,
        latent_channels=latent_channels,
        context_dim=context_dim,
    )


__all__ = [
    "ISPY2BiFlowNonImageConditioner",
    "ISPY2BiFlowOutput",
    "ISPY2BiFlowPreparedContext",
    "ISPY2BiFlowWorldModel",
    "build_ispy2_biflow_world_model",
]
