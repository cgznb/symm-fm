from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from .ispy2_biflow_config import ISPY2BiFlowConfig
from .ispy2_biflow_world_model import ISPY2BiFlowWorldModel


ISPY2_BIFLOW_CHECKPOINT_SCHEMA = "mewm_ispy2_dce0_biflow_checkpoint_v3"
ISPY2_BIFLOW_LEGACY_CHECKPOINT_SCHEMA = "mewm_ispy2_dce0_biflow_checkpoint_v2"
ISPY2_BIFLOW_LEGACY_IDENTITY_SCHEMA = "mewm_ispy2_dce0_biflow_identity_v1"
ISPY2_BIFLOW_CURRENT_IDENTITY_SCHEMA = "mewm_ispy2_dce0_biflow_identity_v2"
ISPY2_BIFLOW_LEGACY_LATENT_NORMALIZATION = "continuous_codebook_minmax_v1"
ISPY2_BIFLOW_CHECKPOINT_STATE_POLICY = (
    "omit_rebuildable_frozen_medgemma_v1"
)
_FROZEN_TEXT_STATE_PREFIX = "model.conditioner.text_tower.model."


@dataclass(frozen=True)
class ISPY2BiFlowBatch:
    flow_state: torch.Tensor
    target_velocity: torch.Tensor
    endpoint: torch.Tensor
    noise: torch.Tensor
    flow_time: torch.Tensor


def _validate_latent(value: torch.Tensor, name: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 6
        or value.shape[1] != 1
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"I-SPY2 {name} must be finite floating [B,1,C,D,H,W]")


def build_ispy2_biflow_batch(
    target_latent: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    flow_time: torch.Tensor | None = None,
) -> ISPY2BiFlowBatch:
    """Construct full-target RF without reading the source DCE0 latent."""

    _validate_latent(target_latent, "target latent")
    endpoint = target_latent
    if noise is None:
        noise = torch.randn_like(endpoint)
    if noise.shape != endpoint.shape or not bool(torch.isfinite(noise).all()):
        raise ValueError("I-SPY2 BiFlowNet noise is invalid")
    batch = endpoint.shape[0]
    if flow_time is None:
        flow_time = torch.rand(batch, device=endpoint.device, dtype=endpoint.dtype)
    if (
        flow_time.shape != (batch,)
        or not bool(torch.isfinite(flow_time).all())
        or not bool(((flow_time >= 0.0) & (flow_time <= 1.0)).all())
    ):
        raise ValueError("I-SPY2 BiFlowNet flow time must be finite [B] in [0,1]")
    time = flow_time.reshape(batch, 1, 1, 1, 1, 1)
    return ISPY2BiFlowBatch(
        flow_state=(1.0 - time) * noise + time * endpoint,
        target_velocity=endpoint - noise,
        endpoint=endpoint,
        noise=noise,
        flow_time=flow_time,
    )


@torch.no_grad()
def integrate_ispy2_biflow_euler(
    model: ISPY2BiFlowWorldModel,
    *,
    source_mri: torch.Tensor,
    clinical_text: Sequence[str],
    treatment_text: Sequence[str],
    delta_days: torch.Tensor,
    target_stage: torch.Tensor,
    solver_steps: int,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Integrate caller-provided noise; source latent is not an input."""

    if type(solver_steps) is not int or solver_steps <= 0:
        raise ValueError("I-SPY2 BiFlowNet Euler steps must be positive")
    _validate_latent(noise, "Euler noise")
    state = noise.clone()
    prepared = model.prepare_context(
        source_mri,
        clinical_text=clinical_text,
        treatment_text=treatment_text,
        delta_days=delta_days,
        target_stage=target_stage,
    )
    batch = state.shape[0]
    step_size = 1.0 / solver_steps
    for step in range(solver_steps):
        flow_time = torch.full(
            (batch,),
            step / solver_steps,
            device=state.device,
            dtype=state.dtype,
        )
        velocity = model.velocity_from_context(
            flow_state=state,
            flow_time=flow_time,
            prepared=prepared,
        ).velocity
        state = state + step_size * velocity
    return state


class ISPY2BiFlowTrainingSystem(pl.LightningModule):
    def __init__(
        self,
        model: ISPY2BiFlowWorldModel,
        *,
        config: ISPY2BiFlowConfig,
        checkpoint_identity: dict[str, Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.experiment_config = config
        self.checkpoint_identity = dict(checkpoint_identity)
        self.strict_loading = False

    def _omitted_frozen_text_state_keys(self) -> tuple[str, ...]:
        trainable = {
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        return tuple(
            sorted(
                key for key in self.state_dict()
                if key.startswith(_FROZEN_TEXT_STATE_PREFIX)
                and key not in trainable
            )
        )

    def _step(self, batch: dict[str, Any], prefix: str) -> torch.Tensor:
        flow = build_ispy2_biflow_batch(batch["target_latent"])
        output = self.model(
            source_mri=batch["source_mri"],
            flow_state=flow.flow_state,
            flow_time=flow.flow_time,
            clinical_text=batch["clinical_text"],
            treatment_text=batch["treatment_text"],
            delta_days=batch["delta_days"],
            target_stage=batch["target_stage"],
        )
        loss = F.l1_loss(output.velocity, flow.target_velocity)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("I-SPY2 BiFlowNet velocity loss is non-finite")
        self.log(
            f"{prefix}/velocity_mae",
            loss,
            on_step=prefix == "train",
            on_epoch=True,
            batch_size=flow.endpoint.shape[0],
        )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        self._step(batch, "val")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state = checkpoint.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("I-SPY2 BiFlowNet checkpoint state is missing")
        omitted = self._omitted_frozen_text_state_keys()
        checkpoint["state_dict"] = {
            key: value for key, value in state.items() if key not in omitted
        }
        checkpoint["ispy2_biflow_schema"] = ISPY2_BIFLOW_CHECKPOINT_SCHEMA
        checkpoint["ispy2_biflow_identity"] = self.checkpoint_identity
        checkpoint["ispy2_biflow_run_config"] = self.experiment_config.raw
        checkpoint["ispy2_biflow_state_policy"] = (
            ISPY2_BIFLOW_CHECKPOINT_STATE_POLICY
        )
        checkpoint["ispy2_biflow_omitted_state_keys"] = list(omitted)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        schema = checkpoint.get("ispy2_biflow_schema")
        identity = checkpoint.get("ispy2_biflow_identity")
        current_checkpoint = (
            schema == ISPY2_BIFLOW_CHECKPOINT_SCHEMA
            and identity == self.checkpoint_identity
        )
        legacy_checkpoint = False
        if (
            schema == ISPY2_BIFLOW_LEGACY_CHECKPOINT_SCHEMA
            and isinstance(identity, dict)
            and identity.get("schema") == ISPY2_BIFLOW_LEGACY_IDENTITY_SCHEMA
            and self.experiment_config.base.model.latent_normalization
            == ISPY2_BIFLOW_LEGACY_LATENT_NORMALIZATION
        ):
            upgraded_identity = dict(identity)
            upgraded_identity["schema"] = ISPY2_BIFLOW_CURRENT_IDENTITY_SCHEMA
            legacy_checkpoint = upgraded_identity == self.checkpoint_identity
        if not current_checkpoint and not legacy_checkpoint:
            raise ValueError("I-SPY2 BiFlowNet checkpoint identity mismatch")
        expected_omitted = self._omitted_frozen_text_state_keys()
        state = checkpoint.get("state_dict")
        omitted = checkpoint.get("ispy2_biflow_omitted_state_keys")
        expected_saved = set(self.state_dict()) - set(expected_omitted)
        if (
            checkpoint.get("ispy2_biflow_state_policy")
            != ISPY2_BIFLOW_CHECKPOINT_STATE_POLICY
            or omitted != list(expected_omitted)
            or not isinstance(state, dict)
            or set(state) != expected_saved
        ):
            raise ValueError("I-SPY2 BiFlowNet checkpoint state mismatch")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        text_tower = self.model.conditioner.text_tower
        text = tuple(parameter for parameter in text_tower.parameters() if parameter.requires_grad)
        lora_parameters = getattr(text_tower, "lora_parameters", None)
        if not callable(lora_parameters):
            raise RuntimeError("I-SPY2 text tower does not expose LoRA parameters")
        lora = tuple(parameter for parameter in lora_parameters() if parameter.requires_grad)
        if not lora or {id(value) for value in text} != {id(value) for value in lora}:
            raise RuntimeError("I-SPY2 text parameters must be LoRA-only")
        text_ids = {id(value) for value in text}
        groups: dict[str, list[nn.Parameter]] = defaultdict(list)
        layout = self.experiment_config.training.optimizer_layout
        if layout == "split_backbone_v1":
            dynamics = self.model.dynamics
            backbone = getattr(dynamics, "backbone", None)
            controlnet = getattr(dynamics, "controlnet", None)
            if not isinstance(backbone, nn.Module) or not isinstance(controlnet, nn.Module):
                raise RuntimeError("I-SPY2 split optimizer requires controlled BiFlowNet")
            ownership = {
                id(parameter): "backbone" for parameter in backbone.parameters()
            }
            for parameter in controlnet.parameters():
                identifier = id(parameter)
                if identifier in ownership:
                    raise RuntimeError("I-SPY2 optimizer module ownership overlaps")
                ownership[identifier] = "controlnet"
        elif layout == "legacy_joint_v1":
            ownership = {}
        else:
            raise RuntimeError("I-SPY2 BiFlowNet optimizer layout is unsupported")
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                continue
            identifier = id(parameter)
            if identifier in text_ids:
                groups["text"].append(parameter)
            elif layout == "split_backbone_v1":
                groups[ownership.get(identifier, "conditioner")].append(parameter)
            else:
                groups["dynamics"].append(parameter)
        names = (
            ("backbone", "controlnet", "conditioner", "text")
            if layout == "split_backbone_v1"
            else ("dynamics", "text")
        )
        if any(not groups[name] for name in names):
            raise RuntimeError("I-SPY2 BiFlowNet optimizer group is empty")
        rates = {
            "backbone": self.experiment_config.training.backbone_learning_rate,
            "controlnet": self.experiment_config.training.controlnet_learning_rate,
            "conditioner": self.experiment_config.training.conditioner_learning_rate,
            "dynamics": self.experiment_config.training.dynamics_learning_rate,
            "text": self.experiment_config.training.text_lora_learning_rate,
        }
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": groups[name],
                    "lr": rates[name],
                    "weight_decay": self.experiment_config.training.weight_decay,
                    "name": name,
                }
                for name in names
            ],
            betas=(0.9, 0.999),
        )
        if layout == "legacy_joint_v1":
            return optimizer
        total_steps = int(self.trainer.estimated_stepping_batches)
        if total_steps <= 1:
            raise RuntimeError("I-SPY2 BiFlowNet schedule requires multiple steps")
        warmup_steps = max(
            1, math.ceil(total_steps * self.experiment_config.training.warmup_fraction)
        )
        minimum = self.experiment_config.training.min_learning_rate

        def schedule(peak: float):
            def factor(step: int) -> float:
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                decay_steps = max(1, total_steps - warmup_steps - 1)
                progress = min(1.0, (step - warmup_steps) / decay_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return (minimum + (peak - minimum) * cosine) / peak

            return factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[schedule(rates[name]) for name in names],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_cosine",
            },
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | int | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        if self.experiment_config.training.optimizer_layout == "split_backbone_v1":
            for group in optimizer.param_groups:
                gradients = [
                    parameter.grad
                    for parameter in group["params"]
                    if parameter.grad is not None
                ]
                if not gradients:
                    continue
                norm = torch.nn.utils.get_total_norm(
                    gradients, norm_type=2.0, error_if_nonfinite=True
                )
                self.log(
                    f"train/grad_norm_preclip/{group['name']}",
                    norm,
                    on_step=True,
                    on_epoch=False,
                )
        super().configure_gradient_clipping(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )


__all__ = [
    "ISPY2_BIFLOW_CHECKPOINT_SCHEMA",
    "ISPY2_BIFLOW_LEGACY_CHECKPOINT_SCHEMA",
    "ISPY2_BIFLOW_CHECKPOINT_STATE_POLICY",
    "ISPY2BiFlowBatch",
    "ISPY2BiFlowTrainingSystem",
    "build_ispy2_biflow_batch",
    "integrate_ispy2_biflow_euler",
]
