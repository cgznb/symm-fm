"""Joint pre, first-post and late MRI prediction with the original SymmFlow."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn


PHASES = ("pre_aqc0", "first_post_aqc1", "metadata_late")
IMAGE_SHAPE = (96, 256, 256)
PHASE_LATENT_SHAPE = (8, 24, 64, 64)
JOINT_LATENT_SHAPE = (24, 24, 64, 64)


def import_symmflow(source):
    source = str(Path(source).resolve() / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def join_phase_latents(phases):
    if len(phases) != len(PHASES):
        raise ValueError("Exactly three ordered phase latents are required")
    first = phases[0]
    if first.ndim != 5 or first.shape[1] != PHASE_LATENT_SHAPE[0]:
        raise ValueError("Each phase latent must have shape [B,8,D,H,W]")
    if any(value.shape != first.shape or value.device != first.device or value.dtype != first.dtype
           for value in phases):
        raise ValueError("Phase latents must share shape, device and dtype")
    return torch.cat(tuple(phases), dim=1)


def split_phase_latents(joint):
    if joint.ndim != 5 or joint.shape[1] != JOINT_LATENT_SHAPE[0]:
        raise ValueError("Joint phase latent must have shape [B,24,D,H,W]")
    return joint.split(PHASE_LATENT_SHAPE[0], dim=1)


class SharedThreePhaseCodec(nn.Module):
    """Use one frozen single-channel codec for all three ordered phases."""

    def __init__(self, codec):
        super().__init__()
        self.codec = codec.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, images):
        if images.ndim != 5 or images.shape[1] != len(PHASES):
            raise ValueError("Three-phase MRI must have shape [B,3,D,H,W]")
        return join_phase_latents([
            self.codec.encode_continuous(images[:, index:index + 1])
            for index in range(len(PHASES))
        ])

    @torch.no_grad()
    def decode(self, joint):
        decoded = []
        with torch.autocast(joint.device.type, enabled=False):
            for latent in split_phase_latents(joint):
                quantized, _ = self.codec.quantizer(latent.float())
                decoded.append(self.codec.decode(quantized.float()))
        return torch.cat(decoded, dim=1)


class ThreePhaseSymmFlow(nn.Module):
    def __init__(self, records, *, source, backbone=None):
        super().__init__()
        import_symmflow(source)
        from ispy2_symmflow.config import load_config
        from ispy2_symmflow.flow.path import SymmetricFlowObjective
        from ispy2_symmflow.models.conditioning import StructuredConditionEncoder
        from ispy2_symmflow.models.velocity import build_velocity_model_from_config
        from ispy2_symmflow.training.schema import fit_condition_schema

        config = load_config(Path(source) / "configs/mewm_all_pairs_5090.yaml", resolve_assets=False)
        velocity = copy.deepcopy(config["velocity"])
        velocity["latent_channels"] = JOINT_LATENT_SHAPE[0]
        schema, provenance = fit_condition_schema(records, config["conditions"])
        self.condition_encoder = StructuredConditionEncoder(schema)
        self.velocity_model = build_velocity_model_from_config(velocity, backbone=backbone)
        self.objective = SymmetricFlowObjective(sigma_min=0.0, loss_weight_x=1.0, loss_weight_y=1.0)
        self.description = {
            "phase_order": list(PHASES),
            "image_shape_czyx": [len(PHASES), *IMAGE_SHAPE],
            "latent_shape_czyx": list(JOINT_LATENT_SHAPE),
            "velocity": velocity,
            "condition_schema": schema.to_dict(),
            "condition_fit_split": provenance["fit_split"],
            "unavailable_conditions": provenance["unavailable_fields"],
            "sigma_min": 0.0,
        }

    def tokens(self, records):
        from ispy2_symmflow.training.datasets import pair_conditions
        values = [pair_conditions(record) for record in records]
        fields = self.condition_encoder.schema.field_names
        batched = {key: [row.get(key) for row in values] for key in fields}
        return self.condition_encoder(batched, batch_size=len(records))

    def loss(self, batch):
        source, target = batch["source"], batch["target"]
        split_phase_latents(source)
        split_phase_latents(target)
        result = self.objective(self.velocity_model, target, source, self.tokens(batch["records"]))
        return result.total, {"x": float(result.x.detach()), "y": float(result.y.detach())}

    def sample(self, source, records, noise, steps):
        """Prediction accepts source data and conditions only."""
        from ispy2_symmflow.flow.solver import integrate_ode
        split_phase_latents(source)
        if noise.shape != source.shape or noise.device != source.device or noise.dtype != source.dtype:
            raise ValueError("Sampling noise must match the three-phase source latent")
        tokens = self.tokens(records)
        joint = torch.cat((noise, source), dim=1)
        result = integrate_ode(
            lambda state, tau: self.velocity_model(state, tau, tokens),
            joint, t0=0.0, t1=1.0, steps=steps, method="euler",
        )
        return result.final_state[:, :JOINT_LATENT_SHAPE[0]]
