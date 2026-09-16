"""Original SymmFlow networks/objective on registered 32x128x128 DCE0 crops."""

from __future__ import annotations

import torch
from torch import nn

from ispy2_symmflow.flow.path import SymmetricFlowObjective
from ispy2_symmflow.flow.solver import integrate_ode
from ispy2_symmflow.models.conditioning import StructuredConditionEncoder
from ispy2_symmflow.models.velocity import build_velocity_model_from_config
from ispy2_symmflow.training.datasets import pair_conditions
from ispy2_symmflow.training.schema import fit_condition_schema


def collate_pairs(items):
    records = [item["record"] for item in items]
    values = [pair_conditions(record) for record in records]
    return {"later_latent": torch.stack([item["later_latent"] for item in items]),
            "earlier_latent": torch.stack([item["earlier_latent"] for item in items]),
            "records": records,
            "conditions": {key: [value.get(key) for value in values] for key in values[0]}}


class RegisteredROI32Flow(nn.Module):
    def __init__(self, config, records):
        super().__init__()
        schema, provenance = fit_condition_schema(records, config["conditions"])
        self.condition_encoder = StructuredConditionEncoder(schema)
        self.velocity_model = build_velocity_model_from_config(config["velocity"])
        self.objective = SymmetricFlowObjective(sigma_min=0.0, loss_weight_x=1.0, loss_weight_y=1.0)
        self.description = {"data_interface": "registered_dce0_roi32_firstpostmask_v1",
                            "condition_schema": schema.to_dict(), "fit_split": provenance["fit_split"],
                            "unavailable_conditions": provenance["unavailable_fields"],
                            "velocity_configuration": config["velocity"], "latent_shape_czyx": [8, 8, 32, 32],
                            "joint_branch_order": ["later", "earlier"]}

    def loss(self, batch):
        tokens = self.condition_encoder(batch["conditions"], batch_size=len(batch["later_latent"]))
        return self.objective(self.velocity_model, batch["later_latent"], batch["earlier_latent"], tokens)

    @torch.no_grad()
    def sample(self, source, conditions, noise, *, steps=25, solver="heun"):
        if source.shape != noise.shape or tuple(source.shape[1:]) != (8, 8, 32, 32):
            raise ValueError("ROI32 source/noise must share [B,8,8,32,32]")
        tokens = self.condition_encoder(conditions, batch_size=len(source))
        joint = torch.cat((noise, source), dim=1)
        solution = integrate_ode(lambda state, tau: self.velocity_model(state, tau, tokens),
                                 joint, t0=0.0, t1=1.0, steps=steps, method=solver)
        return solution.final_state[:, :8].float()
