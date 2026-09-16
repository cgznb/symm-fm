"""Original SymmFlow modules bound to the shared first-post comparison data."""

from research_release import path as _release_path, load_yaml as _release_yaml

from pathlib import Path

import torch
import yaml
from torch import nn

from ispy2_symmflow.flow.path import SymmetricFlowObjective
from ispy2_symmflow.flow.solver import integrate_ode
from ispy2_symmflow.models.conditioning import StructuredConditionEncoder
from ispy2_symmflow.models.velocity import build_velocity_model_from_config
from ispy2_symmflow.training.datasets import pair_conditions
from ispy2_symmflow.training.schema import fit_condition_schema


class FirstPostSymmFlow(nn.Module):
    def __init__(self, records):
        super().__init__()
        root = Path(__file__).resolve().parents[3]
        original = _release_yaml((root / 'configs/mewm_all_pairs_5090.yaml').read_text(), resolve_assets=False)
        schema, provenance = fit_condition_schema(records, original['conditions'])
        self.condition_encoder = StructuredConditionEncoder(schema)
        self.velocity_model = build_velocity_model_from_config(original['velocity'])
        self.objective = SymmetricFlowObjective(sigma_min=0.0, loss_weight_x=1.0, loss_weight_y=1.0)
        self.description = {'data_interface': 'first_post_unregistered_tumor_roi_v1',
                            'velocity': original['velocity'], 'condition_schema': schema.to_dict(),
                            'schema_fit_split': provenance['fit_split'],
                            'unavailable_conditions': provenance['unavailable_fields']}

    def tokens(self, records):
        values = [pair_conditions(record) for record in records]
        batched = {key: [v.get(key) for v in values] for key in self.condition_encoder.schema.field_names}
        return self.condition_encoder(batched, batch_size=len(records))

    def loss(self, batch):
        output = self.objective(self.velocity_model, batch['target'], batch['source'], self.tokens(batch['records']))
        return output.total, {'x': float(output.x.detach()), 'y': float(output.y.detach())}

    def sample(self, batch, noise, steps):
        tokens = self.tokens(batch['records'])
        joint = torch.cat((noise, batch['source']), dim=1)
        result = integrate_ode(lambda state, tau: self.velocity_model(state, tau, tokens),
                               joint, t0=0.0, t1=1.0, steps=steps, method='euler')
        return result.final_state[:, :8]
