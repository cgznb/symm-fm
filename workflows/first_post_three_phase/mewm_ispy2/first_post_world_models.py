from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .ispy2_biflow_backbone import enable_local_dit_fp32
from .ispy2_biflow_training import build_ispy2_biflow_batch, integrate_ispy2_biflow_euler
from .ispy2_biflow_world_model import build_ispy2_biflow_world_model
from .mu_glioma_world_model import MUMedicalTextTower


class FirstPostBiFM(nn.Module):
    def __init__(self):
        super().__init__()
        tower = MUMedicalTextTower.from_pretrained(local_files_only=True)
        self.network = build_ispy2_biflow_world_model(text_tower=tower)
        enable_local_dit_fp32(self.network.dynamics, subvolume_batch=16)
        self.description = {'data_interface': 'first_post_unregistered_tumor_roi_v1',
                            'source_modalities': ['first_post', 'ser'],
                            'architecture': self.network.architecture_contract,
                            'initial_distribution': 'standard_normal',
                            'numeric_policy': 'local_dit_fp32_v1', 'loss': 'l1'}

    def conditions(self, batch):
        records = batch['records']
        device = batch['target'].device
        return {'clinical_text': [r['clinical_text'] for r in records],
                'treatment_text': [r['treatment_text'] for r in records],
                'delta_days': torch.tensor([r['delta_days'] for r in records], device=device, dtype=torch.float32),
                'target_stage': torch.tensor([int(r['later_stage'][1:]) for r in records], device=device)}

    def loss(self, batch):
        path = build_ispy2_biflow_batch(batch['target'].unsqueeze(1))
        result = self.network(source_mri=batch['source_mri'], flow_state=path.flow_state,
                              flow_time=path.flow_time, **self.conditions(batch))
        loss = F.l1_loss(result.velocity.float(), path.target_velocity.float())
        return loss, {}

    def sample(self, batch, noise, steps):
        return integrate_ispy2_biflow_euler(self.network, source_mri=batch['source_mri'],
                    noise=noise.unsqueeze(1), solver_steps=steps, **self.conditions(batch)).squeeze(1)


def build_model(cfg, arm, records, device='cuda:0'):
    source = str(Path(cfg['symm_repo']) / 'src')
    if source not in sys.path:
        sys.path.insert(0, source)
    if arm == 'bifm':
        model = FirstPostBiFM()
    elif arm == 'symm':
        from ispy2_symmflow.training.first_post import FirstPostSymmFlow
        model = FirstPostSymmFlow(records)
    else:
        raise ValueError('Unknown comparison arm')
    return model.to(device)


def compact_state(model):
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    prefix = 'network.conditioner.text_tower.model.'
    return {key: value.detach().cpu() for key, value in model.state_dict().items()
            if not key.startswith(prefix) or key in trainable}


def restore_model(model, state):
    result = model.load_state_dict(state, strict=False)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    prefix = 'network.conditioner.text_tower.model.'
    bad = [key for key in result.missing_keys if not key.startswith(prefix) or key in trainable]
    if bad or result.unexpected_keys:
        raise ValueError('World-model checkpoint is incomplete or incompatible')


def optimizer_for(model, arm, max_steps):
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if arm == 'bifm':
        text = [p for name, p in trainable if name.startswith('network.conditioner.text_tower.')]
        dynamics = [p for name, p in trainable if not name.startswith('network.conditioner.text_tower.')]
        optimizer = torch.optim.AdamW([{'params': dynamics, 'name': 'dynamics'},
                                      {'params': text, 'name': 'text_lora'}], lr=1e-4, weight_decay=0.05)
        return optimizer, None, None
    from ispy2_symmflow.training.engine import build_warmup_cosine_scheduler
    from ispy2_symmflow.training.ema import ExponentialMovingAverage
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4, weight_decay=0.01)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=1000, total_steps=max_steps)
    return optimizer, scheduler, ExponentialMovingAverage(model, decay=0.9999)
