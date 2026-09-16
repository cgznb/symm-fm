from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.flow.path import SymmetricFlowObjective
from ispy2_symmflow.training.validation import (
    validate_autoencoder,
    validate_symmflow,
    validate_symmflow_endpoints,
)


class TinyAutoencoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, image, *, sample_posterior=True):
        mean = image * self.scale
        posterior_scale = torch.full_like(mean, 0.5)
        return mean, mean, posterior_scale


class TinyConditionEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.schema = type("Schema", (), {"field_names": ("stage_i",)})()

    def forward(self, conditions, *, batch_size):
        return self.bias.expand(batch_size, 1, 2)


class TinyVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, state, tau, tokens):
        return state * self.scale + tokens.mean() * 0.0


def test_autoencoder_validation_uses_deterministic_posterior_mean() -> None:
    model = TinyAutoencoder().train()
    batches = [torch.ones(2, 1, 3, 3, 3), torch.ones(1, 1, 3, 3, 3)]
    first = validate_autoencoder(
        model, batches, device=torch.device("cpu"), kl_weight=1e-3
    )
    second = validate_autoencoder(
        model, batches, device=torch.device("cpu"), kl_weight=1e-3
    )
    assert first == second
    assert first["sample_count"] == 3
    assert first["reconstruction"] == pytest.approx(0.5)
    assert model.training


def test_symmflow_validation_is_fixed_seed_and_restores_modes() -> None:
    velocity = TinyVelocity().train()
    conditions = TinyConditionEncoder().train()
    batch = {
        "later_latent": torch.ones(2, 1, 2, 2, 2),
        "earlier_latent": torch.zeros(2, 1, 2, 2, 2),
        "conditions": {"stage_i": ["T0", "T0"], "ignored": [1, 2]},
    }
    objective = SymmetricFlowObjective()
    first = validate_symmflow(
        velocity,
        conditions,
        objective,
        [batch],
        device=torch.device("cpu"),
        seed=19,
        repeats=2,
    )
    second = validate_symmflow(
        velocity,
        conditions,
        objective,
        [batch],
        device=torch.device("cpu"),
        seed=19,
        repeats=2,
    )
    assert first == second
    assert first["sample_count"] == 2
    assert first["stochastic_repeats"] == 2
    assert velocity.training and conditions.training


def test_endpoint_validation_is_fixed_seed_and_reports_source_copy() -> None:
    velocity = TinyVelocity().train()
    conditions = TinyConditionEncoder().train()
    batch = {
        "later_latent": torch.ones(1, 1, 2, 2, 2),
        "earlier_latent": torch.zeros(1, 1, 2, 2, 2),
        "conditions": {"stage_i": ["T0"]},
    }

    first = validate_symmflow_endpoints(
        velocity,
        conditions,
        [batch],
        device=torch.device("cpu"),
        seed=31,
        samples_per_pair=2,
        steps=2,
    )
    second = validate_symmflow_endpoints(
        velocity,
        conditions,
        [batch],
        device=torch.device("cpu"),
        seed=31,
        samples_per_pair=2,
        steps=2,
    )

    assert first == second
    assert first["endpoint_pair_count"] == 1
    assert first["endpoint_samples_per_pair"] == 2
    assert first["endpoint_nfe_per_sample"] == 4
    assert first["endpoint_source_copy_mse"] == pytest.approx(1.0)
    assert first["endpoint_source_copy_mae"] == pytest.approx(1.0)
    assert velocity.training and conditions.training
