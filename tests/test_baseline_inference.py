from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.inference.baselines import (
    UnidirectionalCFMSampler,
    sample_deterministic_baseline,
)


class IdentityAutoencoder(torch.nn.Module):
    def encode(self, image, *, normalize=True):
        return image

    def decode(self, latent, *, denormalize=True):
        return latent


class ConditionEncoder(torch.nn.Module):
    def forward(self, conditions, *, batch_size):
        value = float(conditions.get("effect", 0.0))
        return torch.full((batch_size, 1, 1), value)


class DeterministicPredictor(torch.nn.Module):
    def forward(self, source, tokens):
        return source + tokens.mean(dim=(1, 2)).reshape(-1, 1, 1, 1, 1)


class ConstantOneWayVelocity(torch.nn.Module):
    def forward(self, state, source, tau, tokens):
        return source + 0.0 * state + 0.0 * tokens.mean()


def test_deterministic_baseline_returns_one_repeatable_candidate() -> None:
    source = torch.ones(1, 1, 2, 2, 2)
    result = sample_deterministic_baseline(
        IdentityAutoencoder(),
        ConditionEncoder(),
        DeterministicPredictor(),
        source,
        {"effect": 2.0},
    )
    assert result.samples.shape == (1, 1, 1, 2, 2, 2)
    torch.testing.assert_close(result.samples[0], torch.full_like(source, 3.0))
    assert result.nfe_per_sample == 0 and result.seeds == ()


@pytest.mark.parametrize("solver,nfe", [("euler", 3), ("heun", 6)])
def test_one_way_cfm_sampler_has_correct_nfe_and_seed_replay(solver: str, nfe: int) -> None:
    sampler = UnidirectionalCFMSampler(
        IdentityAutoencoder(), ConstantOneWayVelocity(), ConditionEncoder()
    )
    source = torch.full((1, 1, 2, 2, 2), 0.25)
    first = sampler.sample_forward(
        source, {"effect": 0.0}, num_samples=2, seed=7, steps=3, solver=solver
    )
    repeated = sampler.sample_forward(
        source, {"effect": 0.0}, num_samples=2, seed=7, steps=3, solver=solver
    )
    changed = sampler.sample_forward(
        source, {"effect": 0.0}, num_samples=2, seed=8, steps=3, solver=solver
    )
    assert first.nfe_per_sample == nfe
    assert torch.equal(first.samples, repeated.samples)
    assert not torch.equal(first.samples[0], changed.samples[0])
