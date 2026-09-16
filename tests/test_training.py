from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.flow.path import SymmetricFlowObjective
from ispy2_symmflow.models.conditioning import (
    CategoricalField,
    ConditionSchema,
    StructuredConditionEncoder,
)
from ispy2_symmflow.training.engine import (
    SymmFlowTrainer,
    autoencoder_objective,
    fit_latent_statistics,
)


class TinyAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.9))

    def forward(self, image, *, sample_posterior=True):
        mean = image * self.weight
        scale = torch.full_like(mean, 0.5)
        return mean, mean, scale

    def encode(self, image):
        return image * 2


class TinyVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, state, tau, tokens):
        return state * self.weight + 0.0 * tokens.mean()


def test_autoencoder_objective_has_reconstruction_and_kl_gradients() -> None:
    model = TinyAutoencoder()
    image = torch.ones(2, 1, 3, 3, 3)
    loss = autoencoder_objective(model, image, kl_weight=1e-3, gradient_weight=0.1)
    loss.total.backward()
    assert loss.reconstruction > 0
    assert loss.kl > 0
    assert model.weight.grad is not None


def test_latent_statistics_are_shared_across_input_batches() -> None:
    model = TinyAutoencoder()
    mean, std, count = fit_latent_statistics(
        model,
        [torch.zeros(1, 1, 2, 2, 2), torch.ones(1, 1, 2, 2, 2)],
        device=torch.device("cpu"),
    )
    assert mean.item() == pytest.approx(1.0)
    assert std.item() == pytest.approx(1.0)
    assert count == 16


def test_symmflow_trainer_does_not_filter_conditions_outside_schema() -> None:
    velocity = TinyVelocity()
    condition_encoder = StructuredConditionEncoder(
        ConditionSchema(
            categorical_fields=(CategoricalField("stage_i", ("T0",)),),
            numeric_fields=(),
            token_dim=2,
        )
    )
    optimizer = torch.optim.AdamW(
        [*velocity.parameters(), *condition_encoder.parameters()], lr=1e-3
    )
    trainer = SymmFlowTrainer(
        velocity,
        condition_encoder,
        optimizer,
        SymmetricFlowObjective(),
        device=torch.device("cpu"),
    )

    with pytest.raises(KeyError, match="not present"):
        trainer.train_batch(
            torch.ones(1, 1, 2, 2, 2),
            torch.zeros(1, 1, 2, 2, 2),
            {"stage_i": ["T0"], "unconfigured": [1]},
        )
