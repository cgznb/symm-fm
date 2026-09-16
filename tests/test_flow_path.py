from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.flow.path import SymmetricFlowObjective, make_path


@pytest.mark.parametrize("sigma", [0.0, 0.07])
def test_path_endpoints_and_derivatives(sigma: float) -> None:
    later = torch.tensor([[[2.0, 4.0]]], requires_grad=True)
    earlier = torch.tensor([[[1.0, 3.0]]], requires_grad=True)
    eps_x = torch.tensor([[[0.5, -1.0]]])
    eps_y = torch.tensor([[[-0.5, 2.0]]])

    at_zero = make_path(later, earlier, 0.0, epsilon_x=eps_x, epsilon_y=eps_y, sigma_min=sigma)
    at_one = make_path(later, earlier, 1.0, epsilon_x=eps_x, epsilon_y=eps_y, sigma_min=sigma)
    assert torch.equal(at_zero.x_tau, eps_x)
    assert torch.equal(at_zero.y_tau, earlier)
    assert torch.allclose(at_one.x_tau, later + sigma * eps_x)
    assert torch.allclose(at_one.y_tau, eps_y + sigma * earlier)

    tau = torch.tensor([0.37], requires_grad=True)
    middle = make_path(later, earlier, tau, epsilon_x=eps_x, epsilon_y=eps_y, sigma_min=sigma)
    dx = torch.autograd.grad(middle.x_tau.sum(), tau, retain_graph=True)[0]
    dy = torch.autograd.grad(middle.y_tau.sum(), tau)[0]
    assert torch.allclose(dx, middle.target_x.sum().reshape_as(dx))
    assert torch.allclose(dy, middle.target_y.sum().reshape_as(dy))


def test_joint_order_and_both_branch_gradients() -> None:
    class Scale(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale_x = torch.nn.Parameter(torch.tensor(0.2))
            self.scale_y = torch.nn.Parameter(torch.tensor(0.3))

        def forward(self, state, tau, conditions):
            x, y = state.chunk(2, dim=1)
            return torch.cat((self.scale_x * x, self.scale_y * y), dim=1)

    later = torch.ones(2, 1, 2, 2, 2)
    earlier = torch.full_like(later, 2.0)
    eps_x = torch.zeros_like(later)
    eps_y = torch.full_like(later, 3.0)
    model = Scale()
    loss = SymmetricFlowObjective()(model, later, earlier, torch.zeros(2, 1, 4), tau=torch.tensor([0.25, 0.75]), epsilon_x=eps_x, epsilon_y=eps_y)
    loss.total.backward()
    assert model.scale_x.grad is not None and model.scale_x.grad.abs() > 0
    assert model.scale_y.grad is not None and model.scale_y.grad.abs() > 0

    state = make_path(later, earlier, 0.0, epsilon_x=eps_x, epsilon_y=eps_y).joint_state
    assert torch.equal(state[:, :1], eps_x)
    assert torch.equal(state[:, 1:], earlier)


def test_tau_is_not_integer_truncated() -> None:
    later = torch.ones(1, 1, 1)
    earlier = torch.zeros_like(later)
    eps_x = torch.zeros_like(later)
    eps_y = torch.ones_like(later)
    result = make_path(later, earlier, torch.tensor([0.375]), epsilon_x=eps_x, epsilon_y=eps_y)
    assert result.tau.dtype.is_floating_point
    assert torch.allclose(result.x_tau, torch.full_like(later, 0.375))


@pytest.mark.parametrize("tau", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_tau_is_rejected(tau: float) -> None:
    latent = torch.zeros(1, 1, 1)
    with pytest.raises(ValueError, match=r"finite.*\[0, 1\]"):
        make_path(latent, latent, tau)


@pytest.mark.parametrize(
    "weights", [(float("nan"), 1.0), (float("inf"), 1.0), (0.0, 0.0)]
)
def test_invalid_branch_weights_are_rejected(weights: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="weight"):
        SymmetricFlowObjective(loss_weight_x=weights[0], loss_weight_y=weights[1])


def test_nonfloating_or_nonfinite_path_values_are_rejected() -> None:
    with pytest.raises(TypeError, match="floating"):
        make_path(torch.ones(1, 1, 1, dtype=torch.long), torch.ones(1, 1, 1, dtype=torch.long), 0.5)
    latent = torch.ones(1, 1, 1)
    with pytest.raises(ValueError, match="finite"):
        make_path(latent, torch.full_like(latent, float("nan")), 0.5)
