from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.flow.solver import integrate_ode


@pytest.mark.parametrize("method,nfe", [("euler", 8), ("heun", 16)])
def test_constant_field_forward_and_backward(method: str, nfe: int) -> None:
    initial = torch.zeros(2, 4, 2, 2, 2)

    def field(state, tau):
        return torch.full_like(state, 2.5)

    forward = integrate_ode(field, initial, t0=0.0, t1=1.0, steps=8, method=method)
    assert torch.allclose(forward.final_state, torch.full_like(initial, 2.5))
    assert forward.nfe == nfe
    backward = integrate_ode(field, forward.final_state, t0=1.0, t1=0.0, steps=8, method=method)
    assert torch.allclose(backward.final_state, initial, atol=1e-6)


def test_heun_converges_faster_for_linear_field() -> None:
    initial = torch.ones(1, 1, 1)

    def field(state, tau):
        return state

    euler = integrate_ode(field, initial, t0=0, t1=1, steps=8, method="euler")
    heun = integrate_ode(field, initial, t0=0, t1=1, steps=8, method="heun")
    exact = math.e
    assert abs(heun.final_state.item() - exact) < abs(euler.final_state.item() - exact)


def test_trajectory_records_every_joint_update() -> None:
    initial = torch.zeros(1, 2, 1)
    result = integrate_ode(
        lambda state, tau: torch.ones_like(state),
        initial,
        t0=1.0,
        t1=0.0,
        steps=3,
        method="euler",
        return_trajectory=True,
    )
    assert result.trajectory is not None
    assert len(result.trajectory) == 4
    assert result.times[1] - result.times[0] < 0


@pytest.mark.parametrize("endpoint", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_integration_endpoint_is_rejected(endpoint: float) -> None:
    with pytest.raises(ValueError, match="t0 and t1"):
        integrate_ode(
            lambda state, tau: state,
            torch.zeros(1, 1),
            t0=endpoint,
            t1=1.0,
            steps=1,
        )


def test_nonfinite_state_and_vector_field_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite values"):
        integrate_ode(
            lambda state, tau: state,
            torch.full((1, 1), float("nan")),
            t0=0.0,
            t1=1.0,
            steps=1,
        )
    with pytest.raises(FloatingPointError, match="vector field"):
        integrate_ode(
            lambda state, tau: torch.full_like(state, float("inf")),
            torch.zeros(1, 1),
            t0=0.0,
            t1=1.0,
            steps=1,
        )
