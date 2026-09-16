"""Small, auditable fixed-step ODE solvers for joint-state sampling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import Tensor


VectorField = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class ODESolution:
    final_state: Tensor
    nfe: int
    times: Tensor
    trajectory: tuple[Tensor, ...] | None = None


def integrate_ode(
    vector_field: VectorField,
    initial_state: Tensor,
    *,
    t0: float,
    t1: float,
    steps: int,
    method: str = "heun",
    return_trajectory: bool = False,
) -> ODESolution:
    """Integrate a joint tensor without clamping either branch.

    ``steps`` is the number of state updates. Euler uses one network function
    evaluation (NFE) per update and Heun uses two.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    start, end = float(t0), float(t1)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (start, end)):
        raise ValueError("t0 and t1 must be finite and lie in [0, 1]")
    if initial_state.ndim < 2 or any(size < 1 for size in initial_state.shape):
        raise ValueError("ODE state must have non-empty batch and feature dimensions")
    if not torch.is_floating_point(initial_state):
        raise TypeError("ODE state must use a floating dtype")
    if not torch.isfinite(initial_state).all():
        raise ValueError("ODE state must contain only finite values")
    method_normalized = method.lower()
    if method_normalized not in {"euler", "heun"}:
        raise ValueError(f"unsupported solver {method!r}; choose 'euler' or 'heun'")
    times = torch.linspace(
        start,
        end,
        steps + 1,
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    state = initial_state
    trajectory: list[Tensor] | None = [state.clone()] if return_trajectory else None
    nfe = 0
    batch = initial_state.shape[0]
    for index in range(steps):
        current = times[index]
        following = times[index + 1]
        step_size = following - current
        tau = current.expand(batch)
        slope = vector_field(state, tau)
        nfe += 1
        if slope.shape != state.shape:
            raise ValueError("vector field output shape must match its state input")
        if not torch.isfinite(slope).all():
            raise FloatingPointError("vector field returned non-finite values")
        if method_normalized == "euler":
            state = state + step_size * slope
        else:
            predicted = state + step_size * slope
            next_slope = vector_field(predicted, following.expand(batch))
            nfe += 1
            if next_slope.shape != state.shape:
                raise ValueError("vector field output shape must match its state input")
            if not torch.isfinite(next_slope).all():
                raise FloatingPointError("vector field returned non-finite values")
            state = state + 0.5 * step_size * (slope + next_slope)
        if not torch.isfinite(state).all():
            raise FloatingPointError("ODE integration produced a non-finite state")
        if trajectory is not None:
            trajectory.append(state.clone())
    return ODESolution(
        final_state=state,
        nfe=nfe,
        times=times,
        trajectory=tuple(trajectory) if trajectory is not None else None,
    )
