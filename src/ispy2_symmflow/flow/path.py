"""The two-branch interpolation and training objective from SymmFlow."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _validate_sigma(sigma_min: float) -> float:
    sigma = float(sigma_min)
    if not math.isfinite(sigma) or not 0.0 <= sigma < 1.0:
        raise ValueError("sigma_min must satisfy 0 <= sigma_min < 1")
    return sigma


def _batch_time(tau: Tensor | float, reference: Tensor) -> Tensor:
    value = torch.as_tensor(tau, dtype=reference.dtype, device=reference.device)
    if value.ndim == 0:
        value = value.repeat(reference.shape[0])
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 1 or value.shape[0] != reference.shape[0]:
        raise ValueError(
            f"tau must be scalar or [B], received {tuple(value.shape)} for B={reference.shape[0]}"
        )
    if not torch.is_floating_point(value):
        value = value.to(reference.dtype)
    return value


def _broadcast_time(tau: Tensor, reference: Tensor) -> Tensor:
    return tau.reshape(tau.shape[0], *([1] * (reference.ndim - 1)))


def _validate_latents(*values: Tensor) -> None:
    if any(not value.is_floating_point() for value in values):
        raise TypeError("latents and path noise must use floating-point dtypes")
    if any(value.numel() == 0 for value in values):
        raise ValueError("latents and path noise cannot be empty")
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError("latents and path noise must contain only finite values")


@dataclass(frozen=True)
class PathSample:
    """A joint noised state and its two analytic velocity targets."""

    joint_state: Tensor
    x_tau: Tensor
    y_tau: Tensor
    target_x: Tensor
    target_y: Tensor
    epsilon_x: Tensor
    epsilon_y: Tensor
    tau: Tensor
    sigma_min: float


def path_velocity(
    later: Tensor,
    earlier: Tensor,
    epsilon_x: Tensor,
    epsilon_y: Tensor,
    *,
    sigma_min: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Return the constant derivatives ``dx_tau/dtau`` and ``dy_tau/dtau``."""

    sigma = _validate_sigma(sigma_min)
    if not (later.shape == earlier.shape == epsilon_x.shape == epsilon_y.shape):
        raise ValueError("later, earlier, epsilon_x, and epsilon_y must have identical shapes")
    _validate_latents(later, earlier, epsilon_x, epsilon_y)
    target_x = later - (1.0 - sigma) * epsilon_x
    target_y = epsilon_y - (1.0 - sigma) * earlier
    return target_x, target_y


def make_path(
    later: Tensor,
    earlier: Tensor,
    tau: Tensor | float,
    *,
    epsilon_x: Tensor | None = None,
    epsilon_y: Tensor | None = None,
    sigma_min: float = 0.0,
    generator: torch.Generator | None = None,
) -> PathSample:
    """Construct the joint SymmFlow state.

    The channel convention is invariant: x/later is first and y/earlier is
    second.  Noise is independent across branches unless supplied explicitly.
    """

    sigma = _validate_sigma(sigma_min)
    if later.shape != earlier.shape:
        raise ValueError("paired later and earlier latents must have identical shapes")
    if later.ndim < 3:
        raise ValueError("latents must have batch, channel, and spatial dimensions")
    _validate_latents(later, earlier)
    tau_batch = _batch_time(tau, later)
    if not torch.isfinite(tau_batch).all() or torch.any((tau_batch < 0) | (tau_batch > 1)):
        raise ValueError("tau values must be finite and lie in [0, 1]")
    if epsilon_x is None:
        epsilon_x = torch.randn(
            later.shape, dtype=later.dtype, device=later.device, generator=generator
        )
    if epsilon_y is None:
        epsilon_y = torch.randn(
            earlier.shape, dtype=earlier.dtype, device=earlier.device, generator=generator
        )
    target_x, target_y = path_velocity(
        later, earlier, epsilon_x, epsilon_y, sigma_min=sigma
    )
    t = _broadcast_time(tau_batch, later)
    attenuation = 1.0 - (1.0 - sigma) * t
    x_tau = attenuation * epsilon_x + t * later
    y_tau = attenuation * earlier + t * epsilon_y
    return PathSample(
        joint_state=torch.cat((x_tau, y_tau), dim=1),
        x_tau=x_tau,
        y_tau=y_tau,
        target_x=target_x,
        target_y=target_y,
        epsilon_x=epsilon_x,
        epsilon_y=epsilon_y,
        tau=tau_batch,
        sigma_min=sigma,
    )


class JointVelocity(Protocol):
    def __call__(self, joint_state: Tensor, tau: Tensor, condition_tokens: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class FlowLoss:
    total: Tensor
    x: Tensor
    y: Tensor


class SymmetricFlowObjective(nn.Module):
    """One-forward, two-branch SymmFlow velocity objective."""

    def __init__(
        self,
        sigma_min: float = 0.0,
        loss_weight_x: float = 1.0,
        loss_weight_y: float = 1.0,
    ) -> None:
        super().__init__()
        self.sigma_min = _validate_sigma(sigma_min)
        self.loss_weight_x = float(loss_weight_x)
        self.loss_weight_y = float(loss_weight_y)
        weights = (self.loss_weight_x, self.loss_weight_y)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("branch loss weights must be finite and non-negative")
        if not any(weight > 0 for weight in weights):
            raise ValueError("at least one branch loss weight must be positive")

    def forward(
        self,
        model: JointVelocity,
        later: Tensor,
        earlier: Tensor,
        condition_tokens: Tensor,
        *,
        tau: Tensor | None = None,
        epsilon_x: Tensor | None = None,
        epsilon_y: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> FlowLoss:
        if tau is None:
            tau = torch.rand(
                later.shape[0], dtype=later.dtype, device=later.device, generator=generator
            )
        path = make_path(
            later,
            earlier,
            tau,
            epsilon_x=epsilon_x,
            epsilon_y=epsilon_y,
            sigma_min=self.sigma_min,
            generator=generator,
        )
        prediction = model(path.joint_state, path.tau, condition_tokens)
        if prediction.shape != path.joint_state.shape:
            raise ValueError(
                "velocity output must match the joint state shape; "
                f"got {tuple(prediction.shape)} and {tuple(path.joint_state.shape)}"
            )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("velocity model returned non-finite values")
        channels = later.shape[1]
        predicted_x, predicted_y = prediction[:, :channels], prediction[:, channels:]
        loss_x = F.mse_loss(predicted_x, path.target_x, reduction="mean")
        loss_y = F.mse_loss(predicted_y, path.target_y, reduction="mean")
        total = self.loss_weight_x * loss_x + self.loss_weight_y * loss_y
        return FlowLoss(total=total, x=loss_x, y=loss_y)
