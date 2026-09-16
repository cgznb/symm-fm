"""Inference for the deterministic and forward-only CFM comparisons."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ispy2_symmflow.flow.solver import integrate_ode
from ispy2_symmflow.inference.sampler import SampleBatch
from ispy2_symmflow.models.baselines import (
    make_unidirectional_cfm_initial_state,
    validate_cfm_base_distribution,
)


@torch.no_grad()
def sample_deterministic_baseline(
    autoencoder: nn.Module,
    condition_encoder: nn.Module,
    predictor: nn.Module,
    source_image: Tensor,
    conditions: Mapping[str, Any],
) -> SampleBatch:
    """Return the single repeatable forward prediction as a sample batch."""

    autoencoder.eval()
    condition_encoder.eval()
    predictor.eval()
    source_latent = autoencoder.encode(source_image, normalize=True)
    tokens = condition_encoder(conditions, batch_size=source_latent.shape[0])
    target_latent = predictor(source_latent, tokens)
    prediction = autoencoder.decode(target_latent, denormalize=True)
    samples = prediction[None]
    return SampleBatch(
        direction="forward",
        samples=samples,
        mean=prediction,
        std=torch.zeros_like(prediction),
        seeds=(),
        nfe_per_sample=0,
        final_joint_states=None,
    )


class UnidirectionalCFMSampler:
    """Integrate one generated target branch while conditioning on a clean source."""

    def __init__(
        self,
        autoencoder: nn.Module,
        velocity_model: nn.Module,
        condition_encoder: nn.Module,
        *,
        sigma_min: float = 0.0,
        base_distribution: str = "standard_normal",
        noise_scale: float = 1.0,
    ) -> None:
        sigma = float(sigma_min)
        if not math.isfinite(sigma) or not 0.0 <= sigma < 1.0:
            raise ValueError("sigma_min must satisfy 0 <= sigma_min < 1")
        self.autoencoder = autoencoder.eval()
        self.velocity_model = velocity_model.eval()
        self.condition_encoder = condition_encoder.eval()
        self.sigma_min = sigma
        self.base_distribution, self.noise_scale = validate_cfm_base_distribution(
            base_distribution, noise_scale
        )

    @torch.no_grad()
    def sample_forward(
        self,
        source_image: Tensor,
        conditions: Mapping[str, Any],
        *,
        num_samples: int = 8,
        seed: int = 0,
        steps: int = 25,
        solver: str = "heun",
        allow_residual_endpoint: bool = False,
    ) -> SampleBatch:
        source_latent = self.autoencoder.encode(source_image, normalize=True)
        tokens = self.condition_encoder(conditions, batch_size=source_latent.shape[0])
        return self.sample_forward_latent(
            source_latent,
            tokens,
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            solver=solver,
            allow_residual_endpoint=allow_residual_endpoint,
        )

    @torch.no_grad()
    def sample_forward_latent(
        self,
        source_latent: Tensor,
        condition_tokens: Tensor,
        *,
        num_samples: int = 8,
        seed: int = 0,
        steps: int = 25,
        solver: str = "heun",
        return_joint_state: bool = False,
        allow_residual_endpoint: bool = False,
    ) -> SampleBatch:
        """Generate from a normalized cached source without re-encoding its MRI."""

        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if self.sigma_min and not allow_residual_endpoint:
            raise ValueError(
                "sigma_min > 0 has a residual target endpoint; explicitly allow it "
                "only for a compatibility comparison"
            )
        if source_latent.ndim != 5 or any(size < 1 for size in source_latent.shape):
            raise ValueError(
                "normalized source latent must have non-empty shape [B,C,D,H,W]"
            )
        if not source_latent.is_floating_point():
            raise TypeError("normalized source latent must use a floating dtype")
        if not bool(torch.isfinite(source_latent).all()):
            raise ValueError("normalized source latent must contain only finite values")
        if condition_tokens.ndim != 3 or any(
            size < 1 for size in condition_tokens.shape
        ):
            raise ValueError(
                "condition tokens must have non-empty shape [B,L,condition_dim]"
            )
        if not condition_tokens.is_floating_point():
            raise TypeError("condition tokens must use a floating dtype")
        if not bool(torch.isfinite(condition_tokens).all()):
            raise ValueError("condition tokens must contain only finite values")
        if condition_tokens.shape[0] != source_latent.shape[0]:
            raise ValueError("source and condition batch sizes must match")
        if condition_tokens.device != source_latent.device:
            raise ValueError("source latent and condition tokens must share a device")

        def field(target_state: Tensor, tau: Tensor) -> Tensor:
            return self.velocity_model(
                target_state, source_latent, tau, condition_tokens
            )

        outputs: list[Tensor] = []
        final_states: list[Tensor] = []
        seeds: list[int] = []
        nfe = 0
        for index in range(num_samples):
            current_seed = int(seed) + index
            generator = torch.Generator(device=source_latent.device)
            generator.manual_seed(current_seed)
            noise = torch.randn(
                source_latent.shape,
                dtype=source_latent.dtype,
                device=source_latent.device,
                generator=generator,
            )
            initial = make_unidirectional_cfm_initial_state(
                source_latent,
                noise,
                base_distribution=self.base_distribution,
                noise_scale=self.noise_scale,
            )
            solution = integrate_ode(
                field,
                initial,
                t0=0.0,
                t1=1.0,
                steps=steps,
                method=solver,
            )
            outputs.append(
                self.autoencoder.decode(solution.final_state, denormalize=True)
            )
            if return_joint_state:
                final_states.append(solution.final_state)
            seeds.append(current_seed)
            nfe = solution.nfe
        samples = torch.stack(outputs)
        return SampleBatch(
            direction="forward",
            samples=samples,
            mean=samples.mean(0),
            std=samples.std(0, unbiased=False),
            seeds=tuple(seeds),
            nfe_per_sample=nfe,
            final_joint_states=(
                torch.stack(final_states) if return_joint_state else None
            ),
        )


__all__ = ["UnidirectionalCFMSampler", "sample_deterministic_baseline"]
