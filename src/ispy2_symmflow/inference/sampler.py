"""Forward prediction and retrospective reconstruction with one joint ODE."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from ispy2_symmflow.flow.solver import ODESolution, integrate_ode
from ispy2_symmflow.utils.hashing import sha256_file


class Autoencoder(Protocol):
    def encode(self, image: Tensor, *, normalize: bool = True) -> Tensor: ...

    def decode(self, latent: Tensor, *, denormalize: bool = True) -> Tensor: ...


@dataclass(frozen=True)
class SampleBatch:
    direction: str
    samples: Tensor
    mean: Tensor
    std: Tensor
    seeds: tuple[int, ...]
    nfe_per_sample: int
    final_joint_states: Tensor | None


def validate_sample_array_binding(
    prediction: str | Path, record: Mapping[str, Any]
) -> str:
    """Verify that a sampling sidecar still names and hashes its NPZ payload."""

    path = Path(prediction).expanduser().resolve()
    if str(record.get("array_file", "")) != path.name:
        raise ValueError(f"sampling sidecar array_file does not name {path.name}")
    expected = str(record.get("array_sha256", "")).strip().lower()
    if not expected:
        raise ValueError("sampling sidecar has no prediction array SHA-256")
    actual = sha256_file(path)
    if expected != actual:
        raise ValueError("prediction array SHA-256 differs from the sampling sidecar")
    return actual


def validate_source_archive_binding(
    source: str | Path, metadata: Mapping[str, Any]
) -> str:
    """Verify that a sampling sidecar still binds the source archive bytes."""

    expected = str(metadata.get("source_sha256", "")).strip().lower()
    if not expected:
        raise ValueError("sampling sidecar has no source archive SHA-256")
    actual = sha256_file(Path(source).expanduser().resolve())
    if expected != actual:
        raise ValueError("source archive SHA-256 differs from the sampling sidecar")
    return actual


class SymmFlowSampler:
    """Numerically integrate both branches; neither source branch is clamped."""

    def __init__(
        self,
        autoencoder: nn.Module,
        velocity_model: nn.Module,
        *,
        sigma_min: float = 0.0,
    ) -> None:
        if not math.isfinite(float(sigma_min)) or not 0 <= float(sigma_min) < 1:
            raise ValueError("sigma_min must satisfy 0 <= sigma_min < 1")
        self.autoencoder = autoencoder
        self.velocity_model = velocity_model
        self.sigma_min = float(sigma_min)
        # A sampler owns inference-mode views of these modules. This prevents
        # dropout/batch-stat updates from violating fixed-input/seed replay.
        self.autoencoder.eval()
        self.velocity_model.eval()

    def _require_clean_endpoint_path(self, allow_residual_endpoint: bool) -> None:
        if self.sigma_min and not allow_residual_endpoint:
            raise ValueError(
                "sigma_min > 0 has residual endpoint terms; pass "
                "allow_residual_endpoint=True only for explicit compatibility experiments"
            )

    def _field(self, condition_tokens: Tensor):
        def velocity(joint_state: Tensor, tau: Tensor) -> Tensor:
            return self.velocity_model(joint_state, tau, condition_tokens)

        return velocity

    def sample_joint(
        self,
        initial_joint_state: Tensor,
        condition_tokens: Tensor,
        *,
        t0: float,
        t1: float,
        steps: int,
        solver: str = "heun",
        return_trajectory: bool = False,
    ) -> ODESolution:
        """Integrate a supplied complete joint state in either direction."""

        if initial_joint_state.shape[0] != condition_tokens.shape[0]:
            raise ValueError("condition and joint-state batch sizes must match")
        return integrate_ode(
            self._field(condition_tokens),
            initial_joint_state,
            t0=t0,
            t1=t1,
            steps=steps,
            method=solver,
            return_trajectory=return_trajectory,
        )

    def _sample(
        self,
        source_image: Tensor,
        condition_tokens: Tensor,
        *,
        direction: str,
        num_samples: int,
        seed: int,
        steps: int,
        solver: str,
        return_joint_state: bool,
        allow_residual_endpoint: bool,
    ) -> SampleBatch:
        self._require_clean_endpoint_path(allow_residual_endpoint)
        if direction not in {"forward", "backward"}:
            raise ValueError("direction must be 'forward' or 'backward'")
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        with torch.no_grad():
            source_latent = self.autoencoder.encode(source_image, normalize=True)
            return self._sample_latent(
                source_latent,
                condition_tokens,
                direction=direction,
                num_samples=num_samples,
                seed=seed,
                steps=steps,
                solver=solver,
                return_joint_state=return_joint_state,
                allow_residual_endpoint=allow_residual_endpoint,
            )

    def _sample_latent(
        self,
        source_latent: Tensor,
        condition_tokens: Tensor,
        *,
        direction: str,
        num_samples: int,
        seed: int,
        steps: int,
        solver: str,
        return_joint_state: bool,
        allow_residual_endpoint: bool,
    ) -> SampleBatch:
        self._require_clean_endpoint_path(allow_residual_endpoint)
        if direction not in {"forward", "backward"}:
            raise ValueError("direction must be 'forward' or 'backward'")
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if source_latent.ndim != 5 or any(size < 1 for size in source_latent.shape):
            raise ValueError(
                "normalized source latent must have non-empty shape [B,C,D,H,W]"
            )
        if not source_latent.is_floating_point():
            raise TypeError("normalized source latent must use a floating dtype")
        if not torch.isfinite(source_latent).all():
            raise ValueError("normalized source latent must contain only finite values")
        if condition_tokens.ndim != 3 or any(size < 1 for size in condition_tokens.shape):
            raise ValueError(
                "condition tokens must have non-empty shape [B,num_tokens,token_dim]"
            )
        if not condition_tokens.is_floating_point():
            raise TypeError("condition tokens must use a floating dtype")
        if not torch.isfinite(condition_tokens).all():
            raise ValueError("condition tokens must contain only finite values")
        if source_latent.shape[0] != condition_tokens.shape[0]:
            raise ValueError("source and condition batch sizes must match")

        with torch.no_grad():
            outputs: list[Tensor] = []
            joint_outputs: list[Tensor] = []
            seeds: list[int] = []
            expected_nfe: int | None = None
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
                if direction == "forward":
                    initial = torch.cat((noise, source_latent), dim=1)
                    t0, t1 = 0.0, 1.0
                    output_branch = slice(0, source_latent.shape[1])
                else:
                    initial = torch.cat((source_latent, noise), dim=1)
                    t0, t1 = 1.0, 0.0
                    output_branch = slice(source_latent.shape[1], None)
                solution = self.sample_joint(
                    initial,
                    condition_tokens,
                    t0=t0,
                    t1=t1,
                    steps=steps,
                    solver=solver,
                )
                decoded = self.autoencoder.decode(
                    solution.final_state[:, output_branch], denormalize=True
                )
                outputs.append(decoded)
                if return_joint_state:
                    joint_outputs.append(solution.final_state)
                seeds.append(current_seed)
                expected_nfe = solution.nfe
            samples = torch.stack(outputs, dim=0)
            standard_deviation = samples.std(dim=0, unbiased=False)
            return SampleBatch(
                direction=direction,
                samples=samples,
                mean=samples.mean(dim=0),
                std=standard_deviation,
                seeds=tuple(seeds),
                nfe_per_sample=int(expected_nfe or 0),
                final_joint_states=torch.stack(joint_outputs, dim=0) if joint_outputs else None,
            )

    def sample_forward(
        self,
        earlier_image: Tensor,
        condition_tokens: Tensor,
        *,
        num_samples: int = 8,
        seed: int = 0,
        steps: int = 25,
        solver: str = "heun",
        return_joint_state: bool = False,
        allow_residual_endpoint: bool = False,
    ) -> SampleBatch:
        """Predict the later MRI from an earlier source MRI."""

        return self._sample(
            earlier_image,
            condition_tokens,
            direction="forward",
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            solver=solver,
            return_joint_state=return_joint_state,
            allow_residual_endpoint=allow_residual_endpoint,
        )

    def sample_backward(
        self,
        later_image: Tensor,
        condition_tokens: Tensor,
        *,
        num_samples: int = 8,
        seed: int = 0,
        steps: int = 25,
        solver: str = "heun",
        return_joint_state: bool = False,
        allow_residual_endpoint: bool = False,
    ) -> SampleBatch:
        """Retrospectively reconstruct the earlier MRI from a later source MRI."""

        return self._sample(
            later_image,
            condition_tokens,
            direction="backward",
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            solver=solver,
            return_joint_state=return_joint_state,
            allow_residual_endpoint=allow_residual_endpoint,
        )

    def sample_forward_latent(
        self,
        earlier_latent: Tensor,
        condition_tokens: Tensor,
        *,
        num_samples: int = 8,
        seed: int = 0,
        steps: int = 25,
        solver: str = "heun",
        return_joint_state: bool = False,
        allow_residual_endpoint: bool = False,
    ) -> SampleBatch:
        """Predict and decode the later MRI from an already normalized latent."""

        return self._sample_latent(
            earlier_latent,
            condition_tokens,
            direction="forward",
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            solver=solver,
            return_joint_state=return_joint_state,
            allow_residual_endpoint=allow_residual_endpoint,
        )

    def sample_backward_latent(
        self,
        later_latent: Tensor,
        condition_tokens: Tensor,
        *,
        num_samples: int = 8,
        seed: int = 0,
        steps: int = 25,
        solver: str = "heun",
        return_joint_state: bool = False,
        allow_residual_endpoint: bool = False,
    ) -> SampleBatch:
        """Decode the earlier reconstruction from an already normalized latent."""

        return self._sample_latent(
            later_latent,
            condition_tokens,
            direction="backward",
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            solver=solver,
            return_joint_state=return_joint_state,
            allow_residual_endpoint=allow_residual_endpoint,
        )


def save_sample_batch(
    batch: SampleBatch,
    output: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save arrays separately from an auditable JSON sampling record."""

    path = Path(output).expanduser().resolve()
    if path.suffix == "":
        path = path.with_suffix(".npz")
    elif path.suffix != ".npz":
        raise ValueError("sample array output must use the .npz extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "samples": batch.samples.detach().cpu().numpy(),
        "mean": batch.mean.detach().cpu().numpy(),
        "std": batch.std.detach().cpu().numpy(),
    }
    if batch.final_joint_states is not None:
        arrays["final_joint_states"] = batch.final_joint_states.detach().cpu().numpy()
    np.savez_compressed(path, **arrays)
    record = {
        "direction": batch.direction,
        "seeds": list(batch.seeds),
        "nfe_per_sample": batch.nfe_per_sample,
        "array_file": path.name,
        "array_sha256": sha256_file(path),
        "metadata": metadata or {},
    }
    sidecar = path.with_suffix(".json")
    with sidecar.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path, sidecar
