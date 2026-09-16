"""Deterministic held-out validation for checkpoint selection."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ispy2_symmflow.flow.path import SymmetricFlowObjective
from ispy2_symmflow.flow.solver import integrate_ode
from ispy2_symmflow.training.engine import _autocast, autoencoder_objective
from ispy2_symmflow.utils.hashing import stable_hash


def _finite_metrics(metrics: Mapping[str, float]) -> None:
    invalid = {name: value for name, value in metrics.items() if not math.isfinite(value)}
    if invalid:
        raise FloatingPointError(f"validation produced non-finite metrics: {invalid}")


def select_endpoint_validation_records(
    records: Iterable[Mapping[str, Any]], *, limit: int, seed: int
) -> list[dict[str, Any]]:
    """Choose a stable, patient-diverse held-out subset for ODE rollouts."""

    if limit < 1:
        raise ValueError("endpoint validation record limit must be positive")
    candidates = [dict(record) for record in records if record.get("split") == "val"]
    ordered = sorted(
        candidates,
        key=lambda record: stable_hash(
            {
                "namespace": "symmflow-latent-endpoint-validation-v1",
                "seed": int(seed),
                "pair_id": record.get("pair_id"),
            }
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    patients: set[str] = set()
    for record in ordered:
        patient = str(record.get("patient_id"))
        if patient in patients:
            continue
        selected.append(record)
        selected_ids.add(str(record.get("pair_id")))
        patients.add(patient)
        if len(selected) == limit:
            return selected
    for record in ordered:
        pair_id = str(record.get("pair_id"))
        if pair_id in selected_ids:
            continue
        selected.append(record)
        if len(selected) == limit:
            break
    return selected


@torch.no_grad()
def validate_autoencoder(
    model: nn.Module,
    images: Iterable[Tensor],
    *,
    device: torch.device,
    kl_weight: float,
    gradient_weight: float = 0.0,
    precision: str = "fp32",
) -> dict[str, float | int]:
    """Evaluate posterior-mean reconstruction without consuming training RNG."""

    was_training = model.training
    model.eval()
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0, "gradient": 0.0}
    sample_count = 0
    try:
        for image in images:
            image = image.to(device, non_blocking=True)
            with _autocast(device, precision):
                losses = autoencoder_objective(
                    model,
                    image,
                    kl_weight=kl_weight,
                    gradient_weight=gradient_weight,
                    sample_posterior=False,
                )
            batch_size = int(image.shape[0])
            values = {
                "loss": float(losses.total),
                "reconstruction": float(losses.reconstruction),
                "kl": float(losses.kl),
                "gradient": float(losses.gradient),
            }
            _finite_metrics(values)
            for name, value in values.items():
                totals[name] += value * batch_size
            sample_count += batch_size
    finally:
        model.train(was_training)
    if sample_count < 1:
        raise ValueError("autoencoder validation requires at least one held-out visit")
    metrics = {name: value / sample_count for name, value in totals.items()}
    _finite_metrics(metrics)
    return {**metrics, "sample_count": sample_count}


@torch.no_grad()
def validate_symmflow(
    velocity_model: nn.Module,
    condition_encoder: nn.Module,
    objective: SymmetricFlowObjective,
    batches: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    seed: int,
    repeats: int = 1,
    precision: str = "fp32",
) -> dict[str, float | int]:
    """Evaluate fixed-seed two-branch FM loss on held-out patient pairs."""

    if repeats < 1:
        raise ValueError("validation repeats must be at least 1")
    velocity_was_training = velocity_model.training
    condition_was_training = condition_encoder.training
    velocity_model.eval()
    condition_encoder.eval()
    totals = {"loss": 0.0, "loss_x": 0.0, "loss_y": 0.0}
    sample_count = 0
    try:
        for repeat in range(repeats):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) + repeat)
            for batch in batches:
                later = batch["later_latent"].to(device, non_blocking=True)
                earlier = batch["earlier_latent"].to(device, non_blocking=True)
                with _autocast(device, precision):
                    tokens = condition_encoder(
                        batch["conditions"], batch_size=later.shape[0]
                    )
                    losses = objective(
                        velocity_model,
                        later,
                        earlier,
                        tokens,
                        generator=generator,
                    )
                batch_size = int(later.shape[0])
                values = {
                    "loss": float(losses.total),
                    "loss_x": float(losses.x),
                    "loss_y": float(losses.y),
                }
                _finite_metrics(values)
                for name, value in values.items():
                    totals[name] += value * batch_size
                sample_count += batch_size
    finally:
        velocity_model.train(velocity_was_training)
        condition_encoder.train(condition_was_training)
    if sample_count < 1:
        raise ValueError("SymmFlow validation requires at least one held-out pair")
    metrics = {name: value / sample_count for name, value in totals.items()}
    _finite_metrics(metrics)
    return {
        **metrics,
        "sample_count": sample_count // repeats,
        "stochastic_repeats": repeats,
        "seed": int(seed),
    }


@torch.no_grad()
def validate_symmflow_endpoints(
    velocity_model: nn.Module,
    condition_encoder: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    seed: int,
    samples_per_pair: int = 2,
    steps: int = 8,
    solver: str = "heun",
    precision: str = "fp32",
) -> dict[str, float | int]:
    """Roll out fixed-noise forward endpoints and compare them in latent space."""

    if samples_per_pair < 1:
        raise ValueError("endpoint validation samples_per_pair must be at least 1")
    if steps < 1:
        raise ValueError("endpoint validation steps must be at least 1")
    if solver not in {"euler", "heun"}:
        raise ValueError("endpoint validation solver must be 'euler' or 'heun'")
    velocity_was_training = velocity_model.training
    condition_was_training = condition_encoder.training
    velocity_model.eval()
    condition_encoder.eval()
    totals = {
        "endpoint_candidate_mse": 0.0,
        "endpoint_candidate_mae": 0.0,
        "endpoint_mean_mse": 0.0,
        "endpoint_mean_mae": 0.0,
        "endpoint_source_copy_mse": 0.0,
        "endpoint_source_copy_mae": 0.0,
    }
    pair_count = 0
    try:
        for batch_index, batch in enumerate(batches):
            later = batch["later_latent"].to(device, non_blocking=True)
            earlier = batch["earlier_latent"].to(device, non_blocking=True)
            batch_size, channels = later.shape[:2]
            with _autocast(device, precision):
                tokens = condition_encoder(batch["conditions"], batch_size=batch_size)
            predictions: list[Tensor] = []
            for sample_index in range(samples_per_pair):
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    int(seed) + batch_index * samples_per_pair + sample_index
                )
                noise = torch.randn(
                    earlier.shape,
                    dtype=earlier.dtype,
                    device=device,
                    generator=generator,
                )
                initial = torch.cat((noise, earlier), dim=1)

                def field(state: Tensor, tau: Tensor) -> Tensor:
                    with _autocast(device, precision):
                        return velocity_model(state, tau, tokens)

                solution = integrate_ode(
                    field,
                    initial,
                    t0=0.0,
                    t1=1.0,
                    steps=steps,
                    method=solver,
                )
                predicted = solution.final_state[:, :channels].float()
                target = later.float()
                totals["endpoint_candidate_mse"] += (
                    float(F.mse_loss(predicted, target)) * batch_size
                )
                totals["endpoint_candidate_mae"] += (
                    float(F.l1_loss(predicted, target)) * batch_size
                )
                predictions.append(predicted)
            predictive_mean = torch.stack(predictions).mean(dim=0)
            target = later.float()
            source = earlier.float()
            totals["endpoint_mean_mse"] += (
                float(F.mse_loss(predictive_mean, target)) * batch_size
            )
            totals["endpoint_mean_mae"] += (
                float(F.l1_loss(predictive_mean, target)) * batch_size
            )
            totals["endpoint_source_copy_mse"] += (
                float(F.mse_loss(source, target)) * batch_size
            )
            totals["endpoint_source_copy_mae"] += (
                float(F.l1_loss(source, target)) * batch_size
            )
            pair_count += batch_size
    finally:
        velocity_model.train(velocity_was_training)
        condition_encoder.train(condition_was_training)
    if pair_count < 1:
        raise ValueError("endpoint validation requires at least one held-out pair")
    metrics = {
        name: value / (pair_count * samples_per_pair)
        if name.startswith("endpoint_candidate_")
        else value / pair_count
        for name, value in totals.items()
    }
    _finite_metrics(metrics)
    return {
        **metrics,
        "endpoint_pair_count": pair_count,
        "endpoint_samples_per_pair": samples_per_pair,
        "endpoint_steps": steps,
        "endpoint_nfe_per_sample": steps * (2 if solver == "heun" else 1),
        "endpoint_seed": int(seed),
    }


__all__ = [
    "select_endpoint_validation_records",
    "validate_autoencoder",
    "validate_symmflow",
    "validate_symmflow_endpoints",
]
