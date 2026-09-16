"""Train validation-selected deterministic and one-way CFM comparisons."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ispy2_symmflow.flow.solver import integrate_ode
from ispy2_symmflow.models import (
    DeterministicLatentObjective,
    StructuredConditionEncoder,
    UnidirectionalConditionalFMObjective,
    build_deterministic_baseline_from_config,
    build_unidirectional_cfm_from_config,
    make_unidirectional_cfm_initial_state,
)
from ispy2_symmflow.training.checkpoint import load_checkpoint, save_checkpoint
from ispy2_symmflow.training.contracts import require_configured_time_pairs
from ispy2_symmflow.training.datasets import (
    LatentPairDataset,
    collate_latent_pairs,
    read_jsonl,
)
from ispy2_symmflow.training.distributed import (
    DistributedContext,
    patient_balanced_sampler,
)
from ispy2_symmflow.training.ema import ExponentialMovingAverage
from ispy2_symmflow.training.engine import _autocast, build_warmup_cosine_scheduler
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    validate_cached_pair_manifest_binding,
    validate_cached_latent_provenance,
)
from ispy2_symmflow.training.schema import fit_condition_schema
from ispy2_symmflow.training.validation import select_endpoint_validation_records
from ispy2_symmflow.utils.hashing import stable_hash
from ispy2_symmflow.utils.reproducibility import restore_rng_state, seed_everything


BASELINE_KINDS = ("deterministic", "unidirectional_cfm")


def _append_metrics(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")


def _has_error_qc(record: Mapping[str, Any]) -> bool:
    return any(
        isinstance(event, Mapping)
        and str(event.get("severity", "")).strip().lower() == "error"
        for event in (record.get("qc") or ())
    )


def _split_hash(records: list[dict[str, Any]]) -> str:
    assignments: dict[str, str] = {}
    for record in records:
        patient = str(record["patient_id"])
        split = str(record["split"])
        previous = assignments.setdefault(patient, split)
        if previous != split:
            raise ValueError(f"patient {patient} occurs in both {previous} and {split}")
    return stable_hash(assignments)


def _latent_statistics(pair_manifest: str | Path) -> dict[str, Any]:
    path = Path(pair_manifest).resolve().parent / "latent_statistics.json"
    if not path.is_file():
        raise FileNotFoundError(f"cached baseline pairs require {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _condition_tokens(
    encoder: StructuredConditionEncoder,
    values: Mapping[str, Any],
    *,
    batch_size: int,
) -> torch.Tensor:
    return encoder(values, batch_size=batch_size)


@torch.no_grad()
def validate_baseline(
    kind: str,
    model: nn.Module,
    condition_encoder: StructuredConditionEncoder,
    batches,
    *,
    device: torch.device,
    precision: str,
    sigma_min: float,
    seed: int,
    deterministic_loss: str = "mse",
    cfm_base_distribution: str = "standard_normal",
    cfm_noise_scale: float = 1.0,
) -> dict[str, float | int]:
    """Evaluate one comparison with deterministic held-out randomness."""

    if kind not in BASELINE_KINDS:
        raise ValueError(f"unsupported baseline kind {kind!r}")
    model_was_training = model.training
    encoder_was_training = condition_encoder.training
    model.eval()
    condition_encoder.eval()
    objective = (
        DeterministicLatentObjective(deterministic_loss)
        if kind == "deterministic"
        else UnidirectionalConditionalFMObjective(
            sigma_min,
            base_distribution=cfm_base_distribution,
            noise_scale=cfm_noise_scale,
        )
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    total = 0.0
    sample_count = 0
    try:
        for batch in batches:
            source = batch["earlier_latent"].to(device, non_blocking=True)
            target = batch["later_latent"].to(device, non_blocking=True)
            with _autocast(device, precision):
                tokens = _condition_tokens(
                    condition_encoder,
                    batch["conditions"],
                    batch_size=source.shape[0],
                )
                if kind == "deterministic":
                    loss = objective(model, source, target, tokens).total
                else:
                    loss = objective(
                        model,
                        source,
                        target,
                        tokens,
                        generator=generator,
                    ).total
            value = float(loss)
            if not math.isfinite(value):
                raise FloatingPointError("baseline validation produced a non-finite loss")
            total += value * source.shape[0]
            sample_count += source.shape[0]
    finally:
        model.train(model_was_training)
        condition_encoder.train(encoder_was_training)
    if sample_count < 1:
        raise ValueError("baseline validation requires at least one held-out pair")
    return {"loss": total / sample_count, "sample_count": sample_count, "seed": int(seed)}


@torch.no_grad()
def validate_cfm_endpoint(
    model: nn.Module,
    condition_encoder: StructuredConditionEncoder,
    batches,
    *,
    device: torch.device,
    precision: str,
    seed: int,
    base_distribution: str,
    noise_scale: float,
    pair_count: int = 8,
    samples_per_pair: int = 2,
    steps: int = 1,
    solver: str = "euler",
) -> dict[str, float | int | str]:
    """Measure deterministic held-out rollouts in normalized latent space."""

    if pair_count < 1 or samples_per_pair < 1 or steps < 1:
        raise ValueError("CFM endpoint validation counts and steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError("CFM endpoint validation solver must be 'euler' or 'heun'")
    model_was_training = model.training
    encoder_was_training = condition_encoder.training
    model.eval()
    condition_encoder.eval()
    candidate_mse = 0.0
    candidate_mae = 0.0
    mean_mse = 0.0
    mean_mae = 0.0
    source_mse = 0.0
    source_mae = 0.0
    evaluated = 0
    nfe = 0
    try:
        for batch in batches:
            remaining = pair_count - evaluated
            if remaining <= 0:
                break
            source = batch["earlier_latent"].to(device, non_blocking=True)[:remaining]
            target = batch["later_latent"].to(device, non_blocking=True)[:remaining]
            tokens = _condition_tokens(
                condition_encoder,
                batch["conditions"],
                batch_size=batch["earlier_latent"].shape[0],
            )[:remaining]
            candidates: list[torch.Tensor] = []
            for sample_index in range(samples_per_pair):
                sample_seed = int(seed) + evaluated * samples_per_pair + sample_index
                generator = torch.Generator(device=device)
                generator.manual_seed(sample_seed)
                noise = torch.randn(
                    source.shape,
                    dtype=source.dtype,
                    device=device,
                    generator=generator,
                )
                initial = make_unidirectional_cfm_initial_state(
                    source,
                    noise,
                    base_distribution=base_distribution,
                    noise_scale=noise_scale,
                )

                def field(state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
                    return model(state, source, tau, tokens)

                with _autocast(device, precision):
                    solution = integrate_ode(
                        field,
                        initial,
                        t0=0.0,
                        t1=1.0,
                        steps=steps,
                        method=solver,
                    )
                candidates.append(solution.final_state.float())
                nfe = solution.nfe
            stacked = torch.stack(candidates)
            target_float = target.float()
            candidate_error = stacked - target_float[None]
            reduce_dims = tuple(range(2, candidate_error.ndim))
            candidate_mse += float(
                candidate_error.square().mean(dim=reduce_dims).sum()
            )
            candidate_mae += float(candidate_error.abs().mean(dim=reduce_dims).sum())
            predictive_error = stacked.mean(dim=0) - target_float
            volume_dims = tuple(range(1, predictive_error.ndim))
            mean_mse += float(predictive_error.square().mean(dim=volume_dims).sum())
            mean_mae += float(predictive_error.abs().mean(dim=volume_dims).sum())
            source_error = source.float() - target_float
            source_mse += float(source_error.square().mean(dim=volume_dims).sum())
            source_mae += float(source_error.abs().mean(dim=volume_dims).sum())
            evaluated += source.shape[0]
    finally:
        model.train(model_was_training)
        condition_encoder.train(encoder_was_training)
    if evaluated < 1:
        raise ValueError("CFM endpoint validation requires held-out pairs")
    candidate_denominator = evaluated * samples_per_pair
    return {
        "endpoint_candidate_mse": candidate_mse / candidate_denominator,
        "endpoint_candidate_mae": candidate_mae / candidate_denominator,
        "endpoint_mean_mse": mean_mse / evaluated,
        "endpoint_mean_mae": mean_mae / evaluated,
        "endpoint_source_copy_mse": source_mse / evaluated,
        "endpoint_source_copy_mae": source_mae / evaluated,
        "endpoint_pair_count": evaluated,
        "endpoint_samples_per_pair": samples_per_pair,
        "endpoint_steps": steps,
        "endpoint_solver": solver,
        "endpoint_nfe_per_sample": nfe,
        "endpoint_seed": int(seed),
    }


def train_baseline(
    config: Mapping[str, Any],
    pair_manifest: str | Path,
    *,
    kind: str,
    output_dir: str | Path,
    expected_autoencoder_id: str | None = None,
    resume: str | Path | None = None,
    max_steps: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train one forward comparison on the same cached patient pairs."""

    if kind not in BASELINE_KINDS:
        raise ValueError(f"kind must be one of {BASELINE_KINDS}")
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError("WORLD_SIZE must be an integer") from exc
    if world_size != 1:
        raise RuntimeError(
            "baseline training is single-process only; do not launch train-baseline with torchrun"
        )
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = config.get("baseline_training", {})
    seed = int(config["project"]["seed"])
    seed_everything(seed)
    source_records = read_jsonl(pair_manifest)
    statistics = _latent_statistics(pair_manifest)
    validate_cached_pair_manifest_binding(source_records, statistics)
    records = [record for record in source_records if not _has_error_qc(record)]
    validate_cached_latent_provenance(records, statistics, validate_manifest=False)
    rejected_error_qc_count = len(source_records) - len(records)
    require_configured_time_pairs(records, config.get("data", {}))
    autoencoder_ids = {record.get("autoencoder_id") for record in records}
    if len(autoencoder_ids) != 1 or None in autoencoder_ids:
        raise ValueError("baseline pairs must share one explicit autoencoder_id")
    autoencoder_id = str(next(iter(autoencoder_ids)))
    if statistics.get("autoencoder_id") != autoencoder_id:
        raise ValueError("baseline latent statistics and pair autoencoder IDs differ")
    if expected_autoencoder_id is not None and expected_autoencoder_id != autoencoder_id:
        raise ValueError("baseline pairs do not match the supplied autoencoder checkpoint")
    schema, schema_provenance = fit_condition_schema(records, config["conditions"])
    condition_encoder = StructuredConditionEncoder(schema).to(device)
    model = (
        build_deterministic_baseline_from_config(config)
        if kind == "deterministic"
        else build_unidirectional_cfm_from_config(config)
    ).to(device)
    system = nn.ModuleDict({"model": model, "conditions": condition_encoder})

    train_dataset = LatentPairDataset(records, split="train")
    validation_dataset = LatentPairDataset(records, split="val")
    batch_size = int(settings.get("batch_size", config["velocity"].get("batch_size", 1)))
    workers = int(config.get("project", {}).get("num_workers", 0))
    pair_sampling = str(config.get("data", {}).get("pair_sampling", "uniform_pairs"))
    if pair_sampling not in {"uniform_pairs", "patient_balanced"}:
        raise ValueError(
            "data.pair_sampling must be 'uniform_pairs' or 'patient_balanced'"
        )
    train_sampler = (
        patient_balanced_sampler(
            train_dataset.records,
            DistributedContext(
                rank=0,
                local_rank=0,
                world_size=1,
                device=device,
                initialized_here=False,
            ),
            seed=seed,
        )
        if pair_sampling == "patient_balanced"
        else None
    )
    loader_generator = torch.Generator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=loader_generator,
        num_workers=workers,
        collate_fn=collate_latent_pairs,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_latent_pairs,
        pin_memory=device.type == "cuda",
    )
    endpoint_pairs = int(settings.get("endpoint_validation_pairs", 0))
    endpoint_loader = None
    if endpoint_pairs > 0:
        endpoint_records = select_endpoint_validation_records(
            records,
            limit=endpoint_pairs,
            seed=int(settings.get("validation_seed", 2718)),
        )
        expected_endpoint_pairs = min(endpoint_pairs, len(validation_dataset))
        if len(endpoint_records) != expected_endpoint_pairs:
            raise RuntimeError("failed to select the requested CFM endpoint pairs")
        endpoint_loader = DataLoader(
            LatentPairDataset(endpoint_records, split="val"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_latent_pairs,
            pin_memory=device.type == "cuda",
        )
    parameters = list(model.parameters()) + list(condition_encoder.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(settings.get("learning_rate", config["flow"].get("learning_rate", 1e-4))),
        weight_decay=float(settings.get("weight_decay", config["flow"].get("weight_decay", 0.01))),
    )
    planned_steps = int(settings.get("max_steps", config["flow"].get("max_steps", 1)))
    requested_steps = int(max_steps if max_steps is not None else planned_steps)
    if planned_steps < 1 or requested_steps < 1:
        raise ValueError("baseline planned and requested steps must be at least 1")
    if requested_steps > planned_steps:
        raise ValueError(f"--max-steps={requested_steps} exceeds the {planned_steps}-step baseline plan")
    warmup_steps = min(
        int(settings.get("warmup_steps", config["flow"].get("warmup_steps", 0))),
        max(0, planned_steps - 1),
    )
    scheduler = (
        build_warmup_cosine_scheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=planned_steps
        )
        if planned_steps > 1
        else None
    )
    ema = ExponentialMovingAverage(
        model, decay=float(settings.get("ema_decay", config["flow"].get("ema_decay", 0.9999)))
    )
    precision = str(settings.get("precision", config["flow"].get("precision", "fp32")))
    gradient_clip = float(
        settings.get("gradient_clip_norm", config["flow"].get("gradient_clip_norm", 1.0))
    )
    gradient_accumulation = int(settings.get("gradient_accumulation", 1))
    if gradient_accumulation < 1:
        raise ValueError("baseline gradient_accumulation must be at least 1")
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision.lower() == "fp16"
    )
    cfm_base_distribution = str(
        settings.get("cfm_base_distribution", "standard_normal")
    )
    cfm_noise_scale = float(settings.get("cfm_noise_scale", 1.0))
    if kind == "unidirectional_cfm":
        # Constructing the objective here validates the path contract before any
        # checkpoint or metrics artifacts are written.
        cfm_objective = UnidirectionalConditionalFMObjective(
            float(config["flow"].get("sigma_min", 0.0)),
            base_distribution=cfm_base_distribution,
            noise_scale=cfm_noise_scale,
        )
    else:
        cfm_objective = None
    signature = {
        "kind": kind,
        "ordered_manifest_hash": stable_hash(source_records),
        "manifest_record_count": len(source_records),
        CACHED_PAIR_MANIFEST_FINGERPRINT: statistics[
            CACHED_PAIR_MANIFEST_FINGERPRINT
        ],
        CACHED_PAIR_MANIFEST_RECORD_COUNT: statistics[
            CACHED_PAIR_MANIFEST_RECORD_COUNT
        ],
        "config_fingerprint": stable_hash(config),
        "model_parameters": {
            name: {"shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, parameter in system.named_parameters()
        },
        "seed": seed,
        "batch_size": batch_size,
        "planned_steps": planned_steps,
        "warmup_steps": warmup_steps,
        "learning_rate": optimizer.defaults["lr"],
        "weight_decay": optimizer.defaults["weight_decay"],
        "gradient_accumulation": gradient_accumulation,
        "gradient_clip_norm": gradient_clip,
        "pair_sampling": pair_sampling,
        "precision": precision,
        "sigma_min": float(config["flow"].get("sigma_min", 0.0)),
        "deterministic_loss": str(settings.get("deterministic_loss", "mse")),
    }
    if kind == "unidirectional_cfm" and (
        "cfm_base_distribution" in settings or "cfm_noise_scale" in settings
    ):
        signature.update(
            {
                "cfm_base_distribution": cfm_base_distribution,
                "cfm_noise_scale": cfm_noise_scale,
            }
        )
    split_digest = _split_hash(records)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "training_metrics.jsonl"
    latest_path = destination / f"{kind}.pt"
    selected_path = destination / f"{kind}_best.pt"
    best_path = selected_path
    best_loss = math.inf
    history: list[dict[str, Any]] = []
    optimizer_step = 0
    epoch = 0
    batch_cursor = 0
    micro_step = 0
    last_train: dict[str, float | bool] = {}
    if resume is not None:
        payload = load_checkpoint(
            resume,
            model=system,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            expected_autoencoder_id=autoencoder_id,
            expected_feature_schema=schema.to_dict(),
            expected_split_hash=split_digest,
            expected_latent_statistics=statistics,
            expected_training_signature=signature,
            required_extra_keys=(
                "batch_in_epoch",
                "best_validation_loss",
                "last_train_metrics",
                "micro_step",
                "scaler_state",
                "validation_history",
            ),
            map_location=device,
        )
        extra = payload["extra"]
        scaler.load_state_dict(extra["scaler_state"])
        restore_rng_state(payload["rng_state"])
        optimizer_step = int(payload["step"])
        epoch = int(payload["epoch"])
        batch_cursor = int(extra.get("batch_in_epoch", 0))
        micro_step = int(extra["micro_step"])
        best_loss = float(extra.get("best_validation_loss", math.inf))
        history = list(extra.get("validation_history", []))
        last_train = dict(extra.get("last_train_metrics", {}))
        previous_best = extra.get("best_checkpoint")
        if previous_best:
            best_path = Path(previous_best)
        if micro_step % gradient_accumulation != 0:
            raise ValueError(
                "resumable baseline checkpoints must be written at an optimizer boundary"
            )
        if optimizer_step > requested_steps:
            raise ValueError(
                f"resume step {optimizer_step} is beyond requested --max-steps={requested_steps}"
            )

    deterministic_objective = DeterministicLatentObjective(
        str(settings.get("deterministic_loss", "mse"))
    )
    interval = int(settings.get("validation_interval_steps", 1000))
    if interval < 1:
        raise ValueError("baseline validation_interval_steps must be at least 1")
    validation_seed = int(settings.get("validation_seed", 2718))
    endpoint_samples = int(settings.get("endpoint_validation_samples", 2))
    endpoint_steps = int(settings.get("endpoint_validation_steps", 1))
    endpoint_solver = str(settings.get("endpoint_validation_solver", "euler"))
    checkpoint_metric = str(settings.get("checkpoint_metric", "loss"))
    if endpoint_pairs < 0:
        raise ValueError("baseline endpoint_validation_pairs cannot be negative")
    if endpoint_pairs > 0 and (endpoint_samples < 1 or endpoint_steps < 1):
        raise ValueError(
            "baseline endpoint validation samples and steps must be positive"
        )
    if endpoint_pairs > 0 and endpoint_solver not in {"euler", "heun"}:
        raise ValueError("baseline endpoint validation solver must be 'euler' or 'heun'")
    allowed_checkpoint_metrics = {"loss"}
    if kind == "unidirectional_cfm" and endpoint_pairs > 0:
        allowed_checkpoint_metrics.update(
            {"endpoint_candidate_mse", "endpoint_mean_mse"}
        )
    if checkpoint_metric not in allowed_checkpoint_metrics:
        raise ValueError(
            "baseline checkpoint_metric is unavailable for this validation protocol"
        )
    log_interval = int(settings.get("log_interval_steps", 10))
    if log_interval < 1:
        raise ValueError("baseline log_interval_steps must be at least 1")

    def checkpoint_extra() -> dict[str, Any]:
        return {
            "baseline_kind": kind,
            "training_signature": signature,
            "schema_provenance": schema_provenance,
            "batch_in_epoch": batch_cursor,
            "micro_step": micro_step,
            "scaler_state": scaler.state_dict(),
            "last_train_metrics": last_train,
            "validation_history": history,
            "best_validation_loss": best_loss,
            "checkpoint_metric": checkpoint_metric,
            "best_checkpoint": str(best_path),
            "direction": "forward",
        }

    def save(path: Path) -> None:
        save_checkpoint(
            path,
            model=system,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            epoch=epoch,
            step=optimizer_step,
            config=config,
            autoencoder_id=autoencoder_id,
            latent_statistics=statistics,
            feature_schema=schema.to_dict(),
            split_hash=split_digest,
            upstream_commits={
                "symmetricflow_reference_only": "cb14c609c81ba7f6654d66e7de2be6e769fa6104",
                "monai_1.5.1": "9c6d819f97e37f36c72f3bdfad676b455bd2fa0d",
            },
            extra=checkpoint_extra(),
        )

    def validate_and_select() -> dict[str, float | int]:
        nonlocal best_loss, best_path
        with ema.average_parameters(model):
            metrics = validate_baseline(
                kind,
                model,
                condition_encoder,
                validation_loader,
                device=device,
                precision=precision,
                sigma_min=float(config["flow"].get("sigma_min", 0.0)),
                seed=validation_seed,
                deterministic_loss=str(settings.get("deterministic_loss", "mse")),
                cfm_base_distribution=cfm_base_distribution,
                cfm_noise_scale=cfm_noise_scale,
            )
            if kind == "unidirectional_cfm" and endpoint_pairs > 0:
                assert endpoint_loader is not None
                metrics.update(
                    validate_cfm_endpoint(
                        model,
                        condition_encoder,
                        endpoint_loader,
                        device=device,
                        precision=precision,
                        seed=validation_seed,
                        base_distribution=cfm_base_distribution,
                        noise_scale=cfm_noise_scale,
                        pair_count=endpoint_pairs,
                        samples_per_pair=endpoint_samples,
                        steps=endpoint_steps,
                        solver=endpoint_solver,
                    )
                )
        history.append({"step": optimizer_step, "epoch": epoch, **metrics})
        _append_metrics(
            metrics_path,
            {
                "event": "validation",
                "kind": kind,
                "optimizer_step": optimizer_step,
                "epoch": epoch,
                "batch_in_epoch": batch_cursor,
                **metrics,
            },
        )
        if float(metrics[checkpoint_metric]) < best_loss:
            best_loss = float(metrics[checkpoint_metric])
            best_path = selected_path
            save(selected_path)
        return metrics

    optimizer.zero_grad(set_to_none=True)
    if batch_cursor >= len(train_loader):
        epoch += batch_cursor // len(train_loader)
        batch_cursor %= len(train_loader)
    while optimizer_step < requested_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        loader_generator.manual_seed(seed + epoch)
        completed_epoch = True
        deferred_validation = False
        for batch_index, batch in enumerate(train_loader):
            if batch_index < batch_cursor:
                continue
            source = batch["earlier_latent"].to(device, non_blocking=True)
            target = batch["later_latent"].to(device, non_blocking=True)
            with _autocast(device, precision):
                tokens = _condition_tokens(
                    condition_encoder,
                    batch["conditions"],
                    batch_size=source.shape[0],
                )
                if kind == "deterministic":
                    raw_loss = deterministic_objective(model, source, target, tokens).total
                else:
                    assert cfm_objective is not None
                    raw_loss = cfm_objective(model, source, target, tokens).total
                loss = raw_loss / gradient_accumulation
            scaler.scale(loss).backward()
            micro_step += 1
            batch_cursor = batch_index + 1
            updated = micro_step % gradient_accumulation == 0
            gradient_norm_value = math.nan
            if updated:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
                if not torch.isfinite(gradient_norm):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("non-finite baseline gradient norm")
                gradient_norm_value = float(gradient_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                ema.update(model)
                optimizer_step += 1
            last_train = {
                "loss": float(raw_loss.detach()),
                "gradient_norm": gradient_norm_value,
                "optimizer_updated": updated,
            }
            if updated and (
                optimizer_step == 1 or optimizer_step % log_interval == 0
            ):
                peak_memory = (
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda"
                    else None
                )
                _append_metrics(
                    metrics_path,
                    {
                        "event": "train",
                        "kind": kind,
                        "optimizer_step": optimizer_step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_cursor,
                        "micro_step": micro_step,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "peak_cuda_memory_mb": peak_memory,
                        **last_train,
                    },
                )
            if updated and optimizer_step % interval == 0:
                if batch_cursor == len(train_loader):
                    deferred_validation = True
                else:
                    validate_and_select()
            if optimizer_step >= requested_steps:
                completed_epoch = False
                break
        if completed_epoch:
            epoch += 1
            batch_cursor = 0
        elif batch_cursor >= len(train_loader):
            epoch += batch_cursor // len(train_loader)
            batch_cursor %= len(train_loader)
        if deferred_validation:
            validate_and_select()
    if not history or int(history[-1]["step"]) != optimizer_step:
        last_validation = validate_and_select()
    else:
        last_validation = {
            key: history[-1][key] for key in ("loss", "sample_count", "seed")
        }
    save(latest_path)
    return {
        "kind": kind,
        "checkpoint": str(latest_path),
        "best_checkpoint": str(best_path),
        "optimizer_steps": optimizer_step,
        "epochs": epoch,
        "batch_in_epoch": batch_cursor,
        "last_train_metrics": last_train,
        "last_validation_metrics": last_validation,
        "best_validation_loss": best_loss,
        "checkpoint_metric": checkpoint_metric,
        "condition_schema": schema.to_dict(),
        "training_signature": signature,
        "rejected_pair_count": rejected_error_qc_count,
        "rejected_error_qc_count": rejected_error_qc_count,
        "split_hash": split_digest,
    }


__all__ = [
    "BASELINE_KINDS",
    "train_baseline",
    "validate_baseline",
    "validate_cfm_endpoint",
]
