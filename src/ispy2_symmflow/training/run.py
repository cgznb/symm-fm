"""High-level staged training entry points used by the command-line interface."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ispy2_symmflow.flow.path import SymmetricFlowObjective
from ispy2_symmflow.models import (
    StructuredConditionEncoder,
    build_autoencoder_from_config,
    build_velocity_model_from_config,
)
from ispy2_symmflow.training.checkpoint import load_checkpoint, save_checkpoint
from ispy2_symmflow.training.contracts import require_configured_time_pairs
from ispy2_symmflow.training.datasets import (
    LatentPairDataset,
    PreparedVisitDataset,
    collate_latent_pairs,
    read_jsonl,
)
from ispy2_symmflow.training.distributed import (
    distributed_sampler,
    finalize_distributed,
    initialize_distributed,
    patient_balanced_sampler,
    rank_seed,
)
from ispy2_symmflow.training.ema import ExponentialMovingAverage
from ispy2_symmflow.training.engine import (
    AutoencoderTrainer,
    SymmFlowTrainer,
    build_warmup_cosine_scheduler,
)
from ispy2_symmflow.training.schema import fit_condition_schema
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    validate_cached_pair_manifest_binding,
    validate_cached_latent_provenance,
)
from ispy2_symmflow.training.validation import (
    select_endpoint_validation_records,
    validate_autoencoder,
    validate_symmflow,
    validate_symmflow_endpoints,
)
from ispy2_symmflow.utils.hashing import stable_hash
from ispy2_symmflow.utils.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)


UPSTREAM_COMMITS = {
    "symmetricflow": "cb14c609c81ba7f6654d66e7de2be6e769fa6104",
    "medsymmflow": "824eb0f2e23fd062306f7e5c1211f35353bd0260",
    "mewm": "9620f58",
    "monai_1.5.1": "9c6d819f97e37f36c72f3bdfad676b455bd2fa0d",
}


def _workers(config: Mapping[str, Any]) -> int:
    return int(config.get("project", {}).get("num_workers", 0))


def _split_fingerprint(records: list[dict[str, Any]]) -> str:
    assignments: dict[str, str] = {}
    for record in records:
        patient = str(record["patient_id"])
        split = str(record["split"])
        previous = assignments.setdefault(patient, split)
        if previous != split:
            raise ValueError(f"patient {patient} occurs in both {previous} and {split}")
    return stable_hash(assignments)


def _has_error_qc(record: Mapping[str, Any]) -> bool:
    qc = record.get("qc") or ()
    return any(
        isinstance(event, Mapping)
        and str(event.get("severity", "")).strip().lower() == "error"
        for event in qc
    )


def _training_signature(
    stage: str,
    records: list[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    batch_size: int,
    world_size: int,
    planned_steps: int,
    batches_per_epoch: int,
) -> dict[str, Any]:
    """Describe everything that fixes data order and the optimization plan."""

    signature = {
        "stage": stage,
        "ordered_manifest_fingerprint": stable_hash(records),
        "manifest_record_count": len(records),
        "config_fingerprint": stable_hash(config),
        "batch_size_per_rank": int(batch_size),
        "world_size": int(world_size),
        "planned_steps": int(planned_steps),
        "batches_per_epoch_per_rank": int(batches_per_epoch),
    }
    if stage == "symmflow":
        fingerprints = {
            str(record.get(CACHED_PAIR_MANIFEST_FINGERPRINT, "")).strip()
            for record in records
        }
        if len(fingerprints) != 1 or not next(iter(fingerprints), ""):
            raise ValueError(
                "cached pairs do not share one cached-pair manifest fingerprint"
            )
        signature[CACHED_PAIR_MANIFEST_FINGERPRINT] = next(iter(fingerprints))
        signature[CACHED_PAIR_MANIFEST_RECORD_COUNT] = len(records)
    return signature


def _synchronize_validation_metrics(
    metrics: Mapping[str, float | int], context
) -> dict[str, float | int]:
    """Make every rank use rank zero's full-split validation result."""

    local = dict(metrics)
    if context.world_size == 1:
        return local
    gathered: list[dict[str, float | int] | None] = [None] * context.world_size
    torch.distributed.all_gather_object(gathered, local)
    reference = gathered[0]
    if reference is None or any(value is None for value in gathered):
        raise RuntimeError("failed to gather validation metrics from every rank")
    for candidate in gathered[1:]:
        assert candidate is not None
        if set(candidate) != set(reference):
            raise RuntimeError("validation metric fields differ across ranks")
        for name, expected in reference.items():
            observed = candidate[name]
            if isinstance(expected, int) and isinstance(observed, int):
                matches = expected == observed
            else:
                matches = math.isclose(
                    float(expected), float(observed), rel_tol=1e-5, abs_tol=1e-7
                )
            if not matches:
                raise RuntimeError(
                    f"validation metric {name!r} differs across ranks: "
                    f"rank0={expected}, observed={observed}"
                )
    return dict(reference)


def _image_collate(items: list[dict[str, Any]]) -> torch.Tensor:
    return torch.stack([item["image"] for item in items], dim=0)


def _gather_rng_states(context) -> list[dict[str, Any]]:
    local = capture_rng_state()
    if context.world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] = [None] * context.world_size
    torch.distributed.all_gather_object(gathered, local)
    if any(value is None for value in gathered):
        raise RuntimeError("failed to gather all per-rank random states")
    return [value for value in gathered if value is not None]


def _restore_rank_rng(payload: Mapping[str, Any], context) -> None:
    states = payload.get("extra", {}).get("rng_by_rank")
    if states is None:
        if context.world_size > 1:
            raise ValueError("distributed resume requires per-rank RNG states")
        restore_rng_state(payload["rng_state"])
        return
    if len(states) != context.world_size:
        raise ValueError(
            "checkpoint RNG world size differs from the current distributed run"
        )
    restore_rng_state(states[context.rank])


def _resume_best_checkpoint(
    extra: Mapping[str, Any],
    resume: str | Path,
    *,
    legacy_name: str,
    label: str,
) -> Path:
    """Resolve the selected validation artifact carried by a resume checkpoint."""

    recorded = str(extra.get("best_checkpoint", "")).strip()
    candidate = (
        Path(recorded).expanduser().resolve()
        if recorded
        else Path(resume).expanduser().resolve().with_name(legacy_name)
    )
    if not candidate.is_file():
        raise ValueError(
            f"{label} resume checkpoint references no readable best checkpoint: {candidate}"
        )
    return candidate


def train_autoencoder(
    config: Mapping[str, Any],
    manifest: str | Path,
    *,
    output_dir: str | Path,
    resume: str | Path | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    context = initialize_distributed()
    try:
        base_seed = int(config["project"]["seed"])
        seed = rank_seed(base_seed, context)
        seed_everything(seed)
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device.index or 0)
        records = read_jsonl(manifest)
        split_digest = _split_fingerprint(records)
        dataset = PreparedVisitDataset(records, split="train")
        validation_dataset = PreparedVisitDataset(records, split="val")
        sampler = distributed_sampler(dataset, context, shuffle=True, seed=base_seed)
        loader_generator = torch.Generator()
        validation_loader_generator = torch.Generator()
        section = config["autoencoder"]
        batch_size = int(section.get("batch_size", 1))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=_workers(config),
            pin_memory=context.device.type == "cuda",
            collate_fn=_image_collate,
            generator=loader_generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_workers(config),
            pin_memory=context.device.type == "cuda",
            collate_fn=_image_collate,
            generator=validation_loader_generator,
        )
        base_model = build_autoencoder_from_config(config).to(context.device)
        train_model: nn.Module = base_model
        if context.world_size > 1:
            train_model = DistributedDataParallel(
                base_model, device_ids=[context.local_rank], output_device=context.local_rank
            )
        optimizer = torch.optim.AdamW(
            train_model.parameters(),
            lr=float(section.get("learning_rate", 1e-4)),
            weight_decay=float(section.get("weight_decay", 0.01)),
        )
        epochs = int(section.get("max_epochs", 1))
        total_batches = max(1, epochs * len(loader))
        training_signature = _training_signature(
            "autoencoder",
            records,
            config,
            batch_size=batch_size,
            world_size=context.world_size,
            planned_steps=total_batches,
            batches_per_epoch=len(loader),
        )
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=min(int(section.get("warmup_steps", 0)), max(0, total_batches - 1)),
            total_steps=max(1, total_batches),
        ) if total_batches > 1 else None
        trainer = AutoencoderTrainer(
            train_model,
            optimizer,
            device=context.device,
            kl_weight=float(section.get("kl_weight", 1e-6)),
            gradient_weight=float(section.get("gradient_weight", 0.0)),
            precision=str(config.get("flow", {}).get("precision", "fp32")),
            gradient_clip_norm=float(section.get("gradient_clip_norm", 1.0)),
            scheduler=scheduler,
        )
        start_epoch = 0
        start_batch_cursor = 0
        global_step = 0
        best_validation_loss = float("inf")
        validation_history: list[dict[str, Any]] = []
        payload: dict[str, Any] | None = None
        destination = Path(output_dir).expanduser().resolve()
        checkpoint_path = destination / "autoencoder.pt"
        local_best_checkpoint_path = destination / "autoencoder_best.pt"
        best_checkpoint_path = local_best_checkpoint_path
        if resume is not None:
            payload = load_checkpoint(
                resume,
                model=base_model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_split_hash=split_digest,
                expected_training_signature=training_signature,
                required_extra_keys=(
                    "batch_in_epoch",
                    "best_validation_loss",
                    "scaler_state",
                    "validation_history",
                ),
                restore_rng=False,
                map_location=context.device,
            )
            extra = payload["extra"]
            start_epoch = int(payload["epoch"])
            start_batch_cursor = int(extra["batch_in_epoch"])
            global_step = int(payload["step"])
            trainer.scaler.load_state_dict(extra["scaler_state"])
            best_validation_loss = float(extra["best_validation_loss"])
            validation_history = [dict(item) for item in extra["validation_history"]]
            best_checkpoint_path = _resume_best_checkpoint(
                extra,
                resume,
                legacy_name="autoencoder_best.pt",
                label="autoencoder",
            )
            _restore_rank_rng(payload, context)
        last_metrics: dict[str, float] = (
            dict(payload.get("extra", {}).get("last_train_metrics", {}))
            if payload is not None
            else {}
        )
        checkpoint_epoch = start_epoch
        checkpoint_cursor = start_batch_cursor
        target_steps = int(max_steps) if max_steps is not None else total_batches
        if target_steps > total_batches:
            raise ValueError(
                f"--max-steps={target_steps} exceeds the configured {total_batches}-step AE plan"
            )
        if global_step > target_steps:
            raise ValueError(
                f"resume step {global_step} is beyond requested --max-steps={target_steps}"
            )
        if checkpoint_cursor >= len(loader):
            checkpoint_epoch += checkpoint_cursor // len(loader)
            checkpoint_cursor %= len(loader)
        start_epoch = checkpoint_epoch
        start_batch_cursor = checkpoint_cursor
        last_validation_metrics: dict[str, float | int] = (
            dict(validation_history[-1]["metrics"]) if validation_history else {}
        )

        def save_training_state(paths: list[Path]) -> None:
            rng_by_rank = _gather_rng_states(context)
            if context.is_primary:
                extra = {
                    "last_train_metrics": last_metrics,
                    "posterior_for_flow": "mean",
                    "batch_in_epoch": checkpoint_cursor,
                    "rng_by_rank": rng_by_rank,
                    "scaler_state": trainer.scaler.state_dict(),
                    "training_signature": training_signature,
                    "best_validation_loss": best_validation_loss,
                    "best_checkpoint": str(best_checkpoint_path),
                    "validation_history": validation_history,
                }
                for path in paths:
                    save_checkpoint(
                        path,
                        model=base_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=checkpoint_epoch,
                        step=global_step,
                        config=config,
                        autoencoder_id=None,
                        latent_statistics=None,
                        feature_schema=None,
                        split_hash=split_digest,
                        upstream_commits=UPSTREAM_COMMITS,
                        extra=extra,
                    )
            if context.world_size > 1:
                torch.distributed.barrier()

        def run_validation() -> None:
            nonlocal best_validation_loss, best_checkpoint_path, last_validation_metrics
            validation_loader_generator.manual_seed(base_seed)
            metrics = validate_autoencoder(
                base_model,
                validation_loader,
                device=context.device,
                kl_weight=float(section.get("kl_weight", 1e-6)),
                gradient_weight=float(section.get("gradient_weight", 0.0)),
                precision=str(config.get("flow", {}).get("precision", "fp32")),
            )
            last_validation_metrics = _synchronize_validation_metrics(metrics, context)
            validation_history.append(
                {
                    "epoch": checkpoint_epoch,
                    "step": global_step,
                    "batch_in_epoch": checkpoint_cursor,
                    "metrics": last_validation_metrics,
                }
            )
            improved = float(last_validation_metrics["loss"]) < best_validation_loss
            if improved:
                best_validation_loss = float(last_validation_metrics["loss"])
                best_checkpoint_path = local_best_checkpoint_path
            paths = [checkpoint_path]
            if improved:
                paths.insert(0, local_best_checkpoint_path)
            save_training_state(paths)

        validation_interval = int(section.get("validation_interval_epochs", 1))
        if validation_interval < 1:
            raise ValueError("autoencoder validation_interval_epochs must be at least 1")
        stop = global_step >= target_steps
        for epoch in range(start_epoch, epochs):
            if stop:
                break
            if sampler is not None:
                sampler.set_epoch(epoch)
            loader_generator.manual_seed(base_seed + epoch)
            completed_epoch = True
            for batch_index, image in enumerate(loader):
                if epoch == start_epoch and batch_index < start_batch_cursor:
                    continue
                last_metrics = trainer.train_batch(image)
                global_step += 1
                checkpoint_epoch = epoch
                checkpoint_cursor = batch_index + 1
                if global_step >= target_steps:
                    stop = True
                    completed_epoch = False
                    break
            if completed_epoch:
                checkpoint_epoch = epoch + 1
                checkpoint_cursor = 0
                if checkpoint_epoch % validation_interval == 0:
                    run_validation()
            if stop:
                break
        if checkpoint_cursor >= len(loader):
            checkpoint_epoch += checkpoint_cursor // len(loader)
            checkpoint_cursor %= len(loader)
        current_position = (checkpoint_epoch, global_step, checkpoint_cursor)
        last_position = (
            (
                int(validation_history[-1]["epoch"]),
                int(validation_history[-1]["step"]),
                int(validation_history[-1]["batch_in_epoch"]),
            )
            if validation_history
            else None
        )
        if current_position != last_position:
            run_validation()
        return {
            "checkpoint": str(checkpoint_path),
            "best_checkpoint": str(best_checkpoint_path),
            "steps": global_step,
            "epochs": checkpoint_epoch,
            "batch_in_epoch": checkpoint_cursor,
            "last_train_metrics": last_metrics,
            "last_validation_metrics": last_validation_metrics,
            "best_validation_loss": best_validation_loss,
            "validation_history": validation_history,
            "training_signature": training_signature,
            "split_hash": split_digest,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(context.device.index or 0) / (1024**2)
                if context.device.type == "cuda"
                else None
            ),
        }
    finally:
        finalize_distributed(context)


def _read_latent_statistics(pair_manifest: str | Path) -> dict[str, Any]:
    candidate = Path(pair_manifest).resolve().parent / "latent_statistics.json"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"latent statistics must accompany the cached pair manifest: {candidate}"
        )
    return json.loads(candidate.read_text(encoding="utf-8"))


def _append_training_log(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False))
        handle.write("\n")


def train_symmflow(
    config: Mapping[str, Any],
    pair_manifest: str | Path,
    *,
    output_dir: str | Path,
    expected_autoencoder_id: str | None = None,
    resume: str | Path | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    context = initialize_distributed()
    try:
        base_seed = int(config["project"]["seed"])
        seed = rank_seed(base_seed, context)
        seed_everything(seed)
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device.index or 0)
        source_records = read_jsonl(pair_manifest)
        latent_statistics = _read_latent_statistics(pair_manifest)
        validate_cached_pair_manifest_binding(source_records, latent_statistics)
        records = [record for record in source_records if not _has_error_qc(record)]
        validate_cached_latent_provenance(
            records, latent_statistics, validate_manifest=False
        )
        rejected_error_qc_count = len(source_records) - len(records)
        configured_pairs = require_configured_time_pairs(records, config.get("data", {}))
        autoencoder_ids = {record.get("autoencoder_id") for record in records}
        if len(autoencoder_ids) != 1 or None in autoencoder_ids:
            raise ValueError("cached pairs must share one explicit autoencoder_id")
        autoencoder_id = str(next(iter(autoencoder_ids)))
        if str(latent_statistics.get("autoencoder_id")) != autoencoder_id:
            raise ValueError(
                "latent statistics autoencoder_id does not match the cached pair manifest"
            )
        if expected_autoencoder_id is not None and autoencoder_id != expected_autoencoder_id:
            raise ValueError("cached latent autoencoder_id does not match the supplied checkpoint")
        split_digest = _split_fingerprint(records)
        schema, schema_provenance = fit_condition_schema(records, config["conditions"])
        base_condition = StructuredConditionEncoder(schema).to(context.device)
        base_velocity = build_velocity_model_from_config(config).to(context.device)
        velocity: nn.Module = base_velocity
        condition_encoder: nn.Module = base_condition
        if context.world_size > 1:
            velocity = DistributedDataParallel(
                base_velocity, device_ids=[context.local_rank], output_device=context.local_rank
            )
            condition_encoder = DistributedDataParallel(
                base_condition, device_ids=[context.local_rank], output_device=context.local_rank
            )
        dataset = LatentPairDataset(records, split="train")
        validation_dataset = LatentPairDataset(records, split="val")
        pair_sampling = str(
            config.get("data", {}).get(
                "pair_sampling",
                "patient_balanced" if len(configured_pairs) > 1 else "uniform_pairs",
            )
        )
        sampler = (
            patient_balanced_sampler(dataset.records, context, seed=base_seed)
            if pair_sampling == "patient_balanced"
            else distributed_sampler(dataset, context, shuffle=True, seed=base_seed)
        )
        loader_generator = torch.Generator()
        validation_loader_generator = torch.Generator()
        endpoint_loader_generator = torch.Generator()
        flow = config["flow"]
        velocity_config = config["velocity"]
        batch_size = int(velocity_config.get("batch_size", 1))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=_workers(config),
            pin_memory=context.device.type == "cuda",
            collate_fn=collate_latent_pairs,
            generator=loader_generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_workers(config),
            pin_memory=context.device.type == "cuda",
            collate_fn=collate_latent_pairs,
            generator=validation_loader_generator,
        )
        endpoint_pair_limit = int(flow.get("endpoint_validation_pairs", 0))
        endpoint_loader = None
        if endpoint_pair_limit:
            endpoint_records = select_endpoint_validation_records(
                records,
                limit=endpoint_pair_limit,
                seed=int(flow.get("validation_seed", 1729)),
            )
            expected_endpoint_pairs = min(endpoint_pair_limit, len(validation_dataset))
            if len(endpoint_records) != expected_endpoint_pairs:
                raise RuntimeError(
                    "failed to select the requested endpoint validation pairs"
                )
            endpoint_dataset = LatentPairDataset(endpoint_records, split="val")
            endpoint_loader = DataLoader(
                endpoint_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=_workers(config),
                pin_memory=context.device.type == "cuda",
                collate_fn=collate_latent_pairs,
                generator=endpoint_loader_generator,
            )
        parameters = list(velocity.parameters()) + list(condition_encoder.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(flow.get("learning_rate", 1e-4)),
            weight_decay=float(flow.get("weight_decay", 0.01)),
        )
        planned_steps = int(flow.get("max_steps", 1))
        requested_steps = int(max_steps if max_steps is not None else planned_steps)
        if requested_steps > planned_steps:
            raise ValueError(
                f"--max-steps={requested_steps} exceeds flow.max_steps={planned_steps}"
            )
        training_signature = _training_signature(
            "symmflow",
            source_records,
            config,
            batch_size=batch_size,
            world_size=context.world_size,
            planned_steps=planned_steps,
            batches_per_epoch=len(loader),
        )
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=min(int(flow.get("warmup_steps", 0)), max(0, planned_steps - 1)),
            total_steps=planned_steps,
        ) if planned_steps > 1 else None
        ema = ExponentialMovingAverage(base_velocity, decay=float(flow.get("ema_decay", 0.9999)))
        system = nn.ModuleDict({"velocity": base_velocity, "conditions": base_condition})
        objective = SymmetricFlowObjective(
            sigma_min=float(flow.get("sigma_min", 0.0)),
            loss_weight_x=float(flow.get("loss_weight_x", 1.0)),
            loss_weight_y=float(flow.get("loss_weight_y", 1.0)),
        )
        trainer = SymmFlowTrainer(
            velocity,
            condition_encoder,
            optimizer,
            objective,
            device=context.device,
            precision=str(flow.get("precision", "fp32")),
            gradient_clip_norm=float(flow.get("gradient_clip_norm", 1.0)),
            gradient_accumulation=int(flow.get("gradient_accumulation", 1)),
            scheduler=scheduler,
            ema=ema,
        )
        optimizer_step = 0
        epoch = 0
        batch_cursor = 0
        best_validation_loss = float("inf")
        checkpoint_metric = str(flow.get("checkpoint_metric", "loss"))
        validation_history: list[dict[str, Any]] = []
        payload: dict[str, Any] | None = None
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        metrics_log_path = destination / "training_metrics.jsonl"
        log_interval = int(flow.get("log_interval_steps", 10))
        checkpoint_path = destination / "symmflow.pt"
        local_best_checkpoint_path = destination / "symmflow_best.pt"
        best_checkpoint_path = local_best_checkpoint_path
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
                expected_latent_statistics=latent_statistics,
                expected_sigma_min=float(flow.get("sigma_min", 0.0)),
                expected_training_signature=training_signature,
                required_extra_keys=(
                    "batch_in_epoch",
                    "best_validation_loss",
                    "micro_step",
                    "scaler_state",
                    "validation_history",
                ),
                restore_rng=False,
                map_location=context.device,
            )
            extra = payload["extra"]
            optimizer_step = int(payload["step"])
            epoch = int(payload["epoch"])
            batch_cursor = int(extra["batch_in_epoch"])
            trainer.scaler.load_state_dict(extra["scaler_state"])
            trainer.micro_step = int(extra["micro_step"])
            best_validation_loss = float(extra["best_validation_loss"])
            validation_history = [dict(item) for item in extra["validation_history"]]
            best_checkpoint_path = _resume_best_checkpoint(
                extra,
                resume,
                legacy_name="symmflow_best.pt",
                label="SymmFlow",
            )
            _restore_rank_rng(payload, context)
        if trainer.micro_step % trainer.gradient_accumulation != 0:
            raise ValueError(
                "resumable SymmFlow checkpoints must be written at an optimizer boundary"
            )
        if optimizer_step > requested_steps:
            raise ValueError(
                f"resume step {optimizer_step} is beyond requested --max-steps={requested_steps}"
            )
        last_metrics: dict[str, float | bool] = (
            dict(payload.get("extra", {}).get("last_train_metrics", {}))
            if payload is not None
            else {}
        )
        if batch_cursor >= len(loader):
            epoch += batch_cursor // len(loader)
            batch_cursor %= len(loader)
        last_validation_metrics: dict[str, float | int] = (
            dict(validation_history[-1]["metrics"]) if validation_history else {}
        )

        def save_training_state(paths: list[Path]) -> None:
            rng_by_rank = _gather_rng_states(context)
            if context.is_primary:
                extra = {
                    "last_train_metrics": last_metrics,
                    "schema_provenance": schema_provenance,
                    "branch_order": ["later", "earlier"],
                    "batch_in_epoch": batch_cursor,
                    "rng_by_rank": rng_by_rank,
                    "scaler_state": trainer.scaler.state_dict(),
                    "micro_step": trainer.micro_step,
                    "training_signature": training_signature,
                    "best_validation_loss": best_validation_loss,
                    "checkpoint_metric": checkpoint_metric,
                    "best_checkpoint": str(best_checkpoint_path),
                    "validation_history": validation_history,
                    "rejected_pair_count": rejected_error_qc_count,
                    "rejected_error_qc_count": rejected_error_qc_count,
                }
                for path in paths:
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
                        latent_statistics=latent_statistics,
                        feature_schema=schema.to_dict(),
                        split_hash=split_digest,
                        upstream_commits=UPSTREAM_COMMITS,
                        extra=extra,
                    )
            if context.world_size > 1:
                torch.distributed.barrier()

        def run_validation() -> None:
            nonlocal best_validation_loss, best_checkpoint_path, last_validation_metrics
            validation_seed = int(flow.get("validation_seed", 1729))
            validation_loader_generator.manual_seed(validation_seed)
            endpoint_loader_generator.manual_seed(validation_seed)
            with ema.average_parameters(base_velocity):
                metrics = validate_symmflow(
                    base_velocity,
                    base_condition,
                    objective,
                    validation_loader,
                    device=context.device,
                    seed=validation_seed,
                    repeats=int(flow.get("validation_repeats", 1)),
                    precision=str(flow.get("precision", "fp32")),
                )
                if endpoint_loader is not None:
                    metrics.update(
                        validate_symmflow_endpoints(
                            base_velocity,
                            base_condition,
                            endpoint_loader,
                            device=context.device,
                            seed=validation_seed,
                            samples_per_pair=int(
                                flow.get("endpoint_validation_samples", 2)
                            ),
                            steps=int(flow.get("endpoint_validation_steps", 8)),
                            solver=str(
                                flow.get("endpoint_validation_solver", "heun")
                            ),
                            precision=str(flow.get("precision", "fp32")),
                        )
                    )
            last_validation_metrics = _synchronize_validation_metrics(metrics, context)
            if checkpoint_metric not in last_validation_metrics:
                raise ValueError(
                    f"flow.checkpoint_metric {checkpoint_metric!r} is not produced by validation"
                )
            selection_value = float(last_validation_metrics[checkpoint_metric])
            if not math.isfinite(selection_value):
                raise FloatingPointError(
                    f"checkpoint metric {checkpoint_metric!r} is non-finite"
                )
            validation_history.append(
                {
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "batch_in_epoch": batch_cursor,
                    "micro_step": trainer.micro_step,
                    "checkpoint_metric": checkpoint_metric,
                    "checkpoint_metric_value": selection_value,
                    "metrics": last_validation_metrics,
                }
            )
            if context.is_primary:
                _append_training_log(
                    metrics_log_path,
                    {
                        "event": "validation",
                        "optimizer_step": optimizer_step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_cursor,
                        "micro_step": trainer.micro_step,
                        **last_validation_metrics,
                    },
                )
            improved = selection_value < best_validation_loss
            if improved:
                best_validation_loss = selection_value
                best_checkpoint_path = local_best_checkpoint_path
            paths = [checkpoint_path]
            if improved:
                paths.insert(0, local_best_checkpoint_path)
            save_training_state(paths)

        validation_interval = int(flow.get("validation_interval_steps", 1000))
        if validation_interval < 1:
            raise ValueError("flow validation_interval_steps must be at least 1")
        while optimizer_step < requested_steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            loader_generator.manual_seed(base_seed + epoch)
            completed_epoch = True
            deferred_validation = False
            for batch_index, batch in enumerate(loader):
                if batch_index < batch_cursor:
                    continue
                last_metrics = trainer.train_batch(
                    batch["later_latent"], batch["earlier_latent"], batch["conditions"]
                )
                batch_cursor = batch_index + 1
                if last_metrics["optimizer_updated"]:
                    optimizer_step += 1
                    if context.is_primary and (
                        optimizer_step == 1 or optimizer_step % log_interval == 0
                    ):
                        peak_memory = (
                            torch.cuda.max_memory_allocated(context.device.index or 0)
                            / (1024**2)
                            if context.device.type == "cuda"
                            else None
                        )
                        _append_training_log(
                            metrics_log_path,
                            {
                                "event": "train",
                                "optimizer_step": optimizer_step,
                                "epoch": epoch,
                                "batch_in_epoch": batch_cursor,
                                "micro_step": trainer.micro_step,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "peak_cuda_memory_mb": peak_memory,
                                **last_metrics,
                            },
                        )
                    if optimizer_step % validation_interval == 0:
                        if batch_cursor == len(loader):
                            deferred_validation = True
                        else:
                            run_validation()
                if optimizer_step >= requested_steps:
                    completed_epoch = False
                    break
            if completed_epoch:
                epoch += 1
                batch_cursor = 0
            elif batch_cursor >= len(loader):
                epoch += batch_cursor // len(loader)
                batch_cursor %= len(loader)
            if deferred_validation:
                run_validation()
        if batch_cursor >= len(loader):
            epoch += batch_cursor // len(loader)
            batch_cursor %= len(loader)
        current_position = (epoch, optimizer_step, batch_cursor, trainer.micro_step)
        last_position = (
            (
                int(validation_history[-1]["epoch"]),
                int(validation_history[-1]["optimizer_step"]),
                int(validation_history[-1]["batch_in_epoch"]),
                int(validation_history[-1]["micro_step"]),
            )
            if validation_history
            else None
        )
        if current_position != last_position:
            run_validation()
        return {
            "checkpoint": str(checkpoint_path),
            "metrics_log": str(metrics_log_path),
            "best_checkpoint": str(best_checkpoint_path),
            "optimizer_steps": optimizer_step,
            "epochs": epoch,
            "batch_in_epoch": batch_cursor,
            "micro_step": trainer.micro_step,
            "last_train_metrics": last_metrics,
            "last_validation_metrics": last_validation_metrics,
            "best_validation_loss": best_validation_loss,
            "checkpoint_metric": checkpoint_metric,
            "validation_history": validation_history,
            "condition_schema": schema.to_dict(),
            "training_signature": training_signature,
            "rejected_pair_count": rejected_error_qc_count,
            "rejected_error_qc_count": rejected_error_qc_count,
            "split_hash": split_digest,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(context.device.index or 0) / (1024**2)
                if context.device.type == "cuda"
                else None
            ),
        }
    finally:
        finalize_distributed(context)
