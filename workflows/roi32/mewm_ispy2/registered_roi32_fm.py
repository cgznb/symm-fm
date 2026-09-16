"""Train the original SymmFlow objective with the independent ROI32 contract."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import torch

from . import registered_roi32_latents as latents
from . import registered_roi32_runtime as runtime
from .registered_roi32_data import file_identity, read_json


def bridge(config):
    source = Path(config["symm_repo"]).resolve() / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module("ispy2_symmflow.training.registered_roi32")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise ValueError("Imported SymmFlow comes from a different checkout")
    return module


def contract_for(config):
    module = bridge(config)
    source = Path(config["symm_repo"]) / "src" / "ispy2_symmflow"
    dependencies = [source / name for name in ("models/velocity.py", "models/conditioning.py", "flow/path.py", "flow/solver.py",
                                               "training/engine.py", "training/ema.py", "training/validation.py", "training/schema.py", "training/datasets.py")]
    return runtime.stage_contract(config, "fm", [__file__, module.__file__, latents.__file__, runtime.__file__, *dependencies],
                                  codec=file_identity(Path(config["output_dir"]) / "vq" / "best.pt"),
                                  latent_statistics=file_identity(Path(config["output_dir"]) / "latents" / "statistics.json"))


def build(config, records, device):
    module = bridge(config)
    from ispy2_symmflow.training.ema import ExponentialMovingAverage
    from ispy2_symmflow.training.engine import SymmFlowTrainer
    model = module.RegisteredROI32Flow(config["fm"], records).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["fm"]["learning_rate"], weight_decay=config["fm"]["weight_decay"])

    def multiplier(step):
        warmup = config["fm"]["warmup_updates"]
        maximum = config["fm"]["max_updates"]
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, maximum - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    ema = ExponentialMovingAverage(model.velocity_model, decay=config["fm"]["ema_decay"])
    trainer = SymmFlowTrainer(model.velocity_model, model.condition_encoder, optimizer, model.objective,
                              device=device, precision="bf16" if device.type == "cuda" else "fp32",
                              gradient_clip_norm=config["runtime"]["gradient_clip"], scheduler=scheduler, ema=ema)
    return model, optimizer, scheduler, ema, trainer


def train(config, device, batch_size):
    module = bridge(config)
    from ispy2_symmflow.training.validation import validate_symmflow, validate_symmflow_endpoints
    contract = contract_for(config)
    if runtime.stage_complete(config, "fm", contract):
        return True
    root = Path(config["output_dir"]) / "fm"
    runtime.seed_all(config["seed"])
    training = latents.LatentPairs(config, "train")
    validation = latents.LatentPairs(config, "val")
    records = training.inventory["pairs"]
    model, optimizer, scheduler, ema, trainer = build(config, records, device)
    trainer.gradient_accumulation = config["fm"]["batch_size"] // batch_size
    stream = runtime.TrainingBatches(training, batch_size=batch_size, effective_batch=config["fm"]["batch_size"],
                                      seed=config["seed"], workers=config["runtime"]["loader_workers"], balanced=True,
                                      collate_fn=module.collate_pairs)
    state = {"step": 0, "best_loss": None, "best_endpoint": None, "last_validation": None, "last_endpoint": None}
    last, best_loss, best_endpoint = root / "last.pt", root / "best-loss.pt", root / "best-endpoint.pt"
    if last.exists():
        saved = runtime.load_checkpoint(last, contract, device)
        if saved["model_description"] != model.description:
            raise ValueError("FM condition schema changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        ema.load_state_dict(saved["ema"])
        stream.load_state_dict(saved["stream"])
        trainer.micro_step = saved["micro_step"]
        trainer.scaler.load_state_dict(saved["scaler"])
        state = saved["training_state"]
        runtime.restore_rng(saved["rng"])
        del saved
    clock = runtime.UpdateClock(state["step"])

    def checkpoint(path):
        runtime.save_checkpoint(path, {"contract": contract, "model": model.state_dict(), "model_description": model.description,
                                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "ema": ema.state_dict(),
                                      "scaler": trainer.scaler.state_dict(), "micro_step": trainer.micro_step,
                                      "stream": stream.state_dict(), "training_state": dict(state), "rng": runtime.rng_state()})

    try:
        while state["step"] < config["fm"]["max_updates"]:
            for _ in range(trainer.gradient_accumulation):
                batch = next(stream)
                metrics = trainer.train_batch(batch["later_latent"], batch["earlier_latent"], batch["conditions"])
            if not metrics["optimizer_updated"]:
                raise RuntimeError("FM update counter and accumulation disagree")
            if not all(math.isfinite(metrics[key]) for key in ("loss", "loss_x", "loss_y", "gradient_norm")):
                raise FloatingPointError("Non-finite FM update")
            state["step"] += 1
            step = state["step"]
            if step == 1 or step % config["runtime"]["log_interval"] == 0:
                runtime.log_event(config, "fm", "training", step=step, epoch=stream.epoch, batch_size=batch_size,
                                  effective_batch=config["fm"]["batch_size"], **metrics, **clock.metrics(step, config["fm"]["max_updates"]))
                runtime.periodic_guard(config, contract)
            if step % config["fm"]["validation_interval"] == 0:
                loader = runtime.evaluation_loader(validation, batch_size, config["runtime"]["loader_workers"], module.collate_pairs)
                with runtime.fixed_rng(config["fm"]["validation_seed"]), ema.average_parameters(model.velocity_model):
                    metrics = validate_symmflow(model.velocity_model, model.condition_encoder, model.objective, loader,
                                               device=device, seed=config["fm"]["validation_seed"],
                                               precision="bf16" if device.type == "cuda" else "fp32")
                state["last_validation"] = metrics
                if state["best_loss"] is None or metrics["loss"] < state["best_loss"]:
                    state["best_loss"] = metrics["loss"]
                    checkpoint(best_loss)
                runtime.log_event(config, "fm", "validation", step=step, **metrics)
            if step % config["fm"]["endpoint_interval"] == 0:
                loader = runtime.evaluation_loader(validation, batch_size, config["runtime"]["loader_workers"], module.collate_pairs)
                with runtime.fixed_rng(config["fm"]["validation_seed"]), ema.average_parameters(model.velocity_model):
                    metrics = validate_symmflow_endpoints(model.velocity_model, model.condition_encoder, loader,
                                                          device=device, seed=config["fm"]["validation_seed"],
                                                          samples_per_pair=config["fm"]["endpoint_samples"],
                                                          steps=config["fm"]["sampling_steps"], solver=config["fm"]["solver"],
                                                          precision="bf16" if device.type == "cuda" else "fp32")
                state["last_endpoint"] = metrics
                score = metrics["endpoint_candidate_mae"]
                if state["best_endpoint"] is None or score < state["best_endpoint"]:
                    state["best_endpoint"] = score
                    checkpoint(best_endpoint)
                runtime.log_event(config, "fm", "endpoint_validation", step=step, **metrics)
            if step % config["runtime"]["checkpoint_interval"] == 0 or runtime.STOP_REQUESTED:
                checkpoint(last)
            if runtime.STOP_REQUESTED:
                runtime.log_event(config, "fm", "paused", step=step)
                return False
        checkpoint(last)
        if not best_endpoint.exists():
            raise RuntimeError("No endpoint-selected EMA checkpoint")
        runtime.stage_finished(config, "fm", contract, [last, best_loss, best_endpoint],
                               updates=state["step"], best_endpoint_candidate_mae=state["best_endpoint"],
                               selection="fixed_noise_all_validation_pairs_mean_candidate_latent_mae")
        runtime.log_event(config, "fm", "complete", step=state["step"], best_endpoint_candidate_mae=state["best_endpoint"])
        return True
    finally:
        stream.close()


def load_selected(config, device):
    module = bridge(config)
    contract = contract_for(config)
    if not runtime.stage_complete(config, "fm", contract):
        raise ValueError("FM training has not completed")
    records = read_json(Path(config["output_dir"]) / "data" / "inventory.json")["pairs"]
    model = module.RegisteredROI32Flow(config["fm"], records).to(device)
    payload = runtime.load_checkpoint(Path(config["output_dir"]) / "fm" / "best-endpoint.pt", contract, "cpu")
    model.load_state_dict(payload["model"])
    parameters = dict(model.velocity_model.named_parameters())
    with torch.no_grad():
        for name, value in payload["ema"]["shadow"].items():
            parameters[name].copy_(value.to(device))
    return model.eval().requires_grad_(False)
