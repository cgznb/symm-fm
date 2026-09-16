"""Small, resumable three-phase SymmFlow experiment with source-only sampling."""

from __future__ import annotations

import gc
import json
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .first_post_world_data import PatientBalancedBatchSampler, identity, load_codec, now, public, read_json, write_json
from .first_post_world_training import atomic_checkpoint, cpu_tree, restore_rng, rng_state, seed_all, train_step
from .three_phase_pilot_data import (
    SCHEMA, atomic_arrays, build_generated_pillar_input, cached_arrays, phase_errors, render_comparison,
    pcr_helpers, require_geometry_audit, support_coverage,
)
from .three_phase_symmflow import PHASES, SharedThreePhaseCodec, ThreePhaseSymmFlow, import_symmflow


def load_latents(cfg, inventory, *, source_only=False):
    values = {}
    for visit in inventory["visits"]:
        if visit["pilot_split"] == "excluded_geometry":
            continue
        if source_only and (visit["pilot_split"] != "val" or visit["visit"] != "T0"):
            continue
        with np.load(Path(cfg["output_root"]) / visit["cache_file"], allow_pickle=False) as archive:
            latent = np.array(archive["latent"], dtype=np.float32)
        if latent.shape != (24, 24, 64, 64) or not np.isfinite(latent).all():
            raise ValueError("Invalid prepared three-phase latent")
        values[visit["visit_id"]] = torch.from_numpy(latent)
    return values


def fit_latent_statistics(inventory, latents):
    selected = [v for v in inventory["visits"] if v["pilot_split"] == "train"]
    total, squares, count = torch.zeros(24, dtype=torch.float64), torch.zeros(24, dtype=torch.float64), 0
    for visit in selected:
        value = latents[visit["visit_id"]].double().flatten(1)
        if value.shape[0] != 24 or not torch.isfinite(value).all():
            raise ValueError("Invalid training latent values")
        total += value.sum(1)
        squares += value.square().sum(1)
        count += value.shape[1]
    if not count:
        raise ValueError("No training voxels for latent normalization")
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(0).sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Degenerate training latent normalization")
    return {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "pilot_train",
            "unique_visits": len(selected), "patients": len({v["canonical_patient_id"] for v in selected}),
            "phase_order": list(PHASES)}


class PilotDataset(Dataset):
    def __init__(self, inventory, latents, statistics, split):
        self.records = [p for p in inventory["pairs"] if p["split"] == split]
        self.mean = torch.tensor(statistics["mean"], dtype=torch.float32).view(24, 1, 1, 1)
        self.std = torch.tensor(statistics["std"], dtype=torch.float32).view(24, 1, 1, 1)
        self.latents = {key: (value - self.mean) / self.std for key, value in latents.items()}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {"source": self.latents[row["earlier_visit_id"]], "target": self.latents[row["later_visit_id"]],
                "record": row}

    def denormalize(self, value):
        return value * self.std.to(value)[None] + self.mean.to(value)[None]


def collate(items):
    return {"source": torch.stack([row["source"] for row in items]),
            "target": torch.stack([row["target"] for row in items]), "records": [row["record"] for row in items]}


def mean_phase_errors(rows, key):
    if not rows:
        raise ValueError("No validation pairs to evaluate")
    return {phase: {metric: float(np.mean([row[key][phase][metric] for row in rows]))
                   for metric in ("mae", "rmse")} for phase in PHASES}


@torch.inference_mode()
def evaluate(cfg, inventory, model, dataset, *, step, ema=None, save_predictions=False):
    previous_mode, random_state = model.training, rng_state()
    model.eval()
    try:
        with ema.average_parameters(model) if ema is not None else nullcontext():
            return _evaluate(cfg, inventory, model, dataset, step=step, save_predictions=save_predictions)
    finally:
        model.train(previous_mode)
        restore_rng(random_state)
        gc.collect()
        torch.cuda.empty_cache()


def _evaluate(cfg, inventory, model, dataset, *, step, save_predictions):
    root = Path(cfg["output_root"])
    visits = {v["visit_id"]: v for v in inventory["visits"]}
    codec = SharedThreePhaseCodec(load_codec(cfg["codec_checkpoint"], "cuda:0"))
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, record in enumerate(dataset.records):
            source = dataset.latents[record["earlier_visit_id"]][None].to("cuda:0")
            rng = torch.Generator(device="cuda:0").manual_seed(cfg["seed"] + 1000003 * index)
            noise = torch.randn(source.shape, generator=rng, device=source.device, dtype=source.dtype)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted_latent = model.sample(source, [record], noise, cfg["training"]["solver_steps"])
            predicted = codec.decode(dataset.denormalize(predicted_latent)).float()[0].cpu().numpy()
            target_visit, source_visit = visits[record["later_visit_id"]], visits[record["earlier_visit_id"]]
            reference, earlier = cached_arrays(cfg, target_visit), cached_arrays(cfg, source_visit)
            row = {"case_index": target_visit["pilot_case_index"], "stage": record["later_stage"],
                   "generation": phase_errors(predicted, reference["images"], reference["foreground"]),
                   "copy_t0": phase_errors(earlier["images"], reference["images"], reference["foreground"]),
                   "target_codec": phase_errors(reference["reconstruction"], reference["images"], reference["foreground"]),
                   "source_support_coverage": {phase: support_coverage(earlier["source_support"], reference["reference_support"][i])
                                               for i, phase in enumerate(PHASES)}}
            rows.append(row)
            if save_predictions:
                stem = f"case_{target_visit['pilot_case_index']:03d}_{record['later_stage']}"
                atomic_arrays(root / "predictions" / f"{stem}.npz", prediction=predicted,
                              source_support=earlier["source_support"])
                if record["later_stage"] == "T3":
                    z = int(np.argmax(earlier["foreground"][1].sum(axis=(1, 2))))
                    render_comparison(root / "figures" / f"{stem}.png",
                                      {"Real T0": earlier["images"], "Generated": predicted,
                                       "Real target": reference["images"], "Target VQ": reference["reconstruction"]},
                                      z=z, title=f"Three-phase pilot, case {target_visit['pilot_case_index'] + 1}, T0 to T3")
            del source, predicted_latent, noise
    del codec
    gc.collect()
    torch.cuda.empty_cache()
    summary = {key: mean_phase_errors(rows, key) for key in ("generation", "copy_t0", "target_codec")}
    score = float(np.mean([summary["generation"][phase]["mae"] for phase in PHASES]))
    result = {"schema": SCHEMA, "optimizer_step": step, "selection_metric": "mean_phase_foreground_mae",
              "score": score, "summary": summary, "records": rows, "pairs": len(rows),
              "patients": len({row["case_index"] for row in rows}), "solver": "euler",
              "solver_steps": cfg["training"]["solver_steps"], "samples_per_pair": 1,
              "seconds": time.perf_counter() - started, "completed_utc": now(),
              "input_policy": "real_t0_only", "future_geometry_used_for_generation": False,
              "spatial_policy": cfg["spatial_policy"],
              "reference_localization_uses_target_annotations": cfg["spatial_policy"] == "visit_center_with_t0_extent",
              "original_validation_evaluated": False}
    write_json(root / "evaluation" / f"step_{step:06d}.json", result)
    return result


def train(cfg, inventory, *, resume=False, progress=None, stop_after=None):
    import_symmflow(cfg["symm_repo"])
    from ispy2_symmflow.training.ema import ExponentialMovingAverage
    from ispy2_symmflow.training.engine import build_warmup_cosine_scheduler
    root = Path(cfg["output_root"])
    settings = cfg["training"]
    require_geometry_audit(cfg)
    seed_all(settings["seed"])
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    latents = load_latents(cfg, inventory)
    statistics = fit_latent_statistics(inventory, latents)
    write_json(root / "latent_statistics.json", statistics)
    dataset = PilotDataset(inventory, latents, statistics, "train")
    validation = PilotDataset(inventory, latents, statistics, "val")
    del latents
    model = ThreePhaseSymmFlow(dataset.records, source=cfg["symm_repo"]).to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=settings["warmup_steps"],
                                            total_steps=settings["max_optimizer_steps"])
    ema = ExponentialMovingAverage(model, decay=settings["ema_decay"])
    contract = public({"schema": SCHEMA, "config": cfg, "inventory": identity(root / "admitted_inventory.json"),
                "statistics": statistics, "model": model.description,
                "cache_files": {v["cache_file"]: identity(root / v["cache_file"]) for v in inventory["visits"]
                                if v["pilot_split"] != "excluded_geometry"}})
    contract_path = root / "training_contract.json"
    if contract_path.exists() and read_json(contract_path) != contract:
        raise ValueError("Training data, model or configuration changed")
    write_json(contract_path, contract)
    checkpoint = root / "checkpoints/recovery.ckpt"
    step, best = 0, None
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if public(saved["contract"]) != contract:
            raise ValueError("Recovery contract does not match")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        ema.load_state_dict(saved["ema"])
        restore_rng(saved["rng"])
        step, best = saved["optimizer_step"], saved["best_score"]
        del saved
    elif checkpoint.exists():
        raise ValueError("Existing pilot checkpoint requires --resume")
    effective, micro = settings["effective_batch_size"], settings["microbatch"]
    if effective % micro:
        raise ValueError("Effective batch must be divisible by microbatch")
    sampler = PatientBalancedBatchSampler(dataset.records, micro, effective, step,
                                         settings["max_optimizer_steps"], seed=cfg["seed"])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=collate,
                        generator=torch.Generator().manual_seed(cfg["seed"] + 17))
    iterator = iter(loader)
    stopping = []
    def stop(signum, frame):
        stopping.append(signum)
    previous_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)}
    def save_recovery():
        atomic_checkpoint(checkpoint, {"schema": SCHEMA, "contract": contract, "optimizer_step": step,
                          "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                          "scheduler": scheduler.state_dict(), "ema": cpu_tree(ema.state_dict()),
                          "rng": rng_state(), "best_score": best, "updated_utc": now()}, root)
    started = time.perf_counter()
    initial = step
    try:
        while step < settings["max_optimizer_steps"] and not stopping and (stop_after is None or step - initial < stop_after):
            metrics = train_step(model, optimizer, scheduler, ema, iterator, effective // micro, "cuda:0")
            step += 1
            if step == 1 or step % 25 == 0:
                row = {"optimizer_step": step, **metrics,
                       "learning_rate": optimizer.param_groups[0]["lr"], "updated_utc": now()}
                with (root / "training_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                if progress:
                    progress({"stage": "training", **row,
                              "total_steps": settings["max_optimizer_steps"],
                              "elapsed_seconds": time.perf_counter() - started})
            if step % settings["validation_interval"] == 0 or step == settings["max_optimizer_steps"]:
                save_recovery()
                result = evaluate(cfg, inventory, model, validation, step=step, ema=ema)
                if best is None or result["score"] < best:
                    best = result["score"]
                    with ema.average_parameters(model):
                        atomic_checkpoint(root / "checkpoints/best.ckpt", {"schema": SCHEMA, "contract": contract,
                                          "optimizer_step": step, "model": cpu_tree(model.state_dict()),
                                          "score": best, "evaluation_weights": "ema"}, root)
                save_recovery()
                if progress:
                    progress({"stage": "validation_complete", "optimizer_step": step,
                              "score": result["score"], "best_score": best, "summary": result["summary"]})
            elif step % settings["checkpoint_interval"] == 0:
                save_recovery()
        save_recovery()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    if step < settings["max_optimizer_steps"]:
        if progress:
            progress({"stage": "paused", "optimizer_step": step})
        return False
    selected = torch.load(root / "checkpoints/best.ckpt", map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(selected["model"], strict=True)
    result = evaluate(cfg, inventory, model, validation, step=selected["optimizer_step"], save_predictions=True)
    write_json(root / "training_complete.json", {"optimizer_steps": step, "selected_step": selected["optimizer_step"],
               "best_score": best, "new_updates": step - initial, "seconds": time.perf_counter() - started,
               "evaluation": result, "completed_utc": now()})
    if progress:
        progress({"stage": "training_complete", "optimizer_step": step, "selected_step": selected["optimizer_step"],
                  "summary": result["summary"]})
    return True


@torch.inference_mode()
def sample_validation(cfg, inventory):
    """Export future triplets while opening only source-visit image caches."""
    root = Path(cfg["output_root"])
    saved = torch.load(root / "checkpoints/best.ckpt", map_location="cpu", weights_only=False, mmap=True)
    if saved["contract"]["config"] != cfg or saved["contract"]["inventory"] != identity(root / "admitted_inventory.json"):
        raise ValueError("Sampling contract changed")
    records = [p for p in inventory["pairs"] if p["split"] == "train"]
    model = ThreePhaseSymmFlow(records, source=cfg["symm_repo"]).to("cuda:0").eval()
    if public(model.description) != public(saved["contract"]["model"]):
        raise ValueError("Sampling model or condition schema changed")
    model.load_state_dict(saved["model"], strict=True)
    dataset = PilotDataset(inventory, load_latents(cfg, inventory, source_only=True),
                           saved["contract"]["statistics"], "val")
    sources = {v["visit_id"]: v for v in inventory["visits"] if v["visit"] == "T0"}
    codec = SharedThreePhaseCodec(load_codec(cfg["codec_checkpoint"], "cuda:0"))
    exports = []
    for index, record in enumerate(dataset.records):
        visit = sources[record["earlier_visit_id"]]
        source = dataset.latents[record["earlier_visit_id"]][None].to("cuda:0")
        rng = torch.Generator(device="cuda:0").manual_seed(cfg["seed"] + 1000003 * index)
        noise = torch.randn(source.shape, generator=rng, device=source.device, dtype=source.dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = model.sample(source, [record], noise, cfg["training"]["solver_steps"])
        predicted = codec.decode(dataset.denormalize(latent)).float()[0].cpu().numpy()
        if not np.isfinite(predicted).all():
            raise FloatingPointError("Nonfinite generated MRI")
        with np.load(root / visit["cache_file"], allow_pickle=False) as arrays:
            support = np.array(arrays["source_support"], dtype=bool)
        path = root / "predictions" / f"case_{visit['pilot_case_index']:03d}_{record['later_stage']}.npz"
        atomic_arrays(path, prediction=predicted, source_support=support)
        exports.append({"source_visit_id": record["earlier_visit_id"], "target_visit_id": record["later_visit_id"],
                        "stage": record["later_stage"], "case_index": visit["pilot_case_index"],
                        "prediction_file": str(path.relative_to(root))})
    result = {"stage": "sampling_complete", "triplets": len(exports), "optimizer_step": saved["optimizer_step"],
              "phase_order": list(PHASES), "source_only_image_reads": True, "records": exports}
    write_json(root / "sampling_report.json", result)
    return result


def verify_pcr_interface(cfg, inventory):
    import sys
    root = Path(cfg["output_root"])
    sources = {v["visit_id"]: v for v in inventory["visits"] if v["visit"] == "T0"}
    exports = read_json(root / "sampling_report.json")["records"]
    source_ids = sorted({row["source_visit_id"] for row in exports})
    jobs = [{"source_visit_id": key, "target_visit_id": key, "stage": "T0",
             "case_index": sources[key]["pilot_case_index"], "image_source": "real_t0"} for key in source_ids]
    jobs.extend({**row, "image_source": "generated_symm_fm"} for row in exports)
    sys.path.insert(0, cfg["pillar_repo"])
    from scripts.run_first_post_pcr import load_frozen_pillar, pillar_forward
    model = load_frozen_pillar()
    output = root / "pcr_interface_check"
    output.mkdir(exist_ok=True)
    rows = []
    data, _ = pcr_helpers(cfg)
    for row in jobs:
        source = sources[row["source_visit_id"]]
        if row["image_source"] == "real_t0":
            with np.load(root / source["cache_file"], allow_pickle=False) as arrays:
                real = np.array(arrays["images"])
                foreground = np.array(arrays["foreground"], dtype=bool)
            normalization = inventory["image_normalization"]
            raw = real * normalization["std"] + normalization["mean"]
            raw[~foreground] = 0
            spacing = source["source_geometry"]["spacing_xyz_mm"][::-1]
            volume = torch.stack([data.pillar_channel(channel, spacing) for channel in raw])
        else:
            with np.load(root / row["prediction_file"], allow_pickle=False) as arrays:
                prediction, support = np.array(arrays["prediction"]), np.array(arrays["source_support"])
            volume = build_generated_pillar_input(cfg, prediction, source["source_geometry"], support, inventory["image_normalization"])
        vector = pillar_forward(model, volume[None].to("cuda:0"))[0]
        if vector.shape != (1152,) or not torch.isfinite(vector).all() or vector.norm() <= 0:
            raise ValueError("Invalid generated Pillar feature")
        destination = output / "embeddings" / source["canonical_patient_id"]
        destination.mkdir(parents=True, exist_ok=True)
        feature_path = destination / f"{source['canonical_patient_id']}_{row['stage']}.pt"
        torch.save(vector, feature_path)
        rows.append({**row, "feature_file": str(feature_path.relative_to(root)), "finite": True})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if not rows:
        raise ValueError("No generated triplets to export")
    result = {"stage": "pcr_interface_complete", "passed": True, "phase_order": list(PHASES), "generated_shape_czyx": list(prediction.shape),
              "pillar_shape_bchwd": [1, *volume.shape], "feature_shape": list(vector.shape),
              "finite": True, "geometry_source": "real_t0", "features": len(rows), "records": rows,
              "real_t0_features": len(source_ids), "generated_features": len(exports),
              "future_image_or_mask_read": False, "pcr_performance_evaluated": False}
    write_json(output / "report.json", result)
    return result
