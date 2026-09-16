"""Formal Symm-FM optimization, matched validation, and guarded source-only exports."""

from __future__ import annotations

import copy
import gc
import json
import os
import signal
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path

import numpy as np
import torch

from .first_post_world_data import disk_gate, identity, now, public, read_json, write_json
from .first_post_world_training import atomic_checkpoint, cpu_tree, make_loader, restore_rng, rng_state, seed_all, train_step
from .three_phase_all_pairs_data import SCHEMA, PairDataset, fit_statistics, read_latent, read_reference
from .three_phase_pilot_data import atomic_arrays, build_generated_pillar_input, phase_errors, render_comparison
from .three_phase_symmflow import PHASES, SharedThreePhaseCodec, ThreePhaseSymmFlow
from .first_post_world_data import load_codec

_READ_POLICY = ContextVar("three_phase_source_reads", default=None)


def _audit_read(event, arguments):
    policy = _READ_POLICY.get()
    if policy is None or event != "open" or not isinstance(arguments[0], (str, bytes)):
        return
    path = str(Path(os.fsdecode(arguments[0])).resolve())
    if path in policy["allowed"]:
        policy["opened"].add(path)
    elif path in policy["denied"] or any(path.startswith(prefix) for prefix in policy["prefixes"]):
        raise RuntimeError("Source-only inference attempted to open an undeclared image or latent")


sys.addaudithook(_audit_read)


@contextmanager
def source_read_guard(root, inventory, view):
    root = Path(root)
    policy = {"allowed": {str((root / view[k]).resolve()) for k in ("latent_file", "reference_file") if view.get(k)},
              "denied": {str(Path(v[k]).resolve()) for v in inventory["visits"]
                         for k in ("native_first_post", "image_path", "mask_path") if v.get(k)},
              "prefixes": [str((root / folder).resolve()) + os.sep for folder in ("latents", "references", "staging")],
              "opened": set()}
    token = _READ_POLICY.set(policy)
    try:
        yield policy
    finally:
        _READ_POLICY.reset(token)


def validation_indices(records, count, seed):
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["earlier_stage"], record["later_stage"]].append(index)
    rng = np.random.default_rng(seed)
    for values in groups.values():
        rng.shuffle(values)
    result = []
    while len(result) < min(count, len(records)):
        for key in sorted(groups):
            if groups[key] and len(result) < min(count, len(records)):
                result.append(groups[key].pop())
    return sorted(result)


def macro_metrics(rows, name):
    patients = defaultdict(list)
    for row in rows:
        patients[row["patient_id"]].append(row)
    return {phase: {metric: float(np.mean([np.mean([r[name][phase][metric] for r in values])
                                         for values in patients.values()])) for metric in ("mae", "rmse")}
            for phase in PHASES}


def summarize(rows):
    if not rows:
        raise ValueError("Empty validation")
    names = ("generation", "copy_source", "target_codec")
    summary = {name: macro_metrics(rows, name) for name in names}
    by_pair, by_coverage = {}, {}
    for key in sorted({r["transition"] for r in rows}):
        selected = [r for r in rows if r["transition"] == key]
        by_pair[key] = {"pairs": len(selected), **{name: macro_metrics(selected, name) for name in names}}
    for key, predicate in (("below_0.5", lambda r: r["minimum_coverage"] < 0.5),
                           ("at_least_0.5", lambda r: r["minimum_coverage"] >= 0.5)):
        selected = [r for r in rows if predicate(r)]
        by_coverage[key] = {"pairs": len(selected), "generation": macro_metrics(selected, "generation") if selected else None}
    return {"summary": summary, "by_transition": by_pair, "by_coverage": by_coverage,
            "score": float(np.mean([summary["generation"][phase]["mae"] for phase in PHASES])),
            "pairs": len(rows), "patients": len({r["patient_id"] for r in rows})}


def statistics_and_model(cfg, inventory):
    root = Path(cfg["output_root"])
    prepared = read_json(root / "preparation.json")
    if not prepared["passed"]:
        raise ValueError("Data preparation did not pass")
    for filename, recorded in prepared["files"].items():
        if identity(root / filename) != recorded:
            raise ValueError("Prepared files changed")
    statistics_path = root / "latent_statistics.json"
    if statistics_path.exists():
        value = read_json(statistics_path)
        if value["inventory"] != identity(root / "admitted_inventory.json"):
            raise ValueError("Latent normalization inventory changed")
        statistics = value["statistics"]
    else:
        statistics = fit_statistics(inventory, lambda view: read_latent(root, view))
        write_json(statistics_path, {"inventory": identity(root / "admitted_inventory.json"), "statistics": statistics})
    seed_all(cfg["seed"])
    training = PairDataset(root, inventory, statistics, "train")
    validation = PairDataset(root, inventory, statistics, "val")
    model = ThreePhaseSymmFlow(training.records, source=cfg["symm_repo"]).to("cuda:0")
    contract = public({"schema": SCHEMA, "config": cfg, "inventory": identity(root / "admitted_inventory.json"),
                       "statistics": statistics, "model": model.description, "cache_files": prepared["files"]})
    path = root / "training_contract.json"
    if path.exists() and read_json(path) != contract:
        raise ValueError("Formal training contract changed; pilot weights cannot be resumed here")
    if not path.exists():
        write_json(path, contract)
    return model, training, validation, contract


def loader(dataset, cfg, start, stop):
    settings = cfg["training"]
    return make_loader(dataset, {"seed": cfg["seed"], "loader_workers": cfg["runtime"]["loader_workers"]},
                       settings["microbatch"], settings["effective_batch_size"], start, stop)


def make_optimizer(model, cfg):
    from ispy2_symmflow.training.ema import ExponentialMovingAverage
    from ispy2_symmflow.training.engine import build_warmup_cosine_scheduler
    settings = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=settings["warmup_steps"],
                                              total_steps=settings["max_optimizer_steps"])
    return optimizer, scheduler, ExponentialMovingAverage(model, decay=settings["ema_decay"])


@torch.inference_mode()
def generate(cfg, inventory, model, dataset, record, codec):
    root = Path(cfg["output_root"])
    view = dataset.views[record["source_view"]]
    with source_read_guard(root, inventory, view) as audit:
        source = dataset.source(record)[None].to("cuda:0")
        generator = torch.Generator(device=source.device).manual_seed(cfg["seed"] + 1000003 * record["pair_index"])
        noise = torch.randn(source.shape, generator=generator, device=source.device, dtype=source.dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = model.sample(source, [record], noise, cfg["training"]["solver_steps"])
        prediction = codec.decode(dataset.denormalize(latent)).float()[0].cpu().numpy()
        reference = read_reference(root, view)
    if not np.isfinite(prediction).all() or audit["opened"] != audit["allowed"]:
        raise ValueError("Source-only generation or source-read audit failed")
    return prediction, reference, {"passed": True, "source_view": view["view_id"], "opened_source_files": len(audit["opened"]),
                                   "future_image_reads": 0}


@torch.inference_mode()
def evaluate(cfg, inventory, model, dataset, *, step, indices=None, ema=None, label="full", previews=False, progress=None):
    root = Path(cfg["output_root"])
    previous_mode, random_state = model.training, rng_state()
    model.eval()
    rows, exports, audits = [], [], []
    started = time.perf_counter()
    chosen = list(range(len(dataset))) if indices is None else list(indices)
    preview_indices = set(validation_indices(dataset.records, 6, cfg["seed"])) if previews else set()
    codec = None
    try:
        codec = SharedThreePhaseCodec(load_codec(cfg["codec_checkpoint"], "cuda:0"))
        with ema.average_parameters(model) if ema is not None else nullcontext():
            for ordinal, index in enumerate(chosen):
                record = dataset.records[index]
                prediction, source, audit = generate(cfg, inventory, model, dataset, record, codec)
                target_view = dataset.views[record["target_view"]]
                target = read_reference(root, target_view)
                latent = torch.from_numpy(read_latent(root, target_view))[None].to("cuda:0")
                reconstructed = codec.decode(latent).float()[0].cpu().numpy()
                source_support = source["support"][1]
                coverage = [float(mask[source_support].mean()) for mask in target["support"]]
                row = {"patient_id": record["patient_id"], "pair_id": record["pair_id"],
                       "transition": f"{record['earlier_stage']}->{record['later_stage']}",
                       "generation": phase_errors(prediction, target["images"], target["foreground"]),
                       "copy_source": phase_errors(source["images"], target["images"], target["foreground"]),
                       "target_codec": phase_errors(reconstructed, target["images"], target["foreground"]),
                       "minimum_coverage": min(coverage)}
                rows.append(row)
                audits.append(audit)
                if index in preview_indices:
                    stem = f"pair_{record['pair_index']:05d}_{record['earlier_stage']}_{record['later_stage']}"
                    folder = "smoke" if label == "smoke" else "selected"
                    path = root / "predictions" / folder / f"{stem}.npz"
                    disk_gate(root, required=2**29, reserve_gib=cfg["runtime"]["reserve_gib"])
                    atomic_arrays(path, prediction=prediction, source_support=source_support)
                    z = int(np.argmax(source["foreground"][1].sum(axis=(1, 2))))
                    render_comparison(root / "figures" / folder / f"{stem}.png",
                                      {"Real source": source["images"], "Generated": prediction,
                                       "Real target": target["images"], "Target VQ": reconstructed}, z=z,
                                      title=f"Three-phase Symm-FM {record['earlier_stage']} to {record['later_stage']}")
                    exports.append({"record": record, "prediction_file": str(path.relative_to(root)), "source_view": record["source_view"]})
                if progress and (ordinal == 0 or (ordinal + 1) % 25 == 0):
                    progress({"stage": "validating", "optimizer_step": step, "validation_kind": label,
                              "completed_pairs": ordinal + 1, "total_pairs": len(chosen),
                              "eta_seconds": (time.perf_counter() - started) / (ordinal + 1) * (len(chosen) - ordinal - 1)})
        result = {"schema": SCHEMA, "optimizer_step": step, "validation_kind": label, **summarize(rows),
                  "records": rows, "seconds": time.perf_counter() - started, "solver": "euler", "solver_steps": 20,
                  "selection_metric": "patient_macro_equal_phase_foreground_mae", "source_only_audits": audits,
                  "exports": exports, "completed_utc": now(), "independent_pcr_test": False}
        write_json(root / "evaluation" / f"{step:06d}_{label}.json", result)
        return result
    finally:
        model.train(previous_mode)
        restore_rng(random_state)
        del codec
        gc.collect()
        torch.cuda.empty_cache()


def save_state(path, root, contract, model, optimizer, scheduler, ema, step, best):
    atomic_checkpoint(path, {"schema": SCHEMA, "contract": contract, "optimizer_step": step,
                             "model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                             "scheduler": scheduler.state_dict(), "ema": cpu_tree(ema.state_dict()),
                             "rng": rng_state(), "best_score": best, "updated_utc": now()}, root)


def restore_state(path, contract, model, optimizer, scheduler, ema):
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if saved["schema"] != SCHEMA or public(saved["contract"]) != contract:
        raise ValueError("Formal resume contract differs; single-phase/pilot checkpoints are not accepted")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    ema.load_state_dict(saved["ema"])
    restore_rng(saved["rng"])
    return saved["optimizer_step"], saved["best_score"]


def smoke(cfg, inventory, progress):
    root = Path(cfg["output_root"])
    model, dataset, validation, contract = statistics_and_model(cfg, inventory)
    optimizer, scheduler, ema = make_optimizer(model, cfg)
    accumulation = cfg["training"]["effective_batch_size"] // cfg["training"]["microbatch"]
    torch.cuda.reset_peak_memory_stats()
    batches = loader(dataset, cfg, 0, 2)
    iterator = iter(batches)
    # Zero-initialized output layers can block upstream gradients on the first update.
    before = {name: parameter.detach().cpu().clone()
              for name, parameter in model.velocity_model.named_parameters() if parameter.requires_grad}
    first = train_step(model, optimizer, scheduler, ema, iterator, accumulation, "cuda:0")
    updated = [name for name, parameter in model.velocity_model.named_parameters()
               if name in before and not torch.equal(before[name], parameter.detach().cpu())]
    del before
    if not updated or first["gradient_norm"] <= 0:
        raise ValueError("Symm-FM smoke did not update velocity parameters")
    progress({"stage": "smoke_parameter_update_verified", "updated_velocity_parameters": updated, **first})
    path = root / "smoke" / "recovery.ckpt"
    save_state(path, root, contract, model, optimizer, scheduler, ema, 1, None)
    expected_metrics = train_step(model, optimizer, scheduler, ema, iterator, accumulation, "cuda:0")
    expected = copy.deepcopy(cpu_tree(model.state_dict()))
    expected_ema = copy.deepcopy(cpu_tree(ema.state_dict()))
    expected_scheduler = scheduler.state_dict()
    del iterator, batches
    restored_step, _ = restore_state(path, contract, model, optimizer, scheduler, ema)
    batches = loader(dataset, cfg, restored_step, 2)
    actual_metrics = train_step(model, optimizer, scheduler, ema, iter(batches), accumulation, "cuda:0")
    if actual_metrics != expected_metrics or scheduler.state_dict() != expected_scheduler:
        raise ValueError("GPU smoke optimizer/scheduler replay differs")
    for key, value in model.state_dict().items():
        if not torch.equal(value.cpu(), expected[key]):
            raise ValueError("GPU smoke parameter replay differs")
    for key, value in ema.state_dict()["shadow"].items():
        if not torch.equal(value.cpu(), expected_ema["shadow"][key]):
            raise ValueError("GPU smoke EMA replay differs")
    indices = validation_indices(validation.records, 6, cfg["seed"])
    result = evaluate(cfg, inventory, model, validation, step=2, indices=indices, ema=ema,
                      label="smoke", previews=True, progress=progress)
    report = {"passed": True, "contract": contract, "first_step": first, "replay_step": actual_metrics,
              "updated_velocity_parameters": updated,
              "exact_resume": True, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "validation_pairs": result["pairs"], "transitions": sorted(result["by_transition"]),
              "formal_optimizer_updates": 0, "completed_utc": now()}
    write_json(root / "smoke" / "report.json", report)
    del model, optimizer, scheduler, ema, expected, expected_ema, batches, dataset, validation
    gc.collect()
    torch.cuda.empty_cache()
    progress({"stage": "smoke_complete", "passed": True, "exact_resume": True})


def train(cfg, inventory, progress, *, resume=False, stop_after=None):
    root = Path(cfg["output_root"])
    settings = cfg["training"]
    model, dataset, validation, contract = statistics_and_model(cfg, inventory)
    smoke_report = read_json(root / "smoke" / "report.json")
    if not smoke_report["passed"] or smoke_report["contract"] != contract:
        raise ValueError("Matching real-GPU smoke is required")
    optimizer, scheduler, ema = make_optimizer(model, cfg)
    path = root / "checkpoints" / "recovery.ckpt"
    step, best = 0, None
    if resume:
        step, best = restore_state(path, contract, model, optimizer, scheduler, ema)
    elif path.exists():
        raise ValueError("Existing formal training requires --resume")
    batches = loader(dataset, cfg, step, settings["max_optimizer_steps"])
    iterator = iter(batches)
    accumulation = settings["effective_batch_size"] // settings["microbatch"]
    stopping = []
    previous = {sig: signal.signal(sig, lambda signum, frame: stopping.append(signum))
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)}
    initial, started, measured = step, time.perf_counter(), []
    light_indices = validation_indices(validation.records, settings["light_validation_pairs"], cfg["seed"])
    try:
        if not path.exists():
            save_state(path, root, contract, model, optimizer, scheduler, ema, step, best)
        while step < settings["max_optimizer_steps"] and not stopping:
            if stop_after is not None and step - initial >= stop_after:
                break
            began = time.perf_counter()
            metrics = train_step(model, optimizer, scheduler, ema, iterator, accumulation, "cuda:0")
            step += 1
            measured.append(time.perf_counter() - began)
            if step == 1 or step % 25 == 0:
                speed = float(np.median(measured[-100:]))
                row = {"stage": "training", "optimizer_step": step, "max_optimizer_steps": settings["max_optimizer_steps"],
                       **metrics, "learning_rate": optimizer.param_groups[0]["lr"], "seconds_per_update": speed,
                       "training_only_eta_seconds": speed * (settings["max_optimizer_steps"] - step),
                       "effective_batch": settings["effective_batch_size"], "microbatch": settings["microbatch"],
                       "elapsed_seconds": time.perf_counter() - started, "updated_utc": now()}
                with (root / "training_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                progress(row)
            full = step % settings["full_validation_interval"] == 0 or step == settings["max_optimizer_steps"]
            light = step % settings["light_validation_interval"] == 0
            if step == 1 or step % settings["checkpoint_interval"] == 0 or full or light:
                disk_gate(root, required=2 * 2**30, reserve_gib=cfg["runtime"]["reserve_gib"])
                save_state(path, root, contract, model, optimizer, scheduler, ema, step, best)
            if full or light:
                result = evaluate(cfg, inventory, model, validation, step=step, ema=ema,
                                  indices=None if full else light_indices, label="full" if full else "light", progress=progress)
                if full and (best is None or result["score"] < best):
                    best = result["score"]
                    with ema.average_parameters(model):
                        atomic_checkpoint(root / "checkpoints" / "best.ckpt", {"schema": SCHEMA, "contract": contract,
                                          "optimizer_step": step, "model": cpu_tree(model.state_dict()),
                                          "score": best, "evaluation_weights": "ema"}, root)
                save_state(path, root, contract, model, optimizer, scheduler, ema, step, best)
                progress({"stage": "validation_complete", "optimizer_step": step, "validation_kind": "full" if full else "light",
                          "score": result["score"], "best_score": best})
        save_state(path, root, contract, model, optimizer, scheduler, ema, step, best)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        del iterator, batches, model, optimizer, scheduler, ema, dataset, validation
        gc.collect()
        torch.cuda.empty_cache()
    if step < settings["max_optimizer_steps"]:
        progress({"stage": "paused", "optimizer_step": step, "best_score": best})
        return False
    write_json(root / "training_complete.json", {"schema": SCHEMA, "optimizer_steps": step, "best_score": best,
               "contract": contract, "completed_utc": now()})
    progress({"stage": "training_complete", "optimizer_step": step, "best_score": best})
    return True


def evaluate_selected(cfg, inventory, progress):
    root = Path(cfg["output_root"])
    model, training, validation, contract = statistics_and_model(cfg, inventory)
    saved = torch.load(root / "checkpoints" / "best.ckpt", map_location="cpu", weights_only=False, mmap=True)
    if saved["schema"] != SCHEMA or public(saved["contract"]) != contract:
        raise ValueError("Selected checkpoint differs from current data")
    model.load_state_dict(saved["model"], strict=True)
    result = evaluate(cfg, inventory, model, validation, step=saved["optimizer_step"], label="selected", previews=True, progress=progress)
    write_json(root / "sampling_report.json", {"schema": SCHEMA, "optimizer_step": saved["optimizer_step"],
               "exports": result["exports"], "source_only_audits_passed": True})
    del model, training, validation, saved
    gc.collect()
    torch.cuda.empty_cache()
    return result


def verify_pcr_interface(cfg, inventory, progress):
    root = Path(cfg["output_root"])
    sys.path.insert(0, cfg["pillar_repo"])
    from scripts.run_first_post_pcr import load_frozen_pillar, pillar_forward
    exports = read_json(root / "sampling_report.json")["exports"]
    views = {v["view_id"]: v for v in inventory["views"]}
    model = load_frozen_pillar()
    rows = []
    for export in exports:
        view = views[export["source_view"]]
        with source_read_guard(root, inventory, view):
            with np.load(root / export["prediction_file"], allow_pickle=False) as arrays:
                prediction, support = arrays["prediction"], arrays["source_support"]
            volume = build_generated_pillar_input(cfg, prediction, view["geometry"], support, inventory["image_normalization"])
            vector = pillar_forward(model, volume[None].to("cuda:0"))[0].detach().float().cpu()
        if vector.shape != (1152,) or not torch.isfinite(vector).all() or vector.norm() <= 0:
            raise ValueError("Invalid generated Pillar feature")
        path = root / "pcr_interface_check" / "embeddings" / (Path(export["prediction_file"]).stem + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vector, path)
        readback = torch.load(path, weights_only=True)
        if not torch.equal(vector, readback):
            raise ValueError("Pillar feature readback differs")
        rows.append({"pair_id": export["record"]["pair_id"], "feature_file": str(path.relative_to(root)),
                     "source_view": view["view_id"], "shape": [1152], "finite": True})
    if not rows:
        raise ValueError("No generated interface samples")
    report = {"passed": True, "features": len(rows), "records": rows, "input_shape": [1, 3, 384, 384, 192],
              "future_image_reads": 0, "pcr_performance_evaluated": False}
    write_json(root / "pcr_interface_check" / "report.json", report)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    progress({"stage": "pcr_interface_complete", "features": len(rows), "passed": True})
