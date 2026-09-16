"""Run the isolated first-post pCR preparation, fixed recipes, and paired evaluation."""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import argparse
import fcntl
import gc
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import yaml

from src.first_post_pcr_data import (
    ROOT, SCHEMA, build_volume, disk_gate, embedding_path, identity, load_config, now,
    prepare, read_json, release_phases, repo_path, save_tensor, stage_phases,
    validate_embedding, world_imports, write_json,
)


def status(cfg, stage, **fields):
    value = {"schema": SCHEMA, "stage": stage, "updated_utc": now(), "physical_gpu": cfg["gpu"], **fields}
    write_json(Path(cfg["output_dir"]) / "progress.json", value)
    print(json.dumps(value, allow_nan=False), flush=True)


def snapshot_world(cfg):
    output = Path(cfg["output_dir"])
    destination = output / "world_checkpoints"
    destination.mkdir(exist_ok=True)
    manifest = destination / "snapshots.json"
    if manifest.exists():
        result = read_json(manifest)
        for row in result.values():
            if identity(row["path"]) != row["identity"]:
                raise ValueError("Pinned world-model checkpoint changed")
        return result
    result = {}
    for arm in ("symm", "bifm"):
        source = Path(cfg["world_run"]) / arm / "train/checkpoints/best.ckpt"
        path = destination / f"{arm}.ckpt"
        disk_gate(cfg, source.stat().st_size)
        tmp = path.with_suffix(".tmp")
        before = identity(source)
        shutil.copyfile(source, tmp)
        if identity(source) != before:
            tmp.unlink()
            raise RuntimeError("World-model best checkpoint changed during archival; restart preparation")
        payload = torch.load(tmp, map_location="cpu", mmap=True, weights_only=False)
        if payload.get("schema") != "first_post_unregistered_tumor_roi_v1":
            raise ValueError("Unexpected world checkpoint schema")
        step = int(payload["optimizer_step"])
        tmp.replace(path)
        result[arm] = {"path": str(path), "identity": identity(path), "optimizer_step": step,
                       "evaluation_weights": payload.get("evaluation_weights"),
                       "foreground_mae_at_selection": payload["best_foreground_mae"]}
        del payload
    write_json(manifest, result)
    status(cfg, "world_snapshots_pinned", checkpoints={k: v["optimizer_step"] for k, v in result.items()})
    return result


def recipe_configs(cfg):
    from scripts.run_full978_independent_cv import _effective_config
    output = Path(cfg["output_dir"])
    result = {}
    for depth, spec in cfg["recipes"].items():
        original = _release_yaml(repo_path(spec["path"]).read_text())
        effective = _effective_config(original, spec["variant"], depth)
        config = {"data": {"metadata_csv": str(output / "metadata_enriched.csv"),
                            "train_embeddings_dir": str(output / "embeddings/real"), "embedding_dim": 1152},
                  "downstream": effective, "variants": {"prespecified": {}},
                  "independent_cv": {"temporal_depths": [{"name": depth, "max_tp": spec["max_tp"]}]}}
        path = output / "configs" / f"{depth.lower().replace('-', '_')}.yaml"
        path.parent.mkdir(exist_ok=True)
        if path.exists() and _release_yaml(path.read_text()) != config:
            raise ValueError("Frozen pCR recipe changed")
        if not path.exists():
            path.write_text(yaml.safe_dump(config, sort_keys=False))
        result[depth] = str(path)
    return result


def pin_contract(cfg):
    from huggingface_hub.constants import HF_HUB_CACHE
    output = Path(cfg["output_dir"])
    cache = Path(HF_HUB_CACHE) / "models--YalaLab--Pillar0-BreastMRI"
    revision = (cache / "refs/main").read_text().strip()
    snapshot = cache / "snapshots" / revision
    files = {p.name: identity(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
    if not files:
        raise FileNotFoundError("Cached pretrained Pillar files are absent")
    sources = [ROOT / "src/pillar.py", ROOT / "src/tdn.py", ROOT / "src/data.py",
               ROOT / "scripts/run_full978_independent_cv.py", ROOT / "scripts/run_full978_anti_overfit.py",
               ROOT / "src/first_post_pcr_data.py", ROOT / "scripts/run_first_post_pcr.py",
               ROOT / "scripts/evaluate_first_post_pcr.py"]
    expected = {"schema": SCHEMA, "config": cfg, "pillar_model": "YalaLab/Pillar0-BreastMRI",
                "pillar_files": files, "runtime_sources": {str(p.relative_to(ROOT)): identity(p) for p in sources},
                "normalization": {"first_post_mean": 142.555409, "first_post_std": 283.804038},
                "spatial_policy": "native_lps_then_shared_roi_then_1mm_center_pad_crop",
                "fit_policy": "real_only_5fold_10seeds_four_prespecified_depths"}
    path = output / "contract.json"
    if path.exists():
        if read_json(path) != expected:
            raise ValueError("First-post pCR run contract changed")
    else:
        write_json(path, expected)


@torch.inference_mode()
def pillar_forward(model, volume):
    result = model.extract_vision_feats({"breast_mr": volume})
    if not isinstance(result, torch.Tensor) or result.shape != (volume.shape[0], 1152):
        raise ValueError("Pillar returned an unexpected batched embedding shape")
    if not torch.isfinite(result).all() or (result.norm(dim=1) <= 0).any():
        raise FloatingPointError("Pillar returned invalid embeddings")
    return result.float().cpu()


def load_frozen_pillar():
    from src.pillar import load_pillar
    model = load_pillar("cuda:0").float().requires_grad_(False).eval()
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Pillar is not frozen")
    return model


def profile_pillar(cfg, model, visit):
    path = Path(cfg["output_dir"]) / "profiles/pillar.json"
    if path.exists():
        return read_json(path)
    stage_phases(cfg, [visit])
    volume, _ = build_volume(cfg, visit)
    baseline = pillar_forward(model, volume[None].to("cuda:0"))
    rows = []
    total = torch.cuda.get_device_properties(0).total_memory
    for batch in cfg["extraction_batch_candidates"]:
        images = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            images = volume[None].expand(batch, -1, -1, -1, -1).contiguous().to("cuda:0")
            pillar_forward(model, images)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(3):
                result = pillar_forward(model, images)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_reserved()
            difference = float((result - baseline).abs().max())
            if not torch.allclose(result, baseline.expand_as(result), atol=1e-4, rtol=1e-4):
                raise ValueError("Pillar batching changed embeddings beyond tolerance")
            rows.append({"batch_size": batch, "visits_per_second": 3 * batch / elapsed,
                         "peak_reserved_gib": peak / 2**30, "batch_replay_max_difference": difference,
                         "admitted": peak <= total * cfg["memory_fraction"]})
            status(cfg, "profiling_pillar", measurement=rows[-1])
            if not rows[-1]["admitted"]:
                break
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch_size": batch, "admitted": False, "reason": "cuda_oom"})
            break
        finally:
            del images
            gc.collect()
            torch.cuda.empty_cache()
    admitted = [row for row in rows if row["admitted"]]
    if not admitted:
        raise RuntimeError("No Pillar extraction batch fits the memory budget")
    best = max(admitted, key=lambda row: row["visits_per_second"])
    result = {"selected_batch_size": best["batch_size"], "measurements": rows,
              "precision": "float32", "created_utc": now()}
    write_json(path, result)
    return result


def staged_chunks(cfg, visits):
    chunks = [visits[i:i + cfg["staging_visits"]] for i in range(0, len(visits), cfg["staging_visits"])]
    with ThreadPoolExecutor(max_workers=1) as transfer:
        pending = transfer.submit(stage_phases, cfg, chunks[0]) if chunks else None
        for index, chunk in enumerate(chunks):
            pending.result()
            if index + 1 < len(chunks) and cfg.get("phase_prefetch_batches", 0):
                pending = transfer.submit(stage_phases, cfg, chunks[index + 1])
            yield chunk
            release_phases(cfg, chunk)
            if index + 1 < len(chunks) and not cfg.get("phase_prefetch_batches", 0):
                pending = transfer.submit(stage_phases, cfg, chunks[index + 1])


def extract_real(cfg, cohort):
    output = Path(cfg["output_dir"])
    if (output / "REAL_FEATURES_COMPLETE.json").exists():
        for visit in cohort["visits"]:
            validate_embedding(embedding_path(cfg, "real", visit))
        return
    status(cfg, "loading_pillar", purpose="real_feature_extraction", total_visits=len(cohort["visits"]))
    model = load_frozen_pillar()
    train_visit = next(v for v in cohort["visits"] if v["split"] == "train")
    profile = profile_pillar(cfg, model, train_visit)
    batch_size = profile["selected_batch_size"]
    visits = sorted(cohort["visits"], key=lambda v: (v["split"] != "train", v["canonical_patient_id"], v["timepoint"]))
    started, initial_completed = time.perf_counter(), 0
    pending = []
    for v in visits:
        path = embedding_path(cfg, "real", v)
        audit_path = output / "geometry_audit" / f"{v['canonical_patient_id']}_{v['visit']}.json"
        if path.exists() and audit_path.exists():
            validate_embedding(path)
            initial_completed += 1
        else:
            pending.append(v)
    completed = initial_completed
    for chunk in staged_chunks(cfg, pending):
        with ThreadPoolExecutor(max_workers=cfg["preprocess_workers"]) as pool:
            # At most two batches are prepared ahead; full Pillar volumes are large.
            queue = {}
            submitted = 0
            while submitted < min(len(chunk), batch_size * cfg["prefetch_batches"]):
                queue[submitted] = pool.submit(build_volume, cfg, chunk[submitted])
                submitted += 1
            for offset in range(0, len(chunk), batch_size):
                group = chunk[offset:offset + batch_size]
                built = [queue.pop(i).result() for i in range(offset, offset + len(group))]
                while submitted < min(len(chunk), offset + len(group) + batch_size * cfg["prefetch_batches"]):
                    queue[submitted] = pool.submit(build_volume, cfg, chunk[submitted])
                    submitted += 1
                images = torch.stack([item[0] for item in built]).to("cuda:0")
                embeddings = pillar_forward(model, images)
                for visit, vector, (_, audit) in zip(group, embeddings, built):
                    save_tensor(embedding_path(cfg, "real", visit), vector.clone())
                    write_json(output / "geometry_audit" / f"{visit['canonical_patient_id']}_{visit['visit']}.json", audit)
                completed += len(group)
                elapsed = time.perf_counter() - started
                rate = (completed - initial_completed) / max(elapsed, 1e-9)
                status(cfg, "extracting_real", completed_visits=completed, total_visits=len(visits),
                       batch_size=batch_size, visits_per_second=rate,
                       remaining_seconds=(len(visits) - completed) / max(rate, 1e-9))
                del images, built, embeddings
        disk_gate(cfg)
    audits = [read_json(output / "geometry_audit" / f"{v['canonical_patient_id']}_{v['visit']}.json") for v in visits]
    ratios = [r["tumor_retention"] for r in audits if r["tumor_retention"] is not None]
    write_json(output / "REAL_FEATURES_COMPLETE.json", {"schema": SCHEMA, "visits": completed,
               "completed_utc": now(), "nonempty_masks": len(ratios),
               "visits_with_tumor_truncation": sum(x < 1 for x in ratios),
               "minimum_tumor_retention": min(ratios) if ratios else None})
    del model
    gc.collect()
    torch.cuda.empty_cache()


def fold_task(task):
    from scripts.run_full978_independent_cv import _train_fold_task, _training_dir
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    message = _train_fold_task(task)
    _, output, stage, depth, variant, seed, spec, _, _, _ = task
    run = _training_dir(output, stage, depth["name"], variant, seed, spec["fold"])
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if any(not torch.isfinite(value).all() for value in checkpoint["model_state"].values()):
        raise FloatingPointError("Non-finite pCR checkpoint")
    history = pd.read_csv(run / "history.csv")
    if not np.isfinite(history.select_dtypes(include="number").to_numpy()).all():
        raise FloatingPointError("Non-finite pCR training history")
    return {"message": message, "seconds": time.perf_counter() - started,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "epochs": len(history), "best_epoch": checkpoint["selection"]["best_epoch_zero_based"] + 1,
            "depth": depth["name"], "seed": seed, "fold": spec["fold"]}


def training_tasks(cfg, configs, folds, *, smoke=False, count=None, output=None):
    tasks = []
    for depth, spec in cfg["recipes"].items():
        for seed in cfg["seeds"]:
            for fold in folds:
                tasks.append((configs[depth], output or str(Path(cfg["output_dir"]) / "tdn"),
                              "formal", {"name": depth, "max_tp": spec["max_tp"]}, "prespecified",
                              int(seed), fold, "cuda:0", smoke, False))
    return tasks if count is None else tasks[:count]


def profile_training(cfg, configs, folds):
    output = Path(cfg["output_dir"])
    path = output / "profiles/training.json"
    if path.exists():
        return read_json(path)
    records = []
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    for jobs in cfg["training_jobs_candidates"]:
        profile_dir = output / "profiles" / f"train_jobs_{jobs}"
        tasks = [task for task in training_tasks(cfg, configs, folds, smoke=True, output=str(profile_dir))
                 if task[3]["name"] == "T0-T3"][:jobs]
        stopped = threading.Event()
        memory = {"peak_gib": 0.0}

        def measure_memory():
            while not stopped.is_set():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                memory["peak_gib"] = max(memory["peak_gib"], (total_bytes - free_bytes) / 2**30)
                stopped.wait(0.05)

        monitor = threading.Thread(target=measure_memory, daemon=True)
        monitor.start()
        started = time.perf_counter()
        try:
            with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as pool:
                results = list(pool.map(fold_task, tasks))
        finally:
            stopped.set()
            monitor.join()
        seconds = time.perf_counter() - started
        peak = memory["peak_gib"]
        records.append({"jobs": jobs, "seconds": seconds, "fits_per_second": jobs / seconds,
                        "measured_device_peak_gib": peak, "profile_depth": "T0-T3",
                        "admitted": peak <= total * cfg["memory_fraction"], "results": results})
        status(cfg, "profiling_training", jobs=jobs, fits_per_second=jobs / seconds, measured_peak_gib=peak)
        if not records[-1]["admitted"]:
            break
    admitted = [r for r in records if r["admitted"]]
    if not admitted:
        raise RuntimeError("No pCR concurrency level fits the memory budget")
    best = max(admitted, key=lambda r: r["fits_per_second"])
    result = {"selected_jobs": best["jobs"], "measurements": records, "batch_size": 64,
              "precision": "float32", "created_utc": now()}
    write_json(path, result)
    return result


def train(cfg, cohort):
    output = Path(cfg["output_dir"])
    if not (output / "REAL_FEATURES_COMPLETE.json").exists():
        raise ValueError("Real feature extraction has not completed")
    configs = recipe_configs(cfg)
    folds = read_json(output / "folds.json")
    heldout = set(cohort["split"]["val"])
    pool_ids = set(cohort["split"]["train"])
    for fold in folds:
        a, b = set(fold["train_ids"]), set(fold["val_ids"])
        if a & b or (a | b) != pool_ids or (a | b) & heldout:
            raise ValueError("Invalid pCR fold boundaries")
    profile = profile_training(cfg, configs, folds)
    tasks = training_tasks(cfg, configs, folds)
    status(cfg, "training_pcr", completed_fits=0, total_fits=len(tasks), jobs=profile["selected_jobs"])
    started, records = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=profile["selected_jobs"], mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(fold_task, task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            elapsed = time.perf_counter() - started
            status(cfg, "training_pcr", completed_fits=len(records), total_fits=len(tasks),
                   jobs=profile["selected_jobs"], latest=records[-1],
                   estimated_remaining_seconds=elapsed / len(records) * (len(tasks) - len(records)))
            disk_gate(cfg)
    write_json(output / "TRAINING_COMPLETE.json", {"schema": SCHEMA, "fits": len(records),
               "completed_utc": now(), "results": records, "fit_images": "real_only",
               "heldout_used_for_fitting": False})


def check_idle_gpu(gpu):
    result = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    if int(result.stdout.strip()) > 1024:
        raise RuntimeError(f"GPU{gpu} is occupied; existing processes were not changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/first_post_pcr_v1.yaml")
    parser.add_argument("--stage", choices=("prepare", "extract", "train", "generate", "hybrid", "evaluate", "all"), default="all")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        if args.stage != "prepare":
            check_idle_gpu(cfg["gpu"])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["gpu"]), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(repo_path(args.config)),
                   "--stage", args.stage]
        with (output / "workflow.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(output / "launch.json", {"pid": process.pid, "gpu": cfg["gpu"], "stage": args.stage,
                   "created_utc": now(), "log": str(output / "workflow.log")})
        print(json.dumps({"pid": process.pid, "gpu": cfg["gpu"], "log": str(output / "workflow.log")}), flush=True)
        return
    if args.stage != "prepare" and os.environ.get("CUDA_VISIBLE_DEVICES") != str(cfg["gpu"]):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the configured free GPU, or use --detach")
    with (output / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            import SimpleITK as sitk
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
            world_imports(cfg)
            pin_contract(cfg)
            snapshot_world(cfg)
            cohort = prepare(cfg)
            recipe_configs(cfg)
            if args.stage in ("extract", "all"):
                extract_real(cfg, cohort)
            if args.stage in ("train", "all"):
                train(cfg, cohort)
            if args.stage in ("generate", "hybrid", "evaluate", "all"):
                from scripts.evaluate_first_post_pcr import generate, extract_hybrid, evaluate
                if args.stage in ("generate", "all"):
                    generate(cfg, cohort)
                if args.stage in ("hybrid", "all"):
                    extract_hybrid(cfg, cohort)
                if args.stage in ("evaluate", "all"):
                    evaluate(cfg, cohort)
            status(cfg, "complete" if args.stage == "all" else f"{args.stage}_complete")
        except BaseException as error:
            status(cfg, "failed", failed_stage=args.stage, error_type=type(error).__name__, error=str(error))
            raise


if __name__ == "__main__":
    main()
