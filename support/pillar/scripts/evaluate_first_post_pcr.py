"""Previous-real generation and a single frozen pCR evaluator for all four sources."""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import copy
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.data import EmbStore
from src.first_post_pcr_data import (
    SCHEMA, build_volume, disk_gate, embedding_path, identity, now, read_json,
    release_phases, save_tensor, stage_phases, validate_embedding, world_imports, write_json,
)
from src.metrics import METRIC_KEYS, compute_metrics


def routes_for(cohort, world_bundle, base_seed):
    visits = {v["visit_id"]: v for v in cohort["visits"]}
    by_patient = {(v["canonical_patient_id"], v["timepoint"]): v for v in cohort["visits"]}
    pairs = {(p["earlier_visit_id"], p["later_visit_id"]): p for p in world_bundle["pairs"] if p["split"] == "val"}
    result = []
    targets = sorted((v for v in visits.values() if v["split"] == "val" and v["timepoint"] > 0),
                     key=lambda v: (v["canonical_patient_id"], v["timepoint"]))
    for target in targets:
        source = by_patient[target["canonical_patient_id"], target["timepoint"] - 1]
        pair = pairs[source["visit_id"], target["visit_id"]]
        result.append({"patient_id": target["canonical_patient_id"], "source_visit_id": source["visit_id"],
                       "target_visit_id": target["visit_id"], "source_stage": source["timepoint"],
                       "target_stage": target["timepoint"], "pair_id": pair["pair_id"],
                       "noise_seed": int(base_seed) + len(result), "source_policy": "previous_real"})
    return result


def prediction_path(cfg, arm, route):
    return Path(cfg["output_dir"]) / "predictions" / arm / f"{route['patient_id']}_T{route['target_stage']}.pt"


def read_prediction(cfg, arm, route, snapshot):
    payload = torch.load(prediction_path(cfg, arm, route), map_location="cpu", weights_only=False)
    if (payload.get("schema") != SCHEMA or payload["route"] != route
            or payload["checkpoint_identity"] != snapshot["identity"]
            or payload["solver_steps"] != cfg["solver_steps"]):
        raise ValueError("Generated first-post cache contract differs")
    image = payload["image"]
    if image.shape != (96, 256, 256) or not torch.isfinite(image).all():
        raise FloatingPointError("Invalid decoded first-post cache")
    return image.numpy()


def world_batch(dataset, records):
    from mewm_ispy2.first_post_world_data import collate
    from mewm_ispy2.first_post_world_evaluation import to_device
    items = []
    for record in records:
        source = dataset.normalize(dataset.raw_latent(record["earlier_visit_id"]))
        item = {"source": source, "target": torch.zeros_like(source), "record": record}
        if dataset.arm == "bifm":
            item["source_mri"] = dataset.source_mri(record["earlier_visit_id"])
        items.append(item)
    # Target latent values are deliberately absent from the forecast input.
    return to_device(collate(items), "cuda:0")


@torch.inference_mode()
def sample_images(cfg, dataset, model, codec, records, seeds):
    from mewm_ispy2.first_post_world_evaluation import decode
    batch = world_batch(dataset, records)
    noises = [torch.randn(batch["source"].shape[1:], device="cuda:0", dtype=torch.float32,
                          generator=torch.Generator(device="cuda:0").manual_seed(seed)) for seed in seeds]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model.sample(batch, torch.stack(noises), cfg["solver_steps"])
    return decode(codec, dataset.denormalize(prediction)).cpu()[:, 0]


def profile_generation(cfg, dataset, model, codec, arm):
    from scripts.run_first_post_pcr import status
    path = Path(cfg["output_dir"]) / "profiles" / f"generation_{arm}.json"
    if path.exists():
        return read_json(path)["selected_batch_size"]
    records = [r for r in dataset.bundle["pairs"] if r["split"] == "train"]
    total = torch.cuda.get_device_properties(0).total_memory
    rows = []
    for batch in (1, 2, 4, 8):
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            images = sample_images(cfg, dataset, model, codec, records[:batch], list(range(92026, 92026 + batch)))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_reserved()
            rows.append({"batch_size": batch, "images_per_second": batch / elapsed,
                         "peak_reserved_gib": peak / 2**30,
                         "admitted": peak <= cfg["memory_fraction"] * total})
            del images
            status(cfg, "profiling_generation", arm=arm, measurement=rows[-1])
            if not rows[-1]["admitted"]:
                break
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch_size": batch, "admitted": False, "reason": "cuda_oom"})
            gc.collect()
            torch.cuda.empty_cache()
            break
    admitted = [r for r in rows if r["admitted"]]
    if not admitted:
        raise RuntimeError("No world-model inference batch fits the memory budget")
    best = max(admitted, key=lambda r: r["images_per_second"])
    write_json(path, {"selected_batch_size": best["batch_size"], "measurements": rows,
                      "profile_split": "train", "solver_steps": cfg["solver_steps"]})
    return best["batch_size"]


def generate(cfg, cohort):
    from scripts.run_first_post_pcr import status
    world_imports(cfg)
    from mewm_ispy2.first_post_world_data import PairDataset, load_codec
    from mewm_ispy2.first_post_world_models import build_model, restore_model
    output = Path(cfg["output_dir"])
    bundle = read_json(Path(cfg["world_run"]) / "shared/bundle.json")
    routes = routes_for(cohort, bundle, cfg["generation_seed"])
    path = output / "routes.json"
    if path.exists() and read_json(path) != routes:
        raise ValueError("Fixed previous-real routes changed")
    write_json(path, routes)
    snapshots = read_json(output / "world_checkpoints/snapshots.json")
    world_cfg = _release_yaml(Path(cfg["world_config"]).read_text())
    pair_index = {p["pair_id"]: p for p in bundle["pairs"]}
    for arm in ("symm", "bifm"):
        snapshot = snapshots[arm]
        if identity(snapshot["path"]) != snapshot["identity"]:
            raise ValueError("Pinned generator changed")
        pending = []
        for route in routes:
            if prediction_path(cfg, arm, route).exists():
                read_prediction(cfg, arm, route, snapshot)
            else:
                pending.append(route)
        if not pending:
            continue
        disk_gate(cfg, required=len(pending) * 96 * 256 * 256 * 4)
        dataset = PairDataset(cfg["world_run"], arm, "val")
        train_records = [r for r in bundle["pairs"] if r["split"] == "train"]
        model = build_model(world_cfg, arm, train_records, "cuda:0")
        checkpoint = torch.load(snapshot["path"], map_location="cpu", weights_only=False, mmap=True)
        restore_model(model, checkpoint["model"])
        model.eval().requires_grad_(False)
        del checkpoint
        codec = load_codec(Path(cfg["world_run"]) / "shared/codec.pt", "cuda:0")
        batch_size = profile_generation(cfg, dataset, model, codec, arm)
        started = time.perf_counter()
        done = len(routes) - len(pending)
        for offset in range(0, len(pending), batch_size):
            group = pending[offset:offset + batch_size]
            images = sample_images(cfg, dataset, model, codec, [pair_index[r["pair_id"]] for r in group],
                                   [r["noise_seed"] for r in group])
            for route, image in zip(group, images):
                save_tensor(prediction_path(cfg, arm, route), {
                    "schema": SCHEMA, "route": route, "checkpoint_identity": snapshot["identity"],
                    "optimizer_step": snapshot["optimizer_step"], "solver_steps": cfg["solver_steps"],
                    "image": image.clone(), "image_space": "fixed_vqgan_first_post_global_zscore"})
            done += len(group)
            elapsed = time.perf_counter() - started
            status(cfg, "generating_first_post", arm=arm, completed_targets=done, total_targets=len(routes),
                   batch_size=batch_size, optimizer_step=snapshot["optimizer_step"],
                   remaining_seconds=elapsed / (offset + len(group)) * (len(pending) - offset - len(group)))
            disk_gate(cfg)
        del dataset, codec, model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(output / "GENERATION_COMPLETE.json", {"schema": SCHEMA, "targets_per_model": len(routes),
               "completed_utc": now(), "future_target_values_in_forecast_conditions": False})


def extract_hybrid(cfg, cohort):
    from scripts.run_first_post_pcr import load_frozen_pillar, pillar_forward, status
    output = Path(cfg["output_dir"])
    if not (output / "GENERATION_COMPLETE.json").exists():
        raise ValueError("World generation is incomplete")
    routes = read_json(output / "routes.json")
    snapshots = read_json(output / "world_checkpoints/snapshots.json")
    visits = {v["visit_id"]: v for v in cohort["visits"]}
    batch_size = read_json(output / "profiles/pillar.json")["selected_batch_size"]
    model = load_frozen_pillar()
    completed = 0
    for offset in range(0, len(routes), cfg["staging_visits"]):
        chunk = routes[offset:offset + cfg["staging_visits"]]
        target_visits = [visits[r["target_visit_id"]] for r in chunk]
        stage_phases(cfg, target_visits)
        for source in ("symm", "bifm", "copy"):
            pending = []
            for route in chunk:
                path = embedding_path(cfg, source, visits[route["target_visit_id"]])
                if path.exists():
                    validate_embedding(path)
                    completed += 1
                else:
                    pending.append(route)
            for start in range(0, len(pending), batch_size):
                group = pending[start:start + batch_size]
                volumes = []
                for route in group:
                    target = visits[route["target_visit_id"]]
                    if source == "copy":
                        origin = visits[route["source_visit_id"]]
                        if identity(origin["image_path"]) != origin["crop_identity"]:
                            raise ValueError("Copy-source ROI changed")
                        replacement = np.load(origin["image_path"], allow_pickle=False).astype(np.float32)
                    else:
                        replacement = read_prediction(cfg, source, route, snapshots[source])
                    volume, _ = build_volume(cfg, target, replacement=replacement, audit=False,
                                             replacement_valid=(replacement != 0) if source == "copy" else None)
                    volumes.append(volume)
                vectors = pillar_forward(model, torch.stack(volumes).to("cuda:0"))
                for route, vector in zip(group, vectors):
                    save_tensor(embedding_path(cfg, source, visits[route["target_visit_id"]]), vector.clone())
                completed += len(group)
                status(cfg, "extracting_hybrid", source=source, completed_embeddings=completed,
                       total_embeddings=len(routes) * 3, batch_size=batch_size)
                del volumes, vectors
        release_phases(cfg, target_visits)
    write_json(output / "HYBRID_FEATURES_COMPLETE.json", {"schema": SCHEMA, "embeddings": completed,
               "completed_utc": now(), "replaced_channel": "first_post", "future_pre_late": "real"})
    del model
    gc.collect()
    torch.cuda.empty_cache()


def overlay(real, directory):
    result = copy.deepcopy(real)
    expected = {(pid, tp) for i, pid in enumerate(real["pids"]) for tp in range(1, 4) if real["masks"][i, tp]}
    actual = {(p.parent.name, int(p.stem.rsplit("_T", 1)[1])) for p in Path(directory).glob("*/*.pt")}
    if actual != expected:
        raise ValueError("Generated embedding inventory differs from the common heldout prefixes")
    store = EmbStore(str(directory))
    for i, pid in enumerate(real["pids"]):
        for tp in range(1, 4):
            if (pid, tp) in expected:
                result["embs"][i, tp] = store.load(pid, tp, require_finite=True)
    for key in ("masks", "days", "clinical", "labels"):
        if not np.array_equal(real[key], result[key]):
            raise ValueError("Hybrid overlay changed fixed evaluation metadata")
    if not np.array_equal(real["embs"][:, 0], result["embs"][:, 0]):
        raise ValueError("Hybrid overlay changed real T0")
    return result


def oof_threshold(labels, probabilities):
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    scores = (tpr + 1 - fpr) / 2
    eligible = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12) & np.isfinite(thresholds))
    if not len(eligible):
        raise ValueError("No finite OOF threshold")
    index = eligible[np.argmin(np.abs(thresholds[eligible] - 0.5))]
    return float(thresholds[index])


def paired_bootstrap(predictions, cfg):
    from scripts.run_first_post_pcr import status
    rows = []
    rng = np.random.default_rng(cfg["fold_seed"])
    for depth, group in predictions.groupby("temporal_depth", sort=False):
        ids = sorted(group.patient_id.unique())
        labels = group.drop_duplicates("patient_id").set_index("patient_id").loc[ids, "label"].to_numpy()
        matrices = {source: frame.pivot(index="patient_id", columns="seed", values="probability").loc[ids].to_numpy().T
                    for source, frame in group.groupby("source")}
        positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
        if not len(positive) or not len(negative):
            raise ValueError("Paired bootstrap requires both pCR classes")
        measures = {s: np.empty((cfg["bootstrap_samples"], 2)) for s in matrices}
        for i in range(cfg["bootstrap_samples"]):
            indices = np.concatenate((rng.choice(positive, len(positive), replace=True),
                                      rng.choice(negative, len(negative), replace=True)))
            target = labels[indices]
            for source, matrix in matrices.items():
                measures[source][i] = np.mean([[roc_auc_score(target, p[indices]),
                                                average_precision_score(target, p[indices])] for p in matrix], axis=0)
        point = {s: np.mean([[roc_auc_score(labels, p), average_precision_score(labels, p)] for p in matrix], axis=0)
                 for s, matrix in matrices.items()}
        for left, right in (("symm", "real"), ("bifm", "real"), ("copy", "real"),
                            ("symm", "copy"), ("bifm", "copy"), ("symm", "bifm")):
            for index, metric in enumerate(("auroc", "prauc")):
                low, high = np.quantile((measures[left] - measures[right])[:, index], [0.025, 0.975])
                rows.append({"temporal_depth": depth, "comparison": f"{left}_minus_{right}", "metric": metric,
                             "difference": point[left][index] - point[right][index], "ci95_low": low,
                             "ci95_high": high, "patients": len(ids), "bootstrap_samples": cfg["bootstrap_samples"]})
        status(cfg, "paired_bootstrap", completed_depth=depth, bootstrap_samples=cfg["bootstrap_samples"])
    return pd.DataFrame(rows)


def evaluate(cfg, cohort):
    from scripts.run_first_post_pcr import status
    from scripts.run_full978_anti_overfit import _load_split, _prior_from_state
    from scripts.run_full978_independent_cv import _canonical_split, _prediction_frame, _training_dir
    from src.tdn import TDN
    output = Path(cfg["output_dir"])
    if not (output / "TRAINING_COMPLETE.json").exists() or not (output / "HYBRID_FEATURES_COMPLETE.json").exists():
        raise ValueError("pCR training or hybrid feature extraction is incomplete")
    heldout = cohort["split"]["val"]
    real = _load_split(output / "embeddings/real", output / "metadata_enriched.csv", heldout)
    splits = {"real": real, **{source: overlay(real, output / "embeddings" / source) for source in ("symm", "bifm", "copy")}}
    predictions, threshold_rows, replay = [], [], []
    for depth, spec in cfg["recipes"].items():
        depth_spec = {"name": depth, "max_tp": spec["max_tp"]}
        canonical = {s: _canonical_split(value, spec["max_tp"]) for s, value in splits.items()}
        for seed in cfg["seeds"]:
            by_source = {s: [] for s in splits}
            oof = []
            for fold in range(5):
                directory = _training_dir(output / "tdn", "formal", depth, "prespecified", seed, fold)
                checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
                trained = set(checkpoint["train_ids"])
                validated = set(checkpoint["validation_ids"])
                if (trained & validated or (trained | validated) != set(cohort["split"]["train"])
                        or (trained | validated) & set(heldout)):
                    raise ValueError("Classifier checkpoint has incompatible patient boundaries")
                if (checkpoint["temporal_depth"], checkpoint["seed"], checkpoint["fold"]) != (depth, seed, fold):
                    raise ValueError("Classifier checkpoint identity differs")
                model = TDN({"downstream": checkpoint["effective_config"]}).to("cuda:0").eval().requires_grad_(False)
                model.load_state_dict(checkpoint["model_state"], strict=True)
                val = _canonical_split(_load_split(output / "embeddings/real", output / "metadata_enriched.csv",
                                                   checkpoint["validation_ids"]), spec["max_tp"])
                prior = _prior_from_state(val["clinical"], checkpoint["clinical_prior"])
                fresh = _prediction_frame(model, val, prior, depth_spec, seed, fold, "oof_val", "cuda:0", 64)
                saved = pd.read_csv(directory / "val_predictions.csv").set_index("patient_id").loc[fresh.patient_id]
                difference = float(np.max(np.abs(fresh.probability.to_numpy() - saved.probability.to_numpy())))
                if difference > 1e-5 or not np.array_equal(fresh.label.to_numpy(), saved.label.to_numpy()):
                    raise ValueError("Reloaded classifier does not reproduce saved OOF predictions")
                replay.append({"depth": depth, "seed": seed, "fold": fold, "max_probability_difference": difference})
                oof.append(fresh)
                for source, values in canonical.items():
                    prior = _prior_from_state(values["clinical"], checkpoint["clinical_prior"])
                    frame = _prediction_frame(model, values, prior, depth_spec, seed, fold, "world_holdout", "cuda:0", 64)
                    frame["source"] = source
                    by_source[source].append(frame)
                del model
            out_of_fold = pd.concat(oof, ignore_index=True)
            if out_of_fold.patient_id.duplicated().any() or set(out_of_fold.patient_id) != set(cohort["split"]["train"]):
                raise ValueError("OOF probabilities do not cover the training pool exactly once")
            threshold = oof_threshold(out_of_fold.label, out_of_fold.probability)
            threshold_rows.append({"temporal_depth": depth, "seed": seed, "threshold": threshold,
                                   "fitted_on": "training_pool_oof_only"})
            means = {}
            for source, frames in by_source.items():
                combined = pd.concat(frames, ignore_index=True)
                mean = combined.groupby("patient_id", sort=False).probability.mean().reindex(heldout).to_numpy()
                means[source] = mean
                predictions.append(pd.DataFrame({"patient_id": heldout, "label": real["labels"].astype(int),
                    "probability": mean, "source": source, "temporal_depth": depth, "seed": seed,
                    "threshold": threshold, "complete_window": real["masks"][:, :spec["max_tp"]].all(axis=1)}))
                target = output / "evaluation/fold_predictions" / f"{depth}_{seed}_{source}.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                combined.to_csv(target, index=False)
            if depth == "T0" and any(not np.array_equal(means["real"], p) for p in means.values()):
                raise ValueError("T0 prediction differs between sources")
            status(cfg, "evaluating_pcr", temporal_depth=depth, seed=seed)
    frame = pd.concat(predictions, ignore_index=True)
    results = output / "evaluation"
    frame.to_csv(results / "predictions.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(results / "development_thresholds.csv", index=False)
    write_json(results / "checkpoint_replay.json", {"passed": True, "fits": len(replay), "records": replay})
    metric_rows = []
    for (source, depth, seed), values in frame.groupby(["source", "temporal_depth", "seed"]):
        for population, selected in (("all_prefixes", values), ("complete_window", values[values.complete_window])):
            for policy, threshold in (("development_oof_bacc", float(values.threshold.iloc[0])), ("fixed_0.5", 0.5)):
                metric_rows.append({"source": source, "temporal_depth": depth, "seed": seed,
                    "population": population, "patients": len(selected), "threshold_policy": policy,
                    "threshold": threshold, **compute_metrics(selected.label, selected.probability, threshold)})
    per_seed = pd.DataFrame(metric_rows)
    per_seed.to_csv(results / "metrics_per_seed.csv", index=False)
    summary_rows = []
    for keys, values in per_seed.groupby(["source", "temporal_depth", "population", "threshold_policy"]):
        row = dict(zip(("source", "temporal_depth", "population", "threshold_policy"), keys))
        row.update(patients=int(values.patients.iloc[0]), seeds=len(values))
        for metric in METRIC_KEYS:
            row[metric + "_mean"] = float(values[metric].mean())
            row[metric + "_std"] = float(values[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results / "summary.csv", index=False)
    paired_bootstrap(frame, cfg).to_csv(results / "paired_differences.csv", index=False)
    main = summary[(summary.population == "all_prefixes") & (summary.threshold_policy == "development_oof_bacc")]
    lines = ["# First-post pCR comparison", "", "Frozen real-only five-fold classifiers; ten seeds per depth.", "",
             "This is retrospective hybrid-input evaluation on world-model selection validation patients, not an independent test.",
             "Future pre/late and target crop geometry are real. Generator training budgets differ.", "",
             "| Source | Window | Patients | AUROC mean (SD) | PR-AUC mean (SD) |", "|---|---|---:|---:|---:|"]
    for r in main.itertuples():
        lines.append(f"| {r.source} | {r.temporal_depth} | {r.patients} | {r.auroc_mean:.4f} ({r.auroc_std:.4f}) | {r.prauc_mean:.4f} ({r.prauc_std:.4f}) |")
    (results / "report.md").write_text("\n".join(lines) + "\n")
    write_json(output / "EVALUATION_COMPLETE.json", {"schema": SCHEMA, "completed_utc": now(),
               "patients": len(heldout), "classifier_fits": len(replay), "sources": list(splits),
               "world_checkpoints": read_json(output / "world_checkpoints/snapshots.json"),
               "classifier_replay_passed": True, "bootstrap_samples": cfg["bootstrap_samples"]})
