"""Train and evaluate the full978 shared-prefix anti-overfit experiment.

Training uses only the 876-patient development pool. A model is shared across
all four temporal prefixes, with one uniformly sampled prefix per patient and
batch. Checkpoints are selected by the mean validation AUROC over all four
fixed prefixes. The locked 102-patient test data is loaded only in the evaluate
phase, after a variant has been selected from five-fold OOF predictions.
"""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import EmbStore, TABULAR_FEATURE_NAMES, load_ids, load_split
from src.metrics import METRIC_KEYS, compute_metrics
from src.tdn import TDN
from src.temporal import mask_torch_temporal_prefix


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pillar_full978_shared_prefix_anti_overfit_v1"


def _repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _atomic_json(path, payload):
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _read_metadata(path, allowed_ids=None):
    if allowed_ids is None:
        frame = pd.read_csv(path, dtype={"pid": str})
    else:
        allowed = set(allowed_ids)
        try:
            pid_frame = pd.read_csv(path, usecols=["pid"], dtype={"pid": str})
        except ValueError as error:
            if "Usecols do not match columns" in str(error):
                raise ValueError(
                    "metadata must contain unique pid values and pCR labels"
                ) from error
            raise
        if "pid" not in pid_frame or pid_frame["pid"].duplicated().any():
            raise ValueError("metadata must contain unique pid values and pCR labels")
        available = set(pid_frame["pid"])
        if not allowed.issubset(available):
            missing = sorted(allowed - available)
            raise ValueError(f"metadata is missing patient {missing[0]}")
        allowed_rows = {
            row_number + 1
            for row_number, patient_id in enumerate(pid_frame["pid"])
            if patient_id in allowed
        }

        def skip_unallowed_rows(row_number):
            return row_number != 0 and row_number not in allowed_rows

        frame = pd.read_csv(
            path, dtype={"pid": str}, skiprows=skip_unallowed_rows
        )
    if {"pid", "pCR"} - set(frame.columns) or frame["pid"].duplicated().any():
        raise ValueError("metadata must contain unique pid values and pCR labels")
    if allowed_ids is not None:
        if set(frame["pid"]) != allowed:
            missing = sorted(allowed - set(frame["pid"]))
            raise ValueError(f"metadata is missing patient {missing[0]}")
    labels = frame["pCR"].astype(float)
    if not labels.isin([0.0, 1.0]).all():
        raise ValueError("pCR labels must be binary")
    return frame


def _metadata_map(frame, allowed_ids):
    allowed = set(allowed_ids)
    selected = frame[frame["pid"].isin(allowed)]
    if set(selected["pid"]) != allowed:
        missing = sorted(allowed - set(selected["pid"]))
        raise ValueError(f"metadata is missing patient {missing[0]}")
    return {str(row["pid"]): row for _, row in selected.iterrows()}


def _load_split(embeddings_dir, metadata_csv, patient_ids):
    metadata = _read_metadata(metadata_csv, patient_ids)
    mapping = _metadata_map(metadata, patient_ids)
    clinical_dim = len(TABULAR_FEATURE_NAMES)
    return load_split(EmbStore(str(embeddings_dir)), patient_ids, mapping, clinical_dim)


def _fold_specs(config):
    data_cfg = config["data"]
    anti = config["anti_overfit"]
    cohort = _repo_path(data_cfg["cohort_dir"])
    original = {
        split: load_ids(cohort / "splits" / f"{split}_ids.txt")
        for split in ("train", "val", "test")
    }
    sets = {name: set(values) for name, values in original.items()}
    if sets["train"] & sets["val"] or (sets["train"] | sets["val"]) & sets["test"]:
        raise ValueError("original train/validation/test splits overlap")
    pool = sorted(original["train"] + original["val"])
    if len(pool) != int(data_cfg["expected_pool_patients"]):
        raise ValueError("development-pool patient count mismatch")
    if len(original["test"]) != int(data_cfg["expected_test_patients"]):
        raise ValueError("locked-test patient count mismatch")
    reference_path = data_cfg.get("locked_test_reference_json")
    if reference_path:
        reference = json.loads(_repo_path(reference_path).read_text())
        reference_split = str(data_cfg.get("locked_test_reference_split", "val"))
        reference_ids = [str(value) for value in reference.get(reference_split, [])]
        if (
            len(reference_ids) != len(set(reference_ids))
            or set(reference_ids) != sets["test"]
        ):
            raise ValueError("locked test does not exactly match the BiFlow reference split")
    metadata = _read_metadata(_repo_path(data_cfg["metadata_csv"]), pool)
    manifest = pd.read_csv(cohort / "patient_manifest.csv", dtype={"patient_id": str})
    if {"patient_id", "source"} - set(manifest.columns) or manifest["patient_id"].duplicated().any():
        raise ValueError("patient_manifest.csv must contain unique patient IDs and source")
    development = metadata[metadata["pid"].isin(set(pool))].merge(
        manifest[["patient_id", "source"]], left_on="pid", right_on="patient_id",
        how="left", validate="one_to_one",
    ).set_index("pid").loc[pool].reset_index()
    if development[["pCR", "HR_HER2_STATUS", "source", "n_registered_timepoints"]].isna().any().any():
        raise ValueError("fold-stratification fields contain missing values")
    development["base_stratum"] = (
        development["pCR"].astype(int).astype(str)
        + "|" + development["HR_HER2_STATUS"].astype(str)
        + "|" + development["source"].astype(str)
    )
    labels = development["base_stratum"].to_numpy()
    folds = int(anti["folds"])
    if folds != 5:
        raise ValueError("this formal experiment requires exactly five folds")
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=int(anti["fold_seed"])
    )
    pool_array = np.asarray(pool)
    assignment = np.full(len(pool), -1, dtype=int)
    for fold, (_, val_index) in enumerate(splitter.split(pool_array, labels)):
        assignment[val_index] = fold
    if (assignment < 0).any():
        raise RuntimeError("initial fold assignment is incomplete")
    initial_assignment = assignment.copy()

    # Preserve every pCR|HR/HER2|source stratum and every fold size while
    # deterministically swapping patients to balance visit-count marginals.
    visit_counts = development["n_registered_timepoints"].astype(int).to_numpy()
    categories = (1, 2, 3, 4)
    counts = np.asarray(
        [[int(((assignment == fold) & (visit_counts == value)).sum()) for value in categories]
         for fold in range(folds)],
        dtype=float,
    )
    target = np.asarray([(visit_counts == value).sum() / folds for value in categories])

    def cell_cost(count, category_index):
        return (count - target[category_index]) ** 2

    while True:
        best = None
        for left in range(len(pool)):
            for right in range(left + 1, len(pool)):
                fold_left, fold_right = assignment[left], assignment[right]
                visit_left, visit_right = visit_counts[left], visit_counts[right]
                if (
                    fold_left == fold_right
                    or visit_left == visit_right
                    or labels[left] != labels[right]
                ):
                    continue
                left_cat, right_cat = visit_left - 1, visit_right - 1
                before = (
                    cell_cost(counts[fold_left, left_cat], left_cat)
                    + cell_cost(counts[fold_left, right_cat], right_cat)
                    + cell_cost(counts[fold_right, left_cat], left_cat)
                    + cell_cost(counts[fold_right, right_cat], right_cat)
                )
                after = (
                    cell_cost(counts[fold_left, left_cat] - 1, left_cat)
                    + cell_cost(counts[fold_left, right_cat] + 1, right_cat)
                    + cell_cost(counts[fold_right, left_cat] + 1, left_cat)
                    + cell_cost(counts[fold_right, right_cat] - 1, right_cat)
                )
                delta = float(after - before)
                candidate = (delta, left, right)
                if delta < -1e-12 and (best is None or candidate < best):
                    best = candidate
        if best is None:
            break
        _, left, right = best
        fold_left, fold_right = assignment[left], assignment[right]
        left_cat, right_cat = visit_counts[left] - 1, visit_counts[right] - 1
        counts[fold_left, left_cat] -= 1
        counts[fold_left, right_cat] += 1
        counts[fold_right, left_cat] += 1
        counts[fold_right, right_cat] -= 1
        assignment[left], assignment[right] = fold_right, fold_left

    specs = []
    for fold in range(folds):
        val_index = np.flatnonzero(assignment == fold)
        train_index = np.flatnonzero(assignment != fold)
        train_ids = pool_array[train_index].tolist()
        val_ids = pool_array[val_index].tolist()
        if set(train_ids) & set(val_ids) or (set(train_ids) | set(val_ids)) & sets["test"]:
            raise RuntimeError("generated fold leaks locked-test patients")
        specs.append({"fold": fold, "train_ids": train_ids, "val_ids": val_ids})
    fold_audit = development[
        ["pid", "pCR", "HR_HER2_STATUS", "source", "n_registered_timepoints", "base_stratum"]
    ].copy()
    fold_audit["initial_fold"] = initial_assignment
    fold_audit["fold"] = assignment
    return specs, pool, original["test"], fold_audit


def _write_fold_manifests(output_dir, specs, config, fold_audit):
    pool_ids = [pid for spec in specs for pid in spec["val_ids"]]
    metadata = _read_metadata(
        _repo_path(config["data"]["metadata_csv"]), pool_ids
    ).set_index("pid")
    audit_rows = []
    for spec in specs:
        fold_dir = Path(output_dir) / "folds" / f"fold_{spec['fold']}"
        _atomic_text(fold_dir / "train_ids.txt", "\n".join(spec["train_ids"]) + "\n")
        _atomic_text(fold_dir / "val_ids.txt", "\n".join(spec["val_ids"]) + "\n")
        for split_name in ("train", "val"):
            ids = spec[f"{split_name}_ids"]
            selected = metadata.loc[ids]
            row = {
                "fold": int(spec["fold"]),
                "split": split_name,
                "patients": len(ids),
                "pcr_positive": int(selected["pCR"].sum()),
                "pcr_rate": float(selected["pCR"].mean()),
            }
            for column in ("TripleNeg", "HER2pos", "HRposHER2neg"):
                if column in selected:
                    values = pd.to_numeric(selected[column], errors="coerce")
                    row[f"{column}_positive"] = int(values.fillna(0).sum())
                    row[f"{column}_rate"] = float(values.mean())
            if "dataset" in selected:
                for value, count in selected["dataset"].fillna("missing").value_counts().items():
                    row[f"dataset_{value}"] = int(count)
            audit_rows.append(row)
    _atomic_csv(Path(output_dir) / "folds" / "marginal_balance_audit.csv", pd.DataFrame(audit_rows))
    _atomic_csv(Path(output_dir) / "folds" / "patient_fold_assignments.csv", fold_audit)
    balance = []
    for stage, fold_column in (("initial", "initial_fold"), ("balanced", "fold")):
        for fold in range(len(specs)):
            selected = fold_audit[fold_audit[fold_column] == fold]
            balance.append(
                {
                    "stage": stage,
                    "fold": fold,
                    **{
                        f"patients_with_{visits}_visits": int(
                            (selected["n_registered_timepoints"] == visits).sum()
                        )
                        for visits in range(1, 5)
                    },
                }
            )
    _atomic_csv(Path(output_dir) / "folds" / "visit_count_balance.csv", pd.DataFrame(balance))


def _variant_config(config, variant_name, smoke=False):
    variants = config["variants"]
    if variant_name not in variants:
        raise ValueError(f"unknown variant: {variant_name}")
    result = copy.deepcopy(config["downstream"])
    for key, value in variants[variant_name].items():
        if key != "description":
            result[key] = value
    result["input_dim"] = int(config["data"]["embedding_dim"])
    result["clinical_dim"] = len(TABULAR_FEATURE_NAMES)
    result["model_type"] = "tdn"
    if smoke:
        result["epochs"] = 2
        result["patience"] = 2
    return result


def _fit_prior(train, val, seed):
    model = LogisticRegression(
        C=1.0, max_iter=2000, solver="lbfgs", random_state=int(seed)
    )
    model.fit(train["clinical"], train["labels"])

    def logits(split):
        values = model.decision_function(split["clinical"]).astype(np.float32)
        return np.clip(np.nan_to_num(values, nan=0.0, posinf=30.0, neginf=-30.0), -30, 30)

    state = {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "classes": model.classes_.tolist(),
        "coef": model.coef_.astype(float).tolist(),
        "intercept": model.intercept_.astype(float).tolist(),
        "feature_names": list(TABULAR_FEATURE_NAMES),
        "fitted_on": "fold_train_only",
        "n_features_in": int(model.n_features_in_),
        "C": float(model.C),
        "solver": str(model.solver),
        "class_weight": model.class_weight,
        "max_iter": int(model.max_iter),
    }
    return logits(train), logits(val), state


def _prior_from_state(clinical, state):
    coefficient = np.asarray(state["coef"], dtype=np.float32)[0]
    intercept = float(np.asarray(state["intercept"]).reshape(-1)[0])
    values = np.asarray(clinical, dtype=np.float32) @ coefficient + intercept
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=30.0, neginf=-30.0), -30, 30)


def _loader(split, prior, batch_size, shuffle, seed=None):
    dataset = TensorDataset(
        torch.from_numpy(split["embs"]),
        torch.from_numpy(split["masks"]),
        torch.from_numpy(split["clinical"]),
        torch.from_numpy(split["labels"]),
        torch.from_numpy(np.asarray(prior, dtype=np.float32)),
        torch.from_numpy(split["days"]),
    )
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _predict_prefix(model, loader, depth, device):
    labels = []
    probabilities = []
    model.eval()
    with torch.no_grad():
        for embeddings, masks, clinical, target, prior, days in loader:
            embeddings = embeddings.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            prior = prior.to(device, non_blocking=True)
            days = days.to(device, non_blocking=True)
            embeddings, masks, days = mask_torch_temporal_prefix(
                embeddings, masks, days, depth
            )
            logits, _ = model(
                embeddings, masks, clinical, days=days, prior_logit=prior,
                return_residual=True,
            )
            labels.append(target.numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def _training_dir(output_dir, stage, variant, seed, fold):
    return (
        Path(output_dir) / "train" / stage / f"variant_{variant}"
        / f"seed_{seed}" / f"fold_{fold}"
    )


def _validate_long_predictions(frame, expected_ids, seed, fold, split, depths):
    required = {
        "patient_id", "label", "probability", "seed", "fold", "split",
        "temporal_depth", "max_tp",
    }
    if required - set(frame.columns) or not np.isfinite(frame["probability"]).all():
        return False
    if not frame["probability"].between(0, 1).all():
        return False
    if set(frame["seed"]) != {int(seed)} or set(frame["fold"]) != {int(fold)}:
        return False
    if set(frame["split"]) != {split}:
        return False
    for depth in depths:
        rows = frame[frame["temporal_depth"] == str(depth["name"])]
        if rows["patient_id"].tolist() != list(expected_ids):
            return False
        if set(rows["max_tp"]) != {int(depth["max_tp"])}:
            return False
    return len(frame) == len(expected_ids) * len(depths)


def _training_complete(
    run_dir, expected_train_ids, expected_val_ids, stage, variant, seed, fold,
    expected_config, depths
):
    run_dir = Path(run_dir)
    paths = [
        run_dir / "best.pt", run_dir / "history.csv", run_dir / "train_predictions.csv",
        run_dir / "val_predictions.csv",
        run_dir / "TRAINING_COMPLETE.json",
    ]
    if not all(path.is_file() for path in paths):
        return False
    try:
        sentinel = json.loads(paths[-1].read_text())
        checkpoint = torch.load(paths[0], map_location="cpu", weights_only=False)
        train_predictions = pd.read_csv(paths[2], dtype={"patient_id": str})
        val_predictions = pd.read_csv(paths[3], dtype={"patient_id": str})
        return (
            sentinel.get("schema") == SCHEMA
            and sentinel.get("complete") is True
            and checkpoint.get("schema") == SCHEMA
            and checkpoint.get("variant") == variant
            and checkpoint.get("stage") == stage
            and int(checkpoint.get("seed", -1)) == int(seed)
            and int(checkpoint.get("fold", -1)) == int(fold)
            and checkpoint.get("effective_config") == expected_config
            and checkpoint.get("selection", {}).get("criterion")
            == "mean_validation_auroc_across_four_prefixes"
            and checkpoint.get("test_data_loaded_during_training") is False
            and checkpoint.get("clinical_prior", {}).get("fitted_on")
            == "fold_train_only"
            and checkpoint.get("clinical_prior", {}).get("feature_names")
            == list(TABULAR_FEATURE_NAMES)
            and _validate_long_predictions(
                train_predictions, expected_train_ids, seed, fold, "fold_train", depths
            )
            and _validate_long_predictions(
                val_predictions, expected_val_ids, seed, fold, "oof_val", depths
            )
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
        return False


def _train_fold_task(task):
    config_path, output_dir, stage, variant, seed, spec, device_text, smoke, force = task
    with Path(config_path).open() as handle:
        config = _release_yaml(handle)
    depths = config["anti_overfit"]["temporal_depths"]
    fold = int(spec["fold"])
    ds_cfg = _variant_config(config, variant, smoke=smoke)
    run_dir = _training_dir(output_dir, stage, variant, seed, fold)
    if not force and _training_complete(
        run_dir, spec["train_ids"], spec["val_ids"], stage, variant, seed, fold,
        ds_cfg, depths
    ):
        return f"skip variant={variant} seed={seed} fold={fold}"

    _set_seed(int(seed) * 100 + fold)
    torch.set_num_threads(1)
    device = _resolve_device(device_text)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_cfg = config["data"]
    embeddings_dir = _repo_path(data_cfg["train_embeddings_dir"])
    metadata_csv = _repo_path(data_cfg["metadata_csv"])
    train = _load_split(embeddings_dir, metadata_csv, spec["train_ids"])
    val = _load_split(embeddings_dir, metadata_csv, spec["val_ids"])
    if train["dim"] != ds_cfg["input_dim"] or val["dim"] != ds_cfg["input_dim"]:
        raise ValueError("training embedding dimension mismatch")

    train_prior, val_prior, prior_state = _fit_prior(train, val, int(seed) * 100 + fold)
    train_loader = _loader(
        train, train_prior, ds_cfg["batch_size"], True, int(seed) * 100 + fold
    )
    val_loader = _loader(val, val_prior, ds_cfg["batch_size"] * 4, False)
    model = TDN({"downstream": ds_cfg}).to(device)
    optimizer = Adam(model.parameters(), lr=ds_cfg["lr"], weight_decay=ds_cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=ds_cfg["epochs"])
    positives = float((train["labels"] == 1).sum())
    negatives = float((train["labels"] == 0).sum())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], device=device)
    )

    best_score = -np.inf
    best_state = None
    best_epoch = -1
    best_prefix_auroc = None
    patience = 0
    history = []
    for epoch in range(int(ds_cfg["epochs"])):
        model.train()
        loss_sum = 0.0
        seen = 0
        for embeddings, masks, clinical, target, prior, days in train_loader:
            embeddings = embeddings.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prior = prior.to(device, non_blocking=True)
            days = days.to(device, non_blocking=True)
            sampled_depths = torch.randint(
                1, 5, (embeddings.shape[0],), device=device, dtype=torch.long
            )
            random_embeddings, random_masks, random_days = mask_torch_temporal_prefix(
                embeddings, masks, days, sampled_depths
            )
            stacked_embeddings = torch.cat([embeddings, random_embeddings], dim=0)
            stacked_masks = torch.cat([masks, random_masks], dim=0)
            stacked_days = torch.cat([days, random_days], dim=0)
            stacked_clinical = torch.cat([clinical, clinical], dim=0)
            stacked_target = torch.cat([target, target], dim=0)
            stacked_prior = torch.cat([prior, prior], dim=0)
            optimizer.zero_grad(set_to_none=True)
            logits, residual = model(
                stacked_embeddings, stacked_masks, stacked_clinical, days=stacked_days,
                prior_logit=stacked_prior, return_residual=True,
            )
            loss = loss_fn(logits, stacked_target)
            variance_weight = float(ds_cfg.get("centered_residual_variance_weight", 0.0))
            if variance_weight > 0:
                centered = residual - residual.mean()
                loss = loss + variance_weight * centered.square().mean()
            increment_weight = float(ds_cfg.get("prefix_increment_weight", 0.0))
            if increment_weight > 0:
                batch_size = embeddings.shape[0]
                increments = residual[:batch_size] - residual[batch_size:]
                centered_increments = increments - increments.mean()
                loss = loss + increment_weight * centered_increments.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            if float(ds_cfg.get("grad_clip", 0)) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(ds_cfg["grad_clip"]))
            optimizer.step()
            loss_sum += float(loss.item()) * len(target)
            seen += len(target)
        scheduler.step()

        prefix_aurocs = {}
        for depth in depths:
            labels, probabilities = _predict_prefix(
                model, val_loader, int(depth["max_tp"]), device
            )
            prefix_aurocs[str(depth["name"])] = float(
                roc_auc_score(labels, probabilities)
            )
        selection_score = float(np.mean(list(prefix_aurocs.values())))
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(seen, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "residual_scale": float(model.residual_scale().detach().cpu()),
                **{f"val_auroc_{name}": value for name, value in prefix_aurocs.items()},
                "selection_score": selection_score,
            }
        )
        min_delta = float(ds_cfg.get("min_delta", 0.0))
        if np.isfinite(selection_score) and selection_score > best_score + min_delta:
            best_score = selection_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_prefix_auroc = prefix_aurocs
            patience = 0
        else:
            patience += 1
        if patience >= int(ds_cfg["patience"]):
            break
    if best_state is None:
        raise RuntimeError("validation never produced a finite selection score")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})

    train_eval_loader = _loader(train, train_prior, ds_cfg["batch_size"] * 4, False)

    def prediction_frame(split, split_name, loader, prior_values):
        frames = []
        prior_probability = 1.0 / (1.0 + np.exp(-np.asarray(prior_values, dtype=np.float64)))
        for depth in depths:
            labels, probabilities = _predict_prefix(
                model, loader, int(depth["max_tp"]), device
            )
            frames.append(
                pd.DataFrame(
                    {
                        "patient_id": split["pids"],
                        "label": labels.astype(int),
                        "probability": probabilities,
                        "clinical_prior_probability": prior_probability,
                        "seed": int(seed),
                        "fold": fold,
                        "split": split_name,
                        "temporal_depth": str(depth["name"]),
                        "max_tp": int(depth["max_tp"]),
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    train_predictions = prediction_frame(
        train, "fold_train", train_eval_loader, train_prior
    )
    val_predictions = prediction_frame(val, "oof_val", val_loader, val_prior)

    def metrics_by_depth(predictions):
        rows = []
        for depth in depths:
            selected = predictions[
                predictions["temporal_depth"] == str(depth["name"])
            ]
            rows.append(
                {
                    "temporal_depth": str(depth["name"]),
                    "max_tp": int(depth["max_tp"]),
                    **compute_metrics(
                        selected["label"], selected["probability"], threshold=0.5
                    ),
                    "clinical_prior_auroc": float(
                        roc_auc_score(
                            selected["label"], selected["clinical_prior_probability"]
                        )
                    ),
                }
            )
        return rows

    train_metrics = metrics_by_depth(train_predictions)
    val_metrics = metrics_by_depth(val_predictions)
    selection = {
        "criterion": "mean_validation_auroc_across_four_prefixes",
        "best_epoch_zero_based": best_epoch,
        "best_score": best_score,
        "best_prefix_auroc": best_prefix_auroc,
        "epochs_trained": len(history),
    }
    checkpoint = {
        "schema": SCHEMA,
        "stage": stage,
        "variant": variant,
        "seed": int(seed),
        "fold": fold,
        "model_state": best_state,
        "effective_config": ds_cfg,
        "clinical_prior": prior_state,
        "selection": selection,
        "test_data_loaded_during_training": False,
    }
    _atomic_torch(run_dir / "best.pt", checkpoint)
    _atomic_csv(run_dir / "history.csv", pd.DataFrame(history))
    _atomic_csv(run_dir / "train_predictions.csv", train_predictions)
    _atomic_csv(run_dir / "val_predictions.csv", val_predictions)
    _atomic_csv(run_dir / "train_metrics.csv", pd.DataFrame(train_metrics))
    _atomic_csv(run_dir / "val_metrics.csv", pd.DataFrame(val_metrics))
    _atomic_json(
        run_dir / "summary.json",
        {
            "schema": SCHEMA,
            "stage": stage,
            "variant": variant,
            "seed": int(seed),
            "fold": fold,
            "train_patients": len(train["pids"]),
            "validation_patients": len(val["pids"]),
            "selection": selection,
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "test_role": "not_loaded",
        },
    )
    _atomic_json(
        run_dir / "TRAINING_COMPLETE.json",
        {
            "schema": SCHEMA,
            "stage": stage,
            "variant": variant,
            "seed": int(seed),
            "fold": fold,
            "effective_config": ds_cfg,
            "complete": True,
        },
    )
    return f"done stage={stage} variant={variant} seed={seed} fold={fold} score={best_score:.4f}"


def _build_oof_artifacts(
    config, output_dir, specs, stage, variants, seeds, select_variant, smoke
):
    depths = config["anti_overfit"]["temporal_depths"]
    pool_ids = [pid for spec in specs for pid in spec["val_ids"]]
    if len(pool_ids) != len(set(pool_ids)):
        raise RuntimeError("fold validation sets are not a disjoint partition")
    variant_rows = []
    for variant in variants:
        expected_config = _variant_config(config, variant, smoke=smoke)
        for seed in seeds:
            frames = []
            for spec in specs:
                run_dir = _training_dir(output_dir, stage, variant, seed, spec["fold"])
                if not _training_complete(
                    run_dir,
                    spec["train_ids"],
                    spec["val_ids"],
                    stage,
                    variant,
                    seed,
                    spec["fold"],
                    expected_config,
                    depths,
                ):
                    raise RuntimeError(f"incomplete training artifact: {run_dir}")
                frames.append(pd.read_csv(run_dir / "val_predictions.csv", dtype={"patient_id": str}))
            oof = pd.concat(frames, ignore_index=True)
            metrics_rows = []
            for depth in depths:
                rows = oof[oof["temporal_depth"] == str(depth["name"])]
                if set(rows["patient_id"]) != set(pool_ids) or rows["patient_id"].duplicated().any():
                    raise RuntimeError("OOF predictions do not cover the development pool exactly")
                metrics = compute_metrics(rows["label"], rows["probability"], threshold=0.5)
                metrics_rows.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "temporal_depth": str(depth["name"]),
                        "max_tp": int(depth["max_tp"]),
                        **metrics,
                    }
                )
            seed_dir = (
                Path(output_dir) / "oof" / stage / f"variant_{variant}" / f"seed_{seed}"
            )
            _atomic_csv(seed_dir / "oof_predictions.csv", oof)
            _atomic_csv(seed_dir / "oof_metrics.csv", pd.DataFrame(metrics_rows))
            _atomic_json(
                seed_dir / "OOF_COMPLETE.json",
                {
                    "schema": SCHEMA,
                    "stage": stage,
                    "variant": variant,
                    "seed": int(seed),
                    "folds": len(specs),
                    "effective_config": expected_config,
                    "complete": True,
                },
            )
            variant_rows.extend(metrics_rows)
    metrics = pd.DataFrame(variant_rows)
    summary_rows = []
    for order, variant in enumerate(variants):
        rows = metrics[metrics["variant"] == variant]
        per_seed = rows.groupby("seed", sort=True)["auroc"].mean()
        summary_rows.append(
            {
                "variant": variant,
                "variant_order": order,
                "selection_score": float(rows["auroc"].mean()),
                "seed_score_std_ddof1": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
                "n_seeds": len(per_seed),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["selection_score", "variant_order"], ascending=[False, True]
    )
    selected = str(summary.iloc[0]["variant"])
    _atomic_csv(Path(output_dir) / f"{stage}_oof_metrics_all.csv", metrics)
    _atomic_csv(Path(output_dir) / f"{stage}_variant_summary.csv", summary)
    if not select_variant:
        return {
            "schema": SCHEMA,
            "stage": stage,
            "variants": variants,
            "seeds": [int(seed) for seed in seeds],
            "test_data_used": False,
        }
    selection = {
        "schema": SCHEMA,
        "stage": stage,
        "criterion": config["anti_overfit"]["selection"],
        "selected_variant": selected,
        "selection_score": float(summary.iloc[0]["selection_score"]),
        "tune_seeds": [int(seed) for seed in seeds],
        "folds": len(specs),
        "test_data_used": False,
        "variant_configs": {
            variant: _variant_config(config, variant, smoke=smoke) for variant in variants
        },
    }
    _atomic_json(Path(output_dir) / "selected_variant.json", selection)
    return selection


def _load_test_source(config, source_name, test_ids):
    source = config["evaluation_sources"][source_name]
    metadata_csv = _repo_path(config["data"]["metadata_csv"])
    if source["mode"] == "direct":
        result = _load_split(_repo_path(source["embeddings_dir"]), metadata_csv, test_ids)
    elif source["mode"] == "overlay_future":
        result = _load_split(_repo_path(source["base_embeddings_dir"]), metadata_csv, test_ids)
        overlay = EmbStore(str(_repo_path(source["overlay_embeddings_dir"])))
        availability = pd.read_csv(
            _repo_path(config["data"]["availability_csv"]), dtype={"patient_id": str}
        )
        availability = availability[availability["patient_id"].isin(set(test_ids))]
        if availability.duplicated(["patient_id", "visit"]).any():
            raise ValueError("availability.csv contains duplicate patient visits")
        availability_index = {
            (str(row.patient_id), int(str(row.visit).removeprefix("T"))): int(row.valid_mask)
            for row in availability.itertuples(index=False)
        }
        expected_availability = {
            (patient_id, timepoint) for patient_id in test_ids for timepoint in range(4)
        }
        if set(availability_index) != expected_availability:
            raise ValueError("availability.csv does not cover the locked test grid exactly")
        replaced = 0
        for patient_index, patient_id in enumerate(result["pids"]):
            for timepoint in range(4):
                if int(result["masks"][patient_index, timepoint] > 0) != availability_index[
                    (patient_id, timepoint)
                ]:
                    raise ValueError("real embedding mask disagrees with availability.csv")
            for timepoint in range(1, 4):
                if availability_index[(patient_id, timepoint)] <= 0:
                    if overlay.exists(patient_id, timepoint):
                        raise ValueError("generated overlay exists for an invalid future token")
                    continue
                if not overlay.exists(patient_id, timepoint):
                    raise FileNotFoundError(
                        f"generated overlay is missing {patient_id} T{timepoint}"
                    )
                vector = overlay.load(patient_id, timepoint, require_finite=True)
                if vector.shape != (result["dim"],):
                    raise ValueError("generated overlay embedding dimension mismatch")
                result["embs"][patient_index, timepoint] = vector
                replaced += 1
        if replaced != int(source["expected_overlay_tokens"]):
            raise ValueError(
                f"generated overlay count mismatch: expected {source['expected_overlay_tokens']}, found {replaced}"
            )
    else:
        raise ValueError(f"unsupported evaluation source mode: {source['mode']}")
    if result["dim"] != int(config["data"]["embedding_dim"]):
        raise ValueError("test embedding dimension mismatch")
    return result


def _evaluation_dir(output_dir, source_name, seed):
    return Path(output_dir) / "evaluation" / source_name / f"seed_{seed}"


def _evaluation_source_contract(config, source_name):
    data = config["data"]
    return {
        "source_config": copy.deepcopy(config["evaluation_sources"][source_name]),
        "metadata_csv": data["metadata_csv"],
        "availability_csv": data.get("availability_csv"),
        "embedding_dim": int(data["embedding_dim"]),
    }


def _evaluation_complete(
    run_dir, test_ids, source, seed, depths, selected_variant, expected_folds,
    source_contract,
):
    run_dir = Path(run_dir)
    prediction_path = run_dir / "test_predictions.csv"
    sentinel_path = run_dir / "EVALUATION_COMPLETE.json"
    if not prediction_path.is_file() or not sentinel_path.is_file():
        return False
    try:
        sentinel = json.loads(sentinel_path.read_text())
        predictions = pd.read_csv(prediction_path, dtype={"patient_id": str})
        return (
            sentinel.get("schema") == SCHEMA
            and sentinel.get("complete") is True
            and sentinel.get("source") == source
            and sentinel.get("selected_variant") == selected_variant
            and int(sentinel.get("seed", -1)) == int(seed)
            and int(sentinel.get("folds_ensembled", -1)) == int(expected_folds)
            and sentinel.get("source_contract") == source_contract
            and _validate_long_predictions(
                predictions, test_ids, seed, -1, "test_fold_ensemble", depths
            )
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _evaluate_seed_task(task):
    (
        config_path, output_dir, variant, seed, specs, test_ids, source_name,
        device_text, smoke, _force,
    ) = task
    with Path(config_path).open() as handle:
        config = _release_yaml(handle)
    depths = config["anti_overfit"]["temporal_depths"]
    run_dir = _evaluation_dir(output_dir, source_name, seed)
    expected_config = _variant_config(config, variant, smoke=smoke)

    # Evaluation is inexpensive relative to training and is always replayed so
    # in-place source or checkpoint updates cannot silently reuse stale scores.
    for spec in specs:
        fold = int(spec["fold"])
        training_dir = _training_dir(output_dir, "formal", variant, seed, fold)
        if not _training_complete(
            training_dir,
            spec["train_ids"],
            spec["val_ids"],
            "formal",
            variant,
            seed,
            fold,
            expected_config,
            depths,
        ):
            raise RuntimeError(f"formal training contract mismatch: {training_dir}")

    # This is deliberately the first test-data load in the evaluate worker.
    test = _load_test_source(config, source_name, test_ids)
    torch.set_num_threads(1)
    device = _resolve_device(device_text)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fold_frames = []
    for spec in specs:
        fold = int(spec["fold"])
        checkpoint_path = _training_dir(output_dir, "formal", variant, seed, fold) / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("schema") != SCHEMA
            or checkpoint.get("stage") != "formal"
            or checkpoint.get("variant") != variant
            or int(checkpoint.get("seed", -1)) != int(seed)
            or int(checkpoint.get("fold", -1)) != fold
            or checkpoint.get("effective_config") != expected_config
            or checkpoint.get("test_data_loaded_during_training") is not False
            or checkpoint.get("clinical_prior", {}).get("fitted_on")
            != "fold_train_only"
            or checkpoint.get("clinical_prior", {}).get("feature_names")
            != list(TABULAR_FEATURE_NAMES)
            or checkpoint.get("selection", {}).get("criterion")
            != "mean_validation_auroc_across_four_prefixes"
        ):
            raise RuntimeError(f"formal checkpoint contract mismatch: {checkpoint_path}")
        ds_cfg = checkpoint["effective_config"]
        model = TDN({"downstream": ds_cfg}).to(device)
        model.load_state_dict(checkpoint["model_state"])
        prior = _prior_from_state(test["clinical"], checkpoint["clinical_prior"])
        prior_probability = 1.0 / (1.0 + np.exp(-np.asarray(prior, dtype=np.float64)))
        loader = _loader(test, prior, ds_cfg["batch_size"] * 4, False)
        for depth in depths:
            labels, probabilities = _predict_prefix(
                model, loader, int(depth["max_tp"]), device
            )
            fold_frames.append(
                pd.DataFrame(
                    {
                        "patient_id": test["pids"],
                        "label": labels.astype(int),
                        "probability": probabilities,
                        "clinical_prior_probability": prior_probability,
                        "seed": int(seed),
                        "fold": fold,
                        "split": "test_fold",
                        "temporal_depth": str(depth["name"]),
                        "max_tp": int(depth["max_tp"]),
                        "source": source_name,
                    }
                )
            )
        del model
    fold_predictions = pd.concat(fold_frames, ignore_index=True)
    ensemble = (
        fold_predictions.groupby(
            ["patient_id", "label", "seed", "temporal_depth", "max_tp", "source"],
            as_index=False,
            sort=False,
        )[["probability", "clinical_prior_probability"]].mean()
    )
    ensemble["fold"] = -1
    ensemble["split"] = "test_fold_ensemble"
    ensemble = ensemble[
        [
            "patient_id", "label", "probability", "seed", "fold", "split",
            "temporal_depth", "max_tp", "source", "clinical_prior_probability",
        ]
    ]
    # Restore the locked split order within every temporal depth.
    order = {patient_id: index for index, patient_id in enumerate(test_ids)}
    depth_order = {str(row["name"]): index for index, row in enumerate(depths)}
    ensemble["_depth_order"] = ensemble["temporal_depth"].map(depth_order)
    ensemble["_patient_order"] = ensemble["patient_id"].map(order)
    ensemble = ensemble.sort_values(["_depth_order", "_patient_order"]).drop(
        columns=["_depth_order", "_patient_order"]
    )
    _atomic_csv(run_dir / "fold_predictions.csv", fold_predictions)
    _atomic_csv(run_dir / "test_predictions.csv", ensemble)
    _atomic_json(
        run_dir / "EVALUATION_COMPLETE.json",
        {
            "schema": SCHEMA,
            "source": source_name,
            "seed": int(seed),
            "selected_variant": variant,
            "folds_ensembled": len(specs),
            "source_contract": _evaluation_source_contract(config, source_name),
            "complete": True,
        },
    )
    return f"done evaluation source={source_name} seed={seed}"


def _aggregate_evaluation(config, output_dir, sources, seeds, test_ids, expected_folds):
    depths = config["anti_overfit"]["temporal_depths"]
    selected_variant = json.loads(
        (Path(output_dir) / "selected_variant.json").read_text()
    )["selected_variant"]
    for source in sources:
        frames = []
        rows = []
        for seed in seeds:
            run_dir = _evaluation_dir(output_dir, source, seed)
            if not _evaluation_complete(
                run_dir,
                test_ids,
                source,
                seed,
                depths,
                selected_variant,
                expected_folds,
                _evaluation_source_contract(config, source),
            ):
                raise RuntimeError(f"incomplete evaluation artifact: {run_dir}")
            frame = pd.read_csv(run_dir / "test_predictions.csv", dtype={"patient_id": str})
            frames.append(frame)
            for depth in depths:
                selected = frame[frame["temporal_depth"] == str(depth["name"])]
                rows.append(
                    {
                        "source": source,
                        "seed": int(seed),
                        "temporal_depth": str(depth["name"]),
                        "max_tp": int(depth["max_tp"]),
                        **compute_metrics(selected["label"], selected["probability"], threshold=0.5),
                    }
                )
        metrics = pd.DataFrame(rows)
        summary_rows = []
        for depth in depths:
            selected = metrics[metrics["temporal_depth"] == str(depth["name"])]
            row = {
                "source": source,
                "temporal_depth": str(depth["name"]),
                "max_tp": int(depth["max_tp"]),
                "n_seeds": len(seeds),
            }
            for metric in METRIC_KEYS:
                values = selected[metric].to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(np.nanmean(values))
                row[f"{metric}_std"] = (
                    float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
                )
            summary_rows.append(row)
        source_dir = Path(output_dir) / "evaluation" / source
        _atomic_csv(source_dir / "all_test_predictions.csv", pd.concat(frames, ignore_index=True))
        _atomic_csv(source_dir / "metrics_per_seed.csv", metrics)
        _atomic_csv(source_dir / "summary.csv", pd.DataFrame(summary_rows))


def _run_tasks(tasks, jobs, label):
    if not tasks:
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(jobs), mp_context=context) as executor:
        futures = [executor.submit(task[0], task[1:]) for task in tasks]
        for future in as_completed(futures):
            print(f"[{label}] {future.result()}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/mewm_ispy2_full978_locked102_anti_overfit.yaml"
    )
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--tune-seeds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = _repo_path(args.config)
    with config_path.open() as handle:
        config = _release_yaml(handle)
    anti = config["anti_overfit"]
    declared_tune_seeds = [int(seed) for seed in anti["tune_seeds"]]
    declared_formal_seeds = [int(seed) for seed in anti["formal_seeds"]]
    tune_seeds = args.tune_seeds or (
        [declared_tune_seeds[0]] if args.smoke else declared_tune_seeds
    )
    seeds = args.seeds or ([declared_formal_seeds[0]] if args.smoke else declared_formal_seeds)
    if (
        len(tune_seeds) != len(set(tune_seeds))
        or len(seeds) != len(set(seeds))
        or (not args.smoke and set(tune_seeds) - set(declared_tune_seeds))
        or (not args.smoke and set(seeds) - set(declared_formal_seeds))
    ):
        raise SystemExit("tune/formal seeds must be unique and declared by the config")
    variants = args.variants or list(config["variants"])
    if len(variants) != len(set(variants)) or set(variants) - set(config["variants"]):
        raise SystemExit("variants must be unique and declared by the config")
    sources = args.sources or list(config["evaluation_sources"])
    if len(sources) != len(set(sources)) or set(sources) - set(config["evaluation_sources"]):
        raise SystemExit("evaluation sources must be unique and declared by the config")
    jobs = int(args.jobs or anti.get("jobs", 10))
    if jobs < 1:
        raise SystemExit("jobs must be positive")
    output_dir = _repo_path(args.output_dir or anti["output_dir"])
    if args.smoke and args.output_dir is None:
        output_dir = output_dir / "smoke"
    if not args.smoke and (
        variants != list(config["variants"])
        or tune_seeds != declared_tune_seeds
        or seeds != declared_formal_seeds
        or sources != list(config["evaluation_sources"])
    ):
        raise SystemExit(
            "formal runs must use every declared variant, tune seed, formal seed, and source"
        )
    canonical_output_dir = _repo_path(anti["output_dir"])
    if args.smoke and output_dir == canonical_output_dir:
        raise SystemExit("smoke runs may not write to the canonical formal output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    specs, pool_ids, test_ids, fold_audit = _fold_specs(config)
    source_contracts = {
        source: _evaluation_source_contract(config, source) for source in sources
    }
    _atomic_json(
        output_dir / "EXPERIMENT_COMPLETE.json",
        {
            "schema": SCHEMA,
            "phase": args.phase,
            "formal_seeds": [int(seed) for seed in seeds],
            "folds_per_seed": len(specs),
            "evaluation_sources": sources,
            "source_contracts": source_contracts,
            "locked_test_patients": len(test_ids),
            "complete": False,
        },
    )
    _write_fold_manifests(output_dir, specs, config, fold_audit)
    _atomic_text(
        output_dir / "resolved_config.yaml",
        yaml.safe_dump(config, sort_keys=True),
    )
    manifest = {
        "schema": SCHEMA,
        "config": str(config_path),
        "output_dir": str(output_dir),
        "phase": args.phase,
        "variants": variants,
        "tune_seeds": tune_seeds,
        "formal_seeds": seeds,
        "sources": sources,
        "folds": len(specs),
        "development_patients": len(pool_ids),
        "locked_test_patients": len(test_ids),
        "test_policy": "loaded_only_after_oof_variant_selection_in_evaluate_phase",
        "fold_stratification": anti.get("fold_stratification"),
        "parallel_start_method": "spawn",
        "jobs": jobs,
        "smoke": bool(args.smoke),
    }
    _atomic_json(output_dir / "run_manifest.json", manifest)

    selection = None
    if args.phase in ("train", "all"):
        tasks = []
        for variant in variants:
            for seed in tune_seeds:
                for spec in specs:
                    tasks.append(
                        (
                            _train_fold_task,
                            str(config_path),
                            str(output_dir),
                            "tune",
                            variant,
                            int(seed),
                            spec,
                            args.device,
                            bool(args.smoke),
                            bool(args.force),
                        )
                    )
        _run_tasks(tasks, jobs, "train")
        selection = _build_oof_artifacts(
            config, output_dir, specs, "tune", variants, tune_seeds, True,
            bool(args.smoke),
        )
        selected_variant = str(selection["selected_variant"])
        formal_tasks = []
        for seed in seeds:
            for spec in specs:
                formal_tasks.append(
                    (
                        _train_fold_task,
                        str(config_path),
                        str(output_dir),
                        "formal",
                        selected_variant,
                        int(seed),
                        spec,
                        args.device,
                        bool(args.smoke),
                        bool(args.force),
                    )
                )
        _run_tasks(formal_tasks, jobs, "formal-train")
        _build_oof_artifacts(
            config, output_dir, specs, "formal", [selected_variant], seeds, False,
            bool(args.smoke),
        )

    if args.phase in ("evaluate", "all"):
        if selection is None:
            selection_path = output_dir / "selected_variant.json"
            if not selection_path.is_file():
                raise SystemExit("evaluate requires a completed OOF variant selection")
            selection = json.loads(selection_path.read_text())
        if (
            selection.get("schema") != SCHEMA
            or selection.get("criterion") != anti["selection"]
            or int(selection.get("folds", -1)) != len(specs)
            or selection.get("selected_variant") not in config["variants"]
            or [int(seed) for seed in selection.get("tune_seeds", [])]
            != [int(seed) for seed in tune_seeds]
            or selection.get("variant_configs")
            != {
                variant: _variant_config(config, variant, smoke=bool(args.smoke))
                for variant in variants
            }
        ):
            raise SystemExit("selected variant does not match this config and seed set")
        selected_variant = str(selection["selected_variant"])
        tasks = []
        for source in sources:
            for seed in seeds:
                tasks.append(
                    (
                        _evaluate_seed_task,
                        str(config_path),
                        str(output_dir),
                        selected_variant,
                        int(seed),
                        specs,
                        test_ids,
                        source,
                        args.device,
                        bool(args.smoke),
                        bool(args.force),
                    )
                )
        _run_tasks(tasks, jobs, "evaluate")
        _aggregate_evaluation(
            config, output_dir, sources, seeds, test_ids, expected_folds=len(specs)
        )
        _atomic_json(
            output_dir / "EXPERIMENT_COMPLETE.json",
            {
                "schema": SCHEMA,
                "selected_variant": selected_variant,
                "formal_seeds": [int(seed) for seed in seeds],
                "folds_per_seed": len(specs),
                "evaluation_sources": sources,
                "source_contracts": source_contracts,
                "selected_variant_config": _variant_config(
                    config, selected_variant, smoke=bool(args.smoke)
                ),
                "locked_test_patients": len(test_ids),
                "test_used_for_selection": False,
                "complete": True,
            },
        )
        print(
            f"selected variant {selected_variant}; evaluation summaries: "
            + ", ".join(str(output_dir / "evaluation" / source / "summary.csv") for source in sources),
            flush=True,
        )


if __name__ == "__main__":
    main()
