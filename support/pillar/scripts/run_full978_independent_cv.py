"""Tune and evaluate independent T0-T2/T0-T3 models with five-fold OOF CV."""

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
from sklearn.metrics import roc_auc_score
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_full978_anti_overfit import (
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _atomic_torch,
    _fit_prior,
    _fold_specs as _shared_fold_specs,
    _load_split,
    _load_test_source,
    _loader,
    _prior_from_state,
    _repo_path,
    _resolve_device,
    _set_seed,
    _write_fold_manifests,
)
from src.data import TABULAR_FEATURE_NAMES
from src.metrics import METRIC_KEYS, compute_metrics
from src.tdn import TDN
from src.temporal import canonicalize_temporal_prefix, mask_torch_temporal_prefix


SCHEMA = "pillar_full978_independent_depth_cv_v1"
SELECTION_CRITERION = "validation_auroc_at_target_depth"
FINAL_EPOCH_CRITERION = "final_training_epoch"
CHECKPOINT_POLICIES = {"best_validation", "final_epoch"}


def _checkpoint_policy(config):
    policy = str(config.get("checkpoint_policy", "best_validation")).strip().lower()
    if policy not in CHECKPOINT_POLICIES:
        raise ValueError(
            f"unsupported checkpoint_policy: {policy}; expected one of "
            f"{sorted(CHECKPOINT_POLICIES)}"
        )
    return policy


def _selection_criterion(checkpoint_policy):
    return (
        FINAL_EPOCH_CRITERION
        if checkpoint_policy == "final_epoch"
        else SELECTION_CRITERION
    )


def _recorded_checkpoint_policy(artifact):
    selection = artifact.get("selection", {})
    return str(
        artifact.get(
            "checkpoint_policy",
            selection.get("checkpoint_policy", "best_validation"),
        )
    )


def _validation_selection_matches(artifact, checkpoint_policy):
    expected = checkpoint_policy == "best_validation"
    if checkpoint_policy == "best_validation":
        return artifact.get(
            "validation_used_for_checkpoint_selection", True
        ) is expected
    return artifact.get("validation_used_for_checkpoint_selection") is expected


def _training_test_isolation_matches(artifact, checkpoint_policy):
    field = "test_embeddings_or_labels_loaded_during_training"
    if field not in artifact:
        return checkpoint_policy == "best_validation"
    return artifact[field] is False


def _selection_uses_final_epoch(selection):
    return any(
        _recorded_checkpoint_policy(depth_selection) == "final_epoch"
        for depth_selection in selection.get("depths", {}).values()
    )


def _selection_test_isolation_matches(selection):
    field = "test_embeddings_or_labels_loaded"
    if field not in selection:
        return not _selection_uses_final_epoch(selection)
    return selection[field] is False


def _training_phase_complete(path, selection, tune_checkpoints, formal_checkpoints):
    path = Path(path)
    if not path.is_file():
        return not _selection_uses_final_epoch(selection)
    try:
        payload = json.loads(path.read_text())
        return (
            payload.get("schema") == SCHEMA
            and payload.get("complete") is True
            and payload.get("selected_variants") == selection.get("depths")
            and int(payload.get("tune_checkpoints", -1)) == int(tune_checkpoints)
            and int(payload.get("formal_checkpoints", -1)) == int(formal_checkpoints)
            and payload.get("test_data_loaded") is False
            and payload.get("test_embeddings_or_labels_loaded") is False
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _slug(value):
    return str(value).lower().replace("-", "_")


def _depths(config):
    rows = [
        {"name": str(row["name"]), "max_tp": int(row["max_tp"])}
        for row in config["independent_cv"]["temporal_depths"]
    ]
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("temporal-depth names must be unique")
    if any(row["max_tp"] not in (1, 2, 3, 4) for row in rows):
        raise ValueError("max_tp must be between one and four")
    return rows


def _fold_specs(config):
    compatibility = copy.deepcopy(config)
    section = config["independent_cv"]
    compatibility["anti_overfit"] = {
        "folds": int(section["folds"]),
        "fold_seed": int(section["fold_seed"]),
    }
    return _shared_fold_specs(compatibility)


def _effective_config(config, variant, depth_name, smoke=False):
    if variant not in config["variants"]:
        raise ValueError(f"unknown variant: {variant}")
    result = copy.deepcopy(config["downstream"])
    variant_config = config["variants"][variant]
    for key, value in variant_config.items():
        if key not in {"description", "depth_overrides"}:
            result[key] = copy.deepcopy(value)
    overrides = variant_config.get("depth_overrides", {}).get(depth_name, {})
    for key, value in overrides.items():
        result[key] = copy.deepcopy(value)
    result["input_dim"] = int(config["data"]["embedding_dim"])
    result["clinical_dim"] = len(TABULAR_FEATURE_NAMES)
    result["model_type"] = "tdn"
    _checkpoint_policy(result)
    if smoke:
        result["epochs"] = 2
        result["patience"] = 2
        result["scheduler_t_max"] = 2
    return result


def _canonical_split(split, max_tp):
    result = dict(split)
    result["embs"], result["masks"], result["days"] = canonicalize_temporal_prefix(
        np.asarray(split["embs"]),
        np.asarray(split["masks"]),
        np.asarray(split["days"]),
        int(max_tp),
    )
    result["clinical"] = np.asarray(split["clinical"]).copy()
    result["labels"] = np.asarray(split["labels"]).copy()
    result["pids"] = list(split["pids"])
    return result


def _subset_split(split, selector):
    selector = np.asarray(selector, dtype=bool)
    result = dict(split)
    for key in ("embs", "masks", "clinical", "labels", "days"):
        result[key] = np.asarray(split[key])[selector]
    result["pids"] = [
        patient_id
        for patient_id, keep in zip(split["pids"], selector)
        if keep
    ]
    return result


def _sample_prefix_depths(batch_size, max_tp, probability, device, generator=None):
    max_tp = int(max_tp)
    probability = float(probability)
    if max_tp < 1 or not 0 <= probability <= 1:
        raise ValueError("invalid structured prefix-dropout configuration")
    depths = torch.full((int(batch_size),), max_tp, dtype=torch.long, device=device)
    if max_tp == 1 or probability == 0 or int(batch_size) == 0:
        return depths
    draw = torch.rand(int(batch_size), device=device, generator=generator)
    shorten = draw < probability
    count = int(shorten.sum().item())
    if count:
        depths[shorten] = torch.randint(
            1, max_tp, (count,), dtype=torch.long, device=device, generator=generator
        )
    return depths


def _make_optimizer(model, config):
    name = str(config.get("optimizer", "adam")).lower()
    learning_rate = float(config["lr"])
    default_decay = float(config.get("weight_decay", 0.0))
    if name == "adam":
        return Adam(model.parameters(), lr=learning_rate, weight_decay=default_decay)
    if name != "adamw_layerwise":
        raise ValueError(f"unsupported optimizer: {name}")

    projection_decay = float(
        config.get("projection_weight_decay", default_decay)
    )
    groups = {"projection": [], "matrix": [], "no_decay": []}
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter_name == "proj.net.0.weight":
            groups["projection"].append(parameter)
        elif parameter.ndim < 2 or parameter_name in {"q", "alpha", "alpha_logit"}:
            groups["no_decay"].append(parameter)
        else:
            groups["matrix"].append(parameter)
    parameter_groups = [
        {"params": groups["projection"], "weight_decay": projection_decay},
        {"params": groups["matrix"], "weight_decay": default_decay},
        {"params": groups["no_decay"], "weight_decay": 0.0},
    ]
    parameter_groups = [group for group in parameter_groups if group["params"]]
    return AdamW(parameter_groups, lr=learning_rate)


def _positive_class_weight(config, labels):
    setting = config.get("positive_class_weight", "balanced")
    values = np.asarray(labels)
    if isinstance(setting, str) and setting.strip().lower() == "balanced":
        positives = float((values == 1).sum())
        negatives = float((values == 0).sum())
        return negatives / max(positives, 1.0)
    try:
        weight = float(setting)
    except (TypeError, ValueError) as error:
        raise ValueError("positive_class_weight must be 'balanced' or positive") from error
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("positive_class_weight must be 'balanced' or positive")
    return weight


def _parameter_count(config, variant, depth_name, smoke=False):
    effective = _effective_config(config, variant, depth_name, smoke=smoke)
    model = TDN({"downstream": effective})
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _training_dir(output_dir, stage, depth_name, variant, seed, fold):
    return (
        Path(output_dir)
        / "train"
        / stage
        / _slug(depth_name)
        / f"variant_{variant}"
        / f"seed_{int(seed)}"
        / f"fold_{int(fold)}"
    )


def _prediction_frame(model, split, prior, depth, seed, fold, split_name, device, batch_size):
    loader = _loader(split, prior, int(batch_size) * 4, False)
    labels = []
    probabilities = []
    residuals = []
    model.eval()
    with torch.no_grad():
        for embeddings, masks, clinical, target, prior_logit, days in loader:
            embeddings = embeddings.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            prior_logit = prior_logit.to(device, non_blocking=True)
            days = days.to(device, non_blocking=True)
            logits, residual = model(
                embeddings,
                masks,
                clinical,
                days=days,
                prior_logit=prior_logit,
                return_residual=True,
            )
            labels.append(target.numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            residuals.append(residual.cpu().numpy())
    prior_probability = 1.0 / (1.0 + np.exp(-np.asarray(prior, dtype=np.float64)))
    return pd.DataFrame(
        {
            "patient_id": split["pids"],
            "label": np.concatenate(labels).astype(int),
            "probability": np.concatenate(probabilities),
            "residual_logit": np.concatenate(residuals),
            "clinical_prior_probability": prior_probability,
            "seed": int(seed),
            "fold": int(fold),
            "split": split_name,
            "temporal_depth": depth["name"],
            "max_tp": int(depth["max_tp"]),
        }
    )


def _valid_prediction_frame(frame, patient_ids, seed, fold, split_name, depth):
    required = {
        "patient_id",
        "label",
        "probability",
        "residual_logit",
        "clinical_prior_probability",
        "seed",
        "fold",
        "split",
        "temporal_depth",
        "max_tp",
    }
    numeric = frame[["probability", "residual_logit", "clinical_prior_probability"]]
    return (
        required <= set(frame.columns)
        and frame["patient_id"].astype(str).tolist() == list(patient_ids)
        and len(frame) == len(patient_ids)
        and frame["patient_id"].astype(str).is_unique
        and frame["label"].isin([0, 1]).all()
        and np.isfinite(numeric.to_numpy(dtype=float)).all()
        and frame["probability"].between(0, 1).all()
        and frame["clinical_prior_probability"].between(0, 1).all()
        and set(frame["seed"]) == {int(seed)}
        and set(frame["fold"]) == {int(fold)}
        and set(frame["split"]) == {split_name}
        and set(frame["temporal_depth"]) == {depth["name"]}
        and set(frame["max_tp"]) == {int(depth["max_tp"])}
    )


def _training_complete(run_dir, expected):
    run_dir = Path(run_dir)
    paths = {
        "checkpoint": run_dir / "best.pt",
        "history": run_dir / "history.csv",
        "train": run_dir / "train_predictions.csv",
        "val": run_dir / "val_predictions.csv",
        "summary": run_dir / "summary.json",
        "sentinel": run_dir / "TRAINING_COMPLETE.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        sentinel = json.loads(paths["sentinel"].read_text())
        summary = json.loads(paths["summary"].read_text())
        history = pd.read_csv(paths["history"])
        train = pd.read_csv(paths["train"], dtype={"patient_id": str})
        val = pd.read_csv(paths["val"], dtype={"patient_id": str})
        checkpoint_policy = _checkpoint_policy(expected["effective_config"])
        selection = checkpoint.get("selection", {})
        final_epoch_contract = True
        if checkpoint_policy == "final_epoch":
            expected_epochs = int(expected["effective_config"]["epochs"])
            final_epoch_contract = (
                len(history) == expected_epochs
                and "epoch" in history.columns
                and history["epoch"].astype(int).tolist()
                == list(range(expected_epochs))
                and int(selection.get("best_epoch_zero_based", -1))
                == expected_epochs - 1
                and int(selection.get("epochs_trained", -1)) == expected_epochs
                and selection.get("early_stopping_enabled") is False
                and selection.get("validation_used_for_checkpoint_selection") is False
                and selection.get(
                    "test_embeddings_or_labels_loaded_during_training"
                ) is False
                and summary.get("selection") == selection
                and np.isclose(
                    float(selection.get("best_score", np.nan)),
                    float(history.iloc[-1]["validation_auroc"]),
                )
            )
        common = (
            checkpoint.get("schema") == SCHEMA
            and checkpoint.get("stage") == expected["stage"]
            and checkpoint.get("variant") == expected["variant"]
            and checkpoint.get("temporal_depth") == expected["depth"]["name"]
            and int(checkpoint.get("seed", -1)) == int(expected["seed"])
            and int(checkpoint.get("fold", -1)) == int(expected["fold"])
            and checkpoint.get("effective_config") == expected["effective_config"]
            and checkpoint.get("selection", {}).get("criterion")
            == _selection_criterion(checkpoint_policy)
            and _recorded_checkpoint_policy(checkpoint) == checkpoint_policy
            and _validation_selection_matches(checkpoint, checkpoint_policy)
            and checkpoint.get("train_ids") == list(expected["train_ids"])
            and checkpoint.get("validation_ids") == list(expected["validation_ids"])
            and checkpoint.get("test_data_loaded_during_training") is False
            and _training_test_isolation_matches(checkpoint, checkpoint_policy)
            and checkpoint.get("clinical_prior", {}).get("fitted_on") == "fold_train_only"
            and checkpoint.get("clinical_prior", {}).get("feature_names")
            == list(TABULAR_FEATURE_NAMES)
            and sentinel.get("schema") == SCHEMA
            and sentinel.get("complete") is True
            and sentinel.get("effective_config") == expected["effective_config"]
            and _recorded_checkpoint_policy(sentinel) == checkpoint_policy
            and _validation_selection_matches(sentinel, checkpoint_policy)
            and _training_test_isolation_matches(sentinel, checkpoint_policy)
            and summary.get("schema") == SCHEMA
            and _recorded_checkpoint_policy(summary) == checkpoint_policy
            and _validation_selection_matches(summary, checkpoint_policy)
            and _training_test_isolation_matches(summary, checkpoint_policy)
            and final_epoch_contract
        )
        return common and _valid_prediction_frame(
            train,
            expected["train_ids"],
            expected["seed"],
            expected["fold"],
            "fold_train",
            expected["depth"],
        ) and _valid_prediction_frame(
            val,
            expected["validation_ids"],
            expected["seed"],
            expected["fold"],
            "oof_val",
            expected["depth"],
        )
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
        return False


def _train_fold_task(task):
    (
        config_path,
        output_dir,
        stage,
        depth,
        variant,
        seed,
        spec,
        device_text,
        smoke,
        force,
    ) = task
    config = _release_yaml(Path(config_path).read_text())
    fold = int(spec["fold"])
    effective = _effective_config(config, variant, depth["name"], smoke=smoke)
    checkpoint_policy = _checkpoint_policy(effective)
    run_dir = _training_dir(output_dir, stage, depth["name"], variant, seed, fold)
    expected = {
        "stage": stage,
        "depth": depth,
        "variant": variant,
        "seed": int(seed),
        "fold": fold,
        "effective_config": effective,
        "train_ids": spec["train_ids"],
        "validation_ids": spec["val_ids"],
    }
    if not force and _training_complete(run_dir, expected):
        return f"skip {stage} {depth['name']} {variant} seed={seed} fold={fold}"

    task_seed = int(seed) * 1000 + fold * 10 + int(depth["max_tp"])
    _set_seed(task_seed)
    torch.set_num_threads(1)
    device = _resolve_device(device_text)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train = _canonical_split(
        _load_split(
            _repo_path(config["data"]["train_embeddings_dir"]),
            _repo_path(config["data"]["metadata_csv"]),
            spec["train_ids"],
        ),
        depth["max_tp"],
    )
    val = _canonical_split(
        _load_split(
            _repo_path(config["data"]["train_embeddings_dir"]),
            _repo_path(config["data"]["metadata_csv"]),
            spec["val_ids"],
        ),
        depth["max_tp"],
    )
    if train["dim"] != int(effective["input_dim"]) or val["dim"] != int(effective["input_dim"]):
        raise ValueError("embedding dimension mismatch")
    train_prior, val_prior, prior_state = _fit_prior(train, val, task_seed)
    has_t0 = train["masks"][:, 0] > 0
    if not has_t0.any():
        raise ValueError("independent temporal training fold has no valid T0")
    train_for_loss = _subset_split(train, has_t0)
    prior_for_loss = np.asarray(train_prior)[has_t0]
    train_loader = _loader(
        train_for_loss, prior_for_loss, effective["batch_size"], True, task_seed
    )
    model = TDN({"downstream": effective}).to(device)
    optimizer = _make_optimizer(model, effective)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(effective.get("scheduler_t_max", effective["epochs"])),
        eta_min=float(effective.get("scheduler_eta_min", 0.0)),
    )
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [_positive_class_weight(effective, train_for_loss["labels"])],
            device=device,
        )
    )

    best_score = -np.inf
    best_state = None
    best_epoch = -1
    patience = 0
    history = []
    for epoch in range(int(effective["epochs"])):
        model.train()
        losses = []
        for embeddings, masks, clinical, target, prior_logit, days in train_loader:
            embeddings = embeddings.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prior_logit = prior_logit.to(device, non_blocking=True)
            days = days.to(device, non_blocking=True)
            sampled_depths = _sample_prefix_depths(
                embeddings.shape[0],
                depth["max_tp"],
                effective.get("prefix_dropout_probability", 0.0),
                device,
            )
            embeddings, masks, days = mask_torch_temporal_prefix(
                embeddings, masks, days, sampled_depths
            )
            optimizer.zero_grad(set_to_none=True)
            logits, residual = model(
                embeddings,
                masks,
                clinical,
                days=days,
                prior_logit=prior_logit,
                return_residual=True,
            )
            loss = loss_fn(logits, target)
            residual_weight = float(effective.get("residual_l2_weight", 0.0))
            if residual_weight > 0:
                loss = loss + residual_weight * residual.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            if float(effective.get("grad_clip", 0.0)) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(effective["grad_clip"]))
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        val_predictions = _prediction_frame(
            model,
            val,
            val_prior,
            depth,
            seed,
            fold,
            "oof_val",
            device,
            effective["batch_size"],
        )
        score = float(roc_auc_score(val_predictions["label"], val_predictions["probability"]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "residual_scale": float(model.residual_scale().detach().cpu()),
                "validation_auroc": score,
            }
        )
        if checkpoint_policy == "final_epoch":
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            patience = 0
        elif score > best_score + float(effective.get("min_delta", 0.0)):
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if (
            checkpoint_policy == "best_validation"
            and patience >= int(effective["patience"])
        ):
            break
    if best_state is None:
        raise RuntimeError("validation never produced a finite checkpoint")

    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    train_predictions = _prediction_frame(
        model,
        train,
        train_prior,
        depth,
        seed,
        fold,
        "fold_train",
        device,
        effective["batch_size"],
    )
    val_predictions = _prediction_frame(
        model,
        val,
        val_prior,
        depth,
        seed,
        fold,
        "oof_val",
        device,
        effective["batch_size"],
    )
    train_metrics = compute_metrics(
        train_predictions["label"], train_predictions["probability"], threshold=0.5
    )
    val_metrics = compute_metrics(
        val_predictions["label"], val_predictions["probability"], threshold=0.5
    )
    selection = {
        "checkpoint_policy": checkpoint_policy,
        "criterion": _selection_criterion(checkpoint_policy),
        "validation_used_for_checkpoint_selection": checkpoint_policy
        == "best_validation",
        "test_embeddings_or_labels_loaded_during_training": False,
        "best_epoch_zero_based": int(best_epoch),
        "best_score": float(best_score),
        "epochs_trained": len(history),
        "early_stopping_enabled": checkpoint_policy == "best_validation",
    }
    checkpoint = {
        "schema": SCHEMA,
        "stage": stage,
        "variant": variant,
        "temporal_depth": depth["name"],
        "max_tp": int(depth["max_tp"]),
        "seed": int(seed),
        "fold": fold,
        "model_state": best_state,
        "effective_config": effective,
        "checkpoint_policy": checkpoint_policy,
        "validation_used_for_checkpoint_selection": checkpoint_policy
        == "best_validation",
        "clinical_prior": prior_state,
        "selection": selection,
        "train_ids": list(spec["train_ids"]),
        "validation_ids": list(spec["val_ids"]),
        "test_data_loaded_during_training": False,
        "test_embeddings_or_labels_loaded_during_training": False,
        "training_view_policy": "one_view_per_patient_optional_contiguous_suffix_truncation",
    }
    _atomic_torch(run_dir / "best.pt", checkpoint)
    _atomic_csv(run_dir / "history.csv", pd.DataFrame(history))
    _atomic_csv(run_dir / "train_predictions.csv", train_predictions)
    _atomic_csv(run_dir / "val_predictions.csv", val_predictions)
    _atomic_json(
        run_dir / "summary.json",
        {
            "schema": SCHEMA,
            "stage": stage,
            "variant": variant,
            "temporal_depth": depth["name"],
            "max_tp": int(depth["max_tp"]),
            "seed": int(seed),
            "fold": fold,
            "checkpoint_policy": checkpoint_policy,
            "validation_used_for_checkpoint_selection": checkpoint_policy
            == "best_validation",
            "test_embeddings_or_labels_loaded_during_training": False,
            "selection": selection,
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "clinical_prior_training_patients": len(train["pids"]),
            "tdn_training_patients": len(train_for_loss["pids"]),
            "test_role": "not_loaded",
        },
    )
    _atomic_json(
        run_dir / "TRAINING_COMPLETE.json",
        {
            "schema": SCHEMA,
            "stage": stage,
            "variant": variant,
            "temporal_depth": depth["name"],
            "seed": int(seed),
            "fold": fold,
            "effective_config": effective,
            "checkpoint_policy": checkpoint_policy,
            "validation_used_for_checkpoint_selection": checkpoint_policy
            == "best_validation",
            "test_embeddings_or_labels_loaded_during_training": False,
            "complete": True,
        },
    )
    return (
        f"done {stage} {depth['name']} {variant} seed={seed} fold={fold} "
        f"val={best_score:.4f}"
    )


def _select_variant(
    summary,
    variant_order,
    tolerance,
    prauc_tolerance=None,
    tiebreak="parameters_then_gap",
):
    table = summary.copy()
    best = float(table["oof_auroc_mean"].max())
    table["within_oof_tolerance"] = table["oof_auroc_mean"] >= best - float(tolerance)
    if prauc_tolerance is None:
        table["within_prauc_tolerance"] = True
    else:
        prauc_tolerance = float(prauc_tolerance)
        if prauc_tolerance < 0:
            raise ValueError("selection_prauc_tolerance must be non-negative")
        best_prauc = float(
            table.loc[table["within_oof_tolerance"], "oof_prauc_mean"].max()
        )
        table["within_prauc_tolerance"] = (
            table["oof_prauc_mean"] >= best_prauc - prauc_tolerance
        )
    table["variant_order"] = table["variant"].map(
        {name: index for index, name in enumerate(variant_order)}
    )
    eligible = table[
        table["within_oof_tolerance"] & table["within_prauc_tolerance"]
    ].copy()
    if tiebreak == "parameters_then_gap":
        columns = ["parameters", "positive_train_oof_gap", "oof_auroc_mean", "variant_order"]
    elif tiebreak == "gap_then_parameters":
        columns = ["positive_train_oof_gap", "parameters", "oof_auroc_mean", "variant_order"]
    else:
        raise ValueError(f"unsupported selection_tiebreak: {tiebreak}")
    selected = str(
        eligible.sort_values(
            columns,
            ascending=[True, True, False, True],
        ).iloc[0]["variant"]
    )
    table["selected"] = table["variant"] == selected
    return selected, table.sort_values("variant_order")


def _formal_variant_map(config, output_dir, stage, variants):
    if stage != "formal":
        return None
    path = Path(output_dir) / "selected_variants.json"
    selection = json.loads(path.read_text())
    depths = _depths(config)
    expected_depths = {depth["name"] for depth in depths}
    if (
        selection.get("schema") != SCHEMA
        or selection.get("stage") != "tune"
        or selection.get("test_data_used") is not False
        or not _selection_test_isolation_matches(selection)
        or set(selection.get("depths", {})) != expected_depths
    ):
        raise RuntimeError(f"selected variants do not match formal OOF contract: {path}")

    allowed = set(variants)
    result = {}
    for depth in depths:
        selected = selection["depths"][depth["name"]]
        variant = selected.get("variant")
        policy_matches = True
        if "checkpoint_policy" in selected:
            policy_matches = (
                variant in config["variants"]
                and selected["checkpoint_policy"]
                == _checkpoint_policy(
                    _effective_config(config, variant, depth["name"])
                )
            )
        if (
            variant not in allowed
            or variant not in config["variants"]
            or int(selected.get("max_tp", -1)) != int(depth["max_tp"])
            or not policy_matches
        ):
            raise RuntimeError(
                f"invalid selected variant for formal OOF depth {depth['name']}: {variant}"
            )
        result[depth["name"]] = variant
    return result


def _build_oof(config, output_dir, specs, stage, variants, seeds, select, smoke):
    pool_ids = [patient_id for spec in specs for patient_id in spec["val_ids"]]
    if len(pool_ids) != len(set(pool_ids)):
        raise RuntimeError("fold validation sets are not a disjoint development partition")
    formal_variants = _formal_variant_map(config, output_dir, stage, variants)
    metric_rows = []
    for depth in _depths(config):
        depth_variants = (
            (formal_variants[depth["name"]],)
            if formal_variants is not None
            else variants
        )
        for variant in depth_variants:
            effective = _effective_config(config, variant, depth["name"], smoke=smoke)
            checkpoint_policy = _checkpoint_policy(effective)
            for seed in seeds:
                frames = []
                fold_train_aurocs = []
                best_epochs = []
                for spec in specs:
                    fold = int(spec["fold"])
                    run_dir = _training_dir(
                        output_dir, stage, depth["name"], variant, seed, fold
                    )
                    expected = {
                        "stage": stage,
                        "depth": depth,
                        "variant": variant,
                        "seed": int(seed),
                        "fold": fold,
                        "effective_config": effective,
                        "train_ids": spec["train_ids"],
                        "validation_ids": spec["val_ids"],
                    }
                    if not _training_complete(run_dir, expected):
                        raise RuntimeError(f"incomplete training artifact: {run_dir}")
                    frames.append(
                        pd.read_csv(run_dir / "val_predictions.csv", dtype={"patient_id": str})
                    )
                    summary = json.loads((run_dir / "summary.json").read_text())
                    fold_train_aurocs.append(float(summary["train_metrics"]["auroc"]))
                    best_epochs.append(int(summary["selection"]["best_epoch_zero_based"]))
                oof = pd.concat(frames, ignore_index=True)
                if (
                    set(oof["patient_id"]) != set(pool_ids)
                    or oof["patient_id"].duplicated().any()
                    or len(oof) != len(pool_ids)
                ):
                    raise RuntimeError("OOF predictions do not cover the development pool exactly")
                metrics = compute_metrics(oof["label"], oof["probability"], threshold=0.5)
                prior_metrics = compute_metrics(
                    oof["label"], oof["clinical_prior_probability"], threshold=0.5
                )
                mean_train = float(np.mean(fold_train_aurocs))
                metric_rows.append(
                    {
                        "stage": stage,
                        "temporal_depth": depth["name"],
                        "max_tp": int(depth["max_tp"]),
                        "variant": variant,
                        "seed": int(seed),
                        "checkpoint_policy": checkpoint_policy,
                        "mean_fold_train_auroc": mean_train,
                        "train_oof_gap": mean_train - float(metrics["auroc"]),
                        "mean_best_epoch": float(np.mean(best_epochs)),
                        "parameters": _parameter_count(config, variant, depth["name"], smoke=smoke),
                        "brier": float(
                            np.mean(
                                (
                                    oof["probability"].to_numpy(dtype=float)
                                    - oof["label"].to_numpy(dtype=float)
                                )
                                ** 2
                            )
                        ),
                        "clinical_prior_auroc": float(prior_metrics["auroc"]),
                        "clinical_prior_prauc": float(prior_metrics["prauc"]),
                        "clinical_prior_brier": float(
                            np.mean(
                                (
                                    oof["clinical_prior_probability"].to_numpy(dtype=float)
                                    - oof["label"].to_numpy(dtype=float)
                                )
                                ** 2
                            )
                        ),
                        **metrics,
                    }
                )
                oof_dir = (
                    Path(output_dir)
                    / "oof"
                    / stage
                    / _slug(depth["name"])
                    / f"variant_{variant}"
                    / f"seed_{int(seed)}"
                )
                _atomic_csv(oof_dir / "oof_predictions.csv", oof)
                _atomic_json(
                    oof_dir / "OOF_COMPLETE.json",
                    {
                        "schema": SCHEMA,
                        "stage": stage,
                        "temporal_depth": depth["name"],
                        "variant": variant,
                        "seed": int(seed),
                        "checkpoint_policy": checkpoint_policy,
                        "folds": len(specs),
                        "complete": True,
                    },
                )
    metrics = pd.DataFrame(metric_rows)
    _atomic_csv(Path(output_dir) / f"{stage}_oof_metrics_all.csv", metrics)
    summary = (
        metrics.groupby(["temporal_depth", "max_tp", "variant"], sort=False)
        .agg(
            oof_auroc_mean=("auroc", "mean"),
            oof_auroc_std=("auroc", "std"),
            oof_prauc_mean=("prauc", "mean"),
            oof_prauc_std=("prauc", "std"),
            oof_brier_mean=("brier", "mean"),
            clinical_prior_auroc_mean=("clinical_prior_auroc", "mean"),
            clinical_prior_prauc_mean=("clinical_prior_prauc", "mean"),
            clinical_prior_brier_mean=("clinical_prior_brier", "mean"),
            checkpoint_policy=("checkpoint_policy", "first"),
            mean_fold_train_auroc=("mean_fold_train_auroc", "mean"),
            train_oof_gap_mean=("train_oof_gap", "mean"),
            mean_best_epoch=("mean_best_epoch", "mean"),
            parameters=("parameters", "first"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    summary["positive_train_oof_gap"] = summary["train_oof_gap_mean"].clip(lower=0)
    selection = {
        "schema": SCHEMA,
        "stage": stage,
        "criterion": config["independent_cv"]["selection"],
        "selection_tolerance": float(config["independent_cv"]["selection_tolerance"]),
        "tune_seeds": [int(seed) for seed in seeds],
        "folds": len(specs),
        "test_data_used": False,
        "test_embeddings_or_labels_loaded": False,
        "depths": {},
    }
    decorated = []
    for depth in _depths(config):
        depth_summary = summary[summary["temporal_depth"] == depth["name"]].copy()
        if select:
            selected, depth_summary = _select_variant(
                depth_summary,
                variants,
                config["independent_cv"]["selection_tolerance"],
                config["independent_cv"].get("selection_prauc_tolerance"),
                config["independent_cv"].get(
                    "selection_tiebreak", "parameters_then_gap"
                ),
            )
            selected_effective = _effective_config(
                config, selected, depth["name"], smoke=smoke
            )
            selection["depths"][depth["name"]] = {
                "max_tp": int(depth["max_tp"]),
                "variant": selected,
                "checkpoint_policy": _checkpoint_policy(selected_effective),
                "effective_config": selected_effective,
            }
        decorated.append(depth_summary)
    summary = pd.concat(decorated, ignore_index=True)
    _atomic_csv(Path(output_dir) / f"{stage}_variant_summary.csv", summary)
    if select:
        _atomic_json(Path(output_dir) / "selected_variants.json", selection)
        return selection
    return None


def _evaluation_dir(output_dir, source, depth_name, seed):
    return (
        Path(output_dir)
        / "evaluation"
        / source
        / _slug(depth_name)
        / f"seed_{int(seed)}"
    )


def _evaluate_task(task):
    (
        config_path,
        output_dir,
        selection,
        source,
        depth,
        seed,
        specs,
        test_ids,
        device_text,
        smoke,
    ) = task
    config = _release_yaml(Path(config_path).read_text())
    selected = selection["depths"][depth["name"]]
    variant = selected["variant"]
    effective = _effective_config(config, variant, depth["name"], smoke=smoke)
    for spec in specs:
        expected = {
            "stage": "formal",
            "depth": depth,
            "variant": variant,
            "seed": int(seed),
            "fold": int(spec["fold"]),
            "effective_config": effective,
            "train_ids": spec["train_ids"],
            "validation_ids": spec["val_ids"],
        }
        run_dir = _training_dir(
            output_dir,
            "formal",
            depth["name"],
            variant,
            seed,
            spec["fold"],
        )
        if not _training_complete(run_dir, expected):
            raise RuntimeError(f"formal training contract mismatch: {run_dir}")

    test = _canonical_split(_load_test_source(config, source, test_ids), depth["max_tp"])
    device = _resolve_device(device_text)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fold_frames = []
    for spec in specs:
        fold = int(spec["fold"])
        checkpoint_path = (
            _training_dir(output_dir, "formal", depth["name"], variant, seed, fold)
            / "best.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = TDN({"downstream": effective}).to(device)
        model.load_state_dict(checkpoint["model_state"])
        prior = _prior_from_state(test["clinical"], checkpoint["clinical_prior"])
        frame = _prediction_frame(
            model,
            test,
            prior,
            depth,
            seed,
            fold,
            "test_fold",
            device,
            effective["batch_size"],
        )
        frame["source"] = source
        fold_frames.append(frame)
        del model
    fold_predictions = pd.concat(fold_frames, ignore_index=True)
    ensemble = (
        fold_predictions.groupby(
            ["patient_id", "label", "seed", "temporal_depth", "max_tp", "source"],
            as_index=False,
            sort=False,
        )[["probability", "residual_logit", "clinical_prior_probability"]]
        .mean()
    )
    ensemble["fold"] = -1
    ensemble["split"] = "test_fold_ensemble"
    order = {patient_id: index for index, patient_id in enumerate(test_ids)}
    ensemble["_order"] = ensemble["patient_id"].map(order)
    ensemble = ensemble.sort_values("_order").drop(columns="_order")
    ensemble = ensemble[
        [
            "patient_id",
            "label",
            "probability",
            "residual_logit",
            "clinical_prior_probability",
            "seed",
            "fold",
            "split",
            "temporal_depth",
            "max_tp",
            "source",
        ]
    ]
    run_dir = _evaluation_dir(output_dir, source, depth["name"], seed)
    _atomic_csv(run_dir / "fold_predictions.csv", fold_predictions)
    _atomic_csv(run_dir / "test_predictions.csv", ensemble)
    _atomic_json(
        run_dir / "EVALUATION_COMPLETE.json",
        {
            "schema": SCHEMA,
            "source": source,
            "temporal_depth": depth["name"],
            "max_tp": int(depth["max_tp"]),
            "variant": variant,
            "checkpoint_policy": _checkpoint_policy(effective),
            "seed": int(seed),
            "folds_ensembled": len(specs),
            "test_ids": list(test_ids),
            "test_used_for_selection": False,
            "complete": True,
        },
    )
    return f"done evaluation {source} {depth['name']} seed={seed}"


def _aggregate_evaluation(config, output_dir, selection, sources, seeds, test_ids):
    for source in sources:
        frames = []
        metric_rows = []
        ensemble_rows = []
        for depth in _depths(config):
            depth_frames = []
            for seed in seeds:
                run_dir = _evaluation_dir(output_dir, source, depth["name"], seed)
                sentinel = json.loads((run_dir / "EVALUATION_COMPLETE.json").read_text())
                if (
                    sentinel.get("schema") != SCHEMA
                    or sentinel.get("complete") is not True
                    or sentinel.get("test_ids") != list(test_ids)
                    or sentinel.get("variant")
                    != selection["depths"][depth["name"]]["variant"]
                    or _recorded_checkpoint_policy(sentinel)
                    != _recorded_checkpoint_policy(selection["depths"][depth["name"]])
                ):
                    raise RuntimeError(f"invalid evaluation artifact: {run_dir}")
                frame = pd.read_csv(run_dir / "test_predictions.csv", dtype={"patient_id": str})
                if not _valid_prediction_frame(
                    frame,
                    test_ids,
                    seed,
                    -1,
                    "test_fold_ensemble",
                    depth,
                ):
                    raise RuntimeError(f"invalid test predictions: {run_dir}")
                if set(frame["source"]) != {source}:
                    raise RuntimeError("test prediction source mismatch")
                frames.append(frame)
                depth_frames.append(frame)
                metric_rows.append(
                    {
                        "source": source,
                        "temporal_depth": depth["name"],
                        "max_tp": int(depth["max_tp"]),
                        "seed": int(seed),
                        **compute_metrics(
                            frame["label"], frame["probability"], threshold=0.5
                        ),
                    }
                )
            combined = pd.concat(depth_frames, ignore_index=True)
            grand = (
                combined.groupby(["patient_id", "label"], as_index=False, sort=False)[
                    "probability"
                ].mean()
            )
            ensemble_rows.append(
                {
                    "source": source,
                    "temporal_depth": depth["name"],
                    "max_tp": int(depth["max_tp"]),
                    "models_ensembled": len(seeds) * int(config["independent_cv"]["folds"]),
                    **compute_metrics(grand["label"], grand["probability"], threshold=0.5),
                }
            )
        metrics = pd.DataFrame(metric_rows)
        summary_rows = []
        for depth in _depths(config):
            selected = metrics[metrics["temporal_depth"] == depth["name"]]
            row = {
                "source": source,
                "temporal_depth": depth["name"],
                "max_tp": int(depth["max_tp"]),
                "variant": selection["depths"][depth["name"]]["variant"],
                "checkpoint_policy": _recorded_checkpoint_policy(
                    selection["depths"][depth["name"]]
                ),
                "n_seeds": len(seeds),
                "folds_per_seed": int(config["independent_cv"]["folds"]),
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
        _atomic_csv(source_dir / "all_seed_ensemble_metrics.csv", pd.DataFrame(ensemble_rows))


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
        "--config",
        default="configs/mewm_ispy2_full978_locked102_independent_cv.yaml",
    )
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--tune-seeds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = _repo_path(args.config)
    config = _release_yaml(config_path.read_text())
    section = config["independent_cv"]
    declared_variants = list(config["variants"])
    declared_tune_seeds = [int(seed) for seed in section["tune_seeds"]]
    declared_seeds = [int(seed) for seed in section["formal_seeds"]]
    declared_sources = list(config["evaluation_sources"])
    variants = args.variants or declared_variants
    tune_seeds = args.tune_seeds or (
        [declared_tune_seeds[0]] if args.smoke else declared_tune_seeds
    )
    seeds = args.seeds or ([declared_seeds[0]] if args.smoke else declared_seeds)
    sources = args.sources or declared_sources
    if (
        len(variants) != len(set(variants))
        or set(variants) - set(declared_variants)
        or len(tune_seeds) != len(set(tune_seeds))
        or len(seeds) != len(set(seeds))
        or len(sources) != len(set(sources))
        or set(sources) - set(declared_sources)
    ):
        raise SystemExit("variants, seeds, and sources must be unique and declared")
    if not args.smoke and (
        variants != declared_variants
        or tune_seeds != declared_tune_seeds
        or seeds != declared_seeds
        or sources != declared_sources
    ):
        raise SystemExit("formal runs require every declared variant, seed, and source")

    output_dir = _repo_path(args.output_dir or section["output_dir"])
    if args.smoke and args.output_dir is None:
        output_dir = output_dir / "smoke"
    if args.smoke and output_dir == _repo_path(section["output_dir"]):
        raise SystemExit("smoke runs may not write to the formal output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    specs, pool_ids, test_ids, fold_audit = _fold_specs(config)
    jobs = int(args.jobs or section.get("jobs", 10))
    if jobs < 1:
        raise SystemExit("jobs must be positive")
    _write_fold_manifests(output_dir, specs, config, fold_audit)
    _atomic_text(output_dir / "resolved_config.yaml", yaml.safe_dump(config, sort_keys=True))
    _atomic_json(
        output_dir / "run_manifest.json",
        {
            "schema": SCHEMA,
            "config": str(config_path),
            "phase": args.phase,
            "development_patients": len(pool_ids),
            "locked_test_patients": len(test_ids),
            "temporal_depths": _depths(config),
            "variants": variants,
            "checkpoint_policies": {
                depth["name"]: {
                    variant: _checkpoint_policy(
                        _effective_config(config, variant, depth["name"], smoke=args.smoke)
                    )
                    for variant in variants
                }
                for depth in _depths(config)
            },
            "tune_seeds": tune_seeds,
            "formal_seeds": seeds,
            "sources": sources,
            "folds": len(specs),
            "jobs": jobs,
            "parallel_start_method": "spawn",
            "test_policy": "loaded_only_after_depthwise_oof_selection",
            "smoke": bool(args.smoke),
        },
    )

    selection = None
    if args.phase in ("train", "all"):
        tune_tasks = []
        for depth in _depths(config):
            for variant in variants:
                for seed in tune_seeds:
                    for spec in specs:
                        tune_tasks.append(
                            (
                                _train_fold_task,
                                str(config_path),
                                str(output_dir),
                                "tune",
                                depth,
                                variant,
                                seed,
                                spec,
                                args.device,
                                bool(args.smoke),
                                bool(args.force),
                            )
                        )
        _run_tasks(tune_tasks, jobs, "tune")
        selection = _build_oof(
            config, output_dir, specs, "tune", variants, tune_seeds, True, bool(args.smoke)
        )
        formal_tasks = []
        for depth in _depths(config):
            selected_variant = selection["depths"][depth["name"]]["variant"]
            for seed in seeds:
                for spec in specs:
                    formal_tasks.append(
                        (
                            _train_fold_task,
                            str(config_path),
                            str(output_dir),
                            "formal",
                            depth,
                            selected_variant,
                            seed,
                            spec,
                            args.device,
                            bool(args.smoke),
                            bool(args.force),
                        )
                    )
        _run_tasks(formal_tasks, jobs, "formal")
        _build_oof(
            config,
            output_dir,
            specs,
            "formal",
            variants,
            seeds,
            False,
            bool(args.smoke),
        )
        _atomic_json(
            output_dir / "TRAINING_PHASE_COMPLETE.json",
            {
                "schema": SCHEMA,
                "selected_variants": selection["depths"],
                "tune_checkpoints": len(_depths(config)) * len(variants)
                * len(tune_seeds) * len(specs),
                "formal_checkpoints": len(_depths(config)) * len(seeds) * len(specs),
                "test_data_loaded": False,
                "test_embeddings_or_labels_loaded": False,
                "complete": True,
            },
        )

    if args.phase in ("evaluate", "all"):
        if selection is None:
            selection = json.loads((output_dir / "selected_variants.json").read_text())
        expected_selection = {
            depth["name"] for depth in _depths(config)
        }
        if (
            selection.get("schema") != SCHEMA
            or selection.get("test_data_used") is not False
            or not _selection_test_isolation_matches(selection)
            or set(selection.get("depths", {})) != expected_selection
            or int(selection.get("folds", -1)) != len(specs)
        ):
            raise SystemExit("selected variants do not match the experiment contract")
        if not _training_phase_complete(
            output_dir / "TRAINING_PHASE_COMPLETE.json",
            selection,
            len(_depths(config)) * len(variants) * len(tune_seeds) * len(specs),
            len(_depths(config)) * len(seeds) * len(specs),
        ):
            raise SystemExit(
                "training phase does not match the test-data isolation contract"
            )
        evaluation_tasks = []
        for source in sources:
            for depth in _depths(config):
                for seed in seeds:
                    evaluation_tasks.append(
                        (
                            _evaluate_task,
                            str(config_path),
                            str(output_dir),
                            selection,
                            source,
                            depth,
                            seed,
                            specs,
                            test_ids,
                            args.device,
                            bool(args.smoke),
                        )
                    )
        _run_tasks(evaluation_tasks, jobs, "evaluate")
        _aggregate_evaluation(config, output_dir, selection, sources, seeds, test_ids)
        _atomic_json(
            output_dir / "EXPERIMENT_COMPLETE.json",
            {
                "schema": SCHEMA,
                "selected_variants": selection["depths"],
                "formal_seeds": seeds,
                "folds_per_seed": len(specs),
                "evaluation_sources": sources,
                "locked_test_patients": len(test_ids),
                "test_used_for_selection": False,
                "complete": True,
            },
        )
        print(f"complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
