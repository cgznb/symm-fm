"""Held-out posterior-mean reconstruction evaluation for the shared 3D autoencoder."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ispy2_symmflow.models import build_autoencoder_from_config
from ispy2_symmflow.training.checkpoint import load_checkpoint
from ispy2_symmflow.training.datasets import load_prepared_image, read_jsonl
from ispy2_symmflow.utils.hashing import sha256_file, stable_hash

from ._validation import phase_roles, preprocessing_signature, required_text
from .aggregate import aggregate_by_patient, patient_bootstrap_mean_ci
from .metrics import evaluate_prediction


HELD_OUT_SPLITS = frozenset(("val", "test"))
WHOLE_IMAGE_METRICS = ("mae", "mse", "psnr", "ssim")


def _validate_training_manifest(
    records: Sequence[Mapping[str, Any]], header: Mapping[str, Any]
) -> dict[str, Any]:
    if not records:
        raise ValueError("prepared visit manifest is empty")
    patient_splits: dict[str, str] = {}
    visit_ids: set[str] = set()
    for record in records:
        patient = str(record.get("patient_id", "")).strip()
        visit_id = str(record.get("visit_id", "")).strip()
        record_split = str(record.get("split", "")).strip()
        if not patient or not visit_id or not record_split:
            raise ValueError(
                "all prepared visit records require patient_id, visit_id, and split"
            )
        if visit_id in visit_ids:
            raise ValueError(f"prepared visit manifest contains duplicate visit_id {visit_id!r}")
        visit_ids.add(visit_id)
        previous = patient_splits.setdefault(patient, record_split)
        if previous != record_split:
            raise ValueError(
                f"patient {patient!r} occurs in both {previous!r} and {record_split!r}"
            )

    split_fingerprint = stable_hash(patient_splits)
    checkpoint_split_hash = str(header.get("split_hash", "")).strip()
    if not checkpoint_split_hash:
        raise ValueError("autoencoder checkpoint has no training split_hash")
    if split_fingerprint != checkpoint_split_hash:
        raise ValueError(
            "autoencoder checkpoint split_hash differs from the complete visit manifest"
        )

    extra = header.get("extra")
    training_signature = extra.get("training_signature") if isinstance(extra, Mapping) else None
    if not isinstance(training_signature, Mapping):
        raise ValueError("autoencoder checkpoint has no training_signature")
    ordered_fingerprint = stable_hash(records)
    if str(training_signature.get("ordered_manifest_fingerprint", "")) != ordered_fingerprint:
        raise ValueError(
            "autoencoder checkpoint ordered manifest fingerprint differs from the complete visit manifest"
        )
    if int(training_signature.get("manifest_record_count", -1)) != len(records):
        raise ValueError(
            "autoencoder checkpoint manifest record count differs from the complete visit manifest"
        )
    if training_signature.get("stage") != "autoencoder":
        raise ValueError("autoencoder checkpoint training_signature has the wrong stage")
    checkpoint_config = header.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("autoencoder checkpoint has no training configuration")
    if str(training_signature.get("config_fingerprint", "")) != stable_hash(
        checkpoint_config
    ):
        raise ValueError("autoencoder checkpoint training_signature has an invalid config fingerprint")
    return {
        "split_hash": split_fingerprint,
        "ordered_manifest_fingerprint": ordered_fingerprint,
        "record_count": len(records),
    }


def _finite(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("a per-case metric unexpectedly contains more than one value")
        value = value.detach().cpu().item()
    number = float(value)
    return number if math.isfinite(number) else None


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        resolved = device
    elif device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("autoencoder evaluation device must be auto, cpu, or cuda")
    return resolved


def _resolve_prepared_path(value: Any, *, base: Path, visit_id: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"visit {visit_id!r} has no prepared_path")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _masked_mae(error: torch.Tensor, mask: torch.Tensor) -> tuple[float | None, bool]:
    selected = error.abs()[mask]
    if selected.numel() == 0:
        return None, False
    return _finite(selected.mean()), True


def _patient_ci(
    values: Sequence[tuple[str, float | None]],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int | None | str]:
    records = [
        {"patient_id": patient_id, "value": value}
        for patient_id, value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not records:
        return {
            "mean": None,
            "lower": None,
            "upper": None,
            "patient_count": 0,
            "confidence": float(confidence),
            "status": "not_available: no finite patient values",
        }
    patient_values = aggregate_by_patient(records, metric="value")
    return patient_bootstrap_mean_ci(
        patient_values, samples=samples, confidence=confidence, seed=seed
    )


def _aggregate_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    roles: Sequence[str],
    enhancement_roles: Sequence[str],
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    def interval(values: Sequence[tuple[str, float | None]]) -> dict[str, Any]:
        return _patient_ci(
            values,
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )

    whole = {
        name: interval(
            [
                (str(case["patient_id"]), case["whole_image"].get(name))
                for case in cases
            ]
        )
        for name in WHOLE_IMAGE_METRICS
    }
    foreground = {
        name: interval(
            [
                (str(case["patient_id"]), case["foreground"].get(name))
                for case in cases
            ]
        )
        for name in ("mae", "mse")
    }
    by_phase = {
        role: {
            name: interval(
                [
                    (str(case["patient_id"]), case["phase_mae"][role].get(name))
                    for case in cases
                ]
            )
            for name in ("mae", "foreground_mae")
        }
        for role in roles
    }
    enhancement = {
        role: {
            name: interval(
                [
                    (
                        str(case["patient_id"]),
                        case["enhancement_difference"]["phases"][role].get(name),
                    )
                    for case in cases
                ]
            )
            for name in ("mae", "foreground_mae")
        }
        for role in enhancement_roles
    }
    return {
        "whole_image": whole,
        "foreground": foreground,
        "phase_mae": by_phase,
        "enhancement_difference": enhancement,
    }


def evaluate_autoencoder_reconstruction(
    checkpoint: str | Path,
    visit_manifest: str | Path,
    *,
    split: str,
    data_range: float,
    foreground_threshold: float = -0.95,
    bootstrap_samples: int = 1000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 0,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Reconstruct held-out visits using the deterministic posterior mean."""

    if split not in HELD_OUT_SPLITS:
        raise ValueError("autoencoder reconstruction evaluation split must be val or test")
    if not math.isfinite(float(data_range)) or float(data_range) <= 0:
        raise ValueError("data_range must be finite and positive")
    if not math.isfinite(float(foreground_threshold)):
        raise ValueError("foreground_threshold must be finite")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0 < bootstrap_confidence < 1:
        raise ValueError("bootstrap_confidence must lie in (0, 1)")

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_path = Path(visit_manifest).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    records = read_jsonl(manifest_path)
    header = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(header, Mapping) or not isinstance(header.get("config"), Mapping):
        raise ValueError("autoencoder checkpoint has no training configuration")
    manifest_provenance = _validate_training_manifest(records, header)
    selected = [record for record in records if str(record.get("split", "")) == split]
    if not selected:
        raise ValueError(f"prepared visit manifest has no {split!r} records")

    checkpoint_config = header["config"]
    resolved_device = _resolve_device(device)
    model = build_autoencoder_from_config(checkpoint_config).to(resolved_device)
    load_checkpoint(checkpoint_path, model=model, map_location=resolved_device)
    model.freeze()

    configured_roles = tuple(
        str(value) for value in checkpoint_config.get("data", {}).get("phase_channels", ())
    )
    expected_channels = int(checkpoint_config.get("autoencoder", {}).get("in_channels", 0))
    seen_visits: set[str] = set()
    cases: list[dict[str, Any]] = []
    common_roles: tuple[str, ...] | None = None
    common_preprocessing: dict[str, Any] | None = None

    for record in selected:
        visit_id = str(record.get("visit_id", "")).strip()
        patient_id = str(record.get("patient_id", "")).strip()
        visit_stage = str(record.get("visit_stage", "")).strip()
        if not visit_id or not patient_id or not visit_stage:
            raise ValueError("prepared visit records require visit_id, patient_id, and visit_stage")
        if visit_id in seen_visits:
            raise ValueError(f"prepared visit manifest contains duplicate visit_id {visit_id!r}")
        seen_visits.add(visit_id)
        prepared_path = _resolve_prepared_path(
            record.get("prepared_path"), base=manifest_path.parent, visit_id=visit_id
        )
        image, metadata = load_prepared_image(prepared_path)
        identities = {
            "visit_id": visit_id,
            "patient_id": patient_id,
            "visit_stage": visit_stage,
            "split": split,
        }
        for key, expected in identities.items():
            observed = required_text(metadata, key, label=f"prepared visit {visit_id}")
            if observed != expected:
                raise ValueError(
                    f"prepared archive {key}={observed!r} disagrees with manifest {expected!r}"
                )

        roles = phase_roles(metadata, label=f"prepared visit {visit_id}")
        if configured_roles and roles != configured_roles:
            raise ValueError(
                f"visit {visit_id} phase roles {roles} differ from checkpoint {configured_roles}"
            )
        if expected_channels and image.shape[0] != expected_channels:
            raise ValueError(
                f"visit {visit_id} has {image.shape[0]} channels, checkpoint expects {expected_channels}"
            )
        if common_roles is None:
            common_roles = roles
        elif roles != common_roles:
            raise ValueError("held-out visits do not share one phase-role ordering")

        signature = preprocessing_signature(
            metadata, image_shape_cdhw=image.shape, label=f"prepared visit {visit_id}"
        )
        if common_preprocessing is None:
            common_preprocessing = signature
        elif signature != common_preprocessing:
            raise ValueError("held-out visits do not share one preprocessing provenance")

        batch = image[None].to(resolved_device)
        with torch.no_grad():
            posterior_mean = model.encode(batch, normalize=False)
            reconstruction = model.decode(posterior_mean, denormalize=False)
        reconstruction = reconstruction.detach().cpu()
        target = image[None]
        if reconstruction.shape != target.shape:
            raise ValueError(
                f"autoencoder reconstruction shape {tuple(reconstruction.shape)} differs from "
                f"input {tuple(target.shape)}"
            )
        foreground_mask = target > float(foreground_threshold)
        metrics = evaluate_prediction(
            reconstruction,
            target,
            data_range=float(data_range),
            foreground_mask=foreground_mask,
        )
        whole_image = {name: _finite(metrics[name]) for name in WHOLE_IMAGE_METRICS}
        foreground = {
            "mae": _finite(metrics["foreground_mae"]),
            "mse": _finite(metrics["foreground_mse"]),
            "valid": bool(metrics["foreground_valid"].item()),
        }

        error = reconstruction - target
        phase_metrics: dict[str, Any] = {}
        for channel, role in enumerate(roles):
            phase_error = error[:, channel]
            phase_foreground = target[:, channel] > float(foreground_threshold)
            foreground_mae, foreground_valid = _masked_mae(
                phase_error, phase_foreground
            )
            phase_metrics[role] = {
                "mae": _finite(phase_error.abs().mean()),
                "foreground_mae": foreground_mae,
                "foreground_valid": foreground_valid,
            }

        reference_index = roles.index("pre") if "pre" in roles else 0
        reference_role = roles[reference_index]
        enhancement: dict[str, Any] = {}
        for channel, role in enumerate(roles):
            if channel == reference_index:
                continue
            predicted_change = reconstruction[:, channel] - reconstruction[:, reference_index]
            target_change = target[:, channel] - target[:, reference_index]
            change_error = predicted_change - target_change
            change_foreground = (
                (target[:, channel] > float(foreground_threshold))
                | (target[:, reference_index] > float(foreground_threshold))
            )
            foreground_mae, foreground_valid = _masked_mae(
                change_error, change_foreground
            )
            enhancement[role] = {
                "mae": _finite(change_error.abs().mean()),
                "foreground_mae": foreground_mae,
                "foreground_valid": foreground_valid,
            }

        cases.append(
            {
                "patient_id": patient_id,
                "visit_id": visit_id,
                "visit_stage": visit_stage,
                "split": split,
                "prepared_path": str(prepared_path),
                "whole_image": whole_image,
                "foreground": foreground,
                "phase_mae": phase_metrics,
                "enhancement_difference": {
                    "reference_phase": reference_role,
                    "definition": (
                        "MAE((reconstruction_phase - reconstruction_reference), "
                        "(target_phase - target_reference))"
                    ),
                    "phases": enhancement,
                },
            }
        )

    assert common_roles is not None and common_preprocessing is not None
    enhancement_roles = tuple(role for role in common_roles if role != (
        "pre" if "pre" in common_roles else common_roles[0]
    ))
    aggregate = _aggregate_cases(
        cases,
        roles=common_roles,
        enhancement_roles=enhancement_roles,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "task": "autoencoder_reconstruction",
        "split": split,
        "case_count": len(cases),
        "patient_count": len({str(case["patient_id"]) for case in cases}),
        "posterior_representation": "mean",
        "stochastic_posterior_sampling": False,
        "data_range": float(data_range),
        "foreground_rule": f"target > {float(foreground_threshold)} (evaluation only)",
        "aggregation_rule": "average repeated visits within patient before patient bootstrap",
        "phase_roles": list(common_roles),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "training_split_hash": header.get("split_hash"),
            "format_version": header.get("format_version"),
            "upstream_commits": header.get("upstream_commits", {}),
        },
        "trusted_visit_manifest": {
            "path": str(manifest_path),
            **manifest_provenance,
        },
        "preprocessing_provenance": common_preprocessing,
        "aggregate": aggregate,
        "cases": cases,
        "tumor_metrics": {
            "status": "unavailable",
            "reason": (
                "No trusted lesion mask or independently validated tumor measurement model was "
                "provided; Dice, FTV, tumor volume, and tumor-change metrics were not computed."
            ),
        },
    }


__all__ = ["evaluate_autoencoder_reconstruction"]
