"""Deterministic patient-level data splits."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_RATIOS: Mapping[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    if not ratios or any(value < 0 for value in ratios.values()):
        raise ValueError("split ratios must be non-negative and non-empty")
    if abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to one")


def _stable_key(patient_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{patient_id}".encode("utf-8")).hexdigest()


def _quotas(size: int, ratios: Mapping[str, float]) -> dict[str, int]:
    raw = {name: size * ratio for name, ratio in ratios.items()}
    result = {name: int(value) for name, value in raw.items()}
    remaining = size - sum(result.values())
    order = sorted(
        ratios,
        key=lambda name: (raw[name] - result[name], ratios[name], name),
        reverse=True,
    )
    for name in order[:remaining]:
        result[name] += 1
    return result


def assign_patient_splits(
    patient_ids: Sequence[str],
    *,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    seed: int = 2026,
    strata: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Assign every patient exactly once, optionally within known strata.

    Stable SHA-256 ordering avoids Python hash randomization. Exact largest-
    remainder quotas are used for an unstratified cohort; stratified splits use
    the same rule inside each stratum and therefore remain approximate globally.
    """

    _validate_ratios(ratios)
    unique = sorted(set(patient_ids))
    if len(unique) != len(patient_ids):
        raise ValueError("patient_ids contains duplicates")
    if strata is not None and set(strata) != set(unique):
        missing = sorted(set(unique) - set(strata))
        extra = sorted(set(strata) - set(unique))
        raise ValueError(f"strata keys must match patient IDs; missing={missing}, extra={extra}")

    groups: dict[str, list[str]] = defaultdict(list)
    for patient_id in unique:
        groups[strata[patient_id] if strata is not None else "__all__"].append(patient_id)

    assignments: dict[str, str] = {}
    for stratum, members in sorted(groups.items()):
        ordered = sorted(members, key=lambda item: _stable_key(f"{stratum}\0{item}", seed))
        quotas = _quotas(len(ordered), ratios)
        offset = 0
        for split_name in ratios:
            count = quotas[split_name]
            for patient_id in ordered[offset : offset + count]:
                assignments[patient_id] = split_name
            offset += count
    return assignments


def split_hash(assignments: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(assignments.items())), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_split_manifest(assignments: Mapping[str, str], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_hash": split_hash(assignments),
        "assignments": dict(sorted(assignments.items())),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
