"""Strict prepared-volume provenance and physical-grid validation."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np


def _required(metadata: Mapping[str, Any], key: str, *, label: str) -> Any:
    value = metadata.get(key)
    if value is None:
        raise ValueError(f"{label} metadata is missing {key!r}")
    return value


def required_text(metadata: Mapping[str, Any], key: str, *, label: str) -> str:
    value = str(_required(metadata, key, label=label)).strip()
    if not value:
        raise ValueError(f"{label} metadata has an empty {key!r}")
    return value


def phase_roles(metadata: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    raw = _required(metadata, "phase_roles", label=label)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{label} phase_roles must be a non-empty sequence")
    roles = tuple(str(value).strip() for value in raw)
    if not roles or any(not value for value in roles) or len(set(roles)) != len(roles):
        raise ValueError(f"{label} phase_roles must contain unique non-empty names")
    return roles


def grid_arrays(
    metadata: Mapping[str, Any], *, label: str
) -> tuple[np.ndarray, np.ndarray]:
    affine = np.asarray(_required(metadata, "affine_lps", label=label), dtype=np.float64)
    spacing = np.asarray(_required(metadata, "spacing_dhw", label=label), dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{label} affine_lps must be a finite [4,4] matrix")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"{label} spacing_dhw must contain three finite positive values")
    return affine, spacing


def grids_match(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    first_affine, first_spacing = grid_arrays(first, label="first image")
    second_affine, second_spacing = grid_arrays(second, label="second image")
    return bool(
        np.allclose(first_affine, second_affine, rtol=1e-5, atol=1e-4)
        and np.allclose(first_spacing, second_spacing, rtol=1e-6, atol=1e-6)
    )


def parse_provenance(metadata: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    raw = _required(metadata, "provenance_json", label=label)
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} provenance_json is invalid JSON") from exc
    elif isinstance(raw, Mapping):
        value = dict(raw)
    else:
        raise ValueError(f"{label} provenance_json must encode one object")
    if not isinstance(value, dict):
        raise ValueError(f"{label} provenance_json must encode one object")
    return value


def preprocessing_signature(
    metadata: Mapping[str, Any],
    *,
    image_shape_cdhw: Sequence[int],
    label: str,
) -> dict[str, Any]:
    """Validate leakage-sensitive provenance and return compatibility fields."""

    shape = tuple(int(value) for value in image_shape_cdhw)
    if len(shape) != 4 or any(value < 1 for value in shape):
        raise ValueError(f"{label} image shape must be finite [C,D,H,W]")
    affine, spacing = grid_arrays(metadata, label=label)
    roles = phase_roles(metadata, label=label)
    if len(roles) != shape[0]:
        raise ValueError(
            f"{label} phase role count {len(roles)} does not match image channels {shape[0]}"
        )
    provenance = parse_provenance(metadata, label=label)

    # Synthetic archives deliberately use a smaller, explicit provenance contract.
    if provenance.get("synthetic") is True:
        if provenance.get("not_patient_data") is not True:
            raise ValueError(f"{label} synthetic provenance must declare not_patient_data=true")
        return {
            "kind": "synthetic",
            "provenance": provenance,
            "phase_roles": list(roles),
            "output_shape_cdhw": list(shape),
            "spacing_dhw": spacing.tolist(),
        }

    required = (
        "version",
        "source_only",
        "target_content_used",
        "crop_basis",
        "normalization_scope",
        "intensity_stats_hash",
        "output_shape_cdhw",
        "output_spacing_dhw",
        "output_affine_lps",
    )
    missing = [key for key in required if key not in provenance]
    if missing:
        raise ValueError(f"{label} preprocessing provenance is missing {missing}")
    if provenance["source_only"] is not True or provenance["target_content_used"] is not False:
        raise ValueError(f"{label} preprocessing provenance is not source-only")
    if provenance["normalization_scope"] != "global_train_patients_shared_phases":
        raise ValueError(f"{label} normalization was not fitted globally on training patients")
    provenance_shape = tuple(int(value) for value in provenance["output_shape_cdhw"])
    if provenance_shape != shape:
        raise ValueError(
            f"{label} provenance output shape {provenance_shape} differs from archive {shape}"
        )
    provenance_spacing = np.asarray(provenance["output_spacing_dhw"], dtype=np.float64)
    provenance_affine = np.asarray(provenance["output_affine_lps"], dtype=np.float64)
    if provenance_spacing.shape != (3,) or not np.allclose(
        provenance_spacing, spacing, rtol=1e-6, atol=1e-6
    ):
        raise ValueError(f"{label} spacing disagrees with preprocessing provenance")
    if provenance_affine.shape != (4, 4) or not np.allclose(
        provenance_affine, affine, rtol=1e-5, atol=1e-4
    ):
        raise ValueError(f"{label} affine disagrees with preprocessing provenance")
    intensity_hash = str(provenance["intensity_stats_hash"]).strip()
    if not intensity_hash:
        raise ValueError(f"{label} preprocessing provenance has no intensity_stats_hash")
    return {
        "kind": "prepared_patient_volume",
        "version": str(provenance["version"]),
        "crop_basis": str(provenance["crop_basis"]),
        "normalization_scope": str(provenance["normalization_scope"]),
        "intensity_stats_hash": intensity_hash,
        "phase_roles": list(roles),
        "output_shape_cdhw": list(shape),
        "spacing_dhw": spacing.tolist(),
    }


def metadata_agrees(
    recorded: Mapping[str, Any], actual: Mapping[str, Any], *, label: str
) -> None:
    """Verify that a sampling sidecar describes the supplied source archive."""

    for key in ("patient_id", "visit_id", "visit_stage", "split"):
        if required_text(recorded, key, label=f"recorded {label}") != required_text(
            actual, key, label=label
        ):
            raise ValueError(f"recorded {label} {key} differs from the supplied archive")
    if phase_roles(recorded, label=f"recorded {label}") != phase_roles(actual, label=label):
        raise ValueError(f"recorded {label} phase roles differ from the supplied archive")
    if not grids_match(recorded, actual):
        raise ValueError(f"recorded {label} physical grid differs from the supplied archive")
    if parse_provenance(recorded, label=f"recorded {label}") != parse_provenance(
        actual, label=label
    ):
        raise ValueError(f"recorded {label} preprocessing provenance differs from the archive")
