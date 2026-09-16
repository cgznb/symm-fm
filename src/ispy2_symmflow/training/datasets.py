"""Strict readers for prepared MRI volumes and cached paired latents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ispy2_symmflow.models.conditioning import is_forbidden_condition_name
from ispy2_symmflow.training.provenance import LATENT_STATISTICS_FINGERPRINT


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {source}:{line_number} is not an object")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True, default=str))
            handle.write("\n")
    return destination


def load_prepared_image(path: str | Path) -> tuple[Tensor, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        if "image" not in archive:
            raise ValueError(f"prepared archive has no 'image' array: {source}")
        image_array = np.asarray(archive["image"], dtype=np.float32)
        if image_array.ndim != 4:
            raise ValueError(f"prepared image must use [C,D,H,W], got {image_array.shape}")
        if not np.isfinite(image_array).all():
            raise ValueError(f"prepared image contains non-finite values: {source}")
        metadata: dict[str, Any] = {}
        for key in archive.files:
            if key == "image":
                continue
            value = archive[key]
            metadata[key] = value.item() if value.ndim == 0 else value.tolist()
    return torch.from_numpy(image_array), metadata


def _required_text(metadata: Mapping[str, Any], key: str, *, label: str) -> str:
    value = metadata.get(key)
    if value is None:
        raise ValueError(f"{label} is missing {key!r}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} has an empty {key!r}")
    return text


def _phase_roles(metadata: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    raw = metadata.get("phase_roles")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{label} phase_roles must be a non-empty sequence")
    roles = tuple(str(value).strip() for value in raw)
    if not roles or any(not role for role in roles) or len(set(roles)) != len(roles):
        raise ValueError(f"{label} phase_roles must contain unique non-empty names")
    return roles


def _grid_arrays(
    metadata: Mapping[str, Any], *, label: str
) -> tuple[np.ndarray, np.ndarray]:
    if "affine_lps" not in metadata or "spacing_dhw" not in metadata:
        raise ValueError(f"{label} lacks physical-grid provenance")
    affine = np.asarray(metadata["affine_lps"], dtype=np.float64)
    spacing = np.asarray(metadata["spacing_dhw"], dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{label} affine_lps must be a finite [4,4] matrix")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"{label} spacing_dhw must contain three finite positive values")
    return affine, spacing


def _parse_preprocessing_provenance(
    metadata: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    raw = metadata.get("provenance_json")
    if raw is None:
        raise ValueError(f"{label} is missing 'provenance_json'")
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


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        keys = {str(key).lower() for key in value}
        for child in value.values():
            keys.update(_nested_keys(child))
        return keys
    if isinstance(value, (list, tuple)):
        keys: set[str] = set()
        for child in value:
            keys.update(_nested_keys(child))
        return keys
    return set()


def _preprocessing_signature(
    metadata: Mapping[str, Any],
    *,
    image_shape_cdhw: Sequence[int],
    label: str,
) -> dict[str, Any]:
    shape = tuple(int(value) for value in image_shape_cdhw)
    if len(shape) != 4 or any(value < 1 for value in shape):
        raise ValueError(f"{label} image shape must be [C,D,H,W]")
    affine, spacing = _grid_arrays(metadata, label=label)
    roles = _phase_roles(metadata, label=label)
    if len(roles) != shape[0]:
        raise ValueError(
            f"{label} phase role count {len(roles)} does not match image channels {shape[0]}"
        )
    provenance = _parse_preprocessing_provenance(metadata, label=label)

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
    extra = provenance.get("extra", {})
    if not isinstance(extra, Mapping):
        raise ValueError(f"{label} preprocessing provenance extra must be an object")
    forbidden = {
        key
        for key in _nested_keys(extra)
        if key.startswith("target_")
        or key in {"target_mask", "target_bbox", "target_statistics"}
    }
    if forbidden:
        raise ValueError(
            f"{label} preprocessing provenance contains target-derived fields: {sorted(forbidden)}"
        )
    try:
        provenance_shape = tuple(int(value) for value in provenance["output_shape_cdhw"])
        provenance_spacing = np.asarray(provenance["output_spacing_dhw"], dtype=np.float64)
        provenance_affine = np.asarray(provenance["output_affine_lps"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} preprocessing output geometry is invalid") from exc
    if provenance_shape != shape:
        raise ValueError(
            f"{label} provenance output shape {provenance_shape} differs from archive {shape}"
        )
    if provenance_spacing.shape != (3,) or not np.allclose(
        provenance_spacing, spacing, rtol=1e-6, atol=1e-6
    ):
        raise ValueError(f"{label} spacing disagrees with preprocessing provenance")
    if provenance_affine.shape != (4, 4) or not np.allclose(
        provenance_affine, affine, rtol=1e-5, atol=1e-4
    ):
        raise ValueError(f"{label} affine disagrees with preprocessing provenance")
    for key in ("version", "crop_basis", "intensity_stats_hash"):
        if not str(provenance[key]).strip():
            raise ValueError(f"{label} preprocessing provenance has an empty {key!r}")
    recorded_roles = extra.get("phase_roles")
    if recorded_roles is not None:
        if not isinstance(recorded_roles, Sequence) or isinstance(recorded_roles, (str, bytes)):
            raise ValueError(f"{label} preprocessing phase_roles are invalid")
        if tuple(str(value).strip() for value in recorded_roles) != roles:
            raise ValueError(f"{label} phase_roles disagree with preprocessing provenance")
    return {
        "kind": "prepared_patient_volume",
        "version": str(provenance["version"]),
        "crop_basis": str(provenance["crop_basis"]),
        "normalization_scope": str(provenance["normalization_scope"]),
        "intensity_stats_hash": str(provenance["intensity_stats_hash"]),
        "phase_roles": list(roles),
        "output_shape_cdhw": list(shape),
        "spacing_dhw": spacing.tolist(),
    }


def validate_prepared_image(
    image: Tensor,
    metadata: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Validate one prepared image's phase, grid, and preprocessing contract."""

    if image.ndim != 4:
        raise ValueError(f"{label} image must use [C,D,H,W], got {tuple(image.shape)}")
    if not torch.isfinite(image).all():
        raise ValueError(f"{label} image contains non-finite values")
    return _preprocessing_signature(
        metadata,
        image_shape_cdhw=image.shape,
        label=label,
    )


def load_validated_prepared_image(
    record: Mapping[str, Any],
) -> tuple[Tensor, dict[str, Any], dict[str, Any]]:
    """Load one prepared archive and bind it to its manifest identity."""

    visit_id = _required_text(record, "visit_id", label="prepared visit manifest record")
    path = record.get("prepared_path")
    if path is None or not str(path).strip():
        raise ValueError(f"visit {visit_id!r} has no prepared_path")
    image, metadata = load_prepared_image(path)
    label = f"prepared archive for visit {visit_id!r}"
    for key in ("patient_id", "visit_id", "split"):
        expected = _required_text(record, key, label=f"manifest visit {visit_id!r}")
        observed = _required_text(metadata, key, label=label)
        if observed != expected:
            raise ValueError(
                f"{label} {key} mismatch: observed={observed!r}, expected={expected!r}"
            )
    if record.get("phase_roles") is not None:
        expected_roles = _phase_roles(record, label=f"manifest visit {visit_id!r}")
        observed_roles = _phase_roles(metadata, label=label)
        if observed_roles != expected_roles:
            raise ValueError(f"{label} phase_roles disagree with the manifest")
    signature = validate_prepared_image(image, metadata, label=label)
    return image, metadata, signature


def validate_prepared_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate every archive before a training or caching operation can begin."""

    common_signature: dict[str, Any] | None = None
    for record in records:
        _, _, signature = load_validated_prepared_image(record)
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise ValueError("prepared visits do not share one preprocessing contract")
    if common_signature is None:
        raise ValueError("cannot validate an empty prepared visit collection")
    return common_signature


class PreparedVisitDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path | Iterable[Mapping[str, Any]],
        *,
        split: str | None = None,
    ) -> None:
        records = read_jsonl(manifest) if isinstance(manifest, (str, Path)) else [dict(r) for r in manifest]
        self.records = [record for record in records if split is None or record.get("split") == split]
        if not self.records:
            raise ValueError(f"prepared visit dataset is empty for split={split!r}")
        for record in self.records:
            if not record.get("prepared_path"):
                raise ValueError(f"visit {record.get('visit_id')!r} has no prepared_path")
        validate_prepared_records(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image, archive_metadata, _ = load_validated_prepared_image(record)
        return {"image": image, "record": record, "archive_metadata": archive_metadata}


def _load_latent(
    path: str | Path,
    *,
    expected_visit_id: str,
    expected_patient_id: str,
    expected_split: str,
    expected_autoencoder_id: str,
    expected_latent_statistics_fingerprint: str | None = None,
) -> tuple[Tensor, dict[str, np.ndarray]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    metadata: dict[str, np.ndarray] = {}
    with np.load(source, allow_pickle=False) as archive:
        if "latent" not in archive:
            raise ValueError(f"latent archive has no 'latent' array: {source}")
        value = np.asarray(archive["latent"], dtype=np.float32)
        expected = {
            "visit_id": expected_visit_id,
            "patient_id": expected_patient_id,
            "split": expected_split,
            "autoencoder_id": expected_autoencoder_id,
        }
        if expected_latent_statistics_fingerprint is not None:
            expected[LATENT_STATISTICS_FINGERPRINT] = (
                expected_latent_statistics_fingerprint
            )
        for key, wanted in expected.items():
            if key not in archive:
                raise ValueError(f"latent archive has no {key!r} provenance: {source}")
            observed = str(np.asarray(archive[key]).item())
            if observed != str(wanted):
                raise ValueError(
                    f"latent {key} mismatch for {source}: observed={observed!r}, expected={wanted!r}"
                )
        for key, shape in (("affine_lps", (4, 4)), ("spacing_dhw", (3,))):
            if key not in archive:
                raise ValueError(f"latent archive has no {key!r} grid provenance: {source}")
            grid_value = np.asarray(archive[key], dtype=np.float64)
            if grid_value.shape != shape or not np.isfinite(grid_value).all():
                raise ValueError(f"latent {key} provenance is invalid in {source}")
            metadata[key] = grid_value
    if value.ndim != 4 or not np.isfinite(value).all():
        raise ValueError(f"cached latent must be finite [C,D,H,W], got {value.shape}")
    return torch.from_numpy(value), metadata


def pair_conditions(record: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only source-available condition fields, never evaluation outcomes."""

    clinical = dict(record.get("baseline_clinical") or {})
    treatment = dict(record.get("treatment") or {})
    for mapping in (clinical, treatment):
        overlap = {str(key) for key in mapping if is_forbidden_condition_name(str(key))}
        if overlap:
            raise ValueError(f"pair manifest leaks forbidden condition fields: {sorted(overlap)}")
    condition = {**clinical, **treatment}
    condition.update(
        {
            "stage_i": record["earlier_stage"],
            "stage_j": record["later_stage"],
            "delta_days": record.get("delta_days"),
            "interval_missing": (
                "yes" if bool(record.get("interval_missing", True)) else "no"
            ),
            "interval_source": record.get("interval_source"),
        }
    )
    return condition


class LatentPairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        pair_manifest: str | Path | Iterable[Mapping[str, Any]],
        *,
        split: str | None = None,
    ) -> None:
        records = read_jsonl(pair_manifest) if isinstance(pair_manifest, (str, Path)) else [dict(r) for r in pair_manifest]
        self.records = [record for record in records if split is None or record.get("split") == split]
        if not self.records:
            raise ValueError(f"latent pair dataset is empty for split={split!r}")
        for record in self.records:
            if any(
                isinstance(event, Mapping) and event.get("severity") == "error"
                for event in record.get("qc", [])
            ):
                raise ValueError(
                    f"pair {record.get('pair_id')!r} has unresolved error-level QC"
                )
            for key in ("earlier_latent_path", "later_latent_path"):
                if not record.get(key):
                    raise ValueError(f"pair {record.get('pair_id')!r} has no {key}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        provenance = {
            "expected_patient_id": str(record["patient_id"]),
            "expected_split": str(record["split"]),
            "expected_autoencoder_id": str(record["autoencoder_id"]),
            "expected_latent_statistics_fingerprint": (
                str(record[LATENT_STATISTICS_FINGERPRINT])
                if record.get(LATENT_STATISTICS_FINGERPRINT) is not None
                else None
            ),
        }
        earlier, earlier_grid = _load_latent(
                record["earlier_latent_path"],
                expected_visit_id=str(record["earlier_visit_id"]),
                **provenance,
            )
        later, later_grid = _load_latent(
                record["later_latent_path"],
                expected_visit_id=str(record["later_visit_id"]),
                **provenance,
            )
        if earlier.shape != later.shape:
            raise ValueError(f"paired latent shapes differ for pair {record.get('pair_id')!r}")
        affine_match = np.allclose(
            earlier_grid["affine_lps"],
            later_grid["affine_lps"],
            rtol=1e-5,
            atol=1e-4,
        )
        spacing_match = np.allclose(
            earlier_grid["spacing_dhw"],
            later_grid["spacing_dhw"],
            rtol=1e-6,
            atol=1e-6,
        )
        if not (affine_match and spacing_match):
            raise ValueError(
                f"paired latent physical grids differ for pair {record.get('pair_id')!r}"
            )
        return {
            "earlier_latent": earlier,
            "later_latent": later,
            "conditions": pair_conditions(record),
            "record": record,
        }


def collate_latent_pairs(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty pair batch")
    condition_names = set().union(*(item["conditions"] for item in items))
    return {
        "earlier_latent": torch.stack([item["earlier_latent"] for item in items]),
        "later_latent": torch.stack([item["later_latent"] for item in items]),
        "conditions": {
            key: [item["conditions"].get(key) for item in items] for key in sorted(condition_names)
        },
        "record": [item["record"] for item in items],
    }
