"""Import the signed MeWM continuous-latent cache for SymmFlow training.

The upstream VQ-GAN is a different codec from this project's MONAI
AutoencoderKL.  This importer therefore preserves its checkpoint identity and
channel statistics instead of pretending that the two latent spaces match.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ispy2_symmflow.data.manifest import build_pair_manifest, visit_from_dict
from ispy2_symmflow.training.datasets import (
    load_prepared_image,
    read_jsonl,
    validate_prepared_records,
    write_jsonl,
)
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    LATENT_STATISTICS_FINGERPRINT,
    bind_cached_pair_manifest_provenance,
)
from ispy2_symmflow.utils.hashing import sha256_file, stable_hash


MEWM_CACHE_SCHEMA = "mewm_ispy2_biflow_continuous_latents_v2"
MEWM_PAYLOAD_SCHEMA = "mewm_ispy2_biflow_continuous_latent_payload_v2"
MEWM_NORMALIZATION = "continuous_train_unique_visit_channel_zscore_v1"
MEWM_STATISTICS_SCHEMA = "mewm_ispy2_biflow_latent_channel_statistics_v1"
MEWM_STORED_REPRESENTATION = "continuous_prequantization_float16_v1"
MEWM_NUMERIC_CONTRACT = "motfm_registered_global_zscore_v1"
MEWM_BUNDLE_SCHEMA = "motfm_ispy2_registered_strict_a_bundle_v1"
MEWM_ROI_CACHE_SCHEMA = "motfm_ispy2_t0_fixed_roi_cache_v1"

_IDENTITY_FIELDS = frozenset(
    {
        "bundle_contract_sha256",
        "bundle_json_sha256",
        "bundle_schema",
        "codebook_max",
        "codebook_min",
        "codebook_sha256",
        "data_contract_sha256",
        "encoding",
        "input_shape_zyx",
        "latent_dtype",
        "latent_shape_czyx",
        "latent_statistics",
        "normalization",
        "numeric_contract",
        "payload_schema",
        "phase_manifest_sha256",
        "roi_cache_contract",
        "schema",
        "source_cache_identity_sha256",
        "split_visit_counts",
        "stored_representation",
        "visit_count",
        "visit_ids",
        "vqgan_checkpoint",
        "vqgan_sha256",
    }
)
_STATISTICS_FIELDS = frozenset(
    {
        "accumulator_dtype",
        "element_count_per_channel",
        "mean",
        "schema",
        "sha256",
        "source_split",
        "std",
        "variance_estimator",
        "visit_count",
        "visit_ids_sha256",
        "visit_selection",
    }
)
_PAYLOAD_FIELDS = frozenset(
    {
        "codebook_sha256",
        "continuous_latent",
        "data_contract_sha256",
        "latent_statistics_sha256",
        "normalization",
        "schema",
        "split",
        "visit_id",
        "vqgan_sha256",
    }
)


@dataclass(frozen=True)
class MewmLatentImportResult:
    visit_manifest_path: str
    pair_manifest_path: str
    statistics_path: str
    report_path: str
    visit_count: int
    pair_count: int
    missing_pair_ids: tuple[str, ...]
    vqgan_sha256: str


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} is missing or unsafe: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _sha256_text(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result < 1 or result != value:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _validate_statistics(
    value: Any, *, latent_channels: int, split_visit_counts: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _STATISTICS_FIELDS:
        raise ValueError("MeWM latent statistics fields are invalid")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != _canonical_sha256(unsigned):
        raise ValueError("MeWM latent statistics SHA-256 does not match its contents")
    mean = value.get("mean")
    std = value.get("std")
    if (
        value.get("schema") != MEWM_STATISTICS_SCHEMA
        or value.get("source_split") != "train"
        or value.get("visit_selection") != "unique_endpoint_visits"
        or value.get("accumulator_dtype") != "float64"
        or value.get("variance_estimator") != "population_ddof0"
        or not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != latent_channels
        or len(std) != latent_channels
    ):
        raise ValueError("MeWM latent statistics contract is invalid")
    try:
        means = np.asarray(mean, dtype=np.float64)
        standard_deviations = np.asarray(std, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("MeWM latent channel statistics are invalid") from exc
    if (
        not np.isfinite(means).all()
        or not np.isfinite(standard_deviations).all()
        or np.any(standard_deviations <= 0)
    ):
        raise ValueError("MeWM latent channel statistics must be finite and non-degenerate")
    visit_count = _positive_int(value.get("visit_count"), label="statistics visit_count")
    if _positive_int(split_visit_counts.get("train"), label="train visit count") != visit_count:
        raise ValueError("MeWM train visit count differs from its latent statistics")
    _positive_int(
        value.get("element_count_per_channel"),
        label="statistics element_count_per_channel",
    )
    _sha256_text(value.get("visit_ids_sha256"), label="statistics visit_ids_sha256")
    return dict(value)


def validate_mewm_cache_identity(
    identity: Mapping[str, Any],
    *,
    bundle_json: str | Path,
    vqgan_checkpoint: str | Path,
) -> dict[str, Any]:
    """Validate the complete upstream cache and codec identity."""

    if set(identity) != _IDENTITY_FIELDS:
        raise ValueError("MeWM latent cache identity fields are invalid")
    latent_shape = tuple(identity.get("latent_shape_czyx", ()))
    input_shape = tuple(identity.get("input_shape_zyx", ()))
    split_counts = identity.get("split_visit_counts")
    visit_ids = identity.get("visit_ids")
    roi_contract = identity.get("roi_cache_contract")
    if (
        identity.get("schema") != MEWM_CACHE_SCHEMA
        or identity.get("payload_schema") != MEWM_PAYLOAD_SCHEMA
        or identity.get("normalization") != MEWM_NORMALIZATION
        or identity.get("stored_representation") != MEWM_STORED_REPRESENTATION
        or identity.get("numeric_contract") != MEWM_NUMERIC_CONTRACT
        or identity.get("bundle_schema") != MEWM_BUNDLE_SCHEMA
        or identity.get("encoding") != "continuous_without_quantizer"
        or identity.get("latent_dtype") != "float16"
        or latent_shape != (8, 24, 64, 64)
        or input_shape != (96, 256, 256)
        or not isinstance(split_counts, dict)
        or set(split_counts) != {"train", "val"}
        or not isinstance(visit_ids, list)
        or not isinstance(roi_contract, dict)
        or roi_contract.get("schema") != MEWM_ROI_CACHE_SCHEMA
    ):
        raise ValueError("MeWM latent cache identity contract is unsupported")
    count = _positive_int(identity.get("visit_count"), label="identity visit_count")
    normalized_ids = [str(value) for value in visit_ids]
    if len(normalized_ids) != count or len(set(normalized_ids)) != count:
        raise ValueError("MeWM latent cache visit inventory is invalid")
    if sum(_positive_int(value, label=f"split {key} visit count") for key, value in split_counts.items()) != count:
        raise ValueError("MeWM latent split visit counts do not sum to its inventory")
    for key in (
        "bundle_contract_sha256",
        "bundle_json_sha256",
        "codebook_sha256",
        "data_contract_sha256",
        "phase_manifest_sha256",
        "source_cache_identity_sha256",
        "vqgan_sha256",
    ):
        _sha256_text(identity.get(key), label=f"identity {key}")
    for key in ("bundle_contract_sha256", "preprocess_sha256", "normalization_sha256"):
        _sha256_text(roi_contract.get(key), label=f"ROI cache {key}")

    bundle_path = Path(bundle_json).expanduser().resolve()
    if sha256_file(bundle_path) != identity["bundle_json_sha256"]:
        raise ValueError("MeWM bundle.json bytes differ from the latent cache identity")
    bundle = _read_json_object(bundle_path, label="MeWM bundle")
    if (
        bundle.get("schema_version") != identity["bundle_schema"]
        or bundle.get("bundle_contract_sha256") != identity["bundle_contract_sha256"]
        or bundle.get("base_cache_contract") != roi_contract
    ):
        raise ValueError("MeWM bundle contract differs from the latent cache identity")
    source_contracts = bundle.get("source_contracts")
    if not isinstance(source_contracts, Mapping) or identity["phase_manifest_sha256"] not in {
        str(value) for value in source_contracts.values()
    }:
        raise ValueError("MeWM phase-manifest identity is not signed by the bundle")

    checkpoint_path = Path(vqgan_checkpoint).expanduser().resolve()
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("MeWM VQ-GAN checkpoint is missing or unsafe")
    if sha256_file(checkpoint_path) != identity["vqgan_sha256"]:
        raise ValueError("MeWM VQ-GAN checkpoint SHA-256 differs from the latent cache")
    statistics = _validate_statistics(
        identity.get("latent_statistics"),
        latent_channels=latent_shape[0],
        split_visit_counts=split_counts,
    )
    validated = dict(identity)
    validated["visit_ids"] = normalized_ids
    validated["latent_statistics"] = statistics
    return validated


def _upstream_payload_path(root: Path, visit_id: str) -> Path:
    if not visit_id or "/" in visit_id or "\\" in visit_id:
        raise ValueError("MeWM latent visit ID is unsafe")
    return root / "visits" / f"{visit_id.replace(':', '__')}.pt"


def _load_upstream_latent(
    path: Path,
    *,
    visit_id: str,
    split: str,
    identity: Mapping[str, Any],
) -> torch.Tensor:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError) as exc:
        raise ValueError(f"MeWM latent payload is unreadable: {path}") from exc
    statistics = identity["latent_statistics"]
    latent = payload.get("continuous_latent") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or set(payload) != _PAYLOAD_FIELDS
        or payload.get("schema") != identity["payload_schema"]
        or payload.get("visit_id") != visit_id
        or payload.get("split") != split
        or payload.get("vqgan_sha256") != identity["vqgan_sha256"]
        or payload.get("codebook_sha256") != identity["codebook_sha256"]
        or payload.get("data_contract_sha256") != identity["data_contract_sha256"]
        or payload.get("normalization") != identity["normalization"]
        or payload.get("latent_statistics_sha256") != statistics["sha256"]
        or not isinstance(latent, torch.Tensor)
        or latent.dtype != torch.float16
        or tuple(latent.shape) != tuple(identity["latent_shape_czyx"])
        or not bool(torch.isfinite(latent.float()).all())
    ):
        raise ValueError(f"MeWM latent payload contract is invalid: {path}")
    return latent.float().contiguous()


def _without_bindings(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in record.items()
        if str(key)
        not in {
            LATENT_STATISTICS_FINGERPRINT,
            CACHED_PAIR_MANIFEST_FINGERPRINT,
            CACHED_PAIR_MANIFEST_RECORD_COUNT,
        }
    }


def _validate_pair_manifest(
    visits: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> None:
    typed_visits = [visit_from_dict(record) for record in visits]
    assignments: dict[str, str] = {}
    for visit in typed_visits:
        if not visit.split:
            raise ValueError(f"prepared visit {visit.visit_id!r} has no split")
        previous = assignments.setdefault(visit.patient_id, visit.split)
        if previous != visit.split:
            raise ValueError(f"patient {visit.patient_id!r} crosses prepared-data splits")
    stage_pairs = {
        (str(pair.get("earlier_stage", "")), str(pair.get("later_stage", "")))
        for pair in pairs
    }
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for earlier_stage, later_stage in stage_pairs:
        for pair in build_pair_manifest(
            typed_visits,
            assignments,
            earlier_stage=earlier_stage,
            later_stage=later_stage,
            require_three_phase=False,
        ):
            expected[(pair.earlier_visit_id, pair.later_visit_id)] = pair.to_dict()
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        endpoints = (str(pair.get("earlier_visit_id", "")), str(pair.get("later_visit_id", "")))
        if not all(endpoints) or endpoints in seen:
            raise ValueError("MeWM prepared pair endpoints are missing or duplicated")
        seen.add(endpoints)
        reference = expected.get(endpoints)
        if reference is None:
            raise ValueError(f"prepared pair endpoints are not reconstructable: {endpoints}")
        for key, value in reference.items():
            if pair.get(key) != value:
                raise ValueError(
                    f"prepared pair {pair.get('pair_id')!r} field {key!r} differs from visits"
                )
        if any(
            isinstance(event, Mapping)
            and str(event.get("severity", "")).strip().lower() == "error"
            for event in pair.get("qc", ())
        ):
            raise ValueError(f"prepared pair {pair.get('pair_id')!r} has error-level QC")


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def import_mewm_latent_pairs(
    latent_cache_dir: str | Path,
    bundle_json: str | Path,
    visit_manifest: str | Path,
    pair_manifest: str | Path,
    vqgan_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    require_all: bool = True,
) -> MewmLatentImportResult:
    """Convert signed MeWM continuous latents into this project's cache contract."""

    cache_root = Path(latent_cache_dir).expanduser().resolve()
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise ValueError("MeWM latent cache directory is missing or unsafe")
    identity_path = cache_root / "cache_identity.json"
    raw_identity = _read_json_object(identity_path, label="MeWM latent cache identity")
    identity = validate_mewm_cache_identity(
        raw_identity,
        bundle_json=bundle_json,
        vqgan_checkpoint=vqgan_checkpoint,
    )
    autoencoder_id = str(identity["vqgan_sha256"])

    visits = read_jsonl(visit_manifest)
    pairs = read_jsonl(pair_manifest)
    if not visits or not pairs:
        raise ValueError("MeWM prepared visit and pair manifests must be non-empty")
    _validate_pair_manifest(visits, pairs)
    visit_by_id = {str(record["visit_id"]): record for record in visits}
    if len(visit_by_id) != len(visits):
        raise ValueError("MeWM prepared visit manifest duplicates visit IDs")

    available_ids = set(identity["visit_ids"])
    selected_pairs: list[dict[str, Any]] = []
    missing_pair_ids: list[str] = []
    for pair in pairs:
        endpoint_ids = (str(pair["earlier_visit_id"]), str(pair["later_visit_id"]))
        paths = tuple(_upstream_payload_path(cache_root, visit_id) for visit_id in endpoint_ids)
        complete = all(visit_id in available_ids and path.is_file() for visit_id, path in zip(endpoint_ids, paths, strict=True))
        if not complete:
            missing_pair_ids.append(str(pair["pair_id"]))
            continue
        selected_pairs.append(dict(pair))
    if require_all and missing_pair_ids:
        raise ValueError(
            "MeWM latent cache is incomplete for prepared pairs: "
            + ", ".join(missing_pair_ids[:10])
        )
    if not selected_pairs:
        raise ValueError("MeWM latent cache contains no complete prepared pair")
    selected_splits = {str(pair["split"]) for pair in selected_pairs}
    if not {"train", "val"}.issubset(selected_splits):
        raise ValueError("MeWM latent import requires at least one train and one val pair")

    selected_visit_ids = sorted(
        {
            str(pair[key])
            for pair in selected_pairs
            for key in ("earlier_visit_id", "later_visit_id")
        }
    )
    selected_visits = [visit_by_id[visit_id] for visit_id in selected_visit_ids]
    source_signature = validate_prepared_records(selected_visits)
    assignments = {
        str(record["patient_id"]): str(record["split"])
        for record in selected_visits
    }

    destination = Path(output_dir).expanduser().resolve()
    latent_dir = destination / "visits"
    latent_paths = {
        visit_id: latent_dir / f"{visit_id.replace(':', '__')}.npz"
        for visit_id in selected_visit_ids
    }
    unsigned_pairs: list[dict[str, Any]] = []
    for pair in selected_pairs:
        cached = _without_bindings(pair)
        cached.update(
            {
                "earlier_latent_path": str(latent_paths[str(pair["earlier_visit_id"])]),
                "later_latent_path": str(latent_paths[str(pair["later_visit_id"])]),
                "autoencoder_id": autoencoder_id,
                "latent_backend": "mewm_vqgan_continuous_channel_zscore_v1",
            }
        )
        unsigned_pairs.append(cached)

    upstream_statistics = identity["latent_statistics"]
    base_statistics = {
        "mean": list(upstream_statistics["mean"]),
        "std": list(upstream_statistics["std"]),
        "element_count_per_channel": int(upstream_statistics["element_count_per_channel"]),
        "fit_split": "train",
        "fit_scope": "upstream_unique_train_endpoint_visits",
        "fit_visit_count": int(upstream_statistics["visit_count"]),
        "fit_visit_ids_sha256": str(upstream_statistics["visit_ids_sha256"]),
        "autoencoder_id": autoencoder_id,
        "latent_backend": "mewm_vqgan_continuous_channel_zscore_v1",
        "normalization": str(identity["normalization"]),
        "source_preprocessing_signature": source_signature,
        "source_manifest": str(Path(visit_manifest).expanduser().resolve()),
        "source_ordered_manifest_fingerprint": stable_hash(visits),
        "source_manifest_record_count": len(visits),
        "source_split_hash": stable_hash(assignments),
        "upstream_bundle_contract_sha256": str(identity["bundle_contract_sha256"]),
        "upstream_cache_identity_sha256": _canonical_sha256(identity),
        "upstream_latent_statistics_sha256": str(upstream_statistics["sha256"]),
        "vqgan_checkpoint_sha256": autoencoder_id,
        "codebook_sha256": str(identity["codebook_sha256"]),
        "numeric_contract": str(identity["numeric_contract"]),
    }
    bound_pairs, bound_statistics = bind_cached_pair_manifest_provenance(
        unsigned_pairs, base_statistics
    )
    statistics_fingerprint = str(bound_statistics[LATENT_STATISTICS_FINGERPRINT])
    pair_fingerprint = str(bound_statistics[CACHED_PAIR_MANIFEST_FINGERPRINT])
    pair_count = int(bound_statistics[CACHED_PAIR_MANIFEST_RECORD_COUNT])

    means = torch.tensor(upstream_statistics["mean"], dtype=torch.float32).reshape(-1, 1, 1, 1)
    stds = torch.tensor(upstream_statistics["std"], dtype=torch.float32).reshape(-1, 1, 1, 1)
    latent_visit_records: list[dict[str, Any]] = []
    for record in selected_visits:
        visit_id = str(record["visit_id"])
        patient_id = str(record["patient_id"])
        split = str(record["split"])
        source_path = _upstream_payload_path(cache_root, visit_id)
        continuous = _load_upstream_latent(
            source_path,
            visit_id=visit_id,
            split=split,
            identity=identity,
        )
        normalized = (continuous - means) / stds
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError(f"MeWM normalized latent is non-finite: {visit_id}")
        _, prepared_metadata = load_prepared_image(str(record["prepared_path"]))
        provenance = {
            "source_payload": str(source_path),
            "source_payload_sha256": sha256_file(source_path),
            "source_cache_identity_sha256": _canonical_sha256(identity),
            "vqgan_checkpoint_sha256": autoencoder_id,
            "codebook_sha256": str(identity["codebook_sha256"]),
            "encoding": "continuous_prequantization",
            "normalization": str(identity["normalization"]),
            "statistics_split": "upstream_train_unique_endpoint_visits",
            LATENT_STATISTICS_FINGERPRINT: statistics_fingerprint,
            CACHED_PAIR_MANIFEST_FINGERPRINT: pair_fingerprint,
            CACHED_PAIR_MANIFEST_RECORD_COUNT: pair_count,
        }
        output_path = latent_paths[visit_id]
        _atomic_savez(
            output_path,
            latent=normalized.numpy().astype(np.float32, copy=False),
            visit_id=visit_id,
            patient_id=patient_id,
            split=split,
            autoencoder_id=autoencoder_id,
            latent_statistics_fingerprint=statistics_fingerprint,
            cached_pair_manifest_fingerprint=pair_fingerprint,
            cached_pair_manifest_record_count=pair_count,
            affine_lps=np.asarray(prepared_metadata["affine_lps"], dtype=np.float64),
            spacing_dhw=np.asarray(prepared_metadata["spacing_dhw"], dtype=np.float32),
            provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        )
        updated = dict(record)
        updated.update(
            {
                "latent_path": str(output_path),
                "latent_provenance": provenance,
                LATENT_STATISTICS_FINGERPRINT: statistics_fingerprint,
                CACHED_PAIR_MANIFEST_FINGERPRINT: pair_fingerprint,
                CACHED_PAIR_MANIFEST_RECORD_COUNT: pair_count,
            }
        )
        latent_visit_records.append(updated)

    destination.mkdir(parents=True, exist_ok=True)
    visit_output = write_jsonl(destination / "visits.jsonl", latent_visit_records)
    pair_output = write_jsonl(destination / "pairs.jsonl", bound_pairs)
    statistics_path = destination / "latent_statistics.json"
    statistics_path.write_text(
        json.dumps(bound_statistics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema": "ispy2_symmflow_mewm_latent_import_v1",
        "visit_count": len(latent_visit_records),
        "pair_count": len(bound_pairs),
        "missing_pair_ids": missing_pair_ids,
        "vqgan_sha256": autoencoder_id,
        "codebook_sha256": str(identity["codebook_sha256"]),
        "upstream_bundle_contract_sha256": str(identity["bundle_contract_sha256"]),
        "upstream_cache_identity_sha256": _canonical_sha256(identity),
        "upstream_latent_statistics_sha256": str(upstream_statistics["sha256"]),
        "source_preprocessing_signature": source_signature,
        "visit_manifest": str(visit_output),
        "pair_manifest": str(pair_output),
        "statistics": str(statistics_path),
    }
    report_path = destination / "import_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return MewmLatentImportResult(
        visit_manifest_path=str(visit_output),
        pair_manifest_path=str(pair_output),
        statistics_path=str(statistics_path),
        report_path=str(report_path),
        visit_count=len(latent_visit_records),
        pair_count=len(bound_pairs),
        missing_pair_ids=tuple(missing_pair_ids),
        vqgan_sha256=autoencoder_id,
    )


__all__ = [
    "MEWM_CACHE_SCHEMA",
    "MEWM_NORMALIZATION",
    "MEWM_NUMERIC_CONTRACT",
    "MEWM_PAYLOAD_SCHEMA",
    "MewmLatentImportResult",
    "import_mewm_latent_pairs",
    "validate_mewm_cache_identity",
]
