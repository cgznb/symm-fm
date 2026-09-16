"""Formal held-out cohort evaluation for stochastic longitudinal predictions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from ispy2_symmflow.inference.sampler import (
    validate_sample_array_binding,
    validate_source_archive_binding,
)
from ispy2_symmflow.models import ConditionSchema
from ispy2_symmflow.training.datasets import (
    load_prepared_image,
    pair_conditions,
    read_jsonl,
)
from ispy2_symmflow.training.schema import validate_sampling_condition_availability
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    LATENT_STATISTICS_FINGERPRINT,
    cached_pair_manifest_fingerprint,
    require_latent_statistics_fingerprint,
)
from ispy2_symmflow.utils.hashing import sha256_file, stable_hash

from ._validation import (
    grids_match,
    metadata_agrees,
    phase_roles,
    preprocessing_signature,
    required_text,
)
from .aggregate import aggregate_by_patient, patient_bootstrap_mean_ci
from .metrics import evaluate_prediction, evaluate_sample_set


HELD_OUT_SPLITS = frozenset(("val", "test"))
BASELINE_KINDS = frozenset(("deterministic", "unidirectional_cfm"))
IMAGE_METRICS = ("mae", "mse", "psnr", "ssim", "foreground_mae", "foreground_mse")


@dataclass(frozen=True)
class _CaseResult:
    public: dict[str, Any]
    candidate: dict[str, list[float | None]]
    predictive_mean: dict[str, float | None]
    oracle: dict[str, float | None]
    copy_source: dict[str, float | None] | None
    sampling_signature: dict[str, Any]
    preprocessing_signature: dict[str, Any]


@dataclass(frozen=True)
class _TrustedPairs:
    source: str
    by_visits: Mapping[tuple[str, str, str], Mapping[str, Any]]
    split_hash: str
    ordered_manifest_fingerprint: str
    manifest_record_count: int
    autoencoder_id: str
    cached_pair_manifest_fingerprint: str
    latent_statistics_fingerprint: str


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _finite(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("a per-case metric unexpectedly contains more than one value")
        value = value.detach().cpu().item()
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_scalars(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in IMAGE_METRICS:
        if name in metrics:
            result[name] = _finite(metrics[name])
    return result


def _resolve_path(value: Any, *, base: Path, field: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"cohort record is missing {field!r}")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_cohort_records(
    manifest: str | Path | Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Path]:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest).expanduser().resolve()
        records = read_jsonl(path)
        base = path.parent
    else:
        records = [dict(record) for record in manifest]
        base = Path.cwd()
    if not records:
        raise ValueError("cohort manifest is empty")
    return records, base


def _has_error_qc(record: Mapping[str, Any]) -> bool:
    return any(
        isinstance(event, Mapping)
        and str(event.get("severity", "")).strip().lower() == "error"
        for event in (record.get("qc") or ())
    )


def _record_text(record: Mapping[str, Any], key: str, *, label: str) -> str:
    value = str(record.get(key, "")).strip()
    if not value:
        raise ValueError(f"{label} is missing {key!r}")
    return value


def _read_trusted_pairs(
    manifest: str | Path | Iterable[Mapping[str, Any]],
) -> _TrustedPairs:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest).expanduser().resolve()
        records = read_jsonl(path)
        source = str(path)
    else:
        records = [dict(record) for record in manifest]
        source = "<in-memory>"
    if not records:
        raise ValueError("trusted cached pair manifest is empty")

    ordered_pair_fingerprint = cached_pair_manifest_fingerprint(records)
    recorded_pair_fingerprints = {
        _record_text(record, CACHED_PAIR_MANIFEST_FINGERPRINT, label="cached pair")
        for record in records
    }
    if recorded_pair_fingerprints != {ordered_pair_fingerprint}:
        raise ValueError(
            "cached pair ordered manifest fingerprint differs from its record bindings"
        )
    recorded_counts: set[int] = set()
    for record in records:
        try:
            recorded_counts.add(int(record[CACHED_PAIR_MANIFEST_RECORD_COUNT]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"cached pair has no valid {CACHED_PAIR_MANIFEST_RECORD_COUNT}"
            ) from exc
    if recorded_counts != {len(records)}:
        raise ValueError("cached pair manifest record-count bindings are inconsistent")
    statistics_fingerprints = {
        _record_text(record, LATENT_STATISTICS_FINGERPRINT, label="cached pair")
        for record in records
    }
    if len(statistics_fingerprints) != 1:
        raise ValueError("trusted cached pairs do not share one latent-statistics fingerprint")

    assignments: dict[str, str] = {}
    by_visits: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    autoencoder_ids: set[str] = set()
    eligible_count = 0
    for index, record in enumerate(records):
        label = f"cached pair record {index}"
        patient_id = _record_text(record, "patient_id", label=label)
        split = _record_text(record, "split", label=label)
        earlier_visit = _record_text(record, "earlier_visit_id", label=label)
        later_visit = _record_text(record, "later_visit_id", label=label)
        _record_text(record, "earlier_stage", label=label)
        _record_text(record, "later_stage", label=label)
        autoencoder_id = _record_text(record, "autoencoder_id", label=label)
        for key in ("earlier_latent_path", "later_latent_path"):
            _record_text(record, key, label=label)
        if earlier_visit == later_visit:
            raise ValueError(f"{label} uses the same visit at both endpoints")
        if _has_error_qc(record):
            continue
        eligible_count += 1
        previous = assignments.setdefault(patient_id, split)
        if previous != split:
            raise ValueError(
                f"patient {patient_id!r} occurs in both {previous!r} and {split!r}"
            )
        key = (patient_id, earlier_visit, later_visit)
        if key in by_visits:
            raise ValueError(f"trusted cached pair manifest duplicates visits {key!r}")
        by_visits[key] = record
        autoencoder_ids.add(autoencoder_id)

    if eligible_count < 1:
        raise ValueError("trusted cached pair manifest has no QC-passing pairs")
    if len(autoencoder_ids) != 1:
        raise ValueError("trusted cached pairs must share one autoencoder_id")
    return _TrustedPairs(
        source=source,
        by_visits=by_visits,
        split_hash=stable_hash(assignments),
        ordered_manifest_fingerprint=stable_hash(records),
        manifest_record_count=len(records),
        autoencoder_id=next(iter(autoencoder_ids)),
        cached_pair_manifest_fingerprint=ordered_pair_fingerprint,
        latent_statistics_fingerprint=next(iter(statistics_fingerprints)),
    )


def _load_samples(path: Path) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as archive:
        if "samples" not in archive:
            raise ValueError(f"prediction archive has no 'samples' array: {path}")
        array = np.asarray(archive["samples"], dtype=np.float32)
    if array.ndim == 5:
        array = array[:, None]
    if array.ndim != 6 or array.shape[0] < 1 or array.shape[1] != 1:
        raise ValueError(
            f"prediction samples must use [K,1,C,D,H,W] (or [K,C,D,H,W]), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"prediction samples contain non-finite values: {path}")
    return torch.from_numpy(array)


def _load_sampling_record(prediction: Path) -> dict[str, Any]:
    sidecar = prediction.with_suffix(".json")
    if not sidecar.is_file():
        raise ValueError(f"sampling sidecar is required for formal cohort evaluation: {sidecar}")
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid sampling sidecar JSON: {sidecar}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"sampling sidecar must contain one object: {sidecar}")
    validate_sample_array_binding(prediction, value)
    return value


def _time_pair(value: Any, *, label: str) -> tuple[str, str]:
    if isinstance(value, str):
        parts: list[str] | None = None
        for separator in ("->", ",", "/", ":", "-"):
            candidate = [part.strip() for part in value.split(separator)]
            if len(candidate) == 2 and all(candidate):
                parts = candidate
                break
        if parts is None:
            raise ValueError(f"{label} time_pair must identify exactly two stages")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
        parts = [str(part).strip() for part in value]
    else:
        raise ValueError(f"{label} time_pair must be a two-stage sequence or string")
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError(f"{label} time_pair must identify exactly two stages")
    return parts[0], parts[1]


def _optional_hint(record: Mapping[str, Any], *keys: str) -> str | None:
    observed = [str(record[key]).strip() for key in keys if record.get(key) is not None]
    if len(set(observed)) > 1:
        raise ValueError(f"cohort record fields {keys} disagree")
    return observed[0] if observed else None


def _single_condition_value(value: Any, *, field: str, label: str) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError(
                f"{label} condition {field!r} must contain one case value"
            )
        return value[0]
    return value


def _condition_contract(
    values: Mapping[str, Any],
    schema_payload: Mapping[str, Any],
    *,
    label: str,
    reject_unknown: bool = True,
) -> dict[str, Any]:
    """Canonicalize one case exactly as typed model-visible condition fields."""

    schema = ConditionSchema.from_dict(schema_payload)
    unknown = sorted(set(values).difference(schema.field_names))
    if reject_unknown and unknown:
        raise ValueError(f"{label} contains fields outside the checkpoint schema: {unknown}")
    result: dict[str, Any] = {}
    for field in schema.categorical_fields:
        raw = _single_condition_value(values.get(field.name), field=field.name, label=label)
        missing = raw is None or raw == ""
        if isinstance(raw, float) and math.isnan(raw):
            missing = True
        result[field.name] = (
            {"missing": True, "value": None}
            if missing
            else {"missing": False, "value": str(raw)}
        )
    for field in schema.numeric_fields:
        raw = _single_condition_value(values.get(field.name), field=field.name, label=label)
        if raw is None or raw == "":
            result[field.name] = {"missing": True, "value": None}
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} numeric condition {field.name!r} is invalid"
            ) from exc
        if math.isnan(number):
            result[field.name] = {"missing": True, "value": None}
        elif not math.isfinite(number):
            raise ValueError(
                f"{label} numeric condition {field.name!r} must be finite or missing"
            )
        else:
            result[field.name] = {"missing": False, "value": number}
    return result


def _validate_conditions_against_pair(
    supplied: Mapping[str, Any],
    pair: Mapping[str, Any],
    schema_payload: Mapping[str, Any],
    schema_provenance: Mapping[str, Any],
) -> None:
    validate_sampling_condition_availability(supplied, schema_provenance)
    unavailable = schema_provenance.get("unavailable_fields")
    if not isinstance(unavailable, (list, tuple)):
        raise ValueError(
            "checkpoint condition schema provenance has no valid unavailable_fields"
        )
    trusted_conditions = pair_conditions(pair)
    for name in unavailable:
        trusted_conditions[str(name)] = None
    expected = _condition_contract(
        trusted_conditions,
        schema_payload,
        label="trusted pair",
        reject_unknown=False,
    )
    observed = _condition_contract(
        supplied, schema_payload, label="sampling sidecar"
    )
    if observed != expected:
        differing = sorted(
            name for name in expected if observed.get(name) != expected[name]
        )
        raise ValueError(
            "sampling conditions differ from the trusted pair manifest for fields: "
            + ", ".join(differing)
        )


def _checkpoint_provenance(
    metadata: Mapping[str, Any],
    *,
    model_family: str,
    prediction_path: Path,
    stages: tuple[str, str],
    source_preprocessing_signature: Mapping[str, Any],
    trusted_pairs: _TrustedPairs,
    cache: dict[Path, tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    path_key = "flow_checkpoint" if model_family == "symmflow" else "baseline_checkpoint"
    hash_key = f"{path_key}_sha256"
    checkpoint_path = _resolve_path(
        metadata.get(path_key), base=prediction_path.parent, field=path_key
    )
    cached = cache.get(checkpoint_path)
    if cached is None:
        digest = sha256_file(checkpoint_path)
        header = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(header, Mapping):
            raise ValueError(f"model checkpoint must contain one mapping: {checkpoint_path}")
        cache[checkpoint_path] = (digest, header)
    else:
        digest, header = cached
    if str(metadata.get(hash_key, "")) != digest:
        raise ValueError(f"sampling sidecar {hash_key} differs from the checkpoint file")

    checkpoint_split_hash = str(header.get("split_hash", "")).strip()
    if not checkpoint_split_hash:
        raise ValueError("model checkpoint has no training split_hash")
    if checkpoint_split_hash != trusted_pairs.split_hash:
        raise ValueError(
            "model checkpoint split_hash differs from the trusted cached pair manifest"
        )
    if str(metadata.get("training_split_hash", "")) != checkpoint_split_hash:
        raise ValueError("sampling sidecar training_split_hash differs from the model checkpoint")

    extra = header.get("extra")
    checkpoint_signature = extra.get("training_signature") if isinstance(extra, Mapping) else None
    if not isinstance(checkpoint_signature, Mapping):
        raise ValueError("model checkpoint has no training_signature")
    checkpoint_schema_provenance = (
        extra.get("schema_provenance") if isinstance(extra, Mapping) else None
    )
    sidecar_schema_provenance = metadata.get("condition_schema_provenance")
    if not isinstance(checkpoint_schema_provenance, Mapping):
        raise ValueError("model checkpoint has no condition schema provenance")
    if not isinstance(sidecar_schema_provenance, Mapping) or _canonical(
        sidecar_schema_provenance
    ) != _canonical(checkpoint_schema_provenance):
        raise ValueError(
            "sampling sidecar condition schema provenance differs from the model checkpoint"
        )
    sidecar_signature = metadata.get("training_signature")
    if not isinstance(sidecar_signature, Mapping) or _canonical(
        sidecar_signature
    ) != _canonical(checkpoint_signature):
        raise ValueError("sampling sidecar training_signature differs from the model checkpoint")

    ordered_key = (
        "ordered_manifest_fingerprint"
        if model_family == "symmflow"
        else "ordered_manifest_hash"
    )
    ordered_fingerprint = str(checkpoint_signature.get(ordered_key, "")).strip()
    if ordered_fingerprint != trusted_pairs.ordered_manifest_fingerprint:
        raise ValueError(
            "model checkpoint ordered manifest fingerprint differs from the trusted cached pair manifest"
        )
    if int(checkpoint_signature.get("manifest_record_count", -1)) != (
        trusted_pairs.manifest_record_count
    ):
        raise ValueError(
            "model checkpoint manifest record count differs from the trusted cached pair manifest"
        )
    if model_family == "symmflow" and checkpoint_signature.get("stage") != "symmflow":
        raise ValueError("flow checkpoint training_signature has the wrong training stage")
    if model_family == "baseline" and str(checkpoint_signature.get("kind", "")) != str(
        metadata.get("baseline_kind", "")
    ):
        raise ValueError("baseline checkpoint kind differs from the sampling sidecar")

    checkpoint_config = header.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("model checkpoint has no training configuration")
    config_fingerprint = stable_hash(checkpoint_config)
    if str(checkpoint_signature.get("config_fingerprint", "")) != config_fingerprint:
        raise ValueError("model checkpoint training_signature has an invalid config fingerprint")
    if str(metadata.get("training_config_fingerprint", "")) != config_fingerprint:
        raise ValueError("sampling sidecar training config differs from the model checkpoint")
    checkpoint_data = checkpoint_config.get("data")
    if not isinstance(checkpoint_data, Mapping):
        raise ValueError("model checkpoint has no data configuration")
    checkpoint_pair = _time_pair(checkpoint_data.get("time_pair"), label="model checkpoint")
    if checkpoint_pair != stages:
        raise ValueError("sampling stages differ from the model checkpoint time_pair")
    checkpoint_flow = checkpoint_config.get("flow", {})
    if not isinstance(checkpoint_flow, Mapping):
        raise ValueError("model checkpoint flow configuration must be an object")
    checkpoint_sigma = float(checkpoint_flow.get("sigma_min", 0.0))
    expected_sampling_sigma = (
        0.0
        if model_family == "baseline" and metadata.get("baseline_kind") == "deterministic"
        else checkpoint_sigma
    )
    if float(metadata.get("sigma_min", float("nan"))) != expected_sampling_sigma:
        raise ValueError("sampling sigma_min differs from the model checkpoint")
    if _canonical(header.get("feature_schema")) != _canonical(metadata.get("condition_schema")):
        raise ValueError("sampling condition schema differs from the model checkpoint")
    if str(header.get("autoencoder_id", "")) != trusted_pairs.autoencoder_id:
        raise ValueError(
            "model checkpoint autoencoder_id differs from the trusted cached pair manifest"
        )
    if str(metadata.get("autoencoder_checkpoint_sha256", "")) != trusted_pairs.autoencoder_id:
        raise ValueError(
            "sampling autoencoder checkpoint differs from the trusted cached pair manifest"
        )
    recorded_source_signature = metadata.get("source_preprocessing_signature")
    if not isinstance(recorded_source_signature, Mapping) or _canonical(
        recorded_source_signature
    ) != _canonical(source_preprocessing_signature):
        raise ValueError(
            "sampling source_preprocessing_signature differs from the source archive"
        )
    latent_statistics = header.get("latent_statistics")
    if not isinstance(latent_statistics, Mapping):
        raise ValueError("model checkpoint has no latent statistics")
    checkpoint_statistics_fingerprint = require_latent_statistics_fingerprint(
        latent_statistics
    )
    if checkpoint_statistics_fingerprint != trusted_pairs.latent_statistics_fingerprint:
        raise ValueError(
            "model checkpoint latent-statistics fingerprint differs from the trusted pairs"
        )
    if str(latent_statistics.get(CACHED_PAIR_MANIFEST_FINGERPRINT, "")) != (
        trusted_pairs.cached_pair_manifest_fingerprint
    ):
        raise ValueError(
            "model checkpoint cached-pair fingerprint differs from the trusted pair manifest"
        )
    try:
        checkpoint_pair_count = int(
            latent_statistics[CACHED_PAIR_MANIFEST_RECORD_COUNT]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "model checkpoint latent statistics have no cached-pair record count"
        ) from exc
    if checkpoint_pair_count != trusted_pairs.manifest_record_count:
        raise ValueError(
            "model checkpoint cached-pair record count differs from the trusted manifest"
        )
    if str(checkpoint_signature.get(CACHED_PAIR_MANIFEST_FINGERPRINT, "")) != (
        trusted_pairs.cached_pair_manifest_fingerprint
    ):
        raise ValueError(
            "model checkpoint training signature has a different cached-pair fingerprint"
        )
    if int(checkpoint_signature.get(CACHED_PAIR_MANIFEST_RECORD_COUNT, -1)) != (
        trusted_pairs.manifest_record_count
    ):
        raise ValueError(
            "model checkpoint training signature has a different cached-pair record count"
        )
    checkpoint_source_signature = (
        latent_statistics.get("source_preprocessing_signature")
    )
    if not isinstance(checkpoint_source_signature, Mapping) or _canonical(
        checkpoint_source_signature
    ) != _canonical(source_preprocessing_signature):
        raise ValueError(
            "source preprocessing signature differs from the model checkpoint training data"
        )
    return {
        "model_checkpoint": str(checkpoint_path),
        "training_split_hash": checkpoint_split_hash,
        "training_ordered_manifest_fingerprint": ordered_fingerprint,
        "training_manifest_record_count": trusted_pairs.manifest_record_count,
        "condition_schema_provenance": dict(checkpoint_schema_provenance),
    }


def _endpoint_disclosure(
    metadata: Mapping[str, Any], *, direction: str
) -> dict[str, Any]:
    keys = (
        "endpoint_semantics",
        "residual_endpoint_consent",
        "source_endpoint_approximation",
        "decoded_endpoint_contains_sigma_residual",
    )
    missing = [key for key in keys if key not in metadata]
    if missing:
        raise ValueError(f"sampling sidecar lacks endpoint disclosure fields: {missing}")
    sigma_min = float(metadata.get("sigma_min", float("nan")))
    if not math.isfinite(sigma_min) or sigma_min < 0:
        raise ValueError("sampling sidecar sigma_min must be finite and non-negative")
    observed = {key: metadata[key] for key in keys}
    if sigma_min == 0:
        expected = {
            "endpoint_semantics": "paper_clean_endpoints",
            "residual_endpoint_consent": False,
            "source_endpoint_approximation": False,
            "decoded_endpoint_contains_sigma_residual": False,
        }
    else:
        expected = {
            "endpoint_semantics": "upstream_sigma_compatibility_experiment",
            "residual_endpoint_consent": True,
            "source_endpoint_approximation": direction == "backward",
            "decoded_endpoint_contains_sigma_residual": True,
        }
    if observed != expected:
        raise ValueError(
            "sampling endpoint disclosure is inconsistent with sigma_min and direction"
        )
    result = dict(observed)
    note = metadata.get("endpoint_note")
    if sigma_min > 0:
        if not str(note or "").strip():
            raise ValueError("residual-endpoint sampling sidecar has no endpoint_note")
        result["endpoint_note"] = str(note)
    return result


def _trusted_pair_for_case(
    trusted_pairs: _TrustedPairs,
    *,
    source_metadata: Mapping[str, Any],
    target_metadata: Mapping[str, Any],
    direction: str,
    stages: tuple[str, str],
    split: str,
) -> Mapping[str, Any]:
    patient_id = required_text(target_metadata, "patient_id", label="target")
    source_visit = required_text(source_metadata, "visit_id", label="source")
    target_visit = required_text(target_metadata, "visit_id", label="target")
    earlier_visit, later_visit = (
        (source_visit, target_visit)
        if direction == "forward"
        else (target_visit, source_visit)
    )
    pair = trusted_pairs.by_visits.get((patient_id, earlier_visit, later_visit))
    if pair is None:
        raise ValueError(
            "source/target visits are not a QC-passing pair in the trusted cached pair manifest"
        )
    if str(pair.get("split", "")) != split:
        raise ValueError("case split differs from the trusted cached pair manifest")
    pair_stages = (
        str(pair.get("earlier_stage", "")).strip(),
        str(pair.get("later_stage", "")).strip(),
    )
    if pair_stages != stages:
        raise ValueError("case time pair differs from the trusted cached pair manifest")
    return pair


def _sampling_signature(
    metadata: Mapping[str, Any],
    sampling: Mapping[str, Any],
    *,
    sample_count: int,
    checkpoint_provenance: Mapping[str, Any],
    endpoint_disclosure: Mapping[str, Any],
) -> dict[str, Any]:
    common_required = (
        "autoencoder_checkpoint_sha256",
        "training_config_fingerprint",
        "condition_schema",
        "branch_order",
        "solver",
        "steps",
        "sigma_min",
        "source_path",
        "source_sha256",
        "source_preprocessing_provenance",
        "training_split_hash",
        "training_signature",
    )
    model_family = str(metadata.get("model_family", "symmflow")).strip()
    if model_family not in {"symmflow", "baseline"}:
        raise ValueError(f"unsupported sampling model_family {model_family!r}")
    family_required = (
        ("flow_checkpoint_sha256",)
        if model_family == "symmflow"
        else ("baseline_kind", "baseline_checkpoint_sha256", "direction", "time_pair")
    )
    required = common_required + family_required
    missing = [key for key in required if key not in metadata or metadata[key] is None]
    if missing:
        raise ValueError(f"sampling sidecar lacks formal provenance fields: {missing}")
    seeds = sampling.get("seeds")
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
        raise ValueError("sampling sidecar seeds must be a sequence")
    nfe = int(sampling.get("nfe_per_sample", -1))
    signature = {
        "model_family": model_family,
        "autoencoder_checkpoint_sha256": str(metadata["autoencoder_checkpoint_sha256"]),
        "training_config_fingerprint": str(metadata["training_config_fingerprint"]),
        "condition_schema": metadata["condition_schema"],
        "branch_order": list(metadata["branch_order"]),
        "solver": str(metadata["solver"]),
        "steps": int(metadata["steps"]),
        "sigma_min": float(metadata["sigma_min"]),
        "source_sha256": str(metadata["source_sha256"]),
        "nfe_per_sample": nfe,
        "sample_count": int(sample_count),
        "endpoint_semantics": endpoint_disclosure["endpoint_semantics"],
        "residual_endpoint_consent": endpoint_disclosure[
            "residual_endpoint_consent"
        ],
        "decoded_endpoint_contains_sigma_residual": endpoint_disclosure[
            "decoded_endpoint_contains_sigma_residual"
        ],
        **dict(checkpoint_provenance),
    }
    if model_family == "symmflow":
        if list(metadata["branch_order"]) != ["later", "earlier"]:
            raise ValueError("SymmFlow sampling sidecar has an invalid branch_order")
        if len(seeds) != sample_count:
            raise ValueError(
                "SymmFlow sampling sidecar seed count differs from the candidate count"
            )
        solver = signature["solver"]
        steps = signature["steps"]
        if solver not in {"euler", "heun"} or steps < 1:
            raise ValueError("SymmFlow sampling sidecar must record a supported solver and steps")
        expected_nfe = steps if solver == "euler" else 2 * steps
        if nfe != expected_nfe:
            raise ValueError(
                f"SymmFlow {solver} sampling must record nfe_per_sample={expected_nfe}"
            )
        signature["flow_checkpoint_sha256"] = str(metadata["flow_checkpoint_sha256"])
        return signature

    baseline_kind = str(metadata["baseline_kind"]).strip()
    if baseline_kind not in BASELINE_KINDS:
        raise ValueError(f"unsupported baseline_kind {baseline_kind!r}")
    if sampling.get("direction") != "forward" or metadata["direction"] != "forward":
        raise ValueError("baseline cohort predictions must be forward-only")
    if list(metadata["branch_order"]) != ["later"]:
        raise ValueError("baseline sampling sidecar branch_order must be ['later']")
    if _time_pair(metadata["time_pair"], label="baseline sidecar") != (
        str(metadata.get("conditions", {}).get("stage_i", "")),
        str(metadata.get("conditions", {}).get("stage_j", "")),
    ):
        raise ValueError("baseline sidecar time_pair disagrees with its conditions")
    if baseline_kind == "deterministic":
        if sample_count != 1 or len(seeds) != 0 or nfe != 0:
            raise ValueError(
                "deterministic baseline must contain one candidate, no seeds, and zero NFE"
            )
        if signature["steps"] != 0 or signature["solver"] != "none":
            raise ValueError("deterministic baseline must record steps=0 and solver='none'")
    else:
        if len(seeds) != sample_count:
            raise ValueError(
                "unidirectional CFM seed count differs from the prediction candidate count"
            )
        solver = signature["solver"]
        steps = signature["steps"]
        if solver not in {"euler", "heun"} or steps < 1:
            raise ValueError("unidirectional CFM must record a supported solver and steps")
        expected_nfe = steps if solver == "euler" else 2 * steps
        if nfe != expected_nfe:
            raise ValueError(
                f"unidirectional CFM {solver} sampling must record nfe_per_sample={expected_nfe}"
            )
    signature.update(
        baseline_kind=baseline_kind,
        baseline_checkpoint_sha256=str(metadata["baseline_checkpoint_sha256"]),
    )
    return signature


def _candidate_values(candidate_set: Mapping[str, Any]) -> dict[str, list[float | None]]:
    result: dict[str, list[float | None]] = {}
    metrics = candidate_set["candidate_metrics"]
    for name in IMAGE_METRICS:
        if name not in metrics:
            continue
        tensor = torch.as_tensor(metrics[name]).detach().cpu().reshape(-1)
        result[name] = [_finite(value) for value in tensor]
    return result


def _describe(values: Iterable[float | None]) -> dict[str, float | int | None | str]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q05": None,
            "q95": None,
            "status": "not_available: no finite candidate values",
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "median": float(np.median(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q95": float(np.quantile(finite, 0.95)),
        "status": "descriptive_only: candidates and repeated pairs are not independent patients",
    }


def _patient_ci(
    values: Iterable[tuple[str, float | None]],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int | None | str]:
    records = [
        {"patient_id": patient, "value": value}
        for patient, value in values
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


def _evaluate_case(
    record: Mapping[str, Any],
    *,
    base: Path,
    trusted_pairs: _TrustedPairs,
    checkpoint_cache: dict[Path, tuple[str, Mapping[str, Any]]],
    data_range: float,
    foreground_threshold: float,
) -> _CaseResult:
    prediction_path = _resolve_path(record.get("prediction"), base=base, field="prediction")
    target_path = _resolve_path(record.get("target"), base=base, field="target")
    sampling = _load_sampling_record(prediction_path)
    samples = _load_samples(prediction_path)
    target, target_metadata = load_prepared_image(target_path)
    if tuple(samples.shape[2:]) != tuple(target.shape):
        raise ValueError(
            f"prediction/target geometry differs for {prediction_path}: "
            f"{tuple(samples.shape[2:])} vs {tuple(target.shape)}"
        )

    metadata = sampling.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"sampling sidecar metadata is not an object: {prediction_path}")
    recorded_source_metadata = metadata.get("source_metadata")
    if not isinstance(recorded_source_metadata, Mapping):
        raise ValueError(f"sampling sidecar has no source_metadata object: {prediction_path}")
    source_path = _resolve_path(
        metadata.get("source_path"),
        base=prediction_path.parent,
        field="sampling sidecar source_path",
    )
    if record.get("source") is not None:
        supplied_source_path = _resolve_path(record["source"], base=base, field="source")
        if supplied_source_path != source_path:
            raise ValueError("cohort source differs from the path recorded during sampling")
    validate_source_archive_binding(source_path, metadata)
    source, source_metadata = load_prepared_image(source_path)
    validate_source_archive_binding(source_path, metadata)
    metadata_agrees(recorded_source_metadata, source_metadata, label="source")
    if source.shape != target.shape:
        raise ValueError("source and target array geometry differs")
    source_preprocessing_signature = preprocessing_signature(
        source_metadata,
        image_shape_cdhw=source.shape,
        label="sampling source",
    )

    direction = str(sampling.get("direction", "")).strip()
    if direction not in {"forward", "backward"}:
        raise ValueError(f"invalid sampling direction for {prediction_path}: {direction!r}")
    direction_hint = _optional_hint(record, "direction")
    if direction_hint is not None and direction_hint != direction:
        raise ValueError("cohort direction disagrees with the sampling sidecar")
    endpoint_disclosure = _endpoint_disclosure(metadata, direction=direction)

    conditions = metadata.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("sampling sidecar conditions must contain one object")
    stages = (str(conditions.get("stage_i", "")).strip(), str(conditions.get("stage_j", "")).strip())
    if not all(stages):
        raise ValueError("sampling sidecar conditions must include stage_i and stage_j")
    if record.get("time_pair") is not None and _time_pair(
        record["time_pair"], label="cohort"
    ) != stages:
        raise ValueError("cohort time_pair disagrees with sampling conditions")

    model_family = str(metadata.get("model_family", "symmflow")).strip()
    if model_family not in {"symmflow", "baseline"}:
        raise ValueError(f"unsupported sampling model_family {model_family!r}")
    checkpoint_provenance = _checkpoint_provenance(
        metadata,
        model_family=model_family,
        prediction_path=prediction_path,
        stages=stages,
        source_preprocessing_signature=source_preprocessing_signature,
        trusted_pairs=trusted_pairs,
        cache=checkpoint_cache,
    )

    patient_id = required_text(target_metadata, "patient_id", label="target")
    source_patient = required_text(source_metadata, "patient_id", label="sampling source")
    if patient_id != source_patient:
        raise ValueError("sampling source and target belong to different patients")
    patient_hint = _optional_hint(record, "patient", "patient_id")
    if patient_hint is not None and patient_hint != patient_id:
        raise ValueError("cohort patient disagrees with prepared-volume provenance")

    split = required_text(target_metadata, "split", label="target")
    source_split = required_text(source_metadata, "split", label="sampling source")
    split_hint = _optional_hint(record, "split")
    if split not in HELD_OUT_SPLITS:
        raise ValueError(f"formal cohort evaluation only accepts held-out val/test data, got {split!r}")
    if source_split != split or (split_hint is not None and split_hint != split):
        raise ValueError("source, target, and cohort split provenance disagree")
    trusted_pair = _trusted_pair_for_case(
        trusted_pairs,
        source_metadata=source_metadata,
        target_metadata=target_metadata,
        direction=direction,
        stages=stages,
        split=split,
    )
    condition_schema = metadata.get("condition_schema")
    if not isinstance(condition_schema, Mapping):
        raise ValueError("sampling sidecar condition_schema must contain one object")
    _validate_conditions_against_pair(
        conditions,
        trusted_pair,
        condition_schema,
        checkpoint_provenance["condition_schema_provenance"],
    )

    expected_source_stage, expected_target_stage = (
        stages if direction == "forward" else (stages[1], stages[0])
    )
    source_stage = required_text(source_metadata, "visit_stage", label="sampling source")
    target_stage = required_text(target_metadata, "visit_stage", label="target")
    if source_stage != expected_source_stage or target_stage != expected_target_stage:
        raise ValueError(
            f"{direction} source/target stages must be {expected_source_stage}/{expected_target_stage}, "
            f"got {source_stage}/{target_stage}"
        )
    recorded_source_stage = metadata.get("source_visit_stage")
    if recorded_source_stage is not None and str(recorded_source_stage) != source_stage:
        raise ValueError("sampling source_visit_stage disagrees with source_metadata")

    source_roles = phase_roles(source_metadata, label="sampling source")
    target_roles = phase_roles(target_metadata, label="target")
    if source_roles != target_roles:
        raise ValueError("source and target phase roles differ")
    recorded_roles = metadata.get("source_phase_roles")
    if recorded_roles is not None and tuple(str(value) for value in recorded_roles) != source_roles:
        raise ValueError("sampling source_phase_roles disagrees with source_metadata")
    if not grids_match(source_metadata, target_metadata):
        raise ValueError("sampling source and target physical grids differ")

    source_preprocessing = metadata.get("source_preprocessing_provenance")
    if source_preprocessing != source_metadata.get("provenance_json"):
        raise ValueError("sampling source preprocessing provenance fields disagree")
    target_preprocessing_signature = preprocessing_signature(
        target_metadata,
        image_shape_cdhw=target.shape,
        label="target",
    )
    if _canonical(source_preprocessing_signature) != _canonical(target_preprocessing_signature):
        raise ValueError("source and target preprocessing provenance is incompatible")

    foreground = target[None] > float(foreground_threshold)
    candidate_set = evaluate_sample_set(
        samples, target[None], data_range=data_range, foreground_mask=foreground
    )
    candidate = _candidate_values(candidate_set)
    predictive = _metric_scalars(candidate_set["predictive_mean"])
    candidate_foreground_valid = [
        bool(value)
        for value in torch.as_tensor(
            candidate_set["candidate_metrics"]["foreground_valid"]
        ).detach().cpu().reshape(-1)
    ]
    predictive_foreground_valid = bool(
        torch.as_tensor(candidate_set["predictive_mean"]["foreground_valid"]).item()
    )
    oracle = {"mae": _finite(candidate_set["oracle_best_mae"])}
    copy_metrics = _metric_scalars(
        evaluate_prediction(
            source[None],
            target[None],
            data_range=data_range,
            foreground_mask=foreground,
        )
    )

    public = {
        "prediction": str(prediction_path),
        "target": str(target_path),
        "source": str(source_path),
        "source_sha256": str(metadata["source_sha256"]),
        "pair_id": trusted_pair.get("pair_id"),
        "patient_id": patient_id,
        "split": split,
        "direction": direction,
        "time_pair": list(stages),
        "source_visit_stage": source_stage,
        "target_visit_stage": target_stage,
        "candidate_count": int(samples.shape[0]),
        "conditions_verified_against_trusted_pair": True,
        "candidate_distribution": {
            name: {"values": values, "summary": _describe(values)}
            for name, values in candidate.items()
        },
        "predictive_mean": predictive,
        "foreground_valid": {
            "candidates": candidate_foreground_valid,
            "predictive_mean": predictive_foreground_valid,
        },
        "oracle": {
            **oracle,
            "warning": candidate_set["oracle_warning"],
        },
        "copy_source_no_change": copy_metrics,
        "geometry_status": "matched_source_target_physical_grid",
        "endpoint_provenance": endpoint_disclosure,
    }
    return _CaseResult(
        public=public,
        candidate=candidate,
        predictive_mean=predictive,
        oracle=oracle,
        copy_source=copy_metrics,
        sampling_signature=_sampling_signature(
            metadata,
            sampling,
            sample_count=int(samples.shape[0]),
            checkpoint_provenance=checkpoint_provenance,
            endpoint_disclosure=endpoint_disclosure,
        ),
        preprocessing_signature=target_preprocessing_signature,
    )


def _group_report(
    cases: Sequence[_CaseResult],
    *,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    first = cases[0].public
    patient_count = len({str(case.public["patient_id"]) for case in cases})
    candidate_names = sorted({name for case in cases for name in case.candidate})
    predictive_names = sorted({name for case in cases for name in case.predictive_mean})

    candidate_report: dict[str, Any] = {}
    for name in candidate_names:
        pooled = [value for case in cases for value in case.candidate.get(name, [])]
        case_means: list[tuple[str, float | None]] = []
        for case in cases:
            finite = [
                float(value)
                for value in case.candidate.get(name, [])
                if value is not None and math.isfinite(float(value))
            ]
            case_means.append(
                (str(case.public["patient_id"]), float(np.mean(finite)) if finite else None)
            )
        candidate_report[name] = {
            "candidate_descriptive": _describe(pooled),
            "case_candidate_mean_patient_bootstrap": _patient_ci(
                case_means,
                samples=bootstrap_samples,
                confidence=bootstrap_confidence,
                seed=bootstrap_seed,
            ),
        }

    predictive_report = {
        name: _patient_ci(
            (
                (str(case.public["patient_id"]), case.predictive_mean.get(name))
                for case in cases
            ),
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
        for name in predictive_names
    }
    oracle_report = {
        "mae": _patient_ci(
            ((str(case.public["patient_id"]), case.oracle.get("mae")) for case in cases),
            samples=bootstrap_samples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        ),
        "warning": "target-selected best-of-K; not a primary prospective metric",
    }
    copy_names = sorted(
        {
            name
            for case in cases
            if case.copy_source is not None
            for name in case.copy_source
        }
    )
    copy_report = {
        "case_count": sum(case.copy_source is not None for case in cases),
        "metrics": {
            name: _patient_ci(
                (
                    (
                        str(case.public["patient_id"]),
                        case.copy_source.get(name) if case.copy_source is not None else None,
                    )
                    for case in cases
                ),
                samples=bootstrap_samples,
                confidence=bootstrap_confidence,
                seed=bootstrap_seed,
            )
            for name in copy_names
        },
    }
    return {
        "direction": first["direction"],
        "time_pair": first["time_pair"],
        "split": first["split"],
        "endpoint_provenance": first["endpoint_provenance"],
        "case_count": len(cases),
        "patient_count": patient_count,
        "candidate_distribution": candidate_report,
        "predictive_mean": predictive_report,
        "oracle": oracle_report,
        "copy_source_no_change": copy_report,
    }


def evaluate_cohort(
    manifest: str | Path | Iterable[Mapping[str, Any]],
    *,
    pair_manifest: str | Path | Iterable[Mapping[str, Any]],
    data_range: float,
    foreground_threshold: float = -0.95,
    bootstrap_samples: int = 1000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate a formal JSONL cohort, grouping direction and fixed time pair."""

    if not math.isfinite(float(data_range)) or float(data_range) <= 0:
        raise ValueError("data_range must be finite and positive")
    if not math.isfinite(float(foreground_threshold)):
        raise ValueError("foreground_threshold must be finite")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0 < bootstrap_confidence < 1:
        raise ValueError("bootstrap_confidence must lie in (0, 1)")

    trusted_pairs = _read_trusted_pairs(pair_manifest)
    records, base = _read_cohort_records(manifest)
    checkpoint_cache: dict[Path, tuple[str, Mapping[str, Any]]] = {}
    cases = [
        _evaluate_case(
            record,
            base=base,
            trusted_pairs=trusted_pairs,
            checkpoint_cache=checkpoint_cache,
            data_range=float(data_range),
            foreground_threshold=float(foreground_threshold),
        )
        for record in records
    ]
    split_values = {str(case.public["split"]) for case in cases}
    if len(split_values) != 1:
        raise ValueError("one formal cohort report cannot mix validation and test patients")
    sampling_signatures = {
        _canonical(
            {
                key: value
                for key, value in case.sampling_signature.items()
                if key != "source_sha256"
            }
        )
        for case in cases
    }
    if len(sampling_signatures) != 1:
        raise ValueError("cohort predictions do not share one sampling/model provenance")
    preprocessing_signatures = {_canonical(case.preprocessing_signature) for case in cases}
    if len(preprocessing_signatures) != 1:
        raise ValueError("cohort cases do not share one preprocessing provenance")

    grouped: dict[tuple[str, tuple[str, str]], list[_CaseResult]] = defaultdict(list)
    for case in cases:
        grouped[(str(case.public["direction"]), tuple(case.public["time_pair"]))].append(case)
    group_reports = [
        _group_report(
            grouped[key],
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
            bootstrap_seed=bootstrap_seed,
        )
        for key in sorted(grouped)
    ]
    return {
        "task": "longitudinal_generation_cohort",
        "split": next(iter(split_values)),
        "case_count": len(cases),
        "patient_count": len({str(case.public["patient_id"]) for case in cases}),
        "data_range": float(data_range),
        "foreground_rule": f"target > {float(foreground_threshold)} (evaluation only)",
        "aggregation_rule": "average repeated cases within patient before patient bootstrap",
        "sampling_provenance": cases[0].sampling_signature,
        "source_archive_bindings": [
            {
                "prediction": case.public["prediction"],
                "source": case.public["source"],
                "source_sha256": case.sampling_signature["source_sha256"],
            }
            for case in cases
        ],
        "endpoint_provenance": {
            "endpoint_semantics": cases[0].sampling_signature["endpoint_semantics"],
            "residual_endpoint_consent": cases[0].sampling_signature[
                "residual_endpoint_consent"
            ],
            "decoded_endpoint_contains_sigma_residual": cases[0].sampling_signature[
                "decoded_endpoint_contains_sigma_residual"
            ],
            "source_endpoint_approximation_case_count": sum(
                bool(case.public["endpoint_provenance"]["source_endpoint_approximation"])
                for case in cases
            ),
        },
        "trusted_pair_manifest": {
            "path": trusted_pairs.source,
            "split_hash": trusted_pairs.split_hash,
            "ordered_manifest_fingerprint": trusted_pairs.ordered_manifest_fingerprint,
            "record_count": trusted_pairs.manifest_record_count,
            "autoencoder_id": trusted_pairs.autoencoder_id,
            CACHED_PAIR_MANIFEST_FINGERPRINT: (
                trusted_pairs.cached_pair_manifest_fingerprint
            ),
            LATENT_STATISTICS_FINGERPRINT: (
                trusted_pairs.latent_statistics_fingerprint
            ),
        },
        "preprocessing_provenance": cases[0].preprocessing_signature,
        "groups": group_reports,
        "cases": [case.public for case in cases],
        "tumor_metrics": {
            "status": "unavailable",
            "reason": (
                "This cohort contract contains no trusted lesion masks; tumor ROI, Dice, FTV, "
                "tumor-volume, and tumor-change metrics were not computed."
            ),
        },
    }


__all__ = ["HELD_OUT_SPLITS", "evaluate_cohort"]
