"""Deterministic posterior-mean latent caching with train-only statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ispy2_symmflow.data.manifest import build_pair_manifest, visit_from_dict
from ispy2_symmflow.training.datasets import (
    load_validated_prepared_image,
    read_jsonl,
    validate_prepared_records,
    write_jsonl,
)
from ispy2_symmflow.training.provenance import (
    CACHED_PAIR_MANIFEST_FINGERPRINT,
    CACHED_PAIR_MANIFEST_RECORD_COUNT,
    LATENT_STATISTICS_FINGERPRINT,
    bind_cached_pair_manifest_provenance,
    validate_autoencoder_cache_manifest,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _manifest_path(value: Any, *, base: Path, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is missing")
    path = Path(text).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _has_error_qc(record: Mapping[str, Any]) -> bool:
    return any(
        isinstance(event, Mapping)
        and str(event.get("severity", "")).strip().lower() == "error"
        for event in (record.get("qc") or ())
    )


def _validate_pairs_against_prepared_visits(
    pairs: list[dict[str, Any]],
    visits: list[dict[str, Any]],
    *,
    visit_manifest_base: Path,
    pair_manifest_base: Path,
) -> set[str]:
    """Rebuild pair contracts and independently identify unsafe pairs."""

    typed_visits = [visit_from_dict(record) for record in visits]
    visits_by_id: dict[str, dict[str, Any]] = {}
    assignments: dict[str, str] = {}
    for record, visit in zip(visits, typed_visits, strict=True):
        if visit.visit_id in visits_by_id:
            raise ValueError(
                f"prepared visit manifest duplicates visit_id {visit.visit_id!r}"
            )
        visits_by_id[visit.visit_id] = record
        if not visit.split:
            raise ValueError(f"prepared visit {visit.visit_id!r} has no split")
        previous = assignments.setdefault(visit.patient_id, visit.split)
        if previous != visit.split:
            raise ValueError(
                f"patient {visit.patient_id!r} has conflicting prepared visit splits"
            )

    stage_pairs: set[tuple[str, str]] = set()
    for pair in pairs:
        earlier_stage = str(pair.get("earlier_stage", "")).strip()
        later_stage = str(pair.get("later_stage", "")).strip()
        if not earlier_stage or not later_stage:
            raise ValueError(
                f"pair {pair.get('pair_id', '<unknown>')!r} is missing endpoint stages"
            )
        stage_pairs.add((earlier_stage, later_stage))

    expected_by_visits: dict[tuple[str, str], dict[str, Any]] = {}
    for earlier_stage, later_stage in sorted(stage_pairs):
        expected_pairs = build_pair_manifest(
            typed_visits,
            assignments,
            earlier_stage=earlier_stage,
            later_stage=later_stage,
            require_three_phase=False,
        )
        for expected in expected_pairs:
            expected_by_visits[(expected.earlier_visit_id, expected.later_visit_id)] = (
                expected.to_dict()
            )

    seen_endpoints: set[tuple[str, str]] = set()
    scalar_fields = (
        "pair_id",
        "patient_id",
        "collection",
        "split",
        "earlier_stage",
        "later_stage",
        "delta_days",
        "observed_delta_days",
        "interval_missing",
        "interval_source",
    )
    mapping_fields = ("baseline_clinical", "treatment")
    loaded_archives: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {}
    independently_rejected: set[str] = set()
    for pair in pairs:
        label = f"pair {pair.get('pair_id', '<unknown>')!r}"
        pair_id = str(pair.get("pair_id", ""))
        endpoints = (
            str(pair.get("earlier_visit_id", "")).strip(),
            str(pair.get("later_visit_id", "")).strip(),
        )
        if not all(endpoints):
            raise ValueError(f"{label} is missing endpoint visit IDs")
        if endpoints in seen_endpoints:
            raise ValueError(f"pair manifest duplicates endpoint visits {endpoints!r}")
        seen_endpoints.add(endpoints)
        expected = expected_by_visits.get(endpoints)
        if expected is None:
            raise ValueError(
                f"{label} endpoints/stages do not form a pair in the prepared visit manifest"
            )
        for field in scalar_fields:
            if _canonical(pair.get(field)) != _canonical(expected.get(field)):
                raise ValueError(
                    f"{label} {field} differs from the prepared visit manifest"
                )
        for field in mapping_fields:
            observed_mapping = pair.get(field) or {}
            expected_mapping = expected.get(field) or {}
            if not isinstance(observed_mapping, Mapping) or _canonical(
                observed_mapping
            ) != _canonical(expected_mapping):
                raise ValueError(
                    f"{label} {field} differs from the prepared visit manifest"
                )
        for branch in ("earlier", "later"):
            field = f"{branch}_prepared_path"
            observed_path = _manifest_path(
                pair.get(field), base=pair_manifest_base, label=f"{label} {field}"
            )
            expected_path = _manifest_path(
                expected.get(field),
                base=visit_manifest_base,
                label=f"prepared visit {field}",
            )
            if observed_path != expected_path:
                raise ValueError(
                    f"{label} {field} differs from the prepared visit manifest"
                )

        endpoint_records = [visits_by_id[visit_id] for visit_id in endpoints]
        if any(_has_error_qc(record) for record in endpoint_records) or _has_error_qc(
            expected
        ):
            independently_rejected.add(pair_id)
            continue

        endpoint_archives: list[tuple[torch.Tensor, dict[str, Any]]] = []
        for record in endpoint_records:
            visit_id = str(record["visit_id"])
            loaded = loaded_archives.get(visit_id)
            if loaded is None:
                image, metadata, _ = load_validated_prepared_image(record)
                loaded = (image, metadata)
                loaded_archives[visit_id] = loaded
            endpoint_archives.append(loaded)
        (earlier_image, earlier_metadata), (later_image, later_metadata) = (
            endpoint_archives
        )
        shape_match = tuple(earlier_image.shape) == tuple(later_image.shape)
        affine_match = np.allclose(
            np.asarray(earlier_metadata["affine_lps"], dtype=np.float64),
            np.asarray(later_metadata["affine_lps"], dtype=np.float64),
            rtol=1e-5,
            atol=1e-4,
        )
        spacing_match = np.allclose(
            np.asarray(earlier_metadata["spacing_dhw"], dtype=np.float64),
            np.asarray(later_metadata["spacing_dhw"], dtype=np.float64),
            rtol=1e-6,
            atol=1e-6,
        )
        if not (shape_match and affine_match and spacing_match):
            independently_rejected.add(pair_id)

    return independently_rejected


@torch.no_grad()
def cache_latents(
    autoencoder: nn.Module,
    visit_manifest: str | Path,
    output_dir: str | Path,
    *,
    device: torch.device,
    autoencoder_id: str,
    autoencoder_checkpoint_header: Mapping[str, Any],
    pair_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Fit shared moments on train patients, then cache every visit posterior mean."""

    records = read_jsonl(visit_manifest)
    missing_prepared = [
        str(record.get("visit_id", "<unknown>"))
        for record in records
        if not str(record.get("prepared_path", "")).strip()
    ]
    if missing_prepared:
        raise ValueError(
            "complete prepared visit manifest contains visits without prepared_path: "
            f"{missing_prepared}"
        )
    prepared = records
    training = [record for record in prepared if record.get("split") == "train"]
    if not training:
        raise ValueError("cannot fit latent statistics without prepared training visits")
    patient_splits: dict[str, set[str]] = {}
    for record in prepared:
        patient_splits.setdefault(str(record["patient_id"]), set()).add(str(record["split"]))
    leaked = {patient: splits for patient, splits in patient_splits.items() if len(splits) != 1}
    if leaked:
        raise ValueError(f"patient split leakage detected before latent caching: {leaked}")
    pair_records = read_jsonl(pair_manifest) if pair_manifest is not None else []
    rejected_pair_ids = [
        str(pair.get("pair_id"))
        for pair in pair_records
        if _has_error_qc(pair)
    ]
    rejected_pair_ids = list(dict.fromkeys(rejected_pair_ids))
    cacheable_pairs = [
        pair for pair in pair_records if str(pair.get("pair_id")) not in rejected_pair_ids
    ]
    if pair_manifest is not None and not cacheable_pairs:
        raise ValueError(
            "pair manifest has no QC-passing longitudinal pairs; resolve physical-grid "
            "alignment and other pair errors before latent caching"
        )
    prepared_visit_ids = {str(record["visit_id"]) for record in prepared}
    unresolved_pairs = [
        {
            "pair_id": pair.get("pair_id"),
            "missing_earlier": str(pair["earlier_visit_id"]) not in prepared_visit_ids,
            "missing_later": str(pair["later_visit_id"]) not in prepared_visit_ids,
        }
        for pair in cacheable_pairs
        if str(pair["earlier_visit_id"]) not in prepared_visit_ids
        or str(pair["later_visit_id"]) not in prepared_visit_ids
    ]
    if unresolved_pairs:
        raise ValueError(
            "pair manifest references visits absent from the prepared visit manifest: "
            f"{unresolved_pairs}"
        )

    if cacheable_pairs:
        independently_rejected = _validate_pairs_against_prepared_visits(
            cacheable_pairs,
            prepared,
            visit_manifest_base=Path(visit_manifest).expanduser().resolve().parent,
            pair_manifest_base=Path(pair_manifest).expanduser().resolve().parent,
        )
        rejected_pair_ids.extend(
            str(pair.get("pair_id"))
            for pair in cacheable_pairs
            if str(pair.get("pair_id")) in independently_rejected
        )
        rejected_pair_ids = list(dict.fromkeys(rejected_pair_ids))
        cacheable_pairs = [
            pair
            for pair in cacheable_pairs
            if str(pair.get("pair_id")) not in independently_rejected
        ]
        if not cacheable_pairs:
            raise ValueError(
                "pair manifest has no QC-passing longitudinal pairs after independent "
                "endpoint-QC and physical-grid validation"
            )

    # This eager pass binds every path to its manifest identity before statistics are
    # accumulated or any cache artifact can be written.
    source_preprocessing_signature = validate_prepared_records(prepared)
    autoencoder_manifest_provenance = validate_autoencoder_cache_manifest(
        records,
        autoencoder_checkpoint_header,
        preprocessing_signature=source_preprocessing_signature,
    )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    autoencoder = autoencoder.to(device).eval()
    sums: torch.Tensor | None = None
    square_sums: torch.Tensor | None = None
    element_count = 0
    for record in training:
        image, _, _ = load_validated_prepared_image(record)
        latent = autoencoder.encode(image[None].to(device), normalize=False)
        reduced = latent.double().sum(dim=(0, 2, 3, 4))
        squares = latent.double().square().sum(dim=(0, 2, 3, 4))
        sums = reduced if sums is None else sums + reduced
        square_sums = squares if square_sums is None else square_sums + squares
        element_count += latent.shape[0] * latent.shape[2] * latent.shape[3] * latent.shape[4]
    if sums is None or square_sums is None or element_count < 2:
        raise ValueError("insufficient train latent elements for stable statistics")
    mean = sums / element_count
    variance = (square_sums / element_count - mean.square()).clamp_min(1e-12)
    std = variance.sqrt()
    autoencoder.set_latent_statistics(mean.float(), std.float())

    base_statistics = {
        "mean": mean.cpu().tolist(),
        "std": std.cpu().tolist(),
        "element_count_per_channel": element_count,
        "fit_split": "train",
        "fit_patient_ids": sorted(
            {str(record["patient_id"]) for record in training}
        ),
        "fit_visit_ids": sorted(str(record["visit_id"]) for record in training),
        "autoencoder_id": autoencoder_id,
        "source_preprocessing_signature": source_preprocessing_signature,
        "source_manifest": str(Path(visit_manifest).resolve()),
        "source_split_hash": autoencoder_manifest_provenance["split_hash"],
        "source_ordered_manifest_fingerprint": autoencoder_manifest_provenance[
            "ordered_manifest_fingerprint"
        ],
        "source_manifest_record_count": autoencoder_manifest_provenance[
            "manifest_record_count"
        ],
    }

    # Plan all derived paths before writing. This lets the cache sign the complete
    # ordered pair contract first, then include that digest in statistics and every
    # referenced latent without introducing a circular hash dependency.
    by_visit = {
        str(record["visit_id"]): str(destination / f"{record['visit_id']}.npz")
        for record in prepared
    }
    unsigned_cached_pairs: list[dict[str, Any]] = []
    if pair_manifest is not None:
        for pair in cacheable_pairs:
            earlier = by_visit.get(str(pair["earlier_visit_id"]))
            later = by_visit.get(str(pair["later_visit_id"]))
            assert earlier is not None and later is not None
            cached = dict(pair)
            for field in (
                CACHED_PAIR_MANIFEST_FINGERPRINT,
                CACHED_PAIR_MANIFEST_RECORD_COUNT,
                LATENT_STATISTICS_FINGERPRINT,
            ):
                cached.pop(field, None)
            cached["earlier_latent_path"] = earlier
            cached["later_latent_path"] = later
            cached["autoencoder_id"] = autoencoder_id
            unsigned_cached_pairs.append(cached)
    cached_pairs, statistics = bind_cached_pair_manifest_provenance(
        unsigned_cached_pairs, base_statistics
    )
    statistics_fingerprint = statistics[LATENT_STATISTICS_FINGERPRINT]
    pair_manifest_fingerprint = statistics[CACHED_PAIR_MANIFEST_FINGERPRINT]
    pair_manifest_record_count = statistics[CACHED_PAIR_MANIFEST_RECORD_COUNT]

    latent_records: list[dict[str, Any]] = []
    for record in prepared:
        image, metadata, _ = load_validated_prepared_image(record)
        latent = autoencoder.encode(image[None].to(device), normalize=True)[0]
        if not torch.isfinite(latent).all():
            raise FloatingPointError(f"non-finite latent for visit {record['visit_id']}")
        latent_path = Path(by_visit[str(record["visit_id"])])
        provenance = {
            "source_prepared_path": str(Path(record["prepared_path"]).resolve()),
            "autoencoder_id": autoencoder_id,
            "posterior": "mean",
            "normalization": "shared_train_patient_per_channel",
            "statistics_split": "train",
            LATENT_STATISTICS_FINGERPRINT: statistics_fingerprint,
            CACHED_PAIR_MANIFEST_FINGERPRINT: pair_manifest_fingerprint,
            CACHED_PAIR_MANIFEST_RECORD_COUNT: pair_manifest_record_count,
        }
        np.savez_compressed(
            latent_path,
            latent=latent.detach().cpu().float().numpy(),
            visit_id=str(record["visit_id"]),
            patient_id=str(record["patient_id"]),
            split=str(record["split"]),
            autoencoder_id=autoencoder_id,
            latent_statistics_fingerprint=statistics_fingerprint,
            cached_pair_manifest_fingerprint=pair_manifest_fingerprint,
            cached_pair_manifest_record_count=pair_manifest_record_count,
            affine_lps=np.asarray(metadata["affine_lps"], dtype=np.float64),
            spacing_dhw=np.asarray(metadata["spacing_dhw"], dtype=np.float32),
            provenance_json=json.dumps(provenance, sort_keys=True),
        )
        updated = dict(record)
        updated["latent_path"] = str(latent_path)
        updated["latent_provenance"] = provenance
        updated[LATENT_STATISTICS_FINGERPRINT] = statistics_fingerprint
        updated[CACHED_PAIR_MANIFEST_FINGERPRINT] = pair_manifest_fingerprint
        updated[CACHED_PAIR_MANIFEST_RECORD_COUNT] = pair_manifest_record_count
        latent_records.append(updated)

    visit_output = write_jsonl(destination / "visits.jsonl", latent_records)
    pair_output: Path | None = None
    if pair_manifest is not None:
        pair_output = write_jsonl(destination / "pairs.jsonl", cached_pairs)

    stats_path = destination / "latent_statistics.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "visit_manifest": str(visit_output),
        "pair_manifest": str(pair_output) if pair_output else None,
        "statistics_path": str(stats_path),
        "visit_count": len(latent_records),
        "pair_count": len(cached_pairs),
        "rejected_pair_count": len(rejected_pair_ids),
        "rejected_pair_ids": rejected_pair_ids,
        "statistics": statistics,
    }
