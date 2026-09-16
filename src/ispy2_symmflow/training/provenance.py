"""Fail-closed provenance checks for autoencoder and latent cache artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ispy2_symmflow.utils.hashing import stable_hash


LATENT_STATISTICS_FINGERPRINT = "latent_statistics_fingerprint"
CACHED_PAIR_MANIFEST_FINGERPRINT = "cached_pair_manifest_fingerprint"
CACHED_PAIR_MANIFEST_RECORD_COUNT = "cached_pair_manifest_record_count"

_CACHED_PAIR_BINDING_FIELDS = frozenset(
    {
        LATENT_STATISTICS_FINGERPRINT,
        CACHED_PAIR_MANIFEST_FINGERPRINT,
        CACHED_PAIR_MANIFEST_RECORD_COUNT,
    }
)


def cached_pair_manifest_fingerprint(
    records: Sequence[Mapping[str, Any]],
) -> str:
    """Hash ordered pair contracts without their self-referential binding envelope."""

    contracts = [
        {
            str(key): value
            for key, value in record.items()
            if str(key) not in _CACHED_PAIR_BINDING_FIELDS
        }
        for record in records
    ]
    return stable_hash(contracts)


def bind_cached_pair_manifest_provenance(
    records: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind an ordered cached-pair contract to its statistics and records.

    The pair digest excludes only the three binding-envelope fields above. The
    digest is then included in the statistics document before its own digest is
    calculated, keeping the dependency graph acyclic.
    """

    unsigned = [
        {
            str(key): value
            for key, value in record.items()
            if str(key) not in _CACHED_PAIR_BINDING_FIELDS
        }
        for record in records
    ]
    pair_fingerprint = cached_pair_manifest_fingerprint(unsigned)
    pair_count = len(unsigned)
    statistics_payload = {
        str(key): value
        for key, value in statistics.items()
        if str(key) not in _CACHED_PAIR_BINDING_FIELDS
    }
    statistics_payload.update(
        {
            CACHED_PAIR_MANIFEST_FINGERPRINT: pair_fingerprint,
            CACHED_PAIR_MANIFEST_RECORD_COUNT: pair_count,
        }
    )
    bound_statistics = bind_latent_statistics_fingerprint(statistics_payload)
    statistics_fingerprint = bound_statistics[LATENT_STATISTICS_FINGERPRINT]
    bound_records = [
        {
            **record,
            CACHED_PAIR_MANIFEST_FINGERPRINT: pair_fingerprint,
            CACHED_PAIR_MANIFEST_RECORD_COUNT: pair_count,
            LATENT_STATISTICS_FINGERPRINT: statistics_fingerprint,
        }
        for record in unsigned
    ]
    return bound_records, bound_statistics


def latent_statistics_fingerprint(statistics: Mapping[str, Any]) -> str:
    """Hash the complete statistics document without its self-referential field."""

    payload = {
        str(key): value
        for key, value in statistics.items()
        if str(key) != LATENT_STATISTICS_FINGERPRINT
    }
    return stable_hash(payload)


def bind_latent_statistics_fingerprint(
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy carrying the canonical fingerprint of all statistics fields."""

    bound = dict(statistics)
    bound[LATENT_STATISTICS_FINGERPRINT] = latent_statistics_fingerprint(bound)
    return bound


def require_latent_statistics_fingerprint(statistics: Mapping[str, Any]) -> str:
    """Validate and return a statistics document's embedded fingerprint."""

    observed = str(statistics.get(LATENT_STATISTICS_FINGERPRINT, "")).strip()
    if not observed:
        raise ValueError(
            f"latent statistics are missing {LATENT_STATISTICS_FINGERPRINT}"
        )
    expected = latent_statistics_fingerprint(statistics)
    if observed != expected:
        raise ValueError("latent statistics fingerprint does not match its contents")
    return observed


def _required_text(value: Mapping[str, Any], key: str, *, label: str) -> str:
    text = str(value.get(key, "")).strip()
    if not text:
        raise ValueError(f"{label} is missing {key!r}")
    return text


def _patient_split_assignments(
    records: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    visit_ids: set[str] = set()
    for record in records:
        patient_id = _required_text(record, "patient_id", label=label)
        visit_id = _required_text(record, "visit_id", label=label)
        split = _required_text(record, "split", label=label)
        if visit_id in visit_ids:
            raise ValueError(f"{label} contains duplicate visit_id {visit_id!r}")
        visit_ids.add(visit_id)
        previous = assignments.setdefault(patient_id, split)
        if previous != split:
            raise ValueError(
                f"patient {patient_id!r} occurs in both {previous!r} and {split!r}"
            )
    return assignments


def validate_autoencoder_cache_manifest(
    records: Sequence[Mapping[str, Any]],
    checkpoint_header: Mapping[str, Any],
    *,
    preprocessing_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind latent caching to the AE's complete ordered training manifest."""

    if not records:
        raise ValueError("prepared visit manifest is empty")
    assignments = _patient_split_assignments(records, label="prepared visit manifest")
    split_fingerprint = stable_hash(assignments)
    checkpoint_split = str(checkpoint_header.get("split_hash", "")).strip()
    if not checkpoint_split:
        raise ValueError("autoencoder checkpoint has no training split_hash")
    if checkpoint_split != split_fingerprint:
        raise ValueError(
            "autoencoder checkpoint split_hash differs from the complete prepared visit manifest"
        )

    extra = checkpoint_header.get("extra")
    training_signature = (
        extra.get("training_signature") if isinstance(extra, Mapping) else None
    )
    if not isinstance(training_signature, Mapping):
        raise ValueError("autoencoder checkpoint has no training_signature")
    if training_signature.get("stage") != "autoencoder":
        raise ValueError("autoencoder checkpoint training_signature has the wrong stage")
    ordered_fingerprint = stable_hash(records)
    if (
        str(training_signature.get("ordered_manifest_fingerprint", ""))
        != ordered_fingerprint
    ):
        raise ValueError(
            "autoencoder checkpoint ordered manifest fingerprint differs from the complete "
            "prepared visit manifest"
        )
    if int(training_signature.get("manifest_record_count", -1)) != len(records):
        raise ValueError(
            "autoencoder checkpoint manifest record count differs from the complete "
            "prepared visit manifest"
        )

    checkpoint_config = checkpoint_header.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("autoencoder checkpoint has no training configuration")
    if str(training_signature.get("config_fingerprint", "")) != stable_hash(
        checkpoint_config
    ):
        raise ValueError(
            "autoencoder checkpoint training_signature has an invalid config fingerprint"
        )
    data_config = checkpoint_config.get("data")
    raw_roles = data_config.get("phase_channels") if isinstance(data_config, Mapping) else None
    if not isinstance(raw_roles, Sequence) or isinstance(raw_roles, (str, bytes)):
        raise ValueError("autoencoder checkpoint has no phase-channel contract")
    expected_roles = [str(role).strip() for role in raw_roles]
    if not expected_roles or any(not role for role in expected_roles):
        raise ValueError("autoencoder checkpoint phase-channel contract is invalid")
    observed_roles = preprocessing_signature.get("phase_roles")
    if not isinstance(observed_roles, Sequence) or isinstance(
        observed_roles, (str, bytes)
    ):
        raise ValueError("prepared visits have no phase-role provenance")
    if [str(role).strip() for role in observed_roles] != expected_roles:
        raise ValueError(
            "prepared visit phase roles differ from the autoencoder checkpoint configuration"
        )
    return {
        "split_hash": split_fingerprint,
        "ordered_manifest_fingerprint": ordered_fingerprint,
        "manifest_record_count": len(records),
        "phase_roles": expected_roles,
    }


def _archive_scalar(archive: Any, key: str, *, path: Path) -> str:
    if key not in archive:
        raise ValueError(f"latent archive has no {key!r} provenance: {path}")
    value = np.asarray(archive[key])
    if value.ndim != 0:
        raise ValueError(f"latent archive {key!r} provenance is not scalar: {path}")
    return str(value.item())


def _validate_latent_archive_binding(
    path: str | Path,
    *,
    visit_id: str,
    patient_id: str,
    split: str,
    autoencoder_id: str,
    statistics_fingerprint: str,
    pair_manifest_fingerprint: str,
    pair_manifest_record_count: int,
) -> None:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    expected = {
        "visit_id": visit_id,
        "patient_id": patient_id,
        "split": split,
        "autoencoder_id": autoencoder_id,
        LATENT_STATISTICS_FINGERPRINT: statistics_fingerprint,
        CACHED_PAIR_MANIFEST_FINGERPRINT: pair_manifest_fingerprint,
        CACHED_PAIR_MANIFEST_RECORD_COUNT: str(pair_manifest_record_count),
    }
    with np.load(source, allow_pickle=False) as archive:
        if "latent" not in archive:
            raise ValueError(f"latent archive has no 'latent' array: {source}")
        for key, wanted in expected.items():
            observed = _archive_scalar(archive, key, path=source)
            if observed != wanted:
                raise ValueError(
                    f"latent {key} mismatch for {source}: "
                    f"observed={observed!r}, expected={wanted!r}"
                )


def validate_cached_pair_manifest_binding(
    records: Sequence[Mapping[str, Any]], statistics: Mapping[str, Any]
) -> tuple[str, str, int]:
    """Validate the complete ordered pair manifest without opening latent archives."""

    if not records:
        raise ValueError("cached pair manifest is empty")
    statistics_fingerprint = require_latent_statistics_fingerprint(statistics)
    pair_manifest_fingerprint = _required_text(
        statistics,
        CACHED_PAIR_MANIFEST_FINGERPRINT,
        label="latent statistics",
    )
    try:
        pair_manifest_record_count = int(
            statistics[CACHED_PAIR_MANIFEST_RECORD_COUNT]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"latent statistics have no valid {CACHED_PAIR_MANIFEST_RECORD_COUNT}"
        ) from exc
    if pair_manifest_record_count != len(records):
        raise ValueError(
            "cached pair manifest record count differs from latent statistics"
        )
    actual_pair_fingerprint = cached_pair_manifest_fingerprint(records)
    if actual_pair_fingerprint != pair_manifest_fingerprint:
        raise ValueError(
            "cached pair manifest fingerprint differs from latent statistics"
        )
    for record in records:
        pair_id = str(record.get("pair_id", "<unknown>"))
        record_pair_fingerprint = _required_text(
            record,
            CACHED_PAIR_MANIFEST_FINGERPRINT,
            label=f"cached pair {pair_id!r}",
        )
        if record_pair_fingerprint != pair_manifest_fingerprint:
            raise ValueError(
                f"cached pair {pair_id!r} manifest fingerprint differs from "
                "latent statistics"
            )
        try:
            record_pair_count = int(record[CACHED_PAIR_MANIFEST_RECORD_COUNT])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"cached pair {pair_id!r} has no valid "
                f"{CACHED_PAIR_MANIFEST_RECORD_COUNT}"
            ) from exc
        if record_pair_count != pair_manifest_record_count:
            raise ValueError(
                f"cached pair {pair_id!r} manifest record count differs from "
                "latent statistics"
            )
        record_fingerprint = _required_text(
            record,
            LATENT_STATISTICS_FINGERPRINT,
            label=f"cached pair {pair_id!r}",
        )
        if record_fingerprint != statistics_fingerprint:
            raise ValueError(
                f"cached pair {pair_id!r} latent statistics fingerprint differs from "
                "latent_statistics.json"
            )
    return (
        statistics_fingerprint,
        pair_manifest_fingerprint,
        pair_manifest_record_count,
    )


def validate_cached_latent_provenance(
    records: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    *,
    validate_manifest: bool = True,
) -> str:
    """Validate accepted pair records and their latent archives against statistics."""

    if validate_manifest:
        (
            statistics_fingerprint,
            pair_manifest_fingerprint,
            pair_manifest_record_count,
        ) = validate_cached_pair_manifest_binding(records, statistics)
    else:
        statistics_fingerprint = require_latent_statistics_fingerprint(statistics)
        pair_manifest_fingerprint = _required_text(
            statistics,
            CACHED_PAIR_MANIFEST_FINGERPRINT,
            label="latent statistics",
        )
        try:
            pair_manifest_record_count = int(
                statistics[CACHED_PAIR_MANIFEST_RECORD_COUNT]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"latent statistics have no valid {CACHED_PAIR_MANIFEST_RECORD_COUNT}"
            ) from exc

    autoencoder_id = _required_text(
        statistics, "autoencoder_id", label="latent statistics"
    )
    validated: dict[Path, tuple[str, str, str, str, str]] = {}
    for record in records:
        pair_id = str(record.get("pair_id", "<unknown>"))
        if _required_text(
            record,
            CACHED_PAIR_MANIFEST_FINGERPRINT,
            label=f"cached pair {pair_id!r}",
        ) != pair_manifest_fingerprint:
            raise ValueError(
                f"cached pair {pair_id!r} manifest fingerprint differs from "
                "latent statistics"
            )
        try:
            record_pair_count = int(record[CACHED_PAIR_MANIFEST_RECORD_COUNT])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"cached pair {pair_id!r} has no valid "
                f"{CACHED_PAIR_MANIFEST_RECORD_COUNT}"
            ) from exc
        if record_pair_count != pair_manifest_record_count:
            raise ValueError(
                f"cached pair {pair_id!r} manifest record count differs from "
                "latent statistics"
            )
        if _required_text(
            record,
            LATENT_STATISTICS_FINGERPRINT,
            label=f"cached pair {pair_id!r}",
        ) != statistics_fingerprint:
            raise ValueError(
                f"cached pair {pair_id!r} latent statistics fingerprint differs from "
                "latent_statistics.json"
            )
        record_autoencoder = _required_text(
            record, "autoencoder_id", label=f"cached pair {pair_id!r}"
        )
        if record_autoencoder != autoencoder_id:
            raise ValueError(
                f"cached pair {pair_id!r} autoencoder_id differs from latent statistics"
            )
        patient_id = _required_text(
            record, "patient_id", label=f"cached pair {pair_id!r}"
        )
        split = _required_text(record, "split", label=f"cached pair {pair_id!r}")
        for branch in ("earlier", "later"):
            visit_id = _required_text(
                record,
                f"{branch}_visit_id",
                label=f"cached pair {pair_id!r}",
            )
            path_text = _required_text(
                record,
                f"{branch}_latent_path",
                label=f"cached pair {pair_id!r}",
            )
            path = Path(path_text).expanduser().resolve()
            binding = (
                visit_id,
                patient_id,
                split,
                autoencoder_id,
                statistics_fingerprint,
            )
            previous = validated.get(path)
            if previous is not None and previous != binding:
                raise ValueError(
                    f"latent archive {path} is referenced with conflicting provenance"
                )
            if previous is None:
                _validate_latent_archive_binding(
                    path,
                    visit_id=visit_id,
                    patient_id=patient_id,
                    split=split,
                    autoencoder_id=autoencoder_id,
                    statistics_fingerprint=statistics_fingerprint,
                    pair_manifest_fingerprint=pair_manifest_fingerprint,
                    pair_manifest_record_count=pair_manifest_record_count,
                )
                validated[path] = binding
    return statistics_fingerprint
