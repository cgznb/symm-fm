"""Fail-closed contracts shared by paired training entry points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ispy2_symmflow.config import ConfigError, resolve_time_pairs


def require_configured_time_pairs(
    records: Sequence[Mapping[str, Any]], configured: object
) -> tuple[tuple[str, str], ...]:
    """Require every accepted pair to belong to the configured interval set."""

    data = configured if isinstance(configured, Mapping) else {"time_pairs": configured}
    try:
        expected = resolve_time_pairs(data)
    except ConfigError as exc:
        raise ValueError(str(exc)) from exc
    allowed = set(expected)
    observed_pairs: set[tuple[str, str]] = set()
    mismatches: list[dict[str, Any]] = []
    for record in records:
        observed = (
            str(record.get("earlier_stage", "")).strip(),
            str(record.get("later_stage", "")).strip(),
        )
        observed_pairs.add(observed)
        if observed not in allowed:
            mismatches.append(
                {"pair_id": record.get("pair_id"), "observed": list(observed)}
            )
    if mismatches:
        preview = mismatches[:5]
        suffix = f" (and {len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
        raise ValueError(
            f"QC-passing cached pairs must match configured time_pairs "
            f"{[list(pair) for pair in expected]}; mismatches={preview}{suffix}"
        )
    missing = [pair for pair in expected if pair not in observed_pairs]
    if missing:
        raise ValueError(
            "configured time_pairs have no QC-passing cached pairs: "
            + ", ".join(f"{left}->{right}" for left, right in missing)
        )
    return expected


def require_configured_time_pair(
    records: Sequence[Mapping[str, Any]], configured_pair: object
) -> tuple[str, str]:
    """Require every QC-passing pair to use the configured earlier/later stages."""

    if not isinstance(configured_pair, Sequence) or isinstance(
        configured_pair, (str, bytes)
    ) or len(configured_pair) != 2:
        raise ValueError("data.time_pair must contain exactly [earlier, later]")
    expected = tuple(str(value).strip() for value in configured_pair)
    try:
        return require_configured_time_pairs(
            records, {"time_pair": list(expected)}
        )[0]
    except ValueError as exc:
        message = str(exc).replace("configured time_pairs", "configured time_pair")
        message = message.replace(str([list(expected)]), str(list(expected)))
        raise ValueError(message) from exc


__all__ = ["require_configured_time_pair", "require_configured_time_pairs"]
