"""Fit condition vocabularies/statistics from the training-patient split only."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Iterable, Mapping

from ispy2_symmflow.models.conditioning import (
    CategoricalField,
    ConditionSchema,
    NumericField,
)
from ispy2_symmflow.training.datasets import pair_conditions


_MISSING_CATEGORY_LABELS = {
    "",
    "__missing__",
    "__unknown__",
    "missing",
    "unknown",
}


def _is_observed(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _MISSING_CATEGORY_LABELS
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _contains_observed_value(value: Any) -> bool:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_observed_value(item) for item in value)
    return _is_observed(value)


def validate_sampling_condition_availability(
    values: Mapping[str, Any], schema_provenance: Mapping[str, Any]
) -> None:
    """Reject inference values for fields that had no observations in training."""

    unavailable = schema_provenance.get("unavailable_fields")
    if not isinstance(unavailable, (list, tuple)) or any(
        not isinstance(name, str) for name in unavailable
    ):
        raise ValueError(
            "checkpoint condition schema provenance has no valid unavailable_fields"
        )
    supplied = sorted(
        name
        for name in unavailable
        if name in values and _contains_observed_value(values[name])
    )
    if supplied:
        raise ValueError(
            "conditions provide non-missing values for fields unavailable in training: "
            + ", ".join(supplied)
        )


def fit_condition_schema(
    pair_records: Iterable[Mapping[str, Any]],
    condition_config: Mapping[str, Any],
) -> tuple[ConditionSchema, dict[str, Any]]:
    """Fit only from records labeled train and report entirely missing fields."""

    records = [dict(record) for record in pair_records]
    training = [record for record in records if record.get("split") == "train"]
    if not training:
        raise ValueError("condition schema requires at least one training pair")
    conditions = [pair_conditions(record) for record in training]
    all_conditions = [pair_conditions(record) for record in records]
    configured_categorical = condition_config.get("categorical", {})
    if isinstance(configured_categorical, Mapping):
        categorical_names = list(configured_categorical)
    else:
        categorical_names = [str(value) for value in configured_categorical]
        configured_categorical = {}
    numeric_names = [str(value) for value in condition_config.get("numeric", ())]
    configured_names = set(categorical_names) | set(numeric_names)
    if len(configured_names) != len(categorical_names) + len(numeric_names):
        raise ValueError("condition fields cannot be configured as both categorical and numeric")
    observed_names = {
        str(name)
        for condition in all_conditions
        for name in condition
    }
    unconfigured = sorted(observed_names - configured_names)
    if unconfigured:
        raise ValueError(
            "observed condition fields are not configured: " + ", ".join(unconfigured)
        )
    categorical_fields: list[CategoricalField] = []
    unavailable: list[str] = []
    train_observed_coverage: dict[str, dict[str, Any]] = {}
    for name in categorical_names:
        configured = [
            str(value)
            for value in configured_categorical.get(name, ())
            if str(value).strip().lower() not in _MISSING_CATEGORY_LABELS
        ]
        observed = sorted(
            {
                str(condition[name])
                for condition in conditions
                if _is_observed(condition.get(name))
            }
        )
        categories = tuple(dict.fromkeys([*configured, *observed]))
        train_observed_coverage[name] = {
            "kind": "categorical",
            "observation_count": sum(
                _is_observed(condition.get(name)) for condition in conditions
            ),
            "observed_categories": observed,
        }
        if not observed:
            unavailable.append(name)
        if not categories:
            categories = ("unavailable_in_training",)
        categorical_fields.append(CategoricalField(name=name, categories=categories))

    numeric_fields: list[NumericField] = []
    for name in numeric_names:
        values: list[float] = []
        for condition in conditions:
            raw = condition.get(name)
            if raw in {None, ""}:
                continue
            value = float(raw)
            if math.isfinite(value):
                values.append(value)
        if values:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = max(math.sqrt(variance), 1e-6)
        else:
            mean, std = 0.0, 1.0
            unavailable.append(name)
        train_observed_coverage[name] = {
            "kind": "numeric",
            "observation_count": len(values),
        }
        numeric_fields.append(NumericField(name=name, mean=mean, std=std))
    schema = ConditionSchema(
        categorical_fields=tuple(categorical_fields),
        numeric_fields=tuple(numeric_fields),
        token_dim=int(condition_config.get("token_dim", 256)),
    )
    provenance = {
        "fit_split": "train",
        "fit_pair_ids": sorted(str(record.get("pair_id")) for record in training),
        "unavailable_fields": sorted(unavailable),
        "train_observed_coverage": train_observed_coverage,
        "schema_fingerprint": schema.fingerprint,
    }
    return schema, provenance
