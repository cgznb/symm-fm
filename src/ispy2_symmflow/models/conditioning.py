"""Typed conditioning tokens for clinical, treatment, and interval data."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn


MISSING_CATEGORY_INDEX = 0
UNKNOWN_CATEGORY_INDEX = 1
_CATEGORY_OFFSET = 2
_INTERNAL_CATEGORIES = {"__missing__", "__unknown__"}
_RESERVED_CATEGORY_LABELS = {"missing", "unknown"}
_FORBIDDEN_CONDITION_NAMES = {
    "pcr",
    "rcb",
    "outcome",
    "pathology",
    "target",
    "targetimage",
    "targetmask",
    "laterimage",
    "futureimage",
}
_FORBIDDEN_FRAGMENTS = (
    "pcr",
    "rcb",
    "outcome",
    "patholog",
    "response",
    "residualcancerburden",
)
_TARGET_DERIVED_PREFIXES = ("target", "future", "later")
_TARGET_DERIVED_SUFFIXES = (
    "image",
    "mri",
    "mask",
    "path",
    "bbox",
    "volume",
    "statistic",
    "statistics",
    "feature",
    "label",
)


def _normalized_name(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def is_forbidden_condition_name(name: str) -> bool:
    """Reject outcome and target-derived variants after punctuation normalization."""

    normalized = _normalized_name(name)
    if normalized in _FORBIDDEN_CONDITION_NAMES:
        return True
    if any(fragment in normalized for fragment in _FORBIDDEN_FRAGMENTS):
        return True
    return any(
        normalized.startswith(prefix) and normalized.endswith(suffix)
        for prefix in _TARGET_DERIVED_PREFIXES
        for suffix in _TARGET_DERIVED_SUFFIXES
    )


@dataclass(frozen=True)
class CategoricalField:
    """Definition of one unordered categorical condition."""

    name: str
    categories: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("categorical field names must be non-empty")
        categories = tuple(str(value) for value in self.categories)
        if len(categories) != len(set(categories)):
            raise ValueError(f"categorical field {self.name!r} has duplicate categories")
        if _INTERNAL_CATEGORIES.intersection(categories) or any(
            value.strip().lower() in _RESERVED_CATEGORY_LABELS for value in categories
        ):
            raise ValueError(
                f"categorical field {self.name!r} uses a reserved category name"
            )
        object.__setattr__(self, "categories", categories)

    @property
    def num_embeddings(self) -> int:
        """Embedding rows including explicit missing and unknown entries."""

        return len(self.categories) + _CATEGORY_OFFSET

    @property
    def category_to_index(self) -> dict[str, int]:
        return {
            category: index + _CATEGORY_OFFSET
            for index, category in enumerate(self.categories)
        }


@dataclass(frozen=True)
class NumericField:
    """Definition and training-set normalization for one numeric condition."""

    name: str
    mean: float
    std: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("numeric field names must be non-empty")
        if not math.isfinite(self.mean):
            raise ValueError(f"numeric field {self.name!r} mean must be finite")
        if not math.isfinite(self.std) or self.std <= 0:
            raise ValueError(f"numeric field {self.name!r} std must be positive and finite")


@dataclass(frozen=True)
class ConditionSchema:
    """Serializable, ordered schema for model-visible conditions.

    Categorical tokens are emitted first, followed by numeric tokens. Outcome and
    target-derived fields are rejected to make accidental target leakage explicit.
    """

    categorical_fields: tuple[CategoricalField, ...]
    numeric_fields: tuple[NumericField, ...]
    token_dim: int = 256
    version: int = 1

    def __post_init__(self) -> None:
        categorical = tuple(self.categorical_fields)
        numeric = tuple(self.numeric_fields)
        object.__setattr__(self, "categorical_fields", categorical)
        object.__setattr__(self, "numeric_fields", numeric)
        if not categorical and not numeric:
            raise ValueError("a condition schema must define at least one field")
        if self.token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if self.version <= 0:
            raise ValueError("schema version must be positive")

        names = [field.name for field in (*categorical, *numeric)]
        if len(names) != len(set(names)):
            raise ValueError("condition field names must be unique")
        forbidden = [name for name in names if is_forbidden_condition_name(name)]
        if forbidden:
            raise ValueError(
                "outcome or target-derived fields cannot be model conditions: "
                + ", ".join(sorted(forbidden))
            )

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(
            field.name for field in (*self.categorical_fields, *self.numeric_fields)
        )

    @property
    def num_tokens(self) -> int:
        return len(self.field_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "token_dim": self.token_dim,
            "categorical_fields": [asdict(field) for field in self.categorical_fields],
            "numeric_fields": [asdict(field) for field in self.numeric_fields],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConditionSchema":
        categorical_payload = payload.get(
            "categorical_fields", payload.get("categorical", ())
        )
        numeric_payload = payload.get("numeric_fields", payload.get("numeric", ()))
        if isinstance(categorical_payload, Mapping):
            expanded_categorical = []
            for name, definition in categorical_payload.items():
                categories = (
                    definition.get("categories", ())
                    if isinstance(definition, Mapping)
                    else definition
                )
                expanded_categorical.append(
                    {"name": name, "categories": categories}
                )
            categorical_payload = expanded_categorical
        categorical_fields: list[CategoricalField] = []
        for item in categorical_payload:
            if isinstance(item, CategoricalField):
                categorical_fields.append(item)
                continue
            if isinstance(item, str):
                item = {"name": item, "categories": ()}
            categories = tuple(
                str(value)
                for value in item["categories"]
                if str(value).strip().lower() not in _RESERVED_CATEGORY_LABELS
            )
            categorical_fields.append(
                CategoricalField(name=str(item["name"]), categories=categories)
            )

        statistics = payload.get("numeric_statistics", {})
        if isinstance(numeric_payload, Mapping):
            numeric_payload = [
                {"name": name, **definition}
                if isinstance(definition, Mapping)
                else {"name": name, "mean": definition[0], "std": definition[1]}
                for name, definition in numeric_payload.items()
            ]
        numeric_fields: list[NumericField] = []
        for item in numeric_payload:
            if isinstance(item, NumericField):
                numeric_fields.append(item)
                continue
            if isinstance(item, str):
                field_statistics = statistics.get(item)
                if not isinstance(field_statistics, Mapping):
                    raise ValueError(
                        f"numeric field {item!r} needs training-set mean/std statistics"
                    )
                item = {"name": item, **field_statistics}
            numeric_fields.append(
                NumericField(
                    name=str(item["name"]),
                    mean=float(item["mean"]),
                    std=float(item["std"]),
                )
            )
        return cls(
            categorical_fields=tuple(categorical_fields),
            numeric_fields=tuple(numeric_fields),
            token_dim=int(payload.get("token_dim", 256)),
            version=int(payload.get("version", 1)),
        )

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _value_length(value: Any) -> int:
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return 1
        if value.ndim == 2 and value.shape[1] == 1:
            return int(value.shape[0])
        if value.ndim != 1:
            raise ValueError("condition tensors must have shape [B] or [B, 1]")
        return int(value.shape[0])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 1


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return [value.item()]
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 1:
            raise ValueError("condition tensors must have shape [B] or [B, 1]")
        return value.detach().cpu().tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


class StructuredConditionEncoder(nn.Module):
    """Encode typed table fields as one cross-attention token per field."""

    def __init__(
        self,
        schema: ConditionSchema,
        *,
        hidden_dim: int | None = None,
        strict: bool = True,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.strict = strict
        hidden_dim = hidden_dim or schema.token_dim
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.categorical_embeddings = nn.ModuleDict(
            {
                field.name: nn.Embedding(field.num_embeddings, schema.token_dim)
                for field in schema.categorical_fields
            }
        )
        self.numeric_projections = nn.ModuleDict(
            {
                field.name: nn.Sequential(
                    nn.Linear(2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, schema.token_dim),
                )
                for field in schema.numeric_fields
            }
        )
        self.field_embeddings = nn.Embedding(schema.num_tokens, schema.token_dim)
        self.type_embeddings = nn.Embedding(2, schema.token_dim)
        self.output_norm = nn.LayerNorm(schema.token_dim)

    @property
    def output_dim(self) -> int:
        return self.schema.token_dim

    @property
    def num_tokens(self) -> int:
        return self.schema.num_tokens

    def _infer_batch_size(
        self, values: Mapping[str, Any], batch_size: int | None
    ) -> int:
        lengths = {
            _value_length(value)
            for name, value in values.items()
            if name in self.schema.field_names and value is not None
        }
        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            incompatible = lengths.difference({batch_size})
            if incompatible:
                raise ValueError(
                    f"condition batch lengths {sorted(lengths)} do not match {batch_size}"
                )
            return batch_size
        if not lengths:
            return 1
        if len(lengths) != 1:
            raise ValueError(f"condition fields have inconsistent batch lengths: {lengths}")
        inferred = lengths.pop()
        if inferred <= 0:
            raise ValueError("condition batches cannot be empty")
        return inferred

    def _categorical_indices(
        self,
        field: CategoricalField,
        raw_value: Any,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if raw_value is None:
            return torch.full(
                (batch_size,), MISSING_CATEGORY_INDEX, dtype=torch.long, device=device
            )
        values = _as_sequence(raw_value)
        if len(values) != batch_size:
            raise ValueError(f"field {field.name!r} has the wrong batch length")
        lookup = field.category_to_index
        encoded: list[int] = []
        for value in values:
            is_missing = value is None or value == ""
            if isinstance(value, float) and math.isnan(value):
                is_missing = True
            if is_missing:
                encoded.append(MISSING_CATEGORY_INDEX)
            else:
                encoded.append(lookup.get(str(value), UNKNOWN_CATEGORY_INDEX))
        return torch.tensor(encoded, dtype=torch.long, device=device)

    def _numeric_features(
        self,
        field: NumericField,
        raw_value: Any,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if raw_value is None:
            numeric = torch.full((batch_size,), float("nan"), device=device, dtype=dtype)
        elif isinstance(raw_value, Tensor):
            numeric = raw_value.reshape(-1).to(device=device, dtype=dtype)
        else:
            values = _as_sequence(raw_value)
            converted = [
                float("nan") if value is None or value == "" else float(value)
                for value in values
            ]
            numeric = torch.tensor(converted, device=device, dtype=dtype)
        if numeric.numel() != batch_size:
            raise ValueError(f"field {field.name!r} has the wrong batch length")
        if torch.any(torch.isinf(numeric)):
            raise ValueError(f"field {field.name!r} contains an infinite value")
        missing = torch.isnan(numeric)
        filled = torch.where(missing, torch.full_like(numeric, field.mean), numeric)
        standardized = (filled - field.mean) / field.std
        standardized = torch.where(missing, torch.zeros_like(standardized), standardized)
        return torch.stack((standardized, missing.to(dtype=dtype)), dim=-1)

    def forward(
        self,
        values: Mapping[str, Any],
        *,
        batch_size: int | None = None,
    ) -> Tensor:
        """Return condition tokens with shape ``[B, num_fields, token_dim]``."""

        if not isinstance(values, Mapping):
            raise TypeError("conditions must be provided as a mapping")
        unknown = set(values).difference(self.schema.field_names)
        if self.strict and unknown:
            raise KeyError(f"fields are not present in the condition schema: {sorted(unknown)}")

        batch_size = self._infer_batch_size(values, batch_size)
        reference = self.field_embeddings.weight
        device = reference.device
        dtype = reference.dtype
        tokens: list[Tensor] = []

        for field_index, field in enumerate(self.schema.categorical_fields):
            indices = self._categorical_indices(
                field,
                values.get(field.name),
                batch_size=batch_size,
                device=device,
            )
            token = self.categorical_embeddings[field.name](indices)
            token = token + self.field_embeddings.weight[field_index]
            token = token + self.type_embeddings.weight[0]
            tokens.append(token)

        numeric_offset = len(self.schema.categorical_fields)
        for numeric_index, field in enumerate(self.schema.numeric_fields):
            features = self._numeric_features(
                field,
                values.get(field.name),
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            token = self.numeric_projections[field.name](features)
            token = token + self.field_embeddings.weight[numeric_offset + numeric_index]
            token = token + self.type_embeddings.weight[1]
            tokens.append(token)

        return self.output_norm(torch.stack(tokens, dim=1))


def build_condition_encoder_from_config(
    config: Mapping[str, Any] | ConditionSchema,
) -> StructuredConditionEncoder:
    """Build a structured encoder from a direct or nested config mapping."""

    if isinstance(config, ConditionSchema):
        return StructuredConditionEncoder(config)
    section: Mapping[str, Any] = config
    if "model" in section and isinstance(section["model"], Mapping):
        section = section["model"]
    for key in ("conditioning", "conditions"):
        if key in section and isinstance(section[key], Mapping):
            section = section[key]
            break
    schema_payload = section.get("schema", section)
    if not isinstance(schema_payload, Mapping):
        raise TypeError("conditioning schema config must be a mapping")
    schema = ConditionSchema.from_dict(schema_payload)
    hidden_dim = section.get("hidden_dim")
    strict = bool(section.get("strict", True))
    return StructuredConditionEncoder(
        schema,
        hidden_dim=None if hidden_dim is None else int(hidden_dim),
        strict=strict,
    )


__all__ = [
    "CategoricalField",
    "ConditionSchema",
    "MISSING_CATEGORY_INDEX",
    "NumericField",
    "StructuredConditionEncoder",
    "UNKNOWN_CATEGORY_INDEX",
    "build_condition_encoder_from_config",
    "is_forbidden_condition_name",
]
