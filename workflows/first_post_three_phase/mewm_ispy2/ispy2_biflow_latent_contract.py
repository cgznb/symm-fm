from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .ispy2_dce0_world_latents import (
    ISPY2_DCE0_CONTINUOUS_CACHE_SCHEMA as LEGACY_CACHE_SCHEMA,
    ISPY2_DCE0_CONTINUOUS_NORMALIZATION as LEGACY_NORMALIZATION,
    ISPY2_DCE0_CONTINUOUS_PAYLOAD_SCHEMA as LEGACY_PAYLOAD_SCHEMA,
    ISPY2_DCE0_LATENT_SHAPE,
    ISPY2DCE0ContinuousLatentCache,
)


ISPY2_BIFLOW_CONTINUOUS_CACHE_SCHEMA = "mewm_ispy2_biflow_continuous_latents_v2"
ISPY2_BIFLOW_CONTINUOUS_PAYLOAD_SCHEMA = (
    "mewm_ispy2_biflow_continuous_latent_payload_v2"
)
ISPY2_BIFLOW_LATENT_NORMALIZATION = (
    "continuous_train_unique_visit_channel_zscore_v1"
)
ISPY2_BIFLOW_LATENT_STATISTICS_SCHEMA = (
    "mewm_ispy2_biflow_latent_channel_statistics_v1"
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _payload_path(root: Path, visit_id: str) -> Path:
    if not visit_id or "/" in visit_id or "\\" in visit_id:
        raise ValueError("I-SPY2 BiFlow latent visit ID is unsafe")
    return root / "visits" / f"{visit_id.replace(':', '__')}.pt"


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"I-SPY2 BiFlow {name} is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError(f"I-SPY2 BiFlow {name} is unreadable") from None
    if not isinstance(payload, dict):
        raise ValueError(f"I-SPY2 BiFlow {name} must be an object")
    return payload


def _load_legacy_payload(
    source: ISPY2DCE0ContinuousLatentCache,
    visit_id: str,
) -> tuple[torch.Tensor, str]:
    path = _payload_path(source.root, visit_id)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"I-SPY2 BiFlow source latent is missing: {visit_id}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError):
        raise ValueError(
            f"I-SPY2 BiFlow source latent is unreadable: {visit_id}"
        ) from None
    latent = payload.get("continuous_latent") if isinstance(payload, dict) else None
    split = payload.get("split") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != LEGACY_PAYLOAD_SCHEMA
        or payload.get("visit_id") != visit_id
        or split not in ("train", "val")
        or payload.get("vqgan_sha256") != source.identity.get("vqgan_sha256")
        or payload.get("codebook_sha256") != source.identity.get("codebook_sha256")
        or payload.get("data_contract_sha256")
        != source.identity.get("data_contract_sha256")
        or payload.get("normalization") != LEGACY_NORMALIZATION
        or not isinstance(latent, torch.Tensor)
        or latent.dtype != torch.float16
        or tuple(latent.shape) != ISPY2_DCE0_LATENT_SHAPE
        or not bool(torch.isfinite(latent.float()).all())
    ):
        raise ValueError(f"I-SPY2 BiFlow source latent is invalid: {visit_id}")
    return latent.contiguous(), split


def _statistics_payload(
    source: ISPY2DCE0ContinuousLatentCache,
) -> tuple[dict[str, Any], dict[str, str]]:
    count = 0
    mean = torch.zeros(ISPY2_DCE0_LATENT_SHAPE[0], dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    split_by_visit: dict[str, str] = {}
    train_visit_ids: list[str] = []
    elements_per_visit = math.prod(ISPY2_DCE0_LATENT_SHAPE[1:])

    for index, visit_id in enumerate(sorted(source.visit_ids), start=1):
        latent, split = _load_legacy_payload(source, visit_id)
        split_by_visit[visit_id] = split
        if split == "train":
            train_visit_ids.append(visit_id)
            value = latent.to(dtype=torch.float64)
            visit_variance, visit_mean = torch.var_mean(
                value, dim=(1, 2, 3), correction=0
            )
            visit_count = elements_per_visit
            total = count + visit_count
            delta = visit_mean - mean
            mean = mean + delta * (visit_count / total)
            m2 = (
                m2
                + visit_variance * visit_count
                + delta.square() * count * visit_count / total
            )
            count = total
        if index % 250 == 0 or index == len(source.visit_ids):
            print(
                f"I-SPY2 BiFlow latent statistics {index}/{len(source.visit_ids)} "
                f"(train_unique_visits={len(train_visit_ids)})",
                flush=True,
            )

    expected_train = source.identity.get("split_visit_counts", {}).get("train")
    if len(train_visit_ids) != expected_train or count <= 0:
        raise ValueError("I-SPY2 BiFlow training visit inventory changed")
    std = torch.sqrt(m2 / count)
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError("I-SPY2 BiFlow latent channel moments are non-finite")
    if not bool((std > 0.0).all()):
        raise ValueError("I-SPY2 BiFlow latent channel standard deviation is degenerate")

    statistics: dict[str, Any] = {
        "schema": ISPY2_BIFLOW_LATENT_STATISTICS_SCHEMA,
        "source_split": "train",
        "visit_selection": "unique_endpoint_visits",
        "visit_count": len(train_visit_ids),
        "visit_ids_sha256": _canonical_sha256(train_visit_ids),
        "element_count_per_channel": count,
        "accumulator_dtype": "float64",
        "variance_estimator": "population_ddof0",
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    statistics["sha256"] = _canonical_sha256(statistics)
    return statistics, split_by_visit


def _validate_statistics(
    value: Any,
    *,
    split_visit_counts: Any,
) -> dict[str, Any]:
    fields = {
        "schema",
        "source_split",
        "visit_selection",
        "visit_count",
        "visit_ids_sha256",
        "element_count_per_channel",
        "accumulator_dtype",
        "variance_estimator",
        "mean",
        "std",
        "sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("I-SPY2 BiFlow latent statistics fields are invalid")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    visit_count = value["visit_count"]
    element_count = value["element_count_per_channel"]
    expected_elements = (
        visit_count * math.prod(ISPY2_DCE0_LATENT_SHAPE[1:])
        if type(visit_count) is int
        else -1
    )
    mean = value["mean"]
    std = value["std"]
    valid_vectors = (
        isinstance(mean, list)
        and isinstance(std, list)
        and len(mean) == ISPY2_DCE0_LATENT_SHAPE[0]
        and len(std) == ISPY2_DCE0_LATENT_SHAPE[0]
        and all(type(item) in (int, float) and math.isfinite(item) for item in mean)
        and all(
            type(item) in (int, float) and math.isfinite(item) and item > 0.0
            for item in std
        )
    )
    visit_digest = value["visit_ids_sha256"]
    if (
        value["schema"] != ISPY2_BIFLOW_LATENT_STATISTICS_SCHEMA
        or value["source_split"] != "train"
        or value["visit_selection"] != "unique_endpoint_visits"
        or type(visit_count) is not int
        or visit_count <= 0
        or not isinstance(split_visit_counts, dict)
        or split_visit_counts.get("train") != visit_count
        or type(element_count) is not int
        or element_count != expected_elements
        or value["accumulator_dtype"] != "float64"
        or value["variance_estimator"] != "population_ddof0"
        or not valid_vectors
        or not isinstance(visit_digest, str)
        or len(visit_digest) != 64
        or any(character not in "0123456789abcdef" for character in visit_digest)
        or value["sha256"] != _canonical_sha256(unsigned)
    ):
        raise ValueError("I-SPY2 BiFlow latent statistics are invalid")
    return value


def _valid_v2_payload(
    path: Path,
    *,
    visit_id: str,
    split: str,
    identity: dict[str, Any],
) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError, EOFError):
        return False
    latent = payload.get("continuous_latent") if isinstance(payload, dict) else None
    statistics = identity["latent_statistics"]
    return bool(
        isinstance(payload, dict)
        and payload.get("schema") == ISPY2_BIFLOW_CONTINUOUS_PAYLOAD_SCHEMA
        and payload.get("visit_id") == visit_id
        and payload.get("split") == split
        and payload.get("vqgan_sha256") == identity["vqgan_sha256"]
        and payload.get("codebook_sha256") == identity["codebook_sha256"]
        and payload.get("data_contract_sha256") == identity["data_contract_sha256"]
        and payload.get("normalization") == ISPY2_BIFLOW_LATENT_NORMALIZATION
        and payload.get("latent_statistics_sha256") == statistics["sha256"]
        and isinstance(latent, torch.Tensor)
        and latent.dtype == torch.float16
        and tuple(latent.shape) == ISPY2_DCE0_LATENT_SHAPE
        and bool(torch.isfinite(latent.float()).all())
    )


class ISPY2BiFlowLatentCache:
    def __init__(
        self,
        root: str | Path,
        *,
        expected_normalization: str = ISPY2_BIFLOW_LATENT_NORMALIZATION,
    ) -> None:
        root_path = Path(root).expanduser()
        if root_path.is_symlink() or not root_path.is_dir():
            raise ValueError("I-SPY2 BiFlow latent cache is missing or unsafe")
        self.root = root_path.resolve()
        identity = _load_json(
            self.root / "cache_identity.json", "latent cache identity"
        )
        if (
            identity.get("schema") != ISPY2_BIFLOW_CONTINUOUS_CACHE_SCHEMA
            or identity.get("payload_schema")
            != ISPY2_BIFLOW_CONTINUOUS_PAYLOAD_SCHEMA
            or identity.get("normalization") != expected_normalization
            or identity.get("stored_representation")
            != "continuous_prequantization_float16_v1"
            or identity.get("latent_shape_czyx") != list(ISPY2_DCE0_LATENT_SHAPE)
            or identity.get("latent_dtype") != "float16"
            or not isinstance(identity.get("visit_ids"), list)
            or identity.get("visit_count") != len(identity["visit_ids"])
        ):
            raise ValueError("I-SPY2 BiFlow latent cache identity is invalid")
        self.identity = identity
        self.normalization = expected_normalization
        self.visit_ids = frozenset(str(value) for value in identity["visit_ids"])
        if len(self.visit_ids) != identity["visit_count"]:
            raise ValueError("I-SPY2 BiFlow latent visit inventory is invalid")
        self.latent_statistics = _validate_statistics(
            identity.get("latent_statistics"),
            split_visit_counts=identity.get("split_visit_counts"),
        )
        self._channel_mean = torch.tensor(
            self.latent_statistics["mean"], dtype=torch.float64
        ).reshape(ISPY2_DCE0_LATENT_SHAPE[0], 1, 1, 1)
        self._channel_std = torch.tensor(
            self.latent_statistics["std"], dtype=torch.float64
        ).reshape(ISPY2_DCE0_LATENT_SHAPE[0], 1, 1, 1)

    def load_continuous(self, visit_id: str) -> torch.Tensor:
        if visit_id not in self.visit_ids:
            raise ValueError(f"I-SPY2 BiFlow latent visit is not registered: {visit_id}")
        path = _payload_path(self.root, visit_id)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError, EOFError):
            raise ValueError(
                f"I-SPY2 BiFlow latent is unreadable: {visit_id}"
            ) from None
        split = payload.get("split") if isinstance(payload, dict) else None
        if not isinstance(split, str) or not _valid_v2_payload(
            path,
            visit_id=visit_id,
            split=split,
            identity=self.identity,
        ):
            raise ValueError(f"I-SPY2 BiFlow latent is invalid: {visit_id}")
        return payload["continuous_latent"].float().contiguous()

    def normalize(self, continuous: torch.Tensor) -> torch.Tensor:
        self._validate_latent(continuous, "continuous")
        mean = self._channel_mean.to(
            device=continuous.device, dtype=continuous.dtype
        )
        std = self._channel_std.to(device=continuous.device, dtype=continuous.dtype)
        normalized = (continuous - mean) / std
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("I-SPY2 BiFlow normalized latent is non-finite")
        return normalized.contiguous()

    def load(self, visit_id: str) -> torch.Tensor:
        return self.normalize(self.load_continuous(visit_id))

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        self._validate_latent(normalized, "normalized")
        mean = self._channel_mean.to(
            device=normalized.device, dtype=normalized.dtype
        )
        std = self._channel_std.to(device=normalized.device, dtype=normalized.dtype)
        continuous = normalized * std + mean
        if not bool(torch.isfinite(continuous).all()):
            raise ValueError("I-SPY2 BiFlow denormalized latent is non-finite")
        return continuous.contiguous()

    @staticmethod
    def _validate_latent(value: torch.Tensor, name: str) -> None:
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or tuple(value.shape[-4:]) != ISPY2_DCE0_LATENT_SHAPE
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"I-SPY2 BiFlow {name} latent is invalid")


def prepare_ispy2_biflow_latent_cache(
    source_root: str | Path,
    target_root: str | Path,
) -> Path:
    source = ISPY2DCE0ContinuousLatentCache(source_root)
    if (
        source.identity.get("schema") != LEGACY_CACHE_SCHEMA
        or source.identity.get("normalization") != LEGACY_NORMALIZATION
    ):
        raise ValueError("I-SPY2 BiFlow source must be the legacy continuous cache")
    target = Path(target_root).expanduser().resolve()
    if target == source.root:
        raise ValueError("I-SPY2 BiFlow latent cache requires a new root")
    if target.is_symlink():
        raise ValueError("I-SPY2 BiFlow latent cache root is unsafe")
    target.mkdir(parents=True, exist_ok=True)

    statistics, split_by_visit = _statistics_payload(source)
    identity = dict(source.identity)
    identity.update(
        {
            "schema": ISPY2_BIFLOW_CONTINUOUS_CACHE_SCHEMA,
            "payload_schema": ISPY2_BIFLOW_CONTINUOUS_PAYLOAD_SCHEMA,
            "stored_representation": "continuous_prequantization_float16_v1",
            "normalization": ISPY2_BIFLOW_LATENT_NORMALIZATION,
            "latent_statistics": statistics,
            "source_cache_identity_sha256": _canonical_sha256(source.identity),
        }
    )
    identity_path = target / "cache_identity.json"
    if identity_path.exists():
        existing = ISPY2BiFlowLatentCache(target)
        if existing.identity != identity:
            raise ValueError("I-SPY2 BiFlow latent cache identity mismatch")

    built = reused = 0
    for index, visit_id in enumerate(sorted(source.visit_ids), start=1):
        split = split_by_visit[visit_id]
        destination = _payload_path(target, visit_id)
        if _valid_v2_payload(
            destination,
            visit_id=visit_id,
            split=split,
            identity=identity,
        ):
            reused += 1
        else:
            continuous, observed_split = _load_legacy_payload(source, visit_id)
            if observed_split != split:
                raise ValueError("I-SPY2 BiFlow source latent split changed")
            _atomic_torch_save(
                destination,
                {
                    "schema": ISPY2_BIFLOW_CONTINUOUS_PAYLOAD_SCHEMA,
                    "visit_id": visit_id,
                    "split": split,
                    "continuous_latent": continuous,
                    "vqgan_sha256": identity["vqgan_sha256"],
                    "codebook_sha256": identity["codebook_sha256"],
                    "data_contract_sha256": identity["data_contract_sha256"],
                    "normalization": ISPY2_BIFLOW_LATENT_NORMALIZATION,
                    "latent_statistics_sha256": statistics["sha256"],
                },
            )
            built += 1
        if index % 250 == 0 or index == len(source.visit_ids):
            print(
                f"I-SPY2 BiFlow latent cache {index}/{len(source.visit_ids)} "
                f"(built={built}, reused={reused})",
                flush=True,
            )
    _atomic_json(identity_path, identity)
    ISPY2BiFlowLatentCache(target)
    return identity_path


@torch.no_grad()
def decode_ispy2_biflow_continuous(
    vqgan: nn.Module,
    continuous: torch.Tensor,
) -> torch.Tensor:
    if (
        not isinstance(continuous, torch.Tensor)
        or continuous.ndim != 6
        or tuple(continuous.shape[1:]) != (1, *ISPY2_DCE0_LATENT_SHAPE)
        or not continuous.is_floating_point()
        or not bool(torch.isfinite(continuous).all())
    ):
        raise ValueError("I-SPY2 BiFlow continuous decode latent is invalid")
    batch = continuous.shape[0]
    flat = continuous.reshape(batch, *continuous.shape[2:])
    quantized, _ = vqgan.quantizer(flat)
    decoded = vqgan.decode(quantized)
    return decoded.reshape(batch, 1, *decoded.shape[1:])


__all__ = [
    "ISPY2_BIFLOW_CONTINUOUS_CACHE_SCHEMA",
    "ISPY2_BIFLOW_CONTINUOUS_PAYLOAD_SCHEMA",
    "ISPY2_BIFLOW_LATENT_NORMALIZATION",
    "ISPY2_BIFLOW_LATENT_STATISTICS_SCHEMA",
    "ISPY2BiFlowLatentCache",
    "decode_ispy2_biflow_continuous",
    "prepare_ispy2_biflow_latent_cache",
]
