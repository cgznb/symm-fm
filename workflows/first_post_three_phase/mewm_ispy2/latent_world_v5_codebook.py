from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .latent_statistics import sha256_file
from .latent_world_v5_config import V5_CODEBOOK_LATENT_CONTRACT


V5_CODEBOOK_STATISTICS_SCHEMA = "mewm_ispy2_codebook_latent_statistics_v1"
V5_CODEBOOK_CACHE_SCHEMA = "mewm_ispy2_codebook_visit_cache_v1"
V5_CODEBOOK_INDEX_DTYPE = torch.int16
V5_CODEBOOK_LATENT_SHAPE = (8, 24, 64, 64)
V5_CODEBOOK_INDEX_SHAPE = (24, 64, 64)
V5_CODEBOOK_MASK_SHAPE = (1, 24, 64, 64)
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def codebook_sha256(embeddings: torch.Tensor) -> str:
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or embeddings.shape[1] != V5_CODEBOOK_LATENT_SHAPE[0]
        or not embeddings.is_floating_point()
        or not bool(torch.isfinite(embeddings).all())
    ):
        raise ValueError("V5 codebook embeddings must be finite [N,8] floating data")
    value = embeddings.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(b"mewm_ispy2_v5_codebook_float32_n8_v1\n")
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class V5CodebookStatistics:
    schema: str
    latent_contract: str
    algorithm: str
    channel_mean: tuple[float, ...]
    channel_std: tuple[float, ...]
    voxel_count_per_channel: int
    visit_count: int
    visit_ids_sha256: str
    vqgan_sha256: str
    codebook_sha256: str
    data_contract_sha256: str

    def __post_init__(self) -> None:
        if self.schema != V5_CODEBOOK_STATISTICS_SCHEMA:
            raise ValueError("V5 codebook statistics schema is invalid")
        if self.latent_contract != V5_CODEBOOK_LATENT_CONTRACT:
            raise ValueError("V5 codebook statistics latent contract is invalid")
        if self.algorithm != "code_index_histogram_float64_population_std_v1":
            raise ValueError("V5 codebook statistics algorithm is invalid")
        if len(self.channel_mean) != 8 or len(self.channel_std) != 8:
            raise ValueError("V5 codebook statistics must contain eight channels")
        if any(not math.isfinite(value) for value in self.channel_mean):
            raise ValueError("V5 codebook statistics means must be finite")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.channel_std):
            raise ValueError("V5 codebook statistics standard deviations must be positive")
        if self.voxel_count_per_channel <= 0 or self.visit_count <= 0:
            raise ValueError("V5 codebook statistics counts must be positive")
        for field in (
            "visit_ids_sha256",
            "vqgan_sha256",
            "codebook_sha256",
            "data_contract_sha256",
        ):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"V5 codebook statistics {field} is invalid")

    def payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["channel_mean"] = list(self.channel_mean)
        payload["channel_std"] = list(self.channel_std)
        return payload

    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def tensors(
        self, *, device: torch.device | str, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, 8, 1, 1, 1)
        mean = torch.tensor(self.channel_mean, device=device, dtype=dtype).reshape(shape)
        std = torch.tensor(self.channel_std, device=device, dtype=dtype).reshape(shape)
        return mean, std

    @classmethod
    def from_payload(cls, payload: Any) -> "V5CodebookStatistics":
        fields = {
            "schema",
            "latent_contract",
            "algorithm",
            "channel_mean",
            "channel_std",
            "voxel_count_per_channel",
            "visit_count",
            "visit_ids_sha256",
            "vqgan_sha256",
            "codebook_sha256",
            "data_contract_sha256",
        }
        if type(payload) is not dict or set(payload) != fields:
            raise ValueError("V5 codebook statistics fields are invalid")
        return cls(
            schema=payload["schema"],
            latent_contract=payload["latent_contract"],
            algorithm=payload["algorithm"],
            channel_mean=tuple(float(value) for value in payload["channel_mean"]),
            channel_std=tuple(float(value) for value in payload["channel_std"]),
            voxel_count_per_channel=int(payload["voxel_count_per_channel"]),
            visit_count=int(payload["visit_count"]),
            visit_ids_sha256=payload["visit_ids_sha256"],
            vqgan_sha256=payload["vqgan_sha256"],
            codebook_sha256=payload["codebook_sha256"],
            data_contract_sha256=payload["data_contract_sha256"],
        )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_v5_codebook_statistics(
    path: str | Path, statistics: V5CodebookStatistics
) -> str:
    if type(statistics) is not V5CodebookStatistics:
        raise TypeError("statistics must be exact V5CodebookStatistics")
    target = Path(path).resolve()
    _atomic_json(target, statistics.payload())
    return sha256_file(target)


def load_v5_codebook_statistics(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_vqgan_sha256: str,
    expected_codebook_sha256: str,
    expected_data_contract_sha256: str,
) -> V5CodebookStatistics:
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError("V5 codebook statistics artifact is missing or unsafe")
    if not _is_sha256(expected_sha256) or sha256_file(source) != expected_sha256:
        raise ValueError("V5 codebook statistics SHA256 mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("V5 codebook statistics artifact is unreadable") from error
    statistics = V5CodebookStatistics.from_payload(payload)
    if (
        statistics.vqgan_sha256 != expected_vqgan_sha256
        or statistics.codebook_sha256 != expected_codebook_sha256
        or statistics.data_contract_sha256 != expected_data_contract_sha256
    ):
        raise ValueError("V5 codebook statistics identity mismatch")
    return statistics


def compute_v5_codebook_statistics(
    visit_ids: Sequence[str],
    *,
    index_loader: Callable[[str], torch.Tensor],
    embeddings: torch.Tensor,
    vqgan_sha256: str,
    data_contract_sha256: str,
) -> V5CodebookStatistics:
    identifiers = tuple(visit_ids)
    if not identifiers or identifiers != tuple(sorted(set(identifiers))):
        raise ValueError("V5 statistics visit IDs must be sorted and unique")
    if not callable(index_loader):
        raise TypeError("V5 statistics index loader must be callable")
    table = embeddings.detach().to(device="cpu", dtype=torch.float64).contiguous()
    table_sha256 = codebook_sha256(embeddings)
    counts = torch.zeros(table.shape[0], dtype=torch.int64)
    for visit_id in identifiers:
        indices = index_loader(visit_id)
        if (
            not isinstance(indices, torch.Tensor)
            or indices.dtype != V5_CODEBOOK_INDEX_DTYPE
            or tuple(indices.shape) != V5_CODEBOOK_INDEX_SHAPE
        ):
            raise ValueError(f"V5 code indices are invalid for visit {visit_id}")
        values = indices.to(dtype=torch.long).flatten()
        if int(values.min()) < 0 or int(values.max()) >= table.shape[0]:
            raise ValueError(f"V5 code indices are out of range for visit {visit_id}")
        counts += torch.bincount(values, minlength=table.shape[0])
    total = int(counts.sum())
    expected = len(identifiers) * math.prod(V5_CODEBOOK_INDEX_SHAPE)
    if total != expected:
        raise RuntimeError("V5 codebook statistics voxel count changed")
    weights = counts.to(dtype=torch.float64)
    mean = weights @ table / total
    second = weights @ table.square() / total
    std = (second - mean.square()).clamp_min(1e-12).sqrt()
    visit_digest = hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()
    return V5CodebookStatistics(
        schema=V5_CODEBOOK_STATISTICS_SCHEMA,
        latent_contract=V5_CODEBOOK_LATENT_CONTRACT,
        algorithm="code_index_histogram_float64_population_std_v1",
        channel_mean=tuple(float(value) for value in mean),
        channel_std=tuple(float(value) for value in std),
        voxel_count_per_channel=total,
        visit_count=len(identifiers),
        visit_ids_sha256=visit_digest,
        vqgan_sha256=vqgan_sha256,
        codebook_sha256=table_sha256,
        data_contract_sha256=data_contract_sha256,
    )


def v5_codebook_cache_identity(
    *,
    data_contract_sha256: str,
    bundle_json_sha256: str,
    vqgan_sha256: str,
    codebook_sha256_value: str,
    statistics_sha256: str,
) -> dict[str, Any]:
    values = {
        "data_contract_sha256": data_contract_sha256,
        "bundle_json_sha256": bundle_json_sha256,
        "vqgan_sha256": vqgan_sha256,
        "codebook_sha256": codebook_sha256_value,
        "statistics_sha256": statistics_sha256,
    }
    if any(not _is_sha256(value) for value in values.values()):
        raise ValueError("V5 cache identity SHA256 field is invalid")
    return {
        "schema": V5_CODEBOOK_CACHE_SCHEMA,
        "latent_contract": V5_CODEBOOK_LATENT_CONTRACT,
        **values,
    }


def _validate_index_payload(payload: Any, visit_id: str, *, n_codes: int) -> bool:
    if type(payload) is not dict or set(payload) != {
        "schema",
        "visit_id",
        "code_indices",
        "mask",
    }:
        return False
    indices, mask = payload["code_indices"], payload["mask"]
    return bool(
        payload["schema"] == V5_CODEBOOK_CACHE_SCHEMA
        and payload["visit_id"] == visit_id
        and isinstance(indices, torch.Tensor)
        and indices.dtype == V5_CODEBOOK_INDEX_DTYPE
        and tuple(indices.shape) == V5_CODEBOOK_INDEX_SHAPE
        and int(indices.min()) >= 0
        and int(indices.max()) < n_codes
        and isinstance(mask, torch.Tensor)
        and mask.dtype == torch.uint8
        and tuple(mask.shape) == V5_CODEBOOK_MASK_SHAPE
        and bool(((mask == 0) | (mask == 1)).all())
    )


class V5CodebookVisitCache:
    def __init__(
        self,
        root: str | Path,
        *,
        expected_identity: dict[str, Any],
        embeddings: torch.Tensor,
        statistics: V5CodebookStatistics,
    ) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError("V5 codebook cache manifest is unavailable") from None
        if (
            type(manifest) is not dict
            or manifest.get("schema") != V5_CODEBOOK_CACHE_SCHEMA
            or manifest.get("identity") != expected_identity
        ):
            raise ValueError("V5 codebook cache identity mismatch")
        records = manifest.get("records")
        if type(records) is not dict or not records:
            raise ValueError("V5 codebook cache records are invalid")
        if statistics.codebook_sha256 != codebook_sha256(embeddings):
            raise ValueError("V5 cache codebook does not match statistics")
        table = embeddings.detach().to(device="cpu", dtype=torch.float32)
        mean = torch.tensor(statistics.channel_mean, dtype=torch.float32)
        std = torch.tensor(statistics.channel_std, dtype=torch.float32)
        self.normalized_codebook = ((table - mean) / std).contiguous()
        self.n_codes = int(table.shape[0])
        self.manifest = manifest
        self.records = records

    @staticmethod
    def write_manifest(
        root: str | Path, *, identity: dict[str, Any], records: dict[str, Any]
    ) -> Path:
        target = Path(root).resolve() / "manifest.json"
        _atomic_json(
            target,
            {
                "schema": V5_CODEBOOK_CACHE_SCHEMA,
                "identity": identity,
                "index_shape_zyx": list(V5_CODEBOOK_INDEX_SHAPE),
                "index_dtype": "int16",
                "latent_shape_czyx": list(V5_CODEBOOK_LATENT_SHAPE),
                "mask_shape_czyx": list(V5_CODEBOOK_MASK_SHAPE),
                "records": records,
            },
        )
        return target

    def _descriptor_path(self, visit_id: str) -> Path:
        descriptor = self.records.get(visit_id)
        if type(descriptor) is not dict or set(descriptor) != {"path", "sha256"}:
            raise ValueError(f"V5 codebook cache is missing visit {visit_id}")
        path = self.root / descriptor["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or not _is_sha256(descriptor["sha256"])
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ValueError(f"V5 codebook cache visit identity mismatch: {visit_id}")
        return path

    @lru_cache(maxsize=16)
    def _load(self, visit_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self._descriptor_path(visit_id)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError, EOFError):
            raise ValueError(f"V5 codebook cache visit is unreadable: {visit_id}") from None
        if not _validate_index_payload(payload, visit_id, n_codes=self.n_codes):
            raise ValueError(f"V5 codebook cache visit payload is invalid: {visit_id}")
        indices = payload["code_indices"].contiguous()
        latent = F.embedding(indices.long(), self.normalized_codebook).movedim(-1, 0)
        return latent.contiguous(), payload["mask"].contiguous(), indices

    def load(self, visit_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        latent, mask, _ = self._load(visit_id)
        return latent.clone(), mask.clone()

    def load_indices(self, visit_id: str) -> torch.Tensor:
        _, _, indices = self._load(visit_id)
        return indices.clone()


class NormalizedCodebookProjector(nn.Module):
    def __init__(
        self,
        embeddings: torch.Tensor,
        statistics: V5CodebookStatistics,
        *,
        chunk_size: int,
        straight_through: bool = True,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("V5 codebook projector chunk size must be positive")
        table = embeddings.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if statistics.codebook_sha256 != codebook_sha256(table):
            raise ValueError("V5 codebook projector statistics identity mismatch")
        mean = torch.tensor(statistics.channel_mean, dtype=torch.float32)
        std = torch.tensor(statistics.channel_std, dtype=torch.float32)
        self.register_buffer("embeddings", table)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.chunk_size = int(chunk_size)
        self.straight_through = bool(straight_through)

    def forward(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            normalized.ndim != 5
            or normalized.shape[1] != 8
            or not normalized.is_floating_point()
        ):
            raise ValueError("V5 projector input must be [B,8,D,H,W] floating data")
        original_dtype = normalized.dtype
        value = normalized.float()
        mean = self.mean.reshape(1, 8, 1, 1, 1)
        std = self.std.reshape(1, 8, 1, 1, 1)
        raw_flat = (value * std + mean).movedim(1, -1).reshape(-1, 8)
        table = self.embeddings.float()
        table_norm = table.square().sum(dim=1)
        pieces: list[torch.Tensor] = []
        for chunk in raw_flat.split(self.chunk_size):
            distances = (
                chunk.square().sum(dim=1, keepdim=True)
                - 2.0 * chunk @ table.t()
                + table_norm[None]
            )
            pieces.append(distances.argmin(dim=1))
        flat_indices = torch.cat(pieces)
        quantized_raw = F.embedding(flat_indices, table)
        quantized = (quantized_raw - self.mean) / self.std
        quantized = quantized.reshape(
            normalized.shape[0], *normalized.shape[2:], 8
        ).movedim(-1, 1)
        quantized = quantized.to(dtype=original_dtype)
        if self.straight_through:
            quantized = normalized + (quantized - normalized).detach()
        indices = flat_indices.reshape(normalized.shape[0], *normalized.shape[2:])
        return quantized, indices


__all__ = [
    "NormalizedCodebookProjector",
    "V5_CODEBOOK_CACHE_SCHEMA",
    "V5_CODEBOOK_INDEX_DTYPE",
    "V5_CODEBOOK_INDEX_SHAPE",
    "V5_CODEBOOK_LATENT_SHAPE",
    "V5_CODEBOOK_MASK_SHAPE",
    "V5_CODEBOOK_STATISTICS_SCHEMA",
    "V5CodebookStatistics",
    "V5CodebookVisitCache",
    "codebook_sha256",
    "compute_v5_codebook_statistics",
    "load_v5_codebook_statistics",
    "v5_codebook_cache_identity",
    "write_v5_codebook_statistics",
]
