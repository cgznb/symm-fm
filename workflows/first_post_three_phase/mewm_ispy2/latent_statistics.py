from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


LATENT_CHANNEL_STATISTICS_SCHEMA = "mewm_ispy2_latent_channel_statistics_v1"
LATENT_CHANNEL_COUNT = 8
_HEX_DIGITS = frozenset("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LatentChannelStatistics:
    count: tuple[int, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    visit_count: int
    visit_ids_sha256: str
    vqgan_sha256: str
    data_contract_sha256: str
    latent_shape_czyx: tuple[int, int, int, int]
    schema: str = LATENT_CHANNEL_STATISTICS_SCHEMA
    population: str = "unique_train_source_target_endpoints_all_latent_voxels"
    algorithm: str = "float64_parallel_welford_population_std_v1"

    def __post_init__(self) -> None:
        if self.schema != LATENT_CHANNEL_STATISTICS_SCHEMA:
            raise ValueError("latent statistics schema is unsupported")
        if self.population != "unique_train_source_target_endpoints_all_latent_voxels":
            raise ValueError("latent statistics population is unsupported")
        if self.algorithm != "float64_parallel_welford_population_std_v1":
            raise ValueError("latent statistics algorithm is unsupported")
        for name in ("count", "mean", "std"):
            value = getattr(self, name)
            if type(value) is not tuple or len(value) != LATENT_CHANNEL_COUNT:
                raise ValueError(f"latent statistics {name} must contain eight values")
        if any(type(value) is not int or value <= 0 for value in self.count):
            raise ValueError("latent statistics counts must be positive integers")
        if len(set(self.count)) != 1:
            raise ValueError("latent statistics channel counts must match")
        if any(type(value) is not float or not math.isfinite(value) for value in self.mean):
            raise ValueError("latent statistics means must be finite floats")
        if any(
            type(value) is not float or not math.isfinite(value) or value <= 0.0
            for value in self.std
        ):
            raise ValueError("latent statistics standard deviations must be positive")
        if type(self.visit_count) is not int or self.visit_count <= 0:
            raise ValueError("latent statistics visit count must be positive")
        for name in ("visit_ids_sha256", "vqgan_sha256", "data_contract_sha256"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"latent statistics {name} is invalid")
        if (
            type(self.latent_shape_czyx) is not tuple
            or len(self.latent_shape_czyx) != 4
            or self.latent_shape_czyx[0] != LATENT_CHANNEL_COUNT
            or any(type(value) is not int or value <= 0 for value in self.latent_shape_czyx)
        ):
            raise ValueError("latent statistics shape must be a positive 8-channel shape")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "population": self.population,
            "algorithm": self.algorithm,
            "count": list(self.count),
            "mean": list(self.mean),
            "std": list(self.std),
            "visit_count": self.visit_count,
            "visit_ids_sha256": self.visit_ids_sha256,
            "vqgan_sha256": self.vqgan_sha256,
            "data_contract_sha256": self.data_contract_sha256,
            "latent_shape_czyx": list(self.latent_shape_czyx),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> LatentChannelStatistics:
        expected = {
            "schema",
            "population",
            "algorithm",
            "count",
            "mean",
            "std",
            "visit_count",
            "visit_ids_sha256",
            "vqgan_sha256",
            "data_contract_sha256",
            "latent_shape_czyx",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("latent statistics payload fields are invalid")
        return cls(
            schema=payload["schema"],
            population=payload["population"],
            algorithm=payload["algorithm"],
            count=tuple(payload["count"]),
            mean=tuple(payload["mean"]),
            std=tuple(payload["std"]),
            visit_count=payload["visit_count"],
            visit_ids_sha256=payload["visit_ids_sha256"],
            vqgan_sha256=payload["vqgan_sha256"],
            data_contract_sha256=payload["data_contract_sha256"],
            latent_shape_czyx=tuple(payload["latent_shape_czyx"]),
        )

    def tensors(
        self, *, device: torch.device | str, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, LATENT_CHANNEL_COUNT, 1, 1, 1)
        mean = torch.tensor(self.mean, device=device, dtype=dtype).reshape(shape)
        std = torch.tensor(self.std, device=device, dtype=dtype).reshape(shape)
        return mean, std


class StreamingChannelMoments:
    def __init__(self, channels: int = LATENT_CHANNEL_COUNT) -> None:
        if type(channels) is not int or channels <= 0:
            raise ValueError("streaming latent channel count must be positive")
        self.channels = channels
        self.count = 0
        self.mean = torch.zeros(channels, dtype=torch.float64)
        self.m2 = torch.zeros(channels, dtype=torch.float64)

    def update(self, latent: torch.Tensor) -> None:
        if (
            not isinstance(latent, torch.Tensor)
            or latent.ndim != 5
            or latent.shape[1] != self.channels
            or not latent.is_floating_point()
        ):
            raise ValueError("streaming latent must be [B,C,D,H,W] floating data")
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("streaming latent must be finite")
        values = latent.detach().to(device="cpu", dtype=torch.float64)
        values = values.transpose(0, 1).reshape(self.channels, -1)
        batch_count = values.shape[1]
        batch_mean = values.mean(dim=1)
        centered = values - batch_mean[:, None]
        batch_m2 = centered.square().sum(dim=1)
        if self.count == 0:
            self.count = batch_count
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean.add_(delta * (batch_count / total))
        self.m2.add_(batch_m2 + delta.square() * self.count * batch_count / total)
        self.count = total

    def finalize(
        self,
        *,
        visit_ids: Sequence[str],
        vqgan_sha256: str,
        data_contract_sha256: str,
        latent_shape_czyx: Sequence[int],
    ) -> LatentChannelStatistics:
        identifiers = tuple(visit_ids)
        if not identifiers or identifiers != tuple(sorted(set(identifiers))):
            raise ValueError("latent statistics visit IDs must be sorted and unique")
        if self.count <= 0:
            raise ValueError("cannot finalize empty latent statistics")
        variance = self.m2 / self.count
        std = variance.sqrt()
        visit_digest = hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()
        return LatentChannelStatistics(
            count=(self.count,) * self.channels,
            mean=tuple(float(value) for value in self.mean.tolist()),
            std=tuple(float(value) for value in std.tolist()),
            visit_count=len(identifiers),
            visit_ids_sha256=visit_digest,
            vqgan_sha256=vqgan_sha256,
            data_contract_sha256=data_contract_sha256,
            latent_shape_czyx=tuple(latent_shape_czyx),
        )


def write_latent_statistics(
    path: str | Path, statistics: LatentChannelStatistics
) -> str:
    if type(statistics) is not LatentChannelStatistics:
        raise TypeError("statistics must be exact LatentChannelStatistics")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(
                statistics.to_payload(),
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(target)


def load_latent_statistics(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_vqgan_sha256: str,
    expected_data_contract_sha256: str | None,
) -> LatentChannelStatistics:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("latent statistics artifact is missing or unsafe")
    if not _is_sha256(expected_sha256) or sha256_file(source) != expected_sha256:
        raise ValueError("latent statistics artifact SHA256 does not match")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("latent statistics artifact is unreadable") from error
    statistics = LatentChannelStatistics.from_payload(payload)
    if statistics.vqgan_sha256 != expected_vqgan_sha256:
        raise ValueError("latent statistics VQGAN identity does not match")
    if (
        expected_data_contract_sha256 is not None
        and statistics.data_contract_sha256 != expected_data_contract_sha256
    ):
        raise ValueError("latent statistics data identity does not match")
    return statistics


__all__ = [
    "LATENT_CHANNEL_STATISTICS_SCHEMA",
    "LatentChannelStatistics",
    "StreamingChannelMoments",
    "load_latent_statistics",
    "sha256_file",
    "write_latent_statistics",
]
