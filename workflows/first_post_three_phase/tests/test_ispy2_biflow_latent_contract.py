from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from mewm_ispy2.ispy2_biflow_latent_contract import (
    ISPY2_BIFLOW_CONTINUOUS_CACHE_SCHEMA,
    ISPY2_BIFLOW_CONTINUOUS_PAYLOAD_SCHEMA,
    ISPY2_BIFLOW_LATENT_NORMALIZATION,
    ISPY2_BIFLOW_LATENT_STATISTICS_SCHEMA,
    ISPY2BiFlowLatentCache,
    decode_ispy2_biflow_continuous,
)


SHAPE = (8, 24, 64, 64)


def _canonical_sha256(payload: object) -> str:
    import hashlib

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _cache(tmp_path: Path) -> ISPY2BiFlowLatentCache:
    root = tmp_path / "latents"
    visits = root / "visits"
    visits.mkdir(parents=True)
    train_ids = ["patient-a:T0", "patient-b:T1"]
    means = [float(value) for value in range(8)]
    stds = [float(value + 1) for value in range(8)]
    statistics = {
        "schema": ISPY2_BIFLOW_LATENT_STATISTICS_SCHEMA,
        "source_split": "train",
        "visit_selection": "unique_endpoint_visits",
        "visit_count": len(train_ids),
        "visit_ids_sha256": _canonical_sha256(train_ids),
        "element_count_per_channel": len(train_ids) * 24 * 64 * 64,
        "accumulator_dtype": "float64",
        "variance_estimator": "population_ddof0",
        "mean": means,
        "std": stds,
    }
    statistics["sha256"] = _canonical_sha256(statistics)
    identity = {
        "schema": ISPY2_BIFLOW_CONTINUOUS_CACHE_SCHEMA,
        "payload_schema": ISPY2_BIFLOW_CONTINUOUS_PAYLOAD_SCHEMA,
        "stored_representation": "continuous_prequantization_float16_v1",
        "normalization": ISPY2_BIFLOW_LATENT_NORMALIZATION,
        "latent_statistics": statistics,
        "source_cache_identity_sha256": "s" * 64,
        "visit_count": 2,
        "split_visit_counts": {"train": 2},
        "visit_ids": train_ids,
        "latent_shape_czyx": list(SHAPE),
        "latent_dtype": "float16",
        "vqgan_sha256": "v" * 64,
        "codebook_sha256": "c" * 64,
        "data_contract_sha256": "d" * 64,
    }
    (root / "cache_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    channel_mean = torch.tensor(means).reshape(8, 1, 1, 1)
    for visit_id, offset in zip(train_ids, (-1.0, 1.0), strict=True):
        torch.save(
            {
                "schema": ISPY2_BIFLOW_CONTINUOUS_PAYLOAD_SCHEMA,
                "visit_id": visit_id,
                "split": "train",
                "continuous_latent": (channel_mean + offset).expand(SHAPE).half(),
                "vqgan_sha256": "v" * 64,
                "codebook_sha256": "c" * 64,
                "data_contract_sha256": "d" * 64,
                "normalization": ISPY2_BIFLOW_LATENT_NORMALIZATION,
                "latent_statistics_sha256": statistics["sha256"],
            },
            visits / f"{visit_id.replace(':', '__')}.pt",
        )
    return ISPY2BiFlowLatentCache(root)


def test_channel_zscore_is_unclipped_and_exactly_invertible(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    continuous = torch.stack(
        [
            torch.full((24, 64, 64), 100.0 + channel)
            for channel in range(8)
        ]
    )

    normalized = cache.normalize(continuous)
    recovered = cache.denormalize(normalized)

    assert normalized.max() > 1.0
    torch.testing.assert_close(recovered, continuous, rtol=0.0, atol=1e-5)


def test_cached_train_visits_have_expected_channel_normalization(tmp_path: Path) -> None:
    cache = _cache(tmp_path)

    first = cache.load("patient-a:T0")
    second = cache.load("patient-b:T1")

    for channel in range(8):
        expected = 1.0 / float(channel + 1)
        torch.testing.assert_close(
            first[channel], torch.full_like(first[channel], -expected)
        )
        torch.testing.assert_close(
            second[channel], torch.full_like(second[channel], expected)
        )


class _Quantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observed: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.observed = value.detach().clone()
        return value + 1.0, None


class _VQGAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantizer = _Quantizer()

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :1]


def test_decode_quantizes_the_supplied_continuous_latent_without_clamp() -> None:
    model = _VQGAN()
    continuous = torch.arange(8.0).reshape(1, 1, 8, 1, 1, 1).expand(
        2, 1, 8, 24, 64, 64
    )

    decoded = decode_ispy2_biflow_continuous(model, continuous)

    assert model.quantizer.observed is not None
    torch.testing.assert_close(model.quantizer.observed, continuous[:, 0])
    torch.testing.assert_close(decoded, continuous[:, :, :1] + 1.0)
