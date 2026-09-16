from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .latent_statistics import sha256_file
from .latent_world_v4_config import (
    LatentWorldV4ExperimentConfig,
    load_latent_world_v4_config,
)


LATENT_WORLD_V5_CONFIG_SCHEMA = "mewm_ispy2_treatment_flow_world_model_config_v5"
LATENT_WORLD_V5_CHECKPOINT_SCHEMA = (
    "mewm_ispy2_treatment_flow_world_model_checkpoint_v5"
)
LATENT_WORLD_V5_ARCHITECTURE = (
    "fmbcmri_monai_controlnet_treatment_flow_v5_codebook"
)
V5_CODEBOOK_LATENT_CONTRACT = "vqgan_codebook_embedding_channel_zscore_v1"


def _mapping(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"latent-world v5 config {name} fields are invalid")
    return value


def _path(value: Any, name: str) -> Path:
    if type(value) is not str or not value.strip():
        raise ValueError(f"latent-world v5 config {name} path is invalid")
    return Path(value).expanduser().resolve()


def _sha256(value: Any, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"latent-world v5 config {name} SHA256 is invalid")
    return value


@dataclass(frozen=True)
class V5CodebookConfig:
    cache_dir: Path
    statistics_path: Path
    statistics_sha256: str | None
    latent_contract: str
    project_rollout_endpoints: bool
    straight_through: bool
    nearest_chunk_size: int


@dataclass(frozen=True)
class LatentWorldV5ExperimentConfig:
    path: Path
    v4_config_path: Path
    v4_config_sha256: str
    v4: LatentWorldV4ExperimentConfig
    codebook: V5CodebookConfig
    output_root: Path
    seed: int
    raw: dict[str, Any]

    def identity_payload(self) -> dict[str, Any]:
        if self.codebook.statistics_sha256 is None:
            raise ValueError("V5 resolved config requires codebook statistics SHA256")
        return {
            "schema": LATENT_WORLD_V5_CONFIG_SCHEMA,
            "architecture": LATENT_WORLD_V5_ARCHITECTURE,
            "v4_config_sha256": self.v4_config_sha256,
            "v4_identity_sha256": self.v4.identity_sha256(),
            "codebook": {
                "statistics_sha256": self.codebook.statistics_sha256,
                "latent_contract": self.codebook.latent_contract,
                "project_rollout_endpoints": self.codebook.project_rollout_endpoints,
                "straight_through": self.codebook.straight_through,
                "nearest_chunk_size": self.codebook.nearest_chunk_size,
            },
            "seed": self.seed,
        }

    def identity_sha256(self) -> str:
        encoded = json.dumps(
            self.identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def resolved_payload(self, statistics_sha256: str) -> dict[str, Any]:
        _sha256(statistics_sha256, "codebook.statistics")
        payload = json.loads(json.dumps(self.raw))
        payload["codebook"]["statistics_sha256"] = statistics_sha256
        return payload


def load_latent_world_v5_config(
    path: str | Path, *, require_statistics: bool = True
) -> LatentWorldV5ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    payload = _release_yaml(config_path.read_text(encoding="utf-8"))
    root = _mapping(
        payload,
        "root",
        {"schema_version", "v4_config", "codebook", "runtime"},
    )
    if root["schema_version"] != LATENT_WORLD_V5_CONFIG_SCHEMA:
        raise ValueError("latent-world v5 config schema is unsupported")

    v4_raw = _mapping(root["v4_config"], "v4_config", {"path", "sha256"})
    v4_path = _path(v4_raw["path"], "v4_config.path")
    v4_sha256 = _sha256(v4_raw["sha256"], "v4_config")
    assert isinstance(v4_sha256, str)
    if sha256_file(v4_path) != v4_sha256:
        raise ValueError("latent-world v5 V4 config SHA256 mismatch")
    v4 = load_latent_world_v4_config(v4_path)
    if v4.conditioning.mode != "id":
        raise ValueError("latent-world v5 currently supports ID conditioning only")

    codebook_raw = _mapping(
        root["codebook"],
        "codebook",
        {
            "cache_dir",
            "statistics_path",
            "statistics_sha256",
            "latent_contract",
            "project_rollout_endpoints",
            "straight_through",
            "nearest_chunk_size",
        },
    )
    if require_statistics and codebook_raw["statistics_sha256"] is None:
        raise ValueError("latent-world v5 config is an unresolved preparation spec")
    statistics_sha256 = _sha256(
        codebook_raw["statistics_sha256"],
        "codebook.statistics",
        optional=not require_statistics,
    )
    if codebook_raw["latent_contract"] != V5_CODEBOOK_LATENT_CONTRACT:
        raise ValueError("latent-world v5 codebook latent contract is unsupported")
    if codebook_raw["project_rollout_endpoints"] is not True:
        raise ValueError("latent-world v5 must project rollout endpoints")
    if codebook_raw["straight_through"] is not True:
        raise ValueError("latent-world v5 must use straight-through projection")
    chunk_size = codebook_raw["nearest_chunk_size"]
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("latent-world v5 nearest chunk size must be positive")
    codebook = V5CodebookConfig(
        cache_dir=_path(codebook_raw["cache_dir"], "codebook.cache_dir"),
        statistics_path=_path(
            codebook_raw["statistics_path"], "codebook.statistics_path"
        ),
        statistics_sha256=statistics_sha256,
        latent_contract=V5_CODEBOOK_LATENT_CONTRACT,
        project_rollout_endpoints=True,
        straight_through=True,
        nearest_chunk_size=chunk_size,
    )

    runtime = _mapping(root["runtime"], "runtime", {"output_root", "seed"})
    seed = runtime["seed"]
    if type(seed) is not int or seed not in {2026, 2027, 2028}:
        raise ValueError("latent-world v5 registered seed is invalid")
    return LatentWorldV5ExperimentConfig(
        path=config_path,
        v4_config_path=v4_path,
        v4_config_sha256=v4_sha256,
        v4=v4,
        codebook=codebook,
        output_root=_path(runtime["output_root"], "runtime.output_root"),
        seed=seed,
        raw=root,
    )


__all__ = [
    "LATENT_WORLD_V5_ARCHITECTURE",
    "LATENT_WORLD_V5_CHECKPOINT_SCHEMA",
    "LATENT_WORLD_V5_CONFIG_SCHEMA",
    "LatentWorldV5ExperimentConfig",
    "V5_CODEBOOK_LATENT_CONTRACT",
    "V5CodebookConfig",
    "load_latent_world_v5_config",
]
