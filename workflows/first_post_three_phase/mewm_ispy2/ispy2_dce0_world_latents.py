from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch

from .backend import REGISTERED_STRICT_A_BUNDLE_SCHEMA, load_transition_records
from .cache import RegisteredStrictAROICache
from .ispy2_dce0_world_config import load_ispy2_dce0_world_config
from .ispy2_dce0_world_data import build_ispy2_dce0_world_pairs
from .latent_statistics import sha256_file
from .latent_world_v5_codebook import codebook_sha256
from .vqgan import REGISTERED_VQGAN_NUMERIC_CONTRACT
from .workflows import load_mri_vqgan


ISPY2_DCE0_CONTINUOUS_CACHE_SCHEMA = "mewm_ispy2_dce0_continuous_latents_v1"
ISPY2_DCE0_CONTINUOUS_PAYLOAD_SCHEMA = (
    "mewm_ispy2_dce0_continuous_latent_payload_v1"
)
ISPY2_DCE0_CONTINUOUS_NORMALIZATION = "continuous_codebook_minmax_v1"
ISPY2_DCE0_LATENT_SHAPE = (8, 24, 64, 64)


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


def _visit_filename(visit_id: str) -> str:
    if not visit_id or "/" in visit_id or "\\" in visit_id:
        raise ValueError("I-SPY2 latent visit ID is unsafe")
    return f"{visit_id.replace(':', '__')}.pt"


def _payload_path(root: Path, visit_id: str) -> Path:
    return root / "visits" / _visit_filename(visit_id)


def _latent_root(config: Any) -> Path:
    for owner_name, field in (
        ("data", "continuous_root"),
        ("data", "latent_cache_dir"),
        ("latents", "cache_dir"),
    ):
        owner = getattr(config, owner_name, None)
        value = getattr(owner, field, None) if owner is not None else None
        if value is not None:
            return Path(value).expanduser().resolve()
    raise ValueError("I-SPY2 DCE0 continuous cache root is not configured")


def _vqgan_descriptor(config: Any) -> tuple[Path, str]:
    candidates = (
        getattr(getattr(config, "data", None), "vqgan_checkpoint", None),
        getattr(getattr(config, "vqgan", None), "checkpoint", None),
        getattr(getattr(config, "latents", None), "vqgan_checkpoint", None),
    )
    digest_candidates = (
        getattr(getattr(config, "data", None), "vqgan_sha256", None),
        getattr(getattr(config, "vqgan", None), "checkpoint_sha256", None),
        getattr(getattr(config, "latents", None), "vqgan_sha256", None),
    )
    checkpoint = next((value for value in candidates if value is not None), None)
    digest = next((value for value in digest_candidates if value is not None), None)
    if checkpoint is None or not isinstance(digest, str):
        raise ValueError("I-SPY2 DCE0 VQGAN descriptor is incomplete")
    return Path(checkpoint).expanduser().resolve(), digest


def ispy2_dce0_continuous_cache_identity(
    config: Any,
    *,
    loaded: Any,
    visit_ids: Sequence[str],
    vqgan_checkpoint: Path,
    vqgan_sha256: str,
    embeddings: torch.Tensor,
    roi_cache: RegisteredStrictAROICache,
    split_counts: dict[str, int],
) -> dict[str, Any]:
    identifiers = tuple(sorted(set(str(value) for value in visit_ids)))
    return {
        "schema": ISPY2_DCE0_CONTINUOUS_CACHE_SCHEMA,
        "visit_count": len(identifiers),
        "split_visit_counts": dict(sorted(split_counts.items())),
        "visit_ids": list(identifiers),
        "bundle_schema": loaded.bundle_schema_version,
        "bundle_contract_sha256": loaded.bundle_contract_sha256,
        "data_contract_sha256": loaded.data_contract_sha256,
        "bundle_json_sha256": sha256_file(config.data.bundle_json),
        "phase_manifest_sha256": sha256_file(config.data.phase_manifest_csv),
        "roi_cache_contract": dict(roi_cache.cache_contract),
        "input_shape_zyx": list(roi_cache.output_shape_zyx),
        "latent_shape_czyx": list(ISPY2_DCE0_LATENT_SHAPE),
        "latent_dtype": "float16",
        "vqgan_checkpoint": str(vqgan_checkpoint),
        "vqgan_sha256": vqgan_sha256,
        "codebook_sha256": codebook_sha256(embeddings),
        "codebook_min": config.data.codebook_min,
        "codebook_max": config.data.codebook_max,
        "numeric_contract": REGISTERED_VQGAN_NUMERIC_CONTRACT,
        "normalization": ISPY2_DCE0_CONTINUOUS_NORMALIZATION,
        "encoding": "continuous_without_quantizer",
    }


def _valid_existing(
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
    return bool(
        isinstance(payload, dict)
        and payload.get("schema") == ISPY2_DCE0_CONTINUOUS_PAYLOAD_SCHEMA
        and payload.get("visit_id") == visit_id
        and payload.get("split") == split
        and payload.get("vqgan_sha256") == identity["vqgan_sha256"]
        and payload.get("codebook_sha256") == identity["codebook_sha256"]
        and payload.get("data_contract_sha256") == identity["data_contract_sha256"]
        and payload.get("normalization") == ISPY2_DCE0_CONTINUOUS_NORMALIZATION
        and isinstance(latent, torch.Tensor)
        and latent.dtype == torch.float16
        and tuple(latent.shape) == ISPY2_DCE0_LATENT_SHAPE
        and bool(torch.isfinite(latent.float()).all())
    )


class ISPY2DCE0ContinuousLatentCache:
    def __init__(
        self,
        root: str | Path,
        *,
        expected_identity: dict[str, Any] | None = None,
    ) -> None:
        root_path = Path(root).expanduser()
        if root_path.is_symlink() or not root_path.is_dir():
            raise ValueError("I-SPY2 DCE0 continuous cache is missing or unsafe")
        self.root = root_path.resolve()
        identity_path = self.root / "cache_identity.json"
        if identity_path.is_symlink() or not identity_path.is_file():
            raise ValueError("I-SPY2 DCE0 continuous cache identity is missing")
        try:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError("I-SPY2 DCE0 continuous cache identity is unreadable") from None
        if (
            not isinstance(identity, dict)
            or identity.get("schema") != ISPY2_DCE0_CONTINUOUS_CACHE_SCHEMA
            or identity.get("latent_shape_czyx") != list(ISPY2_DCE0_LATENT_SHAPE)
            or identity.get("latent_dtype") != "float16"
            or identity.get("normalization") != ISPY2_DCE0_CONTINUOUS_NORMALIZATION
        ):
            raise ValueError("I-SPY2 DCE0 continuous cache identity is invalid")
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("I-SPY2 DCE0 continuous cache identity mismatch")
        minimum = float(identity.get("codebook_min", float("nan")))
        maximum = float(identity.get("codebook_max", float("nan")))
        if not torch.isfinite(torch.tensor([minimum, maximum])).all() or minimum >= maximum:
            raise ValueError("I-SPY2 DCE0 codebook range is invalid")
        self.identity = identity
        self.codebook_min = minimum
        self.codebook_max = maximum
        self.visit_ids = frozenset(str(value) for value in identity.get("visit_ids", ()))
        if len(self.visit_ids) != int(identity.get("visit_count", -1)):
            raise ValueError("I-SPY2 DCE0 cache visit inventory is invalid")

    def load(self, visit_id: str) -> torch.Tensor:
        if visit_id not in self.visit_ids:
            raise ValueError(f"I-SPY2 DCE0 latent visit is not registered: {visit_id}")
        path = _payload_path(self.root, visit_id)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"I-SPY2 DCE0 latent is missing: {visit_id}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError, EOFError):
            raise ValueError(f"I-SPY2 DCE0 latent is unreadable: {visit_id}") from None
        split = payload.get("split") if isinstance(payload, dict) else None
        if not isinstance(split, str) or not _valid_existing(
            path, visit_id=visit_id, split=split, identity=self.identity
        ):
            raise ValueError(f"I-SPY2 DCE0 latent payload is invalid: {visit_id}")
        value = payload["continuous_latent"].float()
        normalized = 2.0 * (value - self.codebook_min) / (
            self.codebook_max - self.codebook_min
        ) - 1.0
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError(f"I-SPY2 DCE0 normalized latent is invalid: {visit_id}")
        return normalized.contiguous()

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape[-4:]) != ISPY2_DCE0_LATENT_SHAPE:
            raise ValueError("I-SPY2 normalized latent shape is invalid")
        return (
            (value + 1.0)
            * 0.5
            * (self.codebook_max - self.codebook_min)
            + self.codebook_min
        )


@torch.no_grad()
def prepare_ispy2_dce0_world_latents(
    config_path: str | Path,
    *,
    vqgan_checkpoint: str | Path | None = None,
    device: str = "cuda",
) -> Path:
    config = load_ispy2_dce0_world_config(config_path)
    configured_checkpoint, expected_sha256 = _vqgan_descriptor(config)
    checkpoint = Path(vqgan_checkpoint or configured_checkpoint).expanduser().resolve()
    if checkpoint != configured_checkpoint or sha256_file(checkpoint) != expected_sha256:
        raise ValueError("I-SPY2 DCE0 VQGAN checkpoint does not match config")
    loaded = load_transition_records(
        config.data.bundle_json,
        config.data.phase_manifest_csv,
        backend=config.data.backend,
    )
    if loaded.bundle_schema_version != REGISTERED_STRICT_A_BUNDLE_SCHEMA:
        raise ValueError("I-SPY2 DCE0 world requires registered Strict-A data")
    pairs, _ = build_ispy2_dce0_world_pairs(
        config.data.bundle_json, config.data.phase_manifest_csv
    )
    split_by_visit: dict[str, str] = {}
    for pair in pairs:
        for visit_id in (pair.source_visit_id, pair.target_visit_id):
            previous = split_by_visit.setdefault(visit_id, pair.split)
            if previous != pair.split:
                raise ValueError("I-SPY2 DCE0 endpoint visit crosses data splits")
    visit_ids = tuple(sorted(split_by_visit))
    roi_cache = RegisteredStrictAROICache(
        config.data.roi_cache_dir,
        bundle_json=config.data.bundle_json,
        output_shape_zyx=config.data.output_shape_zyx,
    )
    model = load_mri_vqgan(
        checkpoint,
        expected_numeric_contract=REGISTERED_VQGAN_NUMERIC_CONTRACT,
    )
    embeddings = model.quantizer.embeddings.detach().cpu().float().contiguous()
    observed_min = float(embeddings.min())
    observed_max = float(embeddings.max())
    if not observed_min < observed_max:
        raise ValueError("I-SPY2 DCE0 VQGAN codebook range is degenerate")
    if not (
        math.isclose(observed_min, config.data.codebook_min, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(observed_max, config.data.codebook_max, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError("I-SPY2 DCE0 configured codebook range does not match VQGAN")
    identity = ispy2_dce0_continuous_cache_identity(
        config,
        loaded=loaded,
        visit_ids=visit_ids,
        vqgan_checkpoint=checkpoint,
        vqgan_sha256=expected_sha256,
        embeddings=embeddings,
        roi_cache=roi_cache,
        split_counts={
            split: sum(value == split for value in split_by_visit.values())
            for split in sorted(set(split_by_visit.values()))
        },
    )
    target = torch.device(device)
    model.to(target).eval().requires_grad_(False)
    root = _latent_root(config)
    built = reused = 0
    for index, visit_id in enumerate(visit_ids, start=1):
        split = split_by_visit[visit_id]
        path = _payload_path(root, visit_id)
        if _valid_existing(path, visit_id=visit_id, split=split, identity=identity):
            reused += 1
        else:
            prepared = roi_cache.load(loaded.visits[visit_id])
            image = prepared.image.unsqueeze(0).to(target, dtype=torch.float32)
            continuous = model.encode_continuous(image)
            if tuple(continuous.shape) != (1, *ISPY2_DCE0_LATENT_SHAPE):
                raise RuntimeError("I-SPY2 DCE0 continuous VQ latent shape changed")
            if not bool(torch.isfinite(continuous).all()):
                raise ValueError(f"I-SPY2 DCE0 latent is non-finite: {visit_id}")
            _atomic_torch_save(
                path,
                {
                    "schema": ISPY2_DCE0_CONTINUOUS_PAYLOAD_SCHEMA,
                    "visit_id": visit_id,
                    "split": split,
                    "continuous_latent": continuous[0]
                    .cpu()
                    .to(torch.float16)
                    .contiguous(),
                    "vqgan_sha256": identity["vqgan_sha256"],
                    "codebook_sha256": identity["codebook_sha256"],
                    "data_contract_sha256": identity["data_contract_sha256"],
                    "normalization": ISPY2_DCE0_CONTINUOUS_NORMALIZATION,
                },
            )
            built += 1
        if index % 50 == 0 or index == len(visit_ids):
            print(
                f"I-SPY2 DCE0 continuous encode {index}/{len(visit_ids)} "
                f"(built={built}, reused={reused})"
            )
    identity_path = root / "cache_identity.json"
    _atomic_json(identity_path, identity)
    return identity_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prepare-ispy2-dce0-world-latents")
    parser.add_argument("--config", required=True)
    parser.add_argument("--vqgan-checkpoint")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = prepare_ispy2_dce0_world_latents(
        args.config,
        vqgan_checkpoint=args.vqgan_checkpoint,
        device=args.device,
    )
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ISPY2_DCE0_CONTINUOUS_CACHE_SCHEMA",
    "ISPY2_DCE0_CONTINUOUS_NORMALIZATION",
    "ISPY2_DCE0_CONTINUOUS_PAYLOAD_SCHEMA",
    "ISPY2_DCE0_LATENT_SHAPE",
    "ISPY2DCE0ContinuousLatentCache",
    "ispy2_dce0_continuous_cache_identity",
    "prepare_ispy2_dce0_world_latents",
]
