"""Select the latent codec declared by an experiment configuration."""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

from collections.abc import Mapping, Sequence
from dataclasses import fields
import math
from pathlib import Path
from typing import Any

import yaml
from torch import Tensor, nn

from .autoencoder import SharedAutoencoderKL, build_autoencoder_from_config
from .mewm_vqgan import (
    MEWM_MU_GLIOMA_MODALITIES,
    MEWM_REGISTERED_CODEBOOK_SHA256,
    MEWM_REGISTERED_DATA_BACKEND,
    MEWM_REGISTERED_NUMERIC_CONTRACT,
    MEWM_REGISTERED_VQGAN_DATA_CONTRACT_SHA256,
    MeWMMultimodalVQGANCodec,
    MeWMVQGANCodec,
    MeWMVQGANConfig,
    load_mewm_vqgan_codec,
)


def _declared_path(value: Any, *, config: Mapping[str, Any], label: str) -> Path:
    if type(value) is not str or not value.strip():
        raise ValueError(f"codec.{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        owner = config.get("_config_path")
        base = Path(owner).expanduser().resolve().parent if owner else Path.cwd()
        path = base / path
    return path.resolve()


def load_mewm_vqgan_architecture_config(
    path: str | Path,
    *,
    expected_data_backend: str = MEWM_REGISTERED_DATA_BACKEND,
) -> tuple[MeWMVQGANConfig, tuple[int, int, int]]:
    """Read and validate the relevant generator fields in an upstream YAML."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = _release_yaml(handle)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid MeWM VQ-GAN YAML: {source}") from error
    if type(payload) is not dict:
        raise ValueError("MeWM VQ-GAN config root must be an exact mapping")
    config_fields = {field.name for field in fields(MeWMVQGANConfig)}
    models = payload.get("models")
    raw = models.get("vqgan") if type(models) is dict else None
    mu_glioma_config = payload.get("schema_version") == (
        "mewm_mu_glioma_post_vqgan_config_v1"
    )
    if mu_glioma_config:
        raw = payload.get("model")
        if expected_data_backend != "mu_glioma_post":
            raise ValueError(
                "MU-Glioma VQ-GAN config requires data_backend=mu_glioma_post"
            )
        architecture_label = "MeWM MU-Glioma model"
    else:
        provenance_fields = {"ct_source_repo", "ct_source_revision"}
        architecture_label = "MeWM models.vqgan"
    if type(raw) is not dict:
        raise ValueError("MeWM VQ-GAN config has no supported model mapping")
    allowed_fields = (
        config_fields
        if mu_glioma_config
        else config_fields | provenance_fields
    )
    unknown = set(raw).difference(allowed_fields)
    if unknown:
        raise ValueError(f"unsupported fields in {architecture_label}: {sorted(unknown)}")
    try:
        config = MeWMVQGANConfig(
            **{key: value for key, value in raw.items() if key in config_fields}
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{architecture_label} architecture is invalid") from error

    data = payload.get("data")
    if type(data) is not dict or (
        not mu_glioma_config and data.get("backend") != expected_data_backend
    ):
        raise ValueError(
            "MeWM VQ-GAN config data backend differs from the declared contract"
        )
    if mu_glioma_config and tuple(data.get("modalities", ())) != MEWM_MU_GLIOMA_MODALITIES:
        raise ValueError("MeWM MU-Glioma VQ-GAN modality order is invalid")
    raw_shape = data.get("output_shape_zyx")
    if (
        not isinstance(raw_shape, (list, tuple))
        or len(raw_shape) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_shape)
    ):
        raise ValueError("MeWM VQ-GAN config must declare data.output_shape_zyx")
    image_shape = tuple(int(value) for value in raw_shape)
    config.latent_shape(image_shape)
    return config, image_shape


def build_codec_from_config(
    config: Mapping[str, Any],
    *,
    autoencoder_backend: nn.Module | None = None,
    latent_statistics: Mapping[str, Any] | None = None,
) -> SharedAutoencoderKL | MeWMVQGANCodec | MeWMMultimodalVQGANCodec:
    """Build either the local MONAI codec or the pinned external MeWM VQ-GAN."""

    raw_codec = config.get("codec")
    if raw_codec is None:
        codec: SharedAutoencoderKL | MeWMVQGANCodec | MeWMMultimodalVQGANCodec
        codec = build_autoencoder_from_config(config, backend=autoencoder_backend)
    else:
        if not isinstance(raw_codec, Mapping):
            raise ValueError("codec must be a mapping")
        section = dict(raw_codec)
        backend_name = str(section.get("backend", "")).strip().lower()
        if backend_name in {"monai", "monai_autoencoderkl", "autoencoderkl"}:
            unknown = set(section).difference({"backend"})
            if unknown:
                raise ValueError(
                    f"unsupported MONAI codec config fields: {sorted(unknown)}"
                )
            codec = build_autoencoder_from_config(config, backend=autoencoder_backend)
        elif backend_name in {"mewm_vqgan", "mewm_mu_glioma_vqgan"}:
            if autoencoder_backend is not None:
                raise ValueError("autoencoder_backend cannot override a MeWM VQ-GAN")
            accepted = {
                "backend",
                "config_path",
                "checkpoint_path",
                "checkpoint_sha256",
                "codebook_sha256",
                "data_backend",
                "data_contract_sha256",
                "image_shape",
                "numeric_contract",
            }
            if backend_name == "mewm_mu_glioma_vqgan":
                accepted.update({"modalities", "codebook_min", "codebook_max"})
            unknown = set(section).difference(accepted)
            if unknown:
                raise ValueError(
                    f"unsupported MeWM codec config fields: {sorted(unknown)}"
                )
            missing = [
                key
                for key in ("config_path", "checkpoint_path", "checkpoint_sha256")
                if key not in section
            ]
            if missing:
                raise ValueError(f"MeWM codec config is missing required fields: {missing}")
            expected_backend = str(
                section.get("data_backend", MEWM_REGISTERED_DATA_BACKEND)
            )
            upstream_config_path = _declared_path(
                section["config_path"], config=config, label="config_path"
            )
            checkpoint_path = _declared_path(
                section["checkpoint_path"], config=config, label="checkpoint_path"
            )
            upstream_config, configured_image_shape = (
                load_mewm_vqgan_architecture_config(
                    upstream_config_path,
                    expected_data_backend=expected_backend,
                )
            )
            raw_image_shape = section.get("image_shape", configured_image_shape)
            if not isinstance(raw_image_shape, Sequence) or isinstance(
                raw_image_shape, (str, bytes)
            ):
                raise ValueError("codec.image_shape must be a three-element sequence")
            image_shape = tuple(int(value) for value in raw_image_shape)
            if image_shape != configured_image_shape:
                raise ValueError(
                    "codec.image_shape differs from the upstream VQ-GAN config"
                )
            single_codec = load_mewm_vqgan_codec(
                checkpoint_path,
                expected_sha256=section["checkpoint_sha256"],
                expected_config=upstream_config,
                expected_data_contract_sha256=str(
                    section.get(
                        "data_contract_sha256",
                        MEWM_REGISTERED_VQGAN_DATA_CONTRACT_SHA256,
                    )
                ),
                expected_data_backend=expected_backend,
                expected_numeric_contract=str(
                    section.get(
                        "numeric_contract", MEWM_REGISTERED_NUMERIC_CONTRACT
                    )
                ),
                expected_codebook_sha256=section.get(
                    "codebook_sha256", MEWM_REGISTERED_CODEBOOK_SHA256
                ),
                image_shape=image_shape,
            )
            if backend_name == "mewm_mu_glioma_vqgan":
                modalities = tuple(section.get("modalities", MEWM_MU_GLIOMA_MODALITIES))
                try:
                    minimum = float(section["codebook_min"])
                    maximum = float(section["codebook_max"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        "MU-Glioma codec requires finite codebook_min/codebook_max"
                    ) from error
                if (
                    not math.isfinite(minimum)
                    or not math.isfinite(maximum)
                    or not minimum < maximum
                ):
                    raise ValueError("MU-Glioma codec codebook range is invalid")
                codec = MeWMMultimodalVQGANCodec(
                    single_codec,
                    modalities=modalities,
                    latent_mean=(minimum + maximum) * 0.5,
                    latent_std=(maximum - minimum) * 0.5,
                )
            else:
                codec = single_codec
        else:
            raise ValueError(
                "codec.backend must be 'mewm_vqgan', "
                "'mewm_mu_glioma_vqgan', or 'monai_autoencoderkl'"
            )

    if latent_statistics is not None:
        if not isinstance(latent_statistics, Mapping):
            raise ValueError("latent_statistics must be a mapping")
        if "mean" not in latent_statistics or "std" not in latent_statistics:
            raise ValueError("latent_statistics must contain mean and std")
        codec.set_latent_statistics(
            latent_statistics["mean"], latent_statistics["std"]
        )
    return codec


__all__ = [
    "build_codec_from_config",
    "load_mewm_vqgan_architecture_config",
]
