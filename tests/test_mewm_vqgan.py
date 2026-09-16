from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from ispy2_symmflow.models.mewm_vqgan import (
    MEWM_REGISTERED_IMAGE_SHAPE,
    MEWM_REGISTERED_LATENT_SHAPE,
    MEWM_REGISTERED_NUMERIC_CONTRACT,
    MEWM_REGISTERED_VQGAN_CONFIG,
    MEWM_VQGAN_CHECKPOINT_SCHEMA,
    MRILevelVQGAN,
    MeWMMultimodalVQGANCodec,
    MeWMVQGANCheckpointError,
    MeWMVQGANCodec,
    MeWMVQGANConfig,
    load_mewm_vqgan_codec,
)
from ispy2_symmflow.models.codec import build_codec_from_config


_TEST_DATA_CONTRACT = "1" * 64


def _tiny_config(*, embedding_dim: int = 8) -> MeWMVQGANConfig:
    return MeWMVQGANConfig(
        hidden_channels=4,
        embedding_dim=embedding_dim,
        n_codes=16,
        bottleneck_blocks=1,
        num_groups=4,
        nearest_chunk_size=4,
    )


def _identity(config: MeWMVQGANConfig) -> dict[str, Any]:
    return {
        "schema_version": MEWM_VQGAN_CHECKPOINT_SCHEMA,
        "mri_finetuned": True,
        "numeric_contract": MEWM_REGISTERED_NUMERIC_CONTRACT,
        "data_backend": "registered_t0",
        "data_contract_sha256": _TEST_DATA_CONTRACT,
        "architecture_contract": config.architecture_contract(),
    }


def _write_checkpoint(
    path: Path,
    config: MeWMVQGANConfig,
    *,
    state_transform: Any | None = None,
    identity_transform: Any | None = None,
) -> tuple[MRILevelVQGAN, str]:
    model = MRILevelVQGAN(config)
    state = {
        f"autoencoder.{key}": value.detach().clone()
        for key, value in model.state_dict().items()
    }
    state["image_discriminator.unused"] = torch.ones(1)
    if state_transform is not None:
        state_transform(state)
    identity = _identity(config)
    if identity_transform is not None:
        identity_transform(identity)
    torch.save(
        {
            "epoch": 1,
            "global_step": 2,
            "state_dict": state,
            "mewm_ispy2_vqgan_identity": identity,
        },
        path,
    )
    return model, hashlib.sha256(path.read_bytes()).hexdigest()


class _RecordingQuantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.last_input = latent.detach().clone()
        return latent + 10.0, {"indices": torch.zeros(1, dtype=torch.long)}


class _FakeBackend(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.quantizer = _RecordingQuantizer()
        self.last_decoder_input: torch.Tensor | None = None

    def encode_continuous(self, image: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool3d(image, kernel_size=4, stride=4)
        return pooled.repeat(1, self.embedding_dim, 1, 1, 1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.last_decoder_input = latent.detach().clone()
        return F.interpolate(latent[:, :1], scale_factor=4, mode="nearest")


def test_production_architecture_and_shapes_are_locked() -> None:
    config = MEWM_REGISTERED_VQGAN_CONFIG

    assert config.hidden_channels == 16
    assert config.embedding_dim == 8
    assert config.n_codes == 16384
    assert config.bottleneck_blocks == 1
    assert config.latent_shape(MEWM_REGISTERED_IMAGE_SHAPE) == (
        MEWM_REGISTERED_LATENT_SHAPE
    )
    model = MRILevelVQGAN(config)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_834_217
    assert len(model.state_dict()) == 87


def test_core_vqgan_has_fourfold_continuous_latent_and_reconstruction() -> None:
    model = MRILevelVQGAN(_tiny_config()).eval()
    image = torch.randn(1, 1, 8, 12, 16)

    with torch.no_grad():
        continuous = model.encode_continuous(image)
        quantized, diagnostics = model.quantizer(continuous)
        reconstruction = model.decode(quantized)

    assert continuous.shape == (1, 8, 2, 3, 4)
    assert quantized.shape == continuous.shape
    assert diagnostics["indices"].shape == (1, 2, 3, 4)
    assert reconstruction.shape == image.shape


def test_codec_normalizes_continuous_encoder_output_channelwise() -> None:
    config = _tiny_config(embedding_dim=2)
    backend = _FakeBackend(config.embedding_dim)
    codec = MeWMVQGANCodec(config, backend=backend, image_shape=(8, 8, 8))
    codec.set_latent_statistics([1.0, -2.0], [2.0, 4.0])
    image = torch.arange(8**3, dtype=torch.float32).reshape(1, 1, 8, 8, 8)

    raw = codec.encode(image)
    normalized = codec.encode(image, normalize=True)

    torch.testing.assert_close(normalized[:, 0], (raw[:, 0] - 1.0) / 2.0)
    torch.testing.assert_close(normalized[:, 1], (raw[:, 1] + 2.0) / 4.0)
    torch.testing.assert_close(codec.denormalize_latent(normalized), raw)


def test_decode_inverse_normalizes_then_quantizes_before_decoder() -> None:
    config = _tiny_config(embedding_dim=2)
    backend = _FakeBackend(config.embedding_dim)
    codec = MeWMVQGANCodec(config, backend=backend, image_shape=(8, 8, 8))
    codec.set_latent_statistics([1.0, -2.0], [2.0, 4.0])
    normalized = torch.randn(1, 2, 2, 2, 2)
    expected_continuous = codec.denormalize_latent(normalized)

    reconstruction = codec.decode(normalized, denormalize=True)

    assert backend.quantizer.last_input is not None
    assert backend.last_decoder_input is not None
    torch.testing.assert_close(backend.quantizer.last_input, expected_continuous)
    torch.testing.assert_close(
        backend.last_decoder_input, expected_continuous + 10.0
    )
    assert reconstruction.shape == (1, 1, 8, 8, 8)


def test_codec_is_permanently_eval_to_protect_ema_codebook() -> None:
    config = _tiny_config(embedding_dim=2)
    backend = _FakeBackend(config.embedding_dim)
    codec = MeWMVQGANCodec(config, backend=backend, image_shape=(8, 8, 8))

    codec.train(True)

    assert not codec.training
    assert not backend.training
    assert not backend.quantizer.training
    assert all(not parameter.requires_grad for parameter in codec.parameters())
    with pytest.raises(RuntimeError, match="frozen pretrained codec"):
        codec.unfreeze()


def test_multimodal_codec_preserves_fixed_modality_channel_groups() -> None:
    config = _tiny_config(embedding_dim=2)
    single = MeWMVQGANCodec(
        config, backend=_FakeBackend(config.embedding_dim), image_shape=(8, 8, 8)
    )
    codec = MeWMMultimodalVQGANCodec(single, latent_mean=1.0, latent_std=2.0)
    image = torch.stack(
        tuple(torch.full((8, 8, 8), float(index)) for index in range(4)),
        dim=0,
    ).unsqueeze(0)

    raw = codec.encode(image)
    normalized = codec.encode(image, normalize=True)
    reconstruction = codec.decode(normalized, denormalize=True)

    assert raw.shape == (1, 8, 2, 2, 2)
    for index in range(4):
        torch.testing.assert_close(
            raw[:, index * 2 : (index + 1) * 2],
            torch.full((1, 2, 2, 2, 2), float(index)),
        )
    torch.testing.assert_close(normalized, (raw - 1.0) / 2.0)
    assert reconstruction.shape == image.shape
    torch.testing.assert_close(
        reconstruction,
        image + 10.0,
    )
    assert not codec.training
    assert all(not parameter.requires_grad for parameter in codec.parameters())


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (torch.zeros(1, 2, 2, 2), "shape"),
        (torch.zeros(1, 2, 8, 8, 8), "expected 1 DCE0 channel"),
        (torch.zeros(1, 1, 4, 8, 8), "spatial shape"),
    ),
)
def test_codec_rejects_wrong_image_contract(
    value: torch.Tensor, message: str
) -> None:
    config = MeWMVQGANConfig(
        hidden_channels=2,
        embedding_dim=2,
        n_codes=4,
        num_groups=2,
    )
    codec = MeWMVQGANCodec(
        config, backend=_FakeBackend(2), image_shape=(8, 8, 8)
    )
    with pytest.raises(ValueError, match=message):
        codec.encode(value)


def test_strict_checkpoint_loader_round_trip(tmp_path: Path) -> None:
    config = _tiny_config()
    expected, digest = _write_checkpoint(tmp_path / "vqgan.ckpt", config)

    codec = load_mewm_vqgan_codec(
        tmp_path / "vqgan.ckpt",
        expected_sha256=digest,
        expected_config=config,
        expected_data_contract_sha256=_TEST_DATA_CONTRACT,
        expected_codebook_sha256=None,
        image_shape=(8, 8, 8),
        latent_mean=[0.0] * 8,
        latent_std=[1.0] * 8,
    )

    assert codec.checkpoint_sha256 == digest
    assert codec.checkpoint_identity["numeric_contract"] == (
        MEWM_REGISTERED_NUMERIC_CONTRACT
    )
    for key, value in codec.backend.state_dict().items():
        assert torch.equal(value, expected.state_dict()[key])


def test_generic_factory_loads_mewm_codec_from_relative_paths(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    checkpoint = tmp_path / "vqgan.ckpt"
    _, digest = _write_checkpoint(checkpoint, config)
    upstream = tmp_path / "upstream.yaml"
    architecture = config.architecture_contract()["config"]
    upstream.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "backend": "registered_t0",
                    "output_shape_zyx": [8, 8, 8],
                },
                "models": {"vqgan": architecture},
            }
        ),
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("project: {}\n", encoding="utf-8")

    codec = build_codec_from_config(
        {
            "_config_path": str(experiment),
            "codec": {
                "backend": "mewm_vqgan",
                "config_path": upstream.name,
                "checkpoint_path": checkpoint.name,
                "checkpoint_sha256": digest,
                "data_contract_sha256": _TEST_DATA_CONTRACT,
                "codebook_sha256": None,
            },
        },
        latent_statistics={"mean": [0.0] * 8, "std": [1.0] * 8},
    )

    assert isinstance(codec, MeWMVQGANCodec)
    assert codec.image_shape == (8, 8, 8)
    assert codec.checkpoint_sha256 == digest


def test_factory_loads_mu_glioma_multimodal_codec(tmp_path: Path) -> None:
    config = _tiny_config()
    checkpoint = tmp_path / "vqgan.ckpt"

    def mu_identity(identity: dict[str, Any]) -> None:
        identity["data_backend"] = "mu_glioma_post"
        identity["numeric_contract"] = "mu-numeric-contract"

    _, digest = _write_checkpoint(
        checkpoint, config, identity_transform=mu_identity
    )
    upstream = tmp_path / "mu.yaml"
    upstream.write_text(
        yaml.safe_dump(
            {
                "schema_version": "mewm_mu_glioma_post_vqgan_config_v1",
                "data": {
                    "modalities": ["t1c", "t1n", "t2f", "t2w"],
                    "output_shape_zyx": [8, 8, 8],
                },
                "model": config.architecture_contract()["config"],
            }
        ),
        encoding="utf-8",
    )

    codec = build_codec_from_config(
        {
            "codec": {
                "backend": "mewm_mu_glioma_vqgan",
                "config_path": str(upstream),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": digest,
                "data_backend": "mu_glioma_post",
                "data_contract_sha256": _TEST_DATA_CONTRACT,
                "numeric_contract": "mu-numeric-contract",
                "codebook_sha256": None,
                "image_shape": [8, 8, 8],
                "modalities": ["t1c", "t1n", "t2f", "t2w"],
                "codebook_min": -3.0,
                "codebook_max": 5.0,
            }
        },
        latent_statistics={"mean": [1.0] * 32, "std": [4.0] * 32},
    )

    assert isinstance(codec, MeWMMultimodalVQGANCodec)
    assert codec.latent_shape == (32, 2, 2, 2)
    assert codec.modalities == ("t1c", "t1n", "t2f", "t2w")


def test_checkpoint_hash_is_checked_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config()
    path = tmp_path / "vqgan.ckpt"
    _write_checkpoint(path, config)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint must not be deserialized")
        ),
    )

    with pytest.raises(MeWMVQGANCheckpointError, match="SHA-256 mismatch"):
        load_mewm_vqgan_codec(
            path,
            expected_sha256="0" * 64,
            expected_config=config,
            expected_data_contract_sha256=_TEST_DATA_CONTRACT,
            expected_codebook_sha256=None,
            image_shape=(8, 8, 8),
        )


def test_checkpoint_architecture_identity_is_pinned(tmp_path: Path) -> None:
    config = _tiny_config()
    path = tmp_path / "wrong-architecture.ckpt"

    def alter(identity: dict[str, Any]) -> None:
        identity["architecture_contract"]["config"]["hidden_channels"] = 8

    _, digest = _write_checkpoint(path, config, identity_transform=alter)
    with pytest.raises(MeWMVQGANCheckpointError, match="pinned configuration"):
        load_mewm_vqgan_codec(
            path,
            expected_sha256=digest,
            expected_config=config,
            expected_data_contract_sha256=_TEST_DATA_CONTRACT,
            expected_codebook_sha256=None,
            image_shape=(8, 8, 8),
        )


@pytest.mark.parametrize("corruption", ("missing", "unexpected", "dtype", "nonfinite"))
def test_checkpoint_autoencoder_state_is_strict(
    tmp_path: Path, corruption: str
) -> None:
    config = _tiny_config()
    path = tmp_path / f"{corruption}.ckpt"

    def alter(state: dict[str, torch.Tensor]) -> None:
        key = "autoencoder.encoder.input.weight"
        if corruption == "missing":
            state.pop(key)
        elif corruption == "unexpected":
            state["autoencoder.not_a_real_buffer"] = torch.zeros(1)
        elif corruption == "dtype":
            state[key] = state[key].half()
        else:
            state[key].flatten()[0] = float("nan")

    _, digest = _write_checkpoint(path, config, state_transform=alter)
    with pytest.raises(MeWMVQGANCheckpointError, match="state is incomplete"):
        load_mewm_vqgan_codec(
            path,
            expected_sha256=digest,
            expected_config=config,
            expected_data_contract_sha256=_TEST_DATA_CONTRACT,
            expected_codebook_sha256=None,
            image_shape=(8, 8, 8),
        )


@pytest.mark.integration
@pytest.mark.gpu
def test_copied_checkpoint_matches_copied_continuous_latent() -> None:
    if os.environ.get("RUN_MEWM_VQGAN_INTEGRATION") != "1":
        pytest.skip("set RUN_MEWM_VQGAN_INTEGRATION=1 to exercise copied MeWM data")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for production-volume parity validation")
    root = Path(
        os.environ.get(
            "MEWM_SMOKE_ROOT", _release_path('@external/DATASETS/mewm-ispy2-paired-smoke')
        )
    )
    checkpoint = root / "vqgan" / "best-composite-67-247248.ckpt"
    if not os.environ.get("MEWM_SMOKE_ROI") or not os.environ.get("MEWM_SMOKE_LATENT"):
        pytest.fail("Set MEWM_SMOKE_ROI and MEWM_SMOKE_LATENT to a matched private test pair")
    roi_path = Path(os.environ["MEWM_SMOKE_ROI"])
    latent_path = Path(os.environ["MEWM_SMOKE_LATENT"])
    statistics = {
        "mean": [
            1.0103906584856293,
            0.2506868008964772,
            -0.0038918188331322997,
            -0.6050817792326734,
            -0.6378086730999052,
            0.04597189020914612,
            0.8415360453504631,
            0.3862045789221464,
        ],
        "std": [
            3.98505345722896,
            2.3158803127333027,
            2.7300365366344117,
            2.95361494470835,
            6.340826633978138,
            3.1996728532484098,
            4.710499268425119,
            2.386910113780715,
        ],
    }
    codec = load_mewm_vqgan_codec(
        checkpoint,
        latent_mean=statistics["mean"],
        latent_std=statistics["std"],
    ).cuda()
    roi = torch.load(roi_path, map_location="cpu", weights_only=True)
    cached = torch.load(latent_path, map_location="cpu", weights_only=True)[
        "continuous_latent"
    ]
    image = roi["mri"][0:1].unsqueeze(0).cuda().float()

    with torch.inference_mode():
        actual = codec.encode(image)
        reconstruction = codec.decode(
            codec.normalize_latent(cached.unsqueeze(0).cuda().float()),
            denormalize=True,
        )

    torch.testing.assert_close(actual.cpu(), cached.unsqueeze(0).float(), rtol=2e-3, atol=2e-2)
    assert reconstruction.shape == (1, 1, *MEWM_REGISTERED_IMAGE_SHAPE)
    assert torch.isfinite(reconstruction).all()
