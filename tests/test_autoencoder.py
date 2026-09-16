from __future__ import annotations

import importlib.util

import pytest
import torch
from torch import nn

from ispy2_symmflow.models.autoencoder import (
    AutoencoderKLConfig,
    SharedAutoencoderKL,
    build_autoencoder_from_config,
)


class FakeAutoencoderKL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv3d(3, 2, kernel_size=1)
        self.decoder = nn.Conv3d(2, 3, kernel_size=1)
        self.log_scale = nn.Parameter(torch.full((1, 2, 1, 1, 1), -2.0))

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.encoder(image)
        return mean, self.log_scale.exp().expand_as(mean)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


def _config() -> AutoencoderKLConfig:
    return AutoencoderKLConfig(
        in_channels=3,
        out_channels=3,
        channels=(8, 16),
        num_res_blocks=1,
        attention_levels=(False, False),
        latent_channels=2,
        norm_num_groups=8,
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    )


def test_encode_returns_posterior_mean_and_optional_stats() -> None:
    backend = FakeAutoencoderKL()
    model = SharedAutoencoderKL(_config(), backend=backend)
    image = torch.randn(2, 3, 4, 6, 8)

    latent = model.encode(image)
    returned_latent, posterior = model.encode(image, return_posterior=True)

    torch.testing.assert_close(latent, backend.encoder(image))
    torch.testing.assert_close(returned_latent, latent)
    torch.testing.assert_close(posterior.mean, latent)
    assert posterior.scale.shape == latent.shape
    assert torch.all(posterior.scale > 0)


def test_latent_normalization_is_channelwise_and_reversible() -> None:
    model = SharedAutoencoderKL(_config(), backend=FakeAutoencoderKL())
    model.set_latent_statistics(mean=[2.0, -3.0], std=[0.5, 4.0])
    latent = torch.randn(3, 2, 2, 3, 4)

    normalized = model.normalize_latent(latent)
    restored = model.denormalize_latent(normalized)

    torch.testing.assert_close(restored, latent)
    torch.testing.assert_close(
        normalized[:, 0], (latent[:, 0] - 2.0) / 0.5
    )
    torch.testing.assert_close(
        normalized[:, 1], (latent[:, 1] + 3.0) / 4.0
    )


def test_encode_decode_normalization_keywords_match_explicit_methods() -> None:
    model = SharedAutoencoderKL(_config(), backend=FakeAutoencoderKL())
    model.set_latent_statistics(mean=[0.5, -0.25], std=[2.0, 0.5])
    image = torch.randn(1, 3, 3, 4, 5)

    normalized = model.encode(image, normalize=True)
    torch.testing.assert_close(normalized, model.encode_normalized(image))
    torch.testing.assert_close(
        model.decode(normalized, denormalize=True),
        model.decode_normalized(normalized),
    )


def test_forward_matches_monai_tuple_contract_and_loss_has_gradients() -> None:
    model = SharedAutoencoderKL(_config(), backend=FakeAutoencoderKL())
    image = torch.randn(2, 3, 4, 4, 4)

    reconstruction, mean, scale = model(image, sample_posterior=False)
    expected = model.backend.decode(model.backend.encode(image)[0])

    torch.testing.assert_close(reconstruction, expected)
    assert mean.shape == scale.shape == (2, 2, 4, 4, 4)
    loss = model.reconstruction_loss(
        image, kl_weight=1e-5, sample_posterior=False
    )
    assert loss.total.ndim == loss.reconstruction.ndim == loss.kl.ndim == 0
    assert loss.total >= loss.reconstruction
    loss.total.backward()
    assert model.backend.encoder.weight.grad is not None
    assert model.backend.decoder.weight.grad is not None


def test_latent_statistics_are_checkpointed() -> None:
    first = SharedAutoencoderKL(_config(), backend=FakeAutoencoderKL())
    first.set_latent_statistics(mean=[1.0, 2.0], std=[3.0, 4.0])
    second = SharedAutoencoderKL(_config(), backend=FakeAutoencoderKL())

    second.load_state_dict(first.state_dict())

    torch.testing.assert_close(second.latent_mean, first.latent_mean)
    torch.testing.assert_close(second.latent_std, first.latent_std)


@pytest.mark.parametrize("std", [0.0, -1.0, float("nan")])
def test_invalid_latent_scale_is_rejected(std: float) -> None:
    model = SharedAutoencoderKL(_config(), backend=FakeAutoencoderKL())
    with pytest.raises(ValueError, match="positive and finite"):
        model.set_latent_statistics(0.0, std)


def test_freeze_and_unfreeze_cover_the_shared_backend() -> None:
    model = SharedAutoencoderKL(_config(), backend=FakeAutoencoderKL())
    model.freeze()
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())

    model.unfreeze()
    assert model.training
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_factory_extracts_model_keys_from_training_config() -> None:
    model = build_autoencoder_from_config(
        {
            "autoencoder": {
                "spatial_dims": 3,
                "in_channels": 3,
                "out_channels": 3,
                "channels": [8, 16],
                "num_res_blocks": 1,
                "attention_levels": [False, False],
                "latent_channels": 2,
                "norm_num_groups": 8,
                "learning_rate": 1e-4,
                "kl_weight": 1e-6,
                "gradient_weight": 0.0,
                "gradient_clip_norm": 1.0,
                "warmup_steps": 0,
                "max_epochs": 1,
                "validation_interval_epochs": 1,
            }
        },
        backend=FakeAutoencoderKL(),
    )
    assert model.config.compression_factor == 2


def test_encode_rejects_non_3d_or_wrong_channel_inputs() -> None:
    model = SharedAutoencoderKL(_config(), backend=FakeAutoencoderKL())
    with pytest.raises(ValueError, match="shape"):
        model.encode(torch.randn(1, 3, 8, 8))
    with pytest.raises(ValueError, match="expected 3 MRI channels"):
        model.encode(torch.randn(1, 1, 4, 4, 4))


@pytest.mark.skipif(
    importlib.util.find_spec("monai") is None,
    reason="MONAI train dependency is not installed",
)
def test_real_monai_autoencoder_3d_shape() -> None:
    model = SharedAutoencoderKL(_config())
    image = torch.randn(1, 3, 8, 16, 16)

    with torch.no_grad():
        latent = model.encode(image)
        reconstruction = model.decode(latent)

    assert latent.shape == (1, 2, 4, 8, 8)
    assert reconstruction.shape == image.shape
