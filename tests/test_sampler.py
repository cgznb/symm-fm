from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.inference.sampler import (
    SymmFlowSampler,
    save_sample_batch,
    validate_source_archive_binding,
)
from ispy2_symmflow.utils.hashing import sha256_file


class IdentityAutoencoder(torch.nn.Module):
    def encode(self, image, *, normalize=True):
        return image

    def decode(self, latent, *, denormalize=True):
        return latent


class ZeroVelocity(torch.nn.Module):
    def forward(self, state, tau, condition_tokens):
        return torch.zeros_like(state)


class TrackingAutoencoder(torch.nn.Module):
    def __init__(self, *, encode_offset=0.0):
        super().__init__()
        self.encode_offset = float(encode_offset)
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, image, *, normalize=True):
        self.encode_calls.append((image.detach().clone(), normalize))
        return image + self.encode_offset

    def decode(self, latent, *, denormalize=True):
        self.decode_calls.append((latent.detach().clone(), denormalize))
        return latent


def test_fixed_seed_reproducible_and_new_seed_changes_candidate() -> None:
    sampler = SymmFlowSampler(IdentityAutoencoder(), ZeroVelocity())
    source = torch.ones(1, 2, 2, 2, 2)
    condition = torch.zeros(1, 3, 4)
    first = sampler.sample_forward(source, condition, num_samples=2, seed=41, steps=2)
    second = sampler.sample_forward(source, condition, num_samples=2, seed=41, steps=2)
    third = sampler.sample_forward(source, condition, num_samples=2, seed=42, steps=2)
    assert torch.equal(first.samples, second.samples)
    assert not torch.equal(first.samples[0], third.samples[0])
    assert first.nfe_per_sample == 4


def test_backward_decodes_second_branch_and_returns_joint() -> None:
    sampler = SymmFlowSampler(IdentityAutoencoder(), ZeroVelocity())
    source = torch.ones(1, 2, 2, 2, 2)
    condition = torch.zeros(1, 3, 4)
    result = sampler.sample_backward(
        source, condition, num_samples=1, seed=7, steps=3, solver="euler", return_joint_state=True
    )
    assert result.final_joint_states is not None
    assert result.final_joint_states.shape[2] == 4
    assert torch.equal(result.samples[0], result.final_joint_states[0, :, 2:])
    assert result.nfe_per_sample == 3


@pytest.mark.parametrize(
    ("method_name", "source_branch", "output_branch"),
    (
        ("sample_forward_latent", slice(2, None), slice(0, 2)),
        ("sample_backward_latent", slice(0, 2), slice(2, None)),
    ),
)
def test_direct_latent_sampling_skips_encode_and_preserves_branch_order(
    method_name, source_branch, output_branch
) -> None:
    codec = TrackingAutoencoder()
    sampler = SymmFlowSampler(codec, ZeroVelocity())
    source = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 2, 2)
    condition = torch.zeros(1, 3, 4)

    result = getattr(sampler, method_name)(
        source,
        condition,
        num_samples=2,
        seed=17,
        steps=1,
        solver="euler",
        return_joint_state=True,
    )

    assert codec.encode_calls == []
    assert result.final_joint_states is not None
    assert result.final_joint_states.shape == (2, 1, 4, 2, 2, 2)
    assert torch.equal(
        result.final_joint_states[:, :, source_branch],
        source.unsqueeze(0).expand(2, -1, -1, -1, -1, -1),
    )
    assert all(denormalize is True for _, denormalize in codec.decode_calls)
    for index, (decoded_latent, _) in enumerate(codec.decode_calls):
        assert torch.equal(
            decoded_latent, result.final_joint_states[index, :, output_branch]
        )
        assert torch.equal(result.samples[index], decoded_latent)


def test_image_sampling_reuses_the_direct_normalized_latent_path() -> None:
    class EarlierToLaterVelocity(torch.nn.Module):
        def forward(self, state, tau, condition_tokens):
            channels = state.shape[1] // 2
            return torch.cat(
                (state[:, channels:], torch.zeros_like(state[:, channels:])), dim=1
            )

    codec = TrackingAutoencoder(encode_offset=3.0)
    sampler = SymmFlowSampler(codec, EarlierToLaterVelocity())
    image = torch.ones(1, 1, 2, 2, 2)
    condition = torch.zeros(1, 3, 4)

    from_image = sampler.sample_forward(
        image, condition, num_samples=1, seed=9, steps=1, solver="euler"
    )
    from_latent = sampler.sample_forward_latent(
        image + 3.0, condition, num_samples=1, seed=9, steps=1, solver="euler"
    )

    assert len(codec.encode_calls) == 1
    assert codec.encode_calls[0][1] is True
    assert torch.equal(from_image.samples, from_latent.samples)


@pytest.mark.parametrize(
    ("source", "error_type", "message"),
    (
        (torch.ones(1, 1, 2, 2), ValueError, "shape \\[B,C,D,H,W\\]"),
        (torch.ones(1, 1, 2, 2, 2, dtype=torch.int64), TypeError, "floating dtype"),
        (
            torch.full((1, 1, 2, 2, 2), float("nan")),
            ValueError,
            "only finite values",
        ),
    ),
)
def test_direct_latent_sampling_rejects_invalid_latents(
    source, error_type, message
) -> None:
    sampler = SymmFlowSampler(IdentityAutoencoder(), ZeroVelocity())
    with pytest.raises(error_type, match=message):
        sampler.sample_forward_latent(source, torch.zeros(1, 1, 1), steps=1)


def test_joint_round_trip_for_constant_field() -> None:
    class ConstantVelocity(torch.nn.Module):
        def forward(self, state, tau, condition_tokens):
            return torch.full_like(state, 0.125)

    sampler = SymmFlowSampler(IdentityAutoencoder(), ConstantVelocity())
    initial = torch.randn(1, 4, 2, 2, 2)
    condition = torch.zeros(1, 2, 4)
    forward = sampler.sample_joint(initial, condition, t0=0, t1=1, steps=4, solver="heun")
    reverse = sampler.sample_joint(forward.final_state, condition, t0=1, t1=0, steps=4, solver="heun")
    assert torch.allclose(reverse.final_state, initial, atol=1e-6)


def test_sigma_compatibility_requires_explicit_residual_consent() -> None:
    sampler = SymmFlowSampler(IdentityAutoencoder(), ZeroVelocity(), sigma_min=0.01)
    with pytest.raises(ValueError, match="residual endpoint"):
        sampler.sample_forward(torch.ones(1, 1, 1, 1, 1), torch.zeros(1, 1, 1), steps=1)


def test_sampler_forces_eval_mode_for_seed_replay() -> None:
    class DropoutVelocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=0.5)

        def forward(self, state, tau, condition_tokens):
            return self.dropout(torch.ones_like(state))

    velocity = DropoutVelocity().train()
    sampler = SymmFlowSampler(IdentityAutoencoder(), velocity)
    source = torch.ones(1, 1, 2, 2, 2)
    conditions = torch.zeros(1, 1, 1)
    left = sampler.sample_forward(source, conditions, seed=4, steps=2)
    right = sampler.sample_forward(source, conditions, seed=4, steps=2)
    assert not velocity.training
    assert torch.equal(left.samples, right.samples)


def test_save_normalizes_missing_npz_suffix(tmp_path) -> None:
    sampler = SymmFlowSampler(IdentityAutoencoder(), ZeroVelocity())
    batch = sampler.sample_forward(
        torch.ones(1, 1, 2, 2, 2), torch.zeros(1, 1, 1), num_samples=1, steps=1
    )
    array_path, sidecar = save_sample_batch(batch, tmp_path / "candidate")
    assert array_path.name == "candidate.npz"
    assert array_path.is_file() and sidecar.is_file()
    record = __import__("json").loads(sidecar.read_text(encoding="utf-8"))
    assert record["array_sha256"] == sha256_file(array_path)


def test_source_archive_binding_rejects_replaced_bytes(tmp_path) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"source-v1")
    metadata = {"source_sha256": sha256_file(source)}
    assert validate_source_archive_binding(source, metadata) == metadata["source_sha256"]

    source.write_bytes(b"source-v2")
    with pytest.raises(ValueError, match="source archive SHA-256 differs"):
        validate_source_archive_binding(source, metadata)
