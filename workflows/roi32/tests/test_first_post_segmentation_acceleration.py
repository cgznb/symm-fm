from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("nnunetv2")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from mewm_ispy2.first_post_segmentation_acceleration import BatchedNNUNetPredictor


def predictors(tile_batch=2, mirror_batch=8):
    torch.manual_seed(2026)
    torch.set_num_threads(2)
    baseline = nnUNetPredictor(device=torch.device("cpu"), allow_tqdm=False)
    baseline.network = torch.nn.Conv3d(1, 2, 3, padding=1).eval()
    baseline.allowed_mirroring_axes = (0, 1, 2)
    baseline.configuration_manager = SimpleNamespace(patch_size=[4, 4, 4])
    baseline.label_manager = SimpleNamespace(num_segmentation_heads=2)
    batched = BatchedNNUNetPredictor(
        device=torch.device("cpu"),
        allow_tqdm=False,
        tile_batch_size=tile_batch,
        mirror_batch_size=mirror_batch,
    )
    for key in (
        "network",
        "allowed_mirroring_axes",
        "configuration_manager",
        "label_manager",
    ):
        setattr(batched, key, copy.deepcopy(getattr(baseline, key)))
    return baseline, batched


@pytest.mark.parametrize("mirrors", [1, 2, 4, 8])
def test_batched_mirrors_match_all_official_axes(mirrors):
    baseline, batched = predictors(mirror_batch=mirrors)
    data = torch.randn(3, 1, 4, 4, 4)
    with torch.inference_mode():
        expected = baseline._internal_maybe_mirror_and_predict(data)
        actual = batched._internal_maybe_mirror_and_predict(data)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert batched.largest_forward_batch == 3 * mirrors


@pytest.mark.parametrize("tiles", [1, 2, 4])
def test_batched_tiles_preserve_gaussian_overlap_and_partial_batch(tiles):
    baseline, batched = predictors(tile_batch=tiles)
    data = torch.randn(1, 7, 7, 7)
    slices = baseline._internal_get_sliding_window_slicers(data.shape[1:])
    with torch.inference_mode():
        expected = baseline._internal_predict_sliding_window_return_logits(
            data, slices, False
        )
        actual = batched._internal_predict_sliding_window_return_logits(
            data, slices, False
        )
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.003)
    assert np.array_equal(actual.argmax(0).numpy(), expected.argmax(0).numpy())


def test_oom_reduces_batch_without_skipping_tiles_or_mirrors(monkeypatch):
    baseline, batched = predictors(tile_batch=2)
    data = torch.randn(1, 7, 7, 7)
    slices = baseline._internal_get_sliding_window_slicers(data.shape[1:])
    predict = batched._internal_maybe_mirror_and_predict

    def limited(work):
        if len(work) > 1:
            raise torch.cuda.OutOfMemoryError("simulated activation limit")
        return predict(work)

    monkeypatch.setattr(batched, "_internal_maybe_mirror_and_predict", limited)
    with torch.inference_mode():
        expected = baseline._internal_predict_sliding_window_return_logits(
            data, slices, False
        )
        actual = batched._internal_predict_sliding_window_return_logits(
            data, slices, False
        )
    assert batched.tile_batch_size == 1
    assert batched.oom_reductions == 1
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.003)
