import numpy as np
import pytest
import torch

from mewm_ispy2 import registered_roi32_evaluation as evaluation


def test_bf16_metrics_equal_explicit_float32_conversion(monkeypatch):
    prediction = torch.linspace(-0.5, 2.0, 8**3).reshape(1, 1, 8, 8, 8).bfloat16()
    target = (prediction.float() * 0.8).bfloat16()
    mask = torch.ones_like(prediction, dtype=torch.bool)
    monkeypatch.setattr(evaluation.vq, 'fixed_slices', lambda *args: None)
    monkeypatch.setattr(evaluation.vq, '_orthogonal_slices', lambda value, _: (value[..., 0].float(),) * 3)
    original_ssim = evaluation.structural_similarity

    def checked_ssim(real, generated, **kwargs):
        assert real.dtype == generated.dtype == np.float32
        return original_ssim(real, generated, **kwargs)

    monkeypatch.setattr(evaluation, 'structural_similarity', checked_ssim)
    perceptual = lambda generated, real: (generated - real).square().mean()
    actual = evaluation.image_metrics(prediction, target, mask, mask, perceptual, 'synthetic')
    expected = evaluation.image_metrics(prediction.float(), target.float(), mask, mask, perceptual, 'synthetic')
    assert actual == pytest.approx(expected)
