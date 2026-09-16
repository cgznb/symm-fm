from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.evaluation.metrics import evaluate_prediction, evaluate_sample_set, mask_volume_ml


def test_perfect_prediction_and_foreground() -> None:
    target = torch.ones(2, 1, 4, 4, 4)
    metrics = evaluate_prediction(
        target.clone(), target, data_range=2.0, foreground_mask=torch.ones(2, 1, 4, 4, 4)
    )
    assert torch.equal(metrics["mae"], torch.zeros(2))
    assert torch.isinf(metrics["psnr"]).all()
    assert torch.allclose(metrics["ssim"], torch.ones(2), atol=1e-5)
    assert metrics["foreground_valid"].all()


def test_empty_foreground_is_invalid_not_zero() -> None:
    target = torch.ones(1, 1, 4, 4, 4)
    metrics = evaluate_prediction(
        torch.zeros_like(target), target, data_range=1.0, foreground_mask=torch.zeros_like(target)
    )
    assert not metrics["foreground_valid"].item()
    assert torch.isnan(metrics["foreground_mae"]).item()


def test_unverified_tumor_mask_rejected() -> None:
    image = torch.zeros(1, 1, 4, 4, 4)
    with pytest.raises(ValueError, match="semantics"):
        evaluate_prediction(image, image, data_range=1.0, tumor_mask=torch.ones_like(image))


def test_physical_mask_volume() -> None:
    mask = torch.ones(1, 1, 2, 2, 2)
    assert mask_volume_ml(mask, (2.0, 2.0, 2.0)).item() == pytest.approx(0.064)


def test_sample_distribution_is_separate_from_mean_and_oracle() -> None:
    target = torch.zeros(1, 1, 4, 4, 4)
    samples = torch.stack((torch.zeros_like(target), torch.ones_like(target)), dim=0)
    metrics = evaluate_sample_set(samples, target, data_range=1.0)
    assert metrics["candidate_mae"].shape == (2, 1)
    assert metrics["candidate_metrics"]["ssim"].shape == (2, 1)
    assert metrics["candidate_metrics"]["psnr"].shape == (2, 1)
    assert metrics["predictive_mean"]["mae"].item() == pytest.approx(0.5)
    assert metrics["oracle_best_mae"].item() == 0.0
    assert "not a primary" in metrics["oracle_warning"]


def test_metric_boundary_inputs_have_explicit_errors() -> None:
    image = torch.zeros(1, 2, 4, 4, 4)
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_prediction(image, image, data_range=float("nan"))
    with pytest.raises(ValueError, match="at least one"):
        evaluate_sample_set(torch.empty(0, 1, 2, 4, 4, 4), image, data_range=1.0)
    with pytest.raises(ValueError, match="tumor mask geometry"):
        evaluate_prediction(
            image,
            image,
            data_range=1.0,
            tumor_mask=torch.ones(2, 1, 4, 4, 4),
            mask_semantics_verified=True,
        )
    with pytest.raises(TypeError, match="floating"):
        evaluate_prediction(image.long(), image.long(), data_range=1.0)
    with pytest.raises(ValueError, match="finite"):
        mask_volume_ml(torch.tensor([[[[float("nan")]]]]), (1.0, 1.0, 1.0))
