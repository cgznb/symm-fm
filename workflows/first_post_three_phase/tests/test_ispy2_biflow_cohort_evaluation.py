from types import SimpleNamespace

import math
import numpy as np

from mewm_ispy2.ispy2_biflow_cohort_evaluation import (
    ADJACENT_TRANSITIONS,
    aggregate_patient_rows,
    all_validation_pairs,
    comparison_metrics,
    describe_values,
    region_masks,
    select_validation_pairs,
    valid_ssim_centers,
)


def _pair(patient: str, split: str, source: int, target: int):
    return SimpleNamespace(
        patient_id=patient,
        split=split,
        source_stage=source,
        target_stage=target,
        transition_type=f"T{source}->T{target}",
        pair_id=f"{patient}:T{source}->T{target}",
    )


def test_all_validation_pairs_selects_every_pair_in_stable_order() -> None:
    pairs = [
        _pair("P2", "val", 0, 2),
        _pair("P1", "train", 0, 1),
        _pair("P1", "val", 1, 2),
        _pair("P1", "val", 0, 1),
    ]

    selected = all_validation_pairs(pairs)

    assert [pair.pair_id for pair in selected] == ["P1:T0->T1", "P1:T1->T2", "P2:T0->T2"]
    assert all_validation_pairs(pairs, limit=2) == selected[:2]


def test_metric_contract_uses_regions_and_fixed_window() -> None:
    target = np.zeros((13, 13, 13), dtype=np.float32)
    prediction = np.ones_like(target)
    foreground = np.ones_like(target, dtype=bool)
    tumor = np.zeros_like(target, dtype=bool)
    tumor[6, 6, 6] = True
    regions = region_masks(foreground, foreground, tumor, tumor)

    metrics = comparison_metrics(
        prediction, target, regions, valid_ssim_centers(foreground)
    )

    assert metrics["common_foreground"]["mae_unclipped"] == 1.0
    assert metrics["common_foreground"]["rmse_unclipped"] == 1.0
    assert metrics["target_tumor"]["voxel_count"] == 1
    assert metrics["tumor_union_neighborhood"]["voxel_count"] > 1
    assert math.isfinite(metrics["common_foreground"]["psnr_windowed_db"])
    assert math.isfinite(metrics["common_foreground"]["ssim_3d_windowed"])


def test_describe_values_reports_sample_dispersion_without_ci() -> None:
    summary = describe_values([1.0, 2.0, 3.0])

    assert summary["count"] == 3
    assert summary["mean"] == 2.0
    assert summary["sample_variance"] == 1.0
    assert summary["sample_sd"] == 1.0
    assert "ci95_low" not in summary
    assert describe_values([math.nan])["mean"] is None


def test_patient_aggregation_macro_averages_available_endpoints() -> None:
    rows = []
    for value in (1.0, 3.0):
        rows.append(
            {
                "patient_id": "P1",
                "comparison": "biflow_prediction",
                "region": "common_foreground",
                "voxel_count": 10,
                "ssim_center_count": 5,
                **{metric: value for metric in (
                    "mae_unclipped",
                    "rmse_unclipped",
                    "psnr_windowed_db",
                    "ssim_3d_windowed",
                )},
            }
        )

    aggregated = aggregate_patient_rows(rows)

    assert len(aggregated) == 1
    assert aggregated[0]["case_count"] == 2
    assert aggregated[0]["voxel_count"] == 20
    assert aggregated[0]["mae_unclipped"] == 2.0


def test_adjacent_transition_contract_excludes_long_range_pairs() -> None:
    assert ADJACENT_TRANSITIONS == ("T0->T1", "T1->T2", "T2->T3")
    assert "T0->T2" not in ADJACENT_TRANSITIONS
    assert "T0->T3" not in ADJACENT_TRANSITIONS
    assert "T1->T3" not in ADJACENT_TRANSITIONS


def test_adjacent_selection_preserves_full_cohort_order() -> None:
    pairs = [
        _pair("P1", "val", 0, 1),
        _pair("P1", "val", 0, 2),
        _pair("P1", "val", 1, 2),
        _pair("P2", "val", 0, 3),
        _pair("P2", "val", 2, 3),
    ]

    selected = select_validation_pairs(pairs, pair_mode="adjacent")

    assert [pair.pair_id for pair in selected] == [
        "P1:T0->T1",
        "P1:T1->T2",
        "P2:T2->T3",
    ]
