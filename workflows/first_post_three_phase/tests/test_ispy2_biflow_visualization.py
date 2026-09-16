from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest
import torch

from mewm_ispy2.ispy2_biflow_visualization import (
    BiFlowVisualizationCase,
    _draw_view,
    _mask_centroid,
    _maximum_axial_centroid,
    _plane,
    select_visualization_pairs,
)


def _candidate(patient: str, split: str, transition: str):
    source, target = (int(value[1]) for value in transition.split("->"))
    return SimpleNamespace(
        patient_id=patient,
        split=split,
        transition_type=transition,
        source_stage=source,
        target_stage=target,
    )


def test_visualization_selection_is_stratified_and_patient_distinct() -> None:
    pairs = [
        _candidate("P0", "train", "T0->T1"),
        *[
            _candidate(f"P{index}", "val", transition)
            for index, transition in enumerate(
                ("T0->T1", "T0->T2", "T0->T3", "T1->T2", "T2->T3"),
                start=1,
            )
        ],
    ]

    selected = select_visualization_pairs(pairs, 5)

    assert [pair.transition_type for pair in selected] == [
        "T0->T1",
        "T0->T2",
        "T0->T3",
        "T1->T2",
        "T2->T3",
    ]
    assert len({pair.patient_id for pair in selected}) == 5
    assert all(pair.split == "val" for pair in selected)


def test_visualization_selection_rejects_impossible_case_count() -> None:
    pairs = [_candidate("P1", "val", "T0->T1")]
    with pytest.raises(ValueError, match="distinct validation patients"):
        select_visualization_pairs(pairs, 2)


def test_mask_centers_and_orthogonal_planes_are_zyx_consistent() -> None:
    mask = torch.zeros(1, 4, 5, 6, dtype=torch.uint8)
    mask[0, 2, 1:4, 3:5] = 1
    volume = torch.arange(4 * 5 * 6).reshape(1, 4, 5, 6)

    assert _maximum_axial_centroid(mask) == (2, 2, 4)
    assert _mask_centroid(mask) == (2, 2, 4)
    assert _plane(volume, (2, 2, 4), "axial").shape == (5, 6)
    assert _plane(volume, (2, 2, 4), "coronal").shape == (4, 6)
    assert _plane(volume, (2, 2, 4), "sagittal").shape == (4, 5)


def test_clean_draw_view_omits_mask_contours() -> None:
    image = torch.linspace(-1.0, 1.0, 12 * 13 * 14).reshape(1, 12, 13, 14)
    mask = torch.zeros_like(image, dtype=torch.uint8)
    mask[:, 4:8, 5:9, 6:10] = 1
    case = BiFlowVisualizationCase(
        pair=SimpleNamespace(),
        seed=2026,
        source=image,
        target=image + 0.1,
        target_reconstruction=image + 0.05,
        prediction=image - 0.1,
        source_mask=mask,
        target_mask=mask,
        common_foreground=torch.ones_like(mask, dtype=torch.bool),
        source_max_zyx=(6, 6, 7),
        target_max_zyx=(6, 6, 7),
        target_centroid_zyx=(6, 6, 7),
        metrics={},
    )
    figure, axes = plt.subplots(1, 5)
    _draw_view(
        axes,
        case,
        center=(6, 6, 7),
        plane="axial",
        row_label="clean",
        image_window=(-1.0, 1.0),
        error_max=2.0,
        show_titles=True,
        show_contours=False,
    )
    try:
        assert all(not axis.collections for axis in axes)
    finally:
        plt.close(figure)
