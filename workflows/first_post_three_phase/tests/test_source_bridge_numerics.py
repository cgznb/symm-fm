from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from mewm_ispy2.ispy2_biflow_backbone import (
    ISPY2BiFlowControlNet,
    ISPY2ConditionalBiFlowNet,
    ISPY2ControlledBiFlowNet,
    ORIGINAL_ISPY2_BIFLOW_PRESET,
    enable_local_dit_fp32,
)


@pytest.mark.parametrize("branch_name,method", [
    ("backbone", "_local_features"),
    ("controlnet", "_local_encoder_features"),
])
def test_chunked_local_fp32_matches_reference_forward_backward_and_inference(branch_name, method):
    torch.manual_seed(4)
    preset = replace(
        ORIGINAL_ISPY2_BIFLOW_PRESET, dim=24, dim_mults=(1, 1, 2),
        sub_volume_size=(2, 2, 2), dit_heads=4, attention_heads=2, norm_groups=8,
        attention_levels=(False, False, False), downsample_after=(False, True, False),
        upsample_after=(True, False, False), context_pool_heads=4,
    )
    backbone = ISPY2ConditionalBiFlowNet(
        input_channels=2, output_channels=2, context_dim=12, preset=preset,
    )
    with torch.no_grad():
        for name, parameter in backbone.named_parameters():
            if name.startswith("local_"):
                parameter.normal_(0, 0.05)
    reference = ISPY2ControlledBiFlowNet(backbone, ISPY2BiFlowControlNet(backbone))
    stable = copy.deepcopy(reference)
    original_keys = set(stable.state_dict())
    enable_local_dit_fp32(stable, subvolume_batch=3)
    assert set(stable.state_dict()) == original_keys
    ref_branch, stable_branch = getattr(reference, branch_name), getattr(stable, branch_name)
    value = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
    condition = torch.randn(1, backbone.condition_dim, requires_grad=True) * 20
    stable_value = value.detach().clone().requires_grad_(True)
    stable_condition = condition.detach().clone().requires_grad_(True)
    expected = getattr(ref_branch, method)(value, condition)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = getattr(stable_branch, method)(stable_value, stable_condition)
    for output, target in zip(actual, expected, strict=True):
        assert output.dtype == torch.float32
        torch.testing.assert_close(output, target, atol=1e-5, rtol=1e-5)
    sum(item.square().mean() for item in expected).backward()
    sum(item.square().mean() for item in actual).backward()
    torch.testing.assert_close(stable_value.grad, value.grad, atol=1e-5, rtol=1e-5)
    for (name, parameter), (_, target) in zip(
        stable_branch.named_parameters(), ref_branch.named_parameters(), strict=True
    ):
        if target.grad is not None:
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, target.grad, atol=1e-5, rtol=1e-5)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        inference = getattr(stable_branch, method)(stable_value, stable_condition)
    for output, target in zip(inference, expected, strict=True):
        torch.testing.assert_close(output, target, atol=1e-5, rtol=1e-5)
