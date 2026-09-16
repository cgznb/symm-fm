"""Model/runtime measurements used in completion reports and profiling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ParameterCount:
    total: int
    trainable: int


def count_parameters(model: nn.Module) -> ParameterCount:
    return ParameterCount(
        total=sum(parameter.numel() for parameter in model.parameters()),
        trainable=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    )


def cuda_peak_memory_mb(device: torch.device | str = "cuda") -> float:
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA peak memory requires an available CUDA device")
    return torch.cuda.max_memory_allocated(selected) / (1024**2)
