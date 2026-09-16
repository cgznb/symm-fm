"""Device-preserving exponential moving average for model parameters."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import math
from typing import Iterator

import torch
from torch import Tensor, nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not math.isfinite(float(decay)) or not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must satisfy 0 <= decay < 1")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow: OrderedDict[str, Tensor] = OrderedDict(
            (name, parameter.detach().clone())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        if set(parameters).intersection(self.shadow) != set(self.shadow):
            raise ValueError("EMA parameter names no longer match the model")
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, average in self.shadow.items():
            average.lerp_(parameters[name].detach(), 1.0 - decay)

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        incoming = state["shadow"]
        if not isinstance(incoming, dict):
            raise TypeError("EMA shadow state must be a mapping")
        if set(incoming) != set(self.shadow):
            raise ValueError("EMA checkpoint parameters do not match model")
        for name, value in incoming.items():
            self.shadow[name].copy_(value)

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        parameters = dict(model.named_parameters())
        originals = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            for name, average in self.shadow.items():
                parameters[name].data.copy_(average)
            yield
        finally:
            for name, original in originals.items():
                parameters[name].data.copy_(original)
