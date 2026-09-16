"""Random-state handling used by training checkpoints and tests."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and torch without importing torch for data-only CLI use."""

    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch
    except ImportError:
        return state
    state["torch_cpu"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    try:
        import torch
    except ImportError:
        return
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        available = torch.cuda.device_count()
        saved = [value.detach().cpu() for value in state["torch_cuda"]]
        if len(saved) != available:
            raise ValueError(
                f"checkpoint stores {len(saved)} CUDA RNG states but {available} devices are visible"
            )
        torch.cuda.set_rng_state_all(saved)
