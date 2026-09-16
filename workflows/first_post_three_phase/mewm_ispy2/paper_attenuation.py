from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .paper_contracts import (
    AttenuationLevelConfig,
    PaperRuntimeContract,
    default_paper_runtime_contract,
)


@dataclass(frozen=True)
class AttenuationResult:
    source: torch.Tensor
    soft_mask: torch.Tensor
    level: int


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius == 0:
        return mask
    width = 2 * radius + 1
    padded = F.pad(mask, (radius,) * 6, value=0.0)
    return F.max_pool3d(padded, width, stride=1)


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius == 0:
        return mask
    width = 2 * radius + 1
    padded_complement = F.pad(1.0 - mask, (radius,) * 6, value=1.0)
    return 1.0 - F.max_pool3d(padded_complement, width, stride=1)


def _apply_morphology(
    mask: torch.Tensor, morphology: str, radius: int
) -> torch.Tensor:
    if morphology == "identity" and radius == 0:
        return mask
    if morphology == "erosion":
        return _erode(mask, radius)
    if morphology == "opening":
        return _dilate(_erode(mask, radius), radius)
    raise ValueError(f"unsupported morphology: {morphology!r} with radius {radius}")


class MorphoGaussianAttenuator(nn.Module):
    def __init__(self, contract: PaperRuntimeContract | None = None) -> None:
        super().__init__()
        if contract is None:
            contract = default_paper_runtime_contract()
        if not isinstance(contract, PaperRuntimeContract):
            raise TypeError("contract must be a PaperRuntimeContract")

        expected = default_paper_runtime_contract()
        self._validate_attenuation_contract(contract, expected)

        self.contract = contract
        self._levels = {config.level: config for config in contract.attenuation_levels}

    @staticmethod
    def _validate_attenuation_contract(
        contract: PaperRuntimeContract, expected: PaperRuntimeContract
    ) -> None:
        if type(contract.attenuation_policy) is not str:
            raise TypeError("attenuation_policy must be an exact string")
        if contract.attenuation_policy != expected.attenuation_policy:
            raise ValueError("attenuation policy does not match the paper contract")

        for config in contract.attenuation_levels:
            if type(config) is not AttenuationLevelConfig:
                raise TypeError("attenuation levels must use exact level configs")
            if type(config.level) is not int:
                raise TypeError("attenuation level must be an exact integer")
            if not 1 <= config.level <= 4:
                raise ValueError("attenuation level must be in 1..4")
            if type(config.morphology) is not str:
                raise TypeError("attenuation morphology must be an exact string")
            if config.morphology not in {"identity", "opening", "erosion"}:
                raise ValueError("unsupported attenuation morphology")
            if type(config.radius) is not int:
                raise TypeError("attenuation radius must be an exact integer")
            if config.radius < 0:
                raise ValueError("attenuation radius must be non-negative")

            for field_name in ("sigma", "exponent", "decay"):
                value = getattr(config, field_name)
                if type(value) is not float:
                    raise TypeError(f"attenuation {field_name} must be an exact float")
                if not math.isfinite(value):
                    raise ValueError(f"attenuation {field_name} must be finite")
            if config.sigma <= 0:
                raise ValueError("attenuation sigma must be positive")
            if config.exponent <= 0:
                raise ValueError("attenuation exponent must be positive")
            if not 0 <= config.decay <= 1:
                raise ValueError("attenuation decay must be in [0, 1]")

        if contract.attenuation_levels != expected.attenuation_levels:
            raise ValueError("attenuation levels do not match the exact four-level contract")

    def sample_level(self, generator: torch.Generator | None = None) -> int:
        device = generator.device if generator is not None else torch.device("cpu")
        return int(
            torch.randint(1, 5, (), generator=generator, device=device).item()
        )

    def forward(
        self, source: torch.Tensor, source_mask: torch.Tensor, level: int
    ) -> AttenuationResult:
        config = self._validate_level(level)
        self._validate_tensors(source, source_mask)

        mask = source_mask.to(device=source.device, dtype=source.dtype)
        processed_mask = _apply_morphology(
            mask, config.morphology, config.radius
        )
        kernel = self._gaussian_kernel(config, source)
        soft_mask = F.conv3d(processed_mask, kernel, padding=5)
        soft_mask = soft_mask.clamp(0.0, 1.0).pow(config.exponent)
        soft_mask = soft_mask.to(dtype=source.dtype)
        attenuated = (
            source * (1.0 - soft_mask)
            + config.decay * source * soft_mask
        )

        if not bool(torch.isfinite(soft_mask).all()):
            raise ValueError("attenuation produced a non-finite soft mask")
        if not bool(torch.isfinite(attenuated).all()):
            raise ValueError("attenuation produced a non-finite source")
        return AttenuationResult(
            source=attenuated,
            soft_mask=soft_mask,
            level=config.level,
        )

    def _validate_level(self, level: int) -> AttenuationLevelConfig:
        if type(level) is not int:
            raise TypeError("level must be an integer and not bool")
        if level not in self._levels:
            raise ValueError(f"unknown attenuation level: {level}")
        return self._levels[level]

    @staticmethod
    def _validate_tensors(source: torch.Tensor, source_mask: torch.Tensor) -> None:
        if not isinstance(source, torch.Tensor):
            raise TypeError("source must be a tensor")
        if not isinstance(source_mask, torch.Tensor):
            raise TypeError("source_mask must be a tensor")
        if source.ndim != 5 or source.shape[1] != 1:
            raise ValueError("source must have shape [B, 1, D, H, W]")
        if source.shape[0] <= 0 or any(size <= 0 for size in source.shape[2:]):
            raise ValueError("source batch and spatial dimensions must be positive")
        if source_mask.ndim != 5 or source_mask.shape[1] != 1:
            raise ValueError("source_mask must have shape [B, 1, D, H, W]")
        if source_mask.shape != source.shape:
            raise ValueError("source and source_mask must have exactly aligned shapes")
        if not source.is_floating_point():
            raise TypeError("source must have a floating dtype")
        if not bool(torch.isfinite(source).all()):
            raise ValueError("source must contain only finite values")
        if source_mask.dtype != torch.bool and not source_mask.is_floating_point():
            raise TypeError("source_mask must have a bool or floating dtype")
        if source_mask.requires_grad:
            raise ValueError("source_mask.requires_grad must be False")
        if not bool(torch.isfinite(source_mask).all()):
            raise ValueError("source_mask must contain only finite values")
        if not bool(((source_mask == 0) | (source_mask == 1)).all()):
            raise ValueError("source_mask must be exactly binary")

    @staticmethod
    def _gaussian_kernel(
        config: AttenuationLevelConfig, source: torch.Tensor
    ) -> torch.Tensor:
        coordinate = torch.arange(
            -5, 6, device=source.device, dtype=source.dtype
        )
        squared = coordinate.square()
        squared_distance = (
            squared[:, None, None]
            + squared[None, :, None]
            + squared[None, None, :]
        )
        kernel = torch.exp(-squared_distance / (2.0 * config.sigma**2))
        kernel = kernel / kernel.sum()
        return kernel[None, None]
