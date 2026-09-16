from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DynUNet


@dataclass(frozen=True)
class DynUNetConfig:
    spatial_size: tuple[int, int, int] = (128, 128, 128)
    filters: tuple[int, ...] = (32, 64, 128, 256, 320)
    strides: tuple[int, ...] = (1, 2, 2, 2, 2)
    kernel_size: Sequence[Any] = (3, 3, 3, 3, 3)
    upsample_kernel_size: Sequence[Any] = (2, 2, 2, 2)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if len(self.filters) != len(self.strides):
            raise ValueError("DynUNet filters and strides must have equal lengths")
        if len(self.kernel_size) != len(self.strides):
            raise ValueError("DynUNet needs one kernel per resolution")
        if len(self.upsample_kernel_size) != len(self.strides) - 1:
            raise ValueError("DynUNet needs one fewer upsampling kernels than strides")


def build_dynunet(config: DynUNetConfig = DynUNetConfig()) -> DynUNet:
    return DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=config.kernel_size,
        strides=config.strides,
        upsample_kernel_size=config.upsample_kernel_size,
        filters=config.filters,
        dropout=config.dropout,
        norm_name="instance",
        deep_supervision=False,
        res_block=True,
    )


def soft_dice_score(
    probabilities: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    dimensions = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions)
    return ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


class DynUNetSegmentationSystem(pl.LightningModule):
    def __init__(self, model: nn.Module, *, learning_rate: float = 1e-4) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

    def compute_loss(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        logits = self(image)
        target = mask.float()
        probabilities = logits.sigmoid()
        dice_score = soft_dice_score(probabilities, target)
        dice_loss = 1.0 - dice_score
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return {
            "dice": dice_loss,
            "bce": bce,
            "total": dice_loss + bce,
            "dice_score": dice_score,
        }

    @staticmethod
    def _image_mask(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if "image" in batch and "mask" in batch:
            return batch["image"], batch["mask"]
        if "supervision" in batch:
            return (
                batch["supervision"]["target_dce0"],
                batch["supervision"]["target_mask"],
            )
        return (
            batch["model_inputs"]["source_dce0"],
            batch["model_inputs"]["source_mask"],
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        losses = self.compute_loss(*self._image_mask(batch))
        self.log_dict(
            {f"train/{key}": value for key, value in losses.items()},
            on_step=True,
            on_epoch=True,
        )
        return losses["total"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        losses = self.compute_loss(*self._image_mask(batch))
        self.log("val/dice", losses["dice_score"], prog_bar=True, on_epoch=True)
        self.log("val/loss", losses["total"], on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


@dataclass(frozen=True)
class GeneratedSegmentation:
    samples: torch.Tensor
    sample_probabilities: torch.Tensor
    sample_masks: torch.Tensor
    mean_dce: torch.Tensor
    mean_probability: torch.Tensor
    entropy: torch.Tensor


@torch.inference_mode()
def segment_generated_samples(
    model: nn.Module,
    generated_samples: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> GeneratedSegmentation:
    if generated_samples.ndim != 5 or generated_samples.shape[1] != 1:
        raise ValueError("generated samples must have shape [S,1,Z,Y,X]")
    if generated_samples.shape[0] != 8:
        raise ValueError("inference contract requires exactly eight generated samples")
    logits = model(generated_samples)
    probabilities = logits.sigmoid()
    masks = probabilities >= threshold
    mean_probability = probabilities.mean(dim=0)
    clamped = mean_probability.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(clamped * clamped.log() + (1.0 - clamped) * (1.0 - clamped).log())
    return GeneratedSegmentation(
        samples=generated_samples,
        sample_probabilities=probabilities,
        sample_masks=masks,
        mean_dce=generated_samples.mean(dim=0),
        mean_probability=mean_probability,
        entropy=entropy,
    )
