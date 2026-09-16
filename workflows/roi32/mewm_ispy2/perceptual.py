from __future__ import annotations

import torch
from torch import nn


class UncheckedLPIPSLoss(nn.Module):
    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network.requires_grad_(False).eval()

    @classmethod
    def vgg(cls) -> "UncheckedLPIPSLoss":
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=False
        ).eval()
        network = getattr(metric, "net", None)
        if not isinstance(network, nn.Module):
            raise RuntimeError("TorchMetrics LPIPS does not expose its network")
        return cls(network)

    def train(self, mode: bool = True) -> "UncheckedLPIPSLoss":
        super().train(mode)
        self.network.eval()
        return self

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.network(fake, real, normalize=False)
