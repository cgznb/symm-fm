"""Minimal torchrun setup with patient-pair sampler and rank-aware seeds."""

from __future__ import annotations

from dataclasses import dataclass
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Sized

import torch
from torch.utils.data import DistributedSampler, Sampler


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-process DDP requires CUDA for this 3D training pipeline")
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            initialized_here = True
        device = torch.device("cuda", local_rank)
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.init()
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")
    return DistributedContext(rank, local_rank, world_size, device, initialized_here)


def distributed_sampler(
    dataset: Sized,
    context: DistributedContext,
    *,
    shuffle: bool,
    seed: int,
) -> DistributedSampler | None:
    """Use the same epoch-addressable sampler for one or many processes."""

    if context.world_size > 1 and len(dataset) < context.world_size:
        raise ValueError("DDP dataset must contain at least one item per rank")

    return DistributedSampler(
        dataset,  # type: ignore[arg-type]
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
        seed=int(seed),
        # Equal per-rank lengths are required by DDP. Dropping the rotating
        # permutation tail avoids duplicating patient pairs across ranks.
        drop_last=context.world_size > 1,
    )


class PatientBalancedDistributedSampler(Sampler[int]):
    """Sample pairs with equal expected mass per patient, then shard by rank."""

    def __init__(
        self,
        patient_ids: Sequence[str],
        context: DistributedContext,
        *,
        seed: int,
    ) -> None:
        if not patient_ids or any(not str(value).strip() for value in patient_ids):
            raise ValueError("patient-balanced sampling requires non-empty patient IDs")
        if len(patient_ids) < context.world_size:
            raise ValueError("DDP dataset must contain at least one item per rank")
        self.patient_ids = tuple(str(value).strip() for value in patient_ids)
        self.rank = context.rank
        self.world_size = context.world_size
        self.seed = int(seed)
        self.epoch = 0
        self.global_size = (
            len(self.patient_ids)
            if self.world_size == 1
            else (len(self.patient_ids) // self.world_size) * self.world_size
        )
        self.num_samples = self.global_size // self.world_size
        counts = Counter(self.patient_ids)
        self.weights = torch.tensor(
            [1.0 / counts[patient_id] for patient_id in self.patient_ids],
            dtype=torch.float64,
        )

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.global_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(global_indices[self.rank : self.global_size : self.world_size])


def patient_balanced_sampler(
    records: Sequence[Mapping[str, Any]],
    context: DistributedContext,
    *,
    seed: int,
) -> PatientBalancedDistributedSampler:
    """Create the patient-balanced sampler from pair-manifest records."""

    patient_ids = [str(record.get("patient_id", "")).strip() for record in records]
    return PatientBalancedDistributedSampler(patient_ids, context, seed=seed)


def rank_seed(base_seed: int, context: DistributedContext) -> int:
    return int(base_seed) + 100_003 * context.rank


def finalize_distributed(context: DistributedContext) -> None:
    if context.initialized_here and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
