from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ispy2_symmflow.training.distributed import (
    DistributedContext,
    distributed_sampler,
    patient_balanced_sampler,
    rank_seed,
)


def test_rank_seeds_are_stable_and_distinct() -> None:
    cpu = torch.device("cpu")
    rank0 = DistributedContext(0, 0, 2, cpu, False)
    rank1 = DistributedContext(1, 1, 2, cpu, False)
    assert rank_seed(11, rank0) == 11
    assert rank_seed(11, rank1) != rank_seed(11, rank0)
    assert rank_seed(11, rank1) == rank_seed(11, rank1)


def test_distributed_sampler_partitions_one_shared_permutation() -> None:
    dataset = list(range(12))
    cpu = torch.device("cpu")
    rank0 = DistributedContext(0, 0, 2, cpu, False)
    rank1 = DistributedContext(1, 1, 2, cpu, False)
    left = distributed_sampler(dataset, rank0, shuffle=True, seed=91)
    right = distributed_sampler(dataset, rank1, shuffle=True, seed=91)
    assert left is not None and right is not None
    left.set_epoch(3)
    right.set_epoch(3)
    left_indices, right_indices = list(left), list(right)
    assert not set(left_indices).intersection(right_indices)
    assert set(left_indices).union(right_indices) == set(range(12))


def test_nondivisible_ddp_sampler_drops_tail_without_cross_rank_duplicates() -> None:
    dataset = list(range(11))
    cpu = torch.device("cpu")
    rank0 = DistributedContext(0, 0, 2, cpu, False)
    rank1 = DistributedContext(1, 1, 2, cpu, False)
    left = distributed_sampler(dataset, rank0, shuffle=True, seed=7)
    right = distributed_sampler(dataset, rank1, shuffle=True, seed=7)
    assert left is not None and right is not None
    left.set_epoch(1)
    right.set_epoch(1)
    left_indices, right_indices = list(left), list(right)
    assert len(left_indices) == len(right_indices) == 5
    assert not set(left_indices).intersection(right_indices)


def test_ddp_sampler_rejects_fewer_items_than_ranks() -> None:
    context = DistributedContext(0, 0, 2, torch.device("cpu"), False)
    with pytest.raises(ValueError, match="one item per rank"):
        distributed_sampler([0], context, shuffle=True, seed=7)


def test_patient_balanced_sampler_equalizes_expected_patient_mass() -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    records = [
        {"patient_id": "many"},
        {"patient_id": "many"},
        {"patient_id": "many"},
        {"patient_id": "single"},
    ]
    sampler = patient_balanced_sampler(records, context, seed=19)
    assert sampler.weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1])
    sampler.set_epoch(2)
    first = list(sampler)
    sampler.set_epoch(2)
    assert list(sampler) == first


def test_patient_balanced_sampler_shards_one_global_draw() -> None:
    records = [{"patient_id": f"p{index // 2}"} for index in range(12)]
    left_context = DistributedContext(0, 0, 2, torch.device("cpu"), False)
    right_context = DistributedContext(1, 1, 2, torch.device("cpu"), False)
    left = patient_balanced_sampler(records, left_context, seed=23)
    right = patient_balanced_sampler(records, right_context, seed=23)
    left.set_epoch(4)
    right.set_epoch(4)
    generator = torch.Generator().manual_seed(27)
    expected = torch.multinomial(left.weights, 12, replacement=True, generator=generator).tolist()
    assert list(left) == expected[0::2]
    assert list(right) == expected[1::2]
