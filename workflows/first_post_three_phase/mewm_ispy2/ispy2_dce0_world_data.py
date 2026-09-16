from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.utils.data import Dataset

from .backend import load_transition_records
from .cache import RegisteredStrictAROICache
from .contracts import TransitionRecord
from .ispy2_dce0_world_config import (
    ISPY2_DCE0_WORLD_LATENT_SHAPE,
    ISPY2_DCE0_WORLD_SOURCE_MODALITIES,
    ISPY2_DCE0_WORLD_SOURCE_MRI_SHAPE,
)
from .manifest import VisitRecord


ISPY2_DCE0_WORLD_BUNDLE_CONTRACT_SHA256 = (
    "05e8dff05924c8b78ccaaad08a062fbf5e900b4da118ef0a88f089fb9a743725"
)
ISPY2_DCE0_WORLD_PAIR_COUNT = 4006
ISPY2_DCE0_WORLD_SPLIT_PAIR_COUNTS = {"train": 3468, "val": 538}
ISPY2_DCE0_WORLD_SPLIT_PATIENT_COUNTS = {"train": 765, "val": 102}
ISPY2_DCE0_WORLD_TRANSITION_TYPE_COUNTS = {
    "T0->T1": 837,
    "T0->T2": 698,
    "T0->T3": 581,
    "T1->T2": 698,
    "T1->T3": 581,
    "T2->T3": 611,
}
ISPY2_DCE0_WORLD_SOURCE_VISIT_COUNT = 2146
ISPY2_DCE0_WORLD_ENDPOINT_VISIT_COUNT = 3013
_ADJACENT_TRANSITION = re.compile(r"^T([0-2])->T([1-3])$")


class ContinuousLatentLoader(Protocol):
    def load(self, visit_id: str) -> torch.Tensor: ...


@dataclass(frozen=True)
class ISPY2DCE0WorldPair:
    pair_id: str
    patient_id: str
    split: str
    transition_type: str
    source_visit_id: str
    target_visit_id: str
    source_stage: int
    target_stage: int
    delta_days: int
    source_visit: VisitRecord
    clinical_text: str
    treatment_text: str
    adjacent_transition_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.split not in {"train", "val"}:
            raise ValueError("I-SPY2 DCE0 world pair split is invalid")
        if not 0 <= self.source_stage < self.target_stage <= 3:
            raise ValueError("I-SPY2 DCE0 world pair stages are invalid")
        expected_type = f"T{self.source_stage}->T{self.target_stage}"
        if self.transition_type != expected_type:
            raise ValueError("I-SPY2 DCE0 world pair transition type is invalid")
        if self.pair_id != f"{self.patient_id}:{expected_type}":
            raise ValueError("I-SPY2 DCE0 world pair ID is invalid")
        if self.source_visit_id != f"{self.patient_id}:T{self.source_stage}":
            raise ValueError("I-SPY2 DCE0 world source visit ID is invalid")
        if self.target_visit_id != f"{self.patient_id}:T{self.target_stage}":
            raise ValueError("I-SPY2 DCE0 world target visit ID is invalid")
        if self.source_visit.visit_id != self.source_visit_id:
            raise ValueError("I-SPY2 DCE0 world source visit descriptor is invalid")
        if type(self.delta_days) is not int or self.delta_days <= 0:
            raise ValueError("I-SPY2 DCE0 world delta_days must be positive")
        if not self.clinical_text.strip() or not self.treatment_text.strip():
            raise ValueError("I-SPY2 DCE0 world text condition is empty")
        if len(self.adjacent_transition_ids) != self.target_stage - self.source_stage:
            raise ValueError("I-SPY2 DCE0 world pair chain is incomplete")

    @property
    def source_timepoint(self) -> str:
        return f"T{self.source_stage}"

    @property
    def target_timepoint(self) -> str:
        return f"T{self.target_stage}"


@dataclass(frozen=True)
class ISPY2DCE0WorldPairAudit:
    pair_count: int
    split_pair_counts: dict[str, int]
    split_patient_counts: dict[str, int]
    transition_type_counts: dict[str, int]
    source_visit_count: int
    endpoint_visit_count: int
    adjacent_transition_count: int
    bundle_contract_sha256: str | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def _stage_pair(record: TransitionRecord) -> tuple[int, int]:
    match = _ADJACENT_TRANSITION.fullmatch(record.transition_type)
    if match is None:
        raise ValueError("I-SPY2 DCE0 world requires adjacent Strict-A transitions")
    source_stage, target_stage = (int(value) for value in match.groups())
    if target_stage != source_stage + 1 or record.stage_id != target_stage:
        raise ValueError("I-SPY2 DCE0 world adjacent transition stage is invalid")
    return source_stage, target_stage


def derive_ispy2_dce0_world_pairs(
    records: Sequence[TransitionRecord],
    visits: Mapping[str, VisitRecord],
    *,
    bundle_contract_sha256: str | None = None,
) -> tuple[tuple[ISPY2DCE0WorldPair, ...], ISPY2DCE0WorldPairAudit]:
    """Derive every connected forward endpoint pair from locked adjacent edges."""
    if not records:
        raise ValueError("I-SPY2 DCE0 world adjacent transition set is empty")
    edges_by_patient: dict[str, dict[tuple[int, int], TransitionRecord]] = defaultdict(dict)
    fields_by_patient: dict[str, tuple[str, str, str]] = {}
    for record in records:
        if not isinstance(record, TransitionRecord):
            raise TypeError("I-SPY2 DCE0 world transition record is invalid")
        if record.fold not in {"train", "val"}:
            raise ValueError("I-SPY2 DCE0 world supports train and val folds only")
        source_stage, target_stage = _stage_pair(record)
        expected_source = f"{record.patient_id}:T{source_stage}"
        expected_target = f"{record.patient_id}:T{target_stage}"
        if (
            record.source_visit_id != expected_source
            or record.target_visit_id != expected_target
        ):
            raise ValueError("I-SPY2 DCE0 world transition visit identity is invalid")
        for visit_id in (record.source_visit_id, record.target_visit_id):
            visit = visits.get(visit_id)
            if visit is None or visit.patient_id != record.patient_id:
                raise ValueError("I-SPY2 DCE0 world transition visit is unavailable")
        fields = (record.fold, record.clinical_text, record.action_text)
        if fields_by_patient.setdefault(record.patient_id, fields) != fields:
            raise ValueError(
                "I-SPY2 DCE0 world patient fold or text conditions are inconsistent"
            )
        key = (source_stage, target_stage)
        if key in edges_by_patient[record.patient_id]:
            raise ValueError("I-SPY2 DCE0 world adjacent transition is duplicated")
        edges_by_patient[record.patient_id][key] = record

    pairs: list[ISPY2DCE0WorldPair] = []
    for patient_id in sorted(edges_by_patient):
        edges = edges_by_patient[patient_id]
        split, clinical_text, treatment_text = fields_by_patient[patient_id]
        for source_stage in range(3):
            for target_stage in range(source_stage + 1, 4):
                chain_keys = tuple(
                    (stage, stage + 1) for stage in range(source_stage, target_stage)
                )
                if any(key not in edges for key in chain_keys):
                    continue
                chain = tuple(edges[key] for key in chain_keys)
                source_visit_id = f"{patient_id}:T{source_stage}"
                target_visit_id = f"{patient_id}:T{target_stage}"
                pairs.append(
                    ISPY2DCE0WorldPair(
                        pair_id=f"{patient_id}:T{source_stage}->T{target_stage}",
                        patient_id=patient_id,
                        split=split,
                        transition_type=f"T{source_stage}->T{target_stage}",
                        source_visit_id=source_visit_id,
                        target_visit_id=target_visit_id,
                        source_stage=source_stage,
                        target_stage=target_stage,
                        delta_days=sum(edge.delta_days for edge in chain),
                        source_visit=visits[source_visit_id],
                        clinical_text=clinical_text,
                        treatment_text=treatment_text,
                        adjacent_transition_ids=tuple(
                            edge.transition_id for edge in chain
                        ),
                    )
                )

    split_pair_counts = {
        fold: sum(pair.split == fold for pair in pairs) for fold in ("train", "val")
    }
    split_patient_counts = {
        fold: len({pair.patient_id for pair in pairs if pair.split == fold})
        for fold in ("train", "val")
    }
    transition_counts = Counter(pair.transition_type for pair in pairs)
    source_visits = {pair.source_visit_id for pair in pairs}
    endpoint_visits = {
        visit_id
        for pair in pairs
        for visit_id in (pair.source_visit_id, pair.target_visit_id)
    }
    audit = ISPY2DCE0WorldPairAudit(
        pair_count=len(pairs),
        split_pair_counts=split_pair_counts,
        split_patient_counts=split_patient_counts,
        transition_type_counts={key: transition_counts[key] for key in sorted(transition_counts)},
        source_visit_count=len(source_visits),
        endpoint_visit_count=len(endpoint_visits),
        adjacent_transition_count=len(records),
        bundle_contract_sha256=bundle_contract_sha256,
    )
    return tuple(pairs), audit


def _validate_locked_audit(audit: ISPY2DCE0WorldPairAudit) -> None:
    if (
        audit.bundle_contract_sha256 != ISPY2_DCE0_WORLD_BUNDLE_CONTRACT_SHA256
        or audit.pair_count != ISPY2_DCE0_WORLD_PAIR_COUNT
        or audit.split_pair_counts != ISPY2_DCE0_WORLD_SPLIT_PAIR_COUNTS
        or audit.split_patient_counts != ISPY2_DCE0_WORLD_SPLIT_PATIENT_COUNTS
        or audit.transition_type_counts != ISPY2_DCE0_WORLD_TRANSITION_TYPE_COUNTS
        or audit.source_visit_count != ISPY2_DCE0_WORLD_SOURCE_VISIT_COUNT
        or audit.endpoint_visit_count != ISPY2_DCE0_WORLD_ENDPOINT_VISIT_COUNT
        or audit.adjacent_transition_count != 2146
    ):
        raise ValueError("I-SPY2 DCE0 world locked pair inventory changed")


def build_ispy2_dce0_world_pairs(
    bundle_json: str | Path,
    phase_manifest_csv: str | Path,
) -> tuple[tuple[ISPY2DCE0WorldPair, ...], ISPY2DCE0WorldPairAudit]:
    loaded = load_transition_records(
        bundle_json,
        phase_manifest_csv,
        backend="registered_t0",
    )
    pairs, audit = derive_ispy2_dce0_world_pairs(
        loaded.records,
        loaded.visits,
        bundle_contract_sha256=loaded.bundle_contract_sha256,
    )
    _validate_locked_audit(audit)
    return pairs, audit


class ISPY2DCE0WorldPairDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        pairs: Sequence[ISPY2DCE0WorldPair],
        latent_loader: ContinuousLatentLoader,
        roi_cache: RegisteredStrictAROICache,
        *,
        split: str | None = None,
    ) -> None:
        if split not in (None, "train", "val"):
            raise ValueError("I-SPY2 DCE0 world dataset split is invalid")
        if not hasattr(latent_loader, "load"):
            raise TypeError("I-SPY2 DCE0 world latent loader is invalid")
        if not hasattr(roi_cache, "load_source_mri"):
            raise TypeError("I-SPY2 DCE0 world ROI cache is invalid")
        self.pairs = tuple(pair for pair in pairs if split is None or pair.split == split)
        if any(not isinstance(pair, ISPY2DCE0WorldPair) for pair in self.pairs):
            raise TypeError("I-SPY2 DCE0 world dataset pairs are invalid")
        self.latent_loader = latent_loader
        self.roi_cache = roi_cache

    def __len__(self) -> int:
        return len(self.pairs)

    def _latent(self, visit_id: str) -> torch.Tensor:
        latent = self.latent_loader.load(visit_id)
        if (
            not isinstance(latent, torch.Tensor)
            or not latent.is_floating_point()
            or tuple(latent.shape) != ISPY2_DCE0_WORLD_LATENT_SHAPE
            or not bool(torch.isfinite(latent).all())
        ):
            raise ValueError(f"I-SPY2 DCE0 world latent is invalid: {visit_id}")
        return latent.to(dtype=torch.float32).contiguous().unsqueeze(0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        source_mri = self.roi_cache.load_source_mri(pair.source_visit)
        if (
            not isinstance(source_mri, torch.Tensor)
            or source_mri.dtype != torch.float16
            or tuple(source_mri.shape) != ISPY2_DCE0_WORLD_SOURCE_MRI_SHAPE
            or not bool(torch.isfinite(source_mri).all())
        ):
            raise ValueError("I-SPY2 DCE0 world source MRI is invalid")
        return {
            "source_mri": source_mri.contiguous(),
            "source_latent": self._latent(pair.source_visit_id),
            "target_latent": self._latent(pair.target_visit_id),
            "clinical_text": pair.clinical_text,
            "treatment_text": pair.treatment_text,
            "delta_days": torch.tensor(pair.delta_days, dtype=torch.float32),
            "target_stage": torch.tensor(pair.target_stage, dtype=torch.long),
            "metadata": {
                "pair_id": pair.pair_id,
                "patient_id": pair.patient_id,
                "split": pair.split,
                "transition_type": pair.transition_type,
                "source_visit_id": pair.source_visit_id,
                "target_visit_id": pair.target_visit_id,
                "source_stage": pair.source_stage,
                "target_stage": pair.target_stage,
                "source_modalities": ISPY2_DCE0_WORLD_SOURCE_MODALITIES,
                "adjacent_transition_ids": pair.adjacent_transition_ids,
            },
        }


ISPY2DCE0PairDataset = ISPY2DCE0WorldPairDataset


__all__ = [
    "ContinuousLatentLoader",
    "ISPY2_DCE0_WORLD_BUNDLE_CONTRACT_SHA256",
    "ISPY2_DCE0_WORLD_ENDPOINT_VISIT_COUNT",
    "ISPY2_DCE0_WORLD_PAIR_COUNT",
    "ISPY2_DCE0_WORLD_SOURCE_VISIT_COUNT",
    "ISPY2_DCE0_WORLD_SPLIT_PAIR_COUNTS",
    "ISPY2_DCE0_WORLD_SPLIT_PATIENT_COUNTS",
    "ISPY2_DCE0_WORLD_TRANSITION_TYPE_COUNTS",
    "ISPY2DCE0PairDataset",
    "ISPY2DCE0WorldPair",
    "ISPY2DCE0WorldPairAudit",
    "ISPY2DCE0WorldPairDataset",
    "build_ispy2_dce0_world_pairs",
    "derive_ispy2_dce0_world_pairs",
]
