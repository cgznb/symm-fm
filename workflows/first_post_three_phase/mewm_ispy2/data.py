from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .contracts import DATA_SCHEMA_VERSION, TransitionRecord


@dataclass(frozen=True)
class PreparedVisit:
    visit_id: str
    image: torch.Tensor
    mask: torch.Tensor
    phase_index: int
    n_times: int
    image_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)
    valid_foreground: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.image.ndim != 4 or self.image.shape[0] != 1:
            raise ValueError("prepared image must have shape [1,Z,Y,X]")
        if self.mask.shape != self.image.shape:
            raise ValueError("prepared image and mask shapes must match")
        if self.phase_index != 0 or self.n_times <= 0:
            raise ValueError("prepared visit must contain DCE phase zero")
        if len(self.image_sha256) != 64:
            raise ValueError("prepared image SHA256 is invalid")
        if self.valid_foreground is not None:
            if (
                self.valid_foreground.dtype != torch.uint8
                or self.valid_foreground.shape != self.image.shape
                or not bool(
                    ((self.valid_foreground == 0) | (self.valid_foreground == 1)).all()
                )
            ):
                raise ValueError("prepared valid foreground tensor is invalid")


class DCE0TransitionDataset(torch.utils.data.Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[TransitionRecord],
        *,
        visit_loader: Callable[[str], PreparedVisit],
        pair_loader: Callable[[TransitionRecord], tuple[PreparedVisit, PreparedVisit]] | None = None,
        source_only: bool,
        backend: str = "current",
    ) -> None:
        self.records = tuple(records)
        self.visit_loader = visit_loader
        self.pair_loader = pair_loader
        self.source_only = bool(source_only)
        self.backend = backend

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        target: PreparedVisit | None = None
        if not self.source_only and self.pair_loader is not None:
            source, target = self.pair_loader(record)
        else:
            source = self.visit_loader(record.source_visit_id)
        model_inputs = {
            "source_dce0": source.image.float(),
            "source_mask": source.mask.float(),
            "action_text": record.action_text,
            "clinical_text": record.clinical_text,
            "delta_days": float(record.delta_days),
            "stage_id": record.stage_id,
        }
        metadata = {
            "schema_version": DATA_SCHEMA_VERSION,
            "transition_id": record.transition_id,
            "patient_id": record.patient_id,
            "fold": record.fold,
            "transition_type": record.transition_type,
            "source_visit_id": source.visit_id,
            "source_phase_index": source.phase_index,
            "source_n_times": source.n_times,
            "source_image_sha256": source.image_sha256,
            "data_backend": self.backend,
            "source_preprocessing": source.metadata,
        }
        item: dict[str, Any] = {"model_inputs": model_inputs, "metadata": metadata}
        if self.source_only:
            return item

        if target is None:
            target = self.visit_loader(record.target_visit_id)
        item["supervision"] = {
            "target_dce0": target.image.float(),
            "target_mask": target.mask.float(),
        }
        metadata.update(
            {
                "target_visit_id": target.visit_id,
                "target_phase_index": target.phase_index,
                "target_n_times": target.n_times,
                "target_image_sha256": target.image_sha256,
                "target_preprocessing": target.metadata,
            }
        )
        return item


def _collate_section(items: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if all(isinstance(value, torch.Tensor) for value in values):
            result[key] = torch.stack(values)
        elif all(isinstance(value, (int, float)) for value in values):
            dtype = torch.long if all(isinstance(value, int) for value in values) else torch.float32
            result[key] = torch.tensor(values, dtype=dtype)
        else:
            result[key] = values
    return result


def collate_transitions(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty transition batch")
    output = {
        "model_inputs": _collate_section([item["model_inputs"] for item in items]),
        "metadata": [item["metadata"] for item in items],
    }
    if "supervision" in items[0]:
        if not all("supervision" in item for item in items):
            raise ValueError("cannot mix source-only and supervised samples")
        output["supervision"] = _collate_section(
            [item["supervision"] for item in items]
        )
    return output
