from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from .contracts import TransitionRecord
from .data import DCE0TransitionDataset, PreparedVisit, collate_transitions
from .paper_conditioning import (
    ARM_TO_COMPONENTS,
    StructuredClinicalFields,
    parse_action_text,
    parse_clinical_text,
)


_PAPER_SUPERVISED_ITEM_KEYS = frozenset(
    {"model_inputs", "metadata", "supervision"}
)
PAPER_CCL_MODEL_INPUT_KEYS = frozenset(
    {
        "source_dce0",
        "source_mask",
        "action_text",
        "clinical_text",
        "delta_days",
        "stage_id",
    }
)


@dataclass(frozen=True)
class CCLLossResult:
    loss: torch.Tensor
    positive_similarity: torch.Tensor
    negative_similarity: torch.Tensor


@dataclass(frozen=True)
class CCLSelection:
    anchor_transition_id: str
    positive_transition_id: str | None
    negative_action_texts: Sequence[str]
    valid: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.anchor_transition_id, str)
            or not self.anchor_transition_id
        ):
            raise ValueError("anchor transition ID must be a nonempty string")
        if type(self.valid) is not bool:
            raise TypeError("CCL validity must be a bool")
        if type(self.negative_action_texts) is not tuple:
            raise TypeError("negative action texts must be a tuple")
        if not all(
            isinstance(action, str) and action for action in self.negative_action_texts
        ):
            raise ValueError("negative action texts must be nonempty strings")
        if self.valid:
            if (
                not isinstance(self.positive_transition_id, str)
                or not self.positive_transition_id
            ):
                raise ValueError("valid CCL selection requires a positive transition")
            if len(self.negative_action_texts) != 2 or len(
                set(self.negative_action_texts)
            ) != 2:
                raise ValueError("valid CCL selection requires two distinct negatives")
        elif self.positive_transition_id is not None or self.negative_action_texts:
            raise ValueError(
                "invalid CCL selection cannot contain positives or negatives"
            )


PositiveKey = tuple[str, str, str, int, int, int]


def _positive_key(
    record: TransitionRecord, clinical: StructuredClinicalFields
) -> PositiveKey:
    return (
        record.fold,
        record.action_text,
        record.transition_type,
        clinical.hr,
        clinical.her2,
        clinical.mp,
    )


def _sha256_integer(payload: str) -> int:
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")


class CCLPairIndex:
    def __init__(self, records: Iterable[TransitionRecord], *, seed: int) -> None:
        if type(seed) is not int:
            raise TypeError("seed must be an exact integer")
        materialized = tuple(records)
        if not materialized:
            raise ValueError("CCL pair index cannot use an empty record sequence")
        if not all(isinstance(record, TransitionRecord) for record in materialized):
            raise TypeError("CCL pair index records must be TransitionRecord values")

        records_by_id: dict[str, TransitionRecord] = {}
        clinical_by_id: dict[str, StructuredClinicalFields] = {}
        component_sets: dict[str, frozenset[str]] = {}
        grouped_ids: dict[PositiveKey, list[str]] = defaultdict(list)
        for record in materialized:
            if record.transition_id in records_by_id:
                raise ValueError(f"duplicate transition ID: {record.transition_id}")
            action = parse_action_text(record.action_text)
            clinical = parse_clinical_text(record.clinical_text)
            records_by_id[record.transition_id] = record
            clinical_by_id[record.transition_id] = clinical
            component_sets[record.transition_id] = frozenset(action.components)
            grouped_ids[_positive_key(record, clinical)].append(record.transition_id)

        self.seed = seed
        self.records = materialized
        self.transition_ids = frozenset(records_by_id)
        self._records_by_id = records_by_id
        self._clinical_by_id = clinical_by_id
        self._component_sets = component_sets
        self._positive_candidates: dict[str, tuple[str, ...]] = {}
        for transition_id, record in records_by_id.items():
            key = _positive_key(record, clinical_by_id[transition_id])
            self._positive_candidates[transition_id] = tuple(
                sorted(
                    candidate_id
                    for candidate_id in grouped_ids[key]
                    if records_by_id[candidate_id].patient_id != record.patient_id
                )
            )

    def select(self, transition_id: str, *, epoch: int) -> CCLSelection:
        if type(epoch) is not int:
            raise TypeError("epoch must be an exact integer")
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        if transition_id not in self._records_by_id:
            raise KeyError(
                f"transition ID is missing from CCL pair index: {transition_id}"
            )

        candidates = self._positive_candidates[transition_id]
        if not candidates:
            return CCLSelection(
                anchor_transition_id=transition_id,
                positive_transition_id=None,
                negative_action_texts=(),
                valid=False,
            )
        offset = _sha256_integer(f"{self.seed}:{epoch}:{transition_id}") % len(
            candidates
        )
        record = self._records_by_id[transition_id]
        return CCLSelection(
            anchor_transition_id=transition_id,
            positive_transition_id=candidates[offset],
            negative_action_texts=self._negative_actions(record, epoch=epoch),
            valid=True,
        )

    def _negative_actions(
        self, record: TransitionRecord, *, epoch: int
    ) -> tuple[str, str]:
        anchor_components = self._component_sets[record.transition_id]
        arms_by_distance: dict[Fraction, list[str]] = defaultdict(list)
        for arm in ARM_TO_COMPONENTS:
            if arm == record.action_text:
                continue
            components = frozenset(ARM_TO_COMPONENTS[arm])
            union = anchor_components | components
            distance = Fraction(
                len(union) - len(anchor_components & components), len(union)
            )
            arms_by_distance[distance].append(arm)

        ranked: list[str] = []
        for distance in sorted(arms_by_distance, reverse=True):
            tied = sorted(arms_by_distance[distance])
            tie_id = f"{distance.numerator}/{distance.denominator}"
            rotation = _sha256_integer(
                f"{self.seed}:{epoch}:{record.transition_id}:{tie_id}"
            ) % len(tied)
            ranked.extend(tied[rotation:] + tied[:rotation])
        if len(ranked) < 2:
            raise RuntimeError(
                "canonical treatment vocabulary has fewer than two negatives"
            )
        return ranked[0], ranked[1]


class PaperCCLTransitionDataset:
    def __init__(
        self,
        records: Sequence[TransitionRecord],
        *,
        visit_loader: Callable[[str], PreparedVisit],
        pair_index: CCLPairIndex,
        pair_loader: Callable[
            [TransitionRecord], tuple[PreparedVisit, PreparedVisit]
        ]
        | None = None,
        backend: str = "current",
    ) -> None:
        materialized = tuple(records)
        record_ids = [record.transition_id for record in materialized]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("paper CCL dataset contains duplicate transition IDs")
        if frozenset(record_ids) != pair_index.transition_ids:
            raise ValueError(
                "paper CCL dataset record universe does not match pair index"
            )
        if any(
            record != pair_index._records_by_id[record.transition_id]
            for record in materialized
        ):
            raise ValueError("paper CCL dataset record payload does not match pair index")

        self.records = materialized
        self.pair_index = pair_index
        self.epoch = 0
        self._record_index = {
            record.transition_id: index for index, record in enumerate(materialized)
        }
        self._base = DCE0TransitionDataset(
            materialized,
            visit_loader=visit_loader,
            pair_loader=pair_loader,
            source_only=False,
            backend=backend,
        )

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int:
            raise TypeError("epoch must be an exact integer")
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        anchor = self._base[index]
        if not bool(torch.any(anchor["model_inputs"]["source_mask"] != 0)):
            return {
                "anchor": anchor,
                "positive": None,
                "negative_action_texts": (),
                "ccl_valid": False,
            }
        selection = self.pair_index.select(record.transition_id, epoch=self.epoch)
        if not selection.valid:
            return {
                "anchor": anchor,
                "positive": None,
                "negative_action_texts": (),
                "ccl_valid": False,
            }
        positive_id = selection.positive_transition_id
        if positive_id is None:
            raise RuntimeError("valid CCL selection omitted its positive transition")
        positive = self._base[self._record_index[positive_id]]
        if not bool(torch.any(positive["model_inputs"]["source_mask"] != 0)):
            return {
                "anchor": anchor,
                "positive": None,
                "negative_action_texts": (),
                "ccl_valid": False,
            }
        return {
            "anchor": anchor,
            "positive": positive,
            "negative_action_texts": selection.negative_action_texts,
            "ccl_valid": True,
        }


class PaperAnchorTransitionDataset:
    """Supervised transition dataset for objectives that do not use CCL pairs."""

    def __init__(
        self,
        records: Sequence[TransitionRecord],
        *,
        visit_loader: Callable[[str], PreparedVisit],
        pair_loader: Callable[
            [TransitionRecord], tuple[PreparedVisit, PreparedVisit]
        ]
        | None = None,
        backend: str = "current",
    ) -> None:
        materialized = tuple(records)
        record_ids = [record.transition_id for record in materialized]
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("paper anchor dataset contains duplicate transition IDs")
        self.records = materialized
        self.epoch = 0
        self._base = DCE0TransitionDataset(
            materialized,
            visit_loader=visit_loader,
            pair_loader=pair_loader,
            source_only=False,
            backend=backend,
        )

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int:
            raise TypeError("epoch must be an exact integer")
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"anchor": self._base[index]}


def _validate_supervised_item(item: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TypeError(f"paper CCL {name} must be a transition item")
    if set(item) != _PAPER_SUPERVISED_ITEM_KEYS:
        raise ValueError(f"paper CCL {name} transition structure is malformed")
    model_inputs = item["model_inputs"]
    if not isinstance(model_inputs, dict):
        raise TypeError(f"paper CCL {name} model inputs must be a dictionary")
    if set(model_inputs) != PAPER_CCL_MODEL_INPUT_KEYS:
        raise ValueError(f"paper CCL {name} model inputs are malformed")
    return item


def collate_paper_ccl(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty paper CCL batch")
    anchors = []
    positives = []
    valid_rows = []
    negative_arms: tuple[list[str], list[str]] = ([], [])
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "anchor",
            "positive",
            "negative_action_texts",
            "ccl_valid",
        }:
            raise ValueError("paper CCL item has a malformed nested structure")

        valid = item["ccl_valid"]
        if type(valid) is not bool:
            raise TypeError("paper CCL validity must be a bool")
        anchor = _validate_supervised_item(item["anchor"], name="anchor")
        negatives = item["negative_action_texts"]
        if type(negatives) is not tuple:
            raise TypeError("paper CCL negative action texts must be a tuple")

        if valid:
            positive = _validate_supervised_item(item["positive"], name="positive")
            if (
                len(negatives) != 2
                or len(set(negatives)) != 2
                or not all(isinstance(action, str) and action for action in negatives)
            ):
                raise ValueError(
                    "valid paper CCL item requires two distinct negative strings"
                )
            anchor_action = anchor["model_inputs"].get("action_text")
            if anchor_action in negatives:
                raise ValueError("paper CCL negatives cannot equal the anchor action")
            for action in negatives:
                parse_action_text(action)
            positives.append(positive)
            for arm, action in zip(negative_arms, negatives, strict=True):
                arm.append(action)
        elif item["positive"] is not None or negatives != ():
            raise ValueError(
                "invalid paper CCL item cannot contain positive or negatives"
            )
        anchors.append(anchor)
        valid_rows.append(valid)

    return {
        "anchor": collate_transitions(anchors),
        "positive": collate_transitions(positives) if positives else None,
        "negative_action_texts": tuple(tuple(arm) for arm in negative_arms),
        "ccl_valid_mask": torch.tensor(valid_rows, dtype=torch.bool),
    }


def collate_paper_anchor(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty paper anchor batch")
    anchors = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {"anchor"}:
            raise ValueError("paper anchor item has a malformed nested structure")
        anchors.append(_validate_supervised_item(item["anchor"], name="anchor"))
    return {"anchor": collate_transitions(anchors)}


def _validate_floating_tensor(
    value: torch.Tensor, *, name: str, rank: int
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if any(size <= 0 for size in value.shape):
        raise ValueError(f"{name} dimensions must be positive")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")


def _validate_latent_shape(latent_shape: Sequence[int]) -> tuple[int, int, int]:
    if isinstance(latent_shape, (str, bytes)) or not isinstance(
        latent_shape, Sequence
    ):
        raise TypeError("latent_shape must be a sequence of three integers")
    shape = tuple(latent_shape)
    if len(shape) != 3:
        raise ValueError("latent_shape must contain exactly three dimensions")
    if any(type(size) is not int for size in shape):
        raise TypeError("latent_shape dimensions must be exact integers")
    if any(size <= 0 for size in shape):
        raise ValueError("latent_shape dimensions must be positive")
    return shape


def _downsample_mask_with_occupancy_fallback(
    mask: torch.Tensor, latent_shape: Sequence[int]
) -> torch.Tensor:
    shape = _validate_latent_shape(latent_shape)
    _validate_floating_tensor(mask, name="mask", rank=5)
    if mask.shape[1] != 1:
        raise ValueError("mask must have exactly one channel")

    with torch.autocast(device_type=mask.device.type, enabled=False):
        nearest = F.interpolate(mask, size=shape, mode="nearest")
    source_nonempty = torch.any(mask != 0, dim=(1, 2, 3, 4))
    nearest_nonempty = torch.any(nearest != 0, dim=(1, 2, 3, 4))
    lost = source_nonempty & ~nearest_nonempty
    if not bool(torch.any(lost)):
        return nearest

    lost_indices = lost.nonzero(as_tuple=False).flatten()
    with torch.autocast(device_type=mask.device.type, enabled=False):
        occupancy = F.adaptive_avg_pool3d(
            mask.index_select(0, lost_indices).float(), shape
        )
    occupancy = occupancy.to(dtype=mask.dtype)
    return nearest.index_copy(0, lost_indices, occupancy)


ValidationCheck = tuple[torch.Tensor, type[Exception], str]


def _raise_on_failed_checks(checks: Sequence[ValidationCheck]) -> None:
    failures = torch.stack(tuple(failed.reshape(()) for failed, _, _ in checks))
    sentinel = failures.new_ones(1)
    first_failure = torch.cat((failures, sentinel)).to(torch.uint8).argmax().item()
    if first_failure < len(checks):
        _, error_type, message = checks[first_failure]
        raise error_type(message)


def _input_nonfinite_check(value: torch.Tensor, *, name: str) -> ValidationCheck:
    return (
        torch.any(~torch.isfinite(value)),
        ValueError,
        f"{name} must contain only finite values",
    )


def _computed_nonfinite_check(
    value: torch.Tensor, *, name: str
) -> ValidationCheck:
    return (
        torch.any(~torch.isfinite(value)),
        FloatingPointError,
        f"computed {name} must be finite",
    )


def _stable_work_dtype(values: Sequence[torch.Tensor]) -> torch.dtype:
    dtype = values[0].dtype
    for value in values[1:]:
        dtype = torch.promote_types(dtype, value.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _scale_safe_normalize_rows(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = value.abs().amax(dim=-1, keepdim=True)
    usable_scales = torch.isfinite(scales) & (scales > 0)
    safe_scales = torch.where(usable_scales, scales, torch.ones_like(scales))
    scaled = value / safe_scales
    return F.normalize(scaled, dim=-1, eps=1e-8), scales


@lru_cache(maxsize=16)
def _fixed_gaussian_kernel(
    device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coordinates = torch.arange(-3, 4, device=device, dtype=dtype)
    z, y, x = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    kernel = torch.exp(-(z.square() + y.square() + x.square()) / 2.0)
    return (kernel / kernel.sum()).reshape(1, 1, 7, 7, 7)


def latent_tumor_neighborhood(
    mask: torch.Tensor, latent_shape: Sequence[int]
) -> torch.Tensor:
    shape = _validate_latent_shape(latent_shape)
    _validate_floating_tensor(mask, name="mask", rank=5)
    if mask.shape[1] != 1:
        raise ValueError("mask must have exactly one channel")
    mask_rows = mask.flatten(start_dim=1)
    _raise_on_failed_checks(
        (
            _input_nonfinite_check(mask, name="mask"),
            (torch.any(mask < 0), ValueError, "mask must be nonnegative"),
            (
                torch.any(mask_rows.abs().amax(dim=1) <= 0),
                ValueError,
                "mask must have nonzero support for every batch item",
            ),
        )
    )

    work_dtype = torch.float64 if mask.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=mask.device.type, enabled=False):
        work_mask = mask.to(dtype=work_dtype)
        resized = _downsample_mask_with_occupancy_fallback(work_mask, shape)
        dilated = F.max_pool3d(resized, kernel_size=3, stride=1, padding=1)
        kernel = _fixed_gaussian_kernel(mask.device, work_dtype)
        neighborhood = F.conv3d(dilated, kernel, padding=3)
        maxima = neighborhood.amax(dim=(1, 2, 3, 4), keepdim=True)
        safe_maxima = torch.where(
            torch.isfinite(maxima) & (maxima > 0),
            maxima,
            torch.ones_like(maxima),
        )
        normalized = neighborhood / safe_maxima
        _raise_on_failed_checks(
            (
                (
                    torch.any(
                        dilated.flatten(start_dim=1).abs().amax(dim=1) <= 0
                    ),
                    ValueError,
                    "resized mask must have nonzero latent support",
                ),
                _computed_nonfinite_check(
                    neighborhood, name="latent tumor neighborhood"
                ),
                (
                    torch.any(maxima <= 0),
                    ValueError,
                    "latent tumor neighborhood must have nonzero support",
                ),
                _computed_nonfinite_check(
                    normalized, name="normalized latent tumor neighborhood"
                ),
            )
        )
    return normalized


def masked_latent_representation(
    predicted_x0: torch.Tensor, neighborhood: torch.Tensor
) -> torch.Tensor:
    _validate_floating_tensor(predicted_x0, name="predicted_x0", rank=5)
    _validate_floating_tensor(neighborhood, name="neighborhood", rank=5)
    if neighborhood.shape[1] != 1:
        raise ValueError("neighborhood must have exactly one channel")
    if predicted_x0.shape[0] != neighborhood.shape[0]:
        raise ValueError("predicted_x0 and neighborhood batch sizes must match")
    if predicted_x0.shape[2:] != neighborhood.shape[2:]:
        raise ValueError("predicted_x0 and neighborhood spatial shapes must match")
    if predicted_x0.device != neighborhood.device:
        raise ValueError("predicted_x0 and neighborhood devices must match")
    predicted_rows = predicted_x0.flatten(start_dim=1)
    neighborhood_rows = neighborhood.flatten(start_dim=1)
    _raise_on_failed_checks(
        (
            _input_nonfinite_check(predicted_x0, name="predicted_x0"),
            _input_nonfinite_check(neighborhood, name="neighborhood"),
            (
                torch.any(predicted_rows.abs().amax(dim=1) <= 0),
                ValueError,
                "predicted_x0 must have nonzero support for every batch item",
            ),
            (
                torch.any(neighborhood < 0),
                ValueError,
                "neighborhood must be nonnegative",
            ),
            (
                torch.any(neighborhood_rows.abs().amax(dim=1) <= 0),
                ValueError,
                "neighborhood must have nonzero support for every batch item",
            ),
        )
    )

    work_dtype = _stable_work_dtype((predicted_x0, neighborhood))
    predicted_work = predicted_x0.to(dtype=work_dtype)
    neighborhood_work = neighborhood.to(dtype=work_dtype)
    weighted = predicted_work * neighborhood_work
    weighted_rows = weighted.flatten(start_dim=1)
    representation, scales = _scale_safe_normalize_rows(weighted_rows)
    _raise_on_failed_checks(
        (
            _computed_nonfinite_check(weighted, name="weighted latent values"),
            (
                torch.any(scales <= 0),
                ValueError,
                "weighted latent representation must have nonzero support "
                "for every batch item",
            ),
            _computed_nonfinite_check(
                representation, name="latent representation"
            ),
        )
    )
    return representation


def combo_contrastive_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    *,
    temperature: float,
) -> CCLLossResult:
    if type(temperature) is not float:
        raise TypeError("temperature must be an exact float")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    _validate_floating_tensor(anchor, name="anchor", rank=2)
    _validate_floating_tensor(positive, name="positive", rank=2)
    _validate_floating_tensor(negatives, name="negatives", rank=3)
    if positive.shape != anchor.shape:
        raise ValueError("positive must match anchor batch and feature shape")
    if negatives.shape != (anchor.shape[0], 2, anchor.shape[1]):
        raise ValueError("negatives must have exact shape [B, 2, F]")
    if positive.device != anchor.device or negatives.device != anchor.device:
        raise ValueError("anchor, positive, and negatives devices must match")
    _raise_on_failed_checks(
        (
            _input_nonfinite_check(anchor, name="anchor"),
            _input_nonfinite_check(positive, name="positive"),
            _input_nonfinite_check(negatives, name="negatives"),
            (
                torch.any(anchor.abs().amax(dim=1) <= 0),
                ValueError,
                "anchor must have nonzero norm for every row",
            ),
            (
                torch.any(positive.abs().amax(dim=1) <= 0),
                ValueError,
                "positive must have nonzero norm for every row",
            ),
            (
                torch.any(negatives.abs().amax(dim=2) <= 0),
                ValueError,
                "negatives must have nonzero norm for every row",
            ),
        )
    )

    work_dtype = _stable_work_dtype((anchor, positive, negatives))
    anchor_work = anchor.to(dtype=work_dtype)
    positive_work = positive.to(dtype=work_dtype)
    negatives_work = negatives.to(dtype=work_dtype)
    anchor_unit, _ = _scale_safe_normalize_rows(anchor_work)
    positive_unit, _ = _scale_safe_normalize_rows(positive_work)
    negative_unit, _ = _scale_safe_normalize_rows(negatives_work)
    positive_similarity = (anchor_unit * positive_unit).sum(dim=1)
    negative_similarity = (anchor_unit[:, None] * negative_unit).sum(dim=2)
    similarities = torch.cat(
        (positive_similarity[:, None], negative_similarity), dim=1
    )
    logits = similarities / temperature
    relative_logits = (
        similarities - positive_similarity[:, None]
    ) / temperature
    loss = torch.logsumexp(relative_logits, dim=1)
    result = CCLLossResult(
        loss=loss.mean(),
        positive_similarity=positive_similarity.mean(),
        negative_similarity=negative_similarity.mean(),
    )
    result_values = torch.stack(
        (
            result.loss,
            result.positive_similarity,
            result.negative_similarity,
        )
    )
    _raise_on_failed_checks(
        (
            _computed_nonfinite_check(logits, name="contrastive logits"),
            _computed_nonfinite_check(
                relative_logits, name="relative contrastive logits"
            ),
            _computed_nonfinite_check(loss, name="contrastive loss values"),
            _computed_nonfinite_check(
                result_values, name="contrastive result"
            ),
        )
    )
    return result
