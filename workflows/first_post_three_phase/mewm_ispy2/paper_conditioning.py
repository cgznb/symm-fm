from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn as nn

from .conditioning import FourierDays
from .contracts import ClinicalTextPolicy
from .paper_contracts import (
    UNKNOWN_AGE_POLICY,
    PaperRuntimeContract,
    default_paper_runtime_contract,
)


_DEFAULT_PAPER_CONTRACT = default_paper_runtime_contract()
CLIP_MODEL_ID = _DEFAULT_PAPER_CONTRACT.clip_model_id
CLIP_REVISION = _DEFAULT_PAPER_CONTRACT.clip_revision
ACTION_VOCABULARY_SCHEMA_VERSION = "ispy2_action_vocabulary_v1"

_ARM_TO_COMPONENTS_LITERAL = {
    "treatment arm Paclitaxel": ("Paclitaxel",),
    "treatment arm Paclitaxel + ABT 888 + Carboplatin": (
        "Paclitaxel",
        "ABT 888",
        "Carboplatin",
    ),
    "treatment arm Paclitaxel + AMG 386": ("Paclitaxel", "AMG 386"),
    "treatment arm Paclitaxel + AMG 386 + Trastuzumab": (
        "Paclitaxel",
        "AMG 386",
        "Trastuzumab",
    ),
    "treatment arm Paclitaxel + Ganetespib": ("Paclitaxel", "Ganetespib"),
    "treatment arm Paclitaxel + Ganitumab": ("Paclitaxel", "Ganitumab"),
    "treatment arm Paclitaxel + MK-2206": ("Paclitaxel", "MK-2206"),
    "treatment arm Paclitaxel + MK-2206 + Trastuzumab": (
        "Paclitaxel",
        "MK-2206",
        "Trastuzumab",
    ),
    "treatment arm Paclitaxel + Neratinib": ("Paclitaxel", "Neratinib"),
    "treatment arm Paclitaxel + Pembrolizumab": (
        "Paclitaxel",
        "Pembrolizumab",
    ),
    "treatment arm Paclitaxel + Pertuzumab + Trastuzumab": (
        "Paclitaxel",
        "Pertuzumab",
        "Trastuzumab",
    ),
    "treatment arm Paclitaxel + Trastuzumab": (
        "Paclitaxel",
        "Trastuzumab",
    ),
    "treatment arm T-DM1 + Pertuzumab": ("T-DM1", "Pertuzumab"),
}
ARM_TO_COMPONENTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    _ARM_TO_COMPONENTS_LITERAL
)

COMPONENT_VOCABULARY = tuple(
    dict.fromkeys(
        component
        for components in _ARM_TO_COMPONENTS_LITERAL.values()
        for component in components
    )
)
if len(COMPONENT_VOCABULARY) != 12:
    raise RuntimeError("paper action vocabulary must contain exactly 12 components")
_COMPONENT_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {
        component: index
        for index, component in enumerate(COMPONENT_VOCABULARY)
    }
)

MENOPAUSE_CATEGORIES = (
    "premenopausal",
    "perimenopausal",
    "postmenopausal",
    "not_applicable_age_lt_50",
    "not_applicable_age_gt_50",
    "unknown",
)
_MENOPAUSE_TO_INDEX = {
    category: index for index, category in enumerate(MENOPAUSE_CATEGORIES)
}
_RAW_MENOPAUSE_TO_CATEGORY = {
    (
        "Perimenopausal (6-12 months since LMP AND no prior bilateral "
        "ovariectomy AND not on estrogen replacement)"
    ): "perimenopausal",
    (
        "Perimenopausal(6-12 months since LMP AND no prior bilateral "
        "ovariectomy AND not on estrogen replacement)"
    ): "perimenopausal",
    (
        "Postmenopausal (prior bilateral ovariectomy OR > 12 months since LMP "
        "with no prior hysterectomy)"
    ): "postmenopausal",
    (
        "Premenopausal(< 6 months since LMP AND no prior bilateral "
        "ovariectomy AND not on estrogen replacement)"
    ): "premenopausal",
    (
        "Premenopausal(<6 months since LMP AND no prior bilateral "
        "ovariectomy AND not on estrogen replacement)"
    ): "premenopausal",
    "Above categories not applicable AND Age < 50": "not_applicable_age_lt_50",
    "Above categories not applicable AND Age > 50": "not_applicable_age_gt_50",
    "unknown": "unknown",
}

_CLINICAL_PATTERN = re.compile(
    r"age at screening (?P<age>[^;]+);\s*"
    r"HR (?P<hr>[^;]+);\s*"
    r"HER2 (?P<her2>[^;]+);\s*"
    r"MP (?P<mp>[^;]+);\s*"
    r"menopausal status (?P<menopause>.+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedAction:
    text: str
    components: tuple[str, ...]


@dataclass(frozen=True)
class StructuredClinicalFields:
    age: float | None
    hr: int
    her2: int
    mp: int
    menopause: str


@dataclass(frozen=True)
class PaperConditionOutput:
    global_condition: torch.Tensor
    context_tokens: torch.Tensor
    context_mask: torch.Tensor


def parse_action_text(text: str) -> ParsedAction:
    if not isinstance(text, str):
        raise TypeError("action text must be a string")
    if not text.strip():
        raise ValueError("action text must be nonempty")
    components = ARM_TO_COMPONENTS.get(text)
    if components is None:
        raise ValueError("action text must exactly match a canonical treatment arm")
    return ParsedAction(text=text, components=components)


def action_vocabulary_sha256() -> str:
    payload = {
        "schema_version": ACTION_VOCABULARY_SCHEMA_VERSION,
        "arms": {
            arm: list(ARM_TO_COMPONENTS[arm]) for arm in sorted(ARM_TO_COMPONENTS)
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def parse_clinical_text(text: str) -> StructuredClinicalFields:
    if not isinstance(text, str):
        raise TypeError("clinical text must be a string")
    normalized = ClinicalTextPolicy().validate(text)
    match = _CLINICAL_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError("clinical text does not match the structured field contract")

    raw_age = match.group("age")
    if raw_age == "unknown":
        canonical_unknown_text = (
            f"age at screening unknown; HR {match.group('hr')}; "
            f"HER2 {match.group('her2')}; MP {match.group('mp')}; "
            f"menopausal status {match.group('menopause')}"
        )
        if text != canonical_unknown_text:
            raise ValueError("unknown age must use the exact canonical raw text")
        age = None
    else:
        try:
            age = float(raw_age)
        except ValueError as exc:
            raise ValueError("age at screening must be numeric or exact unknown") from exc
        if not math.isfinite(age):
            raise ValueError("age at screening must be finite")

    subtypes: dict[str, int] = {}
    for field in ("hr", "her2", "mp"):
        raw_value = match.group(field)
        if raw_value not in {"0", "1"}:
            raise ValueError(f"{field.upper()} must be binary")
        subtypes[field] = int(raw_value)

    raw_menopause = match.group("menopause")
    menopause = _RAW_MENOPAUSE_TO_CATEGORY.get(raw_menopause)
    if menopause is None:
        raise ValueError("menopausal status is outside the closed vocabulary")
    return StructuredClinicalFields(
        age=age,
        hr=subtypes["hr"],
        her2=subtypes["her2"],
        mp=subtypes["mp"],
        menopause=menopause,
    )


class FrozenCLIPTextTower(nn.Module):
    hidden_size = 512

    def __init__(self, model: nn.Module, tokenizer: Any) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model.requires_grad_(False)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = CLIP_MODEL_ID,
        revision: str = CLIP_REVISION,
        local_files_only: bool = False,
    ) -> "FrozenCLIPTextTower":
        if model_id != CLIP_MODEL_ID or revision != CLIP_REVISION:
            raise ValueError("CLIP base identity must match the locked model revision")
        from transformers import CLIPTextModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        model = CLIPTextModel.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(model, tokenizer)

    def train(self, mode: bool = True) -> "FrozenCLIPTextTower":
        super().train(mode)
        self.model.eval()
        return self

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if (
            isinstance(texts, (str, bytes))
            or not isinstance(texts, Sequence)
            or not texts
            or not all(isinstance(text, str) and text.strip() for text in texts)
        ):
            raise ValueError("text batches must be nonempty sequences of nonempty strings")

        device = next(self.model.parameters()).device
        with torch.no_grad():
            tokens = self.tokenizer(
                list(texts), padding=True, truncation=True, return_tensors="pt"
            )
            if "attention_mask" not in tokens:
                raise RuntimeError("CLIP tokenizer did not return an attention mask")
            tokens = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in tokens.items()
            }
            outputs = self.model(**tokens, return_dict=True)
            pooled = getattr(outputs, "pooler_output", None)
            if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
                raise RuntimeError("CLIP text model did not return pooled text features")
            if pooled.shape != (len(texts), self.hidden_size):
                raise RuntimeError("CLIP pooled text features must be 512D")
            return pooled


class PaperActionConditioner(nn.Module):
    def __init__(
        self,
        text_tower: nn.Module,
        *,
        text_hidden_size: int | None = None,
        contract: PaperRuntimeContract | None = None,
    ) -> None:
        super().__init__()
        self.contract = contract or default_paper_runtime_contract()
        if (
            type(self.contract.unknown_age_policy) is not str
            or self.contract.unknown_age_policy != UNKNOWN_AGE_POLICY
        ):
            raise ValueError(
                "unknown age policy must match the exact paper runtime contract"
            )
        vocabulary_sha256 = action_vocabulary_sha256()
        if vocabulary_sha256 != self.contract.action_vocabulary_sha256:
            raise ValueError(
                "action vocabulary SHA256 mismatch: "
                f"runtime {vocabulary_sha256}, "
                f"contract {self.contract.action_vocabulary_sha256}"
            )
        if self.contract.context_dim != 512 or self.contract.context_tokens != 7:
            raise ValueError("paper context contract must remain [B,7,512]")
        if not math.isfinite(self.contract.age_mean) or not (
            math.isfinite(self.contract.age_std) and self.contract.age_std > 0.0
        ):
            raise ValueError("paper age statistics must be finite with positive std")

        self.text_tower = text_tower
        self.text_tower.requires_grad_(False)
        self.text_tower.eval()
        hidden_size = int(
            text_hidden_size
            if text_hidden_size is not None
            else getattr(text_tower, "hidden_size")
        )
        if hidden_size <= 0:
            raise ValueError("text hidden size must be positive")
        self.text_hidden_size = hidden_size

        self.holistic_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
        )
        self.component_projection = nn.Linear(hidden_size, 512)
        self.component_concepts = nn.Embedding(len(COMPONENT_VOCABULARY), 512)
        self.component_layer_norm = nn.LayerNorm(512)

        self.hr_context_embedding = nn.Embedding(2, 512)
        self.her2_context_embedding = nn.Embedding(2, 512)
        self.mp_context_embedding = nn.Embedding(2, 512)
        self.hr_global_embedding = nn.Embedding(2, 64)
        self.her2_global_embedding = nn.Embedding(2, 64)
        self.mp_global_embedding = nn.Embedding(2, 64)
        self.age_projection = nn.Linear(1, 64)
        self.menopause_embedding = nn.Embedding(len(MENOPAUSE_CATEGORIES), 64)
        self.stage_embedding = nn.Embedding(4, 64)
        self.days_encoder = FourierDays(128)
        self.fusion = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
        )

    def train(self, mode: bool = True) -> "PaperActionConditioner":
        super().train(mode)
        self.text_tower.eval()
        return self

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def forward(
        self,
        action_text: Sequence[str],
        clinical_text: Sequence[str],
        delta_days: torch.Tensor,
        stage_id: torch.Tensor,
    ) -> PaperConditionOutput:
        if (
            isinstance(action_text, (str, bytes))
            or not isinstance(action_text, Sequence)
            or not action_text
        ):
            raise ValueError("action text must be a nonempty sequence")
        if (
            isinstance(clinical_text, (str, bytes))
            or not isinstance(clinical_text, Sequence)
            or len(clinical_text) != len(action_text)
        ):
            raise ValueError("action and clinical text batch sizes must match")
        batch_size = len(action_text)
        if (
            not isinstance(delta_days, torch.Tensor)
            or delta_days.ndim != 1
            or delta_days.shape[0] != batch_size
        ):
            raise ValueError("delta days must have shape [B]")
        if (
            not isinstance(stage_id, torch.Tensor)
            or stage_id.ndim != 1
            or stage_id.shape[0] != batch_size
        ):
            raise ValueError("stage IDs must have shape [B]")
        if delta_days.dtype == torch.bool or delta_days.is_complex():
            raise ValueError("delta days must use a real non-boolean tensor dtype")
        if not torch.all(torch.isfinite(delta_days)):
            raise ValueError("delta days must be finite")
        if torch.any(delta_days <= 0):
            raise ValueError("delta days must be positive")
        if stage_id.dtype == torch.bool or stage_id.is_complex():
            raise ValueError("stage IDs must use a real non-boolean tensor dtype")
        if not torch.all(torch.isfinite(stage_id)):
            raise ValueError("stage IDs must be finite")
        if stage_id.is_floating_point() and not torch.all(
            stage_id == stage_id.round()
        ):
            raise ValueError("stage IDs must be integers")

        parsed_actions = [parse_action_text(text) for text in action_text]
        clinical_fields = [parse_clinical_text(text) for text in clinical_text]

        projection_parameter = self.component_projection.weight
        device = projection_parameter.device
        dtype = projection_parameter.dtype
        stages = stage_id.to(device=device, dtype=torch.long)
        if torch.any((stages < 1) | (stages > 3)):
            raise ValueError("stage IDs must be 1, 2, or 3")

        holistic_phrases = [action.text for action in parsed_actions]
        component_phrases = [
            component
            for action in parsed_actions
            for component in action.components
        ]
        unique_phrases = list(dict.fromkeys((*holistic_phrases, *component_phrases)))
        hidden = self.text_tower.encode(unique_phrases)
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 2
            or hidden.shape != (len(unique_phrases), self.text_hidden_size)
        ):
            raise RuntimeError("text tower returned features with the wrong shape")
        hidden = hidden.to(device=device, dtype=dtype)
        phrase_to_row = {
            phrase: index for index, phrase in enumerate(unique_phrases)
        }
        holistic_rows = torch.tensor(
            [phrase_to_row[phrase] for phrase in holistic_phrases],
            dtype=torch.long,
            device=device,
        )
        component_rows = torch.tensor(
            [phrase_to_row[phrase] for phrase in component_phrases],
            dtype=torch.long,
            device=device,
        )
        holistic_tokens = self.holistic_projection(hidden[holistic_rows])
        component_tokens = self.component_projection(hidden[component_rows])
        component_ids = torch.tensor(
            [_COMPONENT_TO_INDEX[component] for component in component_phrases],
            dtype=torch.long,
            device=device,
        )
        component_tokens = self.component_layer_norm(
            component_tokens + self.component_concepts(component_ids)
        )

        hr = torch.tensor(
            [fields.hr for fields in clinical_fields], dtype=torch.long, device=device
        )
        her2 = torch.tensor(
            [fields.her2 for fields in clinical_fields], dtype=torch.long, device=device
        )
        mp = torch.tensor(
            [fields.mp for fields in clinical_fields], dtype=torch.long, device=device
        )
        subtype_context = (
            self.hr_context_embedding(hr),
            self.her2_context_embedding(her2),
            self.mp_context_embedding(mp),
        )

        context_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        action_means: list[torch.Tensor] = []
        offset = 0
        for row, action in enumerate(parsed_actions):
            count = len(action.components)
            actual_components = component_tokens[offset : offset + count]
            offset += count
            padding = [
                holistic_tokens.new_zeros(512) for _ in range(3 - count)
            ]
            context_rows.append(
                torch.stack(
                    [
                        holistic_tokens[row],
                        *actual_components.unbind(dim=0),
                        *padding,
                        subtype_context[0][row],
                        subtype_context[1][row],
                        subtype_context[2][row],
                    ]
                )
            )
            mask_rows.append(
                torch.tensor(
                    [True, *([True] * count), *([False] * (3 - count)), True, True, True],
                    dtype=torch.bool,
                    device=device,
                )
            )
            action_means.append(
                torch.cat((holistic_tokens[row : row + 1], actual_components)).mean(
                    dim=0
                )
            )

        context_tokens = torch.stack(context_rows)
        context_mask = torch.stack(mask_rows)
        action_global = torch.stack(action_means)
        subtype_global = torch.cat(
            (
                self.hr_global_embedding(hr),
                self.her2_global_embedding(her2),
                self.mp_global_embedding(mp),
            ),
            dim=1,
        )
        ages = torch.tensor(
            [
                self.contract.age_mean if fields.age is None else fields.age
                for fields in clinical_fields
            ],
            dtype=dtype,
            device=device,
        ).reshape(batch_size, 1)
        normalized_ages = (ages - self.contract.age_mean) / self.contract.age_std
        age_global = self.age_projection(normalized_ages)
        menopause_ids = torch.tensor(
            [_MENOPAUSE_TO_INDEX[fields.menopause] for fields in clinical_fields],
            dtype=torch.long,
            device=device,
        )
        menopause_global = self.menopause_embedding(menopause_ids)
        stage_global = self.stage_embedding(stages)
        days_global = self.days_encoder(delta_days.to(device=device)).to(dtype=dtype)

        global_input = torch.cat(
            (
                action_global,
                subtype_global,
                age_global,
                menopause_global,
                stage_global,
                days_global,
            ),
            dim=1,
        )
        if global_input.shape != (batch_size, 1024):
            raise RuntimeError("paper global conditioning input must be 1024D")
        if context_tokens.shape != (batch_size, 7, 512):
            raise RuntimeError("paper context token contract must be [B,7,512]")
        return PaperConditionOutput(
            global_condition=self.fusion(global_input),
            context_tokens=context_tokens,
            context_mask=context_mask,
        )
