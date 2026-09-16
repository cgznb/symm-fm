from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATA_SCHEMA_VERSION = "mewm_ispy2_dce0_v2"
ANCESTRAL_DDPM_SAMPLER = "ancestral_ddpm"
CONTINUOUS_CODEBOOK_MINMAX_CONTRACT = "continuous_codebook_minmax_v1"
CONTINUOUS_TRAIN_CHANNEL_ZSCORE_CONTRACT = "continuous_train_channel_zscore_v1"
FILM_DENOISER_ARCHITECTURE = "film_unet3d_v1"
CT_DENOISER_ARCHITECTURE = "ct_unet3d_spatial32_v1"
PAPER_FAITHFUL_ARCHITECTURE = "mewm_paper_faithful_v1"
REGISTERED_LARGE_PAPER_ARCHITECTURE = "mewm_paper_registered_large_v1"
EPSILON_PREDICTION_TYPE = "epsilon"
X0_PREDICTION_TYPE = "x0"
DIFFUSION_CHECKPOINT_SCHEMA_VERSION = "mewm_ispy2_diffusion_checkpoint_v4"
_PHASE_PATTERN = re.compile(r"_dce_aqc_(\d+)\.nii(?:\.gz)?$")


@dataclass(frozen=True)
class DCE0Phase:
    path: Path
    phase_index: int
    n_times: int


def select_dce0(paths: tuple[Path, ...] | list[Path], *, n_times: int) -> DCE0Phase:
    ordered = tuple(Path(path) for path in paths)
    if type(n_times) is not int or n_times <= 0 or len(ordered) != n_times:
        raise ValueError("DCE phase count does not match n_times")
    indices: list[int] = []
    for path in ordered:
        match = _PHASE_PATTERN.search(path.name)
        if match is None:
            raise ValueError("DCE phase filename is invalid")
        if path.is_symlink() or not path.is_file():
            raise ValueError("DCE phase path is missing or unsafe")
        indices.append(int(match.group(1)))
    if indices != list(range(n_times)):
        raise ValueError("DCE phase indices must be ordered and contiguous")
    return DCE0Phase(path=ordered[0].resolve(), phase_index=0, n_times=n_times)


class ClinicalTextPolicy:
    """Validate the five static baseline fields published by the locked bundle."""

    _allowed_prefixes = (
        "age at screening ",
        "hr ",
        "her2 ",
        "mp ",
        "menopausal status ",
    )
    _forbidden = (
        "patient_id",
        "patient id",
        "ispy2-",
        "pcr",
        "target ftv",
        "target_ftv",
        "target visit",
        "target_visit",
        "future",
        "response",
        "outcome",
    )

    _field_patterns = (
        re.compile(
            r"age at screening (?:unknown|[1-9]\d?(?:\.0)?)", re.IGNORECASE
        ),
        re.compile(r"hr [01]", re.IGNORECASE),
        re.compile(r"her2 [01]", re.IGNORECASE),
        re.compile(r"mp [01]", re.IGNORECASE),
        re.compile(r"menopausal status \S(?:.*\S)?", re.IGNORECASE),
    )
    _menopausal_values = frozenset(
        {
            "Above categories not applicable AND Age < 50",
            "Above categories not applicable AND Age > 50",
            "Perimenopausal (6-12 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)",
            "Perimenopausal(6-12 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)",
            "Postmenopausal (prior bilateral ovariectomy OR > 12 months since LMP with no prior hysterectomy)",
            "Premenopausal(< 6 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)",
            "Premenopausal(<6 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)",
            "unknown",
        }
    )

    def validate(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("clinical text must be a nonempty string")
        normalized = text.strip()
        lowered = normalized.lower()
        if any(token in lowered for token in self._forbidden):
            raise ValueError("clinical text contains a forbidden identity, outcome, or target field")
        clauses = [clause.strip() for clause in normalized.split(";")]
        if len(clauses) != len(self._allowed_prefixes):
            raise ValueError("clinical text must contain exactly five static baseline fields")
        for clause, pattern in zip(clauses, self._field_patterns, strict=True):
            if pattern.fullmatch(clause) is None:
                raise ValueError("clinical text contains an unsupported field")
        menopausal_value = clauses[-1].removeprefix("menopausal status ")
        if menopausal_value not in self._menopausal_values:
            raise ValueError("clinical text contains an unsupported menopausal status")
        return normalized


@dataclass(frozen=True)
class TransitionRecord:
    transition_id: str
    patient_id: str
    fold: str
    transition_type: str
    source_visit_id: str
    target_visit_id: str
    source_dce_paths: tuple[Path, ...]
    target_dce_paths: tuple[Path, ...]
    source_mask_path: Path
    target_mask_path: Path
    action_text: str
    clinical_text: str
    delta_days: int
    stage_id: int

    def __post_init__(self) -> None:
        if self.fold not in {"train", "val", "test"}:
            raise ValueError("transition fold is invalid")
        if self.transition_type not in {"T0->T1", "T1->T2", "T2->T3"}:
            raise ValueError("transition type is invalid")
        if type(self.delta_days) is not int or self.delta_days <= 0:
            raise ValueError("delta_days must be a positive integer")
        if self.stage_id not in {1, 2, 3}:
            raise ValueError("stage_id must be 1, 2, or 3")
        ClinicalTextPolicy().validate(self.clinical_text)

    def source_view(self) -> dict[str, Any]:
        return {
            "schema_version": DATA_SCHEMA_VERSION,
            "transition_id": self.transition_id,
            "patient_id": self.patient_id,
            "fold": self.fold,
            "transition_type": self.transition_type,
            "source_visit_id": self.source_visit_id,
            "source_dce_paths": tuple(str(path) for path in self.source_dce_paths),
            "source_mask_path": str(self.source_mask_path),
            "action_text": self.action_text,
            "clinical_text": self.clinical_text,
            "delta_days": self.delta_days,
            "stage_id": self.stage_id,
        }
