from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .contracts import REGISTERED_LARGE_PAPER_ARCHITECTURE
from .vqgan import REGISTERED_VQGAN_NUMERIC_CONTRACT


UNKNOWN_AGE_POLICY = "exact_literal_mean_impute_zero_v1"


@dataclass(frozen=True)
class AttenuationLevelConfig:
    level: int
    morphology: str
    radius: int
    sigma: float
    exponent: float
    decay: float


@dataclass(frozen=True)
class PaperRuntimeContract:
    architecture_id: str
    paper_version: str
    official_code_revision: str
    clip_model_id: str
    clip_revision: str
    action_vocabulary_sha256: str
    bundle_contract_sha256: str
    phase_manifest_sha256: str
    data_contract_sha256: str
    vqgan_sha256: str
    clinical_policy_version: str
    unknown_age_policy: str
    age_mean: float
    age_std: float
    context_dim: int
    context_tokens: int
    cross_attention_topology: str
    attenuation_policy: str
    attenuation_levels: tuple[AttenuationLevelConfig, ...]
    ccl_pairing_version: str
    ccl_negative_count: int
    ccl_temperature: float
    ccl_weight: float
    ccl_timestep_min: int
    ccl_timestep_max: int
    epsilon_objective: str
    warm_start_path: Path
    warm_start_sha256: str

    def __post_init__(self) -> None:
        if type(self.attenuation_levels) is not tuple or any(
            not isinstance(level, AttenuationLevelConfig)
            for level in self.attenuation_levels
        ):
            raise TypeError(
                "attenuation_levels must be an immutable tuple of level configs"
            )


@dataclass(frozen=True)
class RegisteredLargePaperRuntimeContract(PaperRuntimeContract):
    vqgan_numeric_contract: str
    train_patient_count: int
    numeric_age_train_patient_count: int
    unknown_age_train_patient_count: int
    train_transition_count: int
    train_ccl_valid_count: int
    val_transition_count: int
    val_ccl_valid_count: int
    initialization_method: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.vqgan_sha256) is not str:
            raise TypeError("vqgan_sha256 must be an exact string")
        if len(self.vqgan_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.vqgan_sha256
        ):
            raise ValueError("vqgan_sha256 must contain 64 lowercase hex characters")
        if self.warm_start_path is not None or self.warm_start_sha256 is not None:
            raise ValueError("registered-large initialization cannot have a warm source")


def default_paper_runtime_contract() -> PaperRuntimeContract:
    return PaperRuntimeContract(
        architecture_id="mewm_paper_faithful_v1",
        paper_version="arxiv:2506.02327v1",
        official_code_revision="d8e9d9b31fcb96e7778e6518338433e8ce953e04",
        clip_model_id="openai/clip-vit-base-patch32",
        clip_revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        action_vocabulary_sha256=(
            "c0794245acb535d3cf52e5579b1651199ca7c934b4c420fa70a2b389810b8f2c"
        ),
        bundle_contract_sha256=(
            "940af2a569ff8cedf6da36bede71d1f53c2bc2c687a949dbb45f9b72890ad6aa"
        ),
        phase_manifest_sha256=(
            "3d7835e9a17b08645979bc6000a4969a3b1840dfb7953c0f0cedf3d83fc9f5b4"
        ),
        data_contract_sha256=(
            "e5a89abea7d5fa5d79b2bc8beb69918417d69d84185d8fa76ba076dfdfd737c0"
        ),
        vqgan_sha256=(
            "ff639cb5b456e531d20c1a67b0e3cec47b43cbdc12f38348034b7daed764f33c"
        ),
        clinical_policy_version="ispy2_structured_hr_her2_mp_v1",
        unknown_age_policy=UNKNOWN_AGE_POLICY,
        age_mean=49.633027522935777,
        age_std=10.278680067148208,
        context_dim=512,
        context_tokens=7,
        cross_attention_topology="down4_middle1_up4_v1",
        attenuation_policy="uniform_four_level_source_only_v1",
        attenuation_levels=(
            AttenuationLevelConfig(1, "identity", 0, 2.0, 0.7, 0.7),
            AttenuationLevelConfig(2, "opening", 1, 4.0, 0.5, 0.5),
            AttenuationLevelConfig(3, "erosion", 1, 6.0, 0.4, 0.3),
            AttenuationLevelConfig(4, "erosion", 2, 8.0, 0.3, 0.1),
        ),
        ccl_pairing_version="same_fold_arm_stage_subtype_other_patient_v1",
        ccl_negative_count=2,
        ccl_temperature=0.07,
        ccl_weight=0.1,
        ccl_timestep_min=100,
        ccl_timestep_max=199,
        epsilon_objective="l2",
        warm_start_path=Path(
            _release_path('@artifacts/mewm/runs/current_dce0_ct_unet3d_replica_v1/diffusion/checkpoints/step-0010600.ckpt')
        ),
        warm_start_sha256=(
            "609f9be6a31dbe704fc264ab45dd464b81c458f29075861d866ea92bf8da8285"
        ),
    )


def registered_large_paper_runtime_contract(
    *, vqgan_sha256: str
) -> RegisteredLargePaperRuntimeContract:
    if type(vqgan_sha256) is not str:
        raise TypeError("vqgan_sha256 must be an exact string")
    legacy = default_paper_runtime_contract()
    common = {
        field.name: getattr(legacy, field.name)
        for field in fields(PaperRuntimeContract)
    }
    common.update(
        architecture_id=REGISTERED_LARGE_PAPER_ARCHITECTURE,
        bundle_contract_sha256=(
            "9d78b82ca97b897a99233452dce30e2c4275fe0b944a017f4beb99ec962c31d1"
        ),
        phase_manifest_sha256=(
            "840611a2f13342d93b237ae12ebf9383e7dbbed4ebaf84a082e654bbd5132ee6"
        ),
        data_contract_sha256=(
            "52fd47f608f0d777647340a868341cbc6b0b60165b046aaef18fa636b91b0067"
        ),
        vqgan_sha256=vqgan_sha256,
        age_mean=48.72605363984675,
        age_std=10.443959708268094,
        warm_start_path=None,
        warm_start_sha256=None,
    )
    return RegisteredLargePaperRuntimeContract(
        **common,
        vqgan_numeric_contract=REGISTERED_VQGAN_NUMERIC_CONTRACT,
        train_patient_count=524,
        numeric_age_train_patient_count=522,
        unknown_age_train_patient_count=2,
        train_transition_count=1273,
        train_ccl_valid_count=1257,
        val_transition_count=128,
        val_ccl_valid_count=82,
        initialization_method="random",
    )


def paper_contract_payload(contract: PaperRuntimeContract) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields(contract):
        value = getattr(contract, field.name)
        if field.name == "warm_start_path":
            value = str(value)
        elif field.name == "attenuation_levels":
            value = [
                {
                    level_field.name: getattr(level, level_field.name)
                    for level_field in fields(level)
                }
                for level in value
            ]
        payload[field.name] = value
    return payload


def paper_contract_sha256(contract: PaperRuntimeContract) -> str:
    canonical_json = json.dumps(
        paper_contract_payload(contract),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(canonical_json).hexdigest()


def registered_large_paper_contract_payload(
    contract: RegisteredLargePaperRuntimeContract,
) -> dict[str, Any]:
    if type(contract) is not RegisteredLargePaperRuntimeContract:
        raise TypeError(
            "contract must be an exact RegisteredLargePaperRuntimeContract"
        )
    payload: dict[str, Any] = {}
    for field in fields(contract):
        value = getattr(contract, field.name)
        if field.name == "warm_start_path":
            value = None
        elif field.name == "attenuation_levels":
            value = [
                {
                    level_field.name: getattr(level, level_field.name)
                    for level_field in fields(level)
                }
                for level in value
            ]
        payload[field.name] = value
    return payload


def registered_large_paper_contract_sha256(
    contract: RegisteredLargePaperRuntimeContract,
) -> str:
    canonical_json = json.dumps(
        registered_large_paper_contract_payload(contract),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(canonical_json).hexdigest()
