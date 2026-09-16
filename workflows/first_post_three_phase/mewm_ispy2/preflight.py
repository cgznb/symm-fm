from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import hashlib
import json
import math
import statistics
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .backend import (
    REGISTERED_STRICT_A_BUNDLE_SCHEMA,
    LoadedTransitions,
    load_split_visit_records,
    load_transition_records,
    sha256_file,
)
from .cache import DCE0Cache, RegisteredStrictAROICache
from .config import ExperimentConfig, load_experiment_config
from .manifest import ACCEPTED_REGISTRATION_STATUSES
from .paper_ccl import CCLPairIndex, CCLSelection
from .paper_conditioning import (
    ARM_TO_COMPONENTS,
    COMPONENT_VOCABULARY,
    action_vocabulary_sha256,
    parse_action_text,
    parse_clinical_text,
)
from .paper_contracts import PaperRuntimeContract
from .paper_warm_start import _validate_v4_payload


CT_REPOSITORY = "MrGiovanni/DiffTumor"
CT_REVISION = "089ba0f3f94a7858603a55106791d7d977d7bc0b"
CT_EXPECTED_SHA256 = "be83639891cc1f14527feadef9484b2d3887fc5f513b4026a90faa230856ffed"
CT_DEFAULT_PATH = Path(
    "/root/.cache/mewm_ispy2/difftumor/"
    f"{CT_REVISION}/AutoencoderModel.ckpt"
)
MEDGEMMA_SNAPSHOT = Path(
    _release_path('@cache/huggingface/hub/models--google--medgemma-4b-it/snapshots/290cda5eeccbee130f987c4ad74a59ae6f196408')
)

PAPER_TRAIN_TRANSITION_COUNT = 849
PAPER_TRAIN_CCL_VALID_COUNT = 828
PAPER_TRAIN_CCL_VALID_FRACTION = 0.9752650176678445
PAPER_TRAIN_PATIENT_COUNT = 328
PAPER_TRAIN_NUMERIC_AGE_PATIENT_COUNT = 327
PAPER_TRAIN_UNKNOWN_AGE_PATIENT_COUNT = 1
PAPER_CCL_AUDIT_SEED = 2026
PAPER_CCL_AUDIT_EPOCH = 0
STRICT_A_PAPER_TRAIN_TRANSITION_COUNT = 1273
STRICT_A_PAPER_VAL_TRANSITION_COUNT = 128
STRICT_A_PAPER_TRAIN_PATIENT_COUNT = 524
STRICT_A_PAPER_TRAIN_NUMERIC_AGE_PATIENT_COUNT = 522
STRICT_A_PAPER_TRAIN_UNKNOWN_AGE_PATIENT_COUNT = 2
STRICT_A_PAPER_TRAIN_CCL_VALID_COUNT = 1257
STRICT_A_PAPER_VAL_CCL_VALID_COUNT = 82


def _locked_registration_counts(config: ExperimentConfig) -> dict[str, int]:
    bundle = json.loads(config.data.bundle_json.read_text())
    transition_path = config.data.bundle_json.parent / bundle["artifacts"]["transitions"]["path"]
    transitions = pd.read_csv(transition_path)
    locked = set(transitions["source_visit_id"]) | set(transitions["target_visit_id"])
    manifest = pd.read_csv(config.data.phase_manifest_csv)
    ids = manifest["patient_id"].astype(str) + ":" + manifest["visit"].astype(str)
    statuses = manifest.loc[ids.isin(locked), "registration_status"].fillna("missing")
    return {str(key): int(value) for key, value in statuses.value_counts().items()}


def _resolve_paper_clip_snapshot(contract: PaperRuntimeContract) -> Path:
    from huggingface_hub import snapshot_download

    resolved = snapshot_download(
        contract.clip_model_id,
        revision=contract.clip_revision,
        local_files_only=True,
    )
    path = Path(resolved)
    if not path.is_dir():
        raise FileNotFoundError("resolved CLIP snapshot directory is missing")
    return path


def _validate_paper_warm_start_metadata(
    path: Path, contract: PaperRuntimeContract
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _state, schema, epoch, global_step = _validate_v4_payload(
        payload,
        expected_vqgan_sha256=contract.vqgan_sha256,
        expected_data_contract_sha256=contract.data_contract_sha256,
    )
    return {
        "warm_start_source_schema_version": schema,
        "warm_start_source_epoch": epoch,
        "warm_start_source_global_step": global_step,
    }


def _expected_paper_negative_actions(
    anchor: Any,
    *,
    seed: int,
    epoch: int,
) -> tuple[str, str]:
    anchor_components = frozenset(ARM_TO_COMPONENTS[anchor.action_text])
    arms_by_distance: dict[Fraction, list[str]] = {}
    for arm in ARM_TO_COMPONENTS:
        if arm == anchor.action_text:
            continue
        components = frozenset(ARM_TO_COMPONENTS[arm])
        union = anchor_components | components
        distance = Fraction(
            len(union) - len(anchor_components & components), len(union)
        )
        arms_by_distance.setdefault(distance, []).append(arm)

    ranked: list[str] = []
    for distance in sorted(arms_by_distance, reverse=True):
        tied = sorted(arms_by_distance[distance])
        tie_id = f"{distance.numerator}/{distance.denominator}"
        payload = f"{seed}:{epoch}:{anchor.transition_id}:{tie_id}"
        rotation = int.from_bytes(
            hashlib.sha256(payload.encode("utf-8")).digest(), "big"
        ) % len(tied)
        ranked.extend(tied[rotation:] + tied[:rotation])
    if len(ranked) < 2:
        raise RuntimeError("paper action vocabulary has fewer than two negatives")
    return ranked[0], ranked[1]


def _validate_paper_ccl_selection(
    selection: CCLSelection,
    *,
    anchor: Any,
    records_by_id: dict[str, Any],
    clinical_by_id: dict[str, Any],
    seed: int,
    epoch: int,
) -> bool:
    if not isinstance(selection, CCLSelection):
        raise TypeError("paper CCL selection must be a CCLSelection")
    if selection.anchor_transition_id != anchor.transition_id:
        raise ValueError("paper CCL selection anchor is illegal")
    if not selection.valid:
        if (
            selection.positive_transition_id is not None
            or selection.negative_action_texts != ()
        ):
            raise ValueError("paper singleton CCL selection is illegal")
        return False

    positive_id = selection.positive_transition_id
    if positive_id not in records_by_id:
        raise ValueError("paper CCL positive selection is missing")
    positive = records_by_id[positive_id]
    if positive.patient_id == anchor.patient_id:
        raise ValueError("paper CCL positive must use a different patient")
    anchor_clinical = clinical_by_id[anchor.transition_id]
    positive_clinical = clinical_by_id[positive.transition_id]
    if (
        positive.fold != anchor.fold
        or positive.action_text != anchor.action_text
        or positive.transition_type != anchor.transition_type
        or (positive_clinical.hr, positive_clinical.her2, positive_clinical.mp)
        != (anchor_clinical.hr, anchor_clinical.her2, anchor_clinical.mp)
    ):
        raise ValueError("paper CCL positive pairing fields are illegal")

    negatives = selection.negative_action_texts
    if (
        type(negatives) is not tuple
        or len(negatives) != 2
        or len(set(negatives)) != 2
        or anchor.action_text in negatives
    ):
        raise ValueError("paper CCL negative selection is illegal")
    for action in negatives:
        parse_action_text(action)
    expected_negatives = _expected_paper_negative_actions(
        anchor,
        seed=seed,
        epoch=epoch,
    )
    if negatives != expected_negatives:
        raise ValueError(
            "paper CCL negative Jaccard-hard tuple or stable order changed"
        )
    return True


def _audit_paper_data(
    loaded: LoadedTransitions,
    contract: PaperRuntimeContract,
    *,
    ccl_seed: int = PAPER_CCL_AUDIT_SEED,
    ccl_epoch: int = PAPER_CCL_AUDIT_EPOCH,
) -> dict[str, int | float]:
    if not isinstance(loaded, LoadedTransitions):
        raise TypeError("paper data audit requires LoadedTransitions")
    if not isinstance(contract, PaperRuntimeContract):
        raise TypeError("paper data audit requires PaperRuntimeContract")
    if type(ccl_seed) is not int:
        raise TypeError("paper CCL audit seed must be an exact integer")
    if type(ccl_epoch) is not int:
        raise TypeError("paper CCL audit epoch must be an exact integer")
    if ccl_epoch < 0:
        raise ValueError("paper CCL audit epoch must be nonnegative")
    if action_vocabulary_sha256() != contract.action_vocabulary_sha256:
        raise ValueError("paper action vocabulary identity changed")

    parsed_clinical: dict[str, Any] = {}
    for record in loaded.records:
        try:
            parse_action_text(record.action_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"paper action parsing failed for {record.transition_id}: {error}"
            ) from error
        try:
            parsed_clinical[record.transition_id] = parse_clinical_text(
                record.clinical_text
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"paper clinical parsing failed for {record.transition_id}: {error}"
            ) from error

    strict_a = loaded.bundle_schema_version == REGISTERED_STRICT_A_BUNDLE_SCHEMA
    expected_train_count = (
        STRICT_A_PAPER_TRAIN_TRANSITION_COUNT
        if strict_a
        else PAPER_TRAIN_TRANSITION_COUNT
    )
    expected_patient_count = (
        STRICT_A_PAPER_TRAIN_PATIENT_COUNT if strict_a else PAPER_TRAIN_PATIENT_COUNT
    )
    expected_numeric_age_count = (
        STRICT_A_PAPER_TRAIN_NUMERIC_AGE_PATIENT_COUNT
        if strict_a
        else PAPER_TRAIN_NUMERIC_AGE_PATIENT_COUNT
    )
    expected_unknown_age_count = (
        STRICT_A_PAPER_TRAIN_UNKNOWN_AGE_PATIENT_COUNT
        if strict_a
        else PAPER_TRAIN_UNKNOWN_AGE_PATIENT_COUNT
    )
    expected_train_ccl_count = (
        STRICT_A_PAPER_TRAIN_CCL_VALID_COUNT
        if strict_a
        else PAPER_TRAIN_CCL_VALID_COUNT
    )

    train = tuple(record for record in loaded.records if record.fold == "train")
    if len(train) != expected_train_count:
        raise ValueError("paper train transition/patient count changed")
    records_by_id = {record.transition_id: record for record in train}
    if len(records_by_id) != len(train):
        raise ValueError("paper train transition IDs are duplicated")

    patient_records: dict[str, list[Any]] = {}
    for record in train:
        patient_records.setdefault(record.patient_id, []).append(record)
    if len(patient_records) != expected_patient_count:
        raise ValueError("paper unique train patient count changed")

    patient_fields: dict[str, Any] = {}
    for patient_id, records in patient_records.items():
        ordered = sorted(records, key=lambda record: record.transition_id)
        stable = parsed_clinical[ordered[0].transition_id]
        for record in ordered[1:]:
            if parsed_clinical[record.transition_id] != stable:
                raise ValueError(
                    f"paper clinical fields changed within train patient {patient_id}"
                )
        patient_fields[patient_id] = stable

    numeric_ages = sorted(
        float(fields.age)
        for fields in patient_fields.values()
        if fields.age is not None
    )
    unknown_age_count = sum(
        fields.age is None for fields in patient_fields.values()
    )
    if len(numeric_ages) != expected_numeric_age_count:
        raise ValueError("paper numeric-age train patient count changed")
    if unknown_age_count != expected_unknown_age_count:
        raise ValueError("paper unknown-age train patient count changed")
    age_mean = statistics.fmean(numeric_ages)
    age_std = statistics.pstdev(numeric_ages)
    if not math.isclose(age_mean, contract.age_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("paper age mean/statistics changed")
    if not math.isclose(age_std, contract.age_std, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("paper age standard deviation/statistics changed")

    pair_index = CCLPairIndex(train, seed=ccl_seed)
    valid_count = 0
    for anchor in sorted(train, key=lambda record: record.transition_id):
        selection = pair_index.select(anchor.transition_id, epoch=ccl_epoch)
        valid_count += int(
            _validate_paper_ccl_selection(
                selection,
                anchor=anchor,
                records_by_id=records_by_id,
                clinical_by_id=parsed_clinical,
                seed=ccl_seed,
                epoch=ccl_epoch,
            )
        )
    if valid_count != expected_train_ccl_count:
        raise ValueError("paper train CCL coverage changed")

    result: dict[str, int | float] = {
        "train_transition_count": len(train),
        "train_ccl_valid_count": valid_count,
        "train_ccl_valid_fraction": (
            valid_count / len(train)
            if strict_a
            else PAPER_TRAIN_CCL_VALID_FRACTION
        ),
        "action_arm_count": len(ARM_TO_COMPONENTS),
        "action_component_count": len(COMPONENT_VOCABULARY),
        "unique_train_patient_count": len(patient_records),
        "numeric_age_train_patient_count": len(numeric_ages),
        "unknown_age_train_patient_count": unknown_age_count,
        "age_mean": age_mean,
        "age_std": age_std,
    }
    if strict_a:
        validation = tuple(
            record for record in loaded.records if record.fold == "val"
        )
        if len(validation) != STRICT_A_PAPER_VAL_TRANSITION_COUNT:
            raise ValueError("paper validation transition count changed")
        validation_by_id = {
            record.transition_id: record for record in validation
        }
        if len(validation_by_id) != len(validation):
            raise ValueError("paper validation transition IDs are duplicated")
        validation_index = CCLPairIndex(validation, seed=ccl_seed)
        validation_valid_count = 0
        for anchor in sorted(validation, key=lambda record: record.transition_id):
            selection = validation_index.select(
                anchor.transition_id, epoch=ccl_epoch
            )
            validation_valid_count += int(
                _validate_paper_ccl_selection(
                    selection,
                    anchor=anchor,
                    records_by_id=validation_by_id,
                    clinical_by_id=parsed_clinical,
                    seed=ccl_seed,
                    epoch=ccl_epoch,
                )
            )
        if validation_valid_count != STRICT_A_PAPER_VAL_CCL_VALID_COUNT:
            raise ValueError("paper validation CCL coverage changed")
        result.update(
            {
                "val_transition_count": len(validation),
                "val_ccl_valid_count": validation_valid_count,
                "val_ccl_valid_fraction": validation_valid_count / len(validation),
            }
        )
    return result


def _run_paper_preflight(
    config: ExperimentConfig,
    *,
    vqgan_checkpoint: str | Path | None,
) -> dict[str, Any]:
    contract = config.diffusion.paper
    if contract is None:
        raise ValueError("paper preflight requires a paper runtime contract")
    report: dict[str, Any] = {
        "config": str(config.path),
        "backend": config.data.backend,
        "ready": False,
        "data_ready": False,
        "data_identity_ready": False,
        "paper_data_ready": False,
        "clip_snapshot_ready": False,
        "vqgan_checkpoint_ready": False,
        "warm_start_checkpoint_ready": False,
        "warm_start_metadata_ready": False,
        "clip_model_id": contract.clip_model_id,
        "clip_revision": contract.clip_revision,
        "action_vocabulary_sha256": action_vocabulary_sha256(),
    }

    try:
        snapshot = _resolve_paper_clip_snapshot(contract)
        report["clip_snapshot"] = str(snapshot.resolve())
        report["clip_snapshot_ready"] = True
    except Exception as error:
        report["clip_snapshot_error"] = str(error)

    try:
        loaded = load_transition_records(
            config.data.bundle_json,
            config.data.phase_manifest_csv,
            backend=config.data.backend,
        )
        report["data_ready"] = True
        report["transition_count"] = len(loaded.records)
        report["split_counts"] = loaded.split_counts
        report["data_contract_sha256"] = loaded.data_contract_sha256
        report["bundle_contract_sha256"] = loaded.bundle_contract_sha256
    except Exception as error:
        report["data_error"] = str(error)
        return report

    try:
        phase_manifest_sha256 = sha256_file(config.data.phase_manifest_csv)
        report["phase_manifest_sha256"] = phase_manifest_sha256
    except Exception as error:
        report["phase_manifest_error"] = str(error)
        phase_manifest_sha256 = None
    report["data_identity_ready"] = bool(
        loaded.data_contract_sha256 == contract.data_contract_sha256
        and loaded.bundle_contract_sha256 == contract.bundle_contract_sha256
        and phase_manifest_sha256 == contract.phase_manifest_sha256
        and report["action_vocabulary_sha256"]
        == contract.action_vocabulary_sha256
    )

    try:
        report.update(
            _audit_paper_data(
                loaded,
                contract,
                ccl_seed=config.runtime.seed,
            )
        )
        report["paper_data_ready"] = True
    except Exception as error:
        report["paper_data_error"] = str(error)

    strict_a = loaded.bundle_schema_version == REGISTERED_STRICT_A_BUNDLE_SCHEMA
    if strict_a:
        report["roi_cache_inventory_ready"] = False
        report["roi_cache_sample_ready"] = False
        try:
            if config.data.roi_cache_dir is None:
                raise ValueError(
                    "registered Strict-A paper preflight requires its read-only "
                    "ROI cache"
                )
            cache = RegisteredStrictAROICache(
                config.data.roi_cache_dir,
                bundle_json=config.data.bundle_json,
                output_shape_zyx=config.data.output_shape_zyx,
            )
        except Exception as error:
            report["roi_cache_inventory_error"] = str(error)
            report["roi_cache_sample_error"] = str(error)
        else:
            try:
                split_visits = load_split_visit_records(
                    config.data.bundle_json,
                    config.data.phase_manifest_csv,
                    backend=config.data.backend,
                )
                required_visit_ids = sorted(split_visits.visits)
                for visit_id in required_visit_ids:
                    cache_path = cache.root / f"{cache.cache_key(visit_id)}.pt"
                    if cache_path.is_symlink() or not cache_path.is_file():
                        raise ValueError(
                            "registered Strict-A cache is unavailable: "
                            f"{visit_id}"
                        )
                report["roi_cache_inventory_count"] = len(required_visit_ids)
                report["roi_cache_inventory_ready"] = True
            except Exception as error:
                report["roi_cache_inventory_error"] = str(error)
            try:
                sample_record = loaded.records[0]
                sample_visit_ids = [
                    sample_record.source_visit_id,
                    sample_record.target_visit_id,
                ]
                samples = [
                    cache.load(loaded.visits[visit_id])
                    for visit_id in sample_visit_ids
                ]
                report["roi_cache_sample_visit_ids"] = sample_visit_ids
                report["roi_cache_sample_shapes"] = [
                    list(sample.image.shape) for sample in samples
                ]
                report["roi_cache_sample_ready"] = True
            except Exception as error:
                report["roi_cache_sample_error"] = str(error)

    if vqgan_checkpoint is None:
        report["vqgan_checkpoint_error"] = "required"
    else:
        vqgan_path = Path(vqgan_checkpoint)
        report["vqgan_checkpoint_path"] = str(vqgan_path)
        try:
            actual_vqgan_sha256 = sha256_file(vqgan_path)
            report["vqgan_sha256"] = actual_vqgan_sha256
            report["vqgan_checkpoint_ready"] = (
                actual_vqgan_sha256 == contract.vqgan_sha256
            )
            if not report["vqgan_checkpoint_ready"]:
                report["vqgan_checkpoint_error"] = "SHA256 mismatch"
        except Exception as error:
            report["vqgan_checkpoint_error"] = str(error)

    random_initialization = getattr(contract, "initialization_method", None) == "random"
    if random_initialization:
        report["initialization_method"] = "random"
        report["warm_start_not_applicable"] = True
    else:
        warm_path = contract.warm_start_path
        report["warm_start_checkpoint_path"] = str(warm_path)
        try:
            actual_warm_start_sha256 = sha256_file(warm_path)
            report["warm_start_sha256"] = actual_warm_start_sha256
            report["warm_start_checkpoint_ready"] = (
                actual_warm_start_sha256 == contract.warm_start_sha256
            )
            if not report["warm_start_checkpoint_ready"]:
                report["warm_start_checkpoint_error"] = "SHA256 mismatch"
        except Exception as error:
            report["warm_start_checkpoint_error"] = str(error)
        if report["warm_start_checkpoint_ready"]:
            try:
                report.update(_validate_paper_warm_start_metadata(warm_path, contract))
                report["warm_start_metadata_ready"] = True
            except Exception as error:
                report["warm_start_metadata_error"] = str(error)

    required = [
        "data_ready",
        "data_identity_ready",
        "paper_data_ready",
        "clip_snapshot_ready",
        "vqgan_checkpoint_ready",
    ]
    if not random_initialization:
        required.extend(
            ("warm_start_checkpoint_ready", "warm_start_metadata_ready")
        )
    if strict_a:
        required.extend(("roi_cache_inventory_ready", "roi_cache_sample_ready"))
    report["ready"] = all(report[field] is True for field in required)
    return report


def run_preflight(
    config_path: str | Path,
    *,
    prepare_sample: bool = False,
    ct_checkpoint: str | Path = CT_DEFAULT_PATH,
    vqgan_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    if getattr(getattr(config, "diffusion", None), "paper", None) is not None:
        if prepare_sample:
            raise ValueError("paper preflight does not support --prepare-sample")
        return _run_paper_preflight(
            config,
            vqgan_checkpoint=vqgan_checkpoint,
        )
    report: dict[str, Any] = {
        "config": str(config.path),
        "backend": config.data.backend,
        "ready": False,
        "data_ready": False,
        "split_visit_ready": False,
        "ct_checkpoint_ready": False,
        "medgemma_cache_ready": False,
        "ct_repository": CT_REPOSITORY,
        "ct_revision": CT_REVISION,
        "ct_expected_sha256": CT_EXPECTED_SHA256,
    }
    checkpoint_path = Path(ct_checkpoint)
    if checkpoint_path.is_file():
        actual_ct_sha = sha256_file(checkpoint_path)
        report["ct_checkpoint_path"] = str(checkpoint_path.resolve())
        report["ct_checkpoint_sha256"] = actual_ct_sha
        report["ct_checkpoint_ready"] = actual_ct_sha == CT_EXPECTED_SHA256
    else:
        report["ct_checkpoint_path"] = str(checkpoint_path)
        report["ct_checkpoint_error"] = "missing"
    medgemma_files = (
        MEDGEMMA_SNAPSHOT / "config.json",
        MEDGEMMA_SNAPSHOT / "tokenizer.json",
        MEDGEMMA_SNAPSHOT / "model-00001-of-00002.safetensors",
        MEDGEMMA_SNAPSHOT / "model-00002-of-00002.safetensors",
    )
    report["medgemma_snapshot"] = str(MEDGEMMA_SNAPSHOT)
    report["medgemma_cache_ready"] = all(path.is_file() for path in medgemma_files)

    if config.data.backend == "registered_t0":
        counts = _locked_registration_counts(config)
        report["registration_status_counts"] = counts
        report["registration_incomplete_count"] = sum(
            count
            for status, count in counts.items()
            if status not in ACCEPTED_REGISTRATION_STATUSES
        )
    try:
        loaded = load_transition_records(
            config.data.bundle_json,
            config.data.phase_manifest_csv,
            backend=config.data.backend,
        )
    except Exception as exc:
        report["data_error"] = str(exc)
        return report

    patients_by_fold: dict[str, set[str]] = {fold: set() for fold in ("train", "val", "test")}
    for record in loaded.records:
        patients_by_fold[record.fold].add(record.patient_id)
    if any(
        patients_by_fold[left] & patients_by_fold[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("preflight found patient leakage across folds")
    report.update(
        {
            "data_ready": True,
            "transition_count": len(loaded.records),
            "split_counts": loaded.split_counts,
            "patient_counts": {
                fold: len(patients) for fold, patients in patients_by_fold.items()
            },
            "visit_count": len(loaded.visits),
            "locked_transition_phase_count_distribution": dict(
                sorted(Counter(visit.dce0.n_times for visit in loaded.visits.values()).items())
            ),
            "data_contract_sha256": loaded.data_contract_sha256,
        }
    )
    try:
        split_visits = load_split_visit_records(
            config.data.bundle_json,
            config.data.phase_manifest_csv,
            backend=config.data.backend,
        )
        report["split_visit_count"] = len(split_visits.visits)
        report["split_visit_ready"] = True
        report["split_visit_fold_counts"] = dict(
            sorted(Counter(split_visits.folds.values()).items())
        )
        report["split_visit_patient_counts"] = {
            fold: len(
                {
                    split_visits.visits[visit_id].patient_id
                    for visit_id, visit_fold in split_visits.folds.items()
                    if visit_fold == fold
                }
            )
            for fold in ("train", "val", "test")
        }
        report["phase_count_distribution"] = dict(
            sorted(
                Counter(
                    visit.dce0.n_times for visit in split_visits.visits.values()
                ).items()
            )
        )
    except Exception as exc:
        report["split_visit_error"] = str(exc)
    if prepare_sample:
        record = loaded.records[0]
        if (
            getattr(loaded, "bundle_schema_version", None)
            == REGISTERED_STRICT_A_BUNDLE_SCHEMA
        ):
            if config.data.roi_cache_dir is None:
                raise ValueError(
                    "registered Strict-A preflight requires its read-only ROI cache"
                )
            cache = RegisteredStrictAROICache(
                config.data.roi_cache_dir,
                bundle_json=config.data.bundle_json,
                output_shape_zyx=config.data.output_shape_zyx,
            )
            sample = cache.load(loaded.visits[record.source_visit_id])
        else:
            cache = DCE0Cache(
                config.data.cache_dir, output_shape_zyx=config.data.output_shape_zyx
            )
            sample = cache.load_or_create(
                loaded.visits[record.source_visit_id], backend=config.data.backend
            )
        report["sample"] = {
            "transition_id": record.transition_id,
            "source_visit_id": sample.visit_id,
            "phase_index": sample.phase_index,
            "n_times": sample.n_times,
            "shape": list(sample.image.shape),
            "image_sha256": sample.image_sha256,
            "value_range": [float(sample.image.min()), float(sample.image.max())],
            "mask_voxels": int(sample.mask.sum()),
        }
        if sample.valid_foreground is None:
            report["sample"]["normalization_percentiles"] = [0.5, 99.5]
        else:
            report["sample"]["normalization"] = sample.metadata["normalization"]
            report["sample"]["valid_foreground_voxels"] = int(
                sample.valid_foreground.sum()
            )
    report["ready"] = bool(
        report["data_ready"]
        and report["split_visit_ready"]
        and report["ct_checkpoint_ready"]
        and report["medgemma_cache_ready"]
    )
    return report
