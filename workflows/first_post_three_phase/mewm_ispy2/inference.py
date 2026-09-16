from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import nibabel as nib
import numpy as np

from .segmenter import GeneratedSegmentation
from .data import PreparedVisit


REQUIRED_PREDICTION_FILES = (
    *(f"sample_{index:02d}_dce.nii.gz" for index in range(8)),
    *(f"sample_{index:02d}_mask.nii.gz" for index in range(8)),
    *(f"sample_{index:02d}_mask_probability.nii.gz" for index in range(8)),
    "mean_dce.nii.gz",
    "mean_mask_probability.nii.gz",
    "entropy.nii.gz",
    "prediction.json",
)


def _contains_target_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            "target" in str(key).lower() or _contains_target_key(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_target_key(child) for child in value)
    return False


def _save_nifti_atomic(path: Path, array: np.ndarray, affine: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.nii.gz")
    nib.save(nib.Nifti1Image(array, affine), temporary)
    os.replace(temporary, path)


def publish_prediction(
    output_directory: str | Path,
    result: GeneratedSegmentation,
    *,
    metadata: dict[str, Any],
    affine: np.ndarray | None = None,
) -> Path:
    if _contains_target_key(metadata):
        raise ValueError("source-only prediction metadata cannot contain target information")
    if result.samples.shape[0] != 8:
        raise ValueError("prediction publishing requires exactly eight samples")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    output_affine = np.eye(4, dtype=np.float64) if affine is None else np.asarray(affine)
    samples = result.samples.detach().cpu().numpy()
    probabilities = result.sample_probabilities.detach().cpu().numpy()
    masks = result.sample_masks.detach().cpu().numpy()
    for index in range(8):
        _save_nifti_atomic(
            output / f"sample_{index:02d}_dce.nii.gz",
            samples[index, 0].astype(np.float32),
            output_affine,
        )
        _save_nifti_atomic(
            output / f"sample_{index:02d}_mask.nii.gz",
            masks[index, 0].astype(np.uint8),
            output_affine,
        )
        _save_nifti_atomic(
            output / f"sample_{index:02d}_mask_probability.nii.gz",
            probabilities[index, 0].astype(np.float32),
            output_affine,
        )
    _save_nifti_atomic(
        output / "mean_dce.nii.gz",
        result.mean_dce.detach().cpu().numpy()[0].astype(np.float32),
        output_affine,
    )
    _save_nifti_atomic(
        output / "mean_mask_probability.nii.gz",
        result.mean_probability.detach().cpu().numpy()[0].astype(np.float32),
        output_affine,
    )
    _save_nifti_atomic(
        output / "entropy.nii.gz",
        result.entropy.detach().cpu().numpy()[0].astype(np.float32),
        output_affine,
    )
    prediction = {
        "schema_version": "mewm_ispy2_prediction_v1",
        "sample_count": 8,
        **metadata,
        "artifacts": list(REQUIRED_PREDICTION_FILES[:-1]),
    }
    temporary_json = output / f".prediction.json.{os.getpid()}.tmp"
    temporary_json.write_text(json.dumps(prediction, indent=2, sort_keys=True))
    os.replace(temporary_json, output / "prediction.json")
    assert_prediction_complete(output)
    return output


def assert_prediction_complete(output_directory: str | Path) -> None:
    output = Path(output_directory)
    missing = [name for name in REQUIRED_PREDICTION_FILES if not (output / name).is_file()]
    if missing:
        raise ValueError(f"prediction is incomplete; missing {len(missing)} artifacts")
    payload = json.loads((output / "prediction.json").read_text())
    if payload.get("sample_count") != 8 or _contains_target_key(payload):
        raise ValueError("prediction manifest violates the source-only contract")


def validate_prediction_identity(
    prediction_manifest: Mapping[str, Any],
    *,
    expected_backend: str,
    expected_data_contract_sha256: str,
    expected_source: PreparedVisit,
) -> None:
    if prediction_manifest.get("data_backend") != expected_backend:
        raise ValueError("prediction data backend does not match evaluation")
    if (
        prediction_manifest.get("data_contract_sha256")
        != expected_data_contract_sha256
    ):
        raise ValueError("prediction data contract does not match evaluation")
    expected_fields = {
        "source_visit_id": expected_source.visit_id,
        "source_phase_index": expected_source.phase_index,
        "source_n_times": expected_source.n_times,
        "source_image_sha256": expected_source.image_sha256,
    }
    for field, expected in expected_fields.items():
        if prediction_manifest.get(field) != expected:
            raise ValueError(f"prediction source identity mismatch: {field}")


T = TypeVar("T")


def load_target_after_generation(
    output_directory: str | Path, target_loader: Callable[[], T]
) -> T:
    assert_prediction_complete(output_directory)
    return target_loader()
