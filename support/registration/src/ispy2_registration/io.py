from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


TERMINAL_OUTPUT_STATUSES = {"fixed_reference", "deformable", "rigid_fallback"}


def read_nifti(path: Path | str, *, dtype=None) -> np.ndarray:
    image = nib.load(str(path))
    array = np.asanyarray(image.dataobj)
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    else:
        array = np.asarray(array)
    return array


def write_nifti_identity(path: Path | str, array: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array)
    if data.ndim not in {3, 4}:
        raise ValueError("NIfTI output must be a 3D image or 3D vector field")
    if not np.isfinite(data).all():
        raise ValueError(f"refusing to write nonfinite NIfTI: {output}")
    image = nib.Nifti1Image(data, np.eye(4, dtype=np.float64))
    image.set_qform(np.eye(4), code=1)
    image.set_sform(np.eye(4), code=1)
    nib.save(image, str(output))
    return output


def write_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return output


def output_image_path(
    output_root: Path,
    patient_id: str,
    visit: str,
    modality: str,
    source_path: Path,
) -> Path:
    return Path(output_root) / patient_id / visit / modality / source_path.name


@contextmanager
def atomic_output_directory(final_directory: Path | str) -> Iterator[Path]:
    final = Path(final_directory)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=final.parent)
    )
    try:
        yield temporary
        stale: Path | None = None
        if final.exists():
            stale = final.parent / f".{final.name}.stale-{uuid.uuid4().hex}"
            os.replace(final, stale)
        try:
            os.replace(temporary, final)
        except Exception:
            if stale is not None and stale.exists() and not final.exists():
                os.replace(stale, final)
            raise
        if stale is not None:
            shutil.rmtree(stale)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def pair_is_complete(pair_directory: Path | str) -> bool:
    pair_dir = Path(pair_directory)
    status_path = pair_dir / "registration.json"
    if not status_path.is_file():
        return False
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("registration_status") not in TERMINAL_OUTPUT_STATUSES:
        return False
    output_files = payload.get("output_files")
    if not isinstance(output_files, list) or not output_files:
        return False
    return all(isinstance(path, str) and Path(path).is_file() for path in output_files)
