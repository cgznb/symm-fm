import io
import os
import re
import zipfile

import numpy as np
import pandas as pd
import torch

from src.tabular import DRUGS, DRUG_COLS, arm_binaries

_AGE_MEAN = 48.75              # cohort mean (years), used to impute
_REGISTERED_TIMEPOINT = re.compile(r"^T([0-3])$")

# HR, HER2, age, menopause, MP, 12 drugs
TABULAR_DIM = 5 + len(DRUGS)
TABULAR_FEATURE_NAMES = (
    "HR",
    "HER2",
    "age_div_100",
    "menopause",
    "MP",
    *DRUG_COLS,
)


def _binary(row, col, default=0.0):
    v = row.get(col, np.nan)
    return default if pd.isna(v) else float(v)


def build_tabular_full(row):
    """Segmentation-free clinical/treatment vector from the enriched CSV (17-D): HR, HER2, age/100,
    menopause, MP (MammaPrint), and 12 treatment-drug indicators. Drug indicators use the `drug_*`
    columns when present, otherwise `Arm` is parsed on the fly."""
    base = [
        _binary(row, "HR"),
        _binary(row, "HER2"),
        _binary(row, "age", default=_AGE_MEAN) / 100.0,
        _binary(row, "menopause", default=0.5),
        _binary(row, "MP", default=0.0),                  # MammaPrint
    ]
    if DRUG_COLS[0] in row.index:                         # use the drug_* columns when present
        drug_oh = [0.0 if pd.isna(row[c]) else float(row[c]) for c in DRUG_COLS]
    else:                                                 # otherwise derive them from the Arm string
        b = arm_binaries(row.get("Arm"))
        drug_oh = [float(b[d]) for d in DRUGS]
    return np.asarray(base + drug_oh, dtype=np.float32)


def load_ids(path):
    """String patient ids, one per line."""
    with open(path) as f:
        return [x.strip() for x in f if x.strip()]


def _l2norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _vec_from_tensor(e, *, require_finite=False):
    if not isinstance(e, torch.Tensor):
        raise TypeError("embedding artifact must contain a torch.Tensor")
    if e.dim() > 1:
        e = e.squeeze()
    v = e.detach().cpu().float().numpy().astype(np.float32).reshape(-1)
    if require_finite and not np.isfinite(v).all():
        raise ValueError("embedding contains non-finite values")
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)   # replace non-finite values with zero
    return _l2norm(v)


def _expected_registered_timepoints(row):
    value = row.get("registered_timepoints", np.nan)
    if pd.isna(value):
        return None
    tokens = [token.strip() for token in str(value).split(";") if token.strip()]
    parsed = []
    for token in tokens:
        match = _REGISTERED_TIMEPOINT.fullmatch(token)
        if match is None:
            raise ValueError(f"invalid registered timepoint token: {token!r}")
        parsed.append(int(match.group(1)))
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError("registered_timepoints must contain unique T0-T3 tokens")
    return tuple(parsed)


# zip member name -> capture  .../<pid>/<pid>_T<t>.pt   (any prefix / nesting allowed)
_PT_RE = re.compile(r"([^/]+)/\1_T(\d)\.pt$")


class EmbStore:
    """Reads {pid}/{pid}_T{t}.pt from a directory OR a .zip (in-memory, no unzip)."""

    def __init__(self, path):
        if str(path).lower().endswith(".zip"):
            self.mode = "zip"
            self.zf = zipfile.ZipFile(path)
            self.index = {}
            for name in self.zf.namelist():
                m = _PT_RE.search(name)
                if m:
                    self.index[(m.group(1), int(m.group(2)))] = name
            if not self.index:
                raise FileNotFoundError(f"No .pt embeddings found in zip {path}")
        else:
            self.mode = "dir"
            self.root = path

    def exists(self, pid, t):
        if self.mode == "zip":
            return (str(pid), int(t)) in self.index
        return os.path.exists(os.path.join(self.root, str(pid), f"{pid}_T{t}.pt"))

    def load(self, pid, t, *, require_finite=False):
        if self.mode == "zip":
            with self.zf.open(self.index[(str(pid), int(t))]) as f:
                e = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=True)
        else:
            e = torch.load(os.path.join(self.root, str(pid), f"{pid}_T{t}.pt"),
                           map_location="cpu", weights_only=True)
        return _vec_from_tensor(e, require_finite=require_finite)

    def infer_dim(self, pids):
        for pid in pids:
            for t in range(4):
                if self.exists(pid, t):
                    return self.load(pid, t).shape[0]
        raise FileNotFoundError("No .pt embeddings found for the given pids")


def load_split(store, pids, meta_by_pid, cdim):
    """Load one split: the per-timepoint embedding array `embs` [n, 4, dim], the availability mask
    `masks` [n, 4] (a timepoint is present iff its embedding file exists), the clinical vectors,
    labels, and elapsed days from T0 (which feed the elapsed-time positional encoding)."""
    dim = store.infer_dim(pids)
    n = len(pids)
    embs = np.zeros((n, 4, dim), dtype=np.float32)
    masks = np.zeros((n, 4), dtype=np.float32)
    clinical = np.zeros((n, cdim), dtype=np.float32)
    labels = np.zeros(n, dtype=np.float32)
    days = np.zeros((n, 4), dtype=np.float32)

    for i, pid in enumerate(pids):
        row = meta_by_pid[str(pid)]
        clinical[i] = build_tabular_full(row)
        labels[i] = float(row["pCR"])
        expected = _expected_registered_timepoints(row)
        if expected is None:
            timepoints = range(4)
        else:
            actual = {t for t in range(4) if store.exists(pid, t)}
            missing = sorted(set(expected) - actual)
            unexpected = sorted(actual - set(expected))
            if missing:
                raise FileNotFoundError(
                    f"Missing registered embeddings for patient {pid}: {missing}"
                )
            if unexpected:
                raise ValueError(
                    f"Unexpected registered embeddings for patient {pid}: {unexpected}"
                )
            timepoints = expected
        for t in timepoints:
            dt = row.get(f"days_T{t}", np.nan)
            days[i, t] = float(t * 40) if pd.isna(dt) else float(dt)   # nominal 40-day spacing if missing
            if store.exists(pid, t):
                vector = store.load(pid, t, require_finite=expected is not None)
                if vector.shape != (dim,):
                    raise ValueError(
                        f"Embedding dimension mismatch for patient {pid} T{t}: "
                        f"expected {dim}, found {vector.shape[0]}"
                    )
                masks[i, t] = 1.0
                embs[i, t] = vector
        if masks[i].sum() == 0:
            raise FileNotFoundError(f"No embeddings found for patient {pid}")

    return {"pids": list(pids), "embs": embs, "dim": dim, "masks": masks,
            "clinical": clinical, "labels": labels, "days": days}


def load_all_splits(emb_dir, meta_csv, train_ids, val_ids, test_ids, pid_col="pid"):
    """Load the train/val/test splits from one embedding store (directory or .zip) and the enriched
    metadata CSV (which supplies the 17-D clinical vector and the elapsed-days columns)."""
    meta = pd.read_csv(meta_csv)
    if pid_col not in meta.columns or meta[pid_col].astype(str).duplicated().any():
        raise ValueError(f"metadata column {pid_col!r} is missing or contains duplicates")
    meta_by_pid = {str(r[pid_col]): r for _, r in meta.iterrows()}
    split_ids = {
        "train": list(train_ids), "val": list(val_ids), "test": list(test_ids)
    }
    for name, ids in split_ids.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} split contains duplicate patient IDs")
        missing = sorted(set(ids) - set(meta_by_pid))
        if missing:
            raise ValueError(f"{name} split has {len(missing)} patients missing from metadata")
    if (
        set(train_ids) & set(val_ids)
        or set(train_ids) & set(test_ids)
        or set(val_ids) & set(test_ids)
    ):
        raise ValueError("train, validation, and test patient splits overlap")
    store = EmbStore(emb_dir)
    cdim = len(build_tabular_full(next(iter(meta_by_pid.values()))))   # 17
    loaded = {
        "train": load_split(store, split_ids["train"], meta_by_pid, cdim),
        "val": load_split(store, split_ids["val"], meta_by_pid, cdim),
        "test": load_split(store, split_ids["test"], meta_by_pid, cdim),
    }
    dimensions = {split["dim"] for split in loaded.values()}
    if len(dimensions) != 1:
        raise ValueError(f"embedding dimensions differ across splits: {sorted(dimensions)}")
    return loaded
