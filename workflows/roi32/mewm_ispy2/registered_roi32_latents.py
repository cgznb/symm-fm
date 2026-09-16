"""One frozen continuous VQ latent per visit, with train-only channel moments."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import registered_roi32_runtime as runtime
from . import registered_roi32_vq as vq
from .registered_roi32_data import CropDataset, file_identity, read_json, verify_identity, visit_filename, write_json


def latent_filename(visit_id):
    return visit_filename(visit_id).replace(".npz", ".npy")


def save_latent(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def fit_channel_statistics(latents):
    sums = np.zeros(8, dtype=np.float64)
    squares = np.zeros(8, dtype=np.float64)
    count = 0
    visits = []
    for visit_id, latent in latents:
        value = np.asarray(latent, dtype=np.float64).reshape(8, -1)
        if not np.isfinite(value).all():
            raise FloatingPointError("Non-finite cached continuous latent")
        sums += value.sum(axis=1)
        squares += np.square(value).sum(axis=1)
        count += value.shape[1]
        visits.append(visit_id)
    if not count or len(visits) != len(set(visits)):
        raise ValueError("Latent statistics require unique training visits")
    mean = sums / count
    std = np.sqrt(np.maximum(0, squares / count - mean**2))
    if not np.isfinite(std).all() or np.any(std <= 1e-8):
        raise FloatingPointError("Zero-variance latent channel; FM is blocked")
    return {"mean": mean.tolist(), "std": std.tolist(), "element_count_per_channel": count,
            "fit_split": "train", "fit_visit_ids": sorted(visits), "scope": "unique_train_visits_continuous_prequantization"}


def contract_for(config):
    return runtime.stage_contract(config, "latents", [__file__, vq.__file__, runtime.__file__],
                                  codec=file_identity(Path(config["output_dir"]) / "vq" / "best.pt"))


@torch.no_grad()
def extract(config, device, batch_size):
    contract = contract_for(config)
    if runtime.stage_complete(config, "latents", contract):
        return True
    root = Path(config["output_dir"]) / "latents"
    root.mkdir(parents=True, exist_ok=True)
    binding = root / "binding.json"
    if binding.exists():
        if read_json(binding) != contract:
            raise ValueError("Partial latent cache uses another codec or preprocessing")
    else:
        write_json(binding, contract)
    dataset = CropDataset(config)
    pending = [r for r in dataset.records if not (root / "raw" / latent_filename(r["visit_id"])).is_file()]
    loader = runtime.evaluation_loader(CropDataset(config, records=pending), batch_size, config["runtime"]["loader_workers"])
    model, _ = vq.load_frozen(config, device)
    codebook = {k: value.clone() for k, value in model.quantizer.state_dict().items()}
    count = len(dataset) - len(pending)
    for batch in loader:
        with runtime.autocast(device):
            latent = model.encode_continuous(batch["image"].to(device, non_blocking=True)).float()
        if tuple(latent.shape[1:]) != (8, 8, 32, 32) or not torch.isfinite(latent).all():
            raise FloatingPointError("Invalid frozen VQ latent")
        stored = latent.cpu().numpy().astype(np.float16)
        if not np.isfinite(stored).all():
            raise FloatingPointError("Continuous latent overflows FP16 cache storage")
        for visit_id, value in zip(batch["visit_id"], stored, strict=True):
            save_latent(root / "raw" / latent_filename(visit_id), value)
        count += len(latent)
        if count % 100 < len(latent) or count == len(dataset):
            runtime.log_event(config, "latents", "encoding", completed=count, total=len(dataset))
            runtime.periodic_guard(config, contract)
        if runtime.STOP_REQUESTED:
            return False
    for key, value in codebook.items():
        if not torch.equal(value, model.quantizer.state_dict()[key]):
            raise RuntimeError("Encoding changed the frozen VQ codebook")
    identities = []
    for row in dataset.records:
        path = root / "raw" / latent_filename(row["visit_id"])
        array = np.load(path, allow_pickle=False)
        if array.shape != (8, 8, 32, 32) or array.dtype != np.float16 or not np.isfinite(array).all():
            raise ValueError("Invalid cached latent on resume")
        identities.append(file_identity(path))
    training = (r for r in dataset.records if r["fold"] == "train")
    statistics = fit_channel_statistics((r["visit_id"], np.load(root / "raw" / latent_filename(r["visit_id"]), allow_pickle=False)) for r in training)
    statistics["codec"] = contract["codec"]
    write_json(root / "statistics.json", statistics)
    write_json(root / "cache_manifest.json", {"schema": config["schema"], "cache_files": identities, "contract": contract})
    runtime.stage_finished(config, "latents", contract, [root / "statistics.json", root / "cache_manifest.json"], visits=len(dataset))
    return True


class LatentPairs(Dataset):
    def __init__(self, config, split=None):
        self.root = Path(config["output_dir"]) / "latents"
        if not runtime.stage_complete(config, "latents", contract_for(config)):
            raise ValueError("Latent preparation has not passed its gate")
        self.inventory = read_json(Path(config["output_dir"]) / "data" / "inventory.json")
        self.records = [r for r in self.inventory["pairs"] if split is None or r["split"] == split]
        self.statistics = read_json(self.root / "statistics.json")
        expected = sorted(r["visit_id"] for r in self.inventory["visits"] if r["fold"] == "train")
        if self.statistics["fit_split"] != "train" or self.statistics["fit_visit_ids"] != expected:
            raise ValueError("Latent statistics contain the wrong fit cohort")
        self.mean = np.array(self.statistics["mean"], dtype=np.float32).reshape(8, 1, 1, 1)
        self.std = np.array(self.statistics["std"], dtype=np.float32).reshape(8, 1, 1, 1)
        for identity in read_json(self.root / "cache_manifest.json")["cache_files"]:
            verify_identity(identity)

    def __len__(self):
        return len(self.records)

    def read(self, visit_id):
        raw = np.load(self.root / "raw" / latent_filename(visit_id), allow_pickle=False).astype(np.float32)
        value = (raw - self.mean) / self.std
        if value.shape != (8, 8, 32, 32) or not np.isfinite(value).all():
            raise ValueError("Invalid normalized latent")
        return torch.from_numpy(value)

    def __getitem__(self, index):
        record = self.records[index]
        return {"earlier_latent": self.read(record["earlier_visit_id"]), "later_latent": self.read(record["later_visit_id"]), "record": record}


@torch.no_grad()
def decode_normalized(model, latent, statistics):
    mean = torch.tensor(statistics["mean"], device=latent.device, dtype=torch.float32).view(1, 8, 1, 1, 1)
    std = torch.tensor(statistics["std"], device=latent.device, dtype=torch.float32).view(1, 8, 1, 1, 1)
    continuous = latent.float() * std + mean
    if model.training:
        raise ValueError("Decoder requires a frozen evaluation-mode VQ model")
    quantized, _ = model.quantizer(continuous)
    with runtime.autocast(latent.device):
        return model.decode(quantized).float()
