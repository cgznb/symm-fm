"""Small shared runtime for the independent registered ROI32 experiment."""

from __future__ import annotations

import fcntl
import math
import os
import random
import signal
import time
from collections import Counter
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .registered_roi32_data import check_disk, file_identity, read_json, timestamp, verify_identity, write_json

STOP_REQUESTED = False


@contextmanager
def output_lock(root, inherited_fd=None):
    handle = os.fdopen(inherited_fd, "a+") if inherited_fd is not None else (Path(root) / "run.lock").open("a+")
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("An ROI32 controller already owns this output directory") from None
        yield handle


def install_signals():
    def stop(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@contextmanager
def fixed_rng(seed):
    state = rng_state()
    try:
        seed_all(seed)
        yield
    finally:
        restore_rng(state)


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, contract, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["contract"] != contract:
        raise ValueError("Checkpoint configuration, prepared data, or runtime source changed")
    return payload


def stage_contract(config, stage, source_files, **extra):
    root = Path(config["output_dir"])
    identities = [file_identity(root / "data" / filename) for filename in ("inventory.json", "normalization.json", "COMPLETE.json")]
    sources = [file_identity(path) for path in source_files]
    return {"schema": config["schema"], "stage": stage, "configuration": config,
            "prepared_data": identities, "runtime_sources": sources,
            "library_versions": {name: version(name) for name in ("torch", "monai", "numpy", "scipy", "SimpleITK", "nibabel", "torchmetrics")},
            **extra}


def verify_contract_files(contract):
    if isinstance(contract, dict):
        if set(contract) == {"path", "size_bytes", "mtime_ns"}:
            verify_identity(contract)
        else:
            for value in contract.values():
                verify_contract_files(value)
    elif isinstance(contract, list):
        for value in contract:
            verify_contract_files(value)


def log_event(config, stage, event, **values):
    import json
    root = Path(config["output_dir"]) / stage
    root.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "event": event, "updated_at": timestamp(), **values}
    line = json.dumps(payload, allow_nan=False)
    with (root / "metrics.jsonl").open("a") as handle:
        handle.write(line + "\n")
    write_json(root / "progress.json", payload)
    print(line, flush=True)


def finite_gradients(parameters, max_norm):
    parameters = list(parameters)
    if not any(p.grad is not None for p in parameters):
        raise FloatingPointError("Optimizer has no gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite gradient norm")
    return float(norm)


def evaluation_loader(dataset, batch_size, workers=0, collate_fn=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), collate_fn=collate_fn,
                      generator=torch.Generator().manual_seed(1729))


class TrainingBatches:
    """Deterministic epoch order with explicit consumed-batch resume state."""

    def __init__(self, dataset, *, batch_size, effective_batch, seed, workers=0, balanced=False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.effective_batch = effective_batch
        self.seed = seed
        self.workers = workers
        self.balanced = balanced
        self.collate_fn = collate_fn
        self.epoch = 0
        self.cursor = 0
        self.iterator = None
        self.loader = None
        if effective_batch % batch_size:
            raise ValueError("Physical batch must divide effective batch")

    def epoch_batches(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        size = len(self.dataset)
        count = math.ceil(size / self.effective_batch) * self.effective_batch
        if self.balanced:
            frequencies = Counter(r["patient_id"] for r in self.dataset.records)
            weights = torch.tensor([1 / frequencies[r["patient_id"]] for r in self.dataset.records], dtype=torch.float64)
            indices = torch.multinomial(weights, count, replacement=True, generator=generator).tolist()
        else:
            indices = torch.randperm(size, generator=generator).tolist()
            indices += (indices * math.ceil((count - size) / size))[:count - size]
        return [indices[i:i + self.batch_size] for i in range(0, count, self.batch_size)]

    def __next__(self):
        while True:
            if self.iterator is None:
                batches = self.epoch_batches()
                if self.cursor >= len(batches):
                    self.epoch += 1
                    self.cursor = 0
                    batches = self.epoch_batches()
                self.loader = DataLoader(self.dataset, batch_sampler=batches[self.cursor:], num_workers=self.workers,
                                         pin_memory=torch.cuda.is_available(), collate_fn=self.collate_fn,
                                         generator=torch.Generator().manual_seed(self.seed + self.epoch + 1000000))
                self.iterator = iter(self.loader)
            try:
                batch = next(self.iterator)
                self.cursor += 1
                return batch
            except StopIteration:
                self.close()
                self.epoch += 1
                self.cursor = 0

    def close(self):
        if self.iterator is not None and hasattr(self.iterator, "_shutdown_workers"):
            self.iterator._shutdown_workers()
        self.iterator = None
        self.loader = None

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor, "batch_size": self.batch_size,
                "effective_batch": self.effective_batch, "seed": self.seed, "balanced": self.balanced}

    def load_state_dict(self, state):
        for key in ("batch_size", "effective_batch", "seed", "balanced"):
            if state[key] != getattr(self, key):
                raise ValueError("Training sampler contract changed")
        self.close()
        self.epoch, self.cursor = state["epoch"], state["cursor"]


def stage_complete(config, stage, contract):
    marker = Path(config["output_dir"]) / stage / "COMPLETE.json"
    if not marker.exists():
        return False
    result = read_json(marker)
    if result["contract"] != contract or result["status"] != "passed":
        raise ValueError("Completed stage belongs to another experiment/runtime")
    for identity in result.get("artifacts", []):
        verify_identity(identity)
    return True


def stage_finished(config, stage, contract, artifacts, **values):
    write_json(Path(config["output_dir"]) / stage / "COMPLETE.json",
               {"status": "passed", "contract": contract, "updated_at": timestamp(),
                "artifacts": [file_identity(path) for path in artifacts], **values})


def periodic_guard(config, contract):
    check_disk(config)
    verify_contract_files(contract)


class UpdateClock:
    def __init__(self, initial=0):
        self.initial = initial
        self.started = time.monotonic()

    def metrics(self, step, maximum):
        seconds = time.monotonic() - self.started
        rate = seconds / max(1, step - self.initial)
        return {"elapsed_seconds": seconds, "seconds_per_update": rate,
                "remaining_seconds_estimate": max(0, maximum - step) * rate}
