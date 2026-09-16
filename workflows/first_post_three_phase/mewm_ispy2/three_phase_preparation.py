"""Bounded phase staging and exact-grid reuse for formal MRI preparation."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import numpy as np

from .first_post_world_data import identity, read_json

LOCAL_PHASE_SCHEMA = "three_phase_verified_local_sources_v1"


def load_local_phase_sources(settings, inventory):
    root = Path(settings["output_dir"])
    path = root / "local_phase_sources.json"
    if not path.exists():
        return {}
    value = read_json(path)
    if (value["schema"] != LOCAL_PHASE_SCHEMA or value.get("verified_exact_content") is not True
            or value["inventory"] != identity(root / "inventory.json")
            or value["remote_root"] != settings["remote_root"]):
        raise ValueError("Verified local phase source contract changed")
    for evidence in value["evidence"]:
        if identity(evidence["path"]) != evidence["identity"]:
            raise ValueError("Local phase verification evidence changed")
    expected = {str(Path(remote).relative_to(settings["remote_root"])): visit["phase_bytes"][phase]
                for visit in inventory["visits"] for phase, remote in visit["remote_phases"].items()}
    for relative, source in value["files"].items():
        path = Path(source["path"])
        if (relative not in expected or ".." in Path(relative).parts or not path.is_absolute()
                or path.resolve().is_relative_to(root.resolve())
                or source["identity"]["size_bytes"] != expected[relative]):
            raise ValueError("Local phase must be a verified persistent native source")
        if identity(path) != source["identity"]:
            raise ValueError("Verified local native phase changed")
    return value["files"]


def phase_path(settings, visit, phase):
    relative = Path(visit["remote_phases"][phase]).relative_to(settings["remote_root"])
    if ".." in relative.parts:
        raise ValueError("Native phase leaves the staging root")
    source = settings.get("local_phase_sources", {}).get(str(relative))
    if source is not None:
        path = Path(source["path"])
        if identity(path) != source["identity"]:
            raise ValueError("Verified local native phase changed")
        return path
    return Path(settings["output_dir"]) / "staging" / relative


def grid_key(view):
    geometry = view["geometry"]
    return (view["visit_id"], *(tuple(geometry[field]) for field in (
        "shape_zyx", "spacing_xyz_mm", "origin_lps_mm", "direction_lps")))


def multiplex_arguments(arguments, socket):
    result = list(arguments)
    options = {"ControlMaster": "auto", "ControlPath": str(socket), "ControlPersist": "120"}
    for key, value in options.items():
        replaced = False
        for index in range(1, len(result)):
            if result[index - 1] == "-o" and result[index].startswith(key + "="):
                result[index] = f"{key}={value}"
                replaced = True
        if not replaced:
            result.extend(["-o", f"{key}={value}"])
    return result


class PhaseTransfer:
    def __init__(self, data, settings, *, reuse_connections=True):
        self.data, self.settings = data, settings
        self.reuse_connections = reuse_connections
        self.directory = tempfile.TemporaryDirectory(prefix="ispy2_phase_ssh_")
        original = data.ssh_arguments(settings)
        self.arguments = [multiplex_arguments(original, Path(self.directory.name) / f"qingyuan_{index}.sock")
                          if reuse_connections else original for index in range(settings["transfer_workers"])]
        self.processes, self.lock = set(), threading.Lock()
        self.cancelled = threading.Event()

    def __enter__(self):
        return self

    def _transfer(self, shard):
        index, names = shard
        if not names:
            return
        command = ["rsync", "-rt", "--partial-dir=.rsync-partial", "--from0", "--files-from=-",
                   "-e", shlex.join(self.arguments[index]), f"qingyuan:{self.settings['remote_root']}/",
                   str(Path(self.settings["output_dir"]) / "staging") + "/"]
        with self.lock:
            if self.cancelled.is_set():
                raise InterruptedError("Phase staging cancelled")
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            self.processes.add(process)
        try:
            _, error = process.communicate("\0".join(names) + "\0", timeout=1800)
            if process.returncode:
                raise RuntimeError(f"Phase transfer failed: {error[-1000:]}")
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
            raise
        finally:
            with self.lock:
                self.processes.discard(process)

    def stage(self, visits):
        began = time.perf_counter()
        root = Path(self.settings["remote_root"])
        staging = Path(self.settings["output_dir"]) / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        names, expected, sizes = set(), {}, {}
        local, cached = set(), set()
        for visit in visits:
            for phase, remote in visit["remote_phases"].items():
                relative = Path(remote).relative_to(root)
                name = str(relative)
                path = phase_path(self.settings, visit, phase)
                size = visit["phase_bytes"][phase]
                if path in expected and expected[path] != size:
                    raise ValueError("Conflicting native phase sizes")
                expected[path] = size
                sizes[name] = size
                if name in self.settings.get("local_phase_sources", {}):
                    if not path.is_file() or path.stat().st_size != size:
                        raise ValueError("Verified local native phase size differs from inventory")
                    local.add(name)
                elif not path.is_file() or path.stat().st_size != size:
                    names.add(name)
                else:
                    cached.add(name)
        if names:
            self.data.disk_gate(self.settings, required=sum(sizes[name] for name in names))
            unique = sorted(names)
            workers = self.settings["transfer_workers"]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(self._transfer, [(i, unique[i::workers]) for i in range(workers)]))
        if any(not p.is_file() or p.stat().st_size != size for p, size in expected.items()):
            raise ValueError("Staged native phase size differs from inventory")
        return {"transfer_seconds": time.perf_counter() - began, "phase_bytes": sum(sizes.values()),
                "local_phase_files": len(local), "local_phase_bytes": sum(sizes[name] for name in local),
                "staged_reused_files": len(cached), "remote_requested_files": len(names),
                "remote_requested_bytes": sum(sizes[name] for name in names)}

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            for process in self.processes:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def __exit__(self, *exc):
        self.cancel()
        try:
            if self.reuse_connections:
                for arguments in self.arguments:
                    try:
                        subprocess.run([*arguments, "-O", "exit", "qingyuan"], stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=10, check=False)
                    except subprocess.TimeoutExpired:
                        continue
        finally:
            self.directory.cleanup()


class PhasePrefetch:
    """Download at most one future patient's phases while the current one is processed."""

    def __init__(self, transfer, jobs, stopped):
        self.transfer, self.jobs, self.stopped = transfer, jobs, stopped
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = self.pool.submit(transfer.stage, jobs[0][3]) if jobs else None

    def __enter__(self):
        return self

    def take(self, position):
        began = time.perf_counter()
        while True:
            if self.stopped.is_set():
                raise InterruptedError("Preparation pause requested")
            try:
                metrics = self.future.result(timeout=0.2)
                break
            except TimeoutError:
                if self.future.done():
                    raise
                continue
        if position + 1 < len(self.jobs):
            self.future = self.pool.submit(self.transfer.stage, self.jobs[position + 1][3])
        else:
            self.future = None
        return {**metrics, "transfer_wait_seconds": time.perf_counter() - began}

    def __exit__(self, *exc):
        self.transfer.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)


def prepare_arrays(data, settings, views, visits, resample, normalize, normalization, coverage):
    began = time.perf_counter()
    patient_visits = [visits[k] for k in sorted({v["visit_id"] for v in views})]
    images = {}
    for visit in patient_visits:
        paths = [phase_path(settings, visit, "pre"), Path(visit["native_first_post"]),
                 phase_path(settings, visit, "late")]
        images[visit["visit_id"]] = [data.native_image(path, visit) for path in paths]
    loaded = time.perf_counter()
    source_support = {key: resample(images[key][1], visits[key]["crop_geometry"], support=True).astype(bool)
                      for key in {v["source_visit_id"] for v in views}}
    unique, rows = {}, []
    for view in views:
        key = grid_key(view)
        if key not in unique:
            native = images[view["visit_id"]]
            raw = np.stack([resample(image, view["geometry"]) for image in native])
            support = np.stack([resample(image, view["geometry"], support=True).astype(bool) for image in native])
            normalized, foreground = normalize(raw, **normalization)
            if not all(mask.any() for mask in foreground) or not all(mask.any() for mask in support):
                raise ValueError("Empty phase on a source-relative ROI")
            unique[key] = normalized, foreground, support
        _, _, support = unique[key]
        values = [coverage(source_support[view["source_visit_id"]], mask) for mask in support]
        rows.append({"view_id": view["view_id"], "values": values, "minimum": min(values)})
    return unique, rows, {"native_load_seconds": loaded - began,
                          "resample_seconds": time.perf_counter() - loaded,
                          "views": len(views), "unique_visit_grids": len(unique)}
