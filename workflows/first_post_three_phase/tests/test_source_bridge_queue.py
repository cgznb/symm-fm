from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def queue_module():
    path = Path(__file__).parents[1] / "scripts/run_source_bridge_queue.py"
    spec = importlib.util.spec_from_file_location("bridge_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "memory,utilization,idle",
    [
        (0, 0, True),
        (1000, 9, True),
        (1200, 0, False),
        (0, 90, False),
        (28429, 96, False),
    ],
)
def test_queue_requires_both_available_memory_and_idle_compute(
    monkeypatch, memory, utilization, idle
):
    module = queue_module()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"{memory}, {utilization}\n"),
    )
    assert module.gpu_idle(0) == (idle, memory, utilization)


def test_queue_requires_three_consecutive_idle_checks(tmp_path, monkeypatch):
    module = queue_module()
    checks = iter(
        [(True, 0, 0), (False, 2000, 99), (True, 0, 0), (True, 0, 0), (True, 0, 0)]
    )
    monkeypatch.setattr(module, "gpu_idle", lambda gpu: next(checks))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    status = tmp_path / "status.json"
    module.wait_for_gpu(0, status, {"job": "test"})
    assert json.loads(status.read_text())["idle_checks"] == 3


def test_queue_resumes_the_new_run_checkpoint(tmp_path, monkeypatch):
    module = queue_module()
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/last.ckpt").touch()
    (tmp_path / "run_contract.json").write_text("{}")
    seen = {}

    def popen(command, **kwargs):
        seen["command"] = command
        return SimpleNamespace(pid=123, stdout=iter([]), wait=lambda: 0)

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    module.run_stage(
        {"repo": str(tmp_path), "config": "continued.yaml"}, "train",
        tmp_path, "python", 0, tmp_path / "status.json", {},
    )
    assert seen["command"][-2:] == ["--resume", str(tmp_path / "checkpoints/last.ckpt")]


def test_interrupted_stage_terminates_and_reaps_child(tmp_path, monkeypatch):
    module = queue_module()
    seen = []

    def lines():
        raise KeyboardInterrupt()
        yield

    child = SimpleNamespace(
        pid=123, stdout=lines(),
        terminate=lambda: seen.append("terminate"),
        wait=lambda **kwargs: seen.append("wait"),
    )
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(KeyboardInterrupt):
        module.run_stage(
            {"repo": str(tmp_path), "config": "continued.yaml"}, "smoke",
            tmp_path, "python", 0, tmp_path / "status.json", {},
        )
    assert seen == ["terminate", "wait"]


def test_superseded_queue_cannot_restart_old_precision(tmp_path):
    module = queue_module()
    manifest = tmp_path / "queue.json"
    manifest.write_text(json.dumps({
        "schema": "biflow_source_bridge_queue_v1", "superseded_by": "/new/queue.json",
    }))
    with pytest.raises(ValueError, match="superseded"):
        module.run_queue(manifest)
