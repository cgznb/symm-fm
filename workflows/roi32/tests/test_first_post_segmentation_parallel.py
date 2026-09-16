from __future__ import annotations

import os
from pathlib import Path

import pytest

from mewm_ispy2 import first_post_segmentation_parallel as parallel
from mewm_ispy2 import first_post_vqgan


def test_batch_reuses_model_and_skips_committed_cases(tmp_path, monkeypatch):
    records = [{"case_id": name} for name in ("done", "first", "second")]
    parallel.write_json(tmp_path / "inventory.json", {"records": records})
    batch = tmp_path / "batch.json"
    parallel.write_json(batch, {"case_ids": [r["case_id"] for r in records]})
    monkeypatch.setattr(parallel, "finished", lambda root, r: r["case_id"] == "done")
    models = []

    def load(config, device):
        instance = object()
        models.append(instance)
        return instance, {}

    calls = []
    monkeypatch.setattr(parallel, "predictor", load)
    monkeypatch.setattr(
        parallel,
        "run_case",
        lambda config, r, device, **kwargs: calls.append(
            (r["case_id"], kwargs["instance"])
        ),
    )
    parallel.run_batch(
        {"output_dir": str(tmp_path)}, batch, "cuda:0", tmp_path / "status.json"
    )
    assert len(models) == 1
    assert calls == [("first", models[0]), ("second", models[0])]
    assert parallel.read_json(batch.with_suffix(".result.json"))["completed_cases"] == 2


@pytest.mark.parametrize("fail", [False, True])
def test_parallel_queue_dispatch_resume_and_cleanup(tmp_path, monkeypatch, fail):
    (tmp_path / "logs").mkdir()
    records = [{"case_id": str(i)} for i in range(7)]
    completed = {"0"}
    assigned = []
    held = set()
    children = []
    peak_workers = 0
    clock = [1.0]
    queue = {
        "max_workers": 2,
        "cases_per_worker": 2,
        "poll_seconds": 1,
        "resource_poll_seconds": 10,
        "admission_stagger_seconds": 10,
    }
    config = {"output_dir": str(tmp_path), "python": "unused", "queue": queue}
    monkeypatch.setattr(parallel, "prepare", lambda config: {"records": records})
    monkeypatch.setattr(parallel, "finished", lambda root, r: r["case_id"] in completed)
    monkeypatch.setattr(first_post_vqgan, "resources", lambda root: {})
    monkeypatch.setattr(
        first_post_vqgan, "resource_reasons", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(parallel.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        parallel.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    class Event:
        def set(self):
            pass

        def clear(self):
            pass

        def wait(self, seconds):
            clock[0] += seconds

    class Thread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(parallel.threading, "Event", Event)
    monkeypatch.setattr(parallel.threading, "Thread", Thread)

    class Lock:
        def __init__(self, gpu):
            self.gpu = gpu

        def close(self):
            held.discard(self.gpu)

    def select(config):
        for gpu in (1, 0):
            if gpu not in held:
                held.add(gpu)
                return gpu, Lock(gpu)
        return None, None

    class Child:
        def __init__(self, args, **kwargs):
            nonlocal peak_workers
            self.ids = parallel.read_json(Path(args[args.index("--batch") + 1]))[
                "case_ids"
            ]
            assert not set(self.ids) & set(assigned)
            assert not set(self.ids) & completed
            assigned.extend(self.ids)
            self.pid = len(children) + 1000
            self.returncode = None
            self.finish_at = clock[0] + 22
            children.append(self)
            peak_workers = max(
                peak_workers, sum(c.returncode is None for c in children)
            )

        def poll(self):
            if self.returncode is None and clock[0] >= self.finish_at:
                self.returncode = 2 if fail and self.pid == 1000 else 0
                if self.returncode == 0:
                    completed.update(self.ids)
                elif fail:
                    completed.add(self.ids[0])
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(parallel, "select_gpu", select)
    monkeypatch.setattr(parallel.subprocess, "Popen", Child)
    if fail:
        with pytest.raises(RuntimeError, match="batch failed"):
            parallel.run_parallel_queue(config, config, tmp_path / "config.yaml")
        assert "0" in completed and "1" in completed
        assert parallel.read_json(tmp_path / "queue_status.json")["status"] == "failed"
        assert all(c.returncode is not None for c in children)
    else:
        parallel.run_parallel_queue(config, config, tmp_path / "config.yaml")
        assert completed == {r["case_id"] for r in records}
        result = parallel.read_json(tmp_path / "queue_status.json")
        assert result["status"] == "completed"
        assert result["completed_cases"] == 7
    assert peak_workers == 2
    assert not held


@pytest.mark.parametrize("recover", [False, True])
def test_cache_release_only_committed_owned_unchanged_inputs(
    tmp_path, monkeypatch, recover
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    records = []
    for name in ("done", "pending", "changed", "outside", "link"):
        path = (tmp_path if name == "outside" else source_root) / name
        if name == "link":
            path.symlink_to(source_root / "done")
        else:
            path.write_bytes(b"input-data" * 4)
        info = path.stat()
        records.append(
            {
                "case_id": name,
                "source_image": str(path),
                "size_bytes": info.st_size,
                "local_mtime_ns": info.st_mtime_ns,
            }
        )
    records[2]["local_mtime_ns"] -= 1
    monkeypatch.setattr(
        parallel, "finished", lambda root, record: record["case_id"] != "pending"
    )
    advised = []

    def advise(descriptor, offset, length, advice):
        assert os.readlink(f"/proc/self/fd/{descriptor}") == str(source_root / "done")
        assert advice == os.POSIX_FADV_DONTNEED
        advised.append(length)

    monkeypatch.setattr(parallel.os, "posix_fadvise", advise)
    monkeypatch.setattr(
        first_post_vqgan,
        "resources",
        lambda root: {
            "memory_headroom_bytes": 24 if recover and advised else 12,
        },
    )
    result = parallel.release_completed_input_cache(
        tmp_path,
        source_root,
        records,
        target_headroom_bytes=24,
        max_advice_bytes=16,
    )
    assert advised == [16]
    assert result["files_advised"] == 1
    assert result["source_bytes_advised"] == 16
    assert result["skipped_files"] == 3
    assert (source_root / "done").read_bytes() == b"input-data" * 4
    assert result["after"]["memory_headroom_bytes"] == (24 if recover else 12)


def test_cache_release_stops_when_headroom_is_already_sufficient(tmp_path, monkeypatch):
    monkeypatch.setattr(
        first_post_vqgan, "resources", lambda root: {"memory_headroom_bytes": 24}
    )
    monkeypatch.setattr(
        parallel, "finished", lambda *args: pytest.fail("No input should be inspected")
    )
    result = parallel.release_completed_input_cache(
        tmp_path, tmp_path, [{"case_id": "unused"}], target_headroom_bytes=24
    )
    assert result["files_advised"] == 0
