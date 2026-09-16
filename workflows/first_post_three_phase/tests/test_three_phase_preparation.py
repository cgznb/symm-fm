import copy
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mewm_ispy2.first_post_world_data import identity, write_json
from mewm_ispy2.three_phase_preparation import (
    LOCAL_PHASE_SCHEMA,
    PhasePrefetch,
    PhaseTransfer,
    grid_key,
    load_local_phase_sources,
    multiplex_arguments,
    phase_path,
    prepare_arrays,
)


def local_source_fixture(tmp_path):
    run = tmp_path / "run"
    source = tmp_path / "persistent" / "pre.nii.gz"
    source.parent.mkdir()
    source.write_bytes(b"native pre")
    visit = {"visit_id": "a", "remote_phases": {"pre": "/remote/a/pre.nii.gz", "late": "/remote/a/late.nii.gz"},
             "phase_bytes": {"pre": source.stat().st_size, "late": 12}}
    inventory = {"visits": [visit]}
    write_json(run / "inventory.json", inventory)
    evidence = tmp_path / "verification.json"
    write_json(evidence, {"passed": True})
    manifest = {"schema": LOCAL_PHASE_SCHEMA, "verified_exact_content": True,
                "inventory": identity(run / "inventory.json"), "remote_root": "/remote",
                "evidence": [{"path": str(evidence), "identity": identity(evidence)}],
                "files": {"a/pre.nii.gz": {"path": str(source), "identity": identity(source)}}}
    write_json(run / "local_phase_sources.json", manifest)
    settings = {"output_dir": str(run), "remote_root": "/remote", "transfer_workers": 2}
    return settings, inventory, manifest, source


def test_local_sources_are_bound_to_inventory_and_verified_files(tmp_path):
    settings, inventory, manifest, source = local_source_fixture(tmp_path)
    settings["local_phase_sources"] = load_local_phase_sources(settings, inventory)
    visit = inventory["visits"][0]
    assert phase_path(settings, visit, "pre") == source
    assert phase_path(settings, visit, "late") == Path(settings["output_dir"]) / "staging/a/late.nii.gz"
    assert identity(source) == manifest["files"]["a/pre.nii.gz"]["identity"]
    source.write_bytes(b"changed pre")
    with pytest.raises(ValueError, match="native phase changed"):
        phase_path(settings, visit, "pre")
    with pytest.raises(ValueError, match="native phase changed"):
        load_local_phase_sources(settings, inventory)


@pytest.mark.parametrize("change", ["inventory", "unverified", "size", "staging", "unknown_phase", "evidence"])
def test_invalid_local_source_manifests_are_rejected(tmp_path, change):
    settings, inventory, manifest, _ = local_source_fixture(tmp_path)
    run = Path(settings["output_dir"])
    source = manifest["files"]["a/pre.nii.gz"]
    if change == "inventory":
        manifest["inventory"]["size_bytes"] += 1
    elif change == "unverified":
        manifest["verified_exact_content"] = False
    elif change == "size":
        source["identity"]["size_bytes"] += 1
    elif change == "staging":
        source["path"] = str(run / "staging/a/pre.nii.gz")
    elif change == "unknown_phase":
        manifest["files"]["a/other.nii.gz"] = manifest["files"].pop("a/pre.nii.gz")
    else:
        manifest["evidence"][0]["identity"]["size_bytes"] += 1
    write_json(run / "local_phase_sources.json", manifest)
    with pytest.raises(ValueError):
        load_local_phase_sources(settings, inventory)


def test_transfer_requests_only_missing_remote_files(tmp_path, monkeypatch):
    settings, inventory, _, source = local_source_fixture(tmp_path)
    settings["local_phase_sources"] = load_local_phase_sources(settings, inventory)
    staged = Path(settings["output_dir"]) / "staging"
    cached = staged / "a/late.nii.gz"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached phase")
    visits = [*inventory["visits"], {"remote_phases": {"pre": "/remote/b/pre.nii.gz"}, "phase_bytes": {"pre": 3}}]
    requests, disk = [], []
    helper = SimpleNamespace(ssh_arguments=lambda _: ["ssh"], disk_gate=lambda _, required: disk.append(required))
    with PhaseTransfer(helper, settings, reuse_connections=False) as transfer:
        def download(shard):
            _, names = shard
            requests.extend(names)
            for name in names:
                path = staged / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"new")
        monkeypatch.setattr(transfer, "_transfer", download)
        result = transfer.stage(visits)
    assert requests == ["b/pre.nii.gz"] and disk == [3]
    assert result["local_phase_files"] == result["staged_reused_files"] == result["remote_requested_files"] == 1
    assert result["local_phase_bytes"] == source.stat().st_size and result["remote_requested_bytes"] == 3
    assert not (staged / "a/pre.nii.gz").exists()
    assert source.read_bytes() == b"native pre" and cached.read_bytes() == b"cached phase"


def test_local_only_patient_does_not_start_a_transfer(tmp_path, monkeypatch):
    settings, inventory, _, _ = local_source_fixture(tmp_path)
    settings["local_phase_sources"] = load_local_phase_sources(settings, inventory)
    visit = inventory["visits"][0]
    visit["remote_phases"].pop("late")
    visit["phase_bytes"].pop("late")
    def fail(*args, **kwargs):
        raise AssertionError("A local-only patient must not transfer files")
    helper = SimpleNamespace(ssh_arguments=lambda _: ["ssh"], disk_gate=fail)
    with PhaseTransfer(helper, settings, reuse_connections=False) as transfer:
        monkeypatch.setattr(transfer, "_transfer", fail)
        result = transfer.stage([visit])
    assert result["remote_requested_files"] == 0 and result["local_phase_files"] == 1


def test_arrays_read_persistent_local_phase_without_staging_copy(tmp_path):
    settings, inventory, _, source = local_source_fixture(tmp_path)
    settings["local_phase_sources"] = load_local_phase_sources(settings, inventory)
    geometry = {"shape_zyx": [1, 1, 2], "spacing_xyz_mm": [1, 1, 1],
                "origin_lps_mm": [0, 0, 0], "direction_lps": np.eye(3).ravel().tolist()}
    visit = {**inventory["visits"][0], "crop_geometry": geometry, "native_first_post": "/local/first.nii.gz"}
    view = {"view_id": "a", "visit_id": "a", "source_visit_id": "a", "geometry": geometry}
    opened = []
    def native_image(path, _):
        opened.append(path)
        return np.ones((1, 1, 2), dtype=np.float32)
    arrays, _, _ = prepare_arrays(SimpleNamespace(native_image=native_image), settings, [view], {"a": visit},
                                  lambda image, grid, support=False: image,
                                  lambda raw: (raw, raw != 0), {}, lambda source, target: 1.0)
    assert opened == [source, Path("/local/first.nii.gz"), Path(settings["output_dir"]) / "staging/a/late.nii.gz"]
    np.testing.assert_array_equal(arrays[grid_key(view)][0], np.ones((3, 1, 1, 2)))
    assert source.is_file() and not (Path(settings["output_dir"]) / "staging/a/pre.nii.gz").exists()


def test_multiplexing_retains_authentication_and_jump_options(tmp_path):
    original = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o",
                "ProxyCommand=ssh -o ControlMaster=no -o ControlPath=none -W %h:%p gateway"]
    result = multiplex_arguments(original, tmp_path / "control")
    assert "ControlMaster=auto" in result and f"ControlPath={tmp_path / 'control'}" in result
    assert "ControlPersist=120" in result and "StrictHostKeyChecking=yes" in result
    assert result[-3] == original[-1]
    assert original[6] == "ControlMaster=no"


def test_deduplication_requires_same_visit_and_exact_grid():
    a = {"visit_id": "a", "geometry": {"shape_zyx": [2, 3, 4], "spacing_xyz_mm": [1, 1, 1],
                                      "origin_lps_mm": [0, 0, 0], "direction_lps": np.eye(3).ravel().tolist()}}
    b = copy.deepcopy(a)
    assert grid_key(a) == grid_key(b)
    b["visit_id"] = "b"
    assert grid_key(a) != grid_key(b)
    b["visit_id"] = "a"
    b["geometry"]["origin_lps_mm"][0] += 1e-9
    assert grid_key(a) != grid_key(b)


def test_duplicate_arrays_keep_source_specific_coverage(tmp_path):
    geometry = {"shape_zyx": [1, 1, 2], "spacing_xyz_mm": [1, 1, 1],
                "origin_lps_mm": [0, 0, 0], "direction_lps": np.eye(3).ravel().tolist()}
    visits = {name: {"visit_id": name, "crop_geometry": geometry, "native_first_post": f"{name}_first",
                     "remote_phases": {"pre": f"/remote/{name}_pre", "late": f"/remote/{name}_late"}}
              for name in ("s0", "s1", "target")}
    views = [{"view_id": name, "source_visit_id": source, "visit_id": visit, "geometry": geometry}
             for name, source, visit in (("s0", "s0", "s0"), ("s1", "s1", "s1"),
                                          ("v0", "s0", "target"), ("v1", "s1", "target"))]
    calls = []
    def resample(image, grid, support=False):
        calls.append((image, support))
        return np.array([[[True, image.startswith("s1")]]]) if support else np.array([[[10., 20.]]])
    helper = SimpleNamespace(native_image=lambda path, visit: Path(path).name)
    arrays, rows, timing = prepare_arrays(helper, {"output_dir": str(tmp_path), "remote_root": "/remote"},
                                         views, visits, resample, lambda raw: (raw, raw != 0), {},
                                         lambda source, target: float(target[source].mean()))
    assert timing["views"] == 4 and timing["unique_visit_grids"] == 3
    assert len(arrays) == 3
    assert rows[2]["values"] == [1., 1., 1.]
    assert rows[3]["values"] == [0.5, 0.5, 0.5]
    assert len(calls) == 2 + 3 * 6


def test_prefetch_starts_only_one_future_patient():
    calls = []
    next_started = threading.Event()
    def stage(visits):
        calls.append(visits)
        if visits == "second":
            next_started.set()
        return {"transfer_seconds": 1.0}
    transfer = SimpleNamespace(stage=stage, cancel=lambda: None)
    jobs = [(0, None, None, "first"), (1, None, None, "second"), (2, None, None, "third")]
    with PhasePrefetch(transfer, jobs, threading.Event()) as prefetch:
        prefetch.take(0)
        assert next_started.wait(2)
        assert calls == ["first", "second"]
        prefetch.take(1)
        prefetch.take(2)
    assert calls == ["first", "second", "third"]


def test_prefetch_propagates_transfer_errors_and_cancels():
    cancelled = []
    def stage(_):
        raise TimeoutError("transport timeout")
    transfer = SimpleNamespace(stage=stage, cancel=lambda: cancelled.append(True))
    with (pytest.raises(TimeoutError, match="transport timeout"),
          PhasePrefetch(transfer, [(0, None, None, [])], threading.Event()) as prefetch):
        prefetch.take(0)
    assert cancelled == [True]


def test_prefetch_pause_cancels_pending_transport():
    stopped, released = threading.Event(), threading.Event()
    def stage(_):
        released.wait(2)
        return {}
    transfer = SimpleNamespace(stage=stage, cancel=released.set)
    with (pytest.raises(InterruptedError, match="pause"),
          PhasePrefetch(transfer, [(0, None, None, [])], stopped) as prefetch):
        stopped.set()
        prefetch.take(0)
    assert released.is_set()
