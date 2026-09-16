import copy
import fcntl
import signal
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mewm_ispy2 import three_phase_all_pairs_data as data
from mewm_ispy2 import three_phase_all_pairs_training as training
from mewm_ispy2.three_phase_symmflow import PHASES


def fixture(monkeypatch):
    monkeypatch.setattr(data, "IMAGE_SHAPE", (2, 3, 4))
    geometry = {"shape_zyx": [2, 3, 4], "spacing_xyz_mm": [1, 1, 1],
                "origin_lps_mm": [0, 0, 0], "direction_lps": np.eye(3).ravel().tolist()}
    world = {"visits": [], "pairs": [], "normalization": {"image_mean": 1, "image_std": 2}}
    native = {"destination_root": "/native", "records": []}
    metadata = []
    for pid, split, stages in (("a", "train", range(4)), ("b", "val", (1, 2, 3)), ("c", "train", (0, 3))):
        row = {"pid": pid, "pCR": np.nan}
        for stage in stages:
            label = f"T{stage}"
            row.update({f"pre_{label}": 0, f"post_late_{label}": 3})
            grid = {**geometry, "spacing_xyz_mm": [stage + 1] * 3, "origin_lps_mm": [stage * 10] * 3}
            world["visits"].append({"patient_id": pid, "canonical_patient_id": pid, "visit_id": f"{pid}:{label}",
                                    "visit": label, "split": split, "crop_geometry": grid})
            native["records"].append({"patient_id": pid, "visit": label, "source_path": f"/remote/{pid}_{label}_aqc_1.nii.gz",
                                       "relative_path": f"{pid}/{label}.nii.gz"})
        for earlier, later in combinations(stages, 2):
            world["pairs"].append({"patient_id": pid, "split": split, "pair_id": f"{pid}:{earlier}-{later}",
                                   "earlier_visit_id": f"{pid}:T{earlier}", "later_visit_id": f"{pid}:T{later}",
                                   "earlier_stage": f"T{earlier}", "later_stage": f"T{later}", "delta_days": (later - earlier) * 21,
                                   "interval_missing": False, "interval_source": "verified_relative_dicom_study_date",
                                   "baseline_clinical": {"age": 50}, "treatment": {"treatment_arm": "Paclitaxel"}})
        metadata.append(row)
    return world, native, pd.DataFrame(metadata)


def test_all_six_pairs_and_missing_t0_without_pcr_label(monkeypatch):
    world, native, metadata = fixture(monkeypatch)
    inventory = data.build_inventory(world, native, metadata, {})
    assert len(inventory["pairs"]) == 10
    assert set(inventory["pair_counts"]["train"]) == {"T0->T1", "T0->T2", "T0->T3", "T1->T2", "T1->T3", "T2->T3"}
    assert inventory["pair_counts"]["val"] == {"T1->T2": 1, "T1->T3": 1, "T2->T3": 1}
    assert inventory["phase_order"] == list(PHASES)
    for pair in inventory["pairs"]:
        source = next(v for v in inventory["views"] if v["view_id"] == pair["source_view"])
        assert source["source_visit_id"] == source["visit_id"] == pair["earlier_visit_id"]


def test_missing_phase_does_not_remove_valid_later_visits(monkeypatch):
    world, native, metadata = fixture(monkeypatch)
    metadata.loc[metadata.pid == "a", "post_late_T1"] = np.nan
    result = data.build_inventory(world, native, metadata, {})
    assert any(p["pair_id"] == "a:2-3" for p in result["pairs"])
    assert all("a:T1" not in (p["earlier_visit_id"], p["later_visit_id"]) for p in result["pairs"])
    assert len(result["exclusions"]) == 1


def test_same_target_has_distinct_source_relative_views(monkeypatch):
    world, native, metadata = fixture(monkeypatch)
    result = data.build_inventory(world, native, metadata, {})
    views = [v for v in result["views"] if v["visit_id"] == "a:T3"]
    assert len(views) == 3
    assert len({v["latent_file"] for v in views}) == 3
    assert {tuple(v["geometry"]["spacing_xyz_mm"]) for v in views} == {(1, 1, 1), (2, 2, 2), (3, 3, 3)}
    centers = [np.asarray(v["geometry"]["origin_lps_mm"]) + np.asarray([3, 2, 1]) * v["geometry"]["spacing_xyz_mm"] / 2 for v in views]
    for center in centers:
        np.testing.assert_allclose(center, [36, 34, 32])


def crop_override_fixture(world):
    corrections = {"schema": "first_post_t0_crop_overrides_v1", "verified": True,
                   "policy": {"empty_mask": "t0_mask", "reference_mapping": "physical_lps",
                              "outside_center": "exclude_visit", "unavailable_t0": "exclude_visit"},
                   "image_normalization": {"mean": 1, "std": 2},
                   "records": [], "unchanged": [], "exclusions": []}
    for visit in world["visits"]:
        visit["crop_localization"] = "all_predicted_components_union"
        row = {"case_id": f"{visit['patient_id']}_{visit['visit']}_aqc1", "patient_id": visit["patient_id"],
               "visit": visit["visit"], "fold": visit["split"], "dependencies": {}}
        if visit["visit_id"] == "a:T1":
            row["reason"] = "t0_center_outside_acquisition"
            corrections["exclusions"].append(row)
        elif visit["visit_id"] == "a:T2":
            visit["crop_localization"] = "empty_mask_full_image_fallback"
            row.update(crop={"geometry": copy.deepcopy(visit["crop_geometry"]),
                             "localization": "empty_mask_t0_mask_fallback", "source_mask_voxels": 0,
                             "resampled_mask_voxels": 0, "reference_mapping": "physical_lps",
                             "localization_center_in_acquisition": True}, cache_path="/corrected/a_T2.npy", cache_identity={})
            corrections["records"].append(row)
        else:
            corrections["unchanged"].append(row)
    return corrections


@pytest.mark.parametrize("reason", ["t0_center_outside_acquisition", "missing_t0_reference", "empty_t0_reference"])
def test_t0_exclusion_removes_visit_pairs_and_keeps_other_visits(monkeypatch, reason):
    world, native, metadata = fixture(monkeypatch)
    correction = crop_override_fixture(world)
    correction["exclusions"][0]["reason"] = reason
    updated = data.apply_crop_overrides(world, correction)
    result = data.build_inventory(updated, native, metadata, {}, data.T0_GRID_POLICY)
    assert len(result["pairs"]) == 7
    assert all("a:T1" not in (p["earlier_visit_id"], p["later_visit_id"]) for p in result["pairs"])
    assert {p["pair_id"] for p in result["pairs"] if p["patient_id"] == "a"} == {"a:0-2", "a:0-3", "a:2-3"}
    assert result["exclusions"] == [{"visit_id": "a:T1", "reason": reason}]
    assert all(p["split"] == ("val" if p["patient_id"] == "b" else "train") for p in result["pairs"])
    assert len(world["visits"]) == 9


def test_empty_target_preserves_complete_t0_geometry_even_with_small_source(monkeypatch):
    world, native, metadata = fixture(monkeypatch)
    correction = crop_override_fixture(world)
    updated = data.apply_crop_overrides(world, correction)
    result = data.build_inventory(updated, native, metadata, {}, data.T0_GRID_POLICY)
    target = next(v for v in result["views"] if v["visit_id"] == "a:T2" and v["source_visit_id"] == "a:T0")
    assert target["geometry"] == correction["records"][0]["crop"]["geometry"]
    assert target["geometry"]["spacing_xyz_mm"] == [3, 3, 3]
    with pytest.raises(ValueError, match="extent must be preserved"):
        data.build_inventory(updated, native, metadata, {})


def test_t0_correction_cannot_change_split_or_skip_a_visit(monkeypatch):
    world, _, _ = fixture(monkeypatch)
    correction = crop_override_fixture(world)
    correction["records"][0]["fold"] = "val"
    with pytest.raises(ValueError, match="patient split"):
        data.apply_crop_overrides(world, correction)
    correction["records"].clear()
    with pytest.raises(ValueError, match="does not cover"):
        data.apply_crop_overrides(world, correction)


def test_mismatched_stage_is_rejected(monkeypatch):
    world, native, metadata = fixture(monkeypatch)
    world["pairs"][0]["earlier_stage"] = "T1"
    with pytest.raises(ValueError, match="Invalid forward"):
        data.build_inventory(world, native, metadata, {})


def test_admission_drops_whole_training_patient_and_keeps_low_coverage_validation(monkeypatch):
    world, native, metadata = fixture(monkeypatch)
    inventory = data.build_inventory(world, native, metadata, {})
    reports = [{"patient_id": "a", "excluded_training_patient": True},
               {"patient_id": "b", "excluded_training_patient": False, "minimum_coverage": 0.2}]
    result = data.admit_inventory(inventory, reports)
    assert {p["patient_id"] for p in result["pairs"]} == {"b", "c"}
    assert all(v["patient_id"] != "a" for v in result["views"])
    assert result["pair_counts"]["val"] == inventory["pair_counts"]["val"]


def test_statistics_use_unique_training_views_only():
    views = [{"view_id": "train", "patient_id": "a", "split": "train"},
             {"view_id": "train", "patient_id": "a", "split": "train"},
             {"view_id": "val", "patient_id": "b", "split": "val"}]
    opened = []
    def read(view):
        assert view["view_id"] == "train"
        opened.append(view["view_id"])
        return np.tile([-1., 1.], (24, 1))
    stats = data.fit_statistics({"views": views}, read)
    assert stats["mean"] == [0.] * 24 and stats["std"] == [1.] * 24
    assert stats["unique_views"] == 1 and opened == ["train"]


def test_compact_latent_requires_exact_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "JOINT_LATENT_SHAPE", (24, 1, 2, 2))
    value = np.full((24, 1, 2, 2), 1.125, np.float32)
    data.atomic_latent(tmp_path / "latent.npy", value)
    np.testing.assert_array_equal(data.read_latent(tmp_path, {"latent_file": "latent.npy"}), value)
    value[0, 0, 0, 0] = 1.1234567
    with pytest.raises(ValueError, match="round-trip"):
        data.pack_latent(value)


def test_reference_masks_and_values_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "IMAGE_SHAPE", (2, 3, 4))
    images = np.arange(72, dtype=np.float32).reshape(3, 2, 3, 4)
    foreground, support = images != 0, np.ones_like(images, dtype=bool)
    data.write_reference(tmp_path / "ref.npz", images, foreground, support)
    actual = data.read_reference(tmp_path, {"reference_file": "ref.npz"})
    for key, expected in (("images", images), ("foreground", foreground), ("support", support)):
        np.testing.assert_array_equal(actual[key], expected)


@pytest.mark.parametrize("stage", ["T1", "T2"])
def test_source_queries_cannot_read_target_or_t0(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(data, "JOINT_LATENT_SHAPE", (24, 1, 2, 2))
    views = []
    for name in ("T0", "T1", "T2", "T3"):
        view = {"view_id": name, "latent_file": f"latents/{name}.npy", "reference_file": f"references/{name}.npz"}
        data.atomic_latent(tmp_path / view["latent_file"], np.ones((24, 1, 2, 2), np.float32))
        views.append(view)
    record = {"source_view": stage, "target_view": "T3", "split": "val"}
    inventory = {"views": views, "pairs": [record], "visits": []}
    dataset = data.PairDataset(tmp_path, inventory, {"mean": [0] * 24, "std": [1] * 24}, "val")
    with training.source_read_guard(tmp_path, inventory, dataset.views[stage]):
        assert dataset.source(record).shape == (24, 1, 2, 2)
        for forbidden in ("T0", "T3"):
            with pytest.raises(RuntimeError, match="undeclared"):
                np.load(tmp_path / dataset.views[forbidden]["latent_file"])


def test_macro_validation_weights_patients_equally_and_stratifies_coverage():
    rows = []
    for patient, count, value in (("a", 6, 0.), ("b", 1, 2.)):
        for _ in range(count):
            metrics = {phase: {"mae": value, "rmse": value} for phase in PHASES}
            rows.append({"patient_id": patient, "transition": "T1->T2", "minimum_coverage": 0.2 if patient == "b" else 1.,
                         "generation": metrics, "copy_source": metrics, "target_codec": metrics})
    result = training.summarize(rows)
    assert result["score"] == 1.
    assert result["by_coverage"]["below_0.5"]["pairs"] == 1
    assert result["pairs"] == 7 and result["patients"] == 2


def test_light_validation_is_fixed_and_covers_six_transitions(monkeypatch):
    world, _, _ = fixture(monkeypatch)
    records = world["pairs"] * 20
    chosen = training.validation_indices(records, 64, 2026)
    assert len(chosen) == len(set(chosen)) == 64
    assert chosen == training.validation_indices(records, 64, 2026)
    assert len({(records[i]["earlier_stage"], records[i]["later_stage"]) for i in chosen}) == 6


def test_pilot_recovery_is_rejected(tmp_path):
    path = tmp_path / "old.ckpt"
    torch.save({"schema": "three_phase_symmflow_pilot_v1"}, path)
    with pytest.raises(ValueError, match="pilot checkpoints"):
        training.restore_state(path, {}, None, None, None, None)


def test_validation_load_failure_restores_mode_and_random_state(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2).train()
    state = torch.get_rng_state()
    def fail(*args, **kwargs):
        torch.rand(10)
        raise ValueError("codec unavailable")
    monkeypatch.setattr(training, "load_codec", fail)
    with pytest.raises(ValueError, match="codec unavailable"):
        training.evaluate({"output_root": str(tmp_path), "codec_checkpoint": "unused"}, {}, model, [], step=1)
    assert model.training and torch.equal(state, torch.get_rng_state())


def test_active_pcr_workflow_blocks_even_an_idle_gpu(tmp_path, monkeypatch):
    from scripts.run_three_phase_symmflow import gpu_available, workflow_active
    cfg = {"runtime": {"protected_pcr_runs": {1: str(tmp_path)}}}
    with (tmp_path / "workflow.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert workflow_active(tmp_path)
        def fail(*args, **kwargs):
            raise AssertionError("Must not consider GPU use while pCR owns its workflow")
        monkeypatch.setattr("scripts.run_three_phase_symmflow.subprocess.check_output", fail)
        assert gpu_available(cfg, 1) == (False, "pcr_workflow_active")
    assert not workflow_active(tmp_path)


def test_gpu_with_compute_process_is_not_available(tmp_path, monkeypatch):
    from scripts.run_three_phase_symmflow import gpu_available
    cfg = {"runtime": {"protected_pcr_runs": {1: str(tmp_path)}}}
    monkeypatch.setattr("scripts.run_three_phase_symmflow.subprocess.check_output", lambda *a, **k: "123\n")
    assert gpu_available(cfg, 1) == (False, "gpu_compute_process_active")


@pytest.mark.parametrize("t0_fallback", [False, True])
@pytest.mark.parametrize("optimizer_updates", [False, True])
def test_formal_pipeline_prepare_recovery_selection_and_source_exports(tmp_path, monkeypatch, t0_fallback, optimizer_updates):
    from contextlib import nullcontext
    from types import SimpleNamespace

    import SimpleITK as sitk
    from torch import nn

    from mewm_ispy2 import three_phase_pilot_data as pilot
    from mewm_ispy2.first_post_world_data import identity, read_json, write_json
    from mewm_ispy2.three_phase_preparation import grid_key
    from mewm_ispy2.three_phase_symmflow import ThreePhaseSymmFlow
    world, native, metadata = fixture(monkeypatch)
    policy = "target_center_with_source_extent"
    if t0_fallback:
        world = data.apply_crop_overrides(world, crop_override_fixture(world))
        policy = data.T0_GRID_POLICY
    inventory = data.build_inventory(world, native, metadata, {}, policy)
    monkeypatch.setattr(pilot, "IMAGE_SHAPE", (2, 3, 4))
    monkeypatch.setattr(data, "JOINT_LATENT_SHAPE", (24, 2, 3, 4))
    monkeypatch.setattr(data, "disk_gate", lambda *a, **k: None)
    monkeypatch.setattr(training, "disk_gate", lambda *a, **k: None)
    grids = {}
    for visit in inventory["visits"]:
        path = tmp_path / "native" / visit["patient_id"] / f"{visit['visit']}.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        visit.update(native_first_post=str(path), native_first_post_identity=identity(path), phase_bytes={"pre": 16, "late": 16})
        grids[visit["visit_id"]] = visit["crop_geometry"]
    def native_image(path, visit):
        image = sitk.GetImageFromArray(np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 10)
        image.SetSpacing(grids[visit["visit_id"]]["spacing_xyz_mm"])
        image.SetOrigin(grids[visit["visit_id"]]["origin_lps_mm"])
        return image
    helper = SimpleNamespace(release_phases=lambda *a: None, native_image=native_image)
    monkeypatch.setattr(data, "pcr_helpers", lambda cfg: (helper, {"remote_root": "/remote", "output_dir": str(tmp_path)}))
    transfers, encodings = [], []
    class Transfer:
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def stage(self, visits):
            transfers.append(visits[0]["patient_id"])
            return {"transfer_seconds": 0.0}
        def cancel(self):
            pass
    monkeypatch.setattr(data, "PhaseTransfer", Transfer)
    class Codec(nn.Module):
        def encode_continuous(self, image):
            encodings.append(image.shape)
            return image.repeat(1, 8, 1, 1, 1).bfloat16()
        def quantizer(self, value):
            return value, {}
        def decode(self, value):
            return value[:, :1]
    monkeypatch.setattr(data, "load_codec", lambda *a, **k: Codec())
    monkeypatch.setattr(training, "load_codec", lambda *a, **k: Codec())
    tensor_to, module_to = torch.Tensor.to, nn.Module.to
    def cpu_args(args):
        return tuple("cpu" if isinstance(a, str) and a.startswith("cuda") else a for a in args)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: tensor_to(self, *cpu_args(a), **k))
    monkeypatch.setattr(nn.Module, "to", lambda self, *a, **k: module_to(self, *cpu_args(a), **k))
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(training, "render_comparison", lambda *a, **k: None)
    class Velocity(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(48, 48, 1)
            self.context = nn.Linear(256, 48)
            self.out = nn.Conv3d(48, 48, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        def forward(self, x, timesteps, context):
            return self.out(self.conv(x) + self.context(context.mean(1))[:, :, None, None, None])
    monkeypatch.setattr(training, "ThreePhaseSymmFlow", lambda records, source: ThreePhaseSymmFlow(records, source=source, backbone=Velocity()))
    cfg = data.load_config(Path(__file__).resolve().parents[1] / "configs/three_phase_symmflow_all_pairs_v1.yaml")
    cfg["output_root"] = str(tmp_path)
    cfg["spatial_policy"] = policy
    cfg["training"].update(max_optimizer_steps=2, effective_batch_size=1, warmup_steps=1, weight_decay=0.0,
                           light_validation_interval=1, full_validation_interval=2, checkpoint_interval=1)
    cfg["runtime"]["loader_workers"] = 0
    write_json(tmp_path / "inventory.json", inventory)
    states = []
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def pause_after_patient(value):
        states.append(value)
        if value.get("prepared_this_session") == 1:
            signal.raise_signal(signal.SIGTERM)
    assert data.prepare(cfg, inventory, pause_after_patient) is None
    assert states[-1]["stage"] == "paused"
    assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items())
    assert not (tmp_path / "preparation.json").exists()
    completed_marker = tmp_path / "prepared" / "patient_0000.json"
    completed_identity = identity(completed_marker)
    completed = read_json(completed_marker)
    admitted = data.prepare(cfg, inventory, states.append)
    assert identity(completed_marker) == completed_identity
    assert all(identity(tmp_path / name) == recorded for name, recorded in completed["files"].items())
    assert transfers.count("a") == 1
    assert len(encodings) == 3 * len({grid_key(v) for v in admitted["views"]})
    storage = read_json(tmp_path / "storage_admission.json")
    pending_views = [v for v in inventory["views"] if v["patient_id"] != "a"]
    assert storage["latent_bytes"] == len(pending_views) * 24 * 2 * 3 * 4 * 2
    assert storage["completed_patients"] == 1 and storage["remaining_patients"] == 2
    assert storage["maximum_staged_patients"] == 2
    before = len(encodings), len(transfers)
    assert data.prepare(cfg, inventory, states.append) == admitted
    assert (len(encodings), len(transfers)) == before
    assert read_json(tmp_path / "storage_admission.json")["reference_bytes_upper_bound"] == 0
    assert len(admitted["pairs"]) == (7 if t0_fallback else 10)
    assert all(not (tmp_path / v["reference_file"]).exists() for v in admitted["views"] if v["split"] == "train" and v["reference_file"])
    if not optimizer_updates:
        cfg["training"]["learning_rate"] = 0.0
        with pytest.raises(ValueError, match="did not update velocity parameters"):
            training.smoke(cfg, admitted, states.append)
        assert not (tmp_path / "smoke" / "report.json").exists()
        assert not (tmp_path / "checkpoints" / "recovery.ckpt").exists()
        return
    training.smoke(cfg, admitted, states.append)
    smoke_report = read_json(tmp_path / "smoke" / "report.json")
    assert smoke_report["exact_resume"]
    assert smoke_report["updated_velocity_parameters"] == ["backbone.out.weight", "backbone.out.bias"]
    assert not (tmp_path / "checkpoints" / "recovery.ckpt").exists()
    assert not training.train(cfg, admitted, states.append, stop_after=1)
    assert not (tmp_path / "checkpoints" / "best.ckpt").exists()
    assert training.train(cfg, admitted, states.append, resume=True)
    selected = training.evaluate_selected(cfg, admitted, states.append)
    assert selected["optimizer_step"] == 2 and selected["pairs"] == 3
    assert all(a["passed"] and a["future_image_reads"] == 0 for a in selected["source_only_audits"])
    assert len(selected["exports"]) == 3
    assert read_json(tmp_path / "training_complete.json")["optimizer_steps"] == 2
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("changed", [False, True])
def test_preparation_reuse_requires_unchanged_physical_views(tmp_path, monkeypatch, changed):
    from mewm_ispy2.first_post_world_data import identity, read_json, write_json
    world, native, metadata = fixture(monkeypatch)
    old = data.build_inventory(world, native, metadata, {})
    old.update(source_files={"codec": {"size_bytes": 16, "mtime_ns": 1}},
               config={"quality": {"minimum_source_support_coverage": 0.5}})
    previous, root = tmp_path / "old", tmp_path / "new"
    write_json(previous / "inventory.json", old)
    for pid in ("a", "b", "c"):
        views = [v for v in old["views"] if v["patient_id"] == pid]
        files = {}
        for view in views:
            for field in ("latent_file", "reference_file"):
                if not view[field]:
                    continue
                path = previous / view[field]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"verified immutable cache fixture")
                files[view[field]] = identity(path)
        write_json(previous / "prepared" / f"{pid}.json",
                   {"patient_id": pid, "excluded_training_patient": False, "files": files,
                    "inventory": identity(previous / "inventory.json"),
                    "views": [{"view_id": v["view_id"], "minimum": 1.0} for v in views]})
    new = copy.deepcopy(old)
    for view in new["views"]:
        for field in ("latent_file", "reference_file"):
            if view[field]:
                view[field] = view[field].replace("v0", "new_v0")
    if changed:
        new["views"][0]["geometry"]["origin_lps_mm"][0] += 10
    write_json(root / "inventory.json", new)
    cfg = {"output_root": str(root), "reuse_preparation_from": str(previous), "quality": old["config"]["quality"]}
    data.reuse_prepared_patients(cfg, new)
    reports = [read_json(p) for p in (root / "prepared").glob("*.json")]
    assert {r["patient_id"] for r in reports} == ({"b", "c"} if changed else {"a", "b", "c"})
    for report in reports:
        assert report["inventory"] == identity(root / "inventory.json")
        for path, recorded in report["files"].items():
            assert identity(root / path) == recorded
            assert (root / path).stat().st_nlink == 2
