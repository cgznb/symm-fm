"""Map follow-up first-post predictions through the retained registration chain."""

from __future__ import annotations

import json
import multiprocessing
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from nibabel.spaces import vox2out_vox

from .registered_roi32_data import (LPS_RAS, SHAPE, SPACING, acquisition_affine, centered_affine, file_identity,
                                   image_affine, read_json, save_npz, timestamp, verify_identity, write_json)


def affine_transform(matrix):
    result = sitk.AffineTransform(3)
    result.SetMatrix(tuple(matrix[:3, :3].ravel()))
    result.SetTranslation(tuple(matrix[:3, 3]))
    return result


def reference_image(shape_zyx, matrix):
    image = sitk.Image([int(n) for n in shape_zyx[::-1]], sitk.sitkUInt8)
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    image.SetSpacing(tuple(spacing))
    image.SetDirection(tuple((matrix[:3, :3] / spacing).ravel()))
    image.SetOrigin(tuple(matrix[:3, 3]))
    return image


def compose_followup_transform(to_native, phase, longitudinal):
    transform = sitk.CompositeTransform(3)
    for part in (to_native, phase, longitudinal):
        transform.AddTransform(part)
    return transform


def cached_record(marker, input_paths, mask_path):
    if not marker.exists():
        return None
    try:
        record = read_json(marker)
        if record.get("status") != "passed":
            return None
        identities = [file_identity(path) for path in input_paths]
        if "input_sources" in record:
            if record["input_sources"] != identities:
                return None
            verify_identity(record["mask_identity"])
        else:
            # Bind earlier audit rows only when every input predates completion.
            if any(item["mtime_ns"] > marker.stat().st_mtime_ns for item in identities):
                return None
            for identity in record["transform_sources"]:
                verify_identity(identity)
            with np.load(mask_path, allow_pickle=False) as archive:
                mask = archive["mask"]
                if mask.shape != SHAPE or mask.dtype != bool or int(mask.sum()) != record["actual_cropped_voxels"]:
                    return None
            record.update(input_sources=identities, mask_identity=file_identity(mask_path))
            write_json(marker, record)
        return record
    except (KeyError, OSError, ValueError, TypeError):
        return None


class FieldWorker:
    def __init__(self, config, output):
        self.config = config
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.process = None
        self.log = None

    def rebuild(self, meta_path, pair_dir, output_path):
        if self.process is None:
            self.log = (self.output / "itk.log").open("a")
            script = Path(__file__).parents[1] / "scripts" / "roi32_displacement_worker.py"
            self.process = subprocess.Popen([self.config["registration_python"], "-B", str(script), "--registration-repo", self.config["registration_repo"]],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        request = {"meta_path": str(meta_path), "pair_dir": str(pair_dir), "output_path": str(output_path)}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Original ITK field replay worker exited")
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                self.log.write(line)
                continue
            if reply.get("status") != "passed":
                raise RuntimeError(reply.get("reason", "Unknown ITK replay failure"))
            return output_path

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=30)
            self.process.stdout.close()
            self.log.close()


def _run_shard(task):
    config, shard = task
    return run_audit(config, shard=shard)


def summarize(records):
    fractions = [r["retained_fraction"] for r in records if r.get("retained_fraction") is not None]
    return {"visits": len(records), "statuses": dict(Counter(r["status"] for r in records)),
            "status": "passed" if all(r["status"] == "passed" for r in records) else "incomplete",
            "originally_empty_kept": sum(r.get("originally_empty", False) for r in records),
            "nonempty_cropped_empty": sum(r.get("crop_empty", False) and not r.get("originally_empty", False) for r in records),
            "truncated_in_existing_registered_fov": sum(r.get("retained_fraction") is not None and r["retained_fraction"] < 1 for r in records),
            "retained_fraction_mean": float(np.mean(fractions)) if fractions else None,
            "retained_fraction_min": float(min(fractions)) if fractions else None,
            "by_visit": {visit: {"count": sum(r["visit"] == visit for r in records),
                                  "originally_empty_kept": sum(r["visit"] == visit and r.get("originally_empty", False) for r in records),
                                  "nonempty_cropped_empty": sum(r["visit"] == visit and r.get("crop_empty", False) and not r.get("originally_empty", False) for r in records)}
                         for visit in ("T1", "T2", "T3")},
            "training_admission_changed": False, "updated_at": timestamp(),
            "limitation": "Retention measures mapped masks within the existing registered T0 FOV; native mask outside this FOV is not in the denominator."}


def run_audit(config, shard=None):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    output = Path(config["output_dir"]) / "followup_audit"
    output.mkdir(parents=True, exist_ok=True)
    inventory = read_json(Path(config["output_dir"]) / "data" / "inventory.json")
    patients = {p["patient_id"]: p for p in inventory["patients"]}
    retained = output / "retained_transforms"
    tasks, missing = [], []
    for row in inventory["visits"]:
        if row["visit"] == "T0":
            continue
        meta_path = Path(row["metadata_source"]["path"])
        meta = read_json(meta_path)
        status = meta["registration"]["status"]
        relative = Path(patients[row["patient_id"]]["native_patient_id"]) / row["visit"]
        names = ["transforms/intravisit/phase_1_to_phase_0.tfm"]
        names += ["transforms/rigid.tfm"] if status == "rigid_fallback" else ["transforms/elastix/InitialTransformParameters.0.txt", "transforms/elastix/TransformParameters.0.txt"]
        local = meta_path.parent.parent
        for name in names:
            if not (local / name).is_file() and not (retained / relative / name).is_file():
                missing.append((relative / name).as_posix())
        tasks.append((row, meta, local, relative, names))
    transfer = {"requested_files": len(missing), "status": "not_needed"}
    if missing and shard is None and config["followup_audit"]["fetch_missing_transforms"]:
        from research_release import path as release_path, ssh_host
        remote_root = config["followup_audit"]["remote_registered_root"]
        if remote_root == "@remote_registered":
            ssh_host()
            remote_root = release_path(remote_root)
        retained.mkdir(parents=True, exist_ok=True)
        command = ["rsync", "-aR", "--files-from=-", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20",
                   remote_root.rstrip("/") + "/", str(retained) + "/"]
        try:
            result = subprocess.run(command, input="\n".join(missing) + "\n", text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
            transfer.update(status="passed" if result.returncode == 0 else "partial", returncode=result.returncode)
            (output / "transfer.log").write_text(result.stdout + result.stderr)
        except subprocess.TimeoutExpired:
            transfer["status"] = "timeout"
    if shard is None:
        if missing or not (output / "transfer.json").exists():
            write_json(output / "transfer.json", transfer)
        workers = config["followup_audit"].get("workers", 1)
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
                list(executor.map(_run_shard, [(config, i) for i in range(workers)]))
            records = [read_json(output / "visits" / (row["visit_id"].replace(":", "_") + ".json")) for row, *_ in tasks]
            result = summarize(records)
            write_json(output / "summary.json", result)
            return result
    else:
        tasks = tasks[shard::config["followup_audit"]["workers"]]
    worker = FieldWorker(config["followup_audit"], output / "workers" / str(shard if shard is not None else 0))
    records = []
    try:
        for index, (row, meta, local, relative, names) in enumerate(tasks):
            marker = output / "visits" / (row["visit_id"].replace(":", "_") + ".json")
            files = [local / name if (local / name).is_file() else retained / relative / name for name in names]
            report_path = Path(config["output_dir"]) / "data" / "patients" / (row["patient_id"] + ".json")
            mask_path = output / "masks" / (row["visit_id"].replace(":", "_") + ".npz")
            inputs = [row["metadata_source"]["path"], row["model_mask_path"], row["mask_native_metadata_path"], report_path, *files]
            previous = cached_record(marker, inputs, mask_path)
            if previous is not None:
                records.append(previous)
                continue
            record = {"visit_id": row["visit_id"], "patient_id": row["patient_id"], "visit": row["visit"], "fold": row["fold"],
                      "registration_status": meta["registration"]["status"], "status": "unassessed"}
            temporary = output / "temporary" / (row["visit_id"].replace(":", "_") + ".npy")
            try:
                if any(not path.is_file() for path in files):
                    raise FileNotFoundError("Retained phase/longitudinal transform unavailable")
                if not row["model_mask_path"] or not row["mask_native_metadata_path"]:
                    raise FileNotFoundError("Follow-up prediction or saved native metadata unavailable")
                input_sources = [file_identity(path) for path in inputs]
                original = read_json(row["mask_native_metadata_path"])
                mask = sitk.ReadImage(row["model_mask_path"])
                shape = (int(original["n_slices"]), int(original["rows"]), int(original["cols"]))
                native_lps = acquisition_affine(original)
                expected = sitk.DICOMOrient(reference_image(shape, native_lps), "LPS")
                if mask.GetSize() != expected.GetSize() or not np.allclose(image_affine(mask), image_affine(expected), atol=2e-5):
                    raise ValueError("Mask grid differs from saved native physical geometry")
                data = sitk.GetArrayFromImage(mask)
                record["native_mask_voxels"] = int(np.count_nonzero(data))
                if record["registration_status"] == "rigid_fallback":
                    longitudinal = sitk.ReadTransform(str(files[1]))
                else:
                    pair_dir = files[1].parents[2]
                    worker.rebuild(row["metadata_source"]["path"], pair_dir, temporary)
                    expected_field = reference_image(tuple(meta["target_geometry"]["shape_zyx"]), centered_affine(meta["target_geometry"]))
                    values = np.load(temporary, allow_pickle=False).astype(np.float64)
                    if values.shape != (*meta["target_geometry"]["shape_zyx"], 3) or not np.isfinite(values).all():
                        raise ValueError("Invalid field replay array")
                    field = sitk.GetImageFromArray(values, isVector=True)
                    field.CopyInformation(expected_field)
                    if field.GetSize() != expected_field.GetSize() or not np.allclose(image_affine(field), image_affine(expected_field), atol=1e-5):
                        raise ValueError("Rebuilt field grid differs from saved T0 target")
                    longitudinal = sitk.DisplacementFieldTransform(field)
                phase = sitk.ReadTransform(str(files[0]))
                to_native = affine_transform(native_lps @ np.linalg.inv(centered_affine(meta["source_geometry"])))
                transform = compose_followup_transform(to_native, phase, longitudinal)
                report = read_json(report_path)
                native_ras = LPS_RAS @ acquisition_affine(meta)
                to_centered = centered_affine(meta["target_geometry"]) @ np.linalg.inv(native_ras)
                crop = reference_image(SHAPE, to_centered @ np.asarray(report["crop_affine_ras"]))
                full_shape, strict_ras = vox2out_vox((tuple(meta["target_geometry"]["shape_zyx"])[::-1], native_ras), voxel_sizes=SPACING[::-1])
                full = reference_image(full_shape[::-1], to_centered @ strict_ras)
                mapped_full = sitk.GetArrayFromImage(sitk.Resample(mask, full, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0
                mapped_crop = sitk.GetArrayFromImage(sitk.Resample(mask, crop, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)) > 0
                points = np.argwhere(mapped_full)
                inside = np.all(np.abs(points - report["center_strict_zyx"]) <= np.asarray(SHAPE) / 2 + 1e-7, axis=1)
                for xyz in ([0.0, 0.0, 0.0], [63.0, 63.0, 15.0], [127.0, 127.0, 31.0]):
                    fixed_point = crop.TransformContinuousIndexToPhysicalPoint(xyz)
                    sequential = to_native.TransformPoint(phase.TransformPoint(longitudinal.TransformPoint(fixed_point)))
                    if not np.allclose(transform.TransformPoint(fixed_point), sequential, atol=1e-6):
                        raise ValueError("Phase/longitudinal composition order mismatch")
                    mask_index = np.asarray(mask.TransformPhysicalPointToContinuousIndex(sequential))
                    nearest = np.floor(mask_index + 0.5).astype(int)
                    expected_value = bool(data[tuple(nearest[::-1])]) if np.all(nearest >= 0) and np.all(nearest < mask.GetSize()) else False
                    if bool(mapped_crop[tuple(np.asarray(xyz, dtype=int)[::-1])]) != expected_value:
                        raise ValueError("Independent follow-up nearest-label sample mismatch")
                for identity in input_sources:
                    verify_identity(identity)
                save_npz(mask_path, mask=mapped_crop)
                record.update(status="passed", actual_cropped_voxels=int(mapped_crop.sum()), registered_full_voxels=len(points),
                              retained_fraction=float(inside.mean()) if len(points) else None, crop_empty=not bool(mapped_crop.any()),
                              originally_empty=record["native_mask_voxels"] == 0, transform_sources=[file_identity(path) for path in files],
                              input_sources=input_sources, mask_identity=file_identity(mask_path),
                              retention_basis="registered_full_strict_grid_voxel_centers_within_existing_T0_FOV")
            except (ValueError, RuntimeError, OSError) as error:
                record["reason"] = str(error)
            finally:
                if temporary.exists():
                    temporary.unlink()
            write_json(marker, record)
            records.append(record)
            if (index + 1) % 25 == 0:
                progress = {"stage": "followup_audit", "completed": index + 1, "total": len(tasks), "updated_at": timestamp()}
                progress["shard"] = shard
                write_json(output / ("progress.json" if shard is None else f"progress-{shard}.json"), progress)
                print(json.dumps(progress), flush=True)
    finally:
        worker.close()
    summary = summarize(records)
    write_json(output / ("summary.json" if shard is None else f"summary-{shard}.json"), summary)
    return summary
