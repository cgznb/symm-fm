"""Resume-safe CPU preparation and queued VQ-GAN -> SymmFlow execution."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mewm_ispy2 import registered_roi32_runtime as runtime
from mewm_ispy2.registered_roi32_data import (CropDataset, build_inventory, check_disk, file_identity, prepare,
                                             read_config, read_json, timestamp, verify_identity, write_json)


def progress(config, status, **values):
    payload = {"status": status, "updated_at": timestamp(), "pid": os.getpid(), **values}
    write_json(Path(config["output_dir"]) / "progress.json", payload)
    print(json.dumps(payload, allow_nan=False), flush=True)


def gpu_snapshot():
    command = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    rows = list(csv.reader(subprocess.check_output(command, text=True, timeout=15).splitlines()))
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True, timeout=15)
    occupied = {row[0].strip() for row in csv.reader(apps.splitlines()) if row}
    return [{"index": int(row[0]), "memory_mib": int(row[2]), "utilization": int(row[3]),
             "occupied": row[1].strip() in occupied} for row in rows]


def reserve_gpu(config):
    while not runtime.STOP_REQUESTED:
        snapshot = gpu_snapshot()
        for index in config["runtime"]["gpu_candidates"]:
            state = next((row for row in snapshot if row["index"] == index), None)
            if state is None or state["occupied"] or state["memory_mib"] > 512 or state["utilization"] > 5:
                continue
            lock = Path(f"/tmp/registered_roi32_gpu_{index}.lock").open("a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            time.sleep(2)
            latest = next(row for row in gpu_snapshot() if row["index"] == index)
            if latest["occupied"] or latest["memory_mib"] > 512 or latest["utilization"] > 5:
                lock.close()
                continue
            return index, lock
        progress(config, "waiting_for_idle_gpu", gpus=snapshot)
        for _ in range(config["runtime"]["queue_poll_seconds"]):
            if runtime.STOP_REQUESTED:
                return None, None
            time.sleep(1)
    return None, None


def ensure_data(config):
    root = Path(config["output_dir"]) / "data"
    if not (root / "COMPLETE.json").exists():
        prepare(config)
    dataset = CropDataset(config)
    if build_inventory(config) != dataset.inventory:
        raise ValueError("The original cohort or source files changed after preparation")
    for patient in dataset.inventory["patients"]:
        report = read_json(root / "patients" / (patient["patient_id"] + ".json"))
        for identity in report["source_identities"] + report["cache_identities"]:
            verify_identity(identity)


def select_batch(config, stage, device):
    from mewm_ispy2.registered_roi32_smoke import fm_smoke, vq_smoke
    from mewm_ispy2.registered_roi32_vq import contract_for as vq_contract
    from mewm_ispy2.registered_roi32_fm import contract_for as fm_contract
    import torch
    contract = vq_contract(config) if stage == "vq" else fm_contract(config)
    report_path = Path(config["output_dir"]) / "smoke" / (stage + ".json")
    if report_path.exists():
        result = read_json(report_path)
        if result["contract"] == contract and result["status"] == "passed" and result["device"].startswith("cuda"):
            return result["batch_size"]
    action = vq_smoke if stage == "vq" else fm_smoke
    for batch in dict.fromkeys([config[stage]["batch_size"], config[stage]["fallback_batch_size"]]):
        progress(config, "gpu_smoke", stage=stage, batch_size=batch)
        try:
            result = action(config, device, batch)
            gc.collect()
            torch.cuda.empty_cache()
            return result["batch_size"]
        except torch.cuda.OutOfMemoryError:
            progress(config, "gpu_smoke_batch_oom", stage=stage, batch_size=batch)
        gc.collect()
        torch.cuda.empty_cache()
    raise RuntimeError("Both configured physical batches exceed GPU memory")


def execute(config, stage, cpu=False):
    import torch
    from mewm_ispy2 import registered_roi32_evaluation as evaluation
    from mewm_ispy2 import registered_roi32_fm as fm
    from mewm_ispy2 import registered_roi32_latents as latents
    from mewm_ispy2 import registered_roi32_vq as vq
    from mewm_ispy2.registered_roi32_followup import run_audit
    loaded_sources = [file_identity(path) for path in
                      (__file__, runtime.__file__, evaluation.__file__, fm.__file__, latents.__file__, vq.__file__,
                       vq.data_module.__file__, vq.vq_module.__file__, vq.perceptual_module.__file__)]
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    progress(config, "checking_prepared_data")
    ensure_data(config)
    if stage in ("all", "prepare", "preview"):
        evaluation.preview_crops(config)
    if stage in ("prepare", "preview"):
        progress(config, "preparation_complete")
        return
    if stage == "audit":
        progress(config, "auditing_followup_masks")
        run_audit(config)
        progress(config, "followup_audit_complete")
        return
    if stage == "all":
        progress(config, "auditing_followup_masks")
        audit = run_audit(config)
        if audit["statuses"].get("passed", 0) != audit["visits"]:
            raise RuntimeError("Follow-up mapping audit is incomplete; inspect followup_audit/summary.json")
    if cpu:
        if stage != "smoke-vq":
            raise ValueError("--cpu is restricted to the separate VQ smoke test")
        from mewm_ispy2.registered_roi32_smoke import vq_smoke
        vq_smoke(config, torch.device("cpu"), config["vq"]["batch_size"])
        progress(config, "cpu_smoke_complete")
        return
    index, gpu_lock = reserve_gpu(config)
    if index is None:
        progress(config, "paused_waiting_for_gpu")
        return
    try:
        for identity in loaded_sources:
            verify_identity(identity)
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was initialized before selecting an idle GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        device = torch.device("cuda:0")
        torch.use_deterministic_algorithms(config["runtime"]["deterministic_algorithms"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        progress(config, "gpu_reserved", physical_gpu=index, stage=stage)
        check_disk(config, 6 * 1024**3)
        if stage in ("all", "vq", "smoke-vq"):
            batch = select_batch(config, "vq", device)
            if stage == "smoke-vq":
                progress(config, "vq_smoke_complete")
                return
            progress(config, "training_vq", physical_gpu=index)
            if not vq.train(config, device, batch):
                return
            gc.collect()
            torch.cuda.empty_cache()
            if not evaluation.evaluate_vq(config, device):
                return
            if stage == "vq":
                progress(config, "vq_complete")
                return
        if stage in ("all", "latents"):
            progress(config, "encoding_latents", physical_gpu=index)
            if not latents.extract(config, device, config["vq"]["batch_size"]):
                return
            gc.collect()
            torch.cuda.empty_cache()
            if stage == "latents":
                progress(config, "latents_complete")
                return
        if stage in ("all", "fm", "smoke-fm"):
            batch = select_batch(config, "fm", device)
            if stage == "smoke-fm":
                progress(config, "fm_smoke_complete")
                return
            progress(config, "training_fm", physical_gpu=index)
            if not fm.train(config, device, batch):
                return
            gc.collect()
            torch.cuda.empty_cache()
        if stage in ("all", "fm", "evaluate"):
            progress(config, "evaluating_fm", physical_gpu=index)
            if not evaluation.evaluate_fm(config, device):
                return
            write_json(Path(config["output_dir"]) / "COMPLETE.json", {"status": "passed", "configuration": config, "updated_at": timestamp()})
        progress(config, "complete", stage=stage, physical_gpu=index)
    finally:
        gpu_lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["all", "prepare", "preview", "audit", "smoke-vq", "vq", "latents", "smoke-fm", "fm", "evaluate"], default="all")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = read_config(config_path)
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with runtime.output_lock(root, args.lock_fd) as lock:
        if args.detach and not args.worker:
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--config", str(config_path), "--stage", args.stage,
                       "--worker", "--lock-fd", str(lock.fileno())]
            if args.cpu:
                command.append("--cpu")
            environment = dict(os.environ, PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS=str(config["runtime"]["cpu_threads"]),
                               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CUBLAS_WORKSPACE_CONFIG=":4096:8")
            with (root / "controller.log").open("a") as handle:
                child = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL, stdout=handle,
                                         stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
            write_json(root / "launch.json", {"pid": child.pid, "configuration": file_identity(config_path), "stage": args.stage, "started_at": timestamp()})
            print(json.dumps({"pid": child.pid, "log": str(root / "controller.log"), "progress": str(root / "progress.json")}))
            return
        runtime.install_signals()
        try:
            execute(config, args.stage, cpu=args.cpu)
            if runtime.STOP_REQUESTED:
                progress(config, "paused", stage=args.stage)
        except Exception as error:
            progress(config, "failed", stage=args.stage, error=str(error))
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
