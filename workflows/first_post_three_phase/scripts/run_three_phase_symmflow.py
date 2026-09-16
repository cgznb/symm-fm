"""Queue and run the formal all-forward-pairs three-phase Symm-FM experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def workflow_active(path):
    lock = Path(path) / "workflow.lock"
    if not lock.exists():
        return False
    with lock.open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def gpu_available(cfg, gpu):
    protected = cfg["runtime"]["protected_pcr_runs"].get(gpu)
    if protected and workflow_active(protected):
        return False, "pcr_workflow_active"
    query = ["nvidia-smi", "-i", str(gpu)]
    processes = subprocess.check_output([*query, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if processes:
        return False, "gpu_compute_process_active"
    used = int(subprocess.check_output([*query, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
    return used <= 512, "available" if used <= 512 else "gpu_memory_allocated"


def worker(cfg, args, inventory):
    import torch
    from mewm_ispy2.first_post_world_data import read_json
    from mewm_ispy2.three_phase_all_pairs_data import prepare
    from mewm_ispy2.three_phase_all_pairs_training import evaluate_selected, smoke, train, verify_pcr_interface
    root = Path(cfg["output_root"])
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_per_process_memory_fraction(cfg["runtime"]["memory_fraction"])
    if args.stage in ("prepare", "all"):
        inventory = prepare(cfg, inventory, lambda value: progress(cfg, value))
        if inventory is None:
            return
    else:
        inventory = read_json(root / "admitted_inventory.json")
    if args.stage == "prepare":
        return
    if args.stage in ("smoke", "all"):
        report = root / "smoke" / "report.json"
        if not report.exists():
            smoke(cfg, inventory, lambda value: progress(cfg, value))
        elif not read_json(report)["passed"]:
            raise ValueError("Existing GPU smoke did not pass")
    if args.stage in ("train", "all"):
        if not train(cfg, inventory, lambda value: progress(cfg, value), resume=args.resume, stop_after=args.stop_after):
            return
    if args.stage in ("evaluate", "sample", "all"):
        evaluate_selected(cfg, inventory, lambda value: progress(cfg, value))
    if args.stage in ("pcr-interface", "all"):
        verify_pcr_interface(cfg, inventory, lambda value: progress(cfg, value))
    progress(cfg, {"stage": "complete", "requested_stage": args.stage})


def progress(cfg, value):
    from mewm_ispy2.first_post_world_data import now, write_json
    row = {**value, "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "updated_utc": now()}
    write_json(Path(cfg["output_root"]) / "progress.json", row)
    print(json.dumps(row, allow_nan=False), flush=True)


def controller(cfg, args, inventory):
    from mewm_ispy2.first_post_world_data import now, read_json, write_json
    root = Path(cfg["output_root"])
    stopped, child = [], None
    def stop(signum, frame):
        stopped.append(signum)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)}
    lock_root = Path(cfg["runtime"]["gpu_lock_root"])
    lock_root.mkdir(parents=True, exist_ok=True)
    selected_lock = None
    try:
        while not stopped:
            reasons = {}
            for gpu in cfg["runtime"]["gpu_priority"]:
                handle = (lock_root / f"gpu{gpu}.lock").open("a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    available, reason = gpu_available(cfg, gpu)
                except BlockingIOError:
                    available, reason = False, "shared_gpu_lock_held"
                if available:
                    selected_lock = handle
                    break
                handle.close()
                reasons[str(gpu)] = reason
            if selected_lock is not None:
                break
            progress(cfg, {"stage": "waiting_for_gpu", "reasons": reasons,
                           "pair_counts": inventory["pair_counts"], "optimizer_updates": 0})
            write_json(root / "controller_status.json", {"stage": "waiting_for_gpu", "pid": os.getpid(),
                       "reasons": reasons, "updated_utc": now()})
            if not args.wait_for_gpu:
                raise RuntimeError("All candidate GPUs are occupied; use --wait-for-gpu")
            time.sleep(cfg["runtime"]["poll_seconds"])
        if stopped:
            progress(cfg, {"stage": "paused", "reason": "controller_signal_before_gpu_acquisition"})
            return
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
                   "--stage", args.stage, "--worker"]
        if args.resume:
            command.append("--resume")
        if args.stop_after is not None:
            command.extend(["--stop-after", str(args.stop_after)])
        child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                                 "THREE_PHASE_CONTROLLER_PID": str(os.getpid()), "PYTHONDONTWRITEBYTECODE": "1",
                                 "OMP_NUM_THREADS": str(cfg["runtime"]["cpu_threads"])})
        write_json(root / "controller_status.json", {"stage": "running", "pid": os.getpid(), "worker_pid": child.pid,
                   "gpu": gpu, "updated_utc": now()})
        result = child.wait()
        if result:
            raise RuntimeError(f"Formal worker exited with status {result}; see workflow.log and progress.json")
        stage = read_json(root / "progress.json")["stage"]
        write_json(root / "controller_status.json", {"stage": stage, "pid": os.getpid(), "gpu": gpu, "updated_utc": now()})
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait()
        if selected_lock is not None:
            selected_lock.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    from mewm_ispy2.first_post_world_data import now, write_json
    from mewm_ispy2.three_phase_all_pairs_data import load_config, prepare_inventory, storage_requirement
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/three_phase_symmflow_all_pairs_t0_v2.yaml")
    parser.add_argument("--stage", choices=("inventory", "prepare", "smoke", "train", "evaluate", "sample", "pcr-interface", "all"), default="all")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be positive")
    cfg = load_config(args.config)
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "crop_policy_hold.json").exists() and args.stage != "inventory":
        raise ValueError("This experiment is held for a crop-policy correction; use the T0 v2 configuration")
    if args.detach:
        if args.worker:
            parser.error("A worker cannot detach")
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--config", str(args.config.resolve()), "--stage", args.stage]
        command.extend(flag for flag, enabled in (("--wait-for-gpu", args.wait_for_gpu), ("--resume", args.resume)) if enabled)
        if args.stop_after is not None:
            command.extend(["--stop-after", str(args.stop_after)])
        with (root / "workflow.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True,
                                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        result = {"pid": process.pid, "config": str(args.config.resolve()), "created_utc": now(), "log": str(root / "workflow.log")}
        write_json(root / "launch.json", result)
        print(json.dumps(result), flush=True)
        return
    if args.worker:
        if os.environ.get("THREE_PHASE_CONTROLLER_PID") != str(os.getppid()):
            parser.error("Workers must be started by the GPU-lock-owning controller")
        try:
            worker(cfg, args, prepare_inventory(cfg))
        except BaseException as error:
            progress(cfg, {"stage": "failed", "error": f"{type(error).__name__}: {error}"})
            raise
        return
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            progress(cfg, {"stage": "checking_inventory"})
            inventory = prepare_inventory(cfg)
            if args.stage == "inventory":
                progress(cfg, {"stage": "inventory_ready", "pair_counts": inventory["pair_counts"],
                               "views": len(inventory["views"]), "storage_estimate": storage_requirement(inventory)})
                return
            controller(cfg, args, inventory)
        except BaseException as error:
            write_json(root / "controller_status.json", {"stage": "failed", "pid": os.getpid(), "updated_utc": now(),
                       "error": f"{type(error).__name__}: {error}"})
            raise


if __name__ == "__main__":
    main()
