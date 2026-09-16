from __future__ import annotations

import fcntl
import json
import os
import signal
import stat
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .first_post_segmentation import (
    REPO,
    execution_configuration,
    export_case,
    finished,
    infer_logits,
    predictor,
    prepare,
    prepare_case_input,
    read_json,
    run_case,
    select_gpu,
    sustained_runtime_reasons,
    timestamp,
    write_json,
)


def run_batch(
    config: dict[str, Any], batch_path: Path, device: str, status_path: Path
) -> None:
    config = execution_configuration(config)
    root = Path(config["output_dir"])
    ids = read_json(batch_path)["case_ids"]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Worker batch must contain unique cases")
    inventory = {r["case_id"]: r for r in read_json(root / "inventory.json")["records"]}
    records = [inventory[case] for case in ids]
    pending = [r for r in records if not finished(root, r)]
    if not pending:
        return
    started = time.monotonic()
    write_json(
        status_path,
        {
            "status": "loading_model",
            "pid": os.getpid(),
            "device": device,
            "updated_at_utc": timestamp(),
            "batch_cases": len(pending),
        },
    )
    instance, _ = predictor(config, device)
    loading_seconds = time.monotonic() - started
    if config.get("inference_execution"):
        run_pipeline(config, pending, device, status_path, instance)
    else:
        for record in pending:
            run_case(config, record, device, instance=instance, status_path=status_path)
    write_json(
        batch_path.with_suffix(".result.json"),
        {
            "status": "completed",
            "updated_at_utc": timestamp(),
            "completed_cases": len(pending),
            "model_load_seconds": loading_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "device": device,
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )


def run_pipeline(config, records, device, status_path, instance):
    import torch

    prefetch = config["inference_execution"].get("prefetch_cases", 2)
    with (
        ThreadPoolExecutor(max_workers=prefetch) as preparation,
        ThreadPoolExecutor(max_workers=1) as exporting,
    ):
        futures = {
            index: preparation.submit(prepare_case_input, config, record, instance)
            for index, record in enumerate(records[:prefetch])
        }
        previous_export = None
        for index, record in enumerate(records):
            write_json(
                status_path,
                {
                    "status": "waiting_for_preprocessing",
                    "pid": os.getpid(),
                    "case_id": record["case_id"],
                    "device": device,
                    "updated_at_utc": timestamp(),
                },
            )
            prepared = futures.pop(index).result()
            next_index = index + prefetch
            if next_index < len(records):
                futures[next_index] = preparation.submit(
                    prepare_case_input, config, records[next_index], instance
                )
            write_json(
                status_path,
                {
                    "status": "segmenting",
                    "pid": os.getpid(),
                    "case_id": record["case_id"],
                    "device": device,
                    "updated_at_utc": timestamp(),
                    "inference_execution": config["inference_execution"],
                },
            )
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            logits = infer_logits(instance, prepared.pop("data"))
            prepared["stage_seconds"]["segmenting"] = time.monotonic() - started
            stats = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "tile_batch_size": instance.tile_batch_size,
                "mirror_batch_size": instance.mirror_batch_size,
                "largest_forward_batch": instance.largest_forward_batch,
                "worker_oom_reductions": instance.oom_reductions,
            }
            if previous_export is not None:
                previous_export.result()
            previous_export = exporting.submit(
                export_case,
                config,
                record,
                device,
                instance,
                prepared,
                logits,
                inference_stats=stats,
            )
            del logits, prepared
        if previous_export is not None:
            previous_export.result()
    write_json(
        status_path,
        {
            "status": "completed",
            "pid": os.getpid(),
            "device": device,
            "updated_at_utc": timestamp(),
        },
    )


def stop_worker(worker: dict[str, Any]) -> None:
    child = worker["process"]
    try:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    finally:
        worker["lock"].close()


def release_completed_input_cache(
    root: Path,
    source_root: Path,
    records: list[dict[str, Any]],
    *,
    target_headroom_bytes: int,
    max_advice_bytes: int = 16 * 1024**3,
) -> dict[str, Any]:
    from .first_post_vqgan import resources

    before = snapshot = resources(root)
    advised = files = skipped = 0
    source_root = source_root.resolve()
    for record in reversed(records):
        headroom = snapshot["memory_headroom_bytes"]
        if (
            headroom is None
            or headroom >= target_headroom_bytes
            or advised >= max_advice_bytes
        ):
            break
        if not finished(root, record):
            continue
        source = Path(record["source_image"])
        if source.is_symlink() or not source.resolve().is_relative_to(source_root):
            skipped += 1
            continue
        try:
            descriptor = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                info = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_size != record["size_bytes"]
                    or info.st_mtime_ns != record["local_mtime_ns"]
                ):
                    skipped += 1
                    continue
                length = min(info.st_size, max_advice_bytes - advised)
                if length:
                    # This releases clean cached input pages, without modifying file data.
                    os.posix_fadvise(descriptor, 0, length, os.POSIX_FADV_DONTNEED)
                    advised += length
                    files += 1
            finally:
                os.close(descriptor)
        except OSError:
            skipped += 1
        snapshot = resources(root)
    return {
        "event": "completed_input_cache_release",
        "updated_at_utc": timestamp(),
        "files_advised": files,
        "source_bytes_advised": advised,
        "skipped_files": skipped,
        "target_headroom_bytes": target_headroom_bytes,
        "max_advice_bytes": max_advice_bytes,
        "before": before,
        "after": snapshot,
    }


def run_parallel_queue(
    config: dict[str, Any], execution: dict[str, Any], config_path: Path
) -> None:
    from .first_post_vqgan import resource_reasons, resources

    root = Path(config["output_dir"])
    queue = execution["queue"]
    stop = False
    workers: list[dict[str, Any]] = []
    wake = threading.Event()

    def request_stop(*_: Any) -> None:
        nonlocal stop
        stop = True
        wake.set()

    def watch(child):
        child.wait()
        wake.set()

    previous_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = prepare(config)
        records = inventory["records"]
        for directory in ("worker_batches", "worker_status"):
            (root / directory).mkdir(exist_ok=True)
        next_admission = 0.0
        next_resource_check = 0.0
        next_cache_release = 0.0
        pressure_samples = 0
        batch_number = 0
        complete = sum(finished(root, r) for r in records)
        snapshot = resources(root)

        def status(name: str, **extra: Any) -> None:
            active = []
            for worker in workers:
                path = worker["status_path"]
                stage = read_json(path) if path.exists() else {}
                if stage.get("pid") != worker["process"].pid:
                    stage = {}
                active.append(
                    {
                        "physical_gpu": worker["gpu"],
                        "worker_pid": worker["process"].pid,
                        "batch_cases": len(worker["records"]),
                        "stage": stage,
                    }
                )
            value = {
                "status": name,
                "pid": os.getpid(),
                "updated_at_utc": timestamp(),
                "total_cases": len(records),
                "completed_cases": complete,
                "max_workers": queue["max_workers"],
                "cases_per_worker": queue["cases_per_worker"],
                "workers": active,
                "resources": snapshot,
                **extra,
            }
            write_json(root / "queue_status.json", value)
            write_json(root / "worker_status.json", value)

        try:
            while not stop:
                for worker in list(workers):
                    child = worker["process"]
                    if child.poll() is None:
                        continue
                    if child.returncode != 0 or not all(
                        finished(root, r) for r in worker["records"]
                    ):
                        raise RuntimeError(
                            f"Segmentation batch failed on GPU {worker['gpu']}; "
                            f"inspect {worker['log_path']}"
                        )
                    worker["lock"].close()
                    workers.remove(worker)

                pending = [r for r in records if not finished(root, r)]
                complete = len(records) - len(pending)
                if not pending and not workers:
                    status("completed")
                    return
                now = time.monotonic()
                if now >= next_resource_check:
                    snapshot = resources(root)
                    next_resource_check = now + queue["resource_poll_seconds"]
                    reasons, pressure_samples = sustained_runtime_reasons(
                        resource_reasons(snapshot, execution, runtime=True),
                        pressure_samples if workers else 0,
                    )
                    if reasons and workers:
                        for worker in workers:
                            stop_worker(worker)
                        workers.clear()
                        with (root / "resource_events.jsonl").open("a") as events:
                            events.write(
                                json.dumps(
                                    {
                                        "updated_at_utc": timestamp(),
                                        "reasons": reasons,
                                        "resources": snapshot,
                                    }
                                )
                                + "\n"
                            )
                        status("waiting_for_resources", reasons=reasons)
                        wake.wait(queue["resource_poll_seconds"])
                        wake.clear()
                        continue

                claimed = {r["case_id"] for w in workers for r in w["records"]}
                available = [r for r in pending if r["case_id"] not in claimed]
                reasons = []
                if (
                    available
                    and len(workers) < queue["max_workers"]
                    and now >= next_admission
                ):
                    snapshot = resources(root)
                    reasons = resource_reasons(snapshot, execution, runtime=False)
                    if (
                        reasons == ["host_memory"]
                        and now >= next_cache_release
                        and inventory.get("source_root")
                    ):
                        recovery = release_completed_input_cache(
                            root,
                            Path(inventory["source_root"]),
                            records,
                            target_headroom_bytes=(queue["admission_memory_gib"] + 8)
                            * 1024**3,
                        )
                        next_cache_release = time.monotonic() + 300
                        with (root / "resource_events.jsonl").open("a") as events:
                            events.write(json.dumps(recovery) + "\n")
                        snapshot = resources(root)
                        reasons = resource_reasons(snapshot, execution, runtime=False)
                    if not reasons:
                        gpu, gpu_lock = select_gpu(execution)
                        if gpu is not None:
                            batch = available[: queue["cases_per_worker"]]
                            batch_number += 1
                            name = f"gpu{gpu}_{os.getpid()}_{batch_number}"
                            batch_path = root / "worker_batches" / f"{name}.json"
                            log_path = root / "logs" / f"batch_{name}.log"
                            status_path = root / "worker_status" / f"gpu{gpu}.json"
                            environment = os.environ.copy()
                            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                            try:
                                write_json(
                                    batch_path,
                                    {"case_ids": [r["case_id"] for r in batch]},
                                )
                                with log_path.open("a") as log:
                                    child = subprocess.Popen(
                                        [
                                            config["python"],
                                            "-u",
                                            "-m",
                                            "mewm_ispy2.first_post_segmentation",
                                            "worker-batch",
                                            "--config",
                                            str(config_path),
                                            "--batch",
                                            str(batch_path),
                                            "--device",
                                            "cuda:0",
                                            "--status-file",
                                            str(status_path),
                                        ],
                                        cwd=REPO,
                                        env=environment,
                                        stdin=subprocess.DEVNULL,
                                        stdout=log,
                                        stderr=subprocess.STDOUT,
                                    )
                            except BaseException:
                                gpu_lock.close()
                                raise
                            workers.append(
                                {
                                    "process": child,
                                    "gpu": gpu,
                                    "lock": gpu_lock,
                                    "records": batch,
                                    "status_path": status_path,
                                    "log_path": log_path,
                                }
                            )
                            threading.Thread(
                                target=watch, args=(child,), daemon=True
                            ).start()
                            # Let model loading consume memory before admitting the second GPU.
                            next_admission = now + queue["admission_stagger_seconds"]
                state = (
                    "segmenting"
                    if workers
                    else "waiting_for_resources"
                    if reasons
                    else "waiting_for_gpu"
                )
                status(state, reasons=reasons, pressure_samples=pressure_samples)
                deadlines = [next_resource_check]
                if (
                    available
                    and len(workers) < queue["max_workers"]
                    and next_admission > now
                ):
                    deadlines.append(next_admission)
                wake.wait(max(0.05, min(deadlines) - time.monotonic()))
                wake.clear()
        except BaseException as error:
            status("failed", error=str(error))
            raise
        finally:
            for worker in workers:
                stop_worker(worker)
            workers.clear()
            if stop:
                complete = sum(finished(root, r) for r in records)
                status("paused", reasons=["requested_stop"])
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
