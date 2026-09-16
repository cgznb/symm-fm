#!/usr/bin/env python3
"""Run the two approved first-post arms with process-exit stage barriers."""

from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mewm_ispy2.first_post_world_data import config, disk_gate, identity, now, public, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['profile', 'smoke', 'train', 'all'], default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--arms', nargs='+', choices=['bifm', 'symm'], default=['bifm', 'symm'],
                        help='Select existing training arms; single-arm execution requires --stage train')
    args = parser.parse_args()
    if set(args.arms) != {'bifm', 'symm'} and args.stage != 'train':
        parser.error('Single-arm execution is supported only for the train stage')
    config_path = Path(args.config).resolve()
    cfg = config(config_path)
    root = Path(cfg['output_root'])
    if not read_json(root / 'shared/bundle.json').get('verified_for_world_training'):
        raise ValueError('Complete shared data preparation first')
    disk_gate(root, reserve_gib=cfg['disk_reserve_gib'])
    children, mutex, stop = {}, threading.Lock(), threading.Event()
    locks = []
    def halt(signum, frame):
        stop.set()
        with mutex:
            for arm, (child, stage) in children.items():
                if child.poll() is not None:
                    continue
                ready = root / arm / ('smoke' if stage == 'smoke' else 'train') / 'ready.json'
                state = read_json(ready) if ready.exists() else {}
                if state.get('pid') == child.pid and state.get('active'):
                    child.send_signal(signal.SIGUSR1)
                else:
                    os.killpg(child.pid, signal.SIGTERM)
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    with (root / 'controller.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_root = Path(cfg.get('gpu_lock_root', str(root / 'gpu_locks')))
        lock_root.mkdir(parents=True, exist_ok=True)
        for arm in dict.fromkeys(args.arms):
            gpu = cfg['gpu_assignments'][arm]
            lock = (lock_root / f'gpu{gpu}.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        def run(arm, stage):
            env = os.environ.copy()
            temp = Path(cfg['temp_root']) / arm
            temp.mkdir(parents=True, exist_ok=True)
            env.update(CUDA_VISIBLE_DEVICES=str(cfg['gpu_assignments'][arm]), OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', TMPDIR=str(temp),
                       PYTHONUNBUFFERED='1')
            command = [sys.executable, '-u', '-m', 'mewm_ispy2.first_post_world_training', stage,
                       '--config', str(config_path), '--arm', arm]
            if args.resume and stage == 'train':
                command.append('--resume')
            with (root / arm / f'{stage}.log').open('a') as log:
                child = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, start_new_session=True,
                                         pass_fds=tuple(lock.fileno() for lock in locks))
                with mutex:
                    children[arm] = (child, stage)
                write_json(root / arm / 'process.json', {'pid': child.pid, 'stage': stage, 'gpu': cfg['gpu_assignments'][arm],
                                                        'controller_pid': os.getpid(), 'started_utc': now()})
                for line in child.stdout:
                    log.write(public(line))
                    log.flush()
                code = child.wait()
                if code:
                    raise RuntimeError(f'{arm} {stage} exited with code {code}; see {log.name}')
                result_path = root / arm / ('profile.json' if stage == 'profile' else f'{stage}/result.json')
                result = read_json(result_path)
                if result['status'] != 'completed' and not (stop.is_set() and result['status'] == 'paused'):
                    raise RuntimeError(f'{arm} {stage} ended as {result["status"]}')
                return arm
        try:
            stages = ['profile', 'smoke', 'train'] if args.stage == 'all' else [args.stage]
            tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_first_post_world.py'],
                                   cwd=REPO, text=True, capture_output=True)
            write_json(root / 'tests.json', {'passed': tests.returncode == 0, 'summary': tests.stdout[-3000:], 'updated_utc': now()})
            if tests.returncode:
                raise RuntimeError('Focused first-post contract tests failed')
            for stage in stages:
                if stop.is_set():
                    break
                write_json(root / 'controller_status.json', {'status': 'running', 'stage': stage, 'pid': os.getpid(), 'updated_utc': now()})
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(run, arm, stage) for arm in dict.fromkeys(args.arms)]
                    for future in as_completed(futures):
                        try:
                            arm = future.result()
                        except BaseException:
                            halt(signal.SIGTERM, None)
                            raise
                        print(json.dumps({'event': 'stage_complete', 'arm': arm, 'stage': stage}), flush=True)
                if stage == 'profile':
                    profiles = {arm: read_json(root / arm / 'profile.json') for arm in ('bifm', 'symm')}
                    if any(p['bundle_identity'] != identity(root / 'shared/bundle.json') for p in profiles.values()):
                        raise ValueError('Data changed during profiling')
                    batches = {arm: p['selected_microbatch'] for arm, p in profiles.items()}
                    write_json(root / 'execution_profile.json', {'effective_batch': max(4, *batches.values()),
                               'microbatches': batches, 'bundle_identity': identity(root / 'shared/bundle.json'),
                               'fitted_utc': now(), 'selection': 'highest_measured_throughput_below_vram_limit'})
            write_json(root / 'controller_status.json', {'status': 'paused' if stop.is_set() else 'completed',
                       'last_stage': stages[-1], 'updated_utc': now()})
        except BaseException as exc:
            halt(signal.SIGTERM, None)
            write_json(root / 'controller_status.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                                                        'updated_utc': now()})
            raise
        finally:
            with mutex:
                pending = [p for p, _ in children.values() if p.poll() is None]
            for child in pending:
                try:
                    child.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            for lock in locks:
                lock.close()


if __name__ == '__main__':
    main()
