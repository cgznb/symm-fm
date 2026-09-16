from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .first_post_world_data import (
    GIB, SCHEMA, PairDataset, PatientBalancedBatchSampler, collate, config, disk_gate,
    identity, load_codec, now, public, read_json, write_json,
)
from .first_post_world_evaluation import to_device, validate
from .first_post_world_models import build_model, compact_state, optimizer_for, restore_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def atomic_checkpoint(path, payload, root):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (root / 'checkpoint_io.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        temporary = path.with_suffix('.tmp.ckpt')
        try:
            torch.save(payload, temporary)
            with temporary.open('rb') as handle:
                os.fsync(handle.fileno())
            torch.load(temporary, map_location='cpu', weights_only=False, mmap=True)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def save_recovery(path, model, optimizer, scheduler, ema, *, step, contract, best, root):
    payload = {'schema': SCHEMA, 'contract': contract, 'optimizer_step': step,
               'model': compact_state(model), 'optimizer': cpu_tree(optimizer.state_dict()),
               'scheduler': scheduler.state_dict() if scheduler else None,
               'ema': cpu_tree(ema.state_dict()) if ema else None,
               'rng': rng_state(), 'best_foreground_mae': best, 'updated_utc': now()}
    atomic_checkpoint(path, payload, root)


def restore_recovery(path, model, optimizer, scheduler, ema, contract):
    value = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if value['schema'] != SCHEMA or value['contract'] != contract:
        raise ValueError('Resume configuration or shared dataset differs')
    restore_model(model, value['model'])
    optimizer.load_state_dict(value['optimizer'])
    if scheduler:
        scheduler.load_state_dict(value['scheduler'])
    if ema:
        ema.load_state_dict(value['ema'])
    restore_rng(value['rng'])
    return int(value['optimizer_step']), value['best_foreground_mae']


def train_step(model, optimizer, scheduler, ema, iterator, accumulation, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, details = 0.0, {}
    for _ in range(accumulation):
        batch = to_device(next(iterator), device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, parts = model.loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite flow loss')
        (loss / accumulation).backward()
        total += float(loss.detach()) / accumulation
        for key, value in parts.items():
            details[key] = details.get(key, 0) + value / accumulation
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    if scheduler:
        scheduler.step()
    if ema:
        ema.update(model)
    optimizer.zero_grad(set_to_none=True)
    return {'loss': total, 'gradient_norm': float(norm), **details}


def make_loader(dataset, cfg, microbatch, effective, start, stop):
    sampler = PatientBalancedBatchSampler(dataset.records, microbatch, effective, start, stop, cfg['seed'])
    workers = cfg['loader_workers']
    kwargs = {'persistent_workers': True, 'prefetch_factor': 1} if workers else {}
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=workers,
                      pin_memory=True, generator=torch.Generator().manual_seed(cfg['seed']), **kwargs)


def profile(cfg, arm):
    root = Path(cfg['output_root'])
    output = root / arm
    output.mkdir(parents=True, exist_ok=True)
    seed_all(cfg['seed'])
    dataset, validation = PairDataset(root, arm, 'train'), PairDataset(root, arm, 'val')
    model = build_model(cfg, arm, dataset.records)
    optimizer, scheduler, ema = optimizer_for(model, arm, cfg['max_optimizer_steps'])
    codec = load_codec(root / 'shared/codec.pt', 'cuda:0')
    tested, microbatch = [], 1
    duration = cfg['profile_warmup_steps'] + cfg['profile_measure_steps']
    total_memory = torch.cuda.get_device_properties(0).total_memory
    while True:
        loader = iterator = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            loader = make_loader(dataset, cfg, microbatch, microbatch, 0, duration)
            iterator = iter(loader)
            seconds = []
            for i in range(duration):
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = train_step(model, optimizer, scheduler, ema, iterator, 1, 'cuda:0')
                torch.cuda.synchronize()
                if i >= cfg['profile_warmup_steps']:
                    seconds.append(time.perf_counter() - started)
            validation_start = time.perf_counter()
            validate(model, validation, codec, cfg, step=duration, indices=[0], samples=1, ema=ema, reference=False)
            validation_seconds = time.perf_counter() - validation_start
            peak = torch.cuda.max_memory_reserved()
            free, total = torch.cuda.mem_get_info()
            external = max(0, total - free - torch.cuda.memory_reserved())
            row = {'microbatch': microbatch, 'seconds_per_microstep': float(np.median(seconds)),
                   'examples_per_second': microbatch / float(np.median(seconds)),
                   'validation_seconds_per_pair': validation_seconds,
                   'peak_reserved_bytes': peak, 'driver_and_external_bytes': external,
                   'memory_fraction': (peak + external) / total_memory,
                   'finite_loss': result['loss'], 'admitted': peak + external <= total_memory * cfg['maximum_vram_fraction']}
            tested.append(row)
            print(json.dumps({'event': 'profile', 'arm': arm, **row}), flush=True)
            if not row['admitted']:
                break
            microbatch *= 2
        except torch.OutOfMemoryError:
            tested.append({'microbatch': microbatch, 'admitted': False, 'reason': 'cuda_oom'})
            optimizer.zero_grad(set_to_none=True)
            break
        finally:
            del iterator, loader
            gc.collect()
            torch.cuda.empty_cache()
    admitted = [r for r in tested if r['admitted']]
    if not admitted:
        write_json(output / 'profile.json', {'status': 'blocked', 'tested': tested, 'reason': 'no_admissible_microbatch'})
        raise RuntimeError('No microbatch fits the approved VRAM margin')
    selected = max(admitted, key=lambda r: r['examples_per_second'])
    write_json(output / 'profile.json', {'status': 'completed', 'arm': arm, 'gpu': cfg['gpu_assignments'][arm],
               'bundle_identity': identity(root / 'shared/bundle.json'),
               'selected_microbatch': selected['microbatch'], 'selected': selected, 'tested': tested,
               'device': torch.cuda.get_device_name(), 'model': model.description,
               'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 'completed_utc': now()})


def runtime_contract(cfg, arm, profile_values):
    root = Path(cfg['output_root'])
    return {'schema': SCHEMA, 'arm': arm, 'seed': cfg['seed'],
            'max_optimizer_steps': cfg['max_optimizer_steps'],
            'bundle_identity': identity(root / 'shared/bundle.json'),
            'codec_identity': identity(root / 'shared/codec.pt'),
            'effective_batch': profile_values['effective_batch'],
            'microbatch': profile_values['microbatches'][arm],
            'evaluation_solver_steps': cfg['solver_steps'],
            'light_validation_interval': cfg['light_validation_interval'],
            'full_validation_interval': cfg['full_validation_interval']}


def remaining_runtime(cfg, completed, seconds_per_update, validation_pairs, seconds_per_pair):
    total = cfg['max_optimizer_steps']
    full = total // cfg['full_validation_interval'] - completed // cfg['full_validation_interval']
    light = total // cfg['light_validation_interval'] - completed // cfg['light_validation_interval'] - full
    scheduled = full * validation_pairs + light * min(cfg['light_validation_pairs'], validation_pairs)
    final = 2 * cfg['final_samples'] * validation_pairs
    training_seconds = (total - completed) * seconds_per_update
    return {'training_only_eta_hours': training_seconds / 3600,
            'estimated_eta_hours_including_validation': (training_seconds + (scheduled + final) * seconds_per_pair) / 3600,
            'eta_validation_seconds_per_pair': seconds_per_pair,
            'eta_basis': 'measured_train_and_validation; includes_final_best_last; excludes_checkpoint_and_new_reference_cost'}


def verify_validation(cfg, arm):
    root = Path(cfg['output_root'])
    profiles = read_json(root / 'execution_profile.json')
    contract = runtime_contract(cfg, arm, profiles)
    seed_all(cfg['seed'])
    dataset, validation = PairDataset(root, arm, 'train'), PairDataset(root, arm, 'val')
    model = build_model(cfg, arm, dataset.records)
    optimizer, scheduler, ema = optimizer_for(model, arm, cfg['max_optimizer_steps'])
    step, _ = restore_recovery(root / arm / 'train/checkpoints/recovery.ckpt', model, optimizer, scheduler, ema, contract)
    codec = load_codec(root / 'shared/codec.pt', 'cuda:0')
    torch.cuda.reset_peak_memory_stats()
    evaluation = validate(model, validation, codec, cfg, step=step, indices=[0, len(validation) - 1],
                          samples=1, ema=ema, output=root / arm / 'validation_device_probe.json')
    peak = torch.cuda.max_memory_reserved()
    free, total = torch.cuda.mem_get_info()
    external = max(0, total - free - torch.cuda.memory_reserved())
    admitted = peak + external <= total * cfg['maximum_vram_fraction']
    result = {'passed': admitted, 'arm': arm, 'optimizer_step': step,
              'bundle_identity': contract['bundle_identity'], 'memory_fraction': (peak + external) / total,
              'peak_reserved_bytes': peak, 'driver_and_external_bytes': external,
              'timing': evaluation['timing'], 'completed_utc': now()}
    write_json(root / arm / 'validation_device_profile.json', result)
    if not admitted:
        raise RuntimeError('GPU validation exceeds the approved memory margin')
    print(json.dumps(result), flush=True)


def train(cfg, arm, *, smoke=False, resume=False):
    root = Path(cfg['output_root'])
    output = root / arm / ('smoke' if smoke else 'train')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        profiles = read_json(root / 'execution_profile.json')
        contract = runtime_contract(cfg, arm, profiles)
        previous_contract = output / 'contract.json'
        if previous_contract.exists() and read_json(previous_contract) != contract:
            raise ValueError('Existing experiment has a different runtime contract')
        write_json(previous_contract, contract)
        seed_all(cfg['seed'])
        dataset, validation = PairDataset(root, arm, 'train'), PairDataset(root, arm, 'val')
        validation_profile = root / arm / 'validation_device_profile.json'
        if validation_profile.exists():
            measured = read_json(validation_profile)
            if not measured['passed'] or measured['bundle_identity'] != contract['bundle_identity']:
                raise ValueError('Validation resource profile differs')
            validation_seconds = measured['timing']['seconds_per_pair_excluding_references']
        else:
            validation_seconds = read_json(root / arm / 'profile.json')['selected']['validation_seconds_per_pair']
        model = build_model(cfg, arm, dataset.records)
        optimizer, scheduler, ema = optimizer_for(model, arm, cfg['max_optimizer_steps'])
        codec = load_codec(root / 'shared/codec.pt', 'cuda:0')
        if any(p.requires_grad for p in codec.parameters()):
            raise ValueError('Codec is not frozen')
        checkpoint = output / 'checkpoints/recovery.ckpt'
        start, best = 0, None
        if resume:
            start, best = restore_recovery(checkpoint, model, optimizer, scheduler, ema, contract)
        elif checkpoint.exists():
            raise ValueError('Existing recovery state requires explicit resume')
        target = 4 if smoke else cfg['max_optimizer_steps']
        microbatch, effective = contract['microbatch'], contract['effective_batch']
        loader = make_loader(dataset, cfg, microbatch, effective, start, target)
        iterator = iter(loader)
        stopping = []
        def request_pause(signum, frame):
            stopping.append(f'signal_{signum}')
        for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, request_pause)
        write_json(output / 'ready.json', {'pid': os.getpid(), 'active': True, 'arm': arm, 'contract': contract})
        completed = start
        begin = time.perf_counter()
        recent = []
        light_indices = sorted(np.random.default_rng(1729).choice(len(validation),
                               min(cfg['light_validation_pairs'], len(validation)), replace=False).tolist())
        status, error = 'training', None
        try:
            for step in range(start, target):
                began_step = time.perf_counter()
                details = train_step(model, optimizer, scheduler, ema, iterator, effective // microbatch, 'cuda:0')
                completed = step + 1
                recent.append(time.perf_counter() - began_step)
                if completed % 25 == 0 or smoke or completed == 1:
                    seconds = float(np.mean(recent[-100:]))
                    record = {'status': 'training', 'arm': arm, 'optimizer_step': completed,
                              'max_optimizer_steps': cfg['max_optimizer_steps'], 'microbatch': microbatch,
                              'effective_batch': effective, 'seconds_per_update': seconds,
                              **remaining_runtime(cfg, completed, seconds, len(validation), validation_seconds),
                              'elapsed_seconds': time.perf_counter() - begin,
                              'equivalent_epochs': completed * effective / len(dataset),
                              'gpu_reserved_gib': torch.cuda.memory_reserved() / GIB,
                              'best_foreground_mae': best, 'updated_utc': now(), **details}
                    write_json(output / 'progress.json', record)
                    print(json.dumps(record), flush=True)
                if completed % 100 == 0:
                    try:
                        disk_gate(root, reserve_gib=cfg['disk_reserve_gib'])
                    except RuntimeError as exc:
                        stopping.append(str(exc))
                full = not smoke and completed % cfg['full_validation_interval'] == 0
                light = not smoke and completed % cfg['light_validation_interval'] == 0
                if full or light:
                    write_json(output / 'progress.json', {**record, 'status': 'validating', 'optimizer_step': completed})
                    evaluation = validate(model, validation, codec, cfg, step=completed,
                        indices=None if full else light_indices, samples=1, ema=ema,
                        output=output / 'evaluation' / f'{completed:06d}_{"full" if full else "light"}.json')
                    validation_seconds = evaluation['timing']['seconds_per_pair_excluding_references']
                    score = evaluation['summary']['patient_macro']['foreground_mae']
                    if full and (best is None or score < best):
                        best = score
                        with ema.average_parameters(model) if ema else nullcontext():
                            atomic_checkpoint(output / 'checkpoints/best.ckpt',
                                {'schema': SCHEMA, 'contract': contract, 'optimizer_step': completed,
                                 'model': compact_state(model), 'best_foreground_mae': best,
                                 'evaluation_weights': 'ema' if ema else 'ordinary'}, root)
                if light or stopping or completed == target or (smoke and completed == 2):
                    save_recovery(checkpoint, model, optimizer, scheduler, ema, step=completed,
                                  contract=contract, best=best, root=root)
                if smoke and completed == 2:
                    expected_rng = rng_state()
                    parameter = next(p for p in model.parameters() if p.requires_grad and p in optimizer.state)
                    expected_weight = parameter.detach().flatten()[:16].clone()
                    expected_moment = optimizer.state[parameter]['exp_avg'].flatten()[:16].clone()
                    expected_optimizer_step = optimizer.state[parameter]['step'].clone()
                    with torch.no_grad():
                        parameter.flatten()[:16].add_(1)
                        optimizer.state[parameter]['exp_avg'].flatten()[:16].add_(1)
                        optimizer.state[parameter]['step'].add_(1)
                    torch.rand(3)
                    torch.rand(3, device='cuda:0')
                    restored_step, _ = restore_recovery(checkpoint, model, optimizer, scheduler, ema, contract)
                    if (restored_step != completed or not torch.equal(expected_rng['torch'], torch.get_rng_state())
                            or not all(torch.equal(a, b) for a, b in zip(expected_rng['cuda'], torch.cuda.get_rng_state_all()))
                            or not torch.equal(parameter.flatten()[:16], expected_weight)
                            or not torch.equal(optimizer.state[parameter]['exp_avg'].flatten()[:16], expected_moment)
                            or not torch.equal(optimizer.state[parameter]['step'], expected_optimizer_step)):
                        raise ValueError('Real optimizer checkpoint/RNG resume failed')
                    write_json(output / 'resume_check.json', {'passed': True, 'optimizer_step': restored_step,
                                                              'optimizer_states': len(optimizer.state)})
                if stopping:
                    status = 'paused'
                    break
            if status != 'paused':
                status = 'completed'
            if smoke:
                validate(model, validation, codec, cfg, step=completed, indices=[0], samples=1,
                         ema=ema, output=output / 'evaluation.json')
            elif status == 'completed':
                validate(model, validation, codec, cfg, step=completed, samples=cfg['final_samples'],
                         ema=ema, output=output / 'evaluation/final_last_k4.json')
                selected = torch.load(output / 'checkpoints/best.ckpt', map_location='cpu', weights_only=False, mmap=True)
                restore_model(model, selected['model'])
                validate(model, validation, codec, cfg, step=selected['optimizer_step'], samples=cfg['final_samples'],
                         output=output / 'evaluation/final_best_k4.json')
        except BaseException as exc:
            status, error = 'failed', f'{type(exc).__name__}: {exc}'
            if not isinstance(exc, torch.OutOfMemoryError):
                save_recovery(checkpoint, model, optimizer, scheduler, ema, step=completed,
                              contract=contract, best=best, root=root)
            raise
        finally:
            write_json(output / 'result.json', {'status': status, 'arm': arm, 'optimizer_step': completed,
                        'best_foreground_mae': best, 'pause_reasons': stopping, 'error': error, 'updated_utc': now()})
            write_json(output / 'ready.json', {'active': False, 'pid': os.getpid(), 'status': status})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['profile', 'smoke', 'verify', 'train'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm', required=True, choices=['bifm', 'symm'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = config(args.config)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this training plan')
    if args.stage == 'profile':
        profile(cfg, args.arm)
    elif args.stage == 'verify':
        verify_validation(cfg, args.arm)
    else:
        train(cfg, args.arm, smoke=args.stage == 'smoke', resume=args.resume)


if __name__ == '__main__':
    main()
