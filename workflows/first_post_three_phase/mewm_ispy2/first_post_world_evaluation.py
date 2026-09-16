from __future__ import annotations

import math
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .first_post_world_data import IMAGE_SHAPE, collate, identity, now, read_json, reference_image, write_json


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


@torch.inference_mode()
def decode(codec, raw_latent):
    with torch.autocast('cuda', enabled=False):
        quantized, _ = codec.quantizer(raw_latent.float())
        image = codec.decode(quantized.float())
    if tuple(image.shape[1:]) != (1, *IMAGE_SHAPE) or not torch.isfinite(image).all():
        raise FloatingPointError('Invalid decoded first-post image')
    return image.float()


def target_mask(record):
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    if identity(record['mask_path']) != record['mask_identity']:
        raise ValueError('Evaluation localization mask changed')
    source = sitk.ReadImage(record['mask_path'], sitk.sitkUInt8)
    result = sitk.Resample(source, reference_image(record['crop_geometry']),
                          sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return torch.from_numpy(sitk.GetArrayFromImage(result).astype(np.float32))[None, None]


def metrics(prediction, target, mask, data_range):
    from ispy2_symmflow.evaluation.metrics import evaluate_prediction
    result = evaluate_prediction(prediction, target, data_range=data_range, foreground_mask=target != 0,
                                 tumor_mask=mask, mask_semantics_verified=True)
    selected = {}
    for name in ('mae', 'foreground_mae', 'tumor_mae', 'psnr', 'ssim'):
        value = float(result[name].reshape(-1)[0])
        selected[name] = value if math.isfinite(value) else None
    return selected


def aggregate(records):
    def summary(rows):
        output = {}
        for metric in ('mae', 'foreground_mae', 'tumor_mae', 'psnr', 'ssim'):
            by_patient = defaultdict(list)
            for r in rows:
                if r[metric] is not None:
                    by_patient[r['patient_id']].append(r[metric])
            output[metric] = float(np.mean([np.mean(v) for v in by_patient.values()])) if by_patient else None
            output[metric + '_patients'] = len(by_patient)
        return output
    by_transition = defaultdict(list)
    for r in records:
        by_transition[r['transition']].append(r)
    return {'patient_macro': summary(records),
            'transitions': {k: summary(v) for k, v in sorted(by_transition.items())},
            'pairs': len(records), 'patients': len({r['patient_id'] for r in records})}


@torch.inference_mode()
def validate(model, dataset, codec, cfg, *, step, indices=None, samples=1, ema=None, output=None, reference=True):
    started = time.perf_counter()
    prediction_seconds, reference_seconds, references_created = 0.0, 0.0, 0
    previous_training = model.training
    model.eval()
    codec.eval()
    records, references = [], []
    if indices is None:
        indices = list(range(len(dataset)))
    device = next(codec.parameters()).device
    context = ema.average_parameters(model) if ema is not None else nullcontext()
    with context:
        for index in indices:
            item = dataset[index]
            pair = item['record']
            visit = dataset.visits[pair['later_visit_id']]
            target = torch.from_numpy(dataset.image(pair['later_visit_id']))[None, None].to(device)
            mask = target_mask(visit).to(device)
            batch = to_device(collate([item]), device)
            per_sample = []
            for sample in range(samples):
                sample_started = time.perf_counter()
                generator = torch.Generator(device=device).manual_seed(cfg['seed'] + 1000003 * index + sample)
                noise = torch.randn(batch['target'].shape, device=device, generator=generator, dtype=torch.float32)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model.sample(batch, noise, cfg['solver_steps'])
                image = decode(codec, dataset.denormalize(prediction))
                per_sample.append(metrics(image, target, mask, dataset.stats['metric_data_range']))
                prediction_seconds += time.perf_counter() - sample_started
            row = {'pair_id': pair['pair_id'], 'patient_id': pair['patient_id'],
                   'transition': f"{pair['earlier_stage']}->{pair['later_stage']}",
                   'empty_tumor_mask': not bool(mask.any()), 'samples': per_sample}
            for key in per_sample[0]:
                observed = [r[key] for r in per_sample if r[key] is not None]
                row[key] = float(np.mean(observed)) if observed else None
            records.append(row)
            if reference:
                reference_started = time.perf_counter()
                cache = dataset.root / 'shared/evaluation_reference' / f'{index:05d}.json'
                if cache.exists():
                    ref = read_json(cache)
                    if ref['pair_id'] != pair['pair_id'] or ref['codec_identity'] != dataset.bundle['codec_identity']:
                        raise ValueError('Evaluation reference cache differs')
                else:
                    copied = torch.from_numpy(dataset.image(pair['earlier_visit_id']))[None, None].to(device)
                    reconstructed = decode(codec, dataset.raw_latent(pair['later_visit_id'])[None].to(device))
                    ref = {'pair_id': pair['pair_id'], 'patient_id': pair['patient_id'], 'transition': row['transition'],
                           'codec_identity': dataset.bundle['codec_identity'],
                           'copy_source': metrics(copied, target, mask, dataset.stats['metric_data_range']),
                           'vqgan_reconstruction': metrics(reconstructed, target, mask, dataset.stats['metric_data_range'])}
                    write_json(cache, ref)
                    references_created += 1
                references.append(ref)
                reference_seconds += time.perf_counter() - reference_started
            if output and len(records) % 16 == 0:
                write_json(Path(output).parent / 'evaluation_progress.json', {
                    'optimizer_step': step, 'completed_pairs': len(records), 'total_pairs': len(indices),
                    'samples': samples, 'updated_utc': now()})
    model.train(previous_training)
    elapsed = time.perf_counter() - started
    result = {'optimizer_step': step, 'samples_per_pair': samples, 'solver': 'euler',
              'solver_steps': cfg['solver_steps'], 'selection_metric': 'patient_macro.foreground_mae',
              'summary': aggregate(records), 'records': records, 'completed_utc': now(),
              'timing': {'elapsed_seconds': elapsed, 'metric_device': str(device),
                         'prediction_seconds_per_sample': prediction_seconds / (len(records) * samples),
                         'seconds_per_pair_excluding_references': (elapsed - reference_seconds) / len(records),
                         'reference_seconds': reference_seconds, 'references_created': references_created}}
    if references:
        result['references'] = {key: aggregate([{**r[key], 'patient_id': r['patient_id'],
                                                'transition': r['transition']} for r in references])
                                for key in ('copy_source', 'vqgan_reconstruction')}
    if output:
        write_json(output, result)
    return result
