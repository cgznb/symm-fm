from __future__ import annotations

from research_release import path as _release_path, load_yaml as _release_yaml, ssh_host

import argparse
import ast
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .first_post_world_data import (
    GIB, IMAGE_SHAPE, LATENT_SHAPE, SCHEMA, case_name, config, disk_gate,
    identity, load_codec, now, read_json, resample_ser, validate_pairs, write_json,
)

WORKSPACE = Path(_release_path('@workspace'))
DATASETS = Path(_release_path("@data"))
OLD_BUNDLE = WORKSPACE / "MAM/MOTFM-tumor-first/runs/registered_strict_a_z96_target_oracle/data_bundle.qingyuan_20260804.val102_high_transition"
REMOTE_ROOT = _release_path('@remote_data/02_preprocessed_nifti')


def ssh_arguments():
    from research_release import ssh_arguments as configured_ssh
    return configured_ssh()


def remote_python(code, payload, timeout=300):
    code = code.replace("__RAW_DICOM_ROOT__", str(Path(_release_path("@remote_data")) / "01_raw_dicom"))
    code = code.replace("__NIFTI_ROOT__", REMOTE_ROOT)
    result = subprocess.run([*ssh_arguments(), ssh_host(), "python3", "-c", shlex.quote(code)],
                            input=json.dumps(payload), text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Remote read failed ({result.returncode}): {result.stderr[-1500:]}")
    return json.loads(result.stdout)


REMOTE_DATES = r'''
import json,re,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pydicom
requests=json.load(sys.stdin)
root=Path('__RAW_DICOM_ROOT__')
def inspect(request):
    pid=request['patient_id']
    if not re.fullmatch(r'(ISPY2|ACRIN-6698)-[0-9]+',pid):raise ValueError('Invalid patient folder')
    collection='ACRIN-6698' if pid.startswith('ACRIN') else 'ISPY2'
    patient=root/collection/pid
    records=[]
    if not patient.is_dir():return records
    wanted=set(request['study_uids'])
    series_wanted=set(request['series_uids'])
    for study in sorted(patient.iterdir()):
        if not study.is_dir():continue
        for series in sorted(study.iterdir()):
            if not series.is_dir():continue
            files=(x for x in series.iterdir() if x.is_file() and not x.name.startswith('.'))
            for path in files:
                try:
                    d=pydicom.dcmread(path,stop_before_pixels=True,specific_tags=['StudyInstanceUID','SeriesInstanceUID','StudyDate'])
                    uid=str(getattr(d,'StudyInstanceUID',''));sid=str(getattr(d,'SeriesInstanceUID',''));date=str(getattr(d,'StudyDate',''))
                    if uid and date and (uid in wanted or sid in series_wanted):
                        records.append({'patient_id':pid,'study_uid':uid,'series_uid':sid,'study_date':date,
                                        'source':'original_dicom_exact_uid','header_path':str(path)})
                    break
                except (OSError,pydicom.errors.InvalidDicomError):continue
    return records
with ThreadPoolExecutor(max_workers=8) as pool:
    result=[record for rows in pool.map(inspect,requests) for record in rows]
print(json.dumps(result))
'''


def scalar(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    if isinstance(value, (float, int, np.number)):
        return str(int(value)) if float(value).is_integer() else str(float(value))
    return str(value).strip()


def clinical_sources():
    ancillary = DATASETS / "qingyuan_acrin_registered_minimal_20260804/metadata/Full-Collection-Ancillary-Patient-Information-file.xlsx"
    frame = pd.read_excel(ancillary)
    mapping = {str(r['TCIA PATIENT ID']).strip(): 'ISPY2-' + scalar(r['I-SPY 2 Research ID'])
               for r in frame.to_dict('records')}
    clinical = pd.read_excel(DATASETS / "manifest-1781750940386/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx")
    return mapping, {'ISPY2-' + scalar(r['Patient_ID']): r for r in clinical.to_dict('records')}


def known_dates():
    by_visit, by_uid = defaultdict(set), defaultdict(set)
    with (OLD_BUNDLE / 'visits.csv').open() as f:
        for r in csv.DictReader(f):
            d = datetime.strptime(r['visit_date'], '%Y-%m-%d').date().isoformat()
            by_visit[(r['patient_id'], r['visit'])].add(d)
            by_uid[r['study_instance_uid']].add(d)
    for path in (DATASETS / 'manifest-1781750940386/metadata.csv',
                 DATASETS / 'qingyuan_acrin_registered_minimal_20260804/metadata/qingyuan_visit_metadata_20260804.csv'):
        with path.open() as f:
            for raw in csv.DictReader(f):
                r = {k.strip(): v.strip() for k, v in raw.items()}
                d = datetime.strptime(r['Study Date'], '%m-%d-%Y').date().isoformat()
                by_uid[r['Study UID']].add(d)
                if 'patient_id' in r:
                    by_visit[(r['patient_id'], r['visit'])].add(d)
    return by_visit, by_uid


def metadata(cfg):
    root = Path(cfg['output_root'])
    shared = root / 'shared'
    shared.mkdir(parents=True, exist_ok=True)
    source = Path(cfg['vqgan_run'])
    manifest = read_json(source / 'data/manifest.json')
    segmentation = Path(manifest['preparation_config']['data']['tumor_crop']['segmentation_root'])
    split = read_json(source / 'data/split.json')
    mapping, clinical = clinical_sources()
    by_visit, by_uid = known_dates()
    visits, by_patient, requests = [], defaultdict(dict), defaultdict(lambda: {'study_uids': set(), 'series_uids': set()})
    canonical_folds, original_ids = {}, {}
    for r in manifest['records']:
        raw_id, visit = r['patient_id'], r['visit']
        pid = mapping.get(raw_id, raw_id)
        if canonical_folds.setdefault(pid, r['fold']) != r['fold']:
            raise ValueError('Canonical patient leakage across VQGAN folds')
        if original_ids.setdefault(pid, raw_id) != raw_id:
            raise ValueError('Multiple raw identities refer to the same participant')
        if raw_id not in split[r['fold']]:
            raise ValueError('Saved visit fold differs from patient split')
        meta = read_json(r['metadata_source'])
        uid = meta.get('study_uid', '')
        series = meta.get('source_series_uids', {}).get('dce', [])
        if isinstance(series, str):
            series = [series]
        dates = by_visit.get((pid, visit), set()) | by_uid.get(uid, set())
        crop = source / 'data/crops' / f'{case_name(r)}.npy'
        if identity(crop) != r['cache_identity']:
            raise ValueError('VQGAN crop cache identity changed')
        descriptor = {k: r[k] for k in ('patient_id', 'visit', 'shape_zyx', 'image_orientation_patient',
                                        'image_position_patient_first', 'pixel_spacing_yx_mm', 'slice_spacing_mm')}
        descriptor.update(visit_id=f'{raw_id}:{visit}', canonical_patient_id=pid, split=r['fold'],
                          image_path=str(crop), image_identity=identity(crop), metadata_path=r['metadata_source'],
                          metadata_identity=identity(r['metadata_source']), crop_geometry=r['crop']['geometry'],
                          crop_localization=r['crop']['localization'], study_uid=uid, dce_series_uids=series,
                          ser_relative_path=f"{raw_id}/{visit}/ser/{Path(meta['ser_path']).name}",
                          known_dates=sorted(dates), date_source='verified_existing_metadata')
        mask = segmentation / 'masks' / f'{case_name(r)}.nii.gz'
        descriptor.update(mask_path=str(mask), mask_identity=identity(mask))
        visits.append(descriptor)
        by_patient[pid][int(visit[1:])] = descriptor
        if not dates:
            requests[raw_id]['study_uids'].update([uid] if uid else [])
            requests[raw_id]['series_uids'].update(series)
    evidence_path = shared / 'recovered_date_evidence.json'
    if not evidence_path.exists():
        request_rows = [{'patient_id': p, **{k: sorted(v) for k, v in d.items()}} for p, d in sorted(requests.items())]
        write_json(shared / 'date_requests.json', request_rows)
        recovered = remote_python(REMOTE_DATES, request_rows, timeout=900) if request_rows else []
        write_json(evidence_path, recovered)
    recovered = read_json(evidence_path)
    recovered_uids, recovered_series = defaultdict(set), defaultdict(set)
    for r in recovered:
        d = datetime.strptime(r['study_date'], '%Y%m%d').date().isoformat()
        recovered_uids[(r['patient_id'], r['study_uid'])].add(d)
        recovered_series[(r['patient_id'], r['series_uid'])].add(d)
    for r in visits:
        dates = set(r.pop('known_dates'))
        extra = recovered_uids.get((r['patient_id'], r['study_uid']), set()).copy()
        for sid in r['dce_series_uids']:
            extra |= recovered_series.get((r['patient_id'], sid), set())
        if extra:
            dates |= extra
            r['date_source'] = 'original_dicom_exact_uid'
        r['visit_date'] = next(iter(dates)) if len(dates) == 1 else None
        r['date_status'] = 'verified' if len(dates) == 1 else ('conflict' if dates else 'missing')
    pairs, exclusions, candidate_counts = [], [], Counter()
    for pid, vs in sorted(by_patient.items()):
        for i in range(3):
            for j in range(i + 1, 4):
                if not all(k in vs for k in range(i, j + 1)):
                    continue
                a, b = vs[i], vs[j]
                candidate_counts[a['split']] += 1
                pair_id = f'{pid}:T{i}->T{j}'
                reasons = []
                if pid not in clinical:
                    reasons.append('clinical_row_absent')
                chain = [vs[k] for k in range(i, j + 1)]
                if any(v['date_status'] != 'verified' for v in chain):
                    reasons.append('date_missing_or_conflicting')
                elif any(x['visit_date'] >= y['visit_date'] for x, y in zip(chain, chain[1:])):
                    reasons.append('nonpositive_adjacent_interval')
                if reasons:
                    exclusions.append({'pair_id': pair_id, 'split': a['split'], 'reasons': reasons})
                    continue
                c = clinical[pid]
                baseline = {dest: scalar(c[key]) for key, dest in [('HR', 'hr_status'), ('HER2', 'her2_status'),
                            ('MP', 'mammaprint'), ('menopausal_status', 'menopausal_status')]}
                baseline['age'] = None if pd.isna(c['Age_at_Screening']) else float(c['Age_at_Screening'])
                text_values = [('age at screening', scalar(c['Age_at_Screening'])), ('HR', baseline['hr_status']),
                               ('HER2', baseline['her2_status']), ('MP', baseline['mammaprint']),
                               ('menopausal status', baseline['menopausal_status'])]
                arm = scalar(c['Arm'])
                pairs.append({'pair_id': pair_id, 'patient_id': pid, 'split': a['split'],
                              'earlier_visit_id': a['visit_id'], 'later_visit_id': b['visit_id'],
                              'earlier_stage': f'T{i}', 'later_stage': f'T{j}',
                              'delta_days': (datetime.fromisoformat(b['visit_date']) - datetime.fromisoformat(a['visit_date'])).days,
                              'interval_missing': False, 'interval_source': 'verified_relative_dicom_study_date',
                              'baseline_clinical': baseline, 'treatment': {'treatment_arm': arm},
                              'clinical_text': '; '.join(f'{key} {value if value is not None else "unknown"}' for key, value in text_values),
                              'treatment_text': f'treatment arm {arm if arm is not None else "unknown"}',
                              'adjacent_visit_chain': [v['visit_id'] for v in chain]})
    if len(visits) != 3777 or Counter(r['split'] for r in visits) != {'train': 3397, 'val': 380}:
        raise ValueError('The approved VQGAN cohort changed')
    validate_pairs(pairs, {r['visit_id']: r for r in visits})
    bundle = {'schema': SCHEMA, 'ready': False, 'created_utc': now(), 'registered': False,
              'phase_index': 1, 'image_shape_zyx': IMAGE_SHAPE, 'latent_shape_czyx': LATENT_SHAPE,
              'vqgan_run': str(source), 'split': split, 'visits': visits, 'pairs': pairs,
              'normalization': {}, 'exclusions': exclusions}
    write_json(shared / 'metadata_bundle.json', bundle)
    write_json(shared / 'metadata_audit.json', {
        'candidate_pairs': dict(candidate_counts), 'accepted_pairs': dict(Counter(p['split'] for p in pairs)),
        'accepted_patients': {s: len({p['patient_id'] for p in pairs if p['split'] == s}) for s in ('train', 'val')},
        'date_status': dict(Counter(r['date_status'] for r in visits)),
        'excluded_reasons': dict(Counter(reason for p in exclusions for reason in p['reasons'])),
        'canonical_patient_count': len(canonical_folds), 'cross_split_patients': 0})
    print(json.dumps(read_json(shared / 'metadata_audit.json')), flush=True)


def transfer(cfg):
    root = Path(cfg['output_root'])
    bundle = read_json(root / 'shared/metadata_bundle.json')
    sources = {p['earlier_visit_id'] for p in bundle['pairs']}
    selected = [r for r in bundle['visits'] if r['visit_id'] in sources]
    files = [r['ser_relative_path'] for r in selected]
    code = "import json,sys; from pathlib import Path; root=Path('__NIFTI_ROOT__'); paths=json.load(sys.stdin); print(json.dumps([{'path':p,'size_bytes':(root/p).stat().st_size,'mtime_ns':(root/p).stat().st_mtime_ns} for p in paths]))"
    inventory = remote_python(code, files)
    destination = root / 'shared/native_ser'
    destination.mkdir(parents=True, exist_ok=True)
    required = sum(r['size_bytes'] for r in inventory if not (destination / r['path']).exists())
    disk_gate(root, required=required, reserve_gib=cfg['disk_reserve_gib'])
    command = ['rsync', '-rt', '--partial', '--from0', '--files-from=-', '--protect-args', '-e',
               shlex.join(ssh_arguments()), f'{ssh_host()}:{REMOTE_ROOT}/', str(destination) + '/']
    subprocess.run(command, input=b''.join(p.encode() + b'\0' for p in files), check=True)
    for r in inventory:
        if (destination / r['path']).stat().st_size != r['size_bytes']:
            raise ValueError('SER transfer size differs')
    write_json(root / 'shared/ser_transfer.json', {'status': 'complete', 'files': inventory,
                                                 'total_bytes': sum(r['size_bytes'] for r in inventory)})
    print(json.dumps({'stage': 'ser_transfer_complete', 'visits': len(inventory), 'bytes': sum(r['size_bytes'] for r in inventory)}), flush=True)


def prepare_ser_one(args):
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    root, record = args
    path = root / 'shared/ser_crops' / f'{case_name(record)}.npz'
    report = path.with_suffix('.json')
    source = root / 'shared/native_ser' / record['ser_relative_path']
    deps = {'source': identity(source), 'image': identity(record['image_path']),
            'metadata': identity(record['metadata_path']), 'geometry': record['crop_geometry']}
    if report.exists() and path.exists():
        previous = read_json(report)
        if previous['dependencies'] != deps or previous['cache_identity'] != identity(path):
            raise ValueError('SER preparation dependencies changed')
        return previous
    disk_gate(root)
    array = resample_ser(source, record)
    first = np.load(record['image_path'], mmap_mode='r', allow_pickle=False)
    foreground = first != 0
    if not foreground.any():
        raise ValueError('First-post foreground is empty')
    array[~foreground] = 0
    cached = array.astype(np.float16)
    if not np.isfinite(cached).all():
        raise ValueError('SER exceeds float16 range')
    values = cached[foreground].astype(np.float64)
    moments = {'count': len(values), 'sum': float(values.sum()), 'sum_squares': float(np.dot(values, values))}
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, ser=cached)
    with np.load(tmp, allow_pickle=False) as f:
        if not np.array_equal(f['ser'], cached):
            raise ValueError('Compressed SER cache roundtrip differs')
    tmp.replace(path)
    result = {'visit_id': record['visit_id'], 'dependencies': deps, 'moments': moments,
              'cache_path': str(path.relative_to(root)), 'cache_identity': identity(path)}
    write_json(report, result)
    return result


def crops(cfg):
    root = Path(cfg['output_root'])
    (root / 'shared/ser_crops').mkdir(parents=True, exist_ok=True)
    bundle = read_json(root / 'shared/metadata_bundle.json')
    sources = {p['earlier_visit_id'] for p in bundle['pairs']}
    selected = [r for r in bundle['visits'] if r['visit_id'] in sources]
    pilot = sorted(selected, key=lambda r: (r['split'], r['visit'], r['patient_id']))
    pilot = [pilot[i] for i in np.linspace(0, len(pilot) - 1, min(24, len(pilot)), dtype=int)]
    with ThreadPoolExecutor(max_workers=cfg['preparation_workers']) as pool:
        reports = list(pool.map(prepare_ser_one, [(root, r) for r in pilot]))
        projected = int(np.mean([r['cache_identity']['size_bytes'] for r in reports]) * len(selected) * 1.25)
        # Reserve shared latents, both recovery checkpoints and concurrent atomic saves.
        existing = sum(p.stat().st_size for p in (root / 'shared/ser_crops').glob('*.npz'))
        disk_gate(root, required=max(0, projected - existing) + 12 * GIB, reserve_gib=cfg['disk_reserve_gib'])
        write_json(root / 'shared/storage_projection.json', {'pilot_cases': len(pilot),
                   'projected_ser_cache_bytes_with_margin': projected, 'other_reserved_bytes': 12 * GIB,
                   'free_bytes': __import__('shutil').disk_usage(root).free})
        all_reports = {}
        for i, report in enumerate(pool.map(prepare_ser_one, [(root, r) for r in selected]), 1):
            all_reports[report['visit_id']] = report
            if i % 100 == 0:
                print(json.dumps({'stage': 'ser_crops', 'completed': i, 'total': len(selected)}), flush=True)
    for r in bundle['visits']:
        if r['visit_id'] in all_reports:
            report = all_reports[r['visit_id']]
            r.update(ser_crop_path=report['cache_path'], ser_crop_identity=report['cache_identity'])
    training = [all_reports[r['visit_id']]['moments'] for r in selected if r['split'] == 'train']
    n = sum(r['count'] for r in training)
    mean = sum(r['sum'] for r in training) / n
    variance = sum(r['sum_squares'] for r in training) / n - mean**2
    if variance <= 0:
        raise ValueError('Degenerate training SER statistics')
    norm = read_json(Path(cfg['vqgan_run']) / 'data/normalization.json')
    bundle['normalization'].update(ser_mean=mean, ser_std=float(np.sqrt(variance)),
                                    ser_fit_unique_train_sources=len(training),
                                    image_mean=norm['mean'], image_std=norm['std'],
                                    image_fit_visits=norm['fit_visits'], fit_split='train')
    write_json(root / 'shared/crop_bundle.json', bundle)


def encode(cfg, device):
    root = Path(cfg['output_root'])
    bundle = read_json(root / 'shared/crop_bundle.json')
    source = Path(cfg['vqgan_run']) / 'checkpoints/best.ckpt'
    codec_path = root / 'shared/codec.pt'
    codec = load_codec(source)
    if not codec_path.exists():
        from dataclasses import asdict
        torch.save({'schema': SCHEMA, 'model_config': asdict(codec.config),
                    'codec_state': codec.state_dict(), 'source_checkpoint': str(source),
                    'source_identity': identity(source)}, codec_path)
    saved = torch.load(codec_path, map_location='cpu', weights_only=False, mmap=True)
    if saved['source_identity'] != identity(source):
        raise ValueError('Selected VQGAN checkpoint changed')
    del saved
    bundle['codec_identity'] = identity(codec_path)
    codec.to(device)
    destinations = root / 'shared/latents'
    destinations.mkdir(parents=True, exist_ok=True)
    endpoints = {p[k] for p in bundle['pairs'] for k in ('earlier_visit_id', 'later_visit_id')}
    selected = [r for r in bundle['visits'] if r['visit_id'] in endpoints]
    missing = sum(not (destinations / f'{case_name(r)}.npy').exists() for r in selected)
    disk_gate(root, required=missing * np.prod(LATENT_SHAPE) * 2 + 7 * GIB,
              reserve_gib=cfg['disk_reserve_gib'])
    sums, squares, count = np.zeros(8), np.zeros(8), 0
    histogram = np.zeros(65536, dtype=np.int64)
    codebook = codec.quantizer.embeddings.detach().float()
    for i, r in enumerate(selected, 1):
        path = destinations / f'{case_name(r)}.npy'
        image = np.load(r['image_path'], allow_pickle=False)
        if not path.exists():
            with torch.inference_mode():
                latent = codec.encode_continuous(torch.from_numpy(image.astype(np.float32))[None, None].to(device))
            latent = latent[0].float().cpu().numpy().astype(np.float16)
            if latent.shape != LATENT_SHAPE or not np.isfinite(latent).all():
                raise ValueError('Nonfinite or incompatible first-post latent')
            temporary = path.with_suffix('.tmp.npy')
            np.save(temporary, latent, allow_pickle=False)
            temporary.replace(path)
        else:
            latent = np.load(path, allow_pickle=False)
            if latent.shape != LATENT_SHAPE or latent.dtype != np.float16 or not np.isfinite(latent).all():
                raise ValueError('Invalid cached first-post latent')
        r.update(latent_path=str(path.relative_to(root)), latent_identity=identity(path))
        if r['split'] == 'train':
            flat = latent.reshape(8, -1).astype(np.float64)
            sums += flat.sum(1)
            squares += np.square(flat).sum(1)
            count += flat.shape[1]
            # The source crop is float16, so an exact bounded histogram suffices.
            codes = image[image != 0].view(np.uint16)
            histogram += np.bincount(codes, minlength=65536)
        if i % 100 == 0:
            print(json.dumps({'stage': 'continuous_latents', 'completed': i, 'total': len(selected)}), flush=True)
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 1e-12))
    values = np.arange(65536, dtype=np.uint16).view(np.float16)
    indices = np.flatnonzero(histogram)
    order = indices[np.argsort(values[indices].astype(np.float32))]
    cumulative = np.cumsum(histogram[order])
    quantiles = [float(values[order[np.searchsorted(cumulative, cumulative[-1] * q)]]) for q in (0.005, 0.995)]
    bundle['normalization'].update(latent_mean=mean.tolist(), latent_std=std.tolist(),
                                  latent_fit_unique_train_visits=sum(r['split'] == 'train' for r in selected),
                                  codebook_min=float(codebook.min()), codebook_max=float(codebook.max()),
                                  metric_quantiles=quantiles, metric_data_range=quantiles[1] - quantiles[0])
    bundle['visits'] = selected
    bundle['ready'] = True
    bundle['completed_utc'] = now()
    validate_pairs(bundle['pairs'], {r['visit_id']: r for r in selected})
    write_json(root / 'shared/bundle.json', bundle)
    print(json.dumps({'stage': 'data_ready', 'visits': len(selected),
                      'pairs': dict(Counter(p['split'] for p in bundle['pairs']))}), flush=True)


def finalize(cfg):
    root = Path(cfg['output_root'])
    path = root / 'shared/bundle.json'
    bundle = read_json(path)
    metadata = read_json(root / 'shared/metadata_bundle.json')
    if bundle['pairs'] != metadata['pairs'] or bundle['split'] != metadata['split']:
        raise ValueError('Final preparation cohort differs from metadata audit')
    sources = {r['visit_id']: r for r in metadata['visits']}
    for record in bundle['visits']:
        original = sources[record['visit_id']]
        for key in ('image_path', 'image_identity', 'crop_geometry', 'split', 'visit_date'):
            if record[key] != original[key]:
                raise ValueError('Visit preparation identity changed')
        for key in ('mask_path', 'mask_identity'):
            record[key] = original[key]
        if identity(record['mask_path']) != record['mask_identity']:
            raise ValueError('Prepared segmentation changed')
    validate_pairs(bundle['pairs'], {r['visit_id']: r for r in bundle['visits']})
    bundle['verified_for_world_training'] = True
    bundle['codec_source_identity'] = identity(Path(cfg['vqgan_run']) / 'checkpoints/best.ckpt')
    write_json(path, bundle)
    write_json(root / 'shared/preflight.json', {'passed': True, 'schema': SCHEMA,
        'pairs': dict(Counter(p['split'] for p in bundle['pairs'])),
        'unique_endpoints': len(bundle['visits']), 'normalization': bundle['normalization'],
        'codec_identity': bundle['codec_identity'], 'bundle_identity': identity(path), 'completed_utc': now()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['metadata', 'transfer', 'crops', 'encode', 'finalize', 'all'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = config(args.config)
    torch.set_num_threads(4)
    stages = ('metadata', 'transfer', 'crops', 'encode', 'finalize') if args.stage == 'all' else (args.stage,)
    for stage in stages:
        write_json(Path(cfg['output_root']) / 'preparation_status.json', {'status': 'running', 'stage': stage, 'pid': os.getpid(), 'updated_utc': now()})
        if stage == 'encode':
            encode(cfg, args.device)
        else:
            globals()[stage](cfg)
    write_json(Path(cfg['output_root']) / 'preparation_status.json', {'status': 'completed', 'stage': stages[-1], 'updated_utc': now()})


if __name__ == '__main__':
    main()
