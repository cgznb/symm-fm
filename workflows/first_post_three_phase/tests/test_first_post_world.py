import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from mewm_ispy2.first_post_world_data import (
    PairDataset, PatientBalancedBatchSampler, native_zyx_image, public, validate_pairs, write_json,
)


def records():
    return [{'patient_id': 'p1'}, {'patient_id': 'p1'}, {'patient_id': 'p2'}]


def flattened(sampler):
    return [i for batch in sampler for i in batch]


def test_effective_examples_do_not_depend_on_microbatch():
    a = PatientBalancedBatchSampler(records(), 1, 4, 0, 7)
    b = PatientBalancedBatchSampler(records(), 4, 4, 0, 7)
    assert flattened(a) == flattened(b)


def test_resume_does_not_replay_or_skip_prefetched_examples():
    full = flattened(PatientBalancedBatchSampler(records(), 2, 8, 0, 11))
    resumed = flattened(PatientBalancedBatchSampler(records(), 2, 8, 5, 11))
    assert resumed == full[5 * 8:]


def test_patient_balancing_is_not_pair_balancing():
    sample = flattened(PatientBalancedBatchSampler(records(), 4, 4, 0, 2000))
    count_second_patient = sum(i == 2 for i in sample)
    assert 0.47 < count_second_patient / len(sample) < 0.53


def pair_fixture():
    visits = {'p:T0': {'canonical_patient_id': 'p', 'split': 'train', 'visit': 'T0'},
              'p:T1': {'canonical_patient_id': 'p', 'split': 'train', 'visit': 'T1'}}
    pair = {'pair_id': 'p:T0->T1', 'patient_id': 'p', 'split': 'train',
            'earlier_stage': 'T0', 'later_stage': 'T1', 'delta_days': 21,
            'earlier_visit_id': 'p:T0', 'later_visit_id': 'p:T1',
            'baseline_clinical': {'age': None}, 'treatment': {'treatment_arm': None}}
    return pair, visits


def test_patient_split_is_enforced_at_both_endpoints():
    pair, visits = pair_fixture()
    validate_pairs([pair], visits)
    visits['p:T1']['split'] = 'val'
    with pytest.raises(ValueError, match='endpoint'):
        validate_pairs([pair], visits)


@pytest.mark.parametrize('field', ['pcr', 'target_mask', 'target_volume'])
def test_future_conditions_are_rejected(field):
    pair, visits = pair_fixture()
    pair['baseline_clinical'][field] = 1
    with pytest.raises(ValueError, match='conditions'):
        validate_pairs([pair], visits)


@pytest.mark.parametrize('interval', [0, -1, 21.5])
def test_unverified_or_nonpositive_intervals_are_rejected(interval):
    pair, visits = pair_fixture()
    pair['delta_days'] = interval
    with pytest.raises(ValueError, match='temporal'):
        validate_pairs([pair], visits)


@pytest.mark.parametrize('arm', ['symm', 'bifm'])
def test_shared_latents_roundtrip_through_each_original_scale(arm):
    dataset = PairDataset.__new__(PairDataset)
    dataset.arm = arm
    dataset.stats = {'codebook_min': -41.3, 'codebook_max': 61.0,
                     'latent_mean': list(range(8)), 'latent_std': [2.0] * 8}
    raw = torch.randn(8, 2, 3, 4)
    torch.testing.assert_close(dataset.denormalize(dataset.normalize(raw)), raw, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dataset.denormalize(dataset.normalize(raw)[None])[0], raw, atol=1e-5, rtol=1e-5)


def test_native_zyx_axes_restore_physical_points():
    record = {'shape_zyx': [2, 3, 4], 'image_orientation_patient': [0, 1, 0, 0, 0, 1],
              'image_position_patient_first': [10, 20, 30], 'pixel_spacing_yx_mm': [2, 3],
              'slice_spacing_mm': 4}
    image = native_zyx_image(np.arange(24).reshape(2, 3, 4), record)
    assert image.GetSize() == (4, 3, 2)
    assert image.TransformIndexToPhysicalPoint((1, 1, 1)) == (14, 23, 32)


def test_registered_cache_is_not_accepted(tmp_path):
    write_json(tmp_path / 'shared/bundle.json', {'schema': 'registered_t0', 'ready': True})
    with pytest.raises(ValueError, match='preparation'):
        PairDataset(tmp_path, 'bifm', 'train')


def test_reports_do_not_copy_legacy_checksums():
    value = {'ordinary': 42, 'schema_fingerprint': 'a' * 64, 'nested': {'sha256': 'b' * 64}}
    assert public(value) == {'ordinary': 42, 'nested': {}}


def test_bifm_keeps_pure_noise_path():
    from mewm_ispy2.ispy2_biflow_training import build_ispy2_biflow_batch
    target = torch.full((1, 1, 8, 2, 2, 2), 3.)
    noise = torch.full_like(target, -2.)
    initial = build_ispy2_biflow_batch(target, noise=noise, flow_time=torch.zeros(1))
    endpoint = build_ispy2_biflow_batch(target, noise=noise, flow_time=torch.ones(1))
    assert torch.equal(initial.flow_state, noise)
    assert torch.equal(endpoint.flow_state, target)
    assert torch.equal(initial.target_velocity, target - noise)


def test_eta_counts_full_validation_once_and_includes_final_four_sample_evaluations():
    from mewm_ispy2.first_post_world_training import remaining_runtime
    cfg = {'max_optimizer_steps': 100000, 'full_validation_interval': 10000,
           'light_validation_interval': 1000, 'light_validation_pairs': 64, 'final_samples': 4}
    result = remaining_runtime(cfg, 10000, 4.0, 515, 2.0)
    expected_validation_samples = 9 * 515 + 81 * 64 + 2 * 4 * 515
    assert result['training_only_eta_hours'] == 100.0
    assert result['estimated_eta_hours_including_validation'] == (90000 * 4 + expected_validation_samples * 2) / 3600
