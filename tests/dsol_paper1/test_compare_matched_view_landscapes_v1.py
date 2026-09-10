import copy
from pathlib import Path
import unittest

import numpy as np

from scripts.dsol_paper1.compare_matched_view_landscapes_v1 import (
    BANK_FILE_SHA256, BANK_MANIFEST_SHA256, CHECKPOINTS, array_contract,
    catalog_bank, compare_metrics, match_reports, paired_noise_se,
    read_json, summarize_fixed_states,
)


def fixtures():
    root = Path(__file__).resolve().parents[2] / 'configs/dsol_paper1'
    catalog = read_json(root / 'libero_view_catalog_v2_m1.json')
    rules = read_json(root / 'libero_view_catalog_v2_m1_rules.json')
    selected = []
    for i in range(8):
        selected.append({'pair_key': f'state-{i}', 'task_id': f'task-{i}', 'source_group': f'source-{i}',
            'asset_source_pair_key': f'asset-{i}', 'source_state_index': i,
            'environment_seed': i+100, 'split': 'development', 'construction_spec_sha256': f'construction-{i}',
            'demo_name': f'demo_{i}', 'static_assets': {'physics_state_sha256': f'physics-{i}',
            'policy_inputs_sha256': f'images-{i}', 'render_receipt_sha256': f'render-{i}', 'visibility_scan_sha256': f'visibility-{i}'}})
    reports = []
    for label in ('canonical', 'broad'):
        poses, _ = catalog_bank(copy.deepcopy(catalog), rules, 'canonical' if label == 'canonical' else 'broad64')
        states = copy.deepcopy(selected)
        for i, state in enumerate(states):
            state['accel_ensemble_seeds'] = [i*10+j for j in range(8)]
            state['accel_action_horizon'] = 10
        success = np.zeros((8, 97, 32), dtype=np.int8)
        for j in range(97):
            success[:, j, :j % 33] = 1
        accel = np.broadcast_to(np.arange(97)[None, :, None], (8, 97, 8)).copy().astype(float)
        reports.append({'states': states, 'poses': poses, 'state_keys': [s['pair_key'] for s in states],
            'candidate_ids': [p['pose_id'] for p in poses], 'success': success, 'accel': accel,
            'visibility': np.zeros((8, 97)), 'provenance': {'checkpoint_sha256': CHECKPOINTS[label],
                'dense': {'bank_manifest_sha256': BANK_MANIFEST_SHA256, 'bank_file_sha256': BANK_FILE_SHA256}}})
    return reports[0], reports[1], {'states': selected, 'outcomes_or_view_metrics_used_in_selection': False}


class CompareMatchedLandscapesTest(unittest.TestCase):
    def test_valid_join_allows_model_specific_support_fields(self):
        canonical, broad, selection = fixtures()
        c, b, states = match_reports(canonical, broad, selection)
        self.assertEqual(len(states), 8)
        self.assertTrue(c['poses'][0]['is_model_post_training_support'])
        self.assertFalse(b['poses'][0]['is_model_post_training_support'])

    def test_broad_can_have_extra_states_but_canonical_cannot(self):
        canonical, broad, selection = fixtures()
        for report in (canonical, broad):
            extra = copy.deepcopy(report['states'][0]); extra['pair_key'] = 'extra'
            report['states'].append(extra); report['state_keys'].append('extra')
            for field in ('accel', 'success', 'visibility'):
                report[field] = np.concatenate([report[field], report[field][:1]], axis=0)
        array_contract(broad, 'broad')
        with self.assertRaises(ValueError):
            array_contract(canonical, 'canonical')

    def test_missing_outcomes_not_zero_filled(self):
        canonical, _, _ = fixtures()
        canonical['success'][0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'Missing/nonbinary'):
            array_contract(canonical, 'canonical')

    def test_rejects_duplicate_candidate_or_state(self):
        for field in ('states', 'poses'):
            canonical, _, _ = fixtures()
            canonical[field][1] = copy.deepcopy(canonical[field][0])
            with self.subTest(field=field), self.assertRaises(ValueError):
                array_contract(canonical, 'canonical')

    def test_rejects_changed_member_seed_order(self):
        canonical, broad, selection = fixtures()
        broad['states'][0]['accel_ensemble_seeds'].reverse()
        with self.assertRaisesRegex(ValueError, 'member seeds/order'):
            match_reports(canonical, broad, selection)

    def test_rejects_physics_or_image_mismatch(self):
        for field in ('physics_state_sha256', 'policy_inputs_sha256'):
            canonical, broad, selection = fixtures()
            broad['states'][0]['static_assets'][field] = 'wrong'
            with self.subTest(field=field), self.assertRaises(ValueError):
                match_reports(canonical, broad, selection)

    def test_rejects_geometry_and_noise_mismatch(self):
        canonical, broad, selection = fixtures()
        broad['poses'][1]['radius_scale'] += .01
        with self.assertRaisesRegex(ValueError, 'geometry'):
            match_reports(canonical, broad, selection)
        canonical, broad, selection = fixtures()
        broad['provenance']['dense']['bank_file_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'bank O'):
            match_reports(canonical, broad, selection)

    def test_rejects_same_checkpoint(self):
        canonical, broad, selection = fixtures()
        broad['provenance']['checkpoint_sha256'] = CHECKPOINTS['canonical']
        with self.assertRaises(ValueError):
            match_reports(canonical, broad, selection)

    def test_metrics_are_paired_not_pooled(self):
        canonical, broad, selection = fixtures()
        broad['success'][:] = canonical['success']
        c, b, _ = match_reports(canonical, broad, selection)
        summary, states, cells, arrays = compare_metrics(c, b)
        self.assertEqual(len(cells), 8*97)
        self.assertTrue(np.all(arrays['success_difference_pp'] == 0))
        self.assertTrue(np.all(arrays['paired_noise_conditional_se_pp'] == 0))
        self.assertIsNone(summary['source_generalization_confidence_interval'])
        self.assertEqual(states[0]['cross_model_success_rank_rho'], 1)

    def test_crossfit_evaluates_opposite_noise_half(self):
        canonical, broad, _ = fixtures()
        for report in (canonical, broad):
            report['success'][:] = 0
            report['success'][:, 0, :8] = 1
            report['success'][:, 1, :16] = 1
            report['success'][:, 2, 16:] = 1
        summary, _, _, _ = compare_metrics(canonical, broad)
        self.assertEqual(summary['rule_summary']['canonical']['empirical_max_success_diagnostic']['mean'], .5)
        self.assertEqual(summary['rule_summary']['canonical']['noise_crossfit_search_success']['mean'], 0)

    def test_conditional_se_preserves_noise_pairing(self):
        left = np.array([0, 1] * 16)[None, None]
        np.testing.assert_array_equal(paired_noise_se(left, left), 0)
        self.assertGreater(paired_noise_se(left, 1-left)[0, 0], 0)

    def test_no_degenerate_source_confidence_interval(self):
        summary = summarize_fixed_states([.5]*8)
        self.assertEqual(summary['mean'], .5)
        self.assertIsNone(summary['confidence_interval'])
        missing = summarize_fixed_states([float('nan')]*8)
        self.assertEqual(missing['finite_states'], 0)
        self.assertIsNone(missing['mean'])


if __name__ == '__main__':
    unittest.main()
