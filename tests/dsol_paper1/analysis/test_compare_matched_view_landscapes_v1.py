from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
import copy
from pathlib import Path
import unittest

import numpy as np

from scripts.dsol_paper1.analysis.compare_matched_view_landscapes_v1 import BANK_FILE_SHA256, BANK_MANIFEST_SHA256, CHECKPOINTS, array_contract, catalog_bank, compare_metrics, match_reports, paired_noise_se, read_json, summarize_fixed_states


from tests.dsol_paper1.helpers.landscape import fixtures


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
