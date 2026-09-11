import unittest

import numpy as np

from scripts.dsol_paper1.analysis.analyze_view_metric_rules_v1 import RULES, analyze, difficulty_from_selection_half, fixed_rule_weights, rule_outcomes, source_bootstrap, top10_stability
from tests.dsol_paper1.helpers.landscape import fixtures


class ViewMetricRulesTest(unittest.TestCase):
    def setUp(self):
        self.ids = ['canonical'] + [f'broad_{i:03d}' for i in range(96)]
        self.accel = np.arange(97, dtype=float)
        self.visibility = np.arange(97, dtype=float)

    def test_seven_weights_and_including_canonical(self):
        weights, selected = fixed_rule_weights(self.accel, self.visibility, self.ids)
        self.assertEqual(tuple(weights), RULES)
        for vector in weights.values():
            self.assertAlmostEqual(vector.sum(), 1)
        self.assertIn(0, selected['accel_top10_uniform'])
        self.assertEqual(len(selected['accel_top10_uniform']), 10)

    def test_historical_mixed_selection(self):
        _, selected = fixed_rule_weights(self.accel, self.visibility, self.ids)
        self.assertEqual(selected['min_accel'][0], 0)
        self.assertEqual(selected['max_visibility'][0], 96)
        self.assertEqual(selected['accel_top10_max_visibility'][0], 9)
        self.assertEqual(selected['visibility_top10_min_accel'][0], 87)

    def test_ties_use_historical_ids_not_canonical_first(self):
        _, selected = fixed_rule_weights(np.zeros(97), np.zeros(97), self.ids)
        self.assertEqual(selected['canonical'][0], 0)
        self.assertEqual(selected['min_accel'][0], 1)
        self.assertEqual(selected['max_visibility'][0], 1)
        self.assertNotIn(0, selected['accel_top10_uniform'])

    def test_mixed_visibility_tie_prefers_lower_accel_before_id(self):
        a = np.full(97, 100.0)
        a[0], a[1] = 0.1, 0.2
        v = np.zeros(97); v[0] = v[1] = 1
        _, selected = fixed_rule_weights(a, v, self.ids)
        self.assertEqual(selected['accel_top10_max_visibility'][0], 0)

    def test_uniform_rule_is_expected_success_not_new_random_draw(self):
        weights, _ = fixed_rule_weights(self.accel, self.visibility, self.ids)
        outcomes = np.zeros((97, 32), dtype=np.int8); outcomes[0] = 1
        result = rule_outcomes(weights, outcomes)
        np.testing.assert_allclose(result['uniform97'], 1/97)
        np.testing.assert_allclose(result['accel_top10_uniform'], .1)
        np.testing.assert_allclose(result['canonical'], 1)

    def test_missing_or_nonfinite_metric_rejected(self):
        a = self.accel.copy(); a[0] = np.nan
        with self.assertRaises(ValueError):
            fixed_rule_weights(a, self.visibility, self.ids)
        with self.assertRaises(ValueError):
            fixed_rule_weights(self.accel[:-1], self.visibility[:-1], self.ids[:-1])

    def test_top10_noise_stability(self):
        members = np.broadcast_to(self.accel[:, None], (97, 8)).copy()
        stable = top10_stability(members, self.ids)
        self.assertEqual(stable['top10_pairwise_jaccard'], 1)
        self.assertEqual(stable['top10_all_noise_intersection_count'], 10)
        members[:, 4:] *= -1
        unstable = top10_stability(members, self.ids)
        self.assertLess(unstable['top10_pairwise_jaccard'], 1)
        self.assertEqual(unstable['top10_all_noise_intersection_count'], 0)

    def test_difficulty_boundaries(self):
        self.assertEqual(difficulty_from_selection_half(np.ones((97,16))), 'easy')
        self.assertEqual(difficulty_from_selection_half(np.zeros((97,16))), 'hard_floor')
        y = np.zeros((97,16)); y[:, :8] = 1
        self.assertEqual(difficulty_from_selection_half(y), 'middle')

    def test_crossfit_classification_never_uses_evaluation_half(self):
        report, _, _ = fixtures()
        report['success'][:, :, :16] = 1
        report['success'][:, :, 16:] = 0
        result = analyze(report)
        strata = result['summary']['difficulty_crossfit']['strata']
        self.assertEqual(strata['easy']['rules']['uniform97']['success']['mean'], 0)
        self.assertAlmostEqual(strata['hard_floor']['rules']['uniform97']['success']['mean'], 1)
        self.assertEqual(result['summary']['difficulty_crossfit']['changed_stratum_state_count'], 8)
        self.assertEqual(len(result['difficulty_fold_metrics']), 8*2*7)

    def test_same_stratum_directions_are_merged_within_state(self):
        report, _, _ = fixtures()
        result = analyze(report)
        self.assertEqual(len(result['difficulty_state_metrics']), 8*7)
        self.assertTrue(all(row['heldout_direction_count'] == 2 for row in result['difficulty_state_metrics']))

    def test_source_and_task_equal_not_state_count_weighted(self):
        rows = [{'task_id':'a','source_group':'a1','x':1}]*9 + [
            {'task_id':'a','source_group':'a2','x':0},
            {'task_id':'b','source_group':'b1','x':0},
            {'task_id':'b','source_group':'b2','x':0}]
        summary = source_bootstrap(rows, 'x')
        self.assertEqual(summary['mean'], .25)
        self.assertEqual(summary['finite_sources'], 4)
        self.assertEqual(summary['ci_status'], 'DESCRIPTIVE_SOURCE_BOOTSTRAP')

    def test_no_false_source_ci_when_one_source_per_task(self):
        rows = [{'task_id':str(i),'source_group':str(i),'x':.5} for i in range(8)]
        result = source_bootstrap(rows, 'x')
        self.assertEqual(result['ci95'], [None,None])
        self.assertEqual(result['mean'], .5)

    def test_two_comparators_and_all_states_are_retained(self):
        report, _, _ = fixtures()
        result = analyze(report)
        self.assertEqual(len(result['state_rule_metrics']), 8*7)
        for rule in RULES:
            values = result['summary']['overall']['rules'][rule]
            self.assertIn('gain_vs_canonical_pp', values)
            self.assertIn('gain_vs_uniform_pp', values)
        self.assertEqual(len(result['diagnostic_topset_headroom']), 8)


if __name__ == '__main__':
    unittest.main()
