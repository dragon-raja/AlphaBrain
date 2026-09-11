import copy
from pathlib import Path
import tempfile
import unittest

from scripts.dsol_paper1.analysis.analyze_matched_view_metric_rules_v1 import (
    RULES, analyze, match_reports, pair_rule_results, require_new_output,
)
from tests.dsol_paper1.helpers.landscape import fixtures


def fake_rule_results():
    results = []
    for model in range(2):
        rows = []
        for state in range(8):
            for index, rule in enumerate(RULES):
                success = .4 + model*.1 + index*(.01 + model*.01)
                rows.append({'pair_key':f's-{state}', 'task_id':f't-{state}', 'source_group':f'd-{state}',
                    'split':'development', 'rule':rule, 'success':success,
                    'gain_vs_canonical_pp':index*(1+model), 'gain_vs_uniform_pp':(index-1)*(1+model)})
        results.append({'state_rule_metrics':rows})
    return results


class MatchedMetricRulesWrapperTest(unittest.TestCase):
    def test_paired_training_interaction_not_raw_success_difference(self):
        canonical, broad = fake_rule_results()
        summary, rows = pair_rule_results(canonical, broad)
        rule = RULES[3]
        values = summary['rules'][rule]
        self.assertAlmostEqual(values['delta_success_pp']['mean'], 13)
        self.assertAlmostEqual(values['gain_interaction_vs_canonical_pp']['mean'], 3)
        self.assertAlmostEqual(values['gain_interaction_vs_uniform_pp']['mean'], 2)
        self.assertEqual(len(rows), 56)

    def test_no_source_ci_for_one_state_per_task(self):
        summary, _ = pair_rule_results(*fake_rule_results())
        for rule in RULES:
            for metric in summary['rules'][rule].values():
                self.assertEqual(metric['ci95'], [None, None])
        self.assertIsNone(summary['source_generalization_confidence_interval'])

    def test_missing_rule_is_rejected(self):
        canonical, broad = fake_rule_results()
        broad['state_rule_metrics'].pop()
        with self.assertRaisesRegex(ValueError, 'all seven rules'):
            pair_rule_results(canonical, broad)

    def test_duplicate_or_source_mismatch_rejected(self):
        for kind in ('duplicate', 'source'):
            canonical, broad = fake_rule_results()
            if kind == 'duplicate':
                broad['state_rule_metrics'][1] = copy.deepcopy(broad['state_rule_metrics'][0])
            else:
                broad['state_rule_metrics'][0]['source_group'] = 'other'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                pair_rule_results(canonical, broad)

    def test_frozen_core_rule_analysis_roundtrip(self):
        c,b,selection = fixtures()
        c,b,_ = match_reports(c,b,selection)
        c_result, b_result = analyze(c), analyze(b)
        summary, rows = pair_rule_results(c_result,b_result)
        self.assertEqual(summary['state_count'], 8)
        self.assertTrue(all(abs(row['delta_success_pp']) < 1e-10 for row in rows))
        self.assertEqual(c_result['summary']['rule_order'], list(RULES))

    def test_existing_output_even_empty_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'refuses overwrite'):
                require_new_output(Path(directory))
            require_new_output(Path(directory) / 'not-yet-created')


if __name__ == '__main__':
    unittest.main()
