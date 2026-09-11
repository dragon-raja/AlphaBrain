import unittest

import numpy as np

from scripts.policy_response_vla.evaluate_gate_minus1 import (
    Record,
    aggregate_sources,
    feature,
    frozen_source_split,
    record_metrics,
)


class PolicyResponseGateTest(unittest.TestCase):
    def record(self) -> Record:
        candidate = np.arange(16 * 14, dtype=np.float64).reshape(16, 14)
        direct = np.zeros((16, 6), dtype=np.float64)
        response = np.arange(16 * 112, dtype=np.float64).reshape(16, 112)
        profiles = np.zeros((16, 6), dtype=np.float64)
        profiles[:, 5] = np.arange(16, dtype=np.float64) / 16.0
        return Record("example", 7, candidate, direct, response, profiles)

    def test_frozen_split_is_disjoint_and_stable(self):
        fitting, evaluation = frozen_source_split(range(23))
        self.assertEqual(len(evaluation), 5)
        self.assertFalse(set(fitting) & set(evaluation))
        self.assertEqual((fitting, evaluation), frozen_source_split(range(23)))

    def test_features_are_centered_within_state(self):
        record = self.record()
        for method in ("candidate", "direct", "response", "candidate_response"):
            np.testing.assert_allclose(feature(record, method).mean(axis=0), 0.0, atol=1e-12)

    def test_perfect_scores_select_oracle(self):
        record = self.record()
        row = record_metrics(record, record.utilities)
        self.assertEqual(row["top_hit"], 1.0)
        self.assertGreater(row["gain"], 0.0)
        self.assertEqual(row["concordant"], row["comparable"])

    def test_source_aggregation_weights_sources_equally(self):
        rows = {
            1: {"utility_gain": 1.0, "oracle_gain": 2.0, "pairwise_concordance": 0.6,
                "oracle_top_set_hit_rate": 0.2, "stable_grasp_harm_rate": 0.0},
            2: {"utility_gain": 3.0, "oracle_gain": 6.0, "pairwise_concordance": 0.8,
                "oracle_top_set_hit_rate": 0.4, "stable_grasp_harm_rate": 0.1},
        }
        result = aggregate_sources(rows)
        self.assertAlmostEqual(result["utility_gain"], 2.0)
        self.assertAlmostEqual(result["oracle_gain_recovered"], 0.5)
        self.assertAlmostEqual(result["pairwise_concordance"], 0.7)


if __name__ == "__main__":
    unittest.main()
