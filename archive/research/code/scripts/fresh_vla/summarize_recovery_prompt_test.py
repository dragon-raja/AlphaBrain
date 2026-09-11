import unittest

from summarize_recovery_prompt import stable_positive


class RecoveryPromptSummaryTest(unittest.TestCase):
    def test_stable_positive_requires_effect_size_and_direction(self):
        comparison = {
            "candidate_minus_baseline": {"mean": 0.25, "bootstrap_95_low": 0.0},
            "seed_deltas": {"41": 0.1, "42": 0.2, "43": 0.3},
        }
        self.assertTrue(stable_positive(comparison, 0.2))
        comparison["seed_deltas"]["43"] = -0.1
        self.assertFalse(stable_positive(comparison, 0.2))


if __name__ == "__main__":
    unittest.main()
