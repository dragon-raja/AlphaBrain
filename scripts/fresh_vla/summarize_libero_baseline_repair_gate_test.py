from __future__ import annotations

import unittest

from summarize_libero_baseline_repair_gate import baseline_gate


class BaselineRepairGateTest(unittest.TestCase):
    def test_passes_mean_and_seed_count(self) -> None:
        result = baseline_gate(
            {41: 0.46, 42: 0.31, 43: 0.15},
            minimum_cross_seed_mean=0.30,
            minimum_per_seed=0.20,
            minimum_seed_count=2,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["passing_seeds"], [41, 42])

    def test_rejects_when_only_one_seed_is_usable(self) -> None:
        result = baseline_gate(
            {41: 0.65, 42: 0.10, 43: 0.10},
            minimum_cross_seed_mean=0.30,
            minimum_per_seed=0.20,
            minimum_seed_count=2,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["conditions"]["cross_seed_attached_mean"])
        self.assertFalse(result["conditions"]["minimum_seed_count"])


if __name__ == "__main__":
    unittest.main()
