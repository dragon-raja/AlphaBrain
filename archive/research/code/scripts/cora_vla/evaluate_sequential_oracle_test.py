from __future__ import annotations

import unittest

import numpy as np

from evaluate_sequential_oracle import candidate_pool_seed, direct_selection_key, self_consistency_index


class SequentialOracleTest(unittest.TestCase):
    def test_candidate_pool_seed_is_method_independent(self) -> None:
        self.assertEqual(
            candidate_pool_seed(41, "pair", "slipped", 7),
            candidate_pool_seed(41, "pair", "slipped", 7),
        )
        self.assertNotEqual(
            candidate_pool_seed(41, "pair", "slipped", 7),
            candidate_pool_seed(41, "pair", "attached", 7),
        )

    def test_self_consistency_chooses_medoid(self) -> None:
        candidates = np.zeros((3, 10, 7), dtype=np.float32)
        candidates[0, :2] = -1.0
        candidates[1, :2] = 0.0
        candidates[2, :2] = 0.2
        self.assertEqual(self_consistency_index(candidates, 2), 1)

    def test_direct_selection_prefers_non_regressing_progress(self) -> None:
        row = {
            "direct": {
                "regress": False,
                "success": False,
                "transport_reached": False,
                "lift_reached": True,
                "next_stage_reached": True,
                "stable_grasp_at_end": True,
                "progress_auc": 0.5,
            }
        }
        regressing = {"direct": {**row["direct"], "regress": True}}
        self.assertGreater(direct_selection_key(row), direct_selection_key(regressing))


if __name__ == "__main__":
    unittest.main()
