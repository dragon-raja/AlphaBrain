from __future__ import annotations

import unittest

import numpy as np

from analyze_coupling_gate import (
    independent_permutations,
    leave_out_profiles,
    pairwise_mse,
    state_metrics,
)


class CouplingGateAnalysisTest(unittest.TestCase):
    def test_independent_permutations_are_reproducible(self) -> None:
        first = independent_permutations(16, 6, state_id="state", draw=3)
        second = independent_permutations(16, 6, state_id="state", draw=3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (16, 6))

    def test_coupling_removes_shared_noise_from_pair_differences(self) -> None:
        signatures = np.zeros((16, 6, 6), dtype=np.float64)
        shared = np.linspace(0.0, 1.0, 6)
        candidate_effect = np.linspace(0.0, 0.3, 16)
        for candidate in range(16):
            signatures[candidate, :, 5] = shared + candidate_effect[candidate]
        coupled = pairwise_mse(signatures)
        permutations = independent_permutations(16, 6, state_id="state", draw=1)
        independent = pairwise_mse(signatures, permutations)
        self.assertLess(coupled, 1e-12)
        self.assertGreater(independent, 0.01)

    def test_pairwise_target_excludes_the_evaluated_repeat(self) -> None:
        signatures = np.asarray([[[0.0], [2.0]], [[0.0], [0.0]]])
        self.assertEqual(pairwise_mse(signatures), 4.0)

    def test_leave_out_profile_does_not_contain_selected_label(self) -> None:
        signatures = np.zeros((2, 3, 6), dtype=np.float64)
        signatures[0, :, 5] = [1.0, 0.0, 0.0]
        signatures[1, :, 5] = [0.0, 1.0, 1.0]
        indices = np.zeros((2, 1), dtype=np.int64)
        profiles = leave_out_profiles(signatures, indices)
        self.assertEqual(profiles[0, 5], 0.0)
        self.assertEqual(profiles[1, 5], 1.0)

    def test_state_metrics_detect_action_leverage(self) -> None:
        signatures = np.zeros((16, 6, 6), dtype=np.float64)
        signatures[:, :, 4] = 1.0
        signatures[1, :, 0] = 1.0
        metrics = state_metrics(signatures, state_id="state")
        self.assertTrue(metrics["action_leverage"])
        self.assertGreater(metrics["available_oracle_gain"], 0.0)


if __name__ == "__main__":
    unittest.main()
