from __future__ import annotations

import unittest

import numpy as np
import torch

from train_and_evaluate_rankers import (
    CandidateCritic,
    StateRecord,
    calibrate_threshold,
    calibration_split,
    utility_tensor,
)


class RankerTest(unittest.TestCase):
    def test_profile_head_is_monotone(self) -> None:
        model = CandidateCritic("ccv_profile").eval()
        with torch.no_grad():
            profile, score = model(torch.zeros(2, 4104), torch.zeros(2, 16, 14))
        self.assertEqual(tuple(profile.shape), (2, 16, 6))
        self.assertEqual(tuple(score.shape), (2, 16))
        self.assertTrue(bool(torch.all(profile[..., 1:4] <= profile[..., :3])))

    def test_utility_tensor_matches_numpy_definition(self) -> None:
        profile = torch.tensor([[[1.0, 0.5, 0.2, 0.1, 1.0, 0.3]]])
        value = utility_tensor(profile).item()
        from ccv import scalar_viability_utility

        self.assertAlmostEqual(value, scalar_viability_utility(profile.numpy()[0, 0]), places=6)

    def test_calibration_split_is_source_disjoint(self) -> None:
        optimizer, calibration = calibration_split(range(23))
        self.assertEqual(len(calibration), 5)
        self.assertFalse(set(optimizer) & set(calibration))

    def test_abstention_prefers_safe_high_threshold_on_tie(self) -> None:
        profiles = np.zeros((16, 6), dtype=np.float32)
        profiles[:, 4] = 1.0
        record = StateRecord("pair", "state", 1, "fit", np.zeros(4104), np.zeros((16, 14)), profiles)
        threshold, _ = calibrate_threshold([record], [np.zeros(16)])
        self.assertEqual(threshold, max((0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)))


if __name__ == "__main__":
    unittest.main()
