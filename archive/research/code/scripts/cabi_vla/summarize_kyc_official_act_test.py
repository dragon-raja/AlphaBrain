from __future__ import annotations

import unittest

import numpy as np

from summarize_kyc_official_act import hierarchical_bootstrap, paired_delta


class SummarizeKycOfficialActTest(unittest.TestCase):
    def test_paired_delta_uses_episode_identity(self) -> None:
        delta = paired_delta(
            {0: 0.0, 1: 1.0},
            {1: 1.0, 0: 1.0},
        )
        np.testing.assert_array_equal(delta, [1.0, 0.0])

    def test_hierarchical_bootstrap_reports_equal_seed_mean(self) -> None:
        result = hierarchical_bootstrap(
            {
                0: np.asarray([1.0, 1.0]),
                1: np.asarray([0.0, 0.0]),
            },
            resamples=1_000,
            seed=1,
        )
        self.assertEqual(result["delta"], 0.5)
        self.assertEqual(result["training_seed_count"], 2)


if __name__ == "__main__":
    unittest.main()
