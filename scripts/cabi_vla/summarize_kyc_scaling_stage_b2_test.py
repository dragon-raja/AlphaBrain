from __future__ import annotations

import unittest

from summarize_kyc_scaling_stage_b2 import hierarchical_group_bootstrap


class SummarizeKycScalingStageB2Test(unittest.TestCase):
    def test_hierarchical_group_bootstrap_equal_seed_weighting(self) -> None:
        result = hierarchical_group_bootstrap(
            {
                41: {0: 1.0, 1: 1.0},
                42: {0: 0.0, 1: 0.0},
                43: {0: 0.5, 1: 0.5},
            },
            resamples=1_000,
            seed=1,
        )
        self.assertEqual(result["delta"], 0.5)
        self.assertEqual(result["training_seed_count"], 3)
        self.assertEqual(
            result["bootstrap_scheme"],
            "crossed_training_seed_and_snapshot_group",
        )

    def test_bootstrap_requires_the_same_snapshot_groups_for_every_seed(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "identical paired"):
            hierarchical_group_bootstrap(
                {
                    41: {0: 1.0, 1: 1.0},
                    42: {0: 0.0, 2: 0.0},
                },
                resamples=10,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
