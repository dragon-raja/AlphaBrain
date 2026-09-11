from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.basin_vla.policy_relativity import (
    STAGES,
    leave_one_policy_out_choice,
    lexicographic_percentiles,
    pairwise_policy_metrics,
    select_cache_files,
)


class PolicyRelativityTest(unittest.TestCase):
    def test_selection_is_stage_balanced_and_deterministic(self) -> None:
        paths = [
            Path(f"pair-{stage}-{index}--{stage}--r{index}.npz")
            for stage in STAGES
            for index in range(5)
        ]
        first = select_cache_files(paths, per_stage=3)
        second = select_cache_files(list(reversed(paths)), per_stage=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 21)
        self.assertEqual(
            {stage: sum(f"--{stage}--" in path.name for path in first) for stage in STAGES},
            {stage: 3 for stage in STAGES},
        )

    def test_lexicographic_percentiles_preserve_ties(self) -> None:
        ranks = lexicographic_percentiles([[0, 1], [1, 0], [1, 0], [1, 1]])
        np.testing.assert_allclose(ranks, [0.0, 0.5, 0.5, 1.0])
        np.testing.assert_allclose(lexicographic_percentiles([[1], [1]]), [0.5, 0.5])

    def test_pairwise_metrics_count_preference_flips(self) -> None:
        metrics = pairwise_policy_metrics([0.0, 0.5, 1.0], [1.0, 0.5, 0.0])
        self.assertEqual(metrics["comparable_pair_count"], 3)
        self.assertEqual(metrics["preference_flip_rate"], 1.0)
        self.assertEqual(metrics["top_tier_jaccard"], 0.0)

    def test_leave_one_policy_out_exposes_policy_specific_choice(self) -> None:
        result = leave_one_policy_out_choice(
            {
                41: [1.0, 0.0],
                42: [0.0, 1.0],
                43: [0.0, 1.0],
            },
            41,
        )
        self.assertEqual(result["selected_index"], 1)
        self.assertEqual(result["oracle_minus_loo"], 1.0)


if __name__ == "__main__":
    unittest.main()
