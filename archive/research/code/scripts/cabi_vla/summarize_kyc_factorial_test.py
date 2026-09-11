from __future__ import annotations

import unittest
from pathlib import Path

from summarize_kyc_factorial import (
    bootstrap_group_contrast,
    evaluation_path,
)


class SummarizeKycFactorialTest(unittest.TestCase):
    def test_evaluation_path_matches_short_run_name(self) -> None:
        path = evaluation_path(
            Path("/tmp/eval"),
            budget=45,
            train_scene="cue_randomized",
            wrist="off",
            arm="poseaug_control",
            seed=41,
            eval_scene="fixed",
        )
        self.assertEqual(
            path.name,
            "camera_sweep_test.json",
        )
        self.assertEqual(
            path.parent.name,
            "n45-trcue-woff-mctrl-s41-evfx",
        )

    def test_group_contrast_uses_paired_groups(self) -> None:
        result = bootstrap_group_contrast(
            {
                "fx_on": {40: 0.0, 41: 0.1},
                "fx_off": {40: 0.2, 41: 0.3},
            },
            coefficients={"fx_off": 1.0, "fx_on": -1.0},
            bootstrap_resamples=1_000,
            seed=1,
        )
        self.assertAlmostEqual(result["estimate"], 0.2)
        self.assertEqual(result["snapshot_group_count"], 2)


if __name__ == "__main__":
    unittest.main()
