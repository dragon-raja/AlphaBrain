from __future__ import annotations

import unittest

from compare_pi05_libero_plus_multiview_gate import build_report


class CompareMultiviewGateTest(unittest.TestCase):
    def test_selects_best_run_and_computes_paired_gain(self) -> None:
        baseline = {
            "task-a": {"canonical": 1.0, "official_camera": 0.0},
            "task-b": {"canonical": 1.0, "official_camera": 0.5},
        }
        weak = {
            "task-a": {"canonical": 1.0, "official_camera": 0.0},
            "task-b": {"canonical": 1.0, "official_camera": 0.5},
        }
        strong = {
            "task-a": {"canonical": 1.0, "official_camera": 1.0},
            "task-b": {"canonical": 1.0, "official_camera": 1.0},
        }
        report = build_report("official", baseline, {"weak": weak, "strong": strong})
        self.assertEqual(report["best_run"], "strong")
        self.assertAlmostEqual(
            report["runs"]["strong"]["paired_minus_official_baseline"]
            ["official_camera_success"]["mean"],
            0.75,
        )
        self.assertTrue(report["gates"]["MATERIAL_MULTIVIEW_GAIN"])


if __name__ == "__main__":
    unittest.main()
