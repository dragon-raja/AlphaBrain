from __future__ import annotations

import unittest

from analyze_libero_bind_camera_viewpoints import (
    paired_bootstrap_delta,
    rank_nonbaseline,
    summarize_rows,
)


def row(pose: str, state: int, edge: str, success: bool, progress: float) -> dict:
    return {
        "camera_pose": pose,
        "canonical_state_index": state,
        "edge_id": edge,
        "camera_azimuth_deg": 0.0 if pose == "baseline" else 15.0,
        "camera_elevation_deg": 0.0,
        "camera_radius_scale": 1.0,
        "success": success,
        "source_selection_success": success,
        "wrong_source_grasp": False,
        "lift_success": success,
        "transport_success": success,
        "progress": progress,
        "completion_steps": 100 if success else 320,
    }


class AnalyzeCameraSweepTest(unittest.TestCase):
    def test_summary_and_ranking_use_closed_loop_outcomes(self) -> None:
        rows = [
            row("baseline", 0, "a", False, 0.25),
            row("candidate", 0, "a", True, 1.0),
        ]
        summaries = summarize_rows(rows)
        self.assertEqual(rank_nonbaseline(summaries), ["candidate"])
        candidate = next(value for value in summaries if value["camera_pose"] == "candidate")
        self.assertEqual(candidate["success"], 1.0)

    def test_bootstrap_pairs_by_state_not_frame(self) -> None:
        rows = [
            row("baseline", 0, "a", False, 0.0),
            row("baseline", 0, "b", True, 1.0),
            row("candidate", 0, "a", True, 1.0),
            row("candidate", 0, "b", True, 1.0),
            row("baseline", 1, "a", False, 0.0),
            row("candidate", 1, "a", True, 1.0),
        ]
        result = paired_bootstrap_delta(
            rows,
            candidate="candidate",
            baseline="baseline",
            metric="success",
            samples=100,
        )
        self.assertEqual(result["paired_state_count"], 2)
        self.assertAlmostEqual(result["delta"], 0.75)


if __name__ == "__main__":
    unittest.main()
