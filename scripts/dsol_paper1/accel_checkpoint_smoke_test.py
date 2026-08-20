from __future__ import annotations

import unittest

import numpy as np

from run_accel_checkpoint_smoke import CANDIDATE_IDS, build_fixed_state_candidates


def _sample(pose: str) -> dict:
    value = {"canonical": 10, "broad_a": 20, "broad_b": 30}[pose]
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = value / 100.0
    return {
        "image": [
            np.full((4, 4, 3), value, dtype=np.uint8),
            np.full((4, 4, 3), 40, dtype=np.uint8),
        ],
        "lang": "do the task",
        "action": np.ones((10, 7), dtype=np.float32),
        "state": np.arange(8, dtype=np.float32),
        "sample_id": f"pair::{pose}",
        "episode_id": "episode-1",
        "frame_index": 5,
        "camera_pose": pose,
        "camera_to_world_opencv": matrix.tolist(),
        "dsol_pair_id": "pair",
    }


class AccelCheckpointSmokeTest(unittest.TestCase):
    def test_builds_physical_and_blackout_candidates_from_one_state(self) -> None:
        candidates, metadata, audit = build_fixed_state_candidates(
            _sample("canonical"), [_sample("broad_a"), _sample("broad_b")]
        )
        self.assertEqual(len(candidates), len(CANDIDATE_IDS))
        self.assertEqual([row["candidate_id"] for row in metadata], list(CANDIDATE_IDS))
        self.assertTrue(audit["same_robot_state"])
        self.assertTrue(np.all(candidates[3]["image"][0] == 0))
        self.assertTrue(np.all(candidates[3]["image"][1] == 40))
        self.assertTrue(np.all(candidates[4]["image"][0] == 0))
        self.assertTrue(np.all(candidates[4]["image"][1] == 0))

    def test_rejects_nonpaired_state(self) -> None:
        broad_b = _sample("broad_b")
        broad_b["state"][0] = 99
        with self.assertRaisesRegex(ValueError, "different robot states"):
            build_fixed_state_candidates(
                _sample("canonical"), [_sample("broad_a"), broad_b]
            )


if __name__ == "__main__":
    unittest.main()
