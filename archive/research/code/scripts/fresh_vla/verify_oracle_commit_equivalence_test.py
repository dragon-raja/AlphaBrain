import json
import tempfile
import unittest
from pathlib import Path

from verify_oracle_commit_equivalence import semantic_row, verify_run


class OracleEquivalenceTest(unittest.TestCase):
    def test_ignores_method_specific_labels_and_timing_only(self):
        common = {
            "pair_id": "g0",
            "success": True,
            "commit_trace": [{"commit_length": 2, "boundary_step": 5, "source": "a"}],
        }
        left = {**common, "commit_method": "oracle_branch_safe_commit", "inference_wall_seconds": 1.0}
        right = {
            **common,
            "commit_method": "oracle_feedback_reveal_commit",
            "inference_wall_seconds": 2.0,
            "commit_trace": [{"commit_length": 2, "boundary_step": 5, "source": "b"}],
        }
        self.assertEqual(semantic_row(left), semantic_row(right))
        right["success"] = False
        self.assertNotEqual(semantic_row(left), semantic_row(right))

    def test_video_byte_drift_is_reported_separately_from_semantic_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [root / name for name in ("branch", "feedback")]
            for run in runs:
                (run / "videos" / "end_to_end").mkdir(parents=True)
                payload = {"rows": [{"pair_id": "g0", "success": True, "commit_trace": []}]}
                for name in ("closed_loop_isolated.json", "closed_loop_end_to_end.json", "deterministic_reach.json"):
                    (run / name).write_text(json.dumps(payload))
            video_name = "videos/end_to_end/g0.mp4"
            (runs[0] / video_name).write_bytes(b"left")
            (runs[1] / video_name).write_bytes(b"right")

            result = verify_run(runs[0], runs[1], verify_videos=True)
            self.assertFalse(result["videos"]["byte_identical"])
            self.assertEqual(result["videos"]["mismatches"], ["end_to_end/g0.mp4"])
            with self.assertRaisesRegex(ValueError, "Oracle videos differ"):
                verify_run(runs[0], runs[1], verify_videos=True, require_byte_identical_videos=True)


if __name__ == "__main__":
    unittest.main()
