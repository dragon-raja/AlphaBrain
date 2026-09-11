from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_libero_closed_loop_videos import audit_runs, mp4_box_offsets
from video_io import write_h264_video


class ClosedLoopVideoAuditTest(unittest.TestCase):
    def test_audits_complete_moving_h264_video_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            video_dir = run_dir / "closed_loop_videos_val_gate"
            video_dir.mkdir(parents=True)
            rows = []
            for horizon in (1, 2, 3):
                for branch, completion_steps in (("attached", 4), ("slipped", 6)):
                    rows.append(
                        {
                            "execution_horizon": horizon,
                            "pair_id": "pair-000",
                            "branch_outcome": branch,
                            "completion_steps": completion_steps,
                        }
                    )
                frames = []
                for index in range(7):
                    frame = np.zeros((32, 48, 3), dtype=np.uint8)
                    frame[:, :, 1] = 80
                    frame[8:20, index : index + 8, 0] = 255
                    frames.append(frame)
                write_h264_video(
                    video_dir / f"end_to_end-k{horizon}-pair-000.mp4",
                    frames,
                    fps=10.0,
                )
            (run_dir / "closed_loop_end_to_end_val_gate.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "evaluation": "end_to_end",
                        "rows": rows,
                    }
                )
            )

            result = audit_runs([run_dir], tag="val_gate", video_groups=1)
            boxes = mp4_box_offsets(video_dir / "end_to_end-k1-pair-000.mp4")

        self.assertTrue(result["passed"])
        self.assertEqual(result["actual_video_count"], 3)
        self.assertEqual(result["total_decoded_frames"], 21)
        self.assertLess(boxes["moov"][0], boxes["mdat"][0])

    def test_static_video_fails_motion_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            video_dir = run_dir / "closed_loop_videos_val_gate"
            video_dir.mkdir(parents=True)
            rows = []
            frame = np.full((32, 48, 3), 80, dtype=np.uint8)
            for horizon in (1, 2, 3):
                for branch in ("attached", "slipped"):
                    rows.append(
                        {
                            "execution_horizon": horizon,
                            "pair_id": "pair-000",
                            "branch_outcome": branch,
                            "completion_steps": 2,
                        }
                    )
                write_h264_video(
                    video_dir / f"end_to_end-k{horizon}-pair-000.mp4",
                    [frame, frame, frame],
                )
            (run_dir / "closed_loop_end_to_end_val_gate.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "evaluation": "end_to_end",
                        "rows": rows,
                    }
                )
            )

            result = audit_runs([run_dir], tag="val_gate", video_groups=1)

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["motion"])


if __name__ == "__main__":
    unittest.main()
