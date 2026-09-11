from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from transcode_to_av1_webm import inspect_video, transcode_video
from video_io import write_h264_video


class TranscodeToAv1WebmTest(unittest.TestCase):
    def test_transcodes_without_replacing_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            output = root / "output.webm"
            frames = [
                np.full((16, 20, 3), value, dtype=np.uint8)
                for value in (0, 127, 255)
            ]
            write_h264_video(source, frames, fps=5)
            result = transcode_video(source, output, crf=40, cpu_used=8)
            self.assertTrue(source.is_file())
            self.assertTrue(output.is_file())
            self.assertEqual(result["source_video"]["frame_count"], 3)
            self.assertEqual(inspect_video(output)["frame_count"], 3)


if __name__ == "__main__":
    unittest.main()
