from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
from video_io import write_h264_video


class VideoIoTest(unittest.TestCase):
    def test_writes_avc1_yuv420p_and_pads_odd_dimensions(self) -> None:
        frames = [np.full((31, 47, 3), index * 20, dtype=np.uint8) for index in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "test.mp4"
            write_h264_video(output, frames, fps=12.0)
            with av.open(str(output)) as container:
                stream = container.streams.video[0]
                decoded = list(container.decode(stream))
                self.assertEqual(stream.codec_context.name, "h264")
                self.assertEqual(stream.codec_context.codec_tag, "avc1")
                self.assertEqual(stream.codec_context.format.name, "yuv420p")
                self.assertEqual((stream.width, stream.height), (48, 32))
                self.assertEqual(len(decoded), len(frames))

    def test_rejects_empty_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "empty video"):
                write_h264_video(Path(tmp) / "empty.mp4", [])


if __name__ == "__main__":
    unittest.main()
