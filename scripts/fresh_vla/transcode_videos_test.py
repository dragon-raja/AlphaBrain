from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
from transcode_videos import discover_videos, inspect_video, is_compatible, transcode_video


def _write_mpeg4(path: Path, frame_count: int = 7) -> None:
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=7)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            image = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class TranscodeVideosTest(unittest.TestCase):
    def test_transcodes_and_preserves_original_under_backup_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "inputs" / "sample.mp4"
            backup_root = root / "backup"
            source.parent.mkdir()
            _write_mpeg4(source)

            result = transcode_video(source, backup_root)

            self.assertEqual(result.status, "transcoded")
            self.assertTrue(is_compatible(inspect_video(source)))
            self.assertEqual(inspect_video(source).frame_count, 7)
            backup = Path(result.backup or "")
            self.assertTrue(backup.is_file())
            self.assertEqual(inspect_video(backup).codec, "mpeg4")

            repeated = transcode_video(source, backup_root)
            self.assertEqual(repeated.status, "already-compatible")

    def test_discovery_excludes_backup_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            backup_root = root / "backup"
            backup = backup_root / "old.mp4"
            _write_mpeg4(source, frame_count=1)
            backup_root.mkdir()
            _write_mpeg4(backup, frame_count=1)

            self.assertEqual(discover_videos([root], backup_root), [source.resolve()])


if __name__ == "__main__":
    unittest.main()
