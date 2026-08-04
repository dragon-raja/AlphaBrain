import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from tfrecord.writer import TFRecordWriter

from AlphaBrain.dataloader.paligemma_datasets import LiberoPlusTFRecordDataset
from build_libero_plus_training_view import scan_tfrecord_offsets


def jpeg(value: int) -> bytes:
    image = Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


class LiberoPlusTFRecordDatasetTest(unittest.TestCase):
    def test_random_access_and_action_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            shard = source / "tiny.tfrecord"
            writer = TFRecordWriter(str(shard))
            episodes = []
            for episode_index, length in enumerate((3, 2)):
                actions = np.arange(length * 7, dtype=np.float32).reshape(length, 7)
                states = np.arange(length * 8, dtype=np.float32).reshape(length, 8)
                writer.write(
                    {
                        "steps/action": (actions.reshape(-1), "float"),
                        "steps/observation/state": (states.reshape(-1), "float"),
                        "steps/observation/image": (
                            [jpeg(10 + episode_index + frame) for frame in range(length)],
                            "byte",
                        ),
                        "steps/observation/wrist_image": (
                            [jpeg(20 + episode_index + frame) for frame in range(length)],
                            "byte",
                        ),
                    }
                )
                episodes.append((length, actions))
            writer.close()
            offsets = scan_tfrecord_offsets(shard)

            view = root / "view"
            view.mkdir()
            rows = []
            for index, ((length, _), (offset, total_bytes)) in enumerate(zip(episodes, offsets)):
                rows.append(
                    {
                        "episode_id": f"ep-{index}",
                        "shard": "tiny.tfrecord",
                        "record_index": index,
                        "record_offset": offset,
                        "record_total_bytes": total_bytes,
                        "language_instruction": f"task-{index}",
                        "step_count": length,
                        "camera_pose_group_id": f"pose-{index}",
                        "camera_to_world_opencv": np.eye(4).tolist(),
                        "split": "train",
                        "budget_percentile": 0.25,
                    }
                )
            manifest = {
                "status": "complete",
                "dataset_root": str(source),
                "image_schema": {
                    "external_camera_intrinsics_224": np.eye(3).tolist(),
                },
                "action_schema": {"action_dim": 7},
                "episodes": rows,
            }
            (view / "manifest.json").write_text(json.dumps(manifest))

            dataset = LiberoPlusTFRecordDataset(view, action_horizon=4)
            self.assertEqual(len(dataset), 5)
            first = dataset[0]
            self.assertEqual(first["image"][0].shape, (16, 16, 3))
            np.testing.assert_array_equal(first["action"][:3], episodes[0][1])
            np.testing.assert_array_equal(first["action"][3], np.zeros(7))
            boundary = dataset[3]
            self.assertEqual(boundary["episode_id"], "ep-1")
            self.assertEqual(boundary["frame_index"], 0)


if __name__ == "__main__":
    unittest.main()
