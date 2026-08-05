from __future__ import annotations

import unittest

import numpy as np

from serve_alphabrain_pi05_websocket import to_alphabrain_example


class AlphaBrainPi05WebsocketTest(unittest.TestCase):
    def test_converts_openpi_observation_without_changing_values(self) -> None:
        agent = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        wrist = np.full((2, 2, 3), 7, dtype=np.uint8)
        state = np.linspace(-1, 1, 8, dtype=np.float32)
        example, seed = to_alphabrain_example(
            {
                "observation/image": agent,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": "pick up the mug",
                "_eval_seed": 41,
            }
        )
        self.assertEqual(seed, 41)
        self.assertEqual(example["lang"], "pick up the mug")
        self.assertEqual(example["language"], "pick up the mug")
        np.testing.assert_array_equal(example["image"][0], agent)
        np.testing.assert_array_equal(example["image"][1], wrist)
        np.testing.assert_array_equal(example["state"], state)
        self.assertNotIn("camera_intrinsics", example)
        self.assertNotIn("camera_to_world_opencv", example)

    def test_passes_camera_calibration_to_alphabrain_example(self) -> None:
        intrinsics = np.arange(9, dtype=np.float64).reshape(3, 3)
        camera_to_world = np.arange(16, dtype=np.float64).reshape(4, 4)
        example, _ = to_alphabrain_example(
            {
                "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
                "observation/wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "observation/state": np.zeros(8, dtype=np.float32),
                "prompt": "task",
                "_eval_seed": 1,
                "camera_intrinsics": intrinsics,
                "camera_to_world_opencv": camera_to_world,
            }
        )
        np.testing.assert_array_equal(example["camera_intrinsics"], intrinsics)
        np.testing.assert_array_equal(example["camera_to_world_opencv"], camera_to_world)

    def test_rejects_unpaired_camera_calibration(self) -> None:
        observation = {
            "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "task",
            "_eval_seed": 1,
        }
        for key, value in (
            ("camera_intrinsics", np.eye(3)),
            ("camera_to_world_opencv", np.eye(4)),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "requires both"):
                to_alphabrain_example({**observation, key: value})

    def test_rejects_invalid_camera_calibration(self) -> None:
        with self.assertRaisesRegex(ValueError, "matrix shapes"):
            to_alphabrain_example(
                {
                    "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "observation/wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "observation/state": np.zeros(8, dtype=np.float32),
                    "prompt": "task",
                    "_eval_seed": 1,
                    "camera_intrinsics": np.eye(4),
                    "camera_to_world_opencv": np.eye(4),
                }
            )

    def test_rejects_wrong_state_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "8D LIBERO state"):
            to_alphabrain_example(
                {
                    "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "observation/wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "observation/state": np.zeros(7, dtype=np.float32),
                    "prompt": "task",
                    "_eval_seed": 1,
                }
            )


if __name__ == "__main__":
    unittest.main()
