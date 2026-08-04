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
