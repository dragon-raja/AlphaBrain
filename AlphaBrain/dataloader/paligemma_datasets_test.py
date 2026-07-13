import unittest

import numpy as np

from AlphaBrain.dataloader.paligemma_datasets import Pi0DataConfig, Pi0DataTransform


class Pi0DataTransformTest(unittest.TestCase):
    def test_preserves_feedback_horizon(self) -> None:
        transform = Pi0DataTransform(Pi0DataConfig(action_horizon=2))
        sample = {
            "action": np.zeros((2, 7), dtype=np.float32),
            "feedback_horizon": 1,
        }

        result = transform(sample)

        self.assertEqual(result["feedback_horizon"], 1)


if __name__ == "__main__":
    unittest.main()
