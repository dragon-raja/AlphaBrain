import unittest

import numpy as np

from evaluate_libero_closed_loop import _policy_observation


class RecoveryPromptEvaluationTest(unittest.TestCase):
    def test_policy_observation_uses_prompt_override(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        observation = {
            "agentview_image": image,
            "robot0_eye_in_hand_image": image,
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }
        result = _policy_observation(observation, "recover the failed grasp")
        self.assertEqual(result["lang"], "recover the failed grasp")
        self.assertEqual(result["language"], "recover the failed grasp")


if __name__ == "__main__":
    unittest.main()
