import unittest

import numpy as np

from prepare_libero_subset import gripper_feedback_horizon


class GripperFeedbackHorizonTest(unittest.TestCase):
    def test_distance_includes_transition_action(self) -> None:
        actions = np.zeros((6, 2), dtype=np.float32)
        actions[2:, -1] = 1.0
        expected = np.array([3, 2, 1, 10, 10, 10], dtype=np.int16)
        np.testing.assert_array_equal(gripper_feedback_horizon(actions, 10), expected)

    def test_multiple_transitions(self) -> None:
        actions = np.zeros((7, 2), dtype=np.float32)
        actions[2:5, -1] = 1.0
        expected = np.array([3, 2, 1, 3, 2, 1, 10], dtype=np.int16)
        np.testing.assert_array_equal(gripper_feedback_horizon(actions, 10), expected)


if __name__ == "__main__":
    unittest.main()
