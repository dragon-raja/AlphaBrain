import unittest

import numpy as np

from onpolicy_support import classify_boundary_stage, immediate_correct_mode, recall_at_n


class OnPolicySupportTest(unittest.TestCase):
    def test_grasped_mode_requires_hold(self):
        self.assertTrue(
            immediate_correct_mode(
                starts_grasped=True,
                grasp_trace=[True, True],
                actions=np.zeros((2, 7)),
                failure_continuation_seen=False,
                premature_commitment_seen=False,
                recovery_action_seen=False,
            )
        )
        actions = np.zeros((2, 7)); actions[0, -1] = -1
        self.assertFalse(
            immediate_correct_mode(
                starts_grasped=True,
                grasp_trace=[True, True],
                actions=actions,
                failure_continuation_seen=False,
                premature_commitment_seen=False,
                recovery_action_seen=False,
            )
        )

    def test_ungrasped_mode_rejects_empty_lift(self):
        self.assertFalse(
            immediate_correct_mode(
                starts_grasped=False,
                grasp_trace=[False, False],
                actions=np.zeros((2, 7)),
                failure_continuation_seen=True,
                premature_commitment_seen=False,
                recovery_action_seen=True,
            )
        )

    def test_stage_order_uses_observed_state(self):
        self.assertEqual(
            classify_boundary_stage(
                replan_index=2,
                grasped=True,
                previous_failure_continuation=False,
                recovery_started=True,
                eef_object_distance=0.01,
                initial_eef_object_distance=0.1,
                candidate0_closes=False,
            ),
            "post_regrasp",
        )

    def test_recall_prefix(self):
        self.assertEqual(recall_at_n([False, True] + [False] * 14)["4"], True)


if __name__ == "__main__":
    unittest.main()
