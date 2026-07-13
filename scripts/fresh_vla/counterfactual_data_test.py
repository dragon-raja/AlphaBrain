import unittest

import numpy as np

from counterfactual_data import (
    CounterfactualRecord,
    assert_no_future_information,
    build_policy_inputs,
    estimate_branch_divergence,
    first_persistent_divergence,
    threshold_sensitivity,
    validate_record,
)


class CounterfactualDataTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        common = np.zeros((6, 2))
        plus = common.copy()
        minus = common.copy()
        plus[3:, 0] = 1.0
        minus[3:, 0] = -1.0
        self.rollouts = {
            "success": np.stack([plus + rng.normal(0, 0.01, plus.shape) for _ in range(4)]),
            "failure": np.stack([minus + rng.normal(0, 0.01, minus.shape) for _ in range(4)]),
        }

    def test_oracle_horizon_is_first_persistent_branch_divergence(self) -> None:
        estimate = estimate_branch_divergence(self.rollouts, persistence=2)
        self.assertEqual(estimate.action_divergence_time, 3)
        self.assertEqual(estimate.oracle_feedback_horizon, 3)
        self.assertEqual(len(estimate.per_step_branch_divergence), 6)

    def test_single_spike_is_not_persistent_divergence(self) -> None:
        self.assertEqual(first_persistent_divergence(np.array([0.0, 2.0, 0.0, 2.0]), 1.0, 2), 4)

    def test_threshold_sensitivity_reports_each_multiplier(self) -> None:
        estimates = threshold_sensitivity(self.rollouts, multipliers=(0.5, 1.0, 2.0))
        self.assertEqual(set(estimates), {"0.5", "1.0", "2.0"})
        self.assertTrue(all(value.oracle_feedback_horizon == 3 for value in estimates.values()))

    def test_future_information_is_not_a_policy_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "future-only"):
            assert_no_future_information({"observation": [0.0], "branch_outcome": "success"})

    def test_record_validation_and_policy_projection(self) -> None:
        record = CounterfactualRecord(
            pair_id="pair-0",
            branch_id="success",
            branch_outcome="attached",
            observation={"pixels": "reference-only"},
            robot_state=[0.0, 1.0],
            language_instruction="lift the object",
            action_chunk=[[0.0, 0.0]] * 3 + [[1.0, 0.0]] * 3,
            event_time=2,
            feedback_reveal_time=3,
            action_divergence_time=3,
            gripper_transition_horizon=2,
            oracle_feedback_horizon=3,
            per_step_branch_divergence=[0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            is_deterministic_control=False,
        )
        validate_record(record)
        self.assertEqual(set(build_policy_inputs(record)), {"observation", "robot_state", "language_instruction"})


if __name__ == "__main__":
    unittest.main()
