import unittest

import numpy as np

from fixed_k_evaluator import EpisodeSpec, evaluate_fixed_k


class ConstantPolicy:
    def predict_action(self, observation):
        return np.ones((3, 1), dtype=np.float32)


class SnapshotEnvironment:
    def reset_to(self, initial_state, branch_outcome):
        self.steps = 0
        self.branch_outcome = branch_outcome
        return {"state": initial_state}

    def step(self, action):
        self.steps += 1
        success = self.steps >= (2 if self.branch_outcome == "success" else 4)
        info = {
            "success": success,
            "premature_commitment": self.branch_outcome == "failure" and self.steps == 1,
            "failure_continuation": False,
            "recovery_success": self.branch_outcome == "failure" and success,
        }
        return {"step": self.steps}, float(success), success, info

    def close(self):
        self.closed = True


class FixedKEvaluatorTest(unittest.TestCase):
    def test_fixed_k_uses_identical_episodes_and_reports_recovery(self):
        episodes = [
            EpisodeSpec("success-0", [0.0], "success"),
            EpisodeSpec("failure-0", [0.0], "failure"),
            EpisodeSpec("control-0", [0.0], "success", is_deterministic_control=True),
        ]
        result = evaluate_fixed_k(
            SnapshotEnvironment,
            ConstantPolicy(),
            episodes,
            execution_horizons=(1, 3),
            max_steps=6,
        )
        self.assertEqual(result["1"]["episode_count"], 3)
        self.assertEqual(result["3"]["episode_count"], 3)
        self.assertEqual(result["1"]["success_rate"], 1.0)
        self.assertEqual(result["3"]["recovery_success_rate"], 1.0)
        self.assertEqual(result["1"]["premature_commitment_rate"], 1 / 3)

    def test_rejects_execution_horizon_larger_than_chunk(self):
        with self.assertRaisesRegex(ValueError, "policy chunk"):
            evaluate_fixed_k(
                SnapshotEnvironment,
                ConstantPolicy(),
                [EpisodeSpec("one", [0.0], "success")],
                execution_horizons=(4,),
            )


if __name__ == "__main__":
    unittest.main()
