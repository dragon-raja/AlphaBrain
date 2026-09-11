from __future__ import annotations

import unittest

import numpy as np

from ccv import (
    assert_deployable_arrays,
    best_candidate_index,
    close_milestones,
    coupled_policy_continuations,
    frozen_source_split,
    profile_from_signatures,
    scalar_viability_utility,
    summary_signature,
)


class CcvTest(unittest.TestCase):
    def test_frozen_source_split_is_group_preserving_and_stable(self) -> None:
        source_ids = [0, 0, 2, 3, 6, 7, 8, 9, 11, 14]
        fit_a, holdout_a = frozen_source_split(source_ids, holdout_count=2)
        fit_b, holdout_b = frozen_source_split(reversed(source_ids), holdout_count=2)
        self.assertEqual((fit_a, holdout_a), (fit_b, holdout_b))
        self.assertFalse(set(fit_a) & set(holdout_a))
        self.assertEqual(set(fit_a) | set(holdout_a), set(source_ids))

    def test_success_closes_all_prerequisite_milestones(self) -> None:
        np.testing.assert_array_equal(close_milestones([False, False, False, True]), np.ones(4))

    def test_profile_is_monotone_and_lexicographic(self) -> None:
        weak = profile_from_signatures(
            np.asarray([[1, 1, 0, 0, 1, 0.4], [1, 1, 0, 0, 1, 0.5]])
        )
        strong = profile_from_signatures(
            np.asarray([[1, 1, 1, 0, 1, 0.7], [1, 1, 1, 0, 1, 0.8]])
        )
        self.assertEqual(best_candidate_index(np.stack([weak, strong])), 1)
        self.assertGreater(scalar_viability_utility(strong), scalar_viability_utility(weak))

    def test_utility_preserves_one_sixth_lexicographic_margin(self) -> None:
        later_terms_maxed = np.asarray([1, 1, 1, 0, 1, 1], dtype=np.float64)
        success_margin = np.asarray([0, 0, 0, 1 / 6, 0, 0], dtype=np.float64)
        self.assertGreater(
            scalar_viability_utility(success_margin),
            scalar_viability_utility(later_terms_maxed),
        )

    def test_summary_signature_uses_only_outcome_summary(self) -> None:
        signature = summary_signature(
            {
                "regrasp_reached": False,
                "lift_reached": False,
                "transport_reached": False,
                "success": True,
                "regress": False,
                "progress_auc": 1.0,
            }
        )
        np.testing.assert_array_equal(signature[:4], np.ones(4))

    def test_deployable_schema_rejects_privileged_fields(self) -> None:
        valid = {
            "agentview_image": np.zeros((2, 2, 3)),
            "wrist_image": np.zeros((2, 2, 3)),
            "robot_state": np.zeros(8),
            "vla_feature": np.zeros(4),
            "candidates": np.zeros((2, 2, 7)),
            "candidate_seeds": np.zeros(2),
        }
        assert_deployable_arrays(valid)
        with self.assertRaises(ValueError):
            assert_deployable_arrays({**valid, "sim_state": np.zeros(3)})

    def test_coupled_continuations_require_attested_policy_path(self) -> None:
        class Pool:
            def reset_continuation(self, count):
                return [{"index": index} for index in range(count)]

            def advance(self, chunks, actions_done, execution_horizon, lookahead_actions):
                return [
                    {"observation": {"index": index}, "executed": execution_horizon}
                    for index in range(len(chunks))
                ]

            def summaries(self, count):
                return [{"candidate": index} for index in range(count)]

        class Policy:
            def __init__(self):
                self.seeds = []

            def predict_observation_batch_coupled(self, observations, *, seed):
                self.seeds.append(seed)
                return np.zeros((len(observations), 2, 7)), 0.0

        policy = Policy()
        results, cost = coupled_policy_continuations(
            Pool(),
            policy,
            endpoint_count=3,
            seed=41,
            pair_id="pair",
            state_id="stage-r2",
            execution_horizon=2,
            lookahead_actions=4,
            repeats=2,
        )
        self.assertEqual([len(rows) for rows in results], [2, 2, 2])
        self.assertEqual(cost["policy_batch_calls"], 4)
        self.assertEqual(cost["simulator_actions"], 24)
        self.assertEqual(len(set(policy.seeds)), 4)


if __name__ == "__main__":
    unittest.main()
