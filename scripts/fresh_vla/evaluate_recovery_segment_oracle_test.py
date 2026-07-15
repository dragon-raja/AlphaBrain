import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import evaluate_recovery_segment_oracle as segment_oracle
from evaluate_libero_closed_loop import stable_seed
from evaluate_physical_process_oracle import aggregate_outcomes
from evaluate_recovery_segment_oracle import (
    DECISION_CONFIG,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_PREREGISTRATION_SHA256,
    PROGRESS_TIE_EPSILON,
    aggregate_random_schedule_results,
    build_audit_bank,
    build_training_bank,
    candidate_seed_schedule,
    choose_candidate,
    recovery_preference_key,
    validate_run_config,
    write_method_comparison_video,
)


def outcome(*, success=False, next_stage=False, progress=0.0):
    return {
        "success": success,
        "regress": False,
        "next_stage_reached": next_stage,
        "transport_reached": success,
        "lift_reached": success,
        "stable_grasp_at_end": next_stage or success,
        "first_regrasp_step": 1 if next_stage or success else None,
        "first_transport_step": 2 if success else None,
        "drop": False,
        "progress_auc": progress,
        "object_to_bowl_progress": progress,
        "object_height_progress": progress,
    }


class RecoverySegmentOracleTest(unittest.TestCase):
    def test_candidate_zero_uses_natural_policy_seed(self):
        seeds = candidate_seed_schedule(41, "pair-1", 3, 4)
        self.assertEqual(seeds[0], stable_seed(41, "pair-1", 3))
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_selection_never_uses_heldout_summary(self):
        weak = aggregate_outcomes([outcome(progress=0.1)])
        strong = aggregate_outcomes([outcome(next_stage=True, progress=0.5)])
        rows = [
            {"selection_summary": strong, "heldout_summary": weak},
            {"selection_summary": weak, "heldout_summary": strong},
        ]
        self.assertEqual(choose_candidate(rows, "selection_summary"), 0)
        self.assertEqual(choose_candidate(rows, "heldout_summary"), 1)

    def test_training_bank_excludes_simulator_state_and_unexecuted_suffix(self):
        record = {
            "images": np.zeros((2, 8, 8, 3), dtype=np.uint8),
            "robot_state": np.zeros((8,), dtype=np.float32),
            "candidate_action_prefix": np.zeros((4, 3, 7), dtype=np.float32),
            "candidate_action_mask": np.ones((4, 3), dtype=bool),
            "oracle_index": np.int64(2),
            "replan_index": np.int64(0),
            "decision_uid": np.asarray("seed41:pair-1:replan0"),
            "source_initial_state_index": np.int64(7),
            "model_seed": np.int64(41),
            "candidate_selection_metrics": np.zeros((4, 10), dtype=np.float32),
        }
        bank = build_training_bank([record, record])
        self.assertEqual(bank["images"].shape, (2, 2, 8, 8, 3))
        self.assertEqual(bank["candidate_action_prefix"].shape, (2, 4, 3, 7))
        self.assertEqual(bank["candidate_action_mask"].shape, (2, 4, 3))
        self.assertNotIn("candidate_actions", bank)
        self.assertNotIn("snapshot_sim_state", bank)
        self.assertNotIn("candidate_seeds", bank)
        self.assertNotIn("heldout_oracle_index", bank)
        self.assertNotIn("candidate_heldout_metrics", bank)

    def test_audit_bank_pads_variable_constraint_shapes(self):
        bank = build_audit_bank(
            [
                {"sim_data_efc_force": np.array([1.0, 2.0])},
                {"sim_data_efc_force": np.array([3.0, 4.0, 5.0])},
            ]
        )
        self.assertEqual(bank["sim_data_efc_force"].shape, (2, 3))
        self.assertTrue(
            np.array_equal(
                bank["sim_data_efc_force__valid_shapes"],
                np.array([[2], [3]]),
            )
        )

    def test_preference_key_uses_ordered_recovery_not_object_tiebreaks(self):
        first = outcome(next_stage=True, progress=0.2)
        second = {**first, "object_to_bowl_progress": 99.0, "object_height_progress": 99.0}
        self.assertEqual(recovery_preference_key(first), recovery_preference_key(second))

    def test_progress_is_quantized_before_tiebreak(self):
        first = outcome(progress=0.2)
        second = outcome(progress=0.2 + PROGRESS_TIE_EPSILON / 4)
        self.assertEqual(recovery_preference_key(first), recovery_preference_key(second))

    def test_decision_config_fails_closed_on_custom_value(self):
        values = dict(DECISION_CONFIG)
        values.update(run_kind="decision", seed=41)
        values["sample_count"] = 8
        args = SimpleNamespace(**values)
        with self.assertRaisesRegex(ValueError, "sample_count=4"):
            validate_run_config(args)

    def test_decision_config_requires_frozen_identity(self):
        values = dict(DECISION_CONFIG)
        values.update(run_kind="decision", seed=41)
        args = SimpleNamespace(**values)
        environment = {
            "FRESH_GIT_DIRTY": "0",
            "FRESH_GIT_SHA": "abc123",
            "FRESH_CHECKPOINT_SHA256": EXPECTED_CHECKPOINT_SHA256[41],
            "FRESH_PREREGISTRATION_SHA256": EXPECTED_PREREGISTRATION_SHA256,
        }
        with mock.patch.dict("os.environ", environment, clear=True):
            validate_run_config(args)

    def test_random_schedule_aggregation_preserves_repeats(self):
        base_outcome = outcome(success=True, next_stage=True, progress=1.0)
        result = {
            "random_schedule_index": 0,
            "initial_candidate_action_prefixes": [[[0.0]]],
            "natural_outcome": base_outcome,
            "full_heldout_outcomes": [
                {"repeat": 0, "policy_calls": 1, "simulator_actions": 3, "outcome": base_outcome}
            ],
            "cost": {"candidate_inference_count": 4},
        }
        aggregated = aggregate_random_schedule_results(
            [result, {**result, "random_schedule_index": 1}]
        )
        self.assertEqual(aggregated["random_schedule_count"], 2)
        self.assertEqual(len(aggregated["full_heldout_outcomes"]), 2)
        self.assertEqual(aggregated["cost"]["candidate_inference_count"], 8)

    def test_oracle_executes_each_candidate_once_then_reuses_endpoint(self):
        candidates = np.zeros((2, 3, 7), dtype=np.float32)
        candidates[1] = 1.0
        policy = mock.Mock()
        policy.predict_many.return_value = (candidates, 0.01)
        env = mock.Mock()
        env.get_sim_state.return_value = np.zeros((2,), dtype=np.float64)
        observation = {"opaque": True}
        snapshot = {
            "sim_state": np.zeros((2,), dtype=np.float64),
            "controller_state": {
                "model_body_pos": np.zeros((1, 3)),
                "object_friction": np.zeros((1, 3)),
                "gripper_action": np.zeros((1,)),
            },
        }
        direct = outcome(next_stage=True, progress=0.3)
        endpoint_one = {
            "endpoint_snapshot": snapshot,
            "trace": [{}, {}],
            "direct": direct,
            "candidate_actions": 3,
            "direct_sim_state": np.zeros((2,), dtype=np.float64),
            "direct_policy_images": np.zeros((2, 2, 2, 3), dtype=np.uint8),
            "direct_policy_state": np.zeros((8,), dtype=np.float64),
        }
        endpoint_zero = dict(endpoint_one)
        seen_endpoints = []

        def continuation_side_effect(_env, _policy, endpoint, **_kwargs):
            seen_endpoints.append(endpoint)
            return {
                "bridge": direct,
                "continuation_policy_calls": 1,
                "continuation_actions": 3,
            }

        policy_input = {
            "image": [
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.zeros((2, 2, 3), dtype=np.uint8),
            ],
            "state": np.zeros((8,), dtype=np.float64),
        }
        with (
            mock.patch.object(
                segment_oracle,
                "_execute_candidate",
                return_value=(observation, 3),
            ) as execute,
            mock.patch.object(
                segment_oracle,
                "capture_runtime_snapshot",
                return_value=snapshot,
            ),
            mock.patch.object(
                segment_oracle,
                "restore_runtime_snapshot",
                return_value=observation,
            ),
            mock.patch.object(
                segment_oracle,
                "_policy_observation",
                return_value=policy_input,
            ),
            mock.patch.object(
                segment_oracle,
                "summarize_recovery_trace",
                return_value=direct,
            ),
            mock.patch.object(
                segment_oracle,
                "generate_candidate_endpoint",
                side_effect=[endpoint_zero, endpoint_one, endpoint_zero],
            ) as generate,
            mock.patch.object(
                segment_oracle,
                "rollout_endpoint_continuation",
                side_effect=continuation_side_effect,
            ),
        ):
            decision, _, training, _, _ = segment_oracle.evaluate_oracle_decision(
                env,
                policy,
                snapshot,
                observation,
                [{}, {}],
                pair_id="pair-1",
                seed=41,
                source_initial_state_index=7,
                replan_index=0,
                sample_count=2,
                execution_horizon=3,
                lookahead_steps=3,
                selection_continuations=2,
                decision_heldout_continuations=2,
                stage_dwell_steps=2,
            )

        self.assertEqual(execute.call_count, 0)
        # Both selection endpoints live in the branch environment. Candidate 0
        # receives one extra K-step replay used only for parity auditing.
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(len(seen_endpoints), 8)
        self.assertEqual(len({id(endpoint) for endpoint in seen_endpoints}), 2)
        self.assertEqual(training["candidate_action_mask"].shape, (2, 3))
        self.assertTrue(decision["selection_matches_decision_heldout"])
        self.assertTrue(decision["candidate0_branch_replay_semantic_match"])
        self.assertEqual(
            decision["cost"]["candidate0_parity_replay_simulator_actions"], 3
        )

    def test_comparison_video_pads_shorter_method_with_last_frame(self):
        first = [np.zeros((8, 8, 3), dtype=np.uint8)]
        second = [
            np.full((8, 8, 3), value, dtype=np.uint8) for value in (10, 20)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "comparison.mp4"
            write_method_comparison_video(output, [first, second])
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
