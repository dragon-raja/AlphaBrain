import json
import tempfile
import unittest
from pathlib import Path

from evaluate_physical_process_oracle import aggregate_outcomes
from evaluate_recovery_expert_handoff import EXPERT_SANITY_METHOD, HANDOFF_METHODS
from evaluate_recovery_segment_oracle import BINARY_METRICS, METRICS
from summarize_recovery_expert_handoff import (
    build_summary,
    hierarchical_summary,
    load_and_validate,
)


def outcome(success):
    return {
        metric: bool(success) if metric in BINARY_METRICS else float(success)
        for metric in METRICS
    }


def method_result(method, successes, teacher_actions=0):
    rows = [
        {
            "repeat": repeat,
            "teacher_actions": teacher_actions,
            "policy_calls": 0 if method == EXPERT_SANITY_METHOD else 1,
            "policy_actions": 0 if method == EXPERT_SANITY_METHOD else 3,
            "executed_actions": teacher_actions + (
                0 if method == EXPERT_SANITY_METHOD else 3
            ),
            "outcome": outcome(value),
        }
        for repeat, value in enumerate(successes)
    ]
    return {
        "method": method,
        "teacher_actions": teacher_actions,
        "criterion_reached": True,
        "teacher_done": method == EXPERT_SANITY_METHOD,
        "teacher_success_before_policy": method == EXPERT_SANITY_METHOD,
        "teacher_prefix_regrasp_reached": teacher_actions >= 36,
        "teacher_prefix_lift_reached": teacher_actions >= 42,
        "teacher_prefix_transport_reached": teacher_actions >= 57,
        "continuations": rows,
        "summary": aggregate_outcomes([row["outcome"] for row in rows]),
    }


def payload():
    methods = {
        method: method_result(method, [False] * 5, teacher_actions=index * 3)
        for index, method in enumerate(HANDOFF_METHODS)
    }
    methods[EXPERT_SANITY_METHOD] = method_result(
        EXPERT_SANITY_METHOD,
        [True],
        teacher_actions=72,
    )
    row = {
        "pair_id": "pair",
        "source_initial_state_index": 1,
        "feedback_state_index": 57,
        "seed": 41,
        "method_order_invariance": {
            "first_chunk_max_abs_delta": 0.0,
            "feedback_image_max_abs_delta": 0,
            "feedback_robot_state_max_abs_delta": 0.0,
        },
        "methods": methods,
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "episode_root": "/episodes",
        "split": "val",
        "seed": 41,
        "group_offset": 0,
        "expected_rows": 1,
        "completed_rows": 1,
        "methods": list(HANDOFF_METHODS),
        "expert_sanity_method": EXPERT_SANITY_METHOD,
        "execution_horizon": 3,
        "total_action_budget": 120,
        "max_teacher_actions": 90,
        "continuations": 5,
        "stage_dwell_steps": 2,
        "teacher_is_privileged_upper_bound": True,
        "teacher_privileged_inputs": ["grasp/contact state", "environment success"],
        "policy_receives_teacher_or_branch_labels": False,
        "continuation_seed_protocol": (
            "same repeat and global replan index use the same policy seed across methods"
        ),
        "rows": [row],
    }


class RecoveryExpertHandoffSummaryTest(unittest.TestCase):
    def test_valid_payload_recomputes_raw_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload()))
            rows, identity = load_and_validate([path], require_full_grid=False)
        self.assertEqual(len(rows), 1)
        self.assertEqual(identity["execution_horizon"], 3)

    def test_rejects_summary_corruption(self):
        value = payload()
        value["rows"][0]["methods"]["policy_only"]["summary"]["success_rate"] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "does not match raw success"):
                load_and_validate([path], require_full_grid=False)

    def test_rejects_policy_calls_in_teacher_sanity(self):
        value = payload()
        value["rows"][0]["methods"][EXPERT_SANITY_METHOD]["continuations"][0][
            "policy_calls"
        ] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "must never hand control"):
                load_and_validate([path], require_full_grid=False)

    def test_hierarchical_summary_weights_source_clusters_equally(self):
        rows = [
            {"seed": 41, "source_initial_state_index": 0, "value": 0.0},
            {"seed": 42, "source_initial_state_index": 0, "value": 0.0},
            {"seed": 43, "source_initial_state_index": 0, "value": 0.0},
            {"seed": 41, "source_initial_state_index": 1, "value": 1.0},
        ]
        summary = hierarchical_summary(
            rows,
            lambda row: row["value"],
            bootstrap_samples=200,
            seed=1,
        )
        self.assertEqual(summary["source_cluster_level"]["mean"], 0.5)

    def test_build_summary_uses_bootstrap_key_names(self):
        value = payload()["rows"][0]
        summary = build_summary([value], bootstrap_samples=20, seed=1)
        percentage = summary["absolute"]["policy_only"]["success"][
            "percentage_points"
        ]
        self.assertIn("bootstrap_95_low", percentage)
        self.assertIn("bootstrap_95_high", percentage)


if __name__ == "__main__":
    unittest.main()
