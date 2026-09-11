from __future__ import annotations

import unittest

from join_constructed_accel_m1 import join_state, summarize


def _accel() -> dict:
    roles = {
        "canonical": "canonical",
        "strong_info": "strong",
        "matched_control": "control",
        "blind": "blind",
    }
    scores = {"canonical": 0.4, "strong": 0.1, "control": 0.2, "blind": 0.8}
    return {
        "record": {
            "pair_key": "pair",
            "task_id": "task",
            "source_episode_id": "demo",
            "fixed_state_audit": {"physics_state_sha256": "physics"},
            "role_metrics": {
                role: {"candidate_id": candidate_id}
                for role, candidate_id in roles.items()
            },
        },
        "rankings": {
            "complete": {
                "ranking": [
                    {"candidate_id": candidate_id, "accel_3": score}
                    for candidate_id, score in scores.items()
                ]
            }
        },
    }


def _m1() -> dict:
    values = {
        "canonical_both": (False, 520),
        "strong_info_both": (True, 80),
        "matched_control_both": (True, 100),
        "blind_both": (False, 520),
    }
    return {
        condition: {
            "success": success,
            "completion_steps": steps,
            "initial_metrics": {"physics_state_sha256": "physics"},
        }
        for condition, (success, steps) in values.items()
    }


class JoinConstructedAccelM1Test(unittest.TestCase):
    def test_selects_lowest_accel_and_defines_success_and_efficiency_oracles(self) -> None:
        row = join_state(_accel(), _m1())
        self.assertEqual(row["accel_selected_role_evaluated4"], "strong_info")
        self.assertTrue(row["accel_selected_success"])
        self.assertEqual(
            set(row["successful_roles"]), {"strong_info", "matched_control"}
        )
        self.assertEqual(row["efficiency_oracle_roles"], ["strong_info"])
        self.assertTrue(row["accel_exact_efficiency_oracle_match"])

    def test_rejects_physics_state_mismatch(self) -> None:
        m1 = _m1()
        m1["blind_both"]["initial_metrics"]["physics_state_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "physical state mismatch"):
            join_state(_accel(), m1)

    def test_summary_bootstraps_source_groups_not_frames(self) -> None:
        first = join_state(_accel(), _m1())
        second = dict(first)
        second["pair_key"] = "pair-2"
        second["source_episode_group"] = "demo"
        result = summarize([first, second], bootstrap_samples=100, seed=1)
        self.assertEqual(result["paired_state_count"], 2)
        self.assertEqual(result["independent_source_episode_group_count"], 1)
        self.assertEqual(result["accel_selected_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
