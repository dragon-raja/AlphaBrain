from __future__ import annotations

import unittest

from summarize_recovery_support_closed_loop import support_decision


def summary(mean: float, low: float, high: float) -> dict:
    return {
        "mean": mean,
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
    }


def comparisons() -> dict:
    result = {"3": {}}
    for name in ("base_vs_original", "clean_vs_base", "policy_vs_base", "policy_vs_clean"):
        result["3"][name] = {
            metric: {"candidate_minus_baseline": summary(0.0, -0.05, 0.05)}
            for metric in (
                "slip_recovery_success",
                "overall_task_success",
                "attached_task_success",
            )
        }
    return result


class SupportDecisionTest(unittest.TestCase):
    def test_continues_only_when_policy_state_beats_clean(self) -> None:
        values = comparisons()
        values["3"]["policy_vs_clean"]["slip_recovery_success"]["candidate_minus_baseline"] = summary(
            0.15, 0.04, 0.24
        )
        values["3"]["policy_vs_clean"]["attached_task_success"]["candidate_minus_baseline"] = summary(
            -0.02, -0.08, 0.03
        )
        values["3"]["policy_vs_base"]["overall_task_success"]["candidate_minus_baseline"] = summary(
            0.13, 0.03, 0.22
        )
        values["3"]["policy_vs_base"]["attached_task_success"]["candidate_minus_baseline"] = summary(
            -0.01, -0.07, 0.04
        )
        result = support_decision(values)
        self.assertEqual(result["decision"], "CONTINUE_MINIMAL_RECOVERY_BRIDGE")

    def test_does_not_continue_when_policy_only_beats_weak_clean_control(self) -> None:
        values = comparisons()
        values["3"]["policy_vs_clean"]["slip_recovery_success"]["candidate_minus_baseline"] = summary(
            0.15, 0.04, 0.24
        )
        result = support_decision(values)
        self.assertEqual(result["decision"], "STOP_OFFLINE_SUPPORT_EXPANSION")

    def test_adopts_clean_replay_when_it_is_sufficient(self) -> None:
        values = comparisons()
        values["3"]["clean_vs_base"]["overall_task_success"]["candidate_minus_baseline"] = summary(
            0.12, 0.02, 0.21
        )
        result = support_decision(values)
        self.assertEqual(result["decision"], "ADOPT_CLEAN_RECOVERY_REPLAY")

    def test_attributes_plain_continuation_when_it_is_sufficient(self) -> None:
        values = comparisons()
        values["3"]["base_vs_original"]["slip_recovery_success"]["candidate_minus_baseline"] = summary(
            0.11, 0.01, 0.20
        )
        result = support_decision(values)
        self.assertEqual(result["decision"], "BASELINE_UNDERTRAINED")

    def test_stops_when_no_behavior_effect_clears_ci(self) -> None:
        result = support_decision(comparisons())
        self.assertEqual(result["decision"], "STOP_OFFLINE_SUPPORT_EXPANSION")


if __name__ == "__main__":
    unittest.main()
