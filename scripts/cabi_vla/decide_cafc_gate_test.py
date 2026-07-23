from __future__ import annotations

import unittest

from decide_cafc_gate import decide_cafc_gate, evaluate_comparison, state0_summary


def state0(id_success: int, ood_success: int) -> dict:
    rows = []
    for index in range(4):
        rows.append({"action_supervised": True, "success": index < id_success})
    for index in range(2):
        rows.append({"action_supervised": False, "success": index < ood_success})
    return {"status": "complete", "rows": rows}


def comparison(*, passing: bool = True) -> dict:
    gain = 0.2 if passing else 0.0
    source_gain = 0.2 if passing else 0.0
    edge_gain = 0.2 if passing else 0.0
    return {
        "decision_horizon": 3,
        "results": {
            "k3": {
                "id": {
                    "success": {
                        "baseline": 0.8,
                        "method": 0.8,
                        "difference": 0.0,
                    }
                },
                "ood": {
                    "success": {
                        "baseline": 0.0,
                        "method": gain,
                        "difference": gain,
                        "ci95_low": 0.0,
                        "ci95_high": gain,
                    },
                    "source_selection_success": {
                        "difference": source_gain,
                        "ci95_low": 0.0,
                        "ci95_high": source_gain,
                    },
                },
                "by_edge": {
                    "white-right": {"difference": edge_gain},
                    "yellow_white-left": {"difference": edge_gain},
                },
            }
        },
    }


class DecideCafcGateTest(unittest.TestCase):
    def test_state0_requires_three_observed_and_one_action_free(self) -> None:
        summary = state0_summary(state0(3, 1))
        self.assertTrue(summary["eligible_for_validation"])
        self.assertFalse(state0_summary(state0(2, 2))["id_valid"])

    def test_comparison_requires_both_withheld_edges(self) -> None:
        payload = comparison()
        payload["results"]["k3"]["by_edge"]["yellow_white-left"]["difference"] = 0.0
        result = evaluate_comparison(payload)
        self.assertFalse(result["passed"])
        self.assertFalse(result["criteria"]["both_withheld_edges_improve"])

    def test_plain_cafc_has_priority_when_both_arms_pass(self) -> None:
        result = decide_cafc_gate(
            plain_state0=state0(3, 1),
            grounded_state0=state0(4, 2),
            plain_exact=comparison(),
            grounded_exact=comparison(),
            plain_strong=comparison(),
            grounded_strong=comparison(),
        )
        self.assertEqual(result["decision"], "ADVANCE_CAFC")

    def test_grounded_decision_requires_both_comparators(self) -> None:
        result = decide_cafc_gate(
            plain_state0=state0(3, 1),
            grounded_state0=state0(3, 1),
            plain_exact=comparison(passing=False),
            grounded_exact=comparison(),
            plain_strong=comparison(passing=False),
            grounded_strong=comparison(),
        )
        self.assertEqual(result["decision"], "ADVANCE_GROUNDED_CAFC")

    def test_invalid_observed_calibration_is_not_a_method_failure(self) -> None:
        result = decide_cafc_gate(
            plain_state0=state0(2, 1),
            grounded_state0=state0(2, 2),
            plain_exact=comparison(passing=False),
            grounded_exact=comparison(passing=False),
            plain_strong=comparison(passing=False),
            grounded_strong=comparison(passing=False),
        )
        self.assertEqual(result["decision"], "BASELINE_INVALID")


if __name__ == "__main__":
    unittest.main()
