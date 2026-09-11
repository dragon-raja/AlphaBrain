import unittest

from decide_h20_cafc_gate import decide_h20_gate


def state0(id_success: int, ood_success: int) -> dict:
    rows = []
    for index in range(4):
        rows.append({"action_supervised": True, "success": index < id_success})
    for index in range(2):
        rows.append({"action_supervised": False, "success": index < ood_success})
    return {"status": "complete", "rows": rows}


def comparison(*, passed: bool = True) -> dict:
    gain = 0.2 if passed else 0.0
    edge_gain = 1.0 if passed else 0.0
    return {
        "decision_horizon": 3,
        "results": {
            "k3": {
                "id": {
                    "success": {
                        "baseline": 0.75,
                        "method": 0.75,
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
                        "baseline": 0.0,
                        "method": gain,
                        "difference": gain,
                        "ci95_low": 0.0,
                        "ci95_high": gain,
                    },
                },
                "by_edge": {
                    "white-right": {"difference": edge_gain},
                    "yellow_white-left": {"difference": edge_gain},
                },
            }
        },
    }


class H20DecisionTest(unittest.TestCase):
    def inputs(self) -> dict:
        return {
            "bc_state0": state0(3, 0),
            "bridge_state0": state0(3, 0),
            "plain_state0": state0(3, 1),
            "grounded_state0": state0(3, 1),
            "plain_exact": comparison(),
            "grounded_exact": comparison(),
            "plain_strong": comparison(),
            "grounded_strong": comparison(),
        }

    def test_plain_arm_advances_first(self) -> None:
        result = decide_h20_gate(**self.inputs())
        self.assertEqual(result["decision"], "ADVANCE_H20_CAFC")

    def test_grounded_arm_can_advance(self) -> None:
        inputs = self.inputs()
        inputs["plain_exact"] = comparison(passed=False)
        result = decide_h20_gate(**inputs)
        self.assertEqual(result["decision"], "ADVANCE_H20_GROUNDED_CAFC")

    def test_invalid_exact_controls_cannot_create_a_claim(self) -> None:
        inputs = self.inputs()
        inputs["bc_state0"] = state0(2, 0)
        inputs["bridge_state0"] = state0(2, 0)
        result = decide_h20_gate(**inputs)
        self.assertEqual(result["decision"], "BASELINE_INVALID")

    def test_valid_but_failed_comparison_stops_extension(self) -> None:
        inputs = self.inputs()
        inputs["plain_exact"] = comparison(passed=False)
        inputs["grounded_exact"] = comparison(passed=False)
        result = decide_h20_gate(**inputs)
        self.assertEqual(result["decision"], "STOP_HORIZON_EXTENSION")


if __name__ == "__main__":
    unittest.main()

