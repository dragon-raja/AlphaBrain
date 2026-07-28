from __future__ import annotations

import unittest

from select_kyc_scaling_stage_b2 import nearest_log_neighbor, select_stage_b2


def summary_with_gains(gains: dict[int, float], factorial_budget: int = 10) -> dict:
    return {
        "budgets": list(gains),
        "budget_results": {
            str(budget): {
                "primary": {
                    "all": {
                        "comparisons": {
                            "kyc_minus_poseaug_control": {
                                "success": {"delta": gain}
                            }
                        }
                    }
                }
            }
            for budget, gain in gains.items()
        },
        "factorial_budget_selection": {
            "selected_budget": factorial_budget,
        },
    }


class SelectKycScalingStageB2Test(unittest.TestCase):
    def test_nearest_neighbor_uses_log_distance(self) -> None:
        self.assertEqual(nearest_log_neighbor(215, [10, 45, 215, 1000]), 1000)

    def test_selects_largest_qualifying_gain_and_neighbor(self) -> None:
        result = select_stage_b2(
            summary_with_gains({10: 0.05, 45: 0.12, 215: 0.08, 1000: 0.0})
        )
        self.assertEqual(result["scaling_confirmation_budgets"], [10, 45])

    def test_fallback_and_factorial_union(self) -> None:
        result = select_stage_b2(
            summary_with_gains(
                {10: 0.01, 45: 0.02, 215: -0.1, 1000: 0.0},
                factorial_budget=215,
            )
        )
        self.assertEqual(result["scaling_confirmation_budgets"], [10, 45])
        self.assertEqual(result["training_budgets"], [10, 45, 215])


if __name__ == "__main__":
    unittest.main()
