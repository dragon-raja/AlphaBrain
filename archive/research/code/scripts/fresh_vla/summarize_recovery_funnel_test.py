import unittest

from summarize_recovery_funnel import bootstrap_ratio, conditional_rates, stage_indicators


class RecoveryFunnelTest(unittest.TestCase):
    def test_chain_stops_at_first_missing_stage(self) -> None:
        values = stage_indicators(
            {
                "event_time": 0,
                "intervention_triggered": True,
                "recovery_switch_observed": True,
                "regrasp_success": False,
                "lift_subgoal": True,
                "transport_subgoal": True,
                "place_subgoal": False,
                "recovery_success": False,
            }
        )
        self.assertEqual(values["marginal_lift"], 1.0)
        self.assertEqual(values["chain_recovery_action"], 1.0)
        self.assertEqual(values["chain_regrasp"], 0.0)
        self.assertEqual(values["chain_transport"], 0.0)

    def test_conditionals_use_previous_chain_stage(self) -> None:
        first = {f"chain_{stage}": 1.0 for stage in ("event", "recovery_action", "regrasp", "transport", "success")}
        second = dict(first)
        for stage in ("regrasp", "transport", "success"):
            second[f"chain_{stage}"] = 0.0
        values = conditional_rates([first, second])
        self.assertEqual(values["event_to_recovery_action"], 1.0)
        self.assertEqual(values["recovery_action_to_regrasp"], 0.5)
        self.assertEqual(values["regrasp_to_transport"], 1.0)

    def test_bootstrap_ratio_reports_effective_counts(self) -> None:
        values = bootstrap_ratio([(1.0, 1.0), (0.0, 1.0)], seed=7, samples=100)
        self.assertEqual(values["mean"], 0.5)
        self.assertEqual(values["effective_numerator"], 1.0)
        self.assertEqual(values["effective_denominator"], 2.0)


if __name__ == "__main__":
    unittest.main()
