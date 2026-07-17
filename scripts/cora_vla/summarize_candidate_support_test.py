import unittest

from summarize_candidate_support import aggregate, paired_group_bootstrap


def payload(seed, attached, slipped16, slipped32, physical1, physical16, leakage=True):
    rows = []
    for index in range(4):
        rows.append(
            {
                "pair_id": f"g{index}",
                "outcome": "attached",
                "joint_recall": {"16": attached[index], "32": attached[index]},
                "physical_recall": {"1": True, "16": True},
            }
        )
        rows.append(
            {
                "pair_id": f"g{index}",
                "outcome": "slipped",
                "joint_recall": {"16": slipped16[index], "32": slipped32[index]},
                "physical_recall": {"1": physical1[index], "16": physical16[index]},
            }
        )
    return {
        "checkpoint_seed": seed,
        "status": "complete",
        "pre_feedback_leakage_passed": leakage,
        "rows": rows,
    }


class CandidateSupportSummaryTest(unittest.TestCase):
    def test_bootstrap_keeps_group_as_unit(self):
        estimate = paired_group_bootstrap({"a": [1, 1, 1], "b": [0, 0, 0]}, samples=1000)
        self.assertEqual(estimate["group_count"], 2)
        self.assertEqual(estimate["mean"], 0.5)

    def test_gate_passes_only_when_all_thresholds_pass(self):
        inputs = [
            payload(
                seed,
                [True, True, True, True],
                [True, True, False, True],
                [True, True, True, True],
                [False, False, False, False],
                [True, True, True, True],
            )
            for seed in (41, 42, 43)
        ]
        self.assertEqual(aggregate(inputs)["decision"], "PASS_CORA_GATE1")

    def test_low_recovery_support_stops_before_energy(self):
        inputs = [
            payload(
                seed,
                [True] * 4,
                [False] * 4,
                [False] * 4,
                [False] * 4,
                [True] * 4,
            )
            for seed in (41, 42, 43)
        ]
        self.assertEqual(aggregate(inputs)["decision"], "BASE_POLICY_LACKS_RECOVERY_SUPPORT")

    def test_leakage_has_precedence(self):
        inputs = [
            payload(
                seed,
                [True] * 4,
                [True] * 4,
                [True] * 4,
                [False] * 4,
                [True] * 4,
                leakage=seed != 42,
            )
            for seed in (41, 42, 43)
        ]
        self.assertEqual(aggregate(inputs)["decision"], "COUNTERFACTUAL_DATA_LEAKAGE")


if __name__ == "__main__":
    unittest.main()
