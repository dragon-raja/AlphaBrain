import unittest

from summarize_libero_oracle_commit import apply_decision_gate


def summary(value):
    return {"mean": value, "bootstrap_95_low": value - 0.01, "bootstrap_95_high": value + 0.01}


def comparison(value):
    return {
        "candidate_minus_baseline": summary(value),
        "seed_deltas": {"41": value, "42": value, "43": value},
    }


class OracleCommitDecisionTest(unittest.TestCase):
    def test_go_requires_all_registered_controls(self):
        metrics = {
            "overall_task_success": comparison(0.12),
            "slip_recovery_success": comparison(0.15),
            "isolated_recovery_success": comparison(0.08),
            "slip_regrasp_success": comparison(0.08),
            "attached_task_success": comparison(-0.02),
            "failure_continuation_rate": comparison(-0.10),
            "premature_commitment_rate": comparison(-0.08),
        }
        comparisons = {
            "oracle_vs_fixed_k3": metrics,
            "oracle_vs_random_matched_commit": metrics,
            "oracle_vs_gripper_commit": metrics,
            "oracle_vs_self_consistency_commit": metrics,
        }
        aggregate = {
            "oracle_branch_safe_commit": {
                "overall_task_success": summary(0.50),
                "policy_forward_calls": summary(40.0),
            },
            "fixed_k1": {
                "overall_task_success": summary(0.52),
                "policy_forward_calls": summary(100.0),
            },
        }
        decision, gate = apply_decision_gate(aggregate, comparisons)
        self.assertEqual(decision, "GO_COUNTERFACTUAL_COMMITMENT")
        self.assertTrue(gate["fixed_k1_efficiency_gate"])

    def test_small_primary_effect_stops_family(self):
        weak = comparison(0.02)
        negative = comparison(-0.02)
        metrics = {
            "overall_task_success": weak,
            "slip_recovery_success": weak,
            "isolated_recovery_success": weak,
            "slip_regrasp_success": weak,
            "attached_task_success": negative,
            "failure_continuation_rate": negative,
            "premature_commitment_rate": negative,
        }
        comparisons = {
            "oracle_vs_fixed_k3": metrics,
            "oracle_vs_random_matched_commit": metrics,
            "oracle_vs_gripper_commit": metrics,
            "oracle_vs_self_consistency_commit": metrics,
        }
        aggregate = {
            "oracle_branch_safe_commit": {
                "overall_task_success": summary(0.20),
                "policy_forward_calls": summary(40.0),
            },
            "fixed_k1": {
                "overall_task_success": summary(0.20),
                "policy_forward_calls": summary(100.0),
            },
        }
        decision, _ = apply_decision_gate(aggregate, comparisons)
        self.assertEqual(decision, "STOP_FRESH_FAMILY")


if __name__ == "__main__":
    unittest.main()
