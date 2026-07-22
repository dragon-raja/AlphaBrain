from __future__ import annotations

import unittest

from compare_libero_bind_policies import compare_payloads, paired_state_bootstrap


def payload(method_gain: bool) -> dict:
    rows = []
    for state in range(5):
        for edge, supervised in (("id", True), ("ood-a", False), ("ood-b", False)):
            baseline = edge == "id"
            success = baseline or (method_gain and not supervised)
            rows.append(
                {
                    "edge_id": edge,
                    "canonical_state_index": state,
                    "execution_horizon": 3,
                    "action_supervised": supervised,
                    "success": success,
                    "source_selection_success": success,
                    "target_placement_success": success,
                    "wrong_source_grasp": False,
                    "lift_success": success,
                    "transport_success": success,
                    "progress": float(success),
                }
            )
    return {"status": "complete", "rows": rows, "policy_identity": {"name": method_gain}}


class CompareLiberoBindPoliciesTest(unittest.TestCase):
    def test_paired_bootstrap_uses_state_differences(self) -> None:
        result = paired_state_bootstrap({0: 0.0, 1: 1.0}, {0: 1.0, 1: 1.0}, samples=100)
        self.assertEqual(result["difference"], 0.5)
        self.assertEqual(result["state_count"], 2)

    def test_clear_ood_gain_advances_to_controls(self) -> None:
        result = compare_payloads(payload(False), payload(True), bootstrap_samples=100)
        self.assertEqual(result["pilot_decision"], "ADVANCE_TO_FULL_CONTROLS")
        self.assertEqual(result["decision_horizon"], 3)

    def test_missing_preregistered_decision_horizon_is_rejected(self) -> None:
        baseline = payload(False)
        method = payload(True)
        for row in baseline["rows"] + method["rows"]:
            row["execution_horizon"] = 1
        with self.assertRaises(ValueError):
            compare_payloads(baseline, method, bootstrap_samples=20)


if __name__ == "__main__":
    unittest.main()
