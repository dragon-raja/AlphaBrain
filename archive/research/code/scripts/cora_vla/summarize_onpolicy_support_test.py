import unittest

from summarize_onpolicy_support import aggregate


def payload(seed, recall):
    rows = [
        {
            "pair_id": f"g{i}",
            "stage": "feedback_reveal",
            "immediate_recall": {str(n): recall for n in (1, 4, 8, 16)},
            "teacher_recoverable_recall": {str(n): True for n in (1, 4, 8, 16)},
        }
        for i in range(3)
    ]
    return {"seed": seed, "rows": rows, "episodes": [{"success": False}] * 3}


class OnPolicySummaryTest(unittest.TestCase):
    def test_gate_passes_at_sixty_percent(self):
        result = aggregate([payload(41, True), payload(42, True), payload(43, False)])
        self.assertEqual(result["decision"], "PASS_ONPOLICY_SUPPORT")

    def test_gate_stops_below_threshold(self):
        result = aggregate([payload(41, False), payload(42, False), payload(43, True)])
        self.assertEqual(result["decision"], "BASE_POLICY_ONPOLICY_SUPPORT_INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
