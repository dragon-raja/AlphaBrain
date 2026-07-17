import unittest

from audit_candidate_labels import balanced_examples, confusion


class CandidateLabelAuditTest(unittest.TestCase):
    def test_confusion_uses_physical_as_audit_reference(self):
        rows = [
            {"x": True, "physical_compatible": True},
            {"x": True, "physical_compatible": False},
            {"x": False, "physical_compatible": True},
            {"x": False, "physical_compatible": False},
        ]
        self.assertEqual(confusion(rows, "x")["agreement"], 0.5)

    def test_examples_round_robin_across_seed_group(self):
        rows = []
        for seed in (41, 42):
            for index in range(3):
                rows.append(
                    {
                        "checkpoint_seed": seed,
                        "pair_id": f"g{seed}",
                        "candidate_index": index,
                        "joint_compatible": False,
                        "physical_compatible": True,
                    }
                )
        selected = balanced_examples(rows, 4)
        self.assertEqual([row["candidate_index"] for row in selected], [0, 0, 1, 1])
        self.assertTrue(all(row["visualization_reason"] == "joint_false_physical_success" for row in selected))

    def test_examples_fall_back_without_mislabeling(self):
        rows = [
            {
                "checkpoint_seed": 41,
                "pair_id": "g0",
                "candidate_index": index,
                "joint_compatible": index != 0,
                "physical_compatible": True,
            }
            for index in range(3)
        ]
        selected = balanced_examples(rows, 3)
        self.assertEqual(
            [row["visualization_reason"] for row in selected],
            ["joint_false_physical_success", "physical_success_context", "physical_success_context"],
        )


if __name__ == "__main__":
    unittest.main()
