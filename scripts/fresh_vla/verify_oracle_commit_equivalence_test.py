import unittest

from verify_oracle_commit_equivalence import semantic_row


class OracleEquivalenceTest(unittest.TestCase):
    def test_ignores_method_specific_labels_and_timing_only(self):
        common = {
            "pair_id": "g0",
            "success": True,
            "commit_trace": [{"commit_length": 2, "boundary_step": 5, "source": "a"}],
        }
        left = {**common, "commit_method": "oracle_branch_safe_commit", "inference_wall_seconds": 1.0}
        right = {
            **common,
            "commit_method": "oracle_feedback_reveal_commit",
            "inference_wall_seconds": 2.0,
            "commit_trace": [{"commit_length": 2, "boundary_step": 5, "source": "b"}],
        }
        self.assertEqual(semantic_row(left), semantic_row(right))
        right["success"] = False
        self.assertNotEqual(semantic_row(left), semantic_row(right))


if __name__ == "__main__":
    unittest.main()
