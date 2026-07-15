import unittest

from pi05_policy_server import validate_policy_example


class PolicyServerInputTest(unittest.TestCase):
    def test_rejects_oracle_or_branch_metadata(self):
        valid = {"image": [], "lang": "task", "language": "task", "state": []}
        validate_policy_example(valid)
        with self.assertRaises(ValueError):
            validate_policy_example({**valid, "branch_outcome": "slipped"})
        with self.assertRaises(ValueError):
            validate_policy_example({**valid, "action_divergence_time": 4})


if __name__ == "__main__":
    unittest.main()
