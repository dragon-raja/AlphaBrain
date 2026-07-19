import unittest

import numpy as np

from scripts.continuation_distill.build_and_audit_bank import robust_winner, source_split


class ContinuationDistillBankTest(unittest.TestCase):
    def test_source_split_matches_policy_response_gate(self):
        fitting, evaluation = source_split(range(23))
        self.assertEqual(len(evaluation), 5)
        self.assertFalse(set(fitting) & set(evaluation))

    def test_robust_winner_accepts_repeat_stable_gain(self):
        signatures = np.zeros((16, 6, 6), dtype=np.float64)
        signatures[3, :, 5] = 1.0
        profiles = signatures.mean(axis=1)
        row = robust_winner(signatures, profiles)
        self.assertTrue(row["accepted"])
        self.assertEqual(row["winner_index"], 3)
        self.assertEqual(row["strict_repeat_wins"], 6)
        self.assertEqual(row["leave_one_out_agreement"], 6)

    def test_robust_winner_rejects_repeat_instability(self):
        signatures = np.zeros((16, 6, 6), dtype=np.float64)
        signatures[2, :3, 5] = 1.0
        signatures[3, 3:, 5] = 1.0
        profiles = signatures.mean(axis=1)
        self.assertFalse(robust_winner(signatures, profiles)["accepted"])


if __name__ == "__main__":
    unittest.main()

