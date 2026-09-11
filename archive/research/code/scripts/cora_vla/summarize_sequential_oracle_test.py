from __future__ import annotations

import unittest

from summarize_sequential_oracle import paired_bootstrap


class SummarizeSequentialOracleTest(unittest.TestCase):
    def test_paired_bootstrap_constant_difference(self) -> None:
        result = paired_bootstrap([0.25] * 13)
        self.assertEqual(result["mean"], 0.25)
        self.assertEqual(result["ci95_low"], 0.25)
        self.assertEqual(result["ci95_high"], 0.25)


if __name__ == "__main__":
    unittest.main()
