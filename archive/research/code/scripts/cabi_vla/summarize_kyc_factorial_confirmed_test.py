from __future__ import annotations

import unittest
from pathlib import Path

from summarize_kyc_factorial_confirmed import scaling_evaluation_path


class SummarizeKycFactorialConfirmedTest(unittest.TestCase):
    def test_scaling_evaluation_path(self) -> None:
        path = scaling_evaluation_path(
            Path("/tmp/eval"),
            budget=10,
            arm="kyc",
            seed=42,
        )
        self.assertEqual(
            path.parent.name,
            "n10-kyc-s42-fixed-wrist-on",
        )


if __name__ == "__main__":
    unittest.main()
