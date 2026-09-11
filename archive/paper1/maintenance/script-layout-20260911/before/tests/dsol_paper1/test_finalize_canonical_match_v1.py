import copy
import tempfile
import unittest
from pathlib import Path

from scripts.dsol_paper1.finalize_canonical_match_v1 import finalize, inspect_weight_header, validate_metrics


class FinalizeCanonicalMatchTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"step": 1, "examples_seen": 32, "action_dit_loss": 1.0, "learning_rate": 1e-5},
            {"step": 2, "examples_seen": 64, "action_dit_loss": 0.5, "learning_rate": 5e-6},
        ]

    def test_valid_metrics(self):
        result = validate_metrics(self.rows, self.rows, 2, 32)
        self.assertEqual(result["examples_seen"], 64)

    def test_rejects_missing_or_duplicate_steps(self):
        for rows in (self.rows[:1], [self.rows[0], self.rows[0]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_metrics(rows, self.rows, 2, 32)

    def test_rejects_exposure_mismatch(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["examples_seen"] = 31
        with self.assertRaises(ValueError):
            validate_metrics(rows, self.rows, 2, 32)

    def test_rejects_nonfinite_loss(self):
        for value in (float("nan"), float("inf")):
            rows = copy.deepcopy(self.rows)
            rows[0]["action_dit_loss"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_metrics(rows, self.rows, 2, 32)

    def test_rejects_early_lr_mismatch(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["learning_rate"] = 1e-6
        with self.assertRaises(ValueError):
            validate_metrics(rows, self.rows, 2, 32)

    def test_refuses_existing_receipt_before_reading_or_hashing(self):
        with tempfile.TemporaryDirectory(prefix="canonical-finalizer-test-") as directory:
            path = Path(directory)
            (path / "completed_run_receipt.json").touch()
            with self.assertRaises(FileExistsError):
                finalize(path, path / "missing-release.json")

    def test_rejects_empty_weight(self):
        with tempfile.TemporaryDirectory(prefix="canonical-header-test-") as directory:
            path = Path(directory) / "empty.safetensors"
            path.touch()
            with self.assertRaises(ValueError):
                inspect_weight_header(path)


if __name__ == "__main__":
    unittest.main()
