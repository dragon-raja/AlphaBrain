from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from explicit_flow_noise import ExplicitFlowNoiseBank, materialize_bank, tensor_sha256


class ExplicitFlowNoiseTest(unittest.TestCase):
    def test_materialized_tensor_is_exactly_repeatable_and_index_specific(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = materialize_bank(
                output_dir=root,
                bank_id="T",
                state_keys=["state-a", "state-b"],
                repeat_count=2,
                max_replans=3,
                action_horizon=5,
                action_dim=2,
                root_seed=17,
            )
            bank = ExplicitFlowNoiseBank(Path(result["manifest_path"]))
            first = bank.get("state-a", 0, 0)
            repeated = bank.get("state-a", 0, 0)
            other_repeat = bank.get("state-a", 1, 0)
            other_replan = bank.get("state-a", 0, 1)
            np.testing.assert_array_equal(first["noise"], repeated["noise"])
            self.assertEqual(first["noise_sha256"], repeated["noise_sha256"])
            self.assertNotEqual(first["noise_sha256"], other_repeat["noise_sha256"])
            self.assertNotEqual(first["noise_sha256"], other_replan["noise_sha256"])
            self.assertEqual(first["noise_sha256"], tensor_sha256(first["noise"]))

    def test_file_tampering_fails_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = materialize_bank(
                output_dir=root,
                bank_id="T",
                state_keys=["state-a"],
                repeat_count=1,
                max_replans=1,
                action_horizon=2,
                action_dim=2,
                root_seed=17,
            )
            manifest = json.loads(Path(result["manifest_path"]).read_text())
            with Path(manifest["noise_file"]).open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                ExplicitFlowNoiseBank(Path(result["manifest_path"]))


if __name__ == "__main__":
    unittest.main()
