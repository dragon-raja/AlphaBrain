from __future__ import annotations

from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
import importlib.util
from pathlib import Path
import unittest


SCRIPT = REPOSITORY_ROOT / "scripts/dsol_paper1/operations/controllers/run_protocol_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_protocol_smoke", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProtocolSmokeTests(unittest.TestCase):
    def test_smoke_passes_and_never_claims_formal_eligibility(self) -> None:
        receipt = MODULE.run_smoke()

        self.assertEqual(receipt["status"], "DEBUG_PROTOCOL_SMOKE_PASS")
        self.assertIs(receipt["formal_eligible"], False)
        self.assertTrue(all(value is False for value in receipt["operations"].values()))

    def test_output_must_be_debug_scoped(self) -> None:
        with self.assertRaises(ValueError):
            MODULE._debug_output(Path("/tmp/dsol-receipt.json"))


if __name__ == "__main__":
    unittest.main()
