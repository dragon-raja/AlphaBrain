import copy
import json
import unittest
from pathlib import Path

from scripts.dsol_paper1.audit_view_value_expectation_protocol import audit


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/dsol_paper1/view_value_expectation_protocol_v1.json"


class ViewValueExpectationProtocolAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(PROTOCOL.read_text())

    def test_frozen_protocol_passes_and_budget_is_deterministic(self) -> None:
        result = audit(self.payload)
        self.assertEqual(result["status"], "PASS_STATIC_DESIGN_RUNNER_STILL_HOLD")
        self.assertEqual(result["calibration_projected_episodes"], 12864)
        self.assertEqual(result["heldout_primary_projected_episodes"], 15360)
        self.assertEqual(result["projected_primary_total_episodes"], 28224)
        self.assertFalse(result["formal_execution_authorized"])

    def test_seed_coupling_fails_closed(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["randomness_contract"]["environment_seed"]["must_not_be_derived_from_policy_noise_seed"] = False
        with self.assertRaisesRegex(ValueError, "separate"):
            audit(payload)

    def test_posthoc_candidate_replacement_fails_closed(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["calibration_design"]["posthoc_candidate_replacement_after_confirmation"] = True
        with self.assertRaisesRegex(ValueError, "posthoc"):
            audit(payload)


if __name__ == "__main__":
    unittest.main()
