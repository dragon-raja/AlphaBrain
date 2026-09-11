from __future__ import annotations

import unittest

from scripts.verify_vla.probe_gate_common import MODALITIES, NON_RELEASE_PROBES, PROBE_NAMES
from scripts.verify_vla.summarize_probe_gate import decide_gate


def classifier_fixture() -> dict:
    result = {}
    for condition in ("pre", *PROBE_NAMES):
        result[condition] = {}
        for modality in MODALITIES:
            accuracy = 0.5 if condition in {"pre", "hold_closed"} else 0.9
            result[condition][modality] = {
                "sample_accuracy": accuracy,
                "pair_ranking_accuracy": 0.9 if condition not in {"pre", "hold_closed"} else 0.5,
            }
    return result


def viability_fixture() -> dict:
    result = {}
    for condition in ("no_probe", *PROBE_NAMES):
        result[condition] = {
            "attached": {"teacher_success": {"mean": 1.0}},
            "detached": {"teacher_success": {"mean": 0.9}},
            "combined_object_displacement_median_m": 0.01,
            "combined_new_irreversible_failure_count": 0,
        }
    return result


class GateDecisionTest(unittest.TestCase):
    def test_proceeds_when_a_probe_clears_all_frozen_checks(self) -> None:
        decision, details = decide_gate(
            classifier_fixture(),
            viability_fixture(),
            {
                "train": {"valid": 100, "valid_fraction": 0.98},
                "val": {"valid": 13, "valid_fraction": 1.0},
            },
        )
        self.assertEqual(decision, "PROCEED_TO_LEARNED_DVOV")
        self.assertEqual(set(details["passing_probes"]), set(NON_RELEASE_PROBES) - {"hold_closed"})

    def test_invalid_teacher_baseline_cannot_be_method_failure(self) -> None:
        viability = viability_fixture()
        viability["no_probe"]["detached"]["teacher_success"]["mean"] = 0.2
        decision, _ = decide_gate(
            classifier_fixture(),
            viability,
            {
                "train": {"valid": 100, "valid_fraction": 0.98},
                "val": {"valid": 13, "valid_fraction": 1.0},
            },
        )
        self.assertEqual(decision, "GATE0_INVALID")


if __name__ == "__main__":
    unittest.main()
