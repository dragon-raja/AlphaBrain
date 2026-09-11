import unittest

from finalize_embodied_research_reset import SELECTED_PROBLEM, choose_problem


class FinalizeResearchResetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = {
            "stage_a_stopped": True,
            "feedback_observable": True,
            "local_mode_availability": 1.0,
            "supports_mode_selection_bottleneck": False,
            "supports_prompt_grounded_recovery": False,
            "isolated_regrasp_rate": 0.62,
            "regrasp_to_transport_rate": 0.25,
        }

    def test_selects_narrow_problem_when_all_evidence_agrees(self) -> None:
        self.assertEqual(
            choose_problem(self.evidence),
            f"SELECT_NEW_RESEARCH_PROBLEM: {SELECTED_PROBLEM}",
        )

    def test_refuses_selection_when_candidate_composition_gap_is_absent(self) -> None:
        self.evidence["regrasp_to_transport_rate"] = 0.8
        self.assertEqual(choose_problem(self.evidence), "NO_VALID_RESEARCH_PROBLEM_YET")


if __name__ == "__main__":
    unittest.main()
