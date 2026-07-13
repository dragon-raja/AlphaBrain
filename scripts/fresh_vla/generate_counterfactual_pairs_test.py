import unittest

from generate_counterfactual_pairs import generate_records


class GenerateCounterfactualPairsTest(unittest.TestCase):
    def test_generator_covers_branching_and_controls(self) -> None:
        records, metadata = generate_records(pairs_per_task=2, repeats=3, horizon=12, seed=5)
        self.assertEqual(len(records), 16)
        self.assertEqual(len(metadata), 8)
        by_task = {}
        for record in records:
            by_task.setdefault(record.observation["task"], []).append(record)

        self.assertTrue(all(row.oracle_feedback_horizon < 12 for row in by_task["grasp"]))
        self.assertTrue(all(row.oracle_feedback_horizon < 12 for row in by_task["blocked_push"]))
        self.assertTrue(all(row.gripper_transition_horizon == 12 for row in by_task["blocked_push"]))
        self.assertTrue(all(row.oracle_feedback_horizon == 12 for row in by_task["deterministic_reach"]))
        self.assertTrue(all(row.oracle_feedback_horizon == 12 for row in by_task["intent_control"]))

    def test_pair_conditioning_is_identical_across_outcomes(self) -> None:
        records, _ = generate_records(pairs_per_task=1, repeats=3, horizon=12, seed=8)
        grouped = {}
        for record in records:
            grouped.setdefault(record.pair_id, []).append(record)
        for pair in grouped.values():
            conditioning = {
                (str(row.observation), tuple(row.robot_state), row.language_instruction)
                for row in pair
            }
            self.assertEqual(len(conditioning), 1)


if __name__ == "__main__":
    unittest.main()
