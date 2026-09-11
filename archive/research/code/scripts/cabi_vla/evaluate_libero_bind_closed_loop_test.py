from __future__ import annotations

import unittest

import numpy as np

from evaluate_libero_bind_closed_loop import (
    parse_state_indices,
    stable_seed,
    subgoal_state,
)


class LiberoBindClosedLoopTest(unittest.TestCase):
    def test_stable_seed_is_reproducible_and_keyed(self) -> None:
        self.assertEqual(stable_seed("a", 1), stable_seed("a", 1))
        self.assertNotEqual(stable_seed("a", 1), stable_seed("a", 2))

    def test_subgoal_state_uses_requested_source_and_target(self) -> None:
        observation = {
            "mug_pos": np.asarray([0.01, 0.02, 0.20]),
            "plate_pos": np.asarray([0.02, 0.03, 0.10]),
        }
        state = subgoal_state(
            observation,
            source_object="mug",
            target_object="plate",
            initial_source_z=0.10,
            source_grasped=True,
            wrong_source_grasped=False,
        )
        self.assertTrue(state["source_grasp"])
        self.assertTrue(state["lift"])
        self.assertTrue(state["transport"])
        self.assertFalse(state["wrong_source_grasp"])

    def test_state_split_uses_canonical_index_schema(self) -> None:
        manifest = {
            "states": [
                {"canonical_state_index": 35, "split": "val"},
                {"canonical_state_index": 40, "split": "test"},
            ]
        }
        self.assertEqual(parse_state_indices(None, manifest, "val"), [35])


if __name__ == "__main__":
    unittest.main()
