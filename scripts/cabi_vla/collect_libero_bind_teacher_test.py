from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from build_libero_bind_suite import build_manifest
from collect_libero_bind_teacher import (
    CALIBRATIONS,
    action_toward,
    parse_indices,
    select_edges,
)


class LiberoBindTeacherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = build_manifest(
            bddl_dir=Path("/tmp/bddl"),
            canonical_init_path=Path("/tmp/states.pruned_init"),
        )

    def test_parse_indices_supports_ranges_and_rejects_duplicates(self) -> None:
        self.assertEqual(parse_indices("0,3-5,49"), [0, 3, 4, 5, 49])
        with self.assertRaises(ValueError):
            parse_indices("3-1")
        with self.assertRaises(ValueError):
            parse_indices("1,1")
        with self.assertRaises(ValueError):
            parse_indices("50")

    def test_training_selection_rejects_withheld_actions(self) -> None:
        selected = select_edges(self.manifest, ["red-left"], purpose="train")
        self.assertEqual(selected[0]["edge_id"], "red-left")
        with self.assertRaises(ValueError):
            select_edges(self.manifest, ["white-right"], purpose="train")

    def test_action_toward_is_bounded(self) -> None:
        action = action_toward([0, 0, 0], [1, -1, 0.025], gripper=2)
        np.testing.assert_allclose(action, [1, -1, 0.5, 0, 0, 0, 1])

    def test_grasp_candidates_are_object_specific_not_edge_specific(self) -> None:
        red = CALIBRATIONS["red_coffee_mug_1"]
        white = CALIBRATIONS["porcelain_mug_1"]
        self.assertEqual(red.grasp_xy_offsets[0], (0.0, 0.0))
        self.assertGreater(len(red.grasp_xy_offsets), 1)
        self.assertEqual(white.grasp_height, 0.10)
        self.assertEqual(
            CALIBRATIONS["white_yellow_mug_1"].placement_xy_bias,
            (0.0, 0.01),
        )


if __name__ == "__main__":
    unittest.main()
