from __future__ import annotations

import unittest

import numpy as np

from build_libero_bind_training_view import (
    anchor_conditioning_is_identical,
    padded_action_chunk,
    supervised_collection_rows,
    teacher_edge_quality,
    transport_anchor_index,
)


class LiberoBindTrainingViewTest(unittest.TestCase):
    def test_anchor_conditioning_requires_exact_images_and_state(self) -> None:
        anchors = {}
        for edge in ("left", "right"):
            prefix = f"{edge}__state_03__"
            anchors[prefix + "agentview"] = np.zeros((2, 2, 3), np.uint8)
            anchors[prefix + "wrist"] = np.zeros((2, 2, 3), np.uint8)
            anchors[prefix + "state"] = np.zeros((8,), np.float32)
        self.assertTrue(anchor_conditioning_is_identical(anchors, "left", "right", 3))
        anchors["right__state_03__state"][0] = 1e-6
        self.assertFalse(anchor_conditioning_is_identical(anchors, "left", "right", 3))

    def test_action_chunk_zero_pads_without_future_leakage(self) -> None:
        actions = np.arange(21, dtype=np.float32).reshape(3, 7)
        chunk = padded_action_chunk(actions, 2, 4)
        np.testing.assert_array_equal(chunk[0], actions[2])
        np.testing.assert_array_equal(chunk[1:], np.zeros((3, 7), np.float32))

    def test_transport_anchor_is_last_pre_transport_observation(self) -> None:
        phases = np.asarray(["lift", "lift", "transport", "transport"])
        self.assertEqual(transport_anchor_index(phases), 1)

    def test_transport_anchor_requires_transport_phase(self) -> None:
        with self.assertRaises(ValueError):
            transport_anchor_index(np.asarray(["lift", "lift"]))

    def test_teacher_quality_is_computed_per_edge(self) -> None:
        rows = [
            {"edge_id": "a", "success": True},
            {"edge_id": "a", "success": False},
            {"edge_id": "b", "success": True},
        ]
        quality = teacher_edge_quality(rows)
        self.assertEqual(quality["a"]["success_rate"], 0.5)
        self.assertEqual(quality["b"]["successful"], 1)

    def test_failed_row_without_flag_remains_in_quality_denominator(self) -> None:
        edges = {
            "observed": {"action_supervised": True},
            "withheld": {"action_supervised": False},
        }
        rows = [
            {"edge_id": "observed", "success": True, "action_supervised": True},
            {"edge_id": "observed", "success": False},
            {"edge_id": "withheld", "success": False},
        ]
        selected = supervised_collection_rows(rows, edges)
        quality = teacher_edge_quality(selected)
        self.assertEqual(quality["observed"]["total"], 2)
        self.assertEqual(quality["observed"]["success_rate"], 0.5)

    def test_supervision_mismatch_is_rejected(self) -> None:
        edges = {"observed": {"action_supervised": True}}
        with self.assertRaises(ValueError):
            supervised_collection_rows(
                [{"edge_id": "observed", "action_supervised": False}], edges
            )


if __name__ == "__main__":
    unittest.main()
