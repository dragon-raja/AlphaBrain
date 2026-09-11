from __future__ import annotations

import unittest

import numpy as np

from scripts.verify_vla.probe_gate_common import (
    block_average,
    detach_offsets,
    probe_actions,
    ridge_scores,
    summarize_predictions,
)


class ProbeGateCommonTest(unittest.TestCase):
    def test_frozen_probe_actions(self) -> None:
        self.assertEqual(probe_actions("micro_lift").shape, (4, 7))
        np.testing.assert_allclose(probe_actions("micro_lift")[:, 2], 0.25)
        np.testing.assert_allclose(probe_actions("micro_lateral")[:, 0], [0.25, -0.25, 0.25, -0.25])
        np.testing.assert_allclose(probe_actions("release")[:, -1], -1.0)
        with self.assertRaises(ValueError):
            probe_actions("micro_lift", steps=5)

    def test_detach_grid_is_frozen_and_normalized(self) -> None:
        offsets = detach_offsets([3.0, 4.0])
        self.assertEqual(len(offsets), 12)
        self.assertAlmostEqual(float(np.linalg.norm(offsets[0])), 0.00025)
        self.assertAlmostEqual(float(np.linalg.norm(offsets[-1])), 0.003)

    def test_block_average_and_ridge(self) -> None:
        image = np.full((32, 32, 3), 255, dtype=np.uint8)
        pooled = block_average(image, size=16)
        self.assertEqual(pooled.shape, (16 * 16 * 3,))
        np.testing.assert_allclose(pooled, 1.0)
        train = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        labels = np.asarray([-1.0, -1.0, 1.0, 1.0])
        scores = ridge_scores(train, labels, np.asarray([[-3.0], [3.0]]), 1.0)
        self.assertLess(scores[0], 0.0)
        self.assertGreater(scores[1], 0.0)

    def test_prediction_summary_keeps_pairs_as_units(self) -> None:
        metadata = [
            {"pair_id": "a", "outcome": "attached"},
            {"pair_id": "a", "outcome": "detached"},
            {"pair_id": "b", "outcome": "attached"},
            {"pair_id": "b", "outcome": "detached"},
        ]
        labels = np.asarray([1.0, -1.0, 1.0, -1.0])
        summary = summarize_predictions(
            metadata,
            labels,
            np.asarray([2.0, -2.0, 1.0, -1.0]),
            bootstrap_samples=100,
            seed=1,
        )
        self.assertEqual(summary["group_count"], 2)
        self.assertEqual(summary["sample_accuracy"], 1.0)
        self.assertEqual(summary["pair_ranking_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
