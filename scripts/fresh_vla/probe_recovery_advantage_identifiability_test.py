import unittest

import numpy as np

from evaluate_recovery_segment_oracle import METRICS
from probe_recovery_advantage_identifiability import (
    DecisionRecord,
    candidate_feature,
    source_loso_probe,
    strict_pairs,
    top_set,
    training_matrix,
)


def record(source, *, selection=(0,), heldout=(0,), history=True):
    return DecisionRecord(
        seed=41,
        pair_id=f"pair-{source}",
        source=str(source),
        replan_index=1 if history else 0,
        current_context=np.asarray([float(source + 1), 0.5]),
        candidate_actions=np.asarray([[1.0], [0.5], [-0.5], [-1.0]]),
        selection_top_set=selection,
        heldout_top_set=heldout,
        previous_context=(np.asarray([float(source), 0.25]) if history else None),
        previous_selected_action=(np.asarray([0.25]) if history else None),
    )


class RecoveryAdvantageIdentifiabilityTest(unittest.TestCase):
    def test_top_set_preserves_ties(self):
        metrics = np.zeros((4, len(METRICS)), dtype=np.float64)
        self.assertEqual(top_set(metrics), (0, 1, 2, 3))
        metrics[2, 0] = 1.0
        self.assertEqual(top_set(metrics), (2,))
        metrics[3, 0] = 1.0
        self.assertEqual(top_set(metrics), (2, 3))

    def test_strict_pairs_abstain_within_top_set(self):
        self.assertEqual(strict_pairs((0, 2)), [(0, 1), (0, 3), (2, 1), (2, 3)])
        self.assertEqual(strict_pairs((0, 1, 2, 3)), [])

    def test_training_matrix_drops_all_tied_decisions(self):
        tied = record(0, selection=(0, 1, 2, 3), heldout=(0, 1, 2, 3))
        useful = record(1, selection=(0,), heldout=(0,))
        features, labels = training_matrix(
            [tied, useful],
            mode="action",
            target="selection",
        )
        self.assertEqual(features.shape, (3, 1))
        self.assertEqual(labels.tolist(), [1.0, 1.0, 1.0])

    def test_history_feature_contains_action_conditioned_interactions(self):
        value = record(0)
        current = candidate_feature(value, 0, "current")
        history = candidate_feature(value, 0, "history")
        self.assertEqual(current.shape, (3,))
        self.assertEqual(history.shape, (6,))
        self.assertFalse(np.array_equal(history, candidate_feature(value, 1, "history")))

    def test_source_loso_reports_source_disjoint_predictions(self):
        records = [record(source) for source in range(4)]
        result = source_loso_probe(
            records,
            mode="action",
            require_history=True,
            train_target="heldout",
            seed=1,
        )
        self.assertTrue(result["source_disjoint_loso"])
        self.assertEqual(result["summary"]["decision_count"], 4)
        self.assertEqual(
            result["summary"]["unique_best_accuracy"]["source_cluster_level"]["mean"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
