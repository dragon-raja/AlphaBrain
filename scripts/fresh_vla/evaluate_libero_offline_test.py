import unittest

from evaluate_libero_offline import balanced_sample_indices


class BalancedSampleIndicesTest(unittest.TestCase):
    @staticmethod
    def _rows(group_sizes):
        rows = []
        for group_id, size in group_sizes.items():
            rows.extend(({"pair_id": group_id}, 10) for _ in range(size))
        return rows

    def test_round_robin_covers_groups_before_repeating(self):
        rows = self._rows({"a": 5, "b": 5, "c": 5})
        selected = balanced_sample_indices(rows, 6, seed=7)
        groups = [rows[index][0]["pair_id"] for index in selected]
        self.assertEqual(set(groups[:3]), {"a", "b", "c"})
        self.assertEqual(set(groups[3:]), {"a", "b", "c"})

    def test_selection_is_deterministic_and_unique(self):
        rows = self._rows({"a": 4, "b": 3, "c": 2})
        first = balanced_sample_indices(rows, 8, seed=19)
        second = balanced_sample_indices(rows, 8, seed=19)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_unbounded_selection_preserves_dataset_order(self):
        rows = self._rows({"a": 2, "b": 1})
        self.assertEqual(balanced_sample_indices(rows, None, seed=3), [0, 1, 2])
        self.assertEqual(balanced_sample_indices(rows, 10, seed=3), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
