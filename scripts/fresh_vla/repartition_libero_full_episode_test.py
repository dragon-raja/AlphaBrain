import unittest

from repartition_libero_full_episode import source_disjoint_splits, split_quality


class RepartitionTest(unittest.TestCase):
    def test_source_states_are_disjoint_with_exact_group_counts(self) -> None:
        groups = []
        for source, count in enumerate((3, 2, 4, 3, 2, 1, 3, 2, 4, 6, 1, 2, 2, 1, 3, 2, 1, 1, 3, 2, 1, 1, 2, 1, 3, 2, 2, 1, 4, 3, 2, 3, 2, 3, 4, 3, 5, 4, 5, 4, 3, 5, 1, 2, 5, 4, 2, 3)):
            for index in range(count):
                groups.append(
                    {
                        "pair_id": f"source-{source}-group-{index}",
                        "source_initial_state_index": source,
                    }
                )
        self.assertEqual(len(groups), 128)
        splits = source_disjoint_splits(groups, seed=7)
        assigned = [{**group, "split": splits[group["pair_id"]]} for group in groups]
        quality = split_quality(assigned)
        self.assertTrue(quality["source_initial_state_disjoint"])
        self.assertEqual(quality["split_group_counts"], {"test": 13, "train": 102, "val": 13})

    def test_split_is_deterministic(self) -> None:
        groups = [
            {"pair_id": f"g{index}", "source_initial_state_index": index}
            for index in range(128)
        ]
        self.assertEqual(source_disjoint_splits(groups, seed=9), source_disjoint_splits(groups, seed=9))


if __name__ == "__main__":
    unittest.main()
