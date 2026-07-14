import unittest

from summarize_libero_episode_offline import aggregate_seeds, group_averages, summarize_groups


class EpisodeOfflineSummaryTest(unittest.TestCase):
    def test_group_averages_windows_before_statistics(self):
        rows = [
            {"pair_id": "g0", "m": 1.0},
            {"pair_id": "g0", "m": 3.0},
            {"pair_id": "g1", "m": 10.0},
        ]
        groups = group_averages(rows, {"metric": "m"})
        self.assertEqual(groups["g0"]["metric"], 2.0)
        self.assertEqual(summarize_groups(groups, ("metric",))["metric"], 6.0)

    def test_group_averages_preserves_missing_suffix(self):
        groups = group_averages([{"pair_id": "g0", "suffix": None}], {"suffix": "suffix"})
        self.assertIsNone(groups["g0"]["suffix"])

    def test_seed_aggregate_does_not_pool_windows(self):
        result = aggregate_seeds([{"m": 1.0}, {"m": 3.0}], ("m",))
        self.assertEqual(result["m"]["mean"], 2.0)
        self.assertEqual(result["m"]["seed_values"], [1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
