import unittest

from build_random_commit_boundaries import build_boundary_map


class RandomCommitBoundariesTest(unittest.TestCase):
    def test_preserves_runtime_boundary_multiset_and_is_outcome_blind(self):
        groups = [
            {"pair_id": "g0", "split": "test", "action_divergence_time": 4},
            {"pair_id": "g1", "split": "test", "action_divergence_time": 7},
            {"pair_id": "g2", "split": "test", "action_divergence_time": 9},
        ]
        rows = []
        for pair_id, event in (("g0", 5), ("g1", 8), ("g2", None)):
            for outcome in ("attached", "slipped"):
                rows.append(
                    {
                        "pair_id": pair_id,
                        "branch_outcome": outcome,
                        "execution_horizon": 3,
                        "event_time": event,
                        "completion_steps": 10,
                    }
                )
        payload = build_boundary_map(groups, rows, seed=41)
        self.assertCountEqual(payload["boundaries"].values(), [5, 8, None])
        self.assertEqual(set(payload["boundaries"]), {"g0", "g1", "g2"})
        self.assertTrue(
            all(
                boundary is None or boundary <= payload["target_completion_limits"][pair_id]
                for pair_id, boundary in payload["boundaries"].items()
            )
        )


if __name__ == "__main__":
    unittest.main()
