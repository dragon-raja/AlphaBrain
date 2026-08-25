from __future__ import annotations

import unittest

from build_accel_gate_a97_protocol import selection_for_state
from summarize_accel_gate_a97 import summarize_shortlist


class GateA97ProtocolTest(unittest.TestCase):
    def test_selection_separates_single_ensemble_visibility_and_hybrid(self) -> None:
        candidate_ids = ["canonical"] + [f"view_{index:02d}" for index in range(96)]
        metadata = {
            candidate_id: {
                "visibility_score": index / 100,
                "delta_visibility": index / 100,
                "catalog_group": "canonical" if index == 0 else "broad_training_64",
            }
            for index, candidate_id in enumerate(candidate_ids)
        }
        first = {candidate_id: 10.0 + index for index, candidate_id in enumerate(candidate_ids)}
        second = dict(first)
        first["view_01"] = 0.0
        second["view_01"] = 20.0
        first["view_02"] = 2.0
        second["view_02"] = 2.0
        first["view_95"] = 3.0
        second["view_95"] = 3.0

        selected, annotations = selection_for_state(
            pair_key="state",
            metadata=metadata,
            noise_runs=[{"state": first}, {"state": second}],
            random_seed=7,
        )

        self.assertEqual(selected["accel_single_noise"], "view_01")
        self.assertEqual(selected["accel_ensemble"], "view_02")
        self.assertEqual(selected["visibility_top1"], "view_95")
        self.assertIn(selected["accel_top10_visibility"], annotations)
        self.assertEqual(len(annotations), 97)

    def test_shortlist_summary_uses_source_group_bootstrap(self) -> None:
        conditions = ("canonical", "accel_ensemble")
        rows = []
        for group, canonical, ensemble in (("a", False, True), ("b", True, True)):
            for condition, success in zip(conditions, (canonical, ensemble)):
                rows.append(
                    {
                        "episode_id": f"{group}-{condition}",
                        "episode_id_source": group,
                        "pair_key": group,
                        "condition": condition,
                        "status": "complete",
                        "success": success,
                        "completion_steps": 10,
                    }
                )
        result = summarize_shortlist(
            rows,
            {"episode_count": 4},
        )
        comparison = result["paired_source_bootstrap"]["accel_ensemble_vs_canonical"]
        self.assertEqual(comparison["source_episode_groups"], 2)
        self.assertAlmostEqual(comparison["difference_pp"], 50.0)
        self.assertEqual(result["oracle_at_shortlist_state_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
