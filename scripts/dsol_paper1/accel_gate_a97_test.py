from __future__ import annotations

import unittest

from build_accel_gate_a97_protocol import build, selection_for_state
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

    def test_targeted_oracle_uses_requested_pair_keys(self) -> None:
        candidate_ids = ["canonical"] + [f"view_{index:02d}" for index in range(96)]
        metadata = {
            pair_key: {
                candidate_id: {
                    "visibility_score": index / 100,
                    "delta_visibility": index / 100,
                    "catalog_group": "canonical" if index == 0 else "broad_training_64",
                }
                for index, candidate_id in enumerate(candidate_ids)
            }
            for pair_key in ("state-a", "state-b")
        }
        scores = {
            pair_key: {candidate_id: float(index) for index, candidate_id in enumerate(candidate_ids)}
            for pair_key in metadata
        }
        base_protocol = {
            "catalog": "/tmp/catalog.json",
            "selected_state_count": 2,
            "specs": [
                {
                    "pair_key": pair_key,
                    "task_id": "task",
                    "episode_id_source": f"episode-{pair_key}",
                    "source_state_index": 0,
                    "stage_fraction": 0.5,
                }
                for pair_key in metadata
            ],
        }
        catalog = {
            "poses": [
                {"pose_id": candidate_id}
                for candidate_id in candidate_ids
                if candidate_id != "canonical"
            ]
        }

        result = build(
            base_protocol=base_protocol,
            catalog=catalog,
            render_metadata=metadata,
            noise_runs=[scores, scores],
            model="model",
            mode="oracle",
            random_seed=7,
            oracle_states_per_task=1,
            oracle_pair_keys=["state-b"],
        )

        self.assertEqual(result["analysis_role"], "targeted_97_view_exhaustive_oracle")
        self.assertEqual(result["selected_state_count"], 1)
        self.assertEqual(result["episode_count"], 97)
        self.assertEqual(result["selected_states"][0]["pair_key"], "state-b")

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
