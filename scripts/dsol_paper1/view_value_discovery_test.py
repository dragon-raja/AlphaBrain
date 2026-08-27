from __future__ import annotations

import unittest

from analyze_view_value_discovery import analyze, pairwise_auc, random_hit_probability


def episode(pair_key: str, candidate: str, success: bool, visibility: float, accel: float):
    entities = {
        "target": {"visible_fraction": visibility, "visible_pixels": int(100 * visibility)},
        "container": {"visible_fraction": visibility / 2, "visible_pixels": int(50 * visibility)},
    }
    return {
        "episode_id": f"{pair_key}-{candidate}",
        "episode_id_source": f"source-{pair_key}",
        "pair_key": pair_key,
        "task_id": "task",
        "source_state_index": 3,
        "stage_fraction": 0.5,
        "selected_candidate_id": candidate,
        "pose": {"azimuth_deg": 0, "elevation_deg": 0, "radius_scale": 1},
        "selection_metadata": {
            "catalog_group": "canonical" if candidate == "canonical" else "heldout",
            "delta_visibility": visibility,
            "ensemble_accel_3": accel,
            "ensemble_accel_rank": 1,
            "single_noise_accel_3": accel,
        },
        "initial_metrics": {
            "task_entity_visibility": {
                "score": 0.75 * visibility,
                "camera_names": ["agentview", "robot0_eye_in_hand"],
                "entity_names": ["target", "container"],
                "per_camera": {
                    "agentview": {"score": visibility, "entities": entities},
                    "robot0_eye_in_hand": {"score": visibility / 2, "entities": entities},
                },
            }
        },
        "success": success,
        "completion_steps": 10,
    }


class ViewValueDiscoveryTest(unittest.TestCase):
    def test_pairwise_auc(self) -> None:
        self.assertEqual(pairwise_auc([2, 3], [False, True]), 1.0)
        self.assertEqual(pairwise_auc([2, 2], [False, True]), 0.5)
        self.assertIsNone(pairwise_auc([1, 2], [True, True]))

    def test_random_hit_probability(self) -> None:
        self.assertAlmostEqual(random_hit_probability(4, 1, 1), 0.25)
        self.assertAlmostEqual(random_hit_probability(4, 1, 2), 0.5)
        self.assertEqual(random_hit_probability(4, 0, 4), 0.0)

    def test_analysis_is_state_conditional(self) -> None:
        rows = [
            episode("a", "canonical", False, 0.1, 0.8),
            episode("a", "view", True, 0.9, 0.1),
            episode("b", "canonical", True, 0.8, 0.2),
            episode("b", "view", False, 0.2, 0.9),
        ]
        payload, flat = analyze(rows)
        self.assertEqual(payload["states"], 2)
        self.assertEqual(payload["candidate_count_per_state"], 2)
        self.assertEqual(payload["oracle_at_all_success_rate"], 1.0)
        self.assertEqual(payload["score_diagnostics"]["visibility_mean"]["state_conditional_auc"], 1.0)
        self.assertEqual(payload["score_diagnostics"]["accel_ensemble"]["state_conditional_auc"], 1.0)
        self.assertEqual(len(flat), 4)


if __name__ == "__main__":
    unittest.main()
