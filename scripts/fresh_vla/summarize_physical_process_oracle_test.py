import unittest

from summarize_physical_process_oracle import build_summary


def outcome(*, success=False, next_stage=False, progress=0.0):
    return {
        "success": success,
        "next_stage_reached": next_stage,
        "transport_reached": next_stage,
        "lift_reached": next_stage,
        "stable_grasp_at_end": next_stage,
        "drop": False,
        "regress": False,
        "progress_auc": progress,
        "object_to_bowl_progress": progress,
        "object_height_progress": progress,
    }


def selection_summary(value):
    binary = (
        "success",
        "next_stage_reached",
        "transport_reached",
        "lift_reached",
        "stable_grasp_at_end",
        "drop",
        "regress",
    )
    continuous = ("progress_auc", "object_to_bowl_progress", "object_height_progress")
    return {
        **{f"{key}_rate": float(value[key]) for key in binary},
        **{key: float(value[key]) for key in continuous},
    }


class PhysicalProcessSummaryTest(unittest.TestCase):
    def test_aggregates_seeds_at_group_level(self):
        rows = []
        for seed in (41, 42):
            rows.append(
                {
                    "pair_id": "g0",
                    "source_initial_state_index": 7,
                    "seed": seed,
                    "stage": "feedback",
                    "eligible": True,
                    "oracle_index": 1,
                    "oracle_replay_match": True,
                    "unique_outcome_signatures": 2,
                    "candidates": [
                        {"selection_summary": selection_summary(outcome())},
                        {
                            "selection_summary": selection_summary(
                                outcome(success=True, next_stage=True, progress=1.0)
                            )
                        },
                    ],
                    "heldout_continuations": [
                        {
                            "sample0": {"bridge": outcome()},
                            "oracle": {"bridge": outcome(success=True, next_stage=True, progress=1.0)},
                        }
                    ],
                }
            )
        summary = build_summary(rows, bootstrap_samples=100, seed=7)
        gain = summary["paired_oracle_vs_sample0"]["feedback"]["success"]
        self.assertEqual(gain["source_cluster_level"]["count"], 1)
        self.assertEqual(gain["source_cluster_level"]["mean"], 1.0)
        self.assertEqual(gain["per_seed_mean"], {"41": 1.0, "42": 1.0})
        heldout = summary["paired_heldout_oracle_vs_sample0"]["feedback"]["success"]
        self.assertEqual(heldout["source_cluster_level"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
