import unittest

import numpy as np

from counterfactual_data import CounterfactualRecord
from generate_libero_training_data import (
    _rollout_array_payload,
    _validate_rollout_array_payload,
    assign_group_splits,
    build_training_labels,
    upright_image,
)


def make_record(pair_id: str, branch: str, oracle: int, *, deterministic: bool = False):
    return CounterfactualRecord(
        pair_id=pair_id,
        branch_id=branch,
        branch_outcome=branch,
        observation={"snapshot_key": pair_id},
        robot_state=[0.0] * 8,
        language_instruction="test",
        action_chunk=np.zeros((10, 7)).tolist(),
        event_time=oracle,
        feedback_reveal_time=oracle,
        action_divergence_time=oracle,
        gripper_transition_horizon=oracle + (branch == "slipped"),
        oracle_feedback_horizon=oracle,
        per_step_branch_divergence=[0.0] * 10,
        is_deterministic_control=deterministic,
    )


class LiberoTrainingDataTest(unittest.TestCase):
    def test_group_split_is_task_stratified_and_complete(self) -> None:
        pair_tasks = {
            **{f"grasp-{index}": "grasp_slip" for index in range(30)},
            **{f"reach-{index}": "deterministic_reach" for index in range(8)},
        }
        splits = assign_group_splits(pair_tasks, 17)

        self.assertEqual(set(splits), set(pair_tasks))
        self.assertEqual(sum(value == "val" for value in splits.values()), 4)
        self.assertEqual(sum(value == "test" for value in splits.values()), 4)

    def test_random_and_shuffled_labels_are_shared_by_pair(self) -> None:
        records = []
        splits = {}
        for index, oracle in enumerate((2, 3, 4, 2)):
            pair_id = f"pair-{index}"
            splits[pair_id] = "train"
            records.extend(
                [make_record(pair_id, "attached", oracle), make_record(pair_id, "slipped", oracle)]
            )
        labels = build_training_labels(records, splits, horizon=10, seed=5)["records"]

        for index in range(4):
            pair_id = f"pair-{index}"
            attached = labels[f"{pair_id}::attached"]
            slipped = labels[f"{pair_id}::slipped"]
            self.assertEqual(attached["random_feedback_horizon"], slipped["random_feedback_horizon"])
            self.assertEqual(attached["shuffled_oracle_horizon"], slipped["shuffled_oracle_horizon"])

    def test_deterministic_random_control_remains_full_horizon(self) -> None:
        records = []
        splits = {}
        for index in range(3):
            pair_id = f"control-{index}"
            splits[pair_id] = "train"
            records.extend(
                [
                    make_record(pair_id, "repeat_a", 10, deterministic=True),
                    make_record(pair_id, "repeat_b", 10, deterministic=True),
                ]
            )
        labels = build_training_labels(records, splits, horizon=10, seed=5)["records"]

        self.assertTrue(
            all(row["random_feedback_horizon"] == 10 for row in labels.values())
        )
        self.assertTrue(
            all(row["shuffled_oracle_horizon"] == 10 for row in labels.values())
        )

    def test_upright_image_rotates_both_axes(self) -> None:
        image = np.arange(12).reshape(2, 2, 3)
        np.testing.assert_array_equal(upright_image(image), image[::-1, ::-1])

    def test_rollout_payload_contains_every_frame_and_state(self) -> None:
        horizon = 2
        frames = [np.zeros((224, 448, 3), dtype=np.uint8) for _ in range(horizon + 1)]
        payload = _rollout_array_payload(
            {"attached": np.zeros((1, horizon, 7))},
            {
                "attached": [
                    {
                        "robot_state_trajectory": np.zeros((horizon + 1, 8)),
                        "object_trajectory": np.zeros((horizon + 1, 3)),
                    }
                ]
            },
            {"attached": [frames]},
            effect_key="object_trajectory",
        )

        _validate_rollout_array_payload(payload, horizon)
        self.assertEqual(len(payload), 5)
        self.assertEqual(payload["attached_repeat_00_agentview"].shape, (3, 224, 224, 3))
        self.assertEqual(payload["attached_repeat_00_robot_state"].shape, (3, 8))


if __name__ == "__main__":
    unittest.main()
