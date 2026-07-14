import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from AlphaBrain.dataloader import _seeded_shuffle_options
from AlphaBrain.dataloader.paligemma_datasets import (
    FreshEpisodeWindowDataset,
    FreshSnapshotDataset,
    Pi0DataConfig,
    Pi0DataTransform,
)


class Pi0DataTransformTest(unittest.TestCase):
    def test_preserves_feedback_horizon(self) -> None:
        transform = Pi0DataTransform(Pi0DataConfig(action_horizon=2))
        sample = {
            "action": np.zeros((2, 7), dtype=np.float32),
            "feedback_horizon": 1,
        }

        result = transform(sample)

        self.assertEqual(result["feedback_horizon"], 1)


class FreshSnapshotDatasetTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        pair_id = "libero-grasp-slip-000"
        records = []
        labels = {}
        for branch in ("attached", "slipped"):
            record_id = f"{pair_id}::{branch}"
            records.append(
                {
                    "pair_id": pair_id,
                    "branch_id": branch,
                    "oracle_feedback_horizon": 2,
                    "observation": {"snapshot_key": pair_id},
                    "language_instruction": "put the cream cheese in the bowl",
                    "action_chunk": np.zeros((4, 7)).tolist(),
                    "robot_state": np.zeros(8).tolist(),
                }
            )
            labels[record_id] = {"oracle_feedback_horizon": 2, "full_h": 4}
        (root / "records.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records)
        )
        (root / "manifest.json").write_text(
            json.dumps({"pairs": [{"pair_id": pair_id, "task": "grasp_slip"}]})
        )
        (root / "splits.json").write_text(json.dumps({"pair_splits": {pair_id: "train"}}))
        (root / "training_labels.json").write_text(json.dumps({"records": labels}))
        np.savez_compressed(
            root / "policy_observation_snapshots.npz",
            **{
                f"{pair_id}_agentview": np.zeros((224, 224, 3), dtype=np.uint8),
                f"{pair_id}_wrist": np.ones((224, 224, 3), dtype=np.uint8),
            },
        )

    def test_loads_both_branches_from_one_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_fixture(root)
            dataset = FreshSnapshotDataset(root)

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["oracle_feedback_horizon"], 2)
            self.assertEqual(dataset[0]["fresh_sample_id"], "libero-grasp-slip-000::attached")
            self.assertEqual(dataset[0]["task"], "grasp_slip")
            self.assertEqual(dataset[0]["image"][0].shape, (224, 224, 3))
            self.assertTrue(np.array_equal(dataset[0]["image"][0], dataset[1]["image"][0]))

    def test_rejects_split_without_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_fixture(root)
            with self.assertRaisesRegex(ValueError, "no FRESH snapshot records"):
                FreshSnapshotDataset(root, split="test")


class FreshEpisodeWindowDatasetTest(unittest.TestCase):
    def test_loads_images_and_keeps_oracle_separate_from_selected_label(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / "agent.jpg")
            Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8)).save(root / "wrist.jpg")
            row = {
                "sample_id": "pair::attached::0000",
                "pair_id": "pair",
                "branch_id": "attached",
                "branch_outcome": "attached",
                "frame_index": 0,
                "task": "grasp_slip_full_episode",
                "split": "train",
                "observation": {"agentview_path": "agent.jpg", "wrist_path": "wrist.jpg"},
                "oracle_feedback_horizon": 3,
                "language_instruction": "put the object in the bowl",
                "action_chunk": np.zeros((10, 7)).tolist(),
                "robot_state": np.zeros(8).tolist(),
            }
            (root / "records.jsonl").write_text(json.dumps(row) + "\n")
            (root / "training_labels.json").write_text(
                json.dumps({"records": {row["sample_id"]: {"random_feedback_horizon": 7}}})
            )
            dataset = FreshEpisodeWindowDataset(
                root,
                feedback_label="random_feedback_horizon",
                feedback_output_key="feedback_horizon",
            )
            sample = dataset[0]
            self.assertEqual(sample["feedback_horizon"], 7)
            self.assertEqual(sample["oracle_feedback_horizon"], 3)
            self.assertEqual(sample["image"][0].shape, (8, 8, 3))


class SeededShuffleOptionsTest(unittest.TestCase):
    def test_shuffle_is_opt_in_and_reproducible_for_the_same_seed(self) -> None:
        disabled = _seeded_shuffle_options(SimpleNamespace(seed=41), SimpleNamespace(shuffle=False))
        self.assertEqual(disabled, {"shuffle": False})

        first = _seeded_shuffle_options(SimpleNamespace(seed=41), SimpleNamespace(shuffle=True))
        second = _seeded_shuffle_options(SimpleNamespace(seed=41), SimpleNamespace(shuffle=True))
        first_order = list(torch.utils.data.RandomSampler(range(32), generator=first["generator"]))
        second_order = list(torch.utils.data.RandomSampler(range(32), generator=second["generator"]))
        self.assertEqual(first_order, second_order)
        self.assertNotEqual(first_order, list(range(32)))


if __name__ == "__main__":
    unittest.main()
