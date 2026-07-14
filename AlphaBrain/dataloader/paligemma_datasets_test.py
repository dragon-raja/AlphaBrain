import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from AlphaBrain.dataloader.paligemma_datasets import FreshSnapshotDataset, Pi0DataConfig, Pi0DataTransform


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


if __name__ == "__main__":
    unittest.main()
