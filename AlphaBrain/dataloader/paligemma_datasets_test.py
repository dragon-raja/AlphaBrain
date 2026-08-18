import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from AlphaBrain.dataloader import _seeded_shuffle_options
from AlphaBrain.dataloader.paligemma_datasets import (
    DsolLiberoPairDataset,
    FreshEpisodeWindowDataset,
    FreshSnapshotDataset,
    Pi0DataConfig,
    Pi0DataTransform,
)
from scripts.dsol_paper1.libero_pair_records import initialize_shard, write_record


class Pi0DataTransformTest(unittest.TestCase):
    def test_preserves_feedback_horizon(self) -> None:
        transform = Pi0DataTransform(Pi0DataConfig(action_horizon=2))
        sample = {
            "action": np.zeros((2, 7), dtype=np.float32),
            "feedback_horizon": 1,
        }

        result = transform(sample)

        self.assertEqual(result["feedback_horizon"], 1)

    def test_image_augmentation_is_seeded_and_changes_pixels(self) -> None:
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[..., 0] = np.arange(224, dtype=np.uint8)[None, :]
        image[..., 1] = np.arange(224, dtype=np.uint8)[:, None]
        sample = {
            "image": [image, image.copy()],
            "dsol_image_augmentation": True,
        }
        transform = Pi0DataTransform(Pi0DataConfig(use_image_augmentation=True))

        torch.manual_seed(41)
        first = transform(sample)
        torch.manual_seed(41)
        second = transform(sample)

        self.assertEqual(first["image"][0].shape, (224, 224, 3))
        self.assertTrue(np.array_equal(first["image"][0], second["image"][0]))
        self.assertFalse(np.array_equal(first["image"][0], image))
        self.assertTrue(first["dsol_image_augmentation"])


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


class DsolLiberoPairDatasetTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        shard_root = root / "task"
        shard_root.mkdir()
        shard_path = shard_root / "pairs.bin"
        records_path = shard_root / "records.jsonl"
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        images = {
            "canonical": image,
            "broad_a": image + 32,
            "broad_b": image + 64,
            "wrist": image + 96,
        }
        header = {
            "sample_id": "task::demo_0::frame-00000",
            "episode_id": "task::demo_0",
            "frame": 0,
            "language_instruction": "move the object",
            "action_chunk": np.zeros((10, 7), dtype=np.float32).tolist(),
            "robot_state": np.zeros(8, dtype=np.float32).tolist(),
            "canonical_camera_to_world_opencv": np.eye(4).tolist(),
            "camera_a_to_world_opencv": np.eye(4).tolist(),
            "camera_b_to_world_opencv": np.eye(4).tolist(),
        }
        with shard_path.open("wb") as handle:
            initialize_shard(handle)
            location = write_record(handle, header=header, images=images)
        records_path.write_text(
            json.dumps(
                {
                    "sample_id": header["sample_id"],
                    "episode_id": header["episode_id"],
                    "split": "train",
                    "offset": location["offset"],
                }
            )
            + "\n"
        )
        (shard_root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "dsol_libero_hdf5_view_pair_shard_v1",
                    "status": "VERIFIED",
                    "shard": shard_path.name,
                    "records": records_path.name,
                }
            )
        )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "dsol_libero_hdf5_view_pair_collection_v1",
                    "shards": [{"path": shard_root.name}],
                }
            )
        )

    def test_state_matched_and_paired_share_flow_but_only_consistency_has_objective(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_fixture(root)
            state_matched = DsolLiberoPairDataset(
                root, arm="broad_unpaired_state_matched"
            )[0]
            paired_fm = DsolLiberoPairDataset(root, arm="broad_paired_fm")[0]
            paired_consistency = DsolLiberoPairDataset(
                root, arm="broad_paired_consistency"
            )[0]

            self.assertTrue(all(row["dsol_pair_shared_flow"] for row in state_matched))
            self.assertTrue(all(row["dsol_pair_shared_flow"] for row in paired_fm))
            self.assertTrue(
                all(row["dsol_pair_shared_flow"] for row in paired_consistency)
            )
            self.assertFalse(any(row["dsol_pair_objective"] for row in state_matched))
            self.assertFalse(any(row["dsol_pair_objective"] for row in paired_fm))
            self.assertTrue(
                all(row["dsol_pair_objective"] for row in paired_consistency)
            )
            self.assertTrue(
                np.array_equal(state_matched[0]["image"][0], state_matched[1]["image"][0])
            )
            self.assertFalse(
                np.array_equal(paired_fm[0]["image"][0], paired_fm[1]["image"][0])
            )


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
