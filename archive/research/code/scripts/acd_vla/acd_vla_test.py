from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.acd_vla.collect_policy_responses import (
    array_hash,
    assert_unsealed,
    atomic_npz,
    selected_groups,
    stable_seed,
    validate_existing_record,
    validate_splits,
)


class AcdVlaTest(unittest.TestCase):
    def test_validate_splits_rejects_non_development(self) -> None:
        self.assertEqual(validate_splits(["train", "val", "train"]), ("train", "val"))
        with self.assertRaises(ValueError):
            validate_splits(["test"])

    def test_assert_unsealed_rejects_sensitive_evaluation_path(self) -> None:
        assert_unsealed(Path("/share/longjunyu/acd-vla/gate0"))
        with self.assertRaises(ValueError):
            assert_unsealed(Path("/share/longjunyu/acd-vla/confirmation/run"))

    def test_selected_groups_filters_without_returning_test(self) -> None:
        manifest = {
            "groups": [
                {"pair_id": "b", "split": "val"},
                {"pair_id": "sealed", "split": "test"},
                {"pair_id": "a", "split": "train"},
            ]
        }
        selected = selected_groups(manifest, ["train", "val"])
        self.assertEqual([group["pair_id"] for group in selected], ["a", "b"])

    def test_stable_seed_is_repeatable_and_pair_specific(self) -> None:
        self.assertEqual(stable_seed(41, "pair-a"), stable_seed(41, "pair-a"))
        self.assertNotEqual(stable_seed(41, "pair-a"), stable_seed(41, "pair-b"))

    def test_resume_record_validates_hashes(self) -> None:
        arrays = {
            "pre_actions": np.zeros((50, 7), dtype=np.float32),
            "attached_actions": np.ones((50, 7), dtype=np.float32),
            "slipped_actions": np.full((50, 7), 2, dtype=np.float32),
            "pre_feature": np.arange(4, dtype=np.float32),
            "attached_post_feature": np.arange(4, dtype=np.float32) + 1,
            "slipped_post_feature": np.arange(4, dtype=np.float32) + 2,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            array_path = root / "pair.npz"
            metadata_path = root / "pair.json"
            atomic_npz(array_path, **arrays)
            metadata = {
                "pair_id": "pair",
                "split": "train",
                "source_id": 1,
                "feedback_reveal_time": 9,
                "inference_seed": 7,
                "array_hashes": {key: array_hash(value) for key, value in arrays.items()},
            }
            metadata_path.write_text(json.dumps(metadata))
            loaded = validate_existing_record(array_path, metadata_path, metadata)
            self.assertEqual(loaded["pair_id"], "pair")


if __name__ == "__main__":
    unittest.main()
