from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from AlphaBrain.dataloader.paligemma_datasets import (
    LiberoBindTrainingDataset,
    Pi0DataConfig,
    Pi0DataTransform,
    Pi0DatasetWrapper,
    pi0_collate_fn,
)


def test_libero_bind_dataset_masks_fourth_action_and_flattens_bundle() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        collection = root / "collection"
        view = root / "view"
        (collection / "episodes").mkdir(parents=True)
        view.mkdir()
        images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
        actions = np.ones((2, 7), dtype=np.float32)
        states = np.zeros((3, 8), dtype=np.float32)
        np.savez_compressed(
            collection / "episodes" / "one.npz",
            agentview=images,
            wrist=images,
            actions=actions,
            robot_state=states,
        )
        record = {
            "sample_id": "one--frame-0000",
            "episode_file": "episodes/one.npz",
            "edge_id": "red-left",
            "canonical_state_index": 0,
            "split": "train",
            "frame_index": 0,
            "language_instruction": "put the red mug on the left plate",
        }
        second_record = {**record, "sample_id": "one--frame-0001", "frame_index": 1}
        (view / "records.jsonl").write_text(
            json.dumps(record) + "\n" + json.dumps(second_record) + "\n"
        )

        edges = ("red-left", "red-right", "white-left")
        anchors = {}
        for edge in edges:
            prefix = f"{edge}__state_00__"
            anchors[prefix + "agentview"] = images[0]
            anchors[prefix + "wrist"] = images[0]
            anchors[prefix + "state"] = states[0]
            anchors[prefix + "action"] = np.ones((2, 7), dtype=np.float32)
        np.savez_compressed(view / "anchors.npz", **anchors)
        tetrad = {
            "tetrad_id": "state-00--white-right",
            "canonical_state_index": 0,
            "split": "train",
            "corners": {
                "base": {
                    "physical_edge": "red-left",
                    "instruction_edge": "red-left",
                    "action_supervised": True,
                },
                "source_anchor": {
                    "physical_edge": "white-left",
                    "instruction_edge": "white-left",
                    "action_supervised": True,
                },
                "target_anchor": {
                    "physical_edge": "red-right",
                    "instruction_edge": "red-right",
                    "action_supervised": True,
                },
                "fourth_anchor": {
                    "physical_edge": "white-left",
                    "instruction_edge": "white-right",
                    "action_supervised": False,
                },
            },
        }
        manifest = {
            "source_collection": str(collection),
            "action_horizon": 2,
            "edge_instructions": {
                "red-left": "put the red mug on the left plate",
                "red-right": "put the red mug on the right plate",
                "white-left": "put the white mug on the left plate",
                "white-right": "put the white mug on the right plate",
            },
            "tetrads": [tetrad],
        }
        (view / "manifest.json").write_text(json.dumps(manifest))

        dataset = LiberoBindTrainingDataset(view)
        bundle = dataset[0]
        assert len(bundle) == 5
        assert bundle[-1]["cabi_corner"] == "fourth_anchor"
        assert not bundle[-1]["action_supervised"]
        np.testing.assert_array_equal(bundle[-1]["action"], np.zeros((2, 7)))

        wrapped = Pi0DatasetWrapper(
            dataset,
            Pi0DataTransform(Pi0DataConfig(action_horizon=2)),
        )
        flattened = pi0_collate_fn([wrapped[0]])
        assert len(flattened) == 5
        assert flattened[-1]["cabi_tetrad_id"].startswith("state-00")

        sparse_anchors = LiberoBindTrainingDataset(view, anchor_period=2)
        assert len(sparse_anchors[0]) == 5
        assert len(sparse_anchors[1]) == 1


def test_unsupervised_fourth_corner_does_not_access_an_action_field() -> None:
    dataset = object.__new__(LiberoBindTrainingDataset)
    dataset.action_horizon = 2
    dataset.action_dim = 7
    dataset.edge_instructions = {"white-right": "put the white mug on the right plate"}
    dataset._anchors = {
        "white-left__state_00__agentview": np.zeros((8, 8, 3), dtype=np.uint8),
        "white-left__state_00__wrist": np.zeros((8, 8, 3), dtype=np.uint8),
        "white-left__state_00__state": np.zeros(8, dtype=np.float32),
    }
    tetrad = {
        "canonical_state_index": 0,
        "corners": {
            "fourth_anchor": {
                "physical_edge": "white-left",
                "instruction_edge": "white-right",
                "action_supervised": False,
            }
        },
    }
    example = dataset._anchor_example(tetrad, "fourth_anchor", instance_id="audit")
    assert not example["action_supervised"]
    np.testing.assert_array_equal(example["action"], np.zeros((2, 7), np.float32))
