import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from build_libero_bind_horizon_view import build_horizon_view, parse_action_key


def _write_fixture(root: Path) -> Path:
    collection = root / "collection"
    episodes = collection / "episodes"
    episodes.mkdir(parents=True)
    phases = np.asarray(
        ["episode_start", "pre_transport", "last_pre_transport"]
        + ["transport"] * 22
    )
    actions = np.arange(25 * 7, dtype=np.float32).reshape(25, 7)
    np.savez_compressed(
        episodes / "red-left--state-00.npz",
        phase=phases,
        actions=actions,
    )
    (collection / "manifest.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "edge_id": "red-left",
                        "canonical_state_index": 0,
                        "episode_file": "episodes/red-left--state-00.npz",
                        "success": True,
                        "action_supervised": True,
                    }
                ]
            }
        )
    )
    source = root / "source"
    source.mkdir()
    (source / "records.jsonl").write_text('{"split":"train"}\n')
    np.savez_compressed(
        source / "anchors.npz",
        **{
            "red-left__state_00__source_select__agentview": np.zeros((2, 2, 3), np.uint8),
            "red-left__state_00__source_select__wrist": np.zeros((2, 2, 3), np.uint8),
            "red-left__state_00__source_select__state": np.zeros(8, np.float32),
            "red-left__state_00__source_select__action": actions[:10],
            "red-left__state_00__target_select__agentview": np.ones((2, 2, 3), np.uint8),
            "red-left__state_00__target_select__wrist": np.ones((2, 2, 3), np.uint8),
            "red-left__state_00__target_select__state": np.ones(8, np.float32),
            "red-left__state_00__target_select__action": actions[2:12],
        },
    )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source_collection": str(collection),
                "action_horizon": 10,
                "action_dim": 7,
                "record_count": 1,
                "tetrad_count": 1,
                "records_file": "records.jsonl",
                "anchors_file": "anchors.npz",
                "tetrads": [],
                "leakage_guard": {
                    "withheld_action_edges": ["white-right"],
                    "fourth_corner_actions_loaded": False,
                },
            }
        )
    )
    return source


class HorizonViewTest(unittest.TestCase):
    def test_parse_action_key(self) -> None:
        self.assertEqual(
            parse_action_key(
                "yellow_white-right__state_17__target_select__action"
            ),
            ("yellow_white-right", 17, "target_select"),
        )
        with self.assertRaises(ValueError):
            parse_action_key("red-left__state_00__action")

    def test_build_preserves_conditioning_and_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_fixture(root)
            output = root / "h20"
            report = build_horizon_view(source, output, action_horizon=20)

            self.assertEqual(report["action_horizon"], 20)
            self.assertEqual(
                (output / "records.jsonl").read_bytes(),
                (source / "records.jsonl").read_bytes(),
            )
            self.assertFalse(
                report["leakage_guard"]["horizon_reslice_uses_teacher_qa"]
            )
            with np.load(source / "anchors.npz", allow_pickle=False) as old, np.load(
                output / "anchors.npz", allow_pickle=False
            ) as new:
                for key in old.files:
                    if not key.endswith("__action"):
                        self.assertTrue(np.array_equal(old[key], new[key]))
                self.assertEqual(
                    new["red-left__state_00__source_select__action"].shape,
                    (20, 7),
                )
                self.assertTrue(
                    np.array_equal(
                        new["red-left__state_00__target_select__action"],
                        np.arange(25 * 7, dtype=np.float32).reshape(25, 7)[2:22],
                    )
                )


if __name__ == "__main__":
    unittest.main()
