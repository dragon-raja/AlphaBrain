from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from finalize_collection import DEPLOYABLE_SHAPES, LABEL_SHAPES, audit_collection


def write_npz(path: Path, shapes: dict[str, tuple[int, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.zeros(shape, dtype=np.float32) for key, shape in shapes.items()})


class FinalizeCollectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.episodes = self.root / "episodes"
        self.dataset.mkdir()
        self.episodes.mkdir()
        (self.episodes / "manifest.json").write_text(
            json.dumps(
                {
                    "groups": [
                        {"pair_id": "pair-fit", "split": "train", "source_initial_state_index": 1},
                        {"pair_id": "pair-sealed", "split": "train", "source_initial_state_index": 2},
                        {"pair_id": "pair-test", "split": "test", "source_initial_state_index": 99},
                    ]
                }
            )
        )
        (self.dataset / "metadata.json").write_text(
            json.dumps(
                {
                    "all_train_group_count": 2,
                    "fit_source_ids": [1],
                    "holdout_source_ids": [2],
                    "engineering_excluded_source_ids": [],
                }
            )
        )
        self._write_group("pair-fit", 1, "fit")
        self._write_group("pair-sealed", 2, "holdout", sealed_garbage=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_group(
        self,
        pair_id: str,
        source_id: int,
        partition: str,
        *,
        sealed_garbage: bool = False,
    ) -> None:
        state_root = self.dataset / "groups" / pair_id / "states" / "state-r000"
        relative = state_root.relative_to(self.dataset)
        if sealed_garbage:
            state_root.mkdir(parents=True)
            for name in ("deployable.npz", "labels.npz", "audit_snapshot.npz"):
                (state_root / name).write_bytes(b"sealed")
        else:
            write_npz(state_root / "deployable.npz", DEPLOYABLE_SHAPES)
            write_npz(state_root / "labels.npz", LABEL_SHAPES)
            np.savez_compressed(state_root / "audit_snapshot.npz", sim_state=np.zeros(3))
        row = {
            "pair_id": pair_id,
            "source_initial_state_index": source_id,
            "source_partition": partition,
            "success": True,
            "states": [
                {
                    "state_id": "state-r000",
                    "candidate_count": 16,
                    "continuation_repeats": 6,
                    "raw_milestone_violations": 0,
                    "deployable_file": str(relative / "deployable.npz"),
                    "labels_file": str(relative / "labels.npz"),
                    "audit_file": str(relative / "audit_snapshot.npz"),
                }
            ],
        }
        complete = self.dataset / "groups" / pair_id / "complete.json"
        complete.parent.mkdir(parents=True, exist_ok=True)
        complete.write_text(json.dumps(row))

    def test_complete_collection_passes_without_opening_sealed_holdout(self) -> None:
        audit, manifest = audit_collection(self.dataset, self.episodes)
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["inspected_non_holdout_states"], 1)
        self.assertEqual(audit["sealed_holdout_states_not_opened"], 1)
        self.assertEqual(manifest["status"], "complete")

    def test_missing_group_blocks_finalization(self) -> None:
        (self.dataset / "groups" / "pair-fit" / "complete.json").unlink()
        audit, manifest = audit_collection(self.dataset, self.episodes)
        self.assertEqual(audit["status"], "fail")
        self.assertIsNone(manifest)
        self.assertEqual(audit["missing_groups"], ["pair-fit"])


if __name__ == "__main__":
    unittest.main()
