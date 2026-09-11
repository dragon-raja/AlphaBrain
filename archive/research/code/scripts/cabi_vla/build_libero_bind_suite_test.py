from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_libero_bind_suite import (
    SOURCES,
    TARGETS,
    build_manifest,
    render_bddl,
    state_split,
    write_suite,
)


class LiberoBindSuiteTest(unittest.TestCase):
    def test_complete_three_by_two_matrix_has_two_withheld_edges(self) -> None:
        manifest = build_manifest(
            bddl_dir=Path("/tmp/bddl"),
            canonical_init_path=Path("/tmp/states.pruned_init"),
        )
        self.assertEqual(len(manifest["edges"]), 6)
        self.assertEqual(sum(edge["action_supervised"] for edge in manifest["edges"]), 4)
        withheld = {
            edge["edge_id"] for edge in manifest["edges"] if not edge["action_supervised"]
        }
        self.assertEqual(withheld, {"white-right", "yellow_white-left"})

    def test_bddl_changes_only_semantic_goal_fields_across_edges(self) -> None:
        for source in SOURCES:
            for target in TARGETS:
                text = render_bddl(source, target)
                self.assertIn(f"(:language put the {source.phrase} on the {target.phrase})", text)
                self.assertIn(f"(And (On {source.object_name} {target.object_name}))", text)
                self.assertIn(source.object_name, text)
                self.assertIn(target.object_name, text)

    def test_state_split_is_sealed_by_canonical_index(self) -> None:
        self.assertEqual(state_split(0), "train")
        self.assertEqual(state_split(34), "train")
        self.assertEqual(state_split(35), "val")
        self.assertEqual(state_split(39), "val")
        self.assertEqual(state_split(40), "test")
        self.assertEqual(state_split(49), "test")
        with self.assertRaises(ValueError):
            state_split(50)

    def test_writer_is_atomic_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init_path = root / "states.pruned_init"
            init_path.write_bytes(b"fixture")
            output = root / "suite"
            write_suite(output, init_path)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["benchmark"], "LIBERO-Bind-v0")
            self.assertEqual(len(list((output / "bddl").glob("*.bddl"))), 6)
            with self.assertRaises(FileExistsError):
                write_suite(output, init_path)


if __name__ == "__main__":
    unittest.main()
