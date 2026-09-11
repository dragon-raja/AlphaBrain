from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from seal_confirmation_groups import build_seal


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


class ConfirmationSealTest(unittest.TestCase):
    def test_rejects_snapshot_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev, source, episodes = root / "dev", root / "source", root / "episodes"
            group = {
                "pair_id": "g0",
                "task": "grasp_slip",
                "initial_state_index": 1,
                "randomization": {"x": 0.1},
            }
            write_json(dev / "manifest.json", {"pairs": [group]})
            write_json(source / "manifest.json", {"pairs": [group]})
            write_json(source / "quality_report.json", {"passed": True})
            write_json(episodes / "manifest.json", {"groups": [group]})
            write_json(episodes / "quality_report.json", {"passed": True})

            with self.assertRaisesRegex(ValueError, "overlap"):
                build_seal(dev, source, episodes, expected_groups=1)

    @mock.patch("seal_confirmation_groups.subprocess.check_output", return_value="deadbeef\n")
    def test_builds_gate3_only_seal(self, _check_output: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev, source, episodes = root / "dev", root / "source", root / "episodes"
            dev_group = {
                "pair_id": "old",
                "task": "grasp_slip",
                "initial_state_index": 1,
                "randomization": {"x": 0.1},
            }
            new_group = {
                "pair_id": "new",
                "task": "grasp_slip",
                "initial_state_index": 2,
                "randomization": {"x": 0.2},
            }
            write_json(dev / "manifest.json", {"pairs": [dev_group]})
            write_json(source / "manifest.json", {"pairs": [new_group]})
            write_json(source / "quality_report.json", {"passed": True})
            write_json(episodes / "manifest.json", {"groups": [new_group]})
            write_json(episodes / "quality_report.json", {"passed": True})

            seal = build_seal(dev, source, episodes, expected_groups=1)

            self.assertEqual(seal["status"], "SEALED_FOR_GATE3_ONLY")
            self.assertEqual(seal["development_snapshot_overlap"], 0)
            self.assertEqual(seal["confirmation_groups"][0]["confirmation_id"], "cora-confirmation-0000")


if __name__ == "__main__":
    unittest.main()
