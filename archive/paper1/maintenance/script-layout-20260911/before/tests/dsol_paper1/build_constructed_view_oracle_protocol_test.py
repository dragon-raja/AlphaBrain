from __future__ import annotations

import unittest

from build_constructed_view_oracle_protocol import group_for_pose, select_stages


class ConstructedViewOracleProtocolTest(unittest.TestCase):
    def test_group_for_pose(self) -> None:
        self.assertEqual(group_for_pose("canonical"), "canonical")
        self.assertEqual(group_for_pose("broad_train_003"), "broad_training_64")
        self.assertEqual(group_for_pose("broad_heldout_004"), "broad_heldout_32")

    def test_select_stages_without_replacement(self) -> None:
        rows = [
            {"scan_id": "a", "stage_fraction": 0.1},
            {"scan_id": "b", "stage_fraction": 0.3},
            {"scan_id": "c", "stage_fraction": 0.6},
        ]
        selected = select_stages(rows, [0.2, 0.55])
        self.assertEqual([row["scan_id"] for row in selected], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
