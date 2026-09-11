from __future__ import annotations

import unittest
from pathlib import Path

from validate_kyc_scaling_views import (
    parse_view_spec,
    validate_nested_catalogs,
)


class ValidateKycScalingViewsTest(unittest.TestCase):
    def test_parse_view_spec(self) -> None:
        self.assertEqual(
            parse_view_spec("45=cue_randomized=/tmp/view"),
            (45, "cue_randomized", Path("/tmp/view")),
        )

    def test_nested_catalog_accepts_shared_prefix(self) -> None:
        validate_nested_catalogs(
            [
                {
                    "scene_cue_mode": "fixed",
                    "catalog_size": 2,
                    "pose_by_index": {0: (1.0, 2.0, 3.0), 1: (4.0, 5.0, 6.0)},
                },
                {
                    "scene_cue_mode": "fixed",
                    "catalog_size": 3,
                    "pose_by_index": {
                        0: (1.0, 2.0, 3.0),
                        1: (4.0, 5.0, 6.0),
                        2: (7.0, 8.0, 9.0),
                    },
                },
            ]
        )

    def test_nested_catalog_rejects_changed_pose(self) -> None:
        with self.assertRaisesRegex(ValueError, "not nested"):
            validate_nested_catalogs(
                [
                    {
                        "scene_cue_mode": "fixed",
                        "catalog_size": 1,
                        "pose_by_index": {0: (1.0, 2.0, 3.0)},
                    },
                    {
                        "scene_cue_mode": "fixed",
                        "catalog_size": 2,
                        "pose_by_index": {0: (9.0, 2.0, 3.0)},
                    },
                ]
            )


if __name__ == "__main__":
    unittest.main()
