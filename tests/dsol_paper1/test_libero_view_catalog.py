from __future__ import annotations

import json
from pathlib import Path

from scripts.dsol_paper1.build_libero_view_catalog import build


ROOT = Path(__file__).parents[2]


def test_m1_catalog_adds_eval_only_crossed_orbits() -> None:
    rules = json.loads(
        (ROOT / "configs/dsol_paper1/libero_view_catalog_v2_m1_rules.json").read_text()
    )
    catalog = build(rules)
    assert len(catalog["diagnostic_crossed_orbit"]) == 16
    assert "diagnostic_crossed_orbit" in catalog["training_exclusions"]["training"]
    assert {
        pose["pose_id"] for pose in catalog["diagnostic_crossed_orbit"]
    }.isdisjoint(catalog["broad_training_sets"]["broad_64"])
