from __future__ import annotations

from tests.dsol_paper1.paths import ROOT as REPOSITORY_ROOT
import json
from pathlib import Path

from scripts.dsol_paper1.protocols.build_libero_view_catalog import build


ROOT = REPOSITORY_ROOT


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
