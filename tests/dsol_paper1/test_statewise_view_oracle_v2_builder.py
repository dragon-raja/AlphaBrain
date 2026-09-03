from __future__ import annotations

from scripts.dsol_paper1.build_statewise_view_oracle_v2 import (
    build_dense_protocol,
    candidate_sort_key,
)
from scripts.dsol_paper1.evaluate_dsol_libero_hdf5_views import (
    protocol_spec_at,
    protocol_spec_count,
)


def test_candidate_sort_places_canonical_then_train_then_heldout() -> None:
    values = ["broad_heldout_001", "broad_train_002", "canonical", "broad_train_001"]
    assert sorted(values, key=candidate_sort_key) == [
        "canonical",
        "broad_train_001",
        "broad_train_002",
        "broad_heldout_001",
    ]


def test_dense_compact_protocol_expands_full_matrix(tmp_path, monkeypatch) -> None:
    state = {
        "pair_key": "oracle-v2::development::task::demo::frame-00001",
        "asset_source_pair_key": "legacy-state",
        "source_group": "source",
        "task_id": "task",
    }
    records = [
        {
            "status": "PASS",
            "pose_id": "canonical" if index == 0 else f"broad_train_{index:03d}",
            "pose": None,
            "group": "canonical" if index == 0 else "broad_train",
            "visibility_score": 0.5,
            "delta_visibility": 0.0,
            "per_camera_scores": {},
            "camera_displacement_from_canonical": {},
        }
        for index in range(97)
    ]
    scans = {
        "legacy-state": {
            "scan": {"records": records, "scene_construction": {"name": "scene"}}
        }
    }
    for path in (tmp_path / "population.json", tmp_path / "catalog.json", tmp_path / "config.json"):
        path.write_text("{}\n")
    protocol = build_dense_protocol(
        role="development",
        states=[state],
        scan_assets=scans,
        repeat_ids=[4, 5, 6, 7],
        bank_id="O",
        wave_id="wave-01",
        population_path=tmp_path / "population.json",
        catalog_path=tmp_path / "catalog.json",
        protocol_config=tmp_path / "config.json",
    )
    assert protocol["episode_count"] == 388
    assert protocol_spec_count(protocol) == 388
    assert protocol_spec_at(protocol, 0)["selected_candidate_id"] == "canonical"
    assert protocol_spec_at(protocol, 0)["policy_repeat_id"] == 4
    assert protocol_spec_at(protocol, 4)["selected_candidate_id"] == "broad_train_001"
    assert protocol_spec_at(protocol, 387)["policy_repeat_id"] == 7
