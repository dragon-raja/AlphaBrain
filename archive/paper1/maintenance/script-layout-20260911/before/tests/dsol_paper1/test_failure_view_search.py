from copy import deepcopy

import pytest

from scripts.dsol_paper1.failure_view_search import choose_confirmation, paired_source_summary, prepare_payload


def inputs():
    rows = []
    specs = []
    states = []
    for state, successes in [("a", 0), ("b", 2), ("c", 3)]:
        rows.append(
            {
                "selector_method": "canonical",
                "pair_key": state,
                "repeat_successes": successes,
                "source_group": f"source-{state}",
            }
        )
        states.append({"pair_key": state, "task_id": "task", "source_episode_id": f"source-{state}"})
        for candidate in ["canonical", *[f"view-{i:02d}" for i in range(96)]]:
            specs.append({"pair_key": state, "selected_candidate_id": candidate, "episode_id": f"{state}-{candidate}"})
    combined = {"status": "PASS", "expected_repeats": 5, "state_method_rows": rows}
    protocol = {"status": "PASS", "catalog": "catalog", "specs": specs, "selected_states": states}
    return combined, protocol


def test_failure_screen_keeps_whole_bank_and_marks_sources_consumed():
    combined, protocol = inputs()
    p = prepare_payload(combined, [protocol])
    assert p["selected_state_count"] == 2
    assert p["episode_count"] == 194
    assert not p["confirmatory_test_eligible"]
    assert not p["candidate_prefilter_by_visibility_or_accel"]
    assert {s["pair_key"] for s in p["specs"]} == {"a", "b"}


def test_missing_candidate_fails_closed():
    combined, protocol = inputs()
    protocol["specs"].pop(0)
    with pytest.raises(ValueError, match="full 97"):
        prepare_payload(combined, [protocol])


def test_confirmation_uses_behavior_and_keeps_canonical_and_controls():
    candidates = {"canonical": {"mean_success": 0.0, "mean_steps": 100, "pose": None}}
    for i in range(96):
        candidates[f"v{i:02d}"] = {
            "mean_success": 1.0 if i < 6 else 0.0,
            "mean_steps": 10 + i,
            "pose": {"azimuth_deg": i - 48, "elevation_deg": i / 10, "radius_scale": 1},
            "visibility_score": i / 96,
            "ensemble_accel_3": i,
        }
    before = deepcopy(candidates)
    selected = choose_confirmation(candidates)
    assert len(selected) == 8
    assert "primary_frozen_top1" in selected["v00"]
    assert selected["canonical"] == ["canonical"]
    assert sum("discovery_top4" in roles for roles in selected.values()) == 4
    assert sum("pose_diverse_control" in roles for roles in selected.values()) == 2
    assert candidates == before


def test_paired_summary_resamples_sources_not_frames():
    r = paired_source_summary(
        [
            {"source_group": "a", "advantage": 1.0},
            {"source_group": "a", "advantage": 1.0},
            {"source_group": "b", "advantage": -1.0},
        ]
    )
    assert r["source_groups"] == 2
    assert r["source_equal_advantage_pp"] == 0
    assert r["state_equal_advantage_pp"] == pytest.approx(100 / 3)
