from pathlib import Path

import pytest

from scripts.dsol_paper1.audit_dense_selector_common_failures import entity_deltas, evaluation_seed, grouped_summary


def test_evaluation_seed_is_not_the_state_specific_policy_noise():
    path = Path("/runs/seed-20260841/episodes-shard-00.jsonl")
    row = {"policy_noise_seed": 1398877024}
    assert evaluation_seed(row, path) == 20260841
    assert evaluation_seed({**row, "evaluation_seed": 20260842}, Path("/runs/episodes.jsonl")) == 20260842
    with pytest.raises(ValueError, match="missing evaluation-repeat"):
        evaluation_seed(row, Path("/runs/episodes.jsonl"))


def test_fixture_gain_can_hide_no_target_gain():
    def record(target, fixture):
        return {
            "visibility": {
                "camera_names": ["ext", "wrist"],
                "entity_names": ["target", "fixture"],
                "per_camera": {
                    "ext": {
                        "entities": {"target": {"visible_fraction": target}, "fixture": {"visible_fraction": fixture}}
                    },
                    "wrist": {"entities": {"target": {"visible_fraction": 0.02}, "fixture": {"visible_fraction": 0.1}}},
                },
            }
        }

    deltas = entity_deltas(record(0.01, 0.1), record(0.005, 0.2))
    assert deltas["target"] < 0
    assert sum(deltas.values()) / 2 > 0
    assert deltas["fixture"] == pytest.approx(0.05)


def test_source_equal_summary_does_not_count_states_as_sources():
    def row(source, difference):
        return dict(
            source_group=source,
            difference_pp=difference,
            canonical_success=0.5,
            selected_success=0.5 + difference / 100,
            stable_rescue=0,
            stable_harm=0,
            target_delta_pp=1.0,
        )

    result = grouped_summary([row("a", 10), row("a", 10), row("b", -10)])
    assert result["states"] == 3
    assert result["sources"] == 2
    assert result["source_equal_difference_pp"] == 0
    assert result["difference_pp"] == pytest.approx(10 / 3)
