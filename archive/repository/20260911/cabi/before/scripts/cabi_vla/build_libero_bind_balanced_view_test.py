from collections import Counter

from build_libero_bind_balanced_view import (
    common_stage_distribution,
    largest_remainder,
    macro_phase,
    plan_balanced_edge_quotas,
    solve_equal_edge_loss_units,
    source_record_coverage,
    stage_quotas_cover_source,
)


def synthetic_manifest() -> dict:
    return {
        "tetrads": [
            {
                "corners": {
                    "base": {"instruction_edge": "red-left", "action_supervised": True},
                    "source_anchor": {"instruction_edge": "white-left", "action_supervised": True},
                    "target_anchor": {"instruction_edge": "red-right", "action_supervised": True},
                    "fourth_anchor": {"instruction_edge": "white-right", "action_supervised": False},
                }
            },
            {
                "corners": {
                    "base": {"instruction_edge": "red-right", "action_supervised": True},
                    "source_anchor": {"instruction_edge": "yellow_white-right", "action_supervised": True},
                    "target_anchor": {"instruction_edge": "red-left", "action_supervised": True},
                    "fourth_anchor": {"instruction_edge": "yellow_white-left", "action_supervised": False},
                }
            },
        ]
    }


def test_exact_exposure_plan_balances_sources_and_targets() -> None:
    count, quotas, anchor_regular, loss_units, anchor = plan_balanced_edge_quotas(
        synthetic_manifest(),
        ("red-left", "red-right", "white-left", "yellow_white-right"),
        minimum_record_count=5611,
        anchor_period=8,
    )
    assert count == 5616
    assert quotas == {
        "red-left": 840,
        "red-right": 840,
        "white-left": 1968,
        "yellow_white-right": 1968,
    }
    assert anchor_regular == {
        "red-left": 106,
        "red-right": 106,
        "white-left": 245,
        "yellow_white-right": 245,
    }
    assert loss_units == {
        "red-left": 3042,
        "red-right": 3042,
        "white-left": 7137,
        "yellow_white-right": 7137,
    }
    source_units = Counter(anchor["sources"])
    target_units = Counter(anchor["targets"])
    for edge, units in loss_units.items():
        source, target = edge.rsplit("-", 1)
        source_units[source] += units
        target_units[target] += units
    assert set(source_units.values()) == {7488}
    assert set(target_units.values()) == {11232}


def test_equal_edge_plan_removes_incomplete_graph_shortcut() -> None:
    count, quotas, anchor_regular, loss_units, anchor = plan_balanced_edge_quotas(
        synthetic_manifest(),
        ("red-left", "red-right", "white-left", "yellow_white-right"),
        minimum_record_count=5611,
        anchor_period=32,
        balance_objective="observed_edges",
    )
    assert count % 32 == 0
    assert sum(quotas.values()) == count
    assert sum(anchor_regular.values()) == count // 32
    effective = Counter(anchor["edges"])
    effective.update(loss_units)
    assert set(effective.values()) == {count}


def test_equal_edge_solver_accounts_for_anchor_loss_reduction() -> None:
    anchor = {
        "edges": Counter(
            {
                "red-left": 8,
                "red-right": 8,
                "white-left": 4,
                "yellow_white-right": 4,
            }
        )
    }
    units = solve_equal_edge_loss_units(
        ("red-left", "red-right", "white-left", "yellow_white-right"),
        anchor,
        record_count=256,
    )
    assert units == {
        "red-left": 248,
        "red-right": 248,
        "white-left": 252,
        "yellow_white-right": 252,
    }


def test_largest_remainder_is_exact_and_deterministic() -> None:
    result = largest_remainder(11, {"approach": 0.5, "grasp": 0.25, "place": 0.25})
    assert result == {"approach": 5, "grasp": 3, "place": 3}
    assert sum(result.values()) == 11


def test_macro_phase_mapping_covers_teacher_controller() -> None:
    assert macro_phase("episode_start") == "approach"
    assert macro_phase("close_gripper") == "grasp"
    assert macro_phase("lift") == "lift"
    assert macro_phase("transport") == "transport"
    assert macro_phase("release") == "place"


def test_stage_coverage_requires_room_for_every_source_record() -> None:
    grouped = {
        "red-left": {
            "approach": [{"sample_id": "a"}, {"sample_id": "b"}],
            "grasp": [{"sample_id": "c"}],
        }
    }
    assert stage_quotas_cover_source(
        {"red-left": {"approach": 2, "grasp": 1}}, grouped
    )
    assert not stage_quotas_cover_source(
        {"red-left": {"approach": 1, "grasp": 2}}, grouped
    )


def test_common_distribution_keeps_fine_grained_phases_separate() -> None:
    distribution = common_stage_distribution(
        {
            "red-left": {"episode_start": [1], "approach_grasp": [1, 2, 3]},
            "white-left": {"episode_start": [1, 2], "approach_grasp": [1, 2]},
        }
    )
    assert distribution == {"approach_grasp": 0.625, "episode_start": 0.375}


def test_source_coverage_audits_episode_start_states() -> None:
    source = [
        {
            "sample_id": "start-0",
            "edge_id": "red-left",
            "canonical_state_index": 0,
            "teacher_phase": "episode_start",
        },
        {
            "sample_id": "start-1",
            "edge_id": "red-left",
            "canonical_state_index": 1,
            "teacher_phase": "episode_start",
        },
        {
            "sample_id": "lift-0",
            "edge_id": "red-left",
            "canonical_state_index": 0,
            "teacher_phase": "lift",
        },
    ]
    complete = source_record_coverage(source, list(reversed(source)))
    assert complete["all_source_records_preserved"]
    assert complete["all_episode_start_states_preserved"]

    incomplete = source_record_coverage(source, source[1:])
    assert incomplete["missing_source_record_count"] == 1
    assert not incomplete["all_episode_start_states_preserved"]
