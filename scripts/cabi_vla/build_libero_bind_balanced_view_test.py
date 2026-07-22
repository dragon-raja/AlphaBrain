from collections import Counter

from build_libero_bind_balanced_view import (
    largest_remainder,
    macro_phase,
    plan_balanced_edge_quotas,
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
