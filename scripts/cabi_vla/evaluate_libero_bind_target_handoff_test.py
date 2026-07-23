from __future__ import annotations

from evaluate_libero_bind_target_handoff import (
    remaining_policy_budget,
    select_prefix_edge,
)


def test_source_matched_supervised_prefix_is_unique() -> None:
    edges = [
        {"edge_id": "white-left", "source_id": "white", "target_id": "left", "action_supervised": True},
        {"edge_id": "white-right", "source_id": "white", "target_id": "right", "action_supervised": False},
        {"edge_id": "red-right", "source_id": "red", "target_id": "right", "action_supervised": True},
    ]
    selected = select_prefix_edge(edges, edges[1])
    assert selected["edge_id"] == "white-left"


def test_total_budget_includes_replayed_teacher_actions() -> None:
    assert remaining_policy_budget(320, 70) == 250
    try:
        remaining_policy_budget(70, 70)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("handoff must retain a positive policy budget")
