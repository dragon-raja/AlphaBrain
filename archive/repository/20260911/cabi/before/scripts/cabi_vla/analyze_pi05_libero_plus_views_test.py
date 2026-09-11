from __future__ import annotations

from analyze_pi05_libero_plus_views import (
    bootstrap_mean,
    build_quantification_gates,
    summarize_candidates,
    summarize_gap,
    wilson_interval,
)


def _gap_row(pair: int, condition: str, success: bool, difficulty: int = 1) -> dict:
    return {
        "episode_id": f"g-{pair}-{condition}",
        "pair_key": f"gap::libero_goal::{pair}",
        "condition": condition,
        "suite": "libero_goal",
        "base_task": f"task-{pair // 2}",
        "difficulty_level": difficulty,
        "perturbation_family": "orbit_yaw",
        "success": success,
        "completion_steps": 10 if success else 100,
    }


def _candidate_row(
    group: int,
    view: str,
    success: bool,
    variance: float,
    *,
    visible: bool | None = None,
) -> dict:
    row = {
        "episode_id": f"c-{group}-{view}",
        "pair_key": f"candidate::libero_goal::task-{group}",
        "condition": f"candidate:{view}",
        "suite": "libero_goal",
        "base_task": f"task-{group}",
        "success": success,
        "completion_steps": 10 if success else 100,
        "initial_metrics": {
            "action_probe": {"mean_pairwise_rms": variance},
            "agent": {
                "entropy_32bin_bits": 2.0 - variance,
                "mean_edge_strength": 4.0,
                "clipped_fraction": 0.0,
            },
        },
    }
    if visible is not None:
        row["initial_metrics"]["sim_visibility"] = {
            "all_interest_visible": visible,
            "all_interest_visible_at_least_16px": visible,
            "any_interest_border_touch": not visible,
            "minimum_interest_pixel_count": 100 if visible else 0,
        }
    return row


def test_gap_clusters_repeated_views_by_base_task() -> None:
    rows = [
        _gap_row(0, "canonical", True),
        _gap_row(0, "official_camera", False),
        _gap_row(1, "canonical", True),
        _gap_row(1, "official_camera", True),
        _gap_row(2, "canonical", True),
        _gap_row(2, "official_camera", False),
        _gap_row(3, "canonical", True),
        _gap_row(3, "official_camera", False),
    ]
    summary = summarize_gap(rows)
    assert summary is not None
    assert summary["overall"]["episode_pair_count"] == 4
    assert summary["overall"]["independent_base_task_count"] == 2
    assert summary["overall"]["view_generalization_gap"]["mean"] == 0.75


def test_gap_separates_visibility_loss_from_visible_view_shift() -> None:
    rows = [
        _gap_row(0, "canonical", True),
        _gap_row(0, "official_camera", False),
        _gap_row(1, "canonical", True),
        _gap_row(1, "official_camera", True),
    ]
    for row in rows:
        camera_lost = row["pair_key"].endswith("::0") and row["condition"] == "official_camera"
        row["initial_metrics"] = {
            "sim_visibility": {
                "all_interest_visible": not camera_lost,
                "all_interest_visible_at_least_16px": not camera_lost,
                "any_interest_border_touch": camera_lost,
                "minimum_interest_pixel_count": 0 if camera_lost else 100,
            }
        }
    summary = summarize_gap(rows)
    assert summary is not None
    visibility = summary["visibility_diagnostics"]
    assert visibility is not None
    assert visibility["camera_any_interest_out_of_frame_rate"]["mean"] == 0.5
    assert visibility["clearly_visible_pair_count"] == 1
    assert visibility["visibility_lost_pair_count"] == 1


def test_candidate_summary_reports_oracle_headroom() -> None:
    rows = []
    for group in range(8):
        rows.append(_candidate_row(group, "canonical", group % 2 == 0, 0.4))
        rows.append(_candidate_row(group, "side", group % 2 == 1, 0.1))
    summary = summarize_candidates(rows)
    assert summary is not None
    assert summary["oracle_selection_all_views"]["success"] == 1.0
    assert summary["view_outcome_disagreement_rate"] == 1.0
    active = summary["active_uncertainty_selection_all_views"]
    assert 0.0 <= active["success"] <= 1.0
    assert active["selected_view_counts"] == {"side": summary["holdout_initial_state_count"]}


def test_candidate_summary_reports_visibility_per_view() -> None:
    rows = []
    for group in range(8):
        rows.append(_candidate_row(group, "canonical", True, 0.2, visible=True))
        rows.append(_candidate_row(group, "side", False, 0.3, visible=False))
    summary = summarize_candidates(rows)
    assert summary is not None
    visibility = summary["fixed_view_visibility"]
    assert visibility["canonical"]["any_interest_out_of_frame_rate"]["mean"] == 0.0
    assert visibility["side"]["any_interest_out_of_frame_rate"]["mean"] == 1.0
    assert visibility["side"]["minimum_interest_pixel_retention"]["mean"] == 0.0


def test_intervals_validate_and_bound_rates() -> None:
    low, high = wilson_interval(5, 10)
    assert 0.0 < low < 0.5 < high < 1.0
    assert bootstrap_mean([1.0])["ci95"] == [1.0, 1.0]


def test_quantification_gates_require_valid_baseline_and_paired_gain() -> None:
    gap = {
        "overall": {
            "canonical_success": 0.9,
            "view_generalization_gap": {"ci95": [0.1, 0.3]},
        }
    }
    candidates = {
        "canonical_holdout_success": 0.9,
        "global_static_selection": {"minus_canonical": {"ci95": [0.0, 0.2]}},
        "suite_static_selection": {"minus_canonical": {"ci95": [0.01, 0.2]}},
        "active_uncertainty_selection_all_views": {
            "minus_global_static": {"ci95": [0.02, 0.3]}
        },
        "oracle_selection_all_views": {"minus_global_static": {"mean": 0.2}},
    }
    gates = build_quantification_gates(gap, candidates)
    assert gates["BASELINE_VALID"] is True
    assert gates["VIEW_GAP_CONFIRMED"] is True
    assert gates["STATIC_VIEW_GAIN"] is True
    assert gates["ACTIVE_SELECTOR_GAIN"] is True
    assert gates["ORACLE_HEADROOM"] is True
