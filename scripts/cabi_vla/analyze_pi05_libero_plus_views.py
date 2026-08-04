from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def wilson_interval(successes: int, count: int, *, z: float = 1.96) -> list[float]:
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("Wilson interval requires 0 <= successes <= count")
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int = 10000,
    seed: int = 20260804,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a non-empty finite vector")
    if len(array) == 1:
        low = high = float(array[0])
    else:
        generator = np.random.default_rng(seed)
        indices = generator.integers(0, len(array), size=(samples, len(array)))
        means = array[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "mean": float(np.mean(array)),
        "ci95": [float(low), float(high)],
        "independent_group_count": len(array),
        "bootstrap_resamples": samples,
    }


def read_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = str(row["episode_id"])
            if episode_id in seen:
                raise ValueError(f"duplicate episode id: {episode_id}")
            if row.get("status") != "complete":
                raise ValueError(f"incomplete episode: {episode_id}")
            seen.add(episode_id)
            rows.append(row)
    return rows


def _gap_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not str(row["pair_key"]).startswith("gap::"):
            continue
        grouped[str(row["pair_key"])][str(row["condition"])] = row
    pairs = []
    for pair_key, conditions in sorted(grouped.items()):
        if set(conditions) != {"canonical", "official_camera"}:
            raise ValueError(f"incomplete gap pair: {pair_key} has {sorted(conditions)}")
        canonical = conditions["canonical"]
        camera = conditions["official_camera"]
        canonical_visibility = canonical.get("initial_metrics", {}).get("sim_visibility", {})
        camera_visibility = camera.get("initial_metrics", {}).get("sim_visibility", {})
        canonical_pixels = float(canonical_visibility.get("minimum_interest_pixel_count", 0))
        camera_pixels = float(camera_visibility.get("minimum_interest_pixel_count", 0))
        pairs.append(
            {
                "pair_key": pair_key,
                "base_group": f"{camera['suite']}::{camera['base_task']}",
                "suite": str(camera["suite"]),
                "base_task": str(camera["base_task"]),
                "difficulty_level": int(camera["difficulty_level"]),
                "perturbation_family": str(camera["perturbation_family"]),
                "canonical_success": float(bool(canonical["success"])),
                "camera_success": float(bool(camera["success"])),
                "canonical_steps": float(canonical["completion_steps"]),
                "camera_steps": float(camera["completion_steps"]),
                "visibility_available": bool(canonical_visibility) and bool(camera_visibility),
                "canonical_all_interest_visible": bool(
                    canonical_visibility.get("all_interest_visible", True)
                ),
                "camera_all_interest_visible": bool(
                    camera_visibility.get("all_interest_visible", True)
                ),
                "canonical_all_interest_visible_at_least_16px": bool(
                    canonical_visibility.get("all_interest_visible_at_least_16px", True)
                ),
                "camera_all_interest_visible_at_least_16px": bool(
                    camera_visibility.get("all_interest_visible_at_least_16px", True)
                ),
                "camera_any_interest_border_touch": bool(
                    camera_visibility.get("any_interest_border_touch", False)
                ),
                "minimum_interest_pixel_retention": (
                    camera_pixels / canonical_pixels if canonical_pixels > 0 else None
                ),
            }
        )
    return pairs


def _clustered_gap(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clustered: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        clustered[str(row["base_group"])].append(row)
    canonical = [float(np.mean([item["canonical_success"] for item in group])) for group in clustered.values()]
    camera = [float(np.mean([item["camera_success"] for item in group])) for group in clustered.values()]
    deltas = [right - left for left, right in zip(canonical, camera, strict=True)]
    delta = bootstrap_mean(deltas)
    return {
        "episode_pair_count": len(pairs),
        "independent_base_task_count": len(clustered),
        "canonical_success": float(np.mean(canonical)),
        "official_camera_success": float(np.mean(camera)),
        "official_minus_canonical": delta,
        "view_generalization_gap": {
            "mean": -float(delta["mean"]),
            "ci95": [-float(delta["ci95"][1]), -float(delta["ci95"][0])],
            "unit": "absolute_success_rate",
        },
        "robustness_ratio": (
            float(np.mean(camera) / np.mean(canonical)) if np.mean(canonical) > 0 else None
        ),
    }


def _clustered_pair_metric(
    pairs: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    clustered: dict[str, list[float]] = defaultdict(list)
    for row in pairs:
        value = row.get(key)
        if value is not None:
            clustered[str(row["base_group"])].append(float(value))
    if not clustered:
        raise ValueError(f"no finite pair values for {key}")
    values = [float(np.mean(clustered[group])) for group in sorted(clustered)]
    return bootstrap_mean(values)


def summarize_gap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    pairs = _gap_pairs(rows)
    if not pairs:
        return None
    by_difficulty = {}
    for difficulty in sorted({int(row["difficulty_level"]) for row in pairs}):
        by_difficulty[str(difficulty)] = _clustered_gap(
            [row for row in pairs if int(row["difficulty_level"]) == difficulty]
        )
    by_suite = {}
    for suite in sorted({str(row["suite"]) for row in pairs}):
        by_suite[suite] = _clustered_gap([row for row in pairs if row["suite"] == suite])
    by_family = {}
    for family in sorted({str(row["perturbation_family"]) for row in pairs}):
        by_family[family] = _clustered_gap(
            [row for row in pairs if row["perturbation_family"] == family]
        )
    overall = _clustered_gap(pairs)
    overall["balanced_difficulty_camera_auc"] = float(
        np.mean([value["official_camera_success"] for value in by_difficulty.values()])
    )
    visibility_pairs = [row for row in pairs if row["visibility_available"]]
    clearly_visible = [
        row
        for row in visibility_pairs
        if row["canonical_all_interest_visible_at_least_16px"]
        and row["camera_all_interest_visible_at_least_16px"]
    ]
    visibility_lost = [
        row
        for row in visibility_pairs
        if row["canonical_all_interest_visible_at_least_16px"]
        and not row["camera_all_interest_visible_at_least_16px"]
    ]
    visibility_diagnostics = {
        "camera_any_interest_out_of_frame_rate": _clustered_pair_metric(
            [
                {**row, "metric": not row["camera_all_interest_visible"]}
                for row in visibility_pairs
            ],
            "metric",
        ),
        "camera_any_interest_below_16px_rate": _clustered_pair_metric(
            [
                {
                    **row,
                    "metric": not row["camera_all_interest_visible_at_least_16px"],
                }
                for row in visibility_pairs
            ],
            "metric",
        ),
        "camera_any_interest_border_touch_rate": _clustered_pair_metric(
            [
                {**row, "metric": row["camera_any_interest_border_touch"]}
                for row in visibility_pairs
            ],
            "metric",
        ),
        "minimum_interest_pixel_retention": _clustered_pair_metric(
            visibility_pairs,
            "minimum_interest_pixel_retention",
        ),
        "clearly_visible_pair_count": len(clearly_visible),
        "visibility_lost_pair_count": len(visibility_lost),
        "gap_when_all_interest_objects_clearly_visible": (
            _clustered_gap(clearly_visible) if clearly_visible else None
        ),
        "gap_when_camera_loses_clear_visibility": (
            _clustered_gap(visibility_lost) if visibility_lost else None
        ),
    } if visibility_pairs else None
    return {
        "overall": overall,
        "by_difficulty": by_difficulty,
        "by_suite": by_suite,
        "by_perturbation_family": by_family,
        "visibility_diagnostics": visibility_diagnostics,
    }


def _candidate_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    matrix: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not str(row["pair_key"]).startswith("candidate::"):
            continue
        view = str(row["condition"]).removeprefix("candidate:")
        matrix[str(row["pair_key"])][view] = row
    if not matrix:
        return {}
    expected = set(next(iter(matrix.values())))
    for group, values in matrix.items():
        if set(values) != expected:
            raise ValueError(f"incomplete candidate matrix group {group}")
    return dict(matrix)


def _candidate_base_group(row: Mapping[str, Any]) -> str:
    return f"{row['suite']}::{row['base_task']}"


def _candidate_base_groups(
    matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    groups: Sequence[str],
) -> set[str]:
    return {_candidate_base_group(matrix[group]["canonical"]) for group in groups}


def _calibration_split(
    matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    groups: Sequence[str],
) -> tuple[list[str], list[str]]:
    by_suite: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        row = matrix[group]["canonical"]
        by_suite[str(row["suite"])].add(_candidate_base_group(row))
    calibration = []
    holdout = []
    for suite, values in sorted(by_suite.items()):
        ranked = sorted(
            values,
            key=lambda value: hashlib.sha256(f"split::{value}".encode()).hexdigest(),
        )
        count = max(1, min(len(ranked) - 1, math.ceil(0.6 * len(ranked)))) if len(ranked) > 1 else 1
        calibration.extend(ranked[:count])
        holdout.extend(ranked[count:])
    calibration_set = set(calibration)
    holdout_set = set(holdout)
    return (
        [group for group in groups if _candidate_base_group(matrix[group]["canonical"]) in calibration_set],
        [group for group in groups if _candidate_base_group(matrix[group]["canonical"]) in holdout_set],
    )


def _cluster_candidate_values(
    matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    groups: Sequence[str],
    values: Mapping[str, float],
) -> list[float]:
    clustered: dict[str, list[float]] = defaultdict(list)
    for group in groups:
        clustered[_candidate_base_group(matrix[group]["canonical"])].append(float(values[group]))
    return [float(np.mean(clustered[key])) for key in sorted(clustered)]


def _view_success(matrix: Mapping[str, Mapping[str, Mapping[str, Any]]], groups: Sequence[str], view: str) -> float:
    values = {group: float(bool(matrix[group][view]["success"])) for group in groups}
    return float(np.mean(_cluster_candidate_values(matrix, groups, values)))


def _rank_views(
    matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    groups: Sequence[str],
) -> list[str]:
    views = list(next(iter(matrix.values())))
    return sorted(
        views,
        key=lambda view: (
            -_view_success(matrix, groups, view),
            float(np.mean([matrix[group][view]["completion_steps"] for group in groups])),
            view != "canonical",
            view,
        ),
    )


def _selector_delta(
    matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    groups: Sequence[str],
    selections: Mapping[str, str],
) -> dict[str, Any]:
    selected = {
        group: float(bool(matrix[group][selections[group]]["success"])) for group in groups
    }
    canonical = {group: float(bool(matrix[group]["canonical"]["success"])) for group in groups}
    selected_success = _cluster_candidate_values(matrix, groups, selected)
    differences = _cluster_candidate_values(
        matrix,
        groups,
        {group: selected[group] - canonical[group] for group in groups},
    )
    result = bootstrap_mean(differences)
    return {
        "success": float(np.mean(selected_success)),
        "success_ci95": bootstrap_mean(selected_success)["ci95"],
        "minus_canonical": result,
        "selected_view_counts": dict(
            sorted(
                {
                    view: sum(selected == view for selected in selections.values())
                    for view in set(selections.values())
                }.items()
            )
        ),
    }


def _selection_difference(
    matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    groups: Sequence[str],
    selections: Mapping[str, str],
    baseline: Mapping[str, str],
) -> dict[str, Any]:
    values = {
        group: float(bool(matrix[group][selections[group]]["success"]))
        - float(bool(matrix[group][baseline[group]]["success"]))
        for group in groups
    }
    return bootstrap_mean(_cluster_candidate_values(matrix, groups, values))


def _minimum_uncertainty(row: Mapping[str, Any]) -> float:
    value = row["initial_metrics"]["action_probe"]["mean_pairwise_rms"]
    return float(value) if value is not None else float("inf")


def _image_score(row: Mapping[str, Any]) -> float:
    metrics = row["initial_metrics"]["agent"]
    return (
        float(metrics["entropy_32bin_bits"])
        + 0.02 * float(metrics["mean_edge_strength"])
        - 2.0 * float(metrics["clipped_fraction"])
    )


def _candidate_visibility_by_view(
    matrix: Mapping[str, Mapping[str, Mapping[str, Any]]],
    groups: Sequence[str],
) -> dict[str, Any] | None:
    views = list(next(iter(matrix.values())))
    if not any(
        "sim_visibility" in matrix[group][view].get("initial_metrics", {})
        for group in groups
        for view in views
    ):
        return None
    result = {}
    for view in views:
        out_of_frame = {}
        below_16px = {}
        border_touch = {}
        retention = {}
        for group in groups:
            visibility = matrix[group][view]["initial_metrics"]["sim_visibility"]
            canonical = matrix[group]["canonical"]["initial_metrics"]["sim_visibility"]
            out_of_frame[group] = float(not visibility["all_interest_visible"])
            below_16px[group] = float(
                not visibility["all_interest_visible_at_least_16px"]
            )
            border_touch[group] = float(visibility["any_interest_border_touch"])
            canonical_pixels = float(canonical["minimum_interest_pixel_count"])
            retention[group] = (
                float(visibility["minimum_interest_pixel_count"]) / canonical_pixels
                if canonical_pixels > 0
                else 0.0
            )
        result[view] = {
            "any_interest_out_of_frame_rate": bootstrap_mean(
                _cluster_candidate_values(matrix, groups, out_of_frame)
            ),
            "any_interest_below_16px_rate": bootstrap_mean(
                _cluster_candidate_values(matrix, groups, below_16px)
            ),
            "any_interest_border_touch_rate": bootstrap_mean(
                _cluster_candidate_values(matrix, groups, border_touch)
            ),
            "minimum_interest_pixel_retention": bootstrap_mean(
                _cluster_candidate_values(matrix, groups, retention)
            ),
        }
    return result


def summarize_candidates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    matrix = _candidate_matrix(rows)
    if not matrix:
        return None
    groups = sorted(matrix)
    calibration, holdout = _calibration_split(matrix, groups)
    if not holdout:
        raise ValueError("candidate analysis needs at least two base-task groups per suite")
    view_ranking = _rank_views(matrix, calibration)
    fixed_view_success = {
        view: {
            "calibration": _view_success(matrix, calibration, view),
            "holdout": _view_success(matrix, holdout, view),
            "holdout_ci95": bootstrap_mean(
                _cluster_candidate_values(
                    matrix,
                    holdout,
                    {
                        group: float(bool(matrix[group][view]["success"]))
                        for group in holdout
                    },
                )
            )["ci95"],
        }
        for view in view_ranking
    }
    global_view = view_ranking[0]
    global_selection = {group: global_view for group in holdout}

    suite_views = {}
    for suite in sorted({str(matrix[group]["canonical"]["suite"]) for group in groups}):
        suite_calibration = [
            group
            for group in calibration
            if str(matrix[group]["canonical"]["suite"]) == suite
        ]
        suite_views[suite] = _rank_views(matrix, suite_calibration)[0]
    suite_selection = {
        group: suite_views[str(matrix[group]["canonical"]["suite"])] for group in holdout
    }

    uncertainty_selection = {
        group: min(
            matrix[group],
            key=lambda view: (
                _minimum_uncertainty(matrix[group][view]),
                view != "canonical",
                view,
            ),
        )
        for group in holdout
    }
    image_selection = {
        group: max(matrix[group], key=lambda view: (_image_score(matrix[group][view]), view == "canonical", view))
        for group in holdout
    }
    oracle_selection = {
        group: max(
            matrix[group],
            key=lambda view: (
                bool(matrix[group][view]["success"]),
                -float(matrix[group][view]["completion_steps"]),
                view == "canonical",
            ),
        )
        for group in holdout
    }

    budget_curves = {}
    ordered = ["canonical", *[view for view in view_ranking if view != "canonical"]]
    for budget in range(1, len(ordered) + 1):
        available = ordered[:budget]
        uncertainty = {
            group: min(available, key=lambda view: (_minimum_uncertainty(matrix[group][view]), view))
            for group in holdout
        }
        oracle = {
            group: max(
                available,
                key=lambda view: (
                    bool(matrix[group][view]["success"]),
                    -float(matrix[group][view]["completion_steps"]),
                ),
            )
            for group in holdout
        }
        budget_curves[str(budget)] = {
            "available_views": available,
            "uncertainty_selector": _selector_delta(matrix, holdout, uncertainty),
            "uncertainty_minus_global_static": _selection_difference(
                matrix, holdout, uncertainty, global_selection
            ),
            "oracle_selector": _selector_delta(matrix, holdout, oracle),
            "observation_cost": budget,
        }

    random_by_instance = {
        group: float(np.mean([bool(row["success"]) for row in matrix[group].values()]))
        for group in holdout
    }
    random_expected = float(
        np.mean(_cluster_candidate_values(matrix, holdout, random_by_instance))
    )
    disagreement_by_instance = {
        group: float(len({bool(row["success"]) for row in matrix[group].values()}) > 1)
        for group in holdout
    }
    disagreement = float(
        np.mean(_cluster_candidate_values(matrix, holdout, disagreement_by_instance))
    )
    active_uncertainty = _selector_delta(matrix, holdout, uncertainty_selection)
    active_uncertainty["minus_global_static"] = _selection_difference(
        matrix, holdout, uncertainty_selection, global_selection
    )
    active_image = _selector_delta(matrix, holdout, image_selection)
    active_image["minus_global_static"] = _selection_difference(
        matrix, holdout, image_selection, global_selection
    )
    oracle = _selector_delta(matrix, holdout, oracle_selection)
    oracle["minus_global_static"] = _selection_difference(
        matrix, holdout, oracle_selection, global_selection
    )
    independent_groups = _candidate_base_groups(matrix, groups)
    calibration_base_groups = _candidate_base_groups(matrix, calibration)
    holdout_base_groups = _candidate_base_groups(matrix, holdout)
    return {
        "independent_base_task_count": len(independent_groups),
        "candidate_initial_state_count": len(groups),
        "calibration_group_count": len(calibration_base_groups),
        "calibration_initial_state_count": len(calibration),
        "holdout_group_count": len(holdout_base_groups),
        "holdout_initial_state_count": len(holdout),
        "view_ranking_from_calibration": view_ranking,
        "fixed_view_success": fixed_view_success,
        "fixed_view_visibility": _candidate_visibility_by_view(matrix, holdout),
        "canonical_holdout_success": _view_success(matrix, holdout, "canonical"),
        "random_view_expected_success": random_expected,
        "view_outcome_disagreement_rate": disagreement,
        "global_static_selection": {
            "selected_view": global_view,
            **_selector_delta(matrix, holdout, global_selection),
        },
        "suite_static_selection": {
            "selected_views": suite_views,
            **_selector_delta(matrix, holdout, suite_selection),
        },
        "active_uncertainty_selection_all_views": active_uncertainty,
        "active_image_quality_selection_all_views": active_image,
        "oracle_selection_all_views": oracle,
        "active_observation_budget": budget_curves,
    }


def build_quantification_gates(
    gap: Mapping[str, Any] | None,
    candidates: Mapping[str, Any] | None,
    *,
    canonical_threshold: float = 0.70,
) -> dict[str, Any]:
    canonical_success = (
        float(gap["overall"]["canonical_success"])
        if gap is not None
        else (
            float(candidates["canonical_holdout_success"])
            if candidates is not None
            else 0.0
        )
    )
    baseline_valid = canonical_success >= canonical_threshold
    view_gap_confirmed = bool(
        baseline_valid
        and gap is not None
        and float(gap["overall"]["view_generalization_gap"]["ci95"][0]) > 0.0
    )
    static_view_gain = False
    active_selector_gain = False
    oracle_headroom = False
    if baseline_valid and candidates is not None:
        global_static = candidates["global_static_selection"]["minus_canonical"]
        suite_static = candidates["suite_static_selection"]["minus_canonical"]
        static_view_gain = bool(
            float(global_static["ci95"][0]) > 0.0
            or float(suite_static["ci95"][0]) > 0.0
        )
        active_selector_gain = bool(
            float(
                candidates["active_uncertainty_selection_all_views"]
                ["minus_global_static"]["ci95"][0]
            )
            > 0.0
        )
        oracle_headroom = bool(
            float(candidates["oracle_selection_all_views"]["minus_global_static"]["mean"])
            > 0.0
        )
    return {
        "canonical_threshold": canonical_threshold,
        "canonical_success": canonical_success,
        "BASELINE_VALID": baseline_valid,
        "VIEW_GAP_CONFIRMED": view_gap_confirmed,
        "STATIC_VIEW_GAIN": static_view_gain,
        "ACTIVE_SELECTOR_GAIN": active_selector_gain,
        "ORACLE_HEADROOM": oracle_headroom,
    }


def render_summary(report: Mapping[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    gap = report.get("view_generalization")
    if gap:
        difficulties = sorted(gap["by_difficulty"], key=int)
        canonical = [gap["by_difficulty"][key]["canonical_success"] for key in difficulties]
        camera = [gap["by_difficulty"][key]["official_camera_success"] for key in difficulties]
        x = np.arange(len(difficulties))
        axes[0].plot(x, canonical, marker="o", label="Canonical")
        axes[0].plot(x, camera, marker="o", label="Plus camera")
        axes[0].set_xticks(x, difficulties)
        axes[0].set_xlabel("LIBERO-Plus difficulty")
        axes[0].set_ylabel("Task success")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("View generalization gap")
        axes[0].legend()
    candidates = report.get("view_optimization_and_active_sensing")
    if candidates:
        ranking = candidates["view_ranking_from_calibration"]
        values = [candidates["fixed_view_success"][view]["holdout"] for view in ranking]
        axes[1].bar(np.arange(len(ranking)), values, color="#3976af")
        axes[1].set_xticks(np.arange(len(ranking)), ranking, rotation=35, ha="right")
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Holdout task success")
        axes[1].set_title("Static camera candidates")
        visibility = candidates.get("fixed_view_visibility")
        if visibility:
            visibility_axis = axes[1].twinx()
            out_of_frame = [
                visibility[view]["any_interest_out_of_frame_rate"]["mean"]
                for view in ranking
            ]
            visibility_axis.plot(
                np.arange(len(ranking)),
                out_of_frame,
                color="#b4423c",
                marker="x",
                linestyle="--",
                label="Object out of frame",
            )
            visibility_axis.set_ylim(0, 1)
            visibility_axis.set_ylabel("Out-of-frame rate")
        budgets = candidates["active_observation_budget"]
        budget_x = sorted(map(int, budgets))
        uncertainty = [budgets[str(value)]["uncertainty_selector"]["success"] for value in budget_x]
        oracle = [budgets[str(value)]["oracle_selector"]["success"] for value in budget_x]
        axes[2].plot(budget_x, uncertainty, marker="o", label="Action-uncertainty selector")
        axes[2].plot(budget_x, oracle, marker="o", label="Oracle upper bound")
        axes[2].axhline(candidates["canonical_holdout_success"], color="black", linestyle="--", label="Canonical")
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel("Views observed before acting")
        axes[2].set_ylabel("Holdout task success")
        axes[2].set_title("Active sensing budget")
        axes[2].legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_chinese_report(report: Mapping[str, Any], output: Path) -> None:
    gap = report.get("view_generalization")
    candidates = report.get("view_optimization_and_active_sensing")
    lines = [
        "# Pi0.5 × LIBERO-Plus 视角缺口、视角优化与主动感知量化",
        "",
        (
            "> 本报告以 `suite × 基础任务` 为独立统计单位，"
            "Plus 同一任务的多个视角不会被当成独立样本扩大显著性。"
        ),
        "",
    ]
    gates = report["quantification_gates"]
    lines.extend(
        [
            "## 自动判定",
            "",
            f"- Canonical 有效性：{'通过' if gates['BASELINE_VALID'] else '失败'} "
            f"({gates['canonical_success']:.1%} / 门槛 {gates['canonical_threshold']:.1%})",
            f"- 视角泛化缺口成立：{'是' if gates['VIEW_GAP_CONFIRMED'] else '否'}",
            f"- 固定视角优化在留出任务上成立：{'是' if gates['STATIC_VIEW_GAIN'] else '否'}",
            f"- 主动选择显著优于最佳固定视角：{'是' if gates['ACTIVE_SELECTOR_GAIN'] else '否'}",
            f"- 仍存在逐任务视角上界空间：{'是' if gates['ORACLE_HEADROOM'] else '否'}",
            "",
        ]
    )
    if gap:
        overall = gap["overall"]
        gap_ci = overall["view_generalization_gap"]["ci95"]
        robustness_line = (
            f"- 相机鲁棒率：{overall['robustness_ratio']:.1%}"
            if overall["robustness_ratio"] is not None
            else "- 相机鲁棒率：无法计算"
        )
        lines.extend(
            [
                "## 视角泛化缺口",
                "",
                f"- Canonical 成功率：{overall['canonical_success']:.1%}",
                f"- 官方相机扰动成功率：{overall['official_camera_success']:.1%}",
                f"- 绝对缺口：{overall['view_generalization_gap']['mean']:.1%}",
                f"- 配对 95% CI：[{gap_ci[0]:.1%}, {gap_ci[1]:.1%}]",
                robustness_line,
                "",
                "| 难度 | Canonical | 相机扰动 | 缺口 |",
                "|---:|---:|---:|---:|",
            ]
        )
        for difficulty, value in sorted(gap["by_difficulty"].items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {difficulty} | {value['canonical_success']:.1%} | "
                f"{value['official_camera_success']:.1%} | {value['view_generalization_gap']['mean']:.1%} |"
            )
        lines.append("")
        visibility = gap.get("visibility_diagnostics")
        if visibility:
            visible_gap = visibility["gap_when_all_interest_objects_clearly_visible"]
            lost_gap = visibility["gap_when_camera_loses_clear_visibility"]
            visible_gap_line = (
                "- 关键物体仍清楚可见时的视角缺口："
                f"{visible_gap['view_generalization_gap']['mean']:.1%}"
                if visible_gap
                else "- 关键物体仍清楚可见时的视角缺口：无可用配对"
            )
            lost_gap_line = (
                "- 丢失清楚可见性后的视角缺口："
                f"{lost_gap['view_generalization_gap']['mean']:.1%}"
                if lost_gap
                else "- 丢失清楚可见性后的视角缺口：无可用配对"
            )
            lines.extend(
                [
                    "### 目标可见性边界",
                    "",
                    (
                        "- 至少一个关键物体完全出画率："
                        f"{visibility['camera_any_interest_out_of_frame_rate']['mean']:.1%}"
                    ),
                    (
                        "- 至少一个关键物体低于 16 像素率："
                        f"{visibility['camera_any_interest_below_16px_rate']['mean']:.1%}"
                    ),
                    (
                        "- 关键物体触碰画面边缘率："
                        f"{visibility['camera_any_interest_border_touch_rate']['mean']:.1%}"
                    ),
                    (
                        "- 最小关键物体像素保留比例："
                        f"{visibility['minimum_interest_pixel_retention']['mean']:.1%}"
                    ),
                    visible_gap_line,
                    lost_gap_line,
                    "",
                    "这里的分割真值只用于事后诊断，不进入策略或视角选择器。",
                    "",
                ]
            )
    if candidates:
        active = candidates["active_uncertainty_selection_all_views"]
        oracle = candidates["oracle_selection_all_views"]
        states_per_view = (
            candidates["candidate_initial_state_count"]
            // candidates["independent_base_task_count"]
        )
        lines.extend(
            [
                "## 视角优化与主动感知",
                "",
                (
                    f"- 独立基础任务：{candidates['independent_base_task_count']}"
                    f"（校准 {candidates['calibration_group_count']} / "
                    f"留出测试 {candidates['holdout_group_count']}）"
                ),
                f"- 每个候选视角的配对初始状态：{states_per_view}",
                f"- Holdout canonical：{candidates['canonical_holdout_success']:.1%}",
                (
                    "- 校准集选择的全局固定视角 "
                    f"`{candidates['global_static_selection']['selected_view']}`："
                    f"{candidates['global_static_selection']['success']:.1%}"
                ),
                f"- 按 suite 选择固定视角：{candidates['suite_static_selection']['success']:.1%}",
                f"- 动作不确定性主动选择：{active['success']:.1%}",
                (
                    f"  - 相对 canonical：{active['minus_canonical']['mean']:+.1%}，"
                    f"95% CI [{active['minus_canonical']['ci95'][0]:+.1%}, "
                    f"{active['minus_canonical']['ci95'][1]:+.1%}]"
                ),
                (
                    "  - 相对最佳全局固定视角："
                    f"{active['minus_global_static']['mean']:+.1%}，"
                    f"95% CI [{active['minus_global_static']['ci95'][0]:+.1%}, "
                    f"{active['minus_global_static']['ci95'][1]:+.1%}]"
                ),
                f"- 图像质量主动选择：{candidates['active_image_quality_selection_all_views']['success']:.1%}",
                f"- 每个任务事后最佳视角上界：{oracle['success']:.1%}",
                f"  - 相对最佳全局固定视角的可追回空间：{oracle['minus_global_static']['mean']:+.1%}",
                f"- 不同视角会改变成败的任务比例：{candidates['view_outcome_disagreement_rate']:.1%}",
                "",
                (
                    "Oracle 只表示传感器重定位的可用上界，不是可部署算法；"
                    "动作方差选择器才是不读取环境真值的主动感知基线。若 Oracle 有增益而"
                    "可部署选择器没有，说明视角存在价值，但当前置信度指标不足。"
                ),
                "",
            ]
        )
        visibility = candidates.get("fixed_view_visibility")
        if visibility:
            lines.extend(
                [
                    "| 候选视角 | 留出成功率 | 关键物体出画率 | 最小像素保留率 |",
                    "|---|---:|---:|---:|",
                ]
            )
            for view in candidates["view_ranking_from_calibration"]:
                lines.append(
                    f"| `{view}` | {candidates['fixed_view_success'][view]['holdout']:.1%} | "
                    f"{visibility[view]['any_interest_out_of_frame_rate']['mean']:.1%} | "
                    f"{visibility[view]['minimum_interest_pixel_retention']['mean']:.1%} |"
                )
            lines.append("")
    lines.extend(
        [
            "## 解释边界",
            "",
            "- `视角优化` 指执行前选择一个固定外部相机位姿，随后整条轨迹保持不变。",
            (
                "- `主动感知` 指在不推进机器人动力学的情况下查看有限个候选相机画面，"
                "再根据策略不确定性选择执行视角。"
            ),
            (
                "- 当前仿真允许瞬时移动外部相机，因此结果量化的是视觉信息价值；"
                "真实机器人还需加入相机运动时间、可达性和碰撞约束。"
            ),
            (
                "- 若 canonical baseline 本身无效，本报告只能输出 `BASELINE_INVALID`，"
                "不能把失败归因于视角。"
            ),
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired Pi0.5 LIBERO-Plus camera evaluations")
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    for output in (args.output_json, args.output_figure, args.output_report):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite analysis: {output}")
    rows = read_rows(args.episodes)
    gap = summarize_gap(rows)
    candidates = summarize_candidates(rows)
    report = {
        "schema_version": 1,
        "study": "pi05_libero_plus_view_generalization_and_active_sensing",
        "episode_count": len(rows),
        "quantification_gates": build_quantification_gates(gap, candidates),
        "view_generalization": gap,
        "view_optimization_and_active_sensing": candidates,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_name(f".{args.output_json.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output_json)
    render_summary(report, args.output_figure)
    write_chinese_report(report, args.output_report)
    print(json.dumps({"output": str(args.output_json), "episode_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
