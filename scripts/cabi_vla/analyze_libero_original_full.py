from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from analyze_libero_plus_camera_full import bootstrap_mean_ci, load_rows, wilson_interval


EXPECTED_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _rate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty group")
    successes = sum(bool(row["success"]) for row in rows)
    return {
        "success_count": successes,
        "episode_count": len(rows),
        "success_rate": successes / len(rows),
        "wilson_ci95": wilson_interval(successes, len(rows)),
    }


def _task_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['suite']}::{row['base_task']}"] .append(float(bool(row["success"])))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int | None,
    expected_trials_per_task: int | None = None,
) -> None:
    if not rows:
        raise ValueError("no episode rows")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} episodes, found {len(rows)}")
    if any(row.get("condition") != "canonical" for row in rows):
        raise ValueError("Original Full input contains a non-canonical condition")
    episode_ids = [str(row["episode_id"]) for row in rows]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("duplicate episode IDs")
    keys = [
        (str(row["suite"]), str(row["base_task"]), int(row["init_state_index"]))
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate task/init-state keys")
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for suite, base_task, init_state_index in keys:
        grouped[(suite, base_task)].append(init_state_index)
    if expected_trials_per_task is not None:
        expected_indices = set(range(expected_trials_per_task))
        for key, indices in grouped.items():
            if set(indices) != expected_indices:
                raise ValueError(f"incomplete initial-state ledger for {key}")


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int | None = 2000,
    expected_trials_per_task: int | None = 50,
) -> dict[str, Any]:
    validate_rows(
        rows,
        expected_count=expected_count,
        expected_trials_per_task=expected_trials_per_task,
    )
    by_suite_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_suite_rows[str(row["suite"])].append(row)
    by_suite = {suite: _rate(by_suite_rows[suite]) for suite in EXPECTED_SUITES}
    task_values = _task_values(rows)
    return {
        "schema_version": 1,
        "benchmark": "Original LIBERO Full",
        "pooled": _rate(rows),
        "suite_macro_average": float(
            np.mean([by_suite[suite]["success_rate"] for suite in EXPECTED_SUITES])
        ),
        "task_macro": bootstrap_mean_ci(list(task_values.values())),
        "independent_task_count": len(task_values),
        "by_suite": by_suite,
    }


def compare_rows(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def keyed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
        return {
            (str(row["suite"]), str(row["base_task"]), int(row["init_state_index"])): row
            for row in rows
        }

    left = keyed(baseline)
    right = keyed(candidate)
    if set(left) != set(right):
        raise ValueError("baseline and candidate task/init-state keys differ")
    by_task: dict[str, list[float]] = defaultdict(list)
    all_differences = []
    for key in sorted(left):
        difference = float(bool(right[key]["success"])) - float(bool(left[key]["success"]))
        by_task[f"{key[0]}::{key[1]}"] .append(difference)
        all_differences.append(difference)
    return {
        "candidate_minus_baseline_pooled": float(np.mean(all_differences)),
        "candidate_minus_baseline_task_cluster": bootstrap_mean_ci(
            [float(np.mean(values)) for values in by_task.values()]
        ),
        "paired_episode_count": len(all_differences),
        "independent_task_count": len(by_task),
    }


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    comparison: Mapping[str, Any] | None,
) -> None:
    pooled = summary["pooled"]
    lines = [
        "# Original LIBERO Full",
        "",
        f"- Episodes: {pooled['episode_count']}",
        f"- Success: {pooled['success_count']}/{pooled['episode_count']} ({pooled['success_rate']:.2%})",
        f"- Suite macro average: {summary['suite_macro_average']:.2%}",
        f"- Independent tasks: {summary['independent_task_count']}",
    ]
    if comparison is not None:
        clustered = comparison["candidate_minus_baseline_task_cluster"]
        lines.extend(
            [
                "",
                "## Paired comparison",
                "",
                f"- Pooled delta: {comparison['candidate_minus_baseline_pooled']:+.2%}",
                (
                    "- Task-clustered delta: "
                    f"{clustered['mean']:+.2%} "
                    f"[{clustered['ci95'][0]:+.2%}, {clustered['ci95'][1]:+.2%}]"
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Original LIBERO Full episodes")
    parser.add_argument("--episodes", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline-episodes", nargs="+", type=Path)
    parser.add_argument("--expected-count", type=int, default=2000)
    parser.add_argument("--expected-trials-per-task", type=int, default=50)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.episodes)
    summary = summarize_rows(
        rows,
        expected_count=args.expected_count,
        expected_trials_per_task=args.expected_trials_per_task,
    )
    comparison = None
    if args.baseline_episodes:
        baseline = load_rows(args.baseline_episodes)
        validate_rows(
            baseline,
            expected_count=args.expected_count,
            expected_trials_per_task=args.expected_trials_per_task,
        )
        comparison = compare_rows(baseline, rows)
    payload = {"summary": summary, "comparison": comparison}
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.output_report, summary, comparison)


if __name__ == "__main__":
    main()
