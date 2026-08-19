from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def load_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("total must be positive")
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _rate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in rows)
    total = len(rows)
    if total == 0:
        raise ValueError("cannot summarize an empty group")
    return {
        "success_count": successes,
        "task_count": total,
        "success_rate": successes / total,
        "wilson_ci95": wilson_interval(successes, total),
    }


def _base_task_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = f"{row['suite']}::{row['base_task']}"
        grouped[key].append(float(bool(row["success"])))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260819,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("bootstrap requires at least one value")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "independent_group_count": int(array.size),
        "bootstrap_resamples": resamples,
    }


def _group_summary(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {key: _rate(grouped[key]) for key in sorted(grouped)}


def validate_camera_rows(rows: Sequence[Mapping[str, Any]], *, expected_count: int | None) -> None:
    if not rows:
        raise ValueError("no episode rows")
    episode_ids = [str(row["episode_id"]) for row in rows]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("duplicate episode IDs")
    keys = [
        (str(row["suite"]), int(row["official_task_id"]), int(row["init_state_index"]))
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate official camera task keys")
    if any(row.get("condition") != "official_camera" for row in rows):
        raise ValueError("camera-full input contains a non-camera condition")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} tasks, found {len(rows)}")


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int | None = 1599,
) -> dict[str, Any]:
    validate_camera_rows(rows, expected_count=expected_count)
    base_values = _base_task_values(rows)
    return {
        "schema_version": 1,
        "benchmark": "LIBERO-Plus Camera Full",
        "official_pooled": _rate(rows),
        "base_task_macro": bootstrap_mean_ci(list(base_values.values())),
        "independent_base_task_count": len(base_values),
        "by_suite": _group_summary(rows, "suite"),
        "by_difficulty": _group_summary(rows, "difficulty_level"),
        "by_perturbation_family": _group_summary(rows, "perturbation_family"),
    }


def _paired_rows(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    def keyed(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int, int], Mapping[str, Any]]:
        return {
            (str(row["suite"]), int(row["official_task_id"]), int(row["init_state_index"])): row
            for row in rows
        }

    left = keyed(baseline)
    right = keyed(candidate)
    if set(left) != set(right):
        raise ValueError("baseline and candidate task keys differ")
    paired = []
    for key in sorted(left):
        base = left[key]
        cand = right[key]
        if str(base["base_task"]) != str(cand["base_task"]):
            raise ValueError(f"base-task mismatch for {key}")
        paired.append(
            {
                "suite": str(base["suite"]),
                "base_task": str(base["base_task"]),
                "difficulty_level": int(base["difficulty_level"]),
                "perturbation_family": str(base["perturbation_family"]),
                "difference": float(bool(cand["success"])) - float(bool(base["success"])),
            }
        )
    return paired


def compare_rows(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    paired = _paired_rows(baseline, candidate)
    by_base: dict[str, list[float]] = defaultdict(list)
    for row in paired:
        by_base[f"{row['suite']}::{row['base_task']}"] .append(float(row["difference"]))
    return {
        "candidate_minus_baseline_pooled": float(
            np.mean([row["difference"] for row in paired])
        ),
        "candidate_minus_baseline_base_task_cluster": bootstrap_mean_ci(
            [float(np.mean(values)) for values in by_base.values()]
        ),
        "paired_task_count": len(paired),
        "independent_base_task_count": len(by_base),
    }


def write_report(path: Path, summary: Mapping[str, Any], comparison: Mapping[str, Any] | None) -> None:
    pooled = summary["official_pooled"]
    lines = [
        "# LIBERO-Plus Camera Full",
        "",
        f"- Tasks: {pooled['task_count']}",
        f"- Success: {pooled['success_count']}/{pooled['task_count']} ({pooled['success_rate']:.2%})",
        f"- Wilson 95% CI: [{pooled['wilson_ci95'][0]:.2%}, {pooled['wilson_ci95'][1]:.2%}]",
        f"- Independent base tasks: {summary['independent_base_task_count']}",
    ]
    if comparison is not None:
        clustered = comparison["candidate_minus_baseline_base_task_cluster"]
        lines.extend(
            [
                "",
                "## Paired comparison",
                "",
                f"- Pooled delta: {comparison['candidate_minus_baseline_pooled']:+.2%}",
                (
                    "- Base-task clustered delta: "
                    f"{clustered['mean']:+.2%} "
                    f"[{clustered['ci95'][0]:+.2%}, {clustered['ci95'][1]:+.2%}]"
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate LIBERO-Plus Camera Full episodes")
    parser.add_argument("--episodes", nargs="+", type=Path, required=True)
    parser.add_argument("--baseline-episodes", nargs="+", type=Path)
    parser.add_argument("--expected-count", type=int, default=1599)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.episodes)
    summary = summarize_rows(rows, expected_count=args.expected_count)
    comparison = None
    if args.baseline_episodes:
        baseline = load_rows(args.baseline_episodes)
        validate_camera_rows(baseline, expected_count=args.expected_count)
        comparison = compare_rows(baseline, rows)
    payload = {"summary": summary, "comparison": comparison}
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.output_report, summary, comparison)


if __name__ == "__main__":
    main()
