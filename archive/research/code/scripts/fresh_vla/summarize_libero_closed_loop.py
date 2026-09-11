from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scripts.fresh_vla.paired_evaluation import bootstrap_summary
except ModuleNotFoundError:
    from paired_evaluation import bootstrap_summary


METHODS = (
    "full_h",
    "random_soft010",
    "shuffled_oracle_soft010",
    "gripper_soft010",
    "oracle_soft010",
    "short_h",
)
METRICS = (
    "overall_task_success",
    "attached_task_success",
    "slip_recovery_success",
    "isolated_recovery_success",
    "failure_continuation_rate",
    "premature_commitment_rate",
    "recovery_switch_latency",
    "slip_regrasp_success",
    "isolated_regrasp_success",
    "drop_rate",
    "attached_drop_rate",
    "slip_drop_rate",
    "final_progress",
    "attached_final_progress",
    "slip_final_progress",
    "progress_auc",
    "grasp_subgoal_rate",
    "lift_subgoal_rate",
    "transport_subgoal_rate",
    "place_subgoal_rate",
    "event_trigger_rate",
    "completion_steps",
    "normal_no_intervention_success",
    "deterministic_reach_success",
)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _bootstrap_or_empty(values: Sequence[float], *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "standard_error": None,
            "bootstrap_95_low": None,
            "bootstrap_95_high": None,
        }
    return bootstrap_summary(values, bootstrap_samples=bootstrap_samples, seed=seed)


def group_metrics(
    isolated_rows: Sequence[Mapping[str, Any]],
    end_to_end_rows: Sequence[Mapping[str, Any]],
    deterministic_rows: Sequence[Mapping[str, Any]] = (),
    *,
    execution_horizon: int,
) -> dict[str, dict[str, float | None]]:
    isolated_by_group: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    e2e_by_group: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in isolated_rows:
        if int(row["execution_horizon"]) == execution_horizon:
            isolated_by_group[str(row["pair_id"])][str(row["branch_outcome"])] = row
    for row in end_to_end_rows:
        if int(row["execution_horizon"]) == execution_horizon:
            e2e_by_group[str(row["pair_id"])][str(row["branch_outcome"])] = row
    if isolated_by_group.keys() != e2e_by_group.keys():
        raise ValueError("isolated and end-to-end snapshot groups do not match")
    deterministic_by_group = {
        str(row["pair_id"]): row
        for row in deterministic_rows
        if int(row["execution_horizon"]) == execution_horizon
    }
    if deterministic_rows and deterministic_by_group.keys() != e2e_by_group.keys():
        raise ValueError("deterministic and end-to-end snapshot groups do not match")

    result = {}
    for pair_id in sorted(e2e_by_group):
        isolated = isolated_by_group[pair_id]
        e2e = e2e_by_group[pair_id]
        if isolated.keys() < {"attached", "slipped"} or e2e.keys() < {"attached", "slipped"}:
            raise ValueError(f"missing branch row for {pair_id}")
        attached = e2e["attached"]
        slipped = e2e["slipped"]
        result[pair_id] = {
            "overall_task_success": 0.5 * (float(attached["success"]) + float(slipped["success"])),
            "attached_task_success": float(attached["success"]),
            "slip_recovery_success": float(slipped["recovery_success"]),
            "isolated_recovery_success": float(isolated["slipped"]["recovery_success"]),
            "failure_continuation_rate": (
                None if slipped["failure_continuation"] is None else float(slipped["failure_continuation"])
            ),
            "premature_commitment_rate": (
                None if slipped["premature_commitment"] is None else float(slipped["premature_commitment"])
            ),
            "recovery_switch_latency": None
            if slipped["recovery_switch_latency"] is None
            else float(slipped["recovery_switch_latency"]),
            "slip_regrasp_success": (
                None if slipped.get("regrasp_success") is None else float(slipped["regrasp_success"])
            ),
            "isolated_regrasp_success": (
                None
                if isolated["slipped"].get("regrasp_success") is None
                else float(isolated["slipped"]["regrasp_success"])
            ),
            "drop_rate": 0.5 * (float(attached["drop"]) + float(slipped["drop"])),
            "attached_drop_rate": float(attached["drop"]),
            "slip_drop_rate": float(slipped["drop"]),
            "final_progress": 0.5
            * (float(attached["final_progress"]) + float(slipped["final_progress"])),
            "attached_final_progress": float(attached["final_progress"]),
            "slip_final_progress": float(slipped["final_progress"]),
            "progress_auc": 0.5 * (float(attached["progress_auc"]) + float(slipped["progress_auc"])),
            "grasp_subgoal_rate": 0.5
            * (float(attached["grasp_subgoal"]) + float(slipped["grasp_subgoal"])),
            "lift_subgoal_rate": 0.5
            * (float(attached["lift_subgoal"]) + float(slipped["lift_subgoal"])),
            "transport_subgoal_rate": 0.5
            * (float(attached["transport_subgoal"]) + float(slipped["transport_subgoal"])),
            "place_subgoal_rate": 0.5
            * (float(attached["place_subgoal"]) + float(slipped["place_subgoal"])),
            "event_trigger_rate": 0.5
            * (float(attached.get("event_time") is not None) + float(slipped.get("event_time") is not None)),
            "completion_steps": 0.5
            * (float(attached["completion_steps"]) + float(slipped["completion_steps"])),
            "normal_no_intervention_success": float(attached["success"]),
            "deterministic_reach_success": (
                None if not deterministic_rows else float(deterministic_by_group[pair_id]["success"])
            ),
        }
    return result


def summarize_seed(groups: Mapping[str, Mapping[str, float | None]]) -> dict[str, float | None]:
    return {
        metric: _mean([float(row[metric]) for row in groups.values() if row[metric] is not None])
        for metric in METRICS
    }


def paired_group_delta(
    baseline: Mapping[int, Mapping[str, Mapping[str, float | None]]],
    candidate: Mapping[int, Mapping[str, Mapping[str, float | None]]],
    metric: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if baseline.keys() != candidate.keys():
        raise ValueError("paired seeds do not match")
    per_group: dict[str, list[float]] = defaultdict(list)
    for run_seed in sorted(baseline):
        if baseline[run_seed].keys() != candidate[run_seed].keys():
            raise ValueError(f"paired groups do not match for seed {run_seed}")
        for pair_id in baseline[run_seed]:
            base_value = baseline[run_seed][pair_id][metric]
            candidate_value = candidate[run_seed][pair_id][metric]
            if base_value is None or candidate_value is None:
                continue
            per_group[pair_id].append(float(candidate_value) - float(base_value))
    group_deltas = [float(np.mean(values)) for _, values in sorted(per_group.items())]
    summary = _bootstrap_or_empty(group_deltas, bootstrap_samples=bootstrap_samples, seed=seed)
    return {
        "unit": "snapshot_group_after_averaging_seeds",
        "group_count": len(group_deltas),
        "candidate_minus_baseline": summary,
        "group_deltas": group_deltas,
    }


def _aggregate_seed_summaries(summaries: Sequence[Mapping[str, float | None]]) -> dict[str, Any]:
    result = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in summaries if row[metric] is not None]
        result[metric] = {
            "mean": _mean(values),
            "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "seed_values": values,
        }
    return result


def group_bootstrap_across_seeds(
    groups_by_seed: Mapping[int, Mapping[str, Mapping[str, float | None]]],
    metric: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    per_group: dict[str, list[float]] = defaultdict(list)
    for run_seed in sorted(groups_by_seed):
        for pair_id, row in groups_by_seed[run_seed].items():
            value = row.get(metric)
            if value is not None:
                per_group[pair_id].append(float(value))
    group_values = [float(np.mean(values)) for _, values in sorted(per_group.items())]
    summary = _bootstrap_or_empty(group_values, bootstrap_samples=bootstrap_samples, seed=seed)
    return {
        "unit": "snapshot_group_after_averaging_seeds",
        "group_count": len(group_values),
        **summary,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_ci(summary: Mapping[str, Any]) -> str:
    if summary.get("mean") is None:
        return "n/a"
    return (
        f"{float(summary['mean']):.3f} "
        f"[{float(summary['bootstrap_95_low']):.3f}, {float(summary['bootstrap_95_high']):.3f}]"
    )


def _fmt_pp(summary: Mapping[str, Any]) -> str:
    if summary.get("mean") is None:
        return "n/a"
    return (
        f"{100.0 * float(summary['mean']):+.1f} "
        f"[{100.0 * float(summary['bootstrap_95_low']):+.1f}, "
        f"{100.0 * float(summary['bootstrap_95_high']):+.1f}] pp"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-level statistics for FRESH fixed-K closed-loop evaluation")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--execution-horizons", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.runs_root / "closed_loop_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_groups: dict[int, dict[str, dict[int, dict[str, dict[str, float | None]]]]] = {}
    seed_summaries = {}
    aggregate = {}
    group_bootstrap = {}
    comparisons = {}

    for execution_horizon in args.execution_horizons:
        all_groups[execution_horizon] = {}
        seed_summaries[str(execution_horizon)] = {}
        aggregate[str(execution_horizon)] = {}
        group_bootstrap[str(execution_horizon)] = {}
        for method in METHODS:
            all_groups[execution_horizon][method] = {}
            summaries = []
            for seed in args.seeds:
                run = args.runs_root / f"fresh_closed_loop_{method}_seed{seed}"
                isolated = json.loads((run / "closed_loop_isolated.json").read_text())
                end_to_end = json.loads((run / "closed_loop_end_to_end.json").read_text())
                deterministic = json.loads((run / "deterministic_reach.json").read_text())
                groups = group_metrics(
                    isolated["rows"],
                    end_to_end["rows"],
                    deterministic["rows"],
                    execution_horizon=execution_horizon,
                )
                all_groups[execution_horizon][method][seed] = groups
                summary = summarize_seed(groups)
                summaries.append(summary)
                seed_summaries[str(execution_horizon)].setdefault(method, {})[str(seed)] = summary
            aggregate[str(execution_horizon)][method] = _aggregate_seed_summaries(summaries)
            group_bootstrap[str(execution_horizon)][method] = {
                metric: group_bootstrap_across_seeds(
                    all_groups[execution_horizon][method],
                    metric,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=8800 + execution_horizon * 100 + len(group_bootstrap[str(execution_horizon)]) * 40 + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            }

        comparisons[str(execution_horizon)] = {}
        baselines = ("full_h", "random_soft010", "shuffled_oracle_soft010", "gripper_soft010", "short_h")
        for baseline_index, baseline in enumerate(baselines):
            comparisons[str(execution_horizon)][f"oracle_vs_{baseline}"] = {
                metric: paired_group_delta(
                    all_groups[execution_horizon][baseline],
                    all_groups[execution_horizon]["oracle_soft010"],
                    metric,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=9010 + execution_horizon * 100 + baseline_index * 10 + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            }

    payload = {
        "seeds": list(args.seeds),
        "execution_horizons": list(args.execution_horizons),
        "methods": list(METHODS),
        "statistical_unit": "snapshot group; seeds are averaged within group before paired bootstrap",
        "seed_summaries": seed_summaries,
        "aggregate": aggregate,
        "group_bootstrap": group_bootstrap,
        "paired_comparisons": comparisons,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# FRESH-VLA Fixed-K Closed-Loop Results",
        "",
        "All values are cross-seed means. Statistical inference uses paired snapshot groups, not frames.",
        "",
    ]
    for execution_horizon in args.execution_horizons:
        values = aggregate[str(execution_horizon)]
        intervals = group_bootstrap[str(execution_horizon)]
        lines.extend(
            [
                f"## K={execution_horizon}",
                "",
                "Absolute rates show paired-group bootstrap 95% intervals after averaging seeds within group.",
                "",
                "| Method | Overall success | Attached success | Slip recovery | Isolated recovery |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in METHODS:
            lines.append(
                f"| `{method}` | {_fmt_ci(intervals[method]['overall_task_success'])} | "
                f"{_fmt_ci(intervals[method]['attached_task_success'])} | "
                f"{_fmt_ci(intervals[method]['slip_recovery_success'])} | "
                f"{_fmt_ci(intervals[method]['isolated_recovery_success'])} |"
            )
        lines.extend(
            [
                "",
                "| Method | Failure continuation | Premature commitment | Re-grasp | Drop | Final progress | Progress AUC | Grasp | Lift | Transport | Place |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in METHODS:
            row = values[method]
            lines.append(
                f"| `{method}` | {_fmt(row['failure_continuation_rate']['mean'])} | "
                f"{_fmt(row['premature_commitment_rate']['mean'])} | {_fmt(row['slip_regrasp_success']['mean'])} | "
                f"{_fmt(row['drop_rate']['mean'])} | {_fmt(row['final_progress']['mean'])} | "
                f"{_fmt(row['progress_auc']['mean'])} | {_fmt(row['grasp_subgoal_rate']['mean'])} | "
                f"{_fmt(row['lift_subgoal_rate']['mean'])} | {_fmt(row['transport_subgoal_rate']['mean'])} | "
                f"{_fmt(row['place_subgoal_rate']['mean'])} |"
            )
        lines.extend(
            [
                "",
                "### Oracle paired deltas",
                "",
                "Positive success deltas favor Oracle; negative behavior-error deltas favor Oracle.",
                "",
                "| Baseline | Overall | Slip recovery | Isolated recovery | Failure continuation | Premature commitment |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for baseline in ("full_h", "random_soft010", "shuffled_oracle_soft010", "gripper_soft010", "short_h"):
            comparison = comparisons[str(execution_horizon)][f"oracle_vs_{baseline}"]
            lines.append(
                f"| `{baseline}` | {_fmt_pp(comparison['overall_task_success']['candidate_minus_baseline'])} | "
                f"{_fmt_pp(comparison['slip_recovery_success']['candidate_minus_baseline'])} | "
                f"{_fmt_pp(comparison['isolated_recovery_success']['candidate_minus_baseline'])} | "
                f"{_fmt_pp(comparison['failure_continuation_rate']['candidate_minus_baseline'])} | "
                f"{_fmt_pp(comparison['premature_commitment_rate']['candidate_minus_baseline'])} |"
            )
        lines.extend(("", "### Per-seed primary rates", "", "| Method | Seed | Overall | Attached | Slip recovery | Isolated recovery |", "| --- | ---: | ---: | ---: | ---: | ---: |"))
        for method in METHODS:
            for run_seed in args.seeds:
                row = seed_summaries[str(execution_horizon)][method][str(run_seed)]
                lines.append(
                    f"| `{method}` | {run_seed} | {_fmt(row['overall_task_success'])} | "
                    f"{_fmt(row['attached_task_success'])} | {_fmt(row['slip_recovery_success'])} | "
                    f"{_fmt(row['isolated_recovery_success'])} |"
                )
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output_dir": str(output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
