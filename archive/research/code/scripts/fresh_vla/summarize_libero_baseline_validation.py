from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scripts.fresh_vla.paired_evaluation import bootstrap_summary
except ModuleNotFoundError:
    from paired_evaluation import bootstrap_summary


METRICS = (
    "overall_task_success",
    "attached_task_success",
    "slip_task_success",
    "event_trigger_rate",
    "final_progress",
    "grasp_subgoal_rate",
    "lift_subgoal_rate",
    "transport_subgoal_rate",
    "place_subgoal_rate",
)


def group_metrics(rows: Sequence[Mapping[str, Any]], execution_horizon: int) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if int(row["execution_horizon"]) == execution_horizon:
            grouped[str(row["pair_id"])][str(row["branch_outcome"])] = row
    result = {}
    for pair_id, branches in sorted(grouped.items()):
        if branches.keys() != {"attached", "slipped"}:
            raise ValueError(f"missing paired validation branch for {pair_id}")
        attached = branches["attached"]
        slipped = branches["slipped"]
        result[pair_id] = {
            "overall_task_success": 0.5 * (float(attached["success"]) + float(slipped["success"])),
            "attached_task_success": float(attached["success"]),
            "slip_task_success": float(slipped["success"]),
            "event_trigger_rate": 0.5
            * (float(attached.get("event_time") is not None) + float(slipped.get("event_time") is not None)),
            "final_progress": 0.5
            * (float(attached["final_progress"]) + float(slipped["final_progress"])),
            "grasp_subgoal_rate": 0.5
            * (float(attached["grasp_subgoal"]) + float(slipped["grasp_subgoal"])),
            "lift_subgoal_rate": 0.5
            * (float(attached["lift_subgoal"]) + float(slipped["lift_subgoal"])),
            "transport_subgoal_rate": 0.5
            * (float(attached["transport_subgoal"]) + float(slipped["transport_subgoal"])),
            "place_subgoal_rate": 0.5
            * (float(attached["place_subgoal"]) + float(slipped["place_subgoal"])),
        }
    if not result:
        raise ValueError(f"no validation rows for K={execution_horizon}")
    return result


def summarize_across_seeds(
    groups_by_seed: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    first_groups = None
    for seed, groups in groups_by_seed.items():
        if first_groups is None:
            first_groups = set(groups)
        elif set(groups) != first_groups:
            raise ValueError(f"validation groups differ for seed {seed}")
    result = {}
    for metric_index, metric in enumerate(METRICS):
        per_group = [
            float(np.mean([groups[pair_id][metric] for groups in groups_by_seed.values()]))
            for pair_id in sorted(first_groups or ())
        ]
        result[metric] = bootstrap_summary(
            per_group,
            bootstrap_samples=bootstrap_samples,
            seed=17290 + metric_index,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only Full-H training-budget gate")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--tag", default="val_budget27607")
    parser.add_argument("--execution-horizon", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--training-steps", type=int, default=27607)
    parser.add_argument("--minimum-attached-success", type=float, default=0.20)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups_by_seed = {}
    seed_summaries = {}
    for seed in args.seeds:
        path = args.runs_root / f"fresh_closed_loop_full_h_seed{seed}" / f"closed_loop_end_to_end_{args.tag}.json"
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("split") != "val":
            raise ValueError(f"invalid Full-H validation result: {path}")
        groups = group_metrics(payload["rows"], args.execution_horizon)
        groups_by_seed[seed] = groups
        seed_summaries[str(seed)] = {
            metric: float(np.mean([row[metric] for row in groups.values()])) for metric in METRICS
        }
    aggregate = summarize_across_seeds(groups_by_seed, bootstrap_samples=args.bootstrap_samples)
    baseline_valid = aggregate["attached_task_success"]["mean"] >= args.minimum_attached_success
    output_dir = args.output_dir or args.runs_root / f"baseline_validation_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "complete",
        "split": "val",
        "method": "full_h",
        "seeds": list(args.seeds),
        "execution_horizon": args.execution_horizon,
        "training_steps": args.training_steps,
        "statistical_unit": "snapshot group after averaging seeds",
        "minimum_attached_success": args.minimum_attached_success,
        "baseline_valid": baseline_valid,
        "seed_summaries": seed_summaries,
        "aggregate": aggregate,
        "next_action": "freeze_common_budget" if baseline_valid else "extend_full_h_without_opening_test",
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Full-H Validation Budget Gate",
        "",
        f"- Split: validation only",
        f"- Fixed K: {args.execution_horizon}",
        f"- Training steps: {args.training_steps}",
        f"- Baseline valid: `{str(baseline_valid).lower()}`",
        f"- Next action: `{result['next_action']}`",
        "",
        "| Metric | Mean [group bootstrap 95% CI] |",
        "| --- | ---: |",
    ]
    for metric in METRICS:
        row = aggregate[metric]
        lines.append(
            f"| `{metric}` | {row['mean']:.3f} [{row['bootstrap_95_low']:.3f}, {row['bootstrap_95_high']:.3f}] |"
        )
    lines.extend(("", "## Per Seed", "", "| Seed | Attached | Overall | Event trigger | Final progress |", "| ---: | ---: | ---: | ---: | ---: |"))
    for seed in args.seeds:
        row = seed_summaries[str(seed)]
        lines.append(
            f"| {seed} | {row['attached_task_success']:.3f} | {row['overall_task_success']:.3f} | "
            f"{row['event_trigger_rate']:.3f} | {row['final_progress']:.3f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"baseline_valid": baseline_valid, "next_action": result["next_action"], "output_dir": str(output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
