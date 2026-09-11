from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from summarize_libero_closed_loop import METHODS, paired_group_delta


REACH_METRICS = (
    "success",
    "best_target_error",
    "final_target_error",
    "best_target_progress",
    "final_target_progress",
    "first_action_mse",
    "first_translation_cosine",
)


def reach_group_metrics(
    rows: Sequence[Mapping[str, Any]], *, execution_horizon: int
) -> dict[str, dict[str, float]]:
    result = {}
    for row in rows:
        if int(row["execution_horizon"]) != execution_horizon:
            continue
        pair_id = str(row["pair_id"])
        if pair_id in result:
            raise ValueError(f"duplicate reach row for {pair_id} at K={execution_horizon}")
        result[pair_id] = {
            metric: float(row[metric])
            for metric in REACH_METRICS
        }
    if not result:
        raise ValueError(f"no reach rows for K={execution_horizon}")
    return result


def summarize_seed(groups: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {
        metric: float(np.mean([row[metric] for row in groups.values()]))
        for metric in REACH_METRICS
    }


def aggregate_seed_summaries(summaries: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    result = {}
    for metric in REACH_METRICS:
        values = [float(row[metric]) for row in summaries]
        result[metric] = {
            "mean": float(np.mean(values)),
            "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "seed_values": values,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-level statistics for deterministic LIBERO reach")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2"),
    )
    parser.add_argument("--input-name", default="deterministic_reach.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--execution-horizons", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.runs_root / "deterministic_reach_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_groups: dict[int, dict[str, dict[int, dict[str, dict[str, float]]]]] = {}
    seed_summaries: dict[str, Any] = {}
    aggregate: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    fingerprints = set()

    for execution_horizon in args.execution_horizons:
        all_groups[execution_horizon] = {}
        seed_summaries[str(execution_horizon)] = {}
        aggregate[str(execution_horizon)] = {}
        for method in METHODS:
            all_groups[execution_horizon][method] = {}
            summaries = []
            for seed in args.seeds:
                path = args.runs_root / f"fresh_closed_loop_{method}_seed{seed}" / args.input_name
                payload = json.loads(path.read_text())
                fingerprints.add(
                    (
                        payload.get("target_definition"),
                        payload.get("reference_target_step"),
                        payload.get("success_threshold"),
                    )
                )
                groups = reach_group_metrics(payload["rows"], execution_horizon=execution_horizon)
                all_groups[execution_horizon][method][seed] = groups
                summary = summarize_seed(groups)
                summaries.append(summary)
                seed_summaries[str(execution_horizon)].setdefault(method, {})[str(seed)] = summary
            aggregate[str(execution_horizon)][method] = aggregate_seed_summaries(summaries)

        comparisons[str(execution_horizon)] = {}
        baselines = ("full_h", "random_soft010", "shuffled_oracle_soft010", "gripper_soft010", "short_h")
        for baseline_index, baseline in enumerate(baselines):
            comparisons[str(execution_horizon)][f"oracle_vs_{baseline}"] = {
                metric: paired_group_delta(
                    all_groups[execution_horizon][baseline],
                    all_groups[execution_horizon]["oracle_soft010"],
                    metric,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=12000 + execution_horizon * 100 + baseline_index * 10 + metric_index,
                )
                for metric_index, metric in enumerate(REACH_METRICS)
            }

    if len(fingerprints) != 1:
        raise ValueError(f"reach evaluation fingerprints differ: {sorted(fingerprints, key=str)}")
    target_definition, reference_target_step, success_threshold = next(iter(fingerprints))
    payload = {
        "seeds": list(args.seeds),
        "execution_horizons": list(args.execution_horizons),
        "methods": list(METHODS),
        "input_name": args.input_name,
        "target_definition": target_definition,
        "reference_target_step": reference_target_step,
        "success_threshold": success_threshold,
        "statistical_unit": "snapshot group; seeds are averaged within group before paired bootstrap",
        "seed_summaries": seed_summaries,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# FRESH-VLA Deterministic Reach Results",
        "",
        "Target: recorded expert EEF position at step "
        f"{reference_target_step}; success threshold: {success_threshold:.3f} m.",
        "",
        "All values are cross-seed means. Statistical inference uses paired snapshot groups, not frames.",
        "",
    ]
    for execution_horizon in args.execution_horizons:
        lines.extend(
            [
                f"## K={execution_horizon}",
                "",
                "| Method | Success | Best error (m) | Final error (m) | Best progress (m) | First-action MSE | Translation cosine |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in METHODS:
            row = aggregate[str(execution_horizon)][method]
            lines.append(
                f"| `{method}` | {row['success']['mean']:.3f} | {row['best_target_error']['mean']:.3f} | "
                f"{row['final_target_error']['mean']:.3f} | {row['best_target_progress']['mean']:.3f} | "
                f"{row['first_action_mse']['mean']:.3f} | {row['first_translation_cosine']['mean']:.3f} |"
            )
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output_dir": str(output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
