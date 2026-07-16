from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from summarize_libero_closed_loop import (
    METRICS,
    group_bootstrap_across_seeds,
    group_metrics,
    paired_group_delta,
    summarize_seed,
)


METHODS = (
    "original_full_h",
    "base_continuation",
    "clean_recovery_replay",
    "policy_state_recovery",
)
SEEDS = (41, 42, 43)
EXECUTION_HORIZONS = (1, 2, 3)
COMPARISONS = {
    "base_vs_original": ("original_full_h", "base_continuation"),
    "clean_vs_base": ("base_continuation", "clean_recovery_replay"),
    "policy_vs_base": ("base_continuation", "policy_state_recovery"),
    "policy_vs_clean": ("clean_recovery_replay", "policy_state_recovery"),
}


def _aggregate_seed_summaries(
    summaries: list[Mapping[str, float | None]],
) -> dict[str, Any]:
    result = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in summaries if row[metric] is not None]
        result[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "seed_values": values,
        }
    return result


def _effect(
    comparisons: Mapping[str, Any],
    name: str,
    metric: str,
) -> Mapping[str, Any]:
    return comparisons["3"][name][metric]["candidate_minus_baseline"]


def _positive_behavior_effect(
    comparisons: Mapping[str, Any],
    name: str,
    *,
    minimum: float = 0.10,
) -> tuple[bool, str | None]:
    for metric in ("slip_recovery_success", "overall_task_success"):
        effect = _effect(comparisons, name, metric)
        if (
            effect.get("mean") is not None
            and float(effect["mean"]) >= minimum
            and float(effect["bootstrap_95_low"]) > 0.0
        ):
            return True, metric
    return False, None


def _attached_degradation_within(
    comparisons: Mapping[str, Any],
    name: str,
    *,
    maximum: float = 0.05,
) -> bool:
    effect = _effect(comparisons, name, "attached_task_success")
    return effect.get("mean") is not None and float(effect["mean"]) >= -maximum


def support_decision(comparisons: Mapping[str, Any]) -> dict[str, Any]:
    policy_better_clean, policy_clean_metric = _positive_behavior_effect(comparisons, "policy_vs_clean")
    policy_better_base, policy_base_metric = _positive_behavior_effect(comparisons, "policy_vs_base")
    clean_better, clean_metric = _positive_behavior_effect(comparisons, "clean_vs_base")
    base_better, base_metric = _positive_behavior_effect(comparisons, "base_vs_original")
    policy_preserves_clean_attached = _attached_degradation_within(comparisons, "policy_vs_clean")
    policy_preserves_base_attached = _attached_degradation_within(comparisons, "policy_vs_base")
    clean_preserves_attached = _attached_degradation_within(comparisons, "clean_vs_base")
    base_preserves_attached = _attached_degradation_within(comparisons, "base_vs_original")

    if (
        policy_better_clean
        and policy_better_base
        and policy_preserves_clean_attached
        and policy_preserves_base_attached
    ):
        decision = "CONTINUE_MINIMAL_RECOVERY_BRIDGE"
        reason = (
            "policy-state recovery beats both matched clean replay and Base "
            f"on {policy_clean_metric}/{policy_base_metric}"
        )
    elif clean_better and clean_preserves_attached:
        decision = "ADOPT_CLEAN_RECOVERY_REPLAY"
        reason = f"clean recovery replay improves {clean_metric}; policy-state data has no unique gain"
    elif base_better and base_preserves_attached:
        decision = "BASELINE_UNDERTRAINED"
        reason = f"ordinary continuation improves {base_metric}; targeted support has no required gain"
    else:
        decision = "STOP_OFFLINE_SUPPORT_EXPANSION"
        reason = "no offline support arm clears the paired closed-loop effect and normal-task gates"
    return {
        "decision": decision,
        "reason": reason,
        "primary_execution_horizon": 3,
        "minimum_behavior_gain": 0.10,
        "paired_ci_must_exclude_zero": True,
        "maximum_attached_degradation": 0.05,
        "policy_vs_clean_positive": policy_better_clean,
        "policy_vs_clean_metric": policy_clean_metric,
        "policy_vs_base_positive": policy_better_base,
        "policy_vs_base_metric": policy_base_metric,
        "policy_preserves_clean_attached": policy_preserves_clean_attached,
        "policy_preserves_base_attached": policy_preserves_base_attached,
        "clean_vs_base_positive": clean_better,
        "clean_vs_base_metric": clean_metric,
        "clean_preserves_attached": clean_preserves_attached,
        "base_vs_original_positive": base_better,
        "base_vs_original_metric": base_metric,
        "base_preserves_attached": base_preserves_attached,
    }


def _candidate_run(output_root: Path, method: str, seed: int, steps: int) -> Path:
    return output_root / f"recovery_support_{method}_seed{seed}_steps{steps}"


def _load_payload(path: Path, *, evaluation: str, split: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "complete"
        or payload.get("split") != split
        or payload.get("evaluation") != evaluation
    ):
        raise ValueError(f"invalid support evaluation payload: {path}")
    return payload


def _load_method_seed(
    output_root: Path,
    baseline_root: Path,
    method: str,
    seed: int,
    *,
    steps: int,
    tag: str,
    baseline_tag: str,
    split: str,
    execution_horizon: int,
    baseline_steps: int,
) -> dict[str, dict[str, float | None]]:
    if method == "original_full_h":
        run = baseline_root / f"fresh_closed_loop_repair_step{baseline_steps}_seed{seed}"
        isolated_path = run / f"closed_loop_isolated_{baseline_tag}.json"
        end_to_end_path = run / f"closed_loop_end_to_end_{baseline_tag}.json"
        deterministic_path = run / f"deterministic_reach_{baseline_tag}.json"
    else:
        run = _candidate_run(output_root, method, seed, steps)
        isolated_path = run / f"closed_loop_isolated_{tag}.json"
        end_to_end_path = run / f"closed_loop_end_to_end_{tag}.json"
        deterministic_path = run / f"deterministic_reach_{tag}.json"
    isolated = _load_payload(isolated_path, evaluation="isolated", split=split)
    end_to_end = _load_payload(end_to_end_path, evaluation="end_to_end", split=split)
    deterministic = json.loads(deterministic_path.read_text())
    deterministic_rows = list(deterministic.get("rows", ()))
    deterministic_identities = {
        (int(row["execution_horizon"]), str(row["pair_id"]))
        for row in deterministic_rows
    }
    if (
        deterministic.get("split") != split
        or deterministic.get("evaluation") != "deterministic_reach"
        or len(deterministic_rows) != 39
        or len(deterministic_identities) != 39
    ):
        raise ValueError(f"invalid deterministic evaluation payload: {deterministic_path}")
    return group_metrics(
        isolated["rows"],
        end_to_end["rows"],
        deterministic_rows,
        execution_horizon=execution_horizon,
    )


def summarize_support(
    output_root: Path,
    baseline_root: Path,
    *,
    steps: int,
    tag: str,
    baseline_tag: str,
    split: str,
    bootstrap_samples: int,
    baseline_steps: int = 10353,
) -> dict[str, Any]:
    groups: dict[str, dict[str, dict[int, dict[str, dict[str, float | None]]]]] = {}
    seed_summaries: dict[str, Any] = {}
    aggregate: dict[str, Any] = {}
    intervals: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}

    for horizon in EXECUTION_HORIZONS:
        horizon_key = str(horizon)
        groups[horizon_key] = {}
        seed_summaries[horizon_key] = {}
        aggregate[horizon_key] = {}
        intervals[horizon_key] = {}
        for method_index, method in enumerate(METHODS):
            groups[horizon_key][method] = {}
            summaries = []
            for seed in SEEDS:
                per_group = _load_method_seed(
                    output_root,
                    baseline_root,
                    method,
                    seed,
                    steps=steps,
                    tag=tag,
                    baseline_tag=baseline_tag,
                    split=split,
                    execution_horizon=horizon,
                    baseline_steps=baseline_steps,
                )
                groups[horizon_key][method][seed] = per_group
                summary = summarize_seed(per_group)
                summaries.append(summary)
                seed_summaries[horizon_key].setdefault(method, {})[str(seed)] = summary
            aggregate[horizon_key][method] = _aggregate_seed_summaries(summaries)
            intervals[horizon_key][method] = {
                metric: group_bootstrap_across_seeds(
                    groups[horizon_key][method],
                    metric,
                    bootstrap_samples=bootstrap_samples,
                    seed=61_000 + horizon * 1_000 + method_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            }

        comparisons[horizon_key] = {}
        for comparison_index, (name, (baseline, candidate)) in enumerate(COMPARISONS.items()):
            comparisons[horizon_key][name] = {
                metric: paired_group_delta(
                    groups[horizon_key][baseline],
                    groups[horizon_key][candidate],
                    metric,
                    bootstrap_samples=bootstrap_samples,
                    seed=71_000 + horizon * 1_000 + comparison_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            }

    decision = support_decision(comparisons)
    return {
        "schema_version": 1,
        "steps": steps,
        "baseline_steps": baseline_steps,
        "tag": tag,
        "baseline_tag": baseline_tag,
        "split": split,
        "test_split_opened": split == "test",
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "execution_horizons": list(EXECUTION_HORIZONS),
        "statistical_unit": "snapshot group after averaging seeds; paired groups are never expanded into frames",
        "seed_summaries": seed_summaries,
        "aggregate": aggregate,
        "group_bootstrap": intervals,
        "paired_comparisons": comparisons,
        "support_decision": decision,
    }


def _format_interval(row: Mapping[str, Any]) -> str:
    if row.get("mean") is None:
        return "n/a"
    return (
        f"{100 * float(row['mean']):.1f}% "
        f"[{100 * float(row['bootstrap_95_low']):.1f}, {100 * float(row['bootstrap_95_high']):.1f}]"
    )


def report_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Recovery Support Closed-Loop Result",
        "",
        f"Split: `{payload['split']}`. Test opened: `{str(payload['test_split_opened']).lower()}`.",
        "",
        "Inference uses paired snapshot groups after averaging seeds; frames are not statistical samples.",
        "",
    ]
    for horizon in EXECUTION_HORIZONS:
        key = str(horizon)
        lines.extend(
            [
                f"## K={horizon}",
                "",
                "| Method | Overall task | Attached task | Slip recovery | Isolated recovery |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in METHODS:
            rows = payload["group_bootstrap"][key][method]
            lines.append(
                f"| `{method}` | {_format_interval(rows['overall_task_success'])} | "
                f"{_format_interval(rows['attached_task_success'])} | "
                f"{_format_interval(rows['slip_recovery_success'])} | "
                f"{_format_interval(rows['isolated_recovery_success'])} |"
            )
        lines.append("")
    decision = payload["support_decision"]
    lines.extend(
        [
            "## Decision",
            "",
            f"`{decision['decision']}`",
            "",
            decision["reason"],
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired fixed-K statistics for recovery-support controls")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/recovery-support-repaired-v2-step10353"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/baseline-repair-v1/eval_views"),
    )
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--baseline-steps", type=int, default=10353)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--baseline-tag", default="val_gate_final")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.bootstrap_samples < 1:
        raise ValueError("steps and bootstrap-samples must be positive")
    payload = summarize_support(
        args.output_root,
        args.baseline_root,
        steps=args.steps,
        tag=args.tag,
        baseline_tag=args.baseline_tag,
        split=args.split,
        bootstrap_samples=args.bootstrap_samples,
        baseline_steps=args.baseline_steps,
    )
    output_dir = args.output_dir or args.output_root / f"closed_loop_summary_{args.split}_steps{args.steps}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_dir / "report.md").write_text(report_markdown(payload))
    (args.output_root / f"support_decision_{args.split}.json").write_text(
        json.dumps(payload["support_decision"], indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output_dir": str(output_dir), **payload["support_decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
