from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from scripts.fresh_vla.paired_evaluation import bootstrap_summary
except ModuleNotFoundError:
    from paired_evaluation import bootstrap_summary


SEEDS = (41, 42, 43)
METHODS = ("original_task", "explicit_recovery", "false_success_assumption")
METRICS = (
    "success",
    "recovery_success",
    "regrasp_success",
    "failure_continuation",
    "premature_commitment",
    "recovery_switch_observed",
    "recovery_switch_latency",
    "final_progress",
    "progress_auc",
    "completion_steps",
)


def bootstrap_metric(
    rows_by_seed: Mapping[int, Mapping[str, Mapping[str, object]]],
    metric: str,
    *,
    seed: int,
) -> dict[str, object]:
    values = []
    for pair_id in sorted(next(iter(rows_by_seed.values()))):
        per_seed = [
            float(rows_by_seed[run_seed][pair_id][metric])
            for run_seed in sorted(rows_by_seed)
            if rows_by_seed[run_seed][pair_id].get(metric) is not None
        ]
        if per_seed:
            values.append(float(np.mean(per_seed)))
    return bootstrap_summary(values, seed=seed)


def paired_delta(
    baseline: Mapping[int, Mapping[str, Mapping[str, object]]],
    candidate: Mapping[int, Mapping[str, Mapping[str, object]]],
    metric: str,
    *,
    seed: int,
) -> dict[str, object]:
    pair_ids = sorted(next(iter(baseline.values())))
    group_deltas = []
    seed_deltas = {}
    for run_seed in sorted(baseline):
        values = [
            float(candidate[run_seed][pair_id][metric]) - float(baseline[run_seed][pair_id][metric])
            for pair_id in pair_ids
            if baseline[run_seed][pair_id].get(metric) is not None
            and candidate[run_seed][pair_id].get(metric) is not None
        ]
        seed_deltas[str(run_seed)] = float(np.mean(values)) if values else None
    for pair_id in pair_ids:
        values = [
            float(candidate[run_seed][pair_id][metric]) - float(baseline[run_seed][pair_id][metric])
            for run_seed in sorted(baseline)
            if baseline[run_seed][pair_id].get(metric) is not None
            and candidate[run_seed][pair_id].get(metric) is not None
        ]
        if values:
            group_deltas.append(float(np.mean(values)))
    return {
        "candidate_minus_baseline": bootstrap_summary(group_deltas, seed=seed),
        "seed_deltas": seed_deltas,
    }


def load_rows(args: argparse.Namespace) -> dict[str, dict[int, dict[str, dict[str, object]]]]:
    result: dict[str, dict[int, dict[str, dict[str, object]]]] = {
        method: {} for method in METHODS
    }
    for seed in SEEDS:
        baseline_path = args.baseline_root / f"fresh_closed_loop_full_h_seed{seed}" / "closed_loop_isolated.json"
        baseline_payload = json.loads(baseline_path.read_text())
        baseline_rows = [row for row in baseline_payload["rows"] if int(row["execution_horizon"]) == 3]
        for method in METHODS:
            if method == "original_task":
                rows = baseline_rows
            else:
                payload = json.loads((args.runs_root / f"{method}_seed{seed}" / "closed_loop_isolated.json").read_text())
                if payload.get("status") != "complete" or len(payload.get("rows", ())) != 26:
                    raise ValueError(f"incomplete prompt evaluation for {method} seed {seed}")
                rows = payload["rows"]
            slipped = [row for row in rows if row["branch_outcome"] == "slipped"]
            if len(slipped) != 13:
                raise ValueError(f"unexpected slipped row count for {method} seed {seed}")
            result[method][seed] = {str(row["pair_id"]): row for row in slipped}
    return result


def stable_positive(comparison: Mapping[str, object], minimum: float) -> bool:
    summary = comparison["candidate_minus_baseline"]
    seed_deltas = [float(value) for value in comparison["seed_deltas"].values() if value is not None]
    return bool(
        float(summary["mean"]) >= minimum
        and (float(summary["bootstrap_95_low"]) > 0 or all(value > 0 for value in seed_deltas))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize explicit recovery prompt upper-bound evaluation")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2"),
    )
    parser.add_argument("--seed", type=int, default=271828)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args)
    absolute = {
        method: {
            metric: bootstrap_metric(
                rows[method],
                metric,
                seed=args.seed + method_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
        for method_index, method in enumerate(METHODS)
    }
    comparisons = {}
    for baseline_index, baseline in enumerate(("original_task", "false_success_assumption")):
        comparisons[f"explicit_recovery_vs_{baseline}"] = {
            metric: paired_delta(
                rows[baseline],
                rows["explicit_recovery"],
                metric,
                seed=args.seed + 1000 + baseline_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    versus_original = comparisons["explicit_recovery_vs_original_task"]["recovery_success"]
    versus_wrong = comparisons["explicit_recovery_vs_false_success_assumption"]["recovery_success"]
    supports_prompt_grounded_recovery = bool(
        stable_positive(versus_original, 0.20) and stable_positive(versus_wrong, 0.10)
    )
    payload = {
        "experiment": "explicit_recovery_prompt_upper_bound",
        "seeds": list(SEEDS),
        "split": "test",
        "evaluation": "isolated_recovery",
        "execution_horizon": 3,
        "go_criteria": {
            "recovery_success_gain_vs_original_min": 0.20,
            "recovery_success_gain_vs_false_success_prompt_min": 0.10,
            "stability": "paired group CI excludes zero or all three seed deltas are positive",
        },
        "supports_prompt_grounded_recovery": supports_prompt_grounded_recovery,
        "absolute": absolute,
        "paired_comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "supports_prompt_grounded_recovery": supports_prompt_grounded_recovery,
                "explicit_recovery_success": absolute["explicit_recovery"]["recovery_success"]["mean"],
                "original_success": absolute["original_task"]["recovery_success"]["mean"],
                "false_success_prompt": absolute["false_success_assumption"]["recovery_success"]["mean"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
