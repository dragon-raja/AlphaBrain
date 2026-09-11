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


METRICS = (
    "sample0_correct_mode",
    "any_correct_mode",
    "best_candidate_is_correct_mode",
    "correct_mode_fraction",
    "opposite_mode_fraction",
    "covers_both_modes",
    "sample0_correct_rmse",
    "best_correct_rmse",
    "best_of_n_relative_rmse_reduction",
    "expert_mode_distance",
    "candidate_action_variance",
)


def aggregate_rows(rows: Sequence[Mapping[str, object]], metric: str, *, seed: int) -> dict[str, object]:
    by_group: dict[str, list[float]] = defaultdict(list)
    by_source: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        by_group[str(row["pair_id"])].append(value)
        by_source[str(row["source_initial_state_index"])].append(value)
    group_values = [float(np.mean(values)) for values in by_group.values()]
    source_values = [float(np.mean(values)) for values in by_source.values()]
    return {
        "group_bootstrap_95": bootstrap_summary(group_values, seed=seed),
        "source_state_bootstrap_95": bootstrap_summary(source_values, seed=seed + 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize frozen Full-H post-feedback mode coverage")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=314159)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    seeds = [int(payload["seed"]) for payload in payloads]
    if len(set(seeds)) != len(seeds):
        raise ValueError("input seeds must be unique")
    rows = [row for payload in payloads for row in payload["rows"]]
    summary: dict[str, dict[str, dict[str, object]]] = {}
    for offset_index, offset in enumerate(sorted({int(row["offset"]) for row in rows})):
        summary[str(offset)] = {}
        for outcome_index, outcome in enumerate(("attached", "slipped")):
            selected = [row for row in rows if int(row["offset"]) == offset and row["outcome"] == outcome]
            summary[str(offset)][outcome] = {
                metric: aggregate_rows(
                    selected,
                    metric,
                    seed=args.seed + offset_index * 100 + outcome_index * 20 + metric_index * 2,
                )
                for metric_index, metric in enumerate(METRICS)
            }

    slipped = summary["0"]["slipped"]
    sample0 = float(slipped["sample0_correct_mode"]["group_bootstrap_95"]["mean"])
    available = float(slipped["any_correct_mode"]["group_bootstrap_95"]["mean"])
    available_low = float(slipped["any_correct_mode"]["group_bootstrap_95"]["bootstrap_95_low"])
    reduction = float(slipped["best_of_n_relative_rmse_reduction"]["group_bootstrap_95"]["mean"])
    availability_gap = available - sample0
    supports_mode_selection_bottleneck = bool(
        available >= 0.70
        and available_low > 0.50
        and sample0 <= 0.50
        and availability_gap >= 0.20
        and reduction >= 0.20
    )
    result = {
        "experiment": "post_feedback_mode_coverage_summary",
        "seeds": seeds,
        "statistical_unit": "snapshot group after averaging seeds",
        "source_state_sensitivity_unit": "source_initial_state_index after averaging groups and seeds",
        "go_criteria": {
            "slip_any_correct_mode_mean_min": 0.70,
            "slip_any_correct_mode_group_ci_low_strictly_above": 0.50,
            "slip_sample0_correct_mode_mean_max": 0.50,
            "availability_minus_sample0_min": 0.20,
            "best_of_n_relative_rmse_reduction_min": 0.20,
        },
        "supports_mode_selection_bottleneck": supports_mode_selection_bottleneck,
        "slip_availability_gap": availability_gap,
        "summary": summary,
        "inputs": [str(path) for path in args.inputs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "supports_mode_selection_bottleneck": supports_mode_selection_bottleneck,
                "slip_sample0_correct_mode": sample0,
                "slip_any_correct_mode": available,
                "slip_availability_gap": availability_gap,
                "slip_best_of_n_relative_rmse_reduction": reduction,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
