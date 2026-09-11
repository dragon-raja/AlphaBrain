from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from summarize_libero_closed_loop import METHODS, paired_group_delta


OFFLINE_METRICS = ("fixed_k_1", "fixed_k_2", "fixed_k_3", "oracle_prefix", "suffix", "full")
MODE_SOURCES = {
    "suffix_mode_coverage": "covers_both_suffix_modes",
    "mode_balance": "mode_balance",
    "attached_mode_fraction": "attached_mode_fraction",
    "slipped_mode_fraction": "slipped_mode_fraction",
    "common_prefix_mse": "common_prefix_mse",
    "common_prefix_variance": "common_prefix_variance",
    "suffix_variance": "suffix_variance",
    "suffix_min_expert_distance": "suffix_min_expert_distance",
}


def group_averages(
    rows: Sequence[Mapping[str, Any]],
    metric_sources: Mapping[str, str],
) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pair_id = str(row["pair_id"])
        values = grouped[pair_id]
        for metric, source in metric_sources.items():
            value = row.get(source)
            if value is not None:
                values[metric].append(float(value))
    if not grouped:
        raise ValueError("evaluation contains no snapshot groups")
    return {
        pair_id: {
            metric: (float(np.mean(values[metric])) if values.get(metric) else None)
            for metric in metric_sources
        }
        for pair_id, values in sorted(grouped.items())
    }


def summarize_groups(
    groups: Mapping[str, Mapping[str, float | None]], metrics: Sequence[str]
) -> dict[str, float | None]:
    return {
        metric: (
            float(np.mean(values))
            if (values := [float(row[metric]) for row in groups.values() if row[metric] is not None])
            else None
        )
        for metric in metrics
    }


def aggregate_seeds(
    summaries: Sequence[Mapping[str, float | None]], metrics: Sequence[str]
) -> dict[str, Any]:
    result = {}
    for metric in metrics:
        values = [float(row[metric]) for row in summaries if row[metric] is not None]
        result[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "seed_values": values,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-level offline statistics for full-episode FRESH-VLA")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.runs_root / "episode_offline_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    offline_sources = {metric: metric for metric in OFFLINE_METRICS}
    group_data = {"offline": {}, "mode": {}}
    seed_summaries = {"offline": {}, "mode": {}}
    aggregate = {"offline": {}, "mode": {}}
    fingerprints: dict[int, set[str]] = defaultdict(set)

    for method in METHODS:
        group_data["offline"][method] = {}
        group_data["mode"][method] = {}
        offline_seed_rows = []
        mode_seed_rows = []
        for seed in args.seeds:
            run = args.runs_root / f"fresh_closed_loop_{method}_seed{seed}"
            offline = json.loads((run / "offline_eval.json").read_text())
            mode = json.loads((run / "mode_coverage.json").read_text())
            fingerprints[seed].add(str(offline["evaluation_fingerprint"]))
            offline_groups = group_averages(offline["rows"], offline_sources)
            mode_groups = group_averages(mode["rows"], MODE_SOURCES)
            group_data["offline"][method][seed] = offline_groups
            group_data["mode"][method][seed] = mode_groups
            offline_summary = summarize_groups(offline_groups, OFFLINE_METRICS)
            mode_summary = summarize_groups(mode_groups, tuple(MODE_SOURCES))
            offline_seed_rows.append(offline_summary)
            mode_seed_rows.append(mode_summary)
            seed_summaries["offline"].setdefault(method, {})[str(seed)] = offline_summary
            seed_summaries["mode"].setdefault(method, {})[str(seed)] = mode_summary
        aggregate["offline"][method] = aggregate_seeds(offline_seed_rows, OFFLINE_METRICS)
        aggregate["mode"][method] = aggregate_seeds(mode_seed_rows, tuple(MODE_SOURCES))

    bad_fingerprints = {seed: values for seed, values in fingerprints.items() if len(values) != 1}
    if bad_fingerprints:
        raise ValueError(f"offline fingerprints differ within seed: {bad_fingerprints}")

    comparisons = {"offline": {}, "mode": {}}
    baselines = ("full_h", "random_soft010", "shuffled_oracle_soft010", "gripper_soft010", "short_h")
    for baseline_index, baseline in enumerate(baselines):
        comparisons["offline"][f"oracle_vs_{baseline}"] = {
            metric: paired_group_delta(
                group_data["offline"][baseline],
                group_data["offline"]["oracle_soft010"],
                metric,
                bootstrap_samples=args.bootstrap_samples,
                seed=14000 + baseline_index * 20 + metric_index,
            )
            for metric_index, metric in enumerate(OFFLINE_METRICS)
        }
        comparisons["mode"][f"oracle_vs_{baseline}"] = {
            metric: paired_group_delta(
                group_data["mode"][baseline],
                group_data["mode"]["oracle_soft010"],
                metric,
                bootstrap_samples=args.bootstrap_samples,
                seed=15000 + baseline_index * 20 + metric_index,
            )
            for metric_index, metric in enumerate(MODE_SOURCES)
        }

    payload = {
        "seeds": list(args.seeds),
        "methods": list(METHODS),
        "statistical_unit": "snapshot group; windows/branches are averaged within group, then seeds, before paired bootstrap",
        "evaluation_fingerprints": {str(seed): next(iter(values)) for seed, values in fingerprints.items()},
        "seed_summaries": seed_summaries,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# FRESH-VLA Full-Episode Offline Results",
        "",
        "Values are cross-seed means. Snapshot groups, not windows or frames, are the statistical units.",
        "",
        "| Method | K=1 | K=2 | K=3 | Oracle prefix | Suffix | Full | Mode coverage | Mode balance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        offline = aggregate["offline"][method]
        mode = aggregate["mode"][method]
        lines.append(
            f"| `{method}` | {offline['fixed_k_1']['mean']:.4f} | {offline['fixed_k_2']['mean']:.4f} | "
            f"{offline['fixed_k_3']['mean']:.4f} | {offline['oracle_prefix']['mean']:.4f} | "
            f"{offline['suffix']['mean']:.4f} | {offline['full']['mean']:.4f} | "
            f"{mode['suffix_mode_coverage']['mean']:.3f} | {mode['mode_balance']['mean']:.3f} |"
        )
    lines.extend(("", "## Oracle Paired Deltas", "", "Negative MSE deltas favor Oracle FRESH.", ""))
    lines.extend(("| Baseline | K=2 delta [95% CI] | Prefix delta [95% CI] |", "| --- | ---: | ---: |"))
    for baseline in baselines:
        comparison = comparisons["offline"][f"oracle_vs_{baseline}"]
        k2 = comparison["fixed_k_2"]["candidate_minus_baseline"]
        prefix = comparison["oracle_prefix"]["candidate_minus_baseline"]
        lines.append(
            f"| `{baseline}` | {k2['mean']:+.4f} [{k2['bootstrap_95_low']:+.4f}, {k2['bootstrap_95_high']:+.4f}] | "
            f"{prefix['mean']:+.4f} [{prefix['bootstrap_95_low']:+.4f}, {prefix['bootstrap_95_high']:+.4f}] |"
        )
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output_dir": str(output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
