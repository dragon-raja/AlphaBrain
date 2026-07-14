from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.fresh_vla.paired_evaluation import bootstrap_summary
except ModuleNotFoundError:
    from paired_evaluation import bootstrap_summary


METHODS = (
    "full_h",
    "random_soft010",
    "shuffled_oracle_soft010",
    "early_oracle_soft010",
    "late_oracle_soft010",
    "gripper_soft010",
    "oracle_soft010",
    "short_h",
)
METRICS = ("fixed_k_1", "fixed_k_2", "fixed_k_3", "oracle_prefix", "suffix", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FRESH LIBERO training and paired offline evaluation")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/libero-counterfactual-v1"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/libero-counterfactual-v1-128"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _run_summary(run_dir: Path) -> dict[str, object]:
    train_rows = _load_jsonl(run_dir / "metrics.jsonl")
    evaluation = json.loads((run_dir / "offline_eval.json").read_text())
    control_evaluation = json.loads((run_dir / "deterministic_control_eval.json").read_text())
    for result in (evaluation, control_evaluation):
        for row in result["rows"]:
            per_step = np.asarray(row["per_step"], dtype=np.float64)
            for fixed_k in (1, 2, 3):
                if fixed_k <= len(per_step):
                    row[f"fixed_k_{fixed_k}"] = float(per_step[:fixed_k].mean())
        for fixed_k in (1, 2, 3):
            key = f"fixed_k_{fixed_k}"
            values = [float(row[key]) for row in result["rows"] if key in row]
            if values:
                result["summary"].setdefault(key, {"mean": float(statistics.mean(values))})
    max_step = max(int(row["step"]) for row in train_rows)
    windows = {}
    for start, end in ((0, 400), (400, 800), (800, max_step)):
        selected = [row for row in train_rows if start < int(row["step"]) <= end]
        windows[f"{start + 1}-{end}"] = {
            key: _mean([float(row[key]) for row in selected if key in row])
            for key in ("fresh_full_loss", "fresh_prefix_loss", "fresh_suffix_loss")
        }
    last_rows = [row for row in train_rows if int(row["step"]) > max_step - 200]
    horizon = len([key for key in train_rows[-1] if re.fullmatch(r"fresh_step_loss_fraction_\d+", key)])
    return {
        "max_step": max_step,
        "logged_points": len(train_rows),
        "windows": windows,
        "last_200_step_loss_fractions": [
            _mean([float(row[f"fresh_step_loss_fraction_{step:02d}"]) for row in last_rows])
            for step in range(horizon)
        ],
        "offline_summary": evaluation["summary"],
        "control_summary": control_evaluation["summary"],
        "evaluation_fingerprint": evaluation["evaluation_fingerprint"],
        "offline_rows": evaluation["rows"],
        "control_rows": control_evaluation["rows"],
    }


def _aggregate_seed_means(run_summaries: list[dict[str, object]], metric: str) -> dict[str, float]:
    values = [float(run["offline_summary"][metric]["mean"]) for run in run_summaries]
    return {
        "seed_count": len(values),
        "mean": float(statistics.mean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _aggregate_control_seed_means(run_summaries: list[dict[str, object]], metric: str) -> dict[str, float]:
    values = [float(run["control_summary"][metric]["mean"]) for run in run_summaries]
    return {
        "seed_count": len(values),
        "mean": float(statistics.mean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _paired_deltas(
    baseline: list[dict[str, object]],
    candidate: list[dict[str, object]],
    metric: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    deltas = []
    for baseline_run, candidate_run in zip(baseline, candidate, strict=True):
        if baseline_run["evaluation_fingerprint"] != candidate_run["evaluation_fingerprint"]:
            raise ValueError("evaluation fingerprints do not match")
        baseline_rows = {row["sample_id"]: row for row in baseline_run["offline_rows"]}
        candidate_rows = {row["sample_id"]: row for row in candidate_run["offline_rows"]}
        if baseline_rows.keys() != candidate_rows.keys():
            raise ValueError("paired sample IDs do not match")
        deltas.extend(
            float(candidate_rows[sample_id][metric]) - float(baseline_rows[sample_id][metric])
            for sample_id in sorted(baseline_rows)
            if baseline_rows[sample_id][metric] is not None and candidate_rows[sample_id][metric] is not None
        )
    return {
        "candidate_minus_full_h": bootstrap_summary(
            deltas,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "candidate_better": sum(delta < 0 for delta in deltas),
        "full_h_better": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
    }


def _plot_curves(runs_root: Path, seeds: list[int], output: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    keys = ("fresh_full_loss", "fresh_prefix_loss", "fresh_suffix_loss")
    titles = ("Full horizon", "Oracle prefix", "Suffix")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    for method in METHODS:
        curves = []
        for seed in seeds:
            rows = _load_jsonl(runs_root / f"fresh_libero_{method}_seed{seed}" / "metrics.jsonl")
            curves.append({int(row["step"]): row for row in rows})
        common_steps = sorted(set.intersection(*(set(curve) for curve in curves)))
        for axis, key, title in zip(axes, keys, titles, strict=True):
            values = np.asarray([[float(curve[step][key]) for step in common_steps] for curve in curves])
            smooth = np.asarray(
                [np.convolve(row, np.ones(5) / 5, mode="same") for row in values]
            )
            mean = smooth.mean(axis=0)
            std = smooth.std(axis=0)
            axis.plot(common_steps, mean, label=method, linewidth=1.5)
            axis.fill_between(common_steps, mean - std, mean + std, alpha=0.1)
            axis.set_title(title)
            axis.set_xlabel("training step")
            axis.set_ylabel("FM MSE")
            axis.grid(alpha=0.25)
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def _fmt(summary: dict[str, float]) -> str:
    return f"{summary['mean']:.4f} +/- {summary['sample_std']:.4f}"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.runs_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)
    runs: dict[str, list[dict[str, object]]] = defaultdict(list)
    for method in METHODS:
        for seed in seeds:
            run_dir = args.runs_root / f"fresh_libero_{method}_seed{seed}"
            runs[method].append(_run_summary(run_dir))

    aggregate = {}
    for method, method_runs in runs.items():
        aggregate[method] = {metric: _aggregate_seed_means(method_runs, metric) for metric in METRICS}
        aggregate[method]["deterministic_fixed_k_2"] = _aggregate_control_seed_means(
            method_runs, "fixed_k_2"
        )
    paired = {
        method: {
            metric: _paired_deltas(
                runs["full_h"],
                method_runs,
                metric,
                bootstrap_samples=args.bootstrap_samples,
                seed=1701 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
        for method, method_runs in runs.items()
        if method != "full_h"
    }
    quality = json.loads((args.data_root / "quality_report.json").read_text())
    payload = {
        "data_root": str(args.data_root),
        "data_quality": quality,
        "seeds": seeds,
        "methods": list(METHODS),
        "runs": runs,
        "aggregate": aggregate,
        "paired_vs_full_h": paired,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    plotted = _plot_curves(args.runs_root, seeds, output_dir / "training_curves.png")

    lines = [
        "# FRESH-VLA LIBERO Training Results",
        "",
        f"Seeds: `{', '.join(map(str, seeds))}`. Lower FM MSE is better.",
        "",
        "| Method | K=1 | K=2 | K=3 | Oracle prefix | Suffix | Full | Deterministic K=2 | Paired K=2 delta vs Full-H |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        paired_text = "reference"
        if method != "full_h":
            delta = paired[method]["fixed_k_2"]["candidate_minus_full_h"]
            paired_text = f"{delta['mean']:+.4f} [{delta['bootstrap_95_low']:+.4f}, {delta['bootstrap_95_high']:+.4f}]"
        lines.append(
            f"| `{method}` | {_fmt(aggregate[method]['fixed_k_1'])} | "
            f"{_fmt(aggregate[method]['fixed_k_2'])} | {_fmt(aggregate[method]['fixed_k_3'])} | "
            f"{_fmt(aggregate[method]['oracle_prefix'])} | {_fmt(aggregate[method]['suffix'])} | "
            f"{_fmt(aggregate[method]['full'])} | {_fmt(aggregate[method]['deterministic_fixed_k_2'])} | "
            f"{paired_text} |"
        )
    lines.extend(
        [
            "",
            "Paired deltas are candidate minus Full-H with a sample-level bootstrap 95% interval.",
            "This report is an offline fixed-noise FM evaluation; it is not a closed-loop success-rate claim.",
            "",
        ]
    )
    if plotted:
        lines.extend(("![Training curves](training_curves.png)", ""))
    (output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({"output_dir": str(output_dir), "aggregate": aggregate}, sort_keys=True))


if __name__ == "__main__":
    main()
