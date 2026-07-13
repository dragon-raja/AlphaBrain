from __future__ import annotations

import argparse
import json
from pathlib import Path

from paired_evaluation import bootstrap_summary


METRICS = (
    "fixed_k_prefix_mse",
    "safe_prefix_mse",
    "premature_commitment",
    "flow_fixed_k3_mse",
    "flow_oracle_prefix_mse",
    "flow_suffix_mse",
    "flow_full_mse",
)


def compare(
    results: list[dict[str, object]],
    candidate: str,
    baseline: str,
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    indexed = {(str(row["method"]), int(row["seed"])): row for row in results}
    seeds = sorted(seed for method, seed in indexed if method == candidate)
    output = {}
    for metric in METRICS:
        seed_deltas = []
        pooled_deltas = []
        per_seed = {}
        for seed in seeds:
            candidate_row = indexed[(candidate, seed)]
            baseline_row = indexed[(baseline, seed)]
            if candidate_row["evaluation_fingerprint"] != baseline_row["evaluation_fingerprint"]:
                raise ValueError(f"evaluation identity mismatch for seed {seed}")
            candidate_values = candidate_row["per_sample"][metric]
            baseline_values = baseline_row["per_sample"][metric]
            deltas = [left - right for left, right in zip(candidate_values, baseline_values, strict=True)]
            summary = bootstrap_summary(deltas, bootstrap_samples=bootstrap_samples, seed=seed)
            per_seed[str(seed)] = summary
            seed_deltas.append(summary["mean"])
            pooled_deltas.extend(deltas)
        output[metric] = {
            "per_seed": per_seed,
            "seed_mean_delta": bootstrap_summary(
                seed_deltas,
                bootstrap_samples=bootstrap_samples,
                seed=1234,
            ),
            "pooled_sample_delta": bootstrap_summary(
                pooled_deltas,
                bootstrap_samples=bootstrap_samples,
                seed=5678,
            ),
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a seed-aware compact summary of paired FRESH results")
    parser.add_argument("input", type=Path)
    parser.add_argument("--candidate", default="oracle_soft010")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["full_h", "random_soft010", "gripper_soft010", "short_h", "remac_prefix_mask_control"],
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    summary = {
        "source": str(args.input),
        "candidate": args.candidate,
        "method_summary": payload["summary"],
        "comparisons": {
            baseline: compare(
                payload["results"],
                args.candidate,
                baseline,
                bootstrap_samples=args.bootstrap_samples,
            )
            for baseline in args.baselines
        },
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(args.output)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
