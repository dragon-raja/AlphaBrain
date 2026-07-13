from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def method_tail_weight(method: str) -> float | None:
    explicit = {"full_h": 1.0, "oracle_hard000": 0.0, "oracle_soft010": 0.1}
    if method in explicit:
        return explicit[method]
    match = re.fullmatch(r"oracle_soft_(\d{3})", method)
    return int(match.group(1)) / 1000 if match else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FRESH tail-weight trade-offs after the strict gate passes")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    rows = []
    for method, metrics in payload["summary"].items():
        weight = method_tail_weight(method)
        if weight is not None:
            rows.append((weight, method, metrics))
    if len(rows) < 3:
        raise ValueError("lambda plot requires at least three tail weights; do not run the sweep before the gate passes")
    rows.sort()

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    plots = (
        ("safe_prefix_mse_mean", "Common-prefix MSE"),
        ("premature_commitment_mean", "Premature commitment"),
        ("flow_suffix_mse_mean", "Suffix FM MSE"),
        ("fixed_k_prefix_mse_mean", "Fixed-K prefix MSE"),
    )
    weights = [row[0] for row in rows]
    for axis, (metric, title) in zip(axes.flat, plots, strict=True):
        axis.plot(weights, [row[2][metric] for row in rows], marker="o")
        axis.set_title(title)
        axis.set_xlabel("Tail weight")
        axis.grid(alpha=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
