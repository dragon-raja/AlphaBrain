#!/usr/bin/env python3
"""Summarize a fixed-ledger DSOL convergence-calibration run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(run_dir / "metrics.jsonl")
    validation = [
        {
            "step": int(row["step"]),
            "action_loss": float(row["dsol_val_action_loss"]),
            "examples": int(row["dsol_val_examples"]),
        }
        for row in metrics
        if "dsol_val_action_loss" in row
    ]
    if not validation:
        raise ValueError(f"no DSOL validation records in {run_dir / 'metrics.jsonl'}")
    if len({row["examples"] for row in validation}) != 1:
        raise ValueError("validation example count changed within the calibration run")

    expected_match = re.search(r"_steps(\d+)$", run_dir.name)
    expected_steps = int(expected_match.group(1)) if expected_match else None
    last_train_step = max(int(row.get("step", 0)) for row in metrics)
    best = min(validation, key=lambda row: (row["action_loss"], row["step"]))
    threshold = best["action_loss"] * 1.01
    earliest_within_one_percent = min(
        (row for row in validation if row["step"] > 0 and row["action_loss"] <= threshold),
        key=lambda row: row["step"],
        default=best,
    )

    ledger = json.loads((run_dir / "dsol_validation_ledger.json").read_text())
    task_counts = Counter(
        row["episode_id"].rsplit("::demo_", 1)[0] for row in ledger["records"]
    )
    summary = {
        "schema": "dsol_convergence_calibration_summary_v1",
        "status": (
            "COMPLETE"
            if expected_steps is not None and last_train_step >= expected_steps
            else "RUNNING"
        ),
        "run_dir": str(run_dir),
        "expected_steps": expected_steps,
        "last_train_step": last_train_step,
        "validation_points": len(validation),
        "validation_examples": validation[0]["examples"],
        "validation_task_counts": dict(sorted(task_counts.items())),
        "best_validation": best,
        "earliest_within_one_percent_of_best": earliest_within_one_percent,
        "curve": validation,
        "interpretation": (
            "Calibration selects one uniform budget only; it does not rank methods "
            "or replace closed-loop evaluation."
        ),
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "calibration_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "action_loss", "examples"))
        writer.writeheader()
        writer.writerows(validation)

    try:
        import matplotlib.pyplot as plt

        train_steps = [int(row["step"]) for row in metrics if "action_dit_loss" in row]
        train_losses = [float(row["action_dit_loss"]) for row in metrics if "action_dit_loss" in row]
        window = 100
        rolling = [
            sum(train_losses[max(0, index - window + 1) : index + 1])
            / min(window, index + 1)
            for index in range(len(train_losses))
        ]
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
        axes[0].plot(train_steps, rolling, color="#3b82a0", linewidth=1.8)
        axes[0].set(title="Training FM loss (rolling 100)", xlabel="Optimizer step", ylabel="Loss")
        axes[1].plot(
            [row["step"] for row in validation],
            [row["action_loss"] for row in validation],
            marker="o",
            color="#c55463",
            linewidth=1.8,
        )
        axes[1].axvline(best["step"], color="#17212b", linestyle="--", linewidth=1)
        axes[1].set(title="Fixed held-out FM loss", xlabel="Optimizer step", ylabel="Loss")
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.savefig(output_dir / "calibration_curve.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
