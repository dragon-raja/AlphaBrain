#!/usr/bin/env python3
"""Summarize budget-matched DSOL LIBERO training smoke runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ARMS = (
    "canonical_unique",
    "canonical_repeat",
    "image_augmentation_unique",
    "broad_unpaired_practical",
    "broad_unpaired_state_matched",
    "broad_paired_fm",
    "broad_paired_consistency",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--global-examples", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_accounting(path: Path) -> dict[str, int | str]:
    values: dict[str, int | str] = {}
    for key, value in re.findall(r"([a-z_]+)=([^ ]+)", path.read_text().strip()):
        values[key] = int(value) if value.isdigit() else value
    return values


def main() -> None:
    args = parse_args()
    rows = []
    for arm in ARMS:
        run_id = (
            f"dsol_{arm}_{args.tag}_seed{args.seed}_g8_"
            f"gb{args.global_examples}_steps{args.steps}"
        )
        root = args.run_root / run_id
        metrics_path = root / "metrics.jsonl"
        launcher_path = root / "launcher.log"
        accounting_path = root / "batch_accounting.txt"
        metrics = [
            json.loads(line)
            for line in metrics_path.read_text().splitlines()
            if line.strip()
        ] if metrics_path.is_file() else []
        launcher = launcher_path.read_text(errors="replace") if launcher_path.is_file() else ""
        accounting = parse_accounting(accounting_path) if accounting_path.is_file() else {}
        final = metrics[-1] if metrics else {}
        rows.append({
            "arm": arm,
            "run_id": run_id,
            "status": "PASS" if len(metrics) == args.steps and "Training complete" in launcher else "FAIL",
            "optimizer_steps": len(metrics),
            "global_model_examples": accounting.get("global_model_examples"),
            "examples_per_item": accounting.get("examples_per_item"),
            "gradient_accumulation_steps": accounting.get("grad_acc"),
            "examples_seen": final.get("examples_seen"),
            "final_action_dit_loss": final.get("action_dit_loss"),
            "final_flow_matching_loss": final.get("flow_matching_loss"),
            "final_pair_consistency_loss": final.get("dsol_pair_consistency_loss"),
            "traceback_count": launcher.count("Traceback"),
            "runtime_error_count": launcher.count("RuntimeError"),
            "run_root": str(root),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dsol_libero_pair_smoke_summary_v1",
        "tag": args.tag,
        "seed": args.seed,
        "intended_optimizer_steps": args.steps,
        "intended_global_model_examples": args.global_examples,
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "interpretation": "Infrastructure smoke only; losses are not an algorithm ranking.",
        "runs": rows,
    }
    json_path = args.output_dir / "smoke_summary.json"
    csv_path = args.output_dir / "smoke_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": payload["status"], "json": str(json_path), "csv": str(csv_path)}))


if __name__ == "__main__":
    main()
