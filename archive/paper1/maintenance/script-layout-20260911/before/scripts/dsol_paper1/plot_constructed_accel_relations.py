#!/usr/bin/env python3
"""Plot the descriptive fixed-state Accel relation audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROLES = (
    "canonical",
    "strong_info",
    "matched_control",
    "blind",
    "look_away",
    "external_blackout",
    "wrist_blackout",
    "all_camera_blackout",
)


def load_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((root / "states").glob("*/rank_record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rankings = json.loads((path.parent / "rankings.json").read_text(encoding="utf-8"))
        accel = {
            str(row["candidate_id"]): float(row["accel_3"])
            for row in rankings["complete"]["ranking"]
        }
        records.append({"record": record, "accel": accel})
    if not records:
        raise ValueError(f"no rank records in {root}")
    return records


def role_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for payload in records:
        record = payload["record"]
        for role in ROLES:
            values = record["role_metrics"][role]
            candidate_id = str(values["candidate_id"])
            rows.append(
                {
                    "pair_key": record["pair_key"],
                    "task_id": record["task_id"],
                    "source_episode_group": record["source_episode_id"],
                    "role": role,
                    "candidate_id": candidate_id,
                    "accel_3": payload["accel"][candidate_id],
                    "complete_rank": values["complete_rank"],
                    "diagnostic_rank": values["diagnostic_rank"],
                    "visibility_score": values["visibility_score"],
                    "delta_visibility": values["delta_visibility"],
                }
            )
    return rows


def plot(records: list[dict[str, Any]], rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "canonical": "#3f7fa8",
        "strong_info": "#3d9b77",
        "matched_control": "#d39a2c",
        "blind": "#c65b6d",
        "look_away": "#8a6fb1",
        "external_blackout": "#78838f",
        "wrist_blackout": "#4e5964",
        "all_camera_blackout": "#202830",
    }
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

    category_counts = Counter(
        payload["record"]["selected_candidate_categories"]["operational_97"]
        for payload in records
    )
    categories = (
        "canonical",
        "broad64_training_support",
        "broad32_heldout",
    )
    values = [category_counts[value] for value in categories]
    axes[0].bar(range(len(categories)), values, color=("#3f7fa8", "#4d9f72", "#d39a2c"))
    axes[0].set_xticks(range(len(categories)), ("Canonical", "Train-64", "Held-out-32"))
    axes[0].set_ylabel("Selected states")
    axes[0].set_title("A. Accel selection in the 97-view bank")
    for index, value in enumerate(values):
        axes[0].text(index, value + 0.25, str(value), ha="center")

    top1 = []
    for role in ROLES:
        selected = [row for row in rows if row["role"] == role]
        top1.append(100.0 * np.mean([row["diagnostic_rank"] == 1 for row in selected]))
    axes[1].barh(
        range(len(ROLES)),
        top1,
        color=[colors[role] for role in ROLES],
    )
    axes[1].set_yticks(range(len(ROLES)), [role.replace("_", " ") for role in ROLES])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Top-1 selection rate (%)")
    axes[1].set_title("B. Frozen diagnostic shortlist")

    for role in ("strong_info", "matched_control", "blind", "canonical"):
        selected = [row for row in rows if row["role"] == role]
        axes[2].scatter(
            [row["delta_visibility"] for row in selected],
            [row["diagnostic_rank"] for row in selected],
            s=28,
            alpha=0.75,
            color=colors[role],
            label=role.replace("_", " "),
        )
    axes[2].axvline(0.0, color="#202830", linewidth=1, alpha=0.5)
    axes[2].set_xlabel("Visibility change from canonical")
    axes[2].set_ylabel("Accel rank (lower is preferred)")
    axes[2].invert_yaxis()
    axes[2].set_title("C. Visibility gain vs. Accel preference")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Constructed M0 Accel audit: descriptive relation only (21 states)",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accel-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.accel_root)
    rows = role_rows(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "role_state_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot(records, rows, args.output_dir / "accel_relation_audit.png")
    print(
        json.dumps(
            {
                "status": "PASS",
                "state_count": len(records),
                "role_state_rows": len(rows),
                "plot": str(args.output_dir / "accel_relation_audit.png"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
