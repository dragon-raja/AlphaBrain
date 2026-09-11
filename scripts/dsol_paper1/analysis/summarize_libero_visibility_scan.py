#!/usr/bin/env python3
"""Summarize exact-state M0 visibility scans without tuning on test states."""

from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))


import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def diagnostic_plot_data(
    flat_rows: list[dict], state_rows: list[dict]
) -> tuple[dict[str, list[float]], list[str], list[float]]:
    """Return plot inputs using the protocol's actual split identifiers."""
    geometric_groups = sorted(
        {
            row["group"]
            for row in flat_rows
            if row["group"] not in {"canonical", "sensor_controls"}
        }
    )
    calibration_values = {
        group: [
            row["delta_visibility"]
            for row in flat_rows
            if row["split"] == "calibration" and row["group"] == group
        ]
        for group in geometric_groups
    }
    labels = [
        "calibration best",
        "calibration worst",
        "held-out best",
        "held-out worst",
    ]
    headroom_values = [
        float(
            np.mean(
                [
                    row[f"{kind}_delta_visibility"]
                    for row in state_rows
                    if row["split"] == split
                ]
            )
        )
        for split, kind in (
            ("calibration", "best"),
            ("calibration", "worst"),
            ("heldout_test", "best"),
            ("heldout_test", "worst"),
        )
    ]
    if not any(calibration_values.values()):
        raise ValueError("no calibration geometric candidates for diagnostic plot")
    if not np.all(np.isfinite(headroom_values)):
        raise ValueError("missing calibration or held-out state rows for diagnostic plot")
    return calibration_values, labels, headroom_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledgers", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ledger_rows = []
    for path in args.ledgers:
        ledger_rows.extend(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    passed = {}
    for row in ledger_rows:
        if row.get("status") == "PASS":
            passed[str(row["scan_id"])] = row
    if not passed:
        raise ValueError("no PASS visibility scans")

    flat_rows = []
    state_rows = []
    definition = None
    for scan_id, ledger in sorted(passed.items()):
        scan_path = Path(ledger["output_dir"]) / "scan.json"
        scan = json.loads(scan_path.read_text())
        if scan.get("status") != "PASS":
            raise ValueError(f"non-PASS scan behind PASS ledger: {scan_path}")
        geometric = []
        sensor_controls = {}
        for record in scan["records"]:
            if "delta_visibility" not in record:
                continue
            visibility = record.get("visibility")
            if visibility is not None:
                current_definition = visibility["definition"]
                if definition is None:
                    definition = current_definition
                elif definition != current_definition:
                    raise ValueError("visibility definition changed across scans")
            row = {
                "scan_id": scan_id,
                "split": ledger["split"],
                "task_id": ledger["task_id"],
                "episode_id": ledger["episode_id"],
                "frame": int(ledger["frame"]),
                "stage_fraction": float(ledger["stage_fraction"]),
                "pose_id": record["pose_id"],
                "group": record["group"],
                "visibility_score": float(record["visibility_score"]),
                "delta_visibility": float(record["delta_visibility"]),
                "agentview_score": float(
                    record.get("per_camera_scores", {}).get("agentview", 0.0)
                ),
                "wrist_score": float(
                    record.get("per_camera_scores", {}).get(
                        "robot0_eye_in_hand", 0.0
                    )
                ),
                "camera_translation_m": record.get(
                    "camera_displacement_from_canonical", {}
                ).get("translation_m"),
                "camera_rotation_geodesic_deg": record.get(
                    "camera_displacement_from_canonical", {}
                ).get("rotation_geodesic_deg"),
                "scan_path": str(scan_path),
                "montage_path": str(scan_path.with_name("visibility_extremes.png")),
            }
            flat_rows.append(row)
            if row["group"] == "sensor_controls":
                sensor_controls[row["pose_id"]] = row["delta_visibility"]
            elif row["pose_id"] != "canonical":
                geometric.append(row)
        if not geometric:
            raise ValueError(f"scan has no geometric candidates: {scan_id}")
        best = max(geometric, key=lambda item: item["delta_visibility"])
        worst = min(geometric, key=lambda item: item["delta_visibility"])
        state_rows.append(
            {
                "scan_id": scan_id,
                "split": ledger["split"],
                "task_id": ledger["task_id"],
                "episode_id": ledger["episode_id"],
                "frame": int(ledger["frame"]),
                "stage_fraction": float(ledger["stage_fraction"]),
                "best_pose_id": best["pose_id"],
                "best_group": best["group"],
                "best_delta_visibility": best["delta_visibility"],
                "worst_pose_id": worst["pose_id"],
                "worst_group": worst["group"],
                "worst_delta_visibility": worst["delta_visibility"],
                "external_blackout_delta": sensor_controls.get("external_blackout"),
                "wrist_blackout_delta": sensor_controls.get("wrist_blackout"),
                "all_camera_blackout_delta": sensor_controls.get("all_camera_blackout"),
            }
        )

    group_summary = {}
    for split in sorted({row["split"] for row in flat_rows}):
        for group in sorted({row["group"] for row in flat_rows}):
            values = [
                row["delta_visibility"]
                for row in flat_rows
                if row["split"] == split and row["group"] == group
            ]
            if values:
                group_summary[f"{split}::{group}"] = quantiles(values)

    task_counts = Counter(
        f"{row['split']}::{row['task_id']}" for row in state_rows
    )
    summary = {
        "schema": "dsol_libero_visibility_scan_summary_v1",
        "status": "MEASUREMENT_COMPLETE_THRESHOLD_UNFROZEN",
        "scan_count": len(state_rows),
        "candidate_record_count": len(flat_rows),
        "visibility_definition": definition,
        "task_state_counts": dict(sorted(task_counts.items())),
        "group_delta_statistics": group_summary,
        "selection_policy": (
            "Use val scans and manual montages to freeze strong-info and matched-control "
            "thresholds; apply the frozen rule once to test scans."
        ),
        "test_tuning_forbidden": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for name, rows in (("candidate_records.csv", flat_rows), ("state_extremes.csv", state_rows)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt

        calibration_values, labels, values = diagnostic_plot_data(flat_rows, state_rows)
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
        for group, group_values in calibration_values.items():
            if group_values:
                axes[0].hist(group_values, bins=40, alpha=0.45, label=group)
        axes[0].axvline(0.0, color="#17212b", linewidth=1)
        axes[0].set(
            title="Calibration candidate visibility deltas",
            xlabel="Delta visible-pixel fraction",
            ylabel="Candidate count",
        )
        axes[0].legend(fontsize=7)
        axes[1].bar(labels, values, color=("#3f7fa8", "#c55463", "#3f7fa8", "#c55463"))
        axes[1].axhline(0.0, color="#17212b", linewidth=1)
        axes[1].set(title="Per-state geometric headroom", ylabel="Mean delta visibility")
        for axis in axes:
            axis.grid(alpha=0.25)
        figure.savefig(args.output_dir / "visibility_diagnostics.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
