#!/usr/bin/env python3
"""Freeze task-level Blind-Reveal pair direction from validation scans only."""

from __future__ import annotations

import argparse
import glob
import json
import math
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any


SCHEMA = "dsol_constructed_task_pair_freeze_v1"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_rows(patterns: Iterable[str]) -> list[dict[str, Any]]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def episode_medians(values: Sequence[tuple[str, float]]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for episode_id, value in values:
        grouped[str(episode_id)].append(float(value))
    return [float(median(grouped[key])) for key in sorted(grouped)]


def candidate_records(scan: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result = {}
    for record in scan["records"]:
        if record.get("group") != "constructed_task_orbit":
            continue
        pose = record.get("pose", {})
        key = (str(pose.get("pair_id")), str(pose.get("pair_member")))
        if not all(key) or key in result:
            raise ValueError(f"invalid or duplicate constructed candidate: {key}")
        result[key] = record
    return result


def freeze(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_strong_delta: float,
    maximum_control_abs_delta: float,
) -> dict[str, Any]:
    eligible = []
    excluded = defaultdict(int)
    for row in rows:
        if row.get("split") != "val":
            excluded["non_validation"] += 1
            continue
        if row.get("status") != "PASS":
            excluded["scan_not_pass"] += 1
            continue
        scan = json.loads((Path(row["output_dir"]) / "scan.json").read_text())
        if bool(scan.get("initial_task_success")):
            excluded["initial_task_success"] += 1
            continue
        if float(row.get("stage_fraction", 1.0)) > 0.70:
            excluded["stage_fraction_above_070"] += 1
            continue
        eligible.append((row, scan, candidate_records(scan)))

    task_rows: dict[str, list[Any]] = defaultdict(list)
    for item in eligible:
        task_rows[str(item[0]["task_id"])].append(item)
    frozen = {}
    for task_id, items in sorted(task_rows.items()):
        pair_ids = sorted(
            set.intersection(
                *[
                    {pair_id for pair_id, _member in records}
                    for _row, _scan, records in items
                ]
            )
        )
        scored = []
        for pair_id in pair_ids:
            for strong_member, control_member in (
                ("negative", "positive"),
                ("positive", "negative"),
            ):
                strong_values = episode_medians(
                    [
                        (str(row["episode_id"]), float(records[(pair_id, strong_member)]["delta_visibility"]))
                        for row, _scan, records in items
                    ]
                )
                control_values = episode_medians(
                    [
                        (str(row["episode_id"]), float(records[(pair_id, control_member)]["delta_visibility"]))
                        for row, _scan, records in items
                    ]
                )
                strong_median = float(median(strong_values))
                control_median = float(median(control_values))
                scored.append(
                    {
                        "pair_id": pair_id,
                        "strong_member": strong_member,
                        "control_member": control_member,
                        "validation_episode_count": len(strong_values),
                        "strong_episode_median_delta": strong_median,
                        "control_episode_median_delta": control_median,
                        "information_specificity": strong_median - control_median,
                        "passes_strong_threshold": strong_median >= minimum_strong_delta,
                        "passes_control_threshold": abs(control_median) <= maximum_control_abs_delta,
                    }
                )
        passing = [
            row
            for row in scored
            if row["passes_strong_threshold"] and row["passes_control_threshold"]
        ]
        ranking = passing if passing else scored
        selected = max(
            ranking,
            key=lambda row: (
                bool(row["passes_strong_threshold"] and row["passes_control_threshold"]),
                float(row["information_specificity"]),
                float(row["strong_episode_median_delta"]),
                -abs(float(row["control_episode_median_delta"])),
                str(row["pair_id"]),
                str(row["strong_member"]),
            ),
        )
        frozen[task_id] = {
            "status": (
                "PASS"
                if selected["passes_strong_threshold"]
                and selected["passes_control_threshold"]
                else "FAIL"
            ),
            "eligible_validation_state_count": len(items),
            "eligible_validation_episode_count": len(
                {str(row["episode_id"]) for row, _scan, _records in items}
            ),
            "selected": selected,
            "all_candidates": scored,
        }
    status = "PASS" if frozen and all(row["status"] == "PASS" for row in frozen.values()) else "FAIL"
    return {
        "schema": SCHEMA,
        "status": status,
        "selection_split": "validation_only",
        "policy_outcomes_used": False,
        "statistical_unit": "source_episode",
        "minimum_strong_delta": minimum_strong_delta,
        "maximum_control_abs_delta": maximum_control_abs_delta,
        "eligible_validation_state_count": len(eligible),
        "exclusions": dict(sorted(excluded.items())),
        "tasks": frozen,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-strong-delta", type=float, default=0.005)
    parser.add_argument("--maximum-control-abs-delta", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.minimum_strong_delta) or args.minimum_strong_delta <= 0:
        raise ValueError("minimum strong delta must be positive and finite")
    if not math.isfinite(args.maximum_control_abs_delta) or args.maximum_control_abs_delta < 0:
        raise ValueError("maximum control delta must be nonnegative and finite")
    result = freeze(
        load_rows(args.inputs),
        minimum_strong_delta=args.minimum_strong_delta,
        maximum_control_abs_delta=args.maximum_control_abs_delta,
    )
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "tasks": len(result["tasks"])}, sort_keys=True))


if __name__ == "__main__":
    main()
