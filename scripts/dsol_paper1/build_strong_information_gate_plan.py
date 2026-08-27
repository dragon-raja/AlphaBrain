#!/usr/bin/env python3
"""Build an outcome-blind early-state plan for the strong-information gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "dsol_strong_information_gate_scan_plan_v1"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def parse_task_spec(value: str) -> tuple[str, Path]:
    task_id, separator, path = value.partition("=")
    if not separator or not task_id or not path:
        raise argparse.ArgumentTypeError("task spec must be TASK_ID=/absolute/spec.json")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise argparse.ArgumentTypeError(f"construction spec does not exist: {resolved}")
    return task_id, resolved


def build(
    source: dict[str, Any],
    task_specs: dict[str, Path],
    *,
    maximum_stage: float,
) -> dict[str, Any]:
    selected = []
    counts: Counter[str] = Counter()
    for source_row in source["records"]:
        task_id = str(source_row["task_id"])
        if task_id not in task_specs:
            continue
        if float(source_row["stage_fraction"]) > maximum_stage:
            continue
        row = dict(source_row)
        row["construction_spec"] = str(task_specs[task_id])
        selected.append(row)
        counts[f"{row['split']}::{task_id}"] += 1
    if not selected:
        raise ValueError("no source records pass the task and stage filters")
    source_path = Path(source["_source_path"])
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "selection_policy": "task_and_early_stage_only",
        "policy_outcomes_used_for_selection": False,
        "maximum_stage_fraction": maximum_stage,
        "source_plan": str(source_path),
        "source_plan_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "task_construction_specs": {
            task_id: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for task_id, path in sorted(task_specs.items())
        },
        "record_count": len(selected),
        "counts": dict(sorted(counts.items())),
        "records": selected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--task-spec", action="append", type=parse_task_spec, required=True)
    parser.add_argument("--maximum-stage", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.maximum_stage <= 1.0:
        raise ValueError("maximum stage must lie in [0, 1]")
    task_specs = dict(args.task_spec)
    if len(task_specs) != len(args.task_spec):
        raise ValueError("task specifications must have unique task IDs")
    source_path = args.source_plan.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["_source_path"] = str(source_path)
    result = build(source, task_specs, maximum_stage=args.maximum_stage)
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
