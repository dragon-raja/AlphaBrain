#!/usr/bin/env python3
"""Convert the frozen view-value population into a visibility scan plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    if population.get("status") != "PASS":
        raise ValueError("population must PASS before scan planning")
    records = []
    counts = {}
    for split in ("calibration", "heldout_test"):
        for state in population["population"][split]["states"]:
            record = {
                **state,
                "scan_id": state["pair_key"],
                "episode_id": state["source_group"],
                "frame": state["source_state_index"],
                "construction_spec": state["construction_spec"],
            }
            records.append(record)
            key = f"{split}::{state['task_id']}"
            counts[key] = counts.get(key, 0) + 1
    payload = {
        "schema": "dsol_view_value_expectation_scan_plan_v1",
        "status": "PASS",
        "population": str(args.population.resolve()),
        "population_sha256": sha256_file(args.population),
        "record_count": len(records),
        "counts": counts,
        "records": records,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": "PASS", "records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
