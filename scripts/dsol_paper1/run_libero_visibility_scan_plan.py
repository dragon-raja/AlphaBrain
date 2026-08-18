#!/usr/bin/env python3
"""Execute one resumable shard of a DSOL M0 visibility scan plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--groups", default="broad_heldout_32,wide_extrapolation_24,diagnostic_extreme_orbit,diagnostic_look_away")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--render-gpu", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    plan = json.loads(args.plan.read_text())
    selected = [
        row
        for index, row in enumerate(plan["records"])
        if index % args.num_shards == args.shard_index
    ]
    if args.max_states is not None:
        if args.max_states <= 0:
            raise ValueError("max-states must be positive")
        selected = selected[: args.max_states]
    args.output_root.mkdir(parents=True, exist_ok=True)
    ledger = args.output_root / f"shard-{args.shard_index:02d}.jsonl"
    completed = set()
    if ledger.exists():
        completed = {
            json.loads(line)["scan_id"]
            for line in ledger.read_text().splitlines()
            if line.strip() and json.loads(line).get("status") == "PASS"
        }
    scanner = Path(__file__).resolve().with_name("scan_libero_hdf5_views.py")
    for ordinal, row in enumerate(selected, start=1):
        scan_id = str(row["scan_id"])
        if scan_id in completed:
            continue
        safe_name = hashlib.sha256(scan_id.encode()).hexdigest()[:16]
        output_dir = args.output_root / "states" / safe_name
        command = [
            sys.executable,
            str(scanner),
            "--hdf5",
            str(row["hdf5"]),
            "--runtime",
            str(args.runtime),
            "--catalog",
            str(args.catalog),
            "--config-root",
            str(args.config_root),
            "--output-dir",
            str(output_dir),
            "--groups",
            args.groups,
            "--demo-index",
            str(row["demo_index"]),
            "--frame-index",
            str(row["frame"]),
            "--render-gpu",
            str(args.render_gpu),
        ]
        print(
            json.dumps(
                {"ordinal": ordinal, "total": len(selected), "scan_id": scan_id},
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            subprocess.run(command, check=True)
            result = json.loads((output_dir / "scan.json").read_text())
            status = (
                "PASS"
                if result["status"] == "PASS" and int(result["invalid_records"]) == 0
                else "FAIL"
            )
            append_jsonl(
                ledger,
                {
                    **row,
                    "status": status,
                    "output_dir": str(output_dir),
                    "valid_records": result["valid_records"],
                    "invalid_records": result["invalid_records"],
                    "delta_visibility_min": result["delta_visibility_min"],
                    "delta_visibility_max": result["delta_visibility_max"],
                },
            )
        except Exception as error:
            append_jsonl(
                ledger,
                {
                    **row,
                    "status": "ERROR",
                    "output_dir": str(output_dir),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise


if __name__ == "__main__":
    main()
