#!/usr/bin/env python3
"""Build an exact-HDF5 paired closed-loop protocol for the DSOL quick gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stable_index(seed: int, identity: str, size: int) -> int:
    digest = hashlib.sha256(f"{seed}::{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    scan_plan = json.loads(args.scan_plan.read_text())
    catalog = json.loads(args.catalog.read_text())
    episodes = {}
    for row in scan_plan["records"]:
        if row["split"] != "test":
            continue
        episodes.setdefault(row["episode_id"], row)
    heldout = catalog["broad_heldout_32"]
    extrapolation = catalog["wide_extrapolation_24"]
    specs = []
    for episode_id, row in sorted(episodes.items()):
        heldout_pose = heldout[stable_index(args.seed, f"{episode_id}::heldout", len(heldout))]
        extrapolation_pose = extrapolation[
            stable_index(args.seed, f"{episode_id}::extrapolation", len(extrapolation))
        ]
        pair_key = f"{row['task_id']}::{episode_id}"
        common = {
            "pair_key": pair_key,
            "task_id": row["task_id"],
            "diagnostic_role": row["diagnostic_role"],
            "suite": row["suite"],
            "hdf5": row["hdf5"],
            "episode_id_source": episode_id,
            "demo_name": row["demo_name"],
            "demo_index": row["demo_index"],
            "split": "test",
            "source_state_index": 0,
        }
        conditions = (
            ("canonical_both", None, "both"),
            ("canonical_external_only", None, "external_only"),
            ("canonical_wrist_only", None, "wrist_only"),
            ("broad_heldout_both", heldout_pose, "both"),
            ("broad_heldout_external_only", heldout_pose, "external_only"),
            ("broad_heldout_wrist_only", heldout_pose, "wrist_only"),
            ("wide_extrapolation_both", extrapolation_pose, "both"),
        )
        for condition, pose, sensor_control in conditions:
            identity = f"{pair_key}::{condition}"
            specs.append(
                {
                    **common,
                    "condition": condition,
                    "pose": pose,
                    "sensor_control": sensor_control,
                    "episode_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                }
            )
    payload = {
        "schema": "dsol_libero_hdf5_closed_loop_protocol_v1",
        "seed": args.seed,
        "source_scan_plan": str(args.scan_plan.resolve()),
        "source_scan_plan_sha256": hashlib.sha256(args.scan_plan.read_bytes()).hexdigest(),
        "catalog": str(args.catalog.resolve()),
        "catalog_sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        "pair_count": len(episodes),
        "condition_count": 7,
        "episode_count": len(specs),
        "specs": specs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: payload[key] for key in ("pair_count", "condition_count", "episode_count")}, indent=2))


if __name__ == "__main__":
    main()
