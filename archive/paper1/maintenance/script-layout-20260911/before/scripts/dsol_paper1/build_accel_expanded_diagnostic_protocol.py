#!/usr/bin/env python3
"""Build a balanced test-only protocol for expanded Accel diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from select_constructed_m0_candidates import (
    BLIND_GROUPS,
    DEFAULT_PROTOCOL,
    LOOK_AWAY_GROUPS,
    _canonical,
    _operational_records,
    _raw_control_pair,
    _raw_strong,
    _records_in_groups,
    load_scan_inputs,
)


PROTOCOL_SCHEMA = "dsol_constructed_m1_frozen_closed_loop_protocol_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def role_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pose_id": str(record["pose_id"]),
        "group": str(record["group"]),
        "visibility_score": float(record["visibility_score"]),
        "delta_visibility": float(record["delta_visibility"]),
    }


def fallback_control(
    snapshot: Mapping[str, Any], strong: Mapping[str, Any]
) -> Mapping[str, Any]:
    candidates = [
        row
        for row in _operational_records(snapshot)
        if str(row["pose_id"]) != str(strong["pose_id"])
    ]
    if not candidates:
        raise ValueError(f"no control candidate for {snapshot['scan_id']}")
    return min(
        candidates,
        key=lambda row: (abs(float(row["delta_visibility"])), str(row["pose_id"])),
    )


def build(
    *,
    snapshots: list[dict[str, Any]],
    scan_plan: Mapping[str, Any],
    catalog_path: Path,
    input_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if input_audit.get("status") != "PASS":
        raise ValueError("visibility scan input audit did not PASS")
    plan_by_id = {str(row["scan_id"]): row for row in scan_plan["records"]}
    test_snapshots = sorted(
        (row for row in snapshots if row["split"] == "test"),
        key=lambda row: str(row["scan_id"]),
    )
    specs = []
    selected_states = []
    fallback_control_count = 0
    for snapshot in test_snapshots:
        scan_id = str(snapshot["scan_id"])
        source = plan_by_id[scan_id]
        canonical = _canonical(snapshot)
        strong = _raw_strong(snapshot)
        if canonical is None or strong is None:
            raise ValueError(f"missing canonical or max-visibility view: {scan_id}")
        control_result = _raw_control_pair(snapshot, strong, DEFAULT_PROTOCOL)
        if control_result is None:
            control = fallback_control(snapshot, strong)
            fallback_control_count += 1
        else:
            control = control_result[0]
        blind_candidates = _records_in_groups(snapshot, BLIND_GROUPS)
        look_away_candidates = _records_in_groups(snapshot, LOOK_AWAY_GROUPS)
        if not blind_candidates or not look_away_candidates:
            raise ValueError(f"missing diagnostic extremes: {scan_id}")
        blind = min(
            blind_candidates,
            key=lambda row: (float(row["visibility_score"]), str(row["pose_id"])),
        )
        look_away = min(
            look_away_candidates,
            key=lambda row: (float(row["visibility_score"]), str(row["pose_id"])),
        )
        visibility_selection = {
            "canonical": role_record(canonical),
            "strong_info": role_record(strong),
            "matched_control": role_record(control),
            "blind": role_record(blind),
            "look_away": role_record(look_away),
            "all_camera_blackout": {
                "pose_id": "all_camera_blackout",
                "group": "sensor_controls",
                "visibility_score": 0.0,
                "delta_visibility": -float(canonical["visibility_score"]),
            },
        }
        common = {
            "pair_key": scan_id,
            "scan_id": scan_id,
            "task_id": str(source["task_id"]),
            "diagnostic_role": "expanded_accel_training_diagnostic",
            "suite": str(source["suite"]),
            "hdf5": str(source["hdf5"]),
            "episode_id_source": str(source["episode_id"]),
            "demo_name": str(source["demo_name"]),
            "demo_index": int(source["demo_index"]),
            "split": "test",
            "source_state_index": int(source["frame"]),
            "stage_fraction": float(source["stage_fraction"]),
            "visibility_selection": visibility_selection,
            "catalog": str(catalog_path.resolve()),
            "manual_audit_verified": False,
            "condition": "canonical_both",
            "pose": None,
            "sensor_control": "both",
            "episode_id": hashlib.sha256(
                f"{scan_id}::canonical_both".encode()
            ).hexdigest()[:20],
        }
        specs.append(common)
        selected_states.append(
            {
                "snapshot_group_id": scan_id,
                "task_id": str(source["task_id"]),
                "source_episode_id": str(source["episode_id"]),
                "source_frame": int(source["frame"]),
                "max_visibility_pose_id": str(strong["pose_id"]),
                "max_visibility_delta": float(strong["delta_visibility"]),
            }
        )
    task_counts = Counter(row["task_id"] for row in selected_states)
    if len(task_counts) < 2 or len(set(task_counts.values())) != 1:
        raise ValueError(f"expanded test protocol is not task-balanced: {task_counts}")
    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "PASS",
        "analysis_role": "expanded_accel_diagnostic_only_no_closed_loop_claim",
        "candidate_selection_frozen_before_manual_audit": False,
        "manual_audit_verified": False,
        "test_threshold_retuning": False,
        "statistical_unit": "source HDF5 demonstration; frames clustered within source",
        "selected_state_count": len(selected_states),
        "condition_count": 1,
        "episode_count": len(specs),
        "selected_states": selected_states,
        "specs": specs,
        "task_counts": dict(sorted(task_counts.items())),
        "diagnostic_role_semantics": {
            "strong_info": (
                "maximum-visibility operational candidate without a minimum-delta "
                "admission threshold; descriptive only"
            ),
            "matched_control": (
                "geometry-near low-delta candidate when available, otherwise the "
                "minimum-absolute-delta operational candidate"
            ),
            "blind": "minimum-visibility extreme/crossed-orbit candidate",
            "look_away": "minimum-visibility look-away candidate",
        },
        "fallback_control_count": fallback_control_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-plan", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, action="append", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshots, _, audit = load_scan_inputs(args.scan_plan, args.ledger)
    scan_plan = json.loads(args.scan_plan.read_text(encoding="utf-8"))
    result = build(
        snapshots=snapshots,
        scan_plan=scan_plan,
        catalog_path=args.catalog,
        input_audit=audit,
    )
    result.update(
        {
            "scan_plan": str(args.scan_plan.resolve()),
            "scan_plan_sha256": sha256(args.scan_plan),
            "catalog": str(args.catalog.resolve()),
            "catalog_sha256": sha256(args.catalog),
            "ledgers": [str(path.resolve()) for path in args.ledger],
            "ledger_sha256": {str(path.resolve()): sha256(path) for path in args.ledger},
        }
    )
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_state_count": result["selected_state_count"],
                "task_counts": result["task_counts"],
                "fallback_control_count": result["fallback_control_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
