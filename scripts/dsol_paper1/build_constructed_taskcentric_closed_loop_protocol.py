#!/usr/bin/env python3
"""Build a test-only closed-loop protocol from validation-frozen view pairs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "dsol_constructed_taskcentric_closed_loop_protocol_v1"
CONDITIONS = (
    ("canonical_both", "canonical", "both"),
    ("strong_info_both", "strong_info", "both"),
    ("matched_control_both", "matched_control", "both"),
    ("canonical_external_only", "canonical", "external_only"),
    ("strong_info_external_only", "strong_info", "external_only"),
    ("matched_control_external_only", "matched_control", "external_only"),
    ("canonical_wrist_only", "canonical", "wrist_only"),
    ("all_camera_blackout", "canonical", "all_blackout"),
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(patterns: Iterable[str]) -> list[dict[str, Any]]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def constructed_records(scan: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    result = {}
    for record in scan["records"]:
        if record.get("group") != "constructed_task_orbit":
            continue
        pose = record["pose"]
        result[(str(pose["pair_id"]), str(pose["pair_member"]))] = record
    return result


def canonical_record(scan: Mapping[str, Any]) -> Mapping[str, Any]:
    records = [
        row
        for row in scan["records"]
        if row.get("pose_id") == "canonical" and row.get("group") == "canonical"
    ]
    if len(records) != 1:
        raise ValueError("scan must contain exactly one canonical record")
    return records[0]


def fixed_stage_selection(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    targets: Sequence[float],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    remaining = list(rows)
    selected = []
    for target in targets:
        if not remaining:
            break
        choice = min(
            remaining,
            key=lambda item: (
                abs(float(item[0]["stage_fraction"]) - float(target)),
                str(item[0]["scan_id"]),
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return selected


def build(
    rows: Sequence[Mapping[str, Any]],
    frozen: Mapping[str, Any],
    *,
    catalog: Path,
    stage_targets: Sequence[float],
    minimum_strong_delta: float,
    maximum_control_abs_delta: float,
) -> dict[str, Any]:
    if frozen.get("schema") != "dsol_constructed_task_pair_freeze_v1":
        raise ValueError("unexpected frozen-pair schema")
    if frozen.get("status") != "PASS" or frozen.get("selection_split") != "validation_only":
        raise ValueError("validation pair freeze must PASS")
    eligible = []
    exclusions = defaultdict(int)
    for row in rows:
        if row.get("split") != "test":
            exclusions["non_test"] += 1
            continue
        if row.get("status") != "PASS":
            exclusions["scan_not_pass"] += 1
            continue
        scan = json.loads((Path(row["output_dir"]) / "scan.json").read_text())
        if bool(scan.get("initial_task_success")):
            exclusions["initial_task_success"] += 1
            continue
        if float(row["stage_fraction"]) > 0.70:
            exclusions["stage_fraction_above_070"] += 1
            continue
        eligible.append((row, scan))

    by_episode: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for item in eligible:
        row = item[0]
        by_episode[(str(row["task_id"]), str(row["episode_id"]))].append(item)
    selected = []
    for key in sorted(by_episode):
        selected.extend(fixed_stage_selection(by_episode[key], stage_targets))

    specs = []
    state_records = []
    compliance = defaultdict(int)
    for row, scan in selected:
        task_id = str(row["task_id"])
        rule = frozen["tasks"][task_id]["selected"]
        candidates = constructed_records(scan)
        strong = candidates[(str(rule["pair_id"]), str(rule["strong_member"]))]
        control = candidates[(str(rule["pair_id"]), str(rule["control_member"]))]
        canonical = canonical_record(scan)
        strong_pass = float(strong["delta_visibility"]) >= minimum_strong_delta
        control_pass = abs(float(control["delta_visibility"])) <= maximum_control_abs_delta
        compliance["strong_pass"] += int(strong_pass)
        compliance["control_pass"] += int(control_pass)
        compliance["both_pass"] += int(strong_pass and control_pass)
        roles = {
            "canonical": canonical,
            "strong_info": strong,
            "matched_control": control,
        }
        pair_key = str(row["scan_id"])
        visibility_selection = {
            role: {
                "pose_id": str(record["pose_id"]),
                "visibility_score": float(record["visibility_score"]),
                "delta_visibility": float(record["delta_visibility"]),
                "per_camera_scores": record.get("per_camera_scores"),
            }
            for role, record in roles.items()
        }
        common = {
            "pair_key": pair_key,
            "scan_id": pair_key,
            "task_id": task_id,
            "diagnostic_role": str(row["diagnostic_role"]),
            "suite": str(row["suite"]),
            "hdf5": str(row["hdf5"]),
            "episode_id_source": str(row["episode_id"]),
            "demo_name": str(row["demo_name"]),
            "demo_index": int(row["demo_index"]),
            "split": "test",
            "source_state_index": int(row["frame"]),
            "stage_fraction": float(row["stage_fraction"]),
            "scene_construction": scan["scene_construction"],
            "visibility_selection": visibility_selection,
            "validation_frozen_pair": {
                "pair_id": str(rule["pair_id"]),
                "strong_member": str(rule["strong_member"]),
                "control_member": str(rule["control_member"]),
            },
        }
        for condition, role, sensor_control in CONDITIONS:
            pose = roles[role].get("pose")
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
        state_records.append(
            {
                "pair_key": pair_key,
                "task_id": task_id,
                "source_episode_id": str(row["episode_id"]),
                "stage_fraction": float(row["stage_fraction"]),
                "strong_delta_visibility": float(strong["delta_visibility"]),
                "control_delta_visibility": float(control["delta_visibility"]),
                "strong_visibility_pass": strong_pass,
                "control_visibility_pass": control_pass,
                "construction_sha256": str(scan["scene_construction"]["sha256"]),
            }
        )
    state_count = len(state_records)
    return {
        "schema": SCHEMA,
        "status": "PASS" if state_count else "FAIL",
        "analysis_role": "expanded_A_constructed_strong_information_pilot",
        "selection_policy": "fixed_stages_per_test_source_episode",
        "policy_outcomes_used_for_selection": False,
        "validation_and_test_episodes_disjoint": True,
        "stage_targets": list(stage_targets),
        "catalog": str(catalog.resolve()),
        "catalog_sha256": sha256(catalog),
        "selected_state_count": state_count,
        "condition_count": len(CONDITIONS),
        "episode_count": len(specs),
        "test_visibility_compliance": {
            **dict(compliance),
            "state_count": state_count,
            "minimum_strong_delta": minimum_strong_delta,
            "maximum_control_abs_delta": maximum_control_abs_delta,
        },
        "exclusions": dict(sorted(exclusions.items())),
        "selected_states": state_records,
        "specs": specs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--frozen-pairs", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-targets", default="0.15,0.45")
    parser.add_argument("--minimum-strong-delta", type=float, default=0.005)
    parser.add_argument("--maximum-control-abs-delta", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_targets = [float(value) for value in args.stage_targets.split(",")]
    result = build(
        load_rows(args.inputs),
        json.loads(args.frozen_pairs.read_text()),
        catalog=args.catalog,
        stage_targets=stage_targets,
        minimum_strong_delta=args.minimum_strong_delta,
        maximum_control_abs_delta=args.maximum_control_abs_delta,
    )
    result["frozen_pairs"] = str(args.frozen_pairs.resolve())
    result["frozen_pairs_sha256"] = sha256(args.frozen_pairs)
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "states": result["selected_state_count"],
                "episodes": result["episode_count"],
                "compliance": result["test_visibility_compliance"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
