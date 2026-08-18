#!/usr/bin/env python3
"""Freeze a visibility-defined, physically matched LIBERO M1 quick gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            if source["group"] == "sensor_controls":
                continue
            row = dict(source)
            row["delta_visibility"] = float(source["delta_visibility"])
            row["camera_translation_m"] = float(source["camera_translation_m"])
            row["camera_rotation_geodesic_deg"] = float(
                source["camera_rotation_geodesic_deg"]
            )
            grouped[source["scan_id"]].append(row)
    return dict(grouped)


def select_strong(
    rows: Iterable[Mapping[str, Any]], rules: Mapping[str, Any]
) -> Mapping[str, Any]:
    allowed = set(rules["strong_information_groups"])
    candidates = [row for row in rows if row["group"] in allowed]
    if not candidates:
        raise ValueError("state has no allowed strong-information candidate")
    return max(
        candidates,
        key=lambda row: (float(row["delta_visibility"]), str(row["pose_id"])),
    )


def select_matched_control(
    rows: Iterable[Mapping[str, Any]],
    strong: Mapping[str, Any],
    rules: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    visibility_limit = float(rules["matched_control_visibility_abs_max"])
    translation_tolerance = float(rules["matched_control_translation_tolerance_m"])
    rotation_tolerance = float(rules["matched_control_rotation_tolerance_deg"])
    allowed = set(rules["control_groups"])
    candidates = []
    for row in rows:
        if row["group"] not in allowed or row["pose_id"] == strong["pose_id"]:
            continue
        translation_gap = abs(
            float(row["camera_translation_m"])
            - float(strong["camera_translation_m"])
        )
        rotation_gap = abs(
            float(row["camera_rotation_geodesic_deg"])
            - float(strong["camera_rotation_geodesic_deg"])
        )
        if (
            abs(float(row["delta_visibility"])) <= visibility_limit
            and translation_gap <= translation_tolerance
            and rotation_gap <= rotation_tolerance
        ):
            candidates.append((row, translation_gap, rotation_gap))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item[1] / translation_tolerance + item[2] / rotation_tolerance,
            abs(float(item[0]["delta_visibility"])),
            str(item[0]["pose_id"]),
        ),
    )[0]


def select_blind(
    rows: Iterable[Mapping[str, Any]], rules: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    allowed = set(rules["blind_groups"])
    candidates = [row for row in rows if row["group"] in allowed]
    if not candidates:
        return None
    chosen = min(
        candidates,
        key=lambda row: (float(row["delta_visibility"]), str(row["pose_id"])),
    )
    return (
        chosen
        if float(chosen["delta_visibility"]) <= float(rules["blind_delta_max"])
        else None
    )


def pose_index(catalog: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    poses = {}
    for value in catalog.values():
        if not isinstance(value, list):
            continue
        for pose in value:
            if isinstance(pose, dict) and "pose_id" in pose:
                pose_id = str(pose["pose_id"])
                if pose_id in poses:
                    raise ValueError(f"duplicate catalog pose_id: {pose_id}")
                poses[pose_id] = pose
    return poses


def task_gate(
    grouped: Mapping[str, list[Mapping[str, Any]]],
    rules: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    tasks: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any] | None]]] = (
        defaultdict(list)
    )
    for rows in grouped.values():
        if rows[0]["split"] != "val":
            continue
        strong = select_strong(rows, rules)
        control = select_matched_control(rows, strong, rules)
        tasks[str(rows[0]["task_id"])].append((strong, control))
    result = {}
    for task_id, selections in sorted(tasks.items()):
        best_deltas = [float(strong["delta_visibility"]) for strong, _ in selections]
        information_fraction = sum(
            delta >= float(rules["strong_information_delta_min"])
            for delta in best_deltas
        ) / len(selections)
        matched_fraction = sum(control is not None for _, control in selections) / len(
            selections
        )
        median = float(statistics.median(best_deltas))
        eligible = (
            median >= float(rules["minimum_val_median_best_delta"])
            and information_fraction >= float(rules["minimum_val_information_fraction"])
            and matched_fraction >= float(rules["minimum_val_matched_fraction"])
        )
        result[task_id] = {
            "eligible": eligible,
            "val_state_count": len(selections),
            "val_median_best_delta": median,
            "val_information_fraction": information_fraction,
            "val_matched_fraction": matched_fraction,
        }
    return result


def build(
    *,
    rules: Mapping[str, Any],
    scan_plan: Mapping[str, Any],
    catalog: Mapping[str, Any],
    grouped: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    plan_by_id = {str(row["scan_id"]): row for row in scan_plan["records"]}
    poses = pose_index(catalog)
    gates = task_gate(grouped, rules)
    specs = []
    selected_states = []
    excluded_states = []
    for scan_id, rows in sorted(grouped.items()):
        if rows[0]["split"] != "test":
            continue
        task_id = str(rows[0]["task_id"])
        if not gates[task_id]["eligible"]:
            excluded_states.append({"scan_id": scan_id, "reason": "task_failed_val_gate"})
            continue
        strong = select_strong(rows, rules)
        control = select_matched_control(rows, strong, rules)
        blind = select_blind(rows, rules)
        reasons = []
        if float(strong["delta_visibility"]) < float(
            rules["strong_information_delta_min"]
        ):
            reasons.append("strong_information_below_threshold")
        if control is None:
            reasons.append("no_strict_matched_control")
        if blind is None:
            reasons.append("no_strict_blind_view")
        if reasons:
            excluded_states.append({"scan_id": scan_id, "reason": ",".join(reasons)})
            continue
        source = plan_by_id[scan_id]
        state_record = {
            "scan_id": scan_id,
            "task_id": task_id,
            "strong_pose_id": strong["pose_id"],
            "strong_delta_visibility": strong["delta_visibility"],
            "matched_control_pose_id": control["pose_id"],
            "matched_control_delta_visibility": control["delta_visibility"],
            "matched_translation_gap_m": abs(
                float(control["camera_translation_m"])
                - float(strong["camera_translation_m"])
            ),
            "matched_rotation_gap_deg": abs(
                float(control["camera_rotation_geodesic_deg"])
                - float(strong["camera_rotation_geodesic_deg"])
            ),
            "blind_pose_id": blind["pose_id"],
            "blind_delta_visibility": blind["delta_visibility"],
        }
        selected_states.append(state_record)
        pair_key = scan_id
        common = {
            "pair_key": pair_key,
            "scan_id": scan_id,
            "task_id": task_id,
            "diagnostic_role": source["diagnostic_role"],
            "suite": source["suite"],
            "hdf5": source["hdf5"],
            "episode_id_source": source["episode_id"],
            "demo_name": source["demo_name"],
            "demo_index": source["demo_index"],
            "split": "test",
            "source_state_index": source["frame"],
            "stage_fraction": source["stage_fraction"],
            "visibility_selection": state_record,
        }
        conditions = (
            ("canonical_both", None, "both"),
            ("strong_info_both", poses[str(strong["pose_id"])], "both"),
            ("matched_control_both", poses[str(control["pose_id"])], "both"),
            ("blind_both", poses[str(blind["pose_id"])], "both"),
            ("canonical_external_only", None, "external_only"),
            (
                "strong_info_external_only",
                poses[str(strong["pose_id"])],
                "external_only",
            ),
            (
                "matched_control_external_only",
                poses[str(control["pose_id"])],
                "external_only",
            ),
            ("blind_external_only", poses[str(blind["pose_id"])], "external_only"),
            ("canonical_wrist_only", None, "wrist_only"),
            ("all_camera_blackout", None, "all_blackout"),
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
    return {
        "schema": "dsol_libero_m1_visibility_closed_loop_protocol_v1",
        "analysis_role": rules["analysis_role"],
        "test_states_previously_inspected": bool(rules["test_states_previously_inspected"]),
        "task_gate": gates,
        "selected_state_count": len(selected_states),
        "excluded_test_state_count": len(excluded_states),
        "condition_count": 10,
        "episode_count": len(specs),
        "selected_states": selected_states,
        "excluded_test_states": excluded_states,
        "specs": specs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--scan-plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--candidate-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rules = json.loads(args.rules.read_text())
    scan_plan = json.loads(args.scan_plan.read_text())
    catalog = json.loads(args.catalog.read_text())
    grouped = load_candidate_rows(args.candidate_records)
    payload = build(rules=rules, scan_plan=scan_plan, catalog=catalog, grouped=grouped)
    payload.update(
        {
            "rules": str(args.rules.resolve()),
            "rules_sha256": sha256(args.rules),
            "source_scan_plan": str(args.scan_plan.resolve()),
            "source_scan_plan_sha256": sha256(args.scan_plan),
            "catalog": str(args.catalog.resolve()),
            "catalog_sha256": sha256(args.catalog),
            "candidate_records": str(args.candidate_records.resolve()),
            "candidate_records_sha256": sha256(args.candidate_records),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "selected_state_count": payload["selected_state_count"],
                "condition_count": payload["condition_count"],
                "episode_count": payload["episode_count"],
                "eligible_tasks": [
                    task for task, gate in payload["task_gate"].items() if gate["eligible"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
