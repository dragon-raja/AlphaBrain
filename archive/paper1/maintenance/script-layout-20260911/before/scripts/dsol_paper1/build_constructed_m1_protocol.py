#!/usr/bin/env python3
"""Build the constructed-M1 protocol from frozen, manually audited M0 states."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SELECTION_SCHEMA = "dsol_constructed_m0_candidate_selection_v1"
AUDIT_SCHEMA = "dsol_constructed_m0_manual_visual_audit_v1"
PROTOCOL_SCHEMA = "dsol_constructed_m1_frozen_closed_loop_protocol_v1"
EXPECTED_ROLES = {
    "canonical",
    "strong_info",
    "matched_control",
    "blind",
    "look_away",
    "all_camera_blackout",
}
OPERATIONAL_GROUPS = {"broad_heldout_32", "wide_extrapolation_24"}


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


def pose_index(catalog: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for value in catalog.values():
        if not isinstance(value, list):
            continue
        for pose in value:
            if not isinstance(pose, Mapping) or "pose_id" not in pose:
                continue
            pose_id = str(pose["pose_id"])
            if pose_id in result:
                raise ValueError(f"duplicate catalog pose: {pose_id}")
            result[pose_id] = pose
    return result


def _passed_audit_ids(
    audit: Mapping[str, Any], *, selection_sha256: str, minimum_groups: int
) -> set[str]:
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("status") != "PASS":
        raise ValueError("manual audit must use the expected schema and PASS")
    if audit.get("selection_sha256") != selection_sha256:
        raise ValueError("manual audit does not identify the frozen selection")
    records = audit.get("records", [])
    passed = {
        str(row["snapshot_group_id"])
        for row in records
        if row.get("status") == "PASS"
    }
    if len(passed) != len(records):
        raise ValueError("every manually audited record must PASS")
    if len(passed) < minimum_groups:
        raise ValueError(
            f"manual audit covers {len(passed)} groups; minimum is {minimum_groups}"
        )
    return passed


def build(
    *,
    selection: Mapping[str, Any],
    selection_sha256: str,
    audit: Mapping[str, Any],
    scan_plan: Mapping[str, Any],
    catalog: Mapping[str, Any],
    catalog_path: Path,
    minimum_audited_groups: int = 20,
) -> dict[str, Any]:
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("status") != "PASS":
        raise ValueError("constructed M0 automated selection must PASS")
    if selection.get("m1_admission_status") != "HOLD_MANUAL_AUDIT":
        raise ValueError("selection must be frozen at the manual-audit gate")
    passed_ids = _passed_audit_ids(
        audit,
        selection_sha256=selection_sha256,
        minimum_groups=minimum_audited_groups,
    )
    selected_by_id = {
        str(row["snapshot_group_id"]): row
        for row in selection["selected_snapshot_groups"]
    }
    unknown = sorted(passed_ids - set(selected_by_id))
    if unknown:
        raise ValueError(f"audit includes groups absent from selection: {unknown}")
    plan_by_id = {str(row["scan_id"]): row for row in scan_plan["records"]}
    poses = pose_index(catalog)
    specs = []
    selected_states = []
    for snapshot_group_id in sorted(passed_ids):
        selected = selected_by_id[snapshot_group_id]
        if selected.get("split") != "test":
            raise ValueError("M1 may only consume frozen test groups")
        conditions = selected["conditions"]
        if set(conditions) != EXPECTED_ROLES:
            raise ValueError(f"unexpected M0 condition roles: {sorted(conditions)}")
        strong = conditions["strong_info"]
        control = conditions["matched_control"]
        blind = conditions["blind"]
        if strong["source_group"] not in OPERATIONAL_GROUPS:
            raise ValueError("strong-info condition is not operational")
        scan_id = str(selected["scan_id"])
        source = plan_by_id.get(scan_id)
        if source is None:
            raise ValueError(f"selected scan absent from plan: {scan_id}")
        pose_ids = {
            role: str(conditions[role]["source_pose_id"])
            for role in ("strong_info", "matched_control", "blind")
        }
        missing_poses = sorted(set(pose_ids.values()) - set(poses))
        if missing_poses:
            raise ValueError(f"selected poses absent from catalog: {missing_poses}")
        visibility_selection = {
            role: {
                "pose_id": str(value["source_pose_id"]),
                "group": str(value["source_group"]),
                "visibility_score": float(value["visibility_score"]),
                "delta_visibility": float(value["delta_visibility"]),
            }
            for role, value in conditions.items()
        }
        common = {
            "pair_key": scan_id,
            "scan_id": scan_id,
            "task_id": str(selected["task_id"]),
            "diagnostic_role": str(source["diagnostic_role"]),
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
            "manual_audit_verified": True,
        }
        condition_specs = (
            ("canonical_both", None, "both"),
            ("strong_info_both", poses[pose_ids["strong_info"]], "both"),
            ("matched_control_both", poses[pose_ids["matched_control"]], "both"),
            ("blind_both", poses[pose_ids["blind"]], "both"),
            ("canonical_external_only", None, "external_only"),
            (
                "strong_info_external_only",
                poses[pose_ids["strong_info"]],
                "external_only",
            ),
            (
                "matched_control_external_only",
                poses[pose_ids["matched_control"]],
                "external_only",
            ),
            ("blind_external_only", poses[pose_ids["blind"]], "external_only"),
            ("canonical_wrist_only", None, "wrist_only"),
            ("all_camera_blackout", None, "all_blackout"),
        )
        for condition, pose, sensor_control in condition_specs:
            identity = f"{scan_id}::{condition}"
            specs.append(
                {
                    **common,
                    "condition": condition,
                    "pose": pose,
                    "sensor_control": sensor_control,
                    "episode_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                }
            )
        selected_states.append(
            {
                "snapshot_group_id": snapshot_group_id,
                "task_id": str(selected["task_id"]),
                "source_episode_id": str(selected["source_episode_id"]),
                "source_frame": int(selected["source_frame"]),
                "condition_pose_ids": pose_ids,
            }
        )
    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "PASS",
        "analysis_role": "constructed_m1_full_closed_loop_seed41_mechanism_gate",
        "candidate_selection_frozen_before_manual_audit": True,
        "manual_audit_verified": True,
        "test_threshold_retuning": False,
        "statistical_unit": "source HDF5 demonstration; frame states clustered within source",
        "selected_state_count": len(selected_states),
        "condition_count": 10,
        "episode_count": len(specs),
        "selected_states": selected_states,
        "specs": specs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--scan-plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-audited-groups", type=int, default=20)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    audit = json.loads(args.manual_audit.read_text(encoding="utf-8"))
    scan_plan = json.loads(args.scan_plan.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    result = build(
        selection=selection,
        selection_sha256=sha256(args.selection),
        audit=audit,
        scan_plan=scan_plan,
        catalog=catalog,
        catalog_path=args.catalog,
        minimum_audited_groups=args.minimum_audited_groups,
    )
    result.update(
        {
            "selection": str(args.selection.resolve()),
            "selection_sha256": sha256(args.selection),
            "manual_audit": str(args.manual_audit.resolve()),
            "manual_audit_sha256": sha256(args.manual_audit),
            "scan_plan": str(args.scan_plan.resolve()),
            "scan_plan_sha256": sha256(args.scan_plan),
            "catalog": str(args.catalog.resolve()),
            "catalog_sha256": sha256(args.catalog),
        }
    )
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_state_count": result["selected_state_count"],
                "episode_count": result["episode_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
