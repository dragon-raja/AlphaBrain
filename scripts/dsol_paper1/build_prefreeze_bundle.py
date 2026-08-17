#!/usr/bin/env python3
"""Materialize a fail-closed Paper 1 prefreeze bundle without policy execution."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


TASK_FAMILIES = {
    "CloseBlenderLid": "object_fixture_assembly",
    "CloseFridge": "fixture_articulation",
    "CoffeeSetupMug": "pick_place_appliance",
    "NavigateKitchen": "mobile_navigation",
    "OpenCabinet": "fixture_articulation",
    "OpenDrawer": "fixture_articulation",
    "PickPlaceCounterToCabinet": "pick_place_container",
    "PickPlaceSinkToCounter": "pick_place_receptacle",
    "TurnOffStove": "appliance_state_change",
    "TurnOnMicrowave": "appliance_state_change",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--camera-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def assert_debug_output(path: Path) -> None:
    resolved = path.resolve()
    if "formal" in resolved.parts:
        raise ValueError(f"prefreeze output must not use a formal root: {resolved}")
    if not str(resolved).startswith("/workspace/ai2r/debug/"):
        raise ValueError(f"prefreeze output must be under /workspace/ai2r/debug: {resolved}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path}")
    return value


def enumerate_task_templates(
    snapshot_bank: Path, expected_tasks: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifests = sorted(snapshot_bank.glob("*/snapshots/*/seed_*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"snapshot bank has no manifests: {snapshot_bank}")
    grouped: dict[str, dict[str, Any]] = {}
    identities: set[tuple[str, str, int]] = set()
    for path in manifests:
        manifest = load_json(path)
        metadata = manifest.get("metadata", {})
        task = str(metadata.get("task", ""))
        partition = path.relative_to(snapshot_bank).parts[0]
        scene_seed = int(metadata.get("scene_seed"))
        identity = (partition, task, scene_seed)
        if identity in identities:
            raise RuntimeError(f"duplicate snapshot group: {identity}")
        identities.add(identity)
        record = grouped.setdefault(
            task,
            {
                "task": task,
                "task_family": TASK_FAMILIES.get(task),
                "prompts": set(),
                "partitions": set(),
                "scene_seeds": set(),
                "snapshot_groups": 0,
            },
        )
        record["prompts"].add(str(metadata.get("prompt", "")))
        record["partitions"].add(partition)
        record["scene_seeds"].add(scene_seed)
        record["snapshot_groups"] += 1

    actual_tasks = set(grouped)
    if actual_tasks != expected_tasks:
        raise RuntimeError(
            f"task population mismatch: missing={sorted(expected_tasks - actual_tasks)}, "
            f"extra={sorted(actual_tasks - expected_tasks)}"
        )
    templates = []
    for task, record in sorted(grouped.items()):
        if record["task_family"] is None:
            raise RuntimeError(f"task family is not frozen for {task}")
        templates.append(
            {
                "schema": "dsol_task_template_v1",
                "task": task,
                "task_family": record["task_family"],
                "prompts": sorted(record["prompts"]),
                "snapshot_partitions": sorted(record["partitions"]),
                "scene_seeds": sorted(record["scene_seeds"]),
                "snapshot_group_count": record["snapshot_groups"],
                "full_observation_contract": "preregistration.observation_contract",
                "expert_decision_set_status": "NOT_MATERIALIZED",
                "recovery_action_set_status": "NOT_MATERIALIZED",
                "release_status": "HOLD",
            }
        )
    summary = {
        "manifest_count": len(manifests),
        "unique_snapshot_group_count": len(identities),
        "task_count": len(templates),
        "partition_counts": {
            partition: sum(1 for value in identities if value[0] == partition)
            for partition in sorted({value[0] for value in identities})
        },
    }
    return templates, summary


def input_record(label: str, path: Path) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    args = parse_args()
    assert_debug_output(args.output)
    prereg = load_json(args.preregistration)
    if prereg.get("schema") != "dsol_paper1_preregistration_v1":
        raise ValueError("unexpected preregistration schema")
    if prereg.get("status") != "PREFREEZE_NOT_RELEASED":
        raise ValueError("builder accepts only the fail-closed prefreeze status")
    camera = load_json(args.camera_catalog)
    if camera.get("format") != "robocasa_camera_pose_catalog_v1":
        raise ValueError("unexpected camera catalog schema")

    snapshot_bank = Path(prereg["roots"]["snapshot_bank"])
    p0_gate_path = Path(prereg["roots"]["p0_gate"])
    checkpoint_path = Path(prereg["roots"]["checkpoint_receipts"])
    p0_gate = load_json(p0_gate_path)
    checkpoints = load_json(checkpoint_path)
    if "ASSETS_CLOSED" not in str(p0_gate.get("overall_status", "")):
        raise RuntimeError("P0 asset gate is not closed")
    if checkpoints.get("status") != "PASS":
        raise RuntimeError("checkpoint load matrix is not PASS")

    templates, bank_summary = enumerate_task_templates(
        snapshot_bank, set(prereg["population"]["tasks"])
    )
    args.output.mkdir(parents=True, exist_ok=True)
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    task_payload = {
        "schema": "dsol_task_template_collection_v1",
        "created_at": created_at,
        "status": "PREFREEZE_NOT_RELEASED",
        "snapshot_bank": str(snapshot_bank.resolve()),
        "snapshot_bank_summary": bank_summary,
        "tasks": templates,
    }
    task_path = args.output / "task_templates_prefreeze.json"
    atomic_json(task_path, task_payload)

    empty_instrument = {
        "schema": "dsol_relation_instrument_v1",
        "created_at": created_at,
        "status": "EMPTY_NOT_RELEASED",
        "relation_records": [],
        "required_relation_types": ["N", "E_D", "MATCHED_CONTROL"],
        "reason": "B4 expert decision sets and B5 relation/temporal audits are not materialized",
    }
    instrument_path = args.output / "relation_instrument_prefreeze.json"
    atomic_json(instrument_path, empty_instrument)

    provenance = {
        "schema": "dsol_prefreeze_provenance_v1",
        "created_at": created_at,
        "inputs": [
            input_record("preregistration", args.preregistration),
            input_record("camera_catalog", args.camera_catalog),
            input_record("p0_gate", p0_gate_path),
            input_record("checkpoint_load_matrix", checkpoint_path),
        ],
        "snapshot_bank": {
            "path": str(snapshot_bank.resolve()),
            **bank_summary,
        },
    }
    provenance_path = args.output / "provenance.json"
    atomic_json(provenance_path, provenance)

    release = {
        "schema": "dsol_release_receipt_v1",
        "created_at": created_at,
        "release": "P1",
        "status": "HOLD",
        "approved": ["static protocol development", "debug-only instrument smoke"],
        "forbidden": ["formal sample generation", "policy training", "formal rollout"],
        "gates": {
            "B1": "PARTIAL",
            "B2": "PASS",
            "B3": "PARTIAL",
            "B4": "HOLD_NOT_MATERIALIZED",
            "B5": "HOLD_NOT_MATERIALIZED",
            "B6": "HOLD_NOT_IMPLEMENTED",
            "B7": "HOLD_POWER_NOT_RUN",
        },
        "artifacts": {
            "task_templates": str(task_path),
            "relation_instrument": str(instrument_path),
            "provenance": str(provenance_path),
        },
    }
    release_path = args.output / "release_candidate.json"
    atomic_json(release_path, release)

    outputs = [task_path, instrument_path, provenance_path, release_path]
    checksums = "".join(f"{sha256(path)}  {path.name}\n" for path in outputs)
    checksum_path = args.output / "checksums.sha256"
    temporary = checksum_path.with_suffix(".sha256.tmp")
    temporary.write_text(checksums)
    temporary.replace(checksum_path)
    print(release_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
