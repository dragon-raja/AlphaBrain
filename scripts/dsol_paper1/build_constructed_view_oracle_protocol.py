#!/usr/bin/env python3
"""Build a dense-view discovery protocol for constructed LIBERO states."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_rows(patterns: Sequence[str]) -> list[dict[str, Any]]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError("no scan JSONL matched")
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def select_stages(
    rows: Sequence[Mapping[str, Any]], stage_targets: Sequence[float]
) -> list[Mapping[str, Any]]:
    remaining = list(rows)
    selected = []
    for target in stage_targets:
        if not remaining:
            break
        choice = min(
            remaining,
            key=lambda row: (
                round(abs(float(row["stage_fraction"]) - target), 12),
                str(row["scan_id"]),
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return selected


def group_for_pose(pose_id: str) -> str:
    if pose_id == "canonical":
        return "canonical"
    if pose_id.startswith("broad_train_"):
        return "broad_training_64"
    if pose_id.startswith("broad_heldout_"):
        return "broad_heldout_32"
    raise ValueError(f"unsupported pose ID: {pose_id}")


def build(
    rows: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    *,
    catalog_path: Path,
    split: str,
    stage_targets: Sequence[float],
) -> dict[str, Any]:
    eligible = []
    for row in rows:
        if row.get("status") != "PASS" or row.get("split") != split:
            continue
        scan_path = Path(row["output_dir"]) / "scan.json"
        scan = json.loads(scan_path.read_text(encoding="utf-8"))
        if bool(scan.get("initial_task_success")):
            continue
        eligible.append((row, scan))

    by_source: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for row, scan in eligible:
        by_source[str(row["episode_id"])].append((row, scan))

    selected = []
    for source in sorted(by_source):
        source_rows = by_source[source]
        chosen_rows = select_stages([item[0] for item in source_rows], stage_targets)
        scans = {str(item[0]["scan_id"]): item[1] for item in source_rows}
        selected.extend((row, scans[str(row["scan_id"])]) for row in chosen_rows)

    poses = [catalog["canonical"][0], *catalog["broad_training_64"], *catalog["broad_heldout_32"]]
    if len(poses) != 97 or len({pose["pose_id"] for pose in poses}) != 97:
        raise ValueError("catalog must provide exactly 97 unique operational poses")

    specs = []
    selected_states = []
    for row, scan in selected:
        pair_key = str(row["scan_id"])
        common = {
            "pair_key": pair_key,
            "scan_id": pair_key,
            "task_id": str(row["task_id"]),
            "diagnostic_role": "constructed_dense_view_value_discovery",
            "suite": str(row["suite"]),
            "hdf5": str(row["hdf5"]),
            "episode_id_source": str(row["episode_id"]),
            "demo_name": str(row["demo_name"]),
            "demo_index": int(row["demo_index"]),
            "split": split,
            "source_state_index": int(row["frame"]),
            "stage_fraction": float(row["stage_fraction"]),
            "scene_construction": scan["scene_construction"],
            "sensor_control": "both",
            "catalog": str(catalog_path.resolve()),
            "manual_audit_verified": False,
        }
        for pose in poses:
            pose_id = str(pose["pose_id"])
            identity = f"{pair_key}::{pose_id}"
            specs.append(
                {
                    **common,
                    "condition": f"candidate__{pose_id}",
                    "selected_candidate_id": pose_id,
                    "pose": None if pose_id == "canonical" else pose,
                    "selection_metadata": {"catalog_group": group_for_pose(pose_id)},
                    "episode_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                }
            )
        selected_states.append(
            {
                "pair_key": pair_key,
                "task_id": str(row["task_id"]),
                "source_episode_id": str(row["episode_id"]),
                "source_state_index": int(row["frame"]),
                "stage_fraction": float(row["stage_fraction"]),
                "construction_sha256": str(scan["scene_construction"]["sha256"]),
            }
        )

    return {
        "schema": "dsol_constructed_dense_view_oracle_protocol_v1",
        "status": "PASS" if specs else "FAIL",
        "analysis_role": "discovery_only_behavior_oracle_pilot",
        "confirmatory_test_eligible": False,
        "selection_uses_policy_outcomes": False,
        "split": split,
        "stage_targets": list(stage_targets),
        "statistical_unit": "source HDF5 demonstration",
        "catalog": str(catalog_path.resolve()),
        "selected_state_count": len(selected_states),
        "source_episode_count": len({row["source_episode_id"] for row in selected_states}),
        "candidate_count": len(poses),
        "episode_count": len(specs),
        "selected_states": selected_states,
        "specs": specs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--stage-targets", default="0.20,0.55")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    result = build(
        load_rows(args.inputs),
        catalog,
        catalog_path=args.catalog,
        split=args.split,
        stage_targets=[float(value) for value in args.stage_targets.split(",")],
    )
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "states": result["selected_state_count"],
                "sources": result["source_episode_count"],
                "episodes": result["episode_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
