#!/usr/bin/env python3
"""Re-render frozen constructed-M0 conditions for balanced manual review."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


POSE_GROUPS = (
    "broad_heldout_32",
    "wide_extrapolation_24",
    "diagnostic_extreme_orbit",
    "diagnostic_crossed_orbit",
    "diagnostic_look_away",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _spread(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("source_frame", 0)),
            str(row["snapshot_group_id"]),
        ),
    )


def select_balanced_audit_groups(
    selected: Sequence[Mapping[str, Any]], *, per_task: int, target_total: int
) -> list[Mapping[str, Any]]:
    if per_task <= 0:
        raise ValueError("per_task must be positive")
    if target_total <= 0:
        raise ValueError("target_total must be positive")
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_task[str(row["task_id"])].append(row)
    chosen = []
    ordered_by_task: dict[str, list[Mapping[str, Any]]] = {}
    for task_id, task_rows in sorted(by_task.items()):
        by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in task_rows:
            by_episode[str(row["source_episode_id"])].append(row)
        episode_rows = {
            episode: _spread(rows) for episode, rows in sorted(by_episode.items())
        }
        task_chosen = []
        offset = 0
        while len(task_chosen) < min(per_task, len(task_rows)):
            progressed = False
            for episode in sorted(episode_rows):
                rows = episode_rows[episode]
                if offset < len(rows):
                    task_chosen.append(rows[offset])
                    progressed = True
                    if len(task_chosen) >= per_task:
                        break
            if not progressed:
                break
            offset += 1
        ordered_by_task[task_id] = task_chosen + [
            row
            for row in _spread(task_rows)
            if row["snapshot_group_id"]
            not in {value["snapshot_group_id"] for value in task_chosen}
        ]
        chosen.extend(task_chosen)

    cursors = {
        task_id: min(per_task, len(rows))
        for task_id, rows in ordered_by_task.items()
    }
    while len(chosen) < target_total:
        progressed = False
        for task_id in sorted(ordered_by_task):
            rows = ordered_by_task[task_id]
            cursor = cursors[task_id]
            if cursor < len(rows):
                chosen.append(rows[cursor])
                cursors[task_id] += 1
                progressed = True
                if len(chosen) >= target_total:
                    break
        if not progressed:
            break
    return chosen


def selected_pose_ids(row: Mapping[str, Any]) -> list[str]:
    pose_ids = []
    for role in ("strong_info", "matched_control", "blind", "look_away"):
        condition = row["conditions"][role]
        pose_id = str(condition["source_pose_id"])
        if pose_id not in pose_ids:
            pose_ids.append(pose_id)
    return pose_ids


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render selected M0 conditions into review-complete montages."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--scan-plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=7)
    parser.add_argument("--target-total", type=int, default=21)
    parser.add_argument("--render-gpu", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("schema") != "dsol_constructed_m0_candidate_selection_v1":
        raise ValueError("unexpected selection schema")
    if selection.get("status") != "PASS":
        raise ValueError("candidate selection must PASS before audit re-rendering")
    plan = json.loads(args.scan_plan.read_text(encoding="utf-8"))
    plan_by_id = {str(row["scan_id"]): row for row in plan["records"]}
    chosen = select_balanced_audit_groups(
        selection["selected_snapshot_groups"],
        per_task=args.per_task,
        target_total=args.target_total,
    )
    if not chosen:
        raise ValueError("selection contains no snapshot groups")
    if len(chosen) < args.target_total:
        raise ValueError(
            f"selection provides only {len(chosen)} audit groups; "
            f"target is {args.target_total}"
        )

    scanner = Path(__file__).resolve().with_name("scan_libero_hdf5_views.py")
    records = []
    for ordinal, selected in enumerate(chosen, start=1):
        scan_id = str(selected["scan_id"])
        if scan_id not in plan_by_id:
            raise KeyError(f"selected scan is absent from source plan: {scan_id}")
        source = plan_by_id[scan_id]
        safe_id = hashlib.sha256(scan_id.encode()).hexdigest()[:16]
        output_dir = args.output_root / "states" / safe_id
        pose_ids = selected_pose_ids(selected)
        scan_path = output_dir / "scan.json"
        reusable = False
        if scan_path.is_file():
            existing = json.loads(scan_path.read_text(encoding="utf-8"))
            reusable = (
                existing.get("status") == "PASS"
                and existing.get("requested_pose_ids") == pose_ids
                and int(existing.get("invalid_records", 1)) == 0
            )
        if not reusable:
            command = [
                sys.executable,
                str(scanner),
                "--hdf5",
                str(source["hdf5"]),
                "--runtime",
                str(args.runtime),
                "--catalog",
                str(args.catalog),
                "--config-root",
                str(args.config_root),
                "--output-dir",
                str(output_dir),
                "--groups",
                ",".join(POSE_GROUPS),
                "--pose-ids",
                ",".join(pose_ids),
                "--demo-index",
                str(source["demo_index"]),
                "--frame-index",
                str(source["frame"]),
                "--render-gpu",
                str(args.render_gpu),
                "--montage-top-k",
                "16",
            ]
            subprocess.run(command, check=True)
        montage = output_dir / "visibility_extremes.png"
        if not montage.is_file():
            raise FileNotFoundError(f"scanner did not produce montage: {montage}")
        records.append(
            {
                "ordinal": ordinal,
                "snapshot_group_id": selected["snapshot_group_id"],
                "task_id": selected["task_id"],
                "source_episode_id": selected["source_episode_id"],
                "source_frame": selected["source_frame"],
                "pose_ids": pose_ids,
                "condition_roles": {
                    role: value["source_pose_id"]
                    for role, value in selected["conditions"].items()
                },
                "scan_path": str(scan_path.resolve()),
                "montage_path": str(montage.resolve()),
                "montage_sha256": _sha256(montage),
                "manual_visual_audit": "PENDING",
            }
        )
        print(
            json.dumps(
                {"ordinal": ordinal, "total": len(chosen), "scan_id": scan_id},
                sort_keys=True,
            ),
            flush=True,
        )

    task_counts = defaultdict(int)
    episode_counts: dict[str, set[str]] = defaultdict(set)
    for record in records:
        task_counts[record["task_id"]] += 1
        episode_counts[record["task_id"]].add(record["source_episode_id"])
    manifest = {
        "schema": "dsol_constructed_m0_manual_audit_render_v1",
        "status": "RENDERED_PENDING_MANUAL_AUDIT",
        "manual_audit_status": "PENDING",
        "automatically_promoted": False,
        "selection": str(args.selection.resolve()),
        "selection_sha256": _sha256(args.selection),
        "scan_plan": str(args.scan_plan.resolve()),
        "scan_plan_sha256": _sha256(args.scan_plan),
        "catalog": str(args.catalog.resolve()),
        "catalog_sha256": _sha256(args.catalog),
        "requested_per_task": args.per_task,
        "requested_target_total": args.target_total,
        "rendered_snapshot_group_count": len(records),
        "task_counts": dict(sorted(task_counts.items())),
        "source_episode_counts": {
            task: len(episodes) for task, episodes in sorted(episode_counts.items())
        },
        "records": records,
    }
    _atomic_json(args.output_root / "audit_render_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "rendered_snapshot_group_count", "task_counts", "source_episode_counts")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
