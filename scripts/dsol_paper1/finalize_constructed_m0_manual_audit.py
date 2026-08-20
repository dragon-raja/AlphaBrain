#!/usr/bin/env python3
"""Finalize a transparent visual-review ledger for constructed M0."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CHECKS = (
    "render_valid",
    "strong_info_gain_visible",
    "matched_control_comparable_geometry",
    "blind_reduces_task_visibility",
    "look_away_reduces_task_visibility",
    "no_visual_leakage_or_corruption",
)


def sha256(path: Path) -> str:
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


def finalize(
    *,
    render_manifest: Mapping[str, Any],
    selection_sha256: str,
    decisions: Mapping[str, Any],
    minimum_groups: int = 20,
) -> dict[str, Any]:
    if (
        render_manifest.get("schema")
        != "dsol_constructed_m0_manual_audit_render_v1"
        or render_manifest.get("status") != "RENDERED_PENDING_MANUAL_AUDIT"
    ):
        raise ValueError("unexpected audit render manifest")
    if render_manifest.get("selection_sha256") != selection_sha256:
        raise ValueError("render manifest and selection identity differ")
    if decisions.get("schema") != "dsol_constructed_m0_manual_visual_decisions_v1":
        raise ValueError("unexpected visual decisions schema")
    rendered = {
        str(row["snapshot_group_id"]): row
        for row in render_manifest.get("records", [])
    }
    reviewed = {
        str(row["snapshot_group_id"]): row for row in decisions.get("records", [])
    }
    if set(rendered) != set(reviewed):
        raise ValueError("visual decisions must cover exactly the rendered groups")
    if len(rendered) < minimum_groups:
        raise ValueError(
            f"only {len(rendered)} groups were rendered; minimum is {minimum_groups}"
        )
    output_records = []
    all_pass = True
    for snapshot_group_id in sorted(rendered):
        source = rendered[snapshot_group_id]
        decision = reviewed[snapshot_group_id]
        checks = decision.get("checks", {})
        missing = [name for name in CHECKS if not isinstance(checks.get(name), bool)]
        if missing:
            raise ValueError(
                f"non-boolean or missing checks for {snapshot_group_id}: {missing}"
            )
        passed = decision.get("status") == "PASS" and all(checks[name] for name in CHECKS)
        all_pass &= passed
        output_records.append(
            {
                "snapshot_group_id": snapshot_group_id,
                "task_id": source["task_id"],
                "source_episode_id": source["source_episode_id"],
                "source_frame": source["source_frame"],
                "status": "PASS" if passed else "HOLD",
                "checks": {name: checks[name] for name in CHECKS},
                "notes": str(decision.get("notes", "")),
                "montage_path": source["montage_path"],
                "montage_sha256": source["montage_sha256"],
                "condition_roles": source["condition_roles"],
            }
        )
    return {
        "schema": "dsol_constructed_m0_manual_visual_audit_v1",
        "status": "PASS" if all_pass else "HOLD",
        "m1_admission": bool(all_pass),
        "automatically_promoted": False,
        "review_mode": str(decisions.get("review_mode", "unspecified")),
        "reviewer": str(decisions.get("reviewer", "unspecified")),
        "reviewed_at_utc": str(decisions.get("reviewed_at_utc", "unspecified")),
        "selection_sha256": selection_sha256,
        "reviewed_snapshot_group_count": len(output_records),
        "checks_required": list(CHECKS),
        "records": output_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-groups", type=int, default=20)
    args = parser.parse_args()
    manifest = json.loads(args.render_manifest.read_text(encoding="utf-8"))
    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    selection_hash = sha256(args.selection)
    result = finalize(
        render_manifest=manifest,
        selection_sha256=selection_hash,
        decisions=decisions,
        minimum_groups=args.minimum_groups,
    )
    result.update(
        {
            "render_manifest": str(args.render_manifest.resolve()),
            "render_manifest_sha256": sha256(args.render_manifest),
            "selection": str(args.selection.resolve()),
            "decisions": str(args.decisions.resolve()),
            "decisions_sha256": sha256(args.decisions),
        }
    )
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "m1_admission": result["m1_admission"],
                "reviewed_snapshot_group_count": result[
                    "reviewed_snapshot_group_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
