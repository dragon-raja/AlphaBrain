#!/usr/bin/env python3
"""Restrict a visibility scan plan to states frozen by a dense test protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_SCHEMA = "dsol_constructed_dense_view_oracle_protocol_v1"


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


def build(
    source_plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    source_plan_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    if source_plan.get("status") != "PASS":
        raise ValueError("source scan plan must PASS")
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("status") != "PASS":
        raise ValueError("dense protocol must use the expected schema and PASS")
    if protocol.get("split") != "test":
        raise ValueError("only a frozen test protocol may build this plan")
    selected_ids = {str(row["pair_key"]) for row in protocol["selected_states"]}
    records = [
        dict(row)
        for row in source_plan["records"]
        if str(row["scan_id"]) in selected_ids
    ]
    found = {str(row["scan_id"]) for row in records}
    missing = sorted(selected_ids - found)
    if missing:
        raise ValueError(f"protocol states are absent from source plan: {missing}")
    if len(records) != int(protocol["selected_state_count"]):
        raise ValueError("selected test state count differs from dense protocol")
    counts = Counter(f"{row['split']}::{row['task_id']}" for row in records)
    return {
        "schema": "dsol_dense_test_visibility_scan_plan_v1",
        "status": "PASS",
        "selection_policy": "frozen_dense_test_protocol_only",
        "policy_outcomes_used_for_selection": False,
        "source_plan": str(source_plan_path.resolve()),
        "source_plan_sha256": sha256(source_plan_path),
        "dense_test_protocol": str(protocol_path.resolve()),
        "dense_test_protocol_sha256": sha256(protocol_path),
        "record_count": len(records),
        "counts": dict(sorted(counts.items())),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_plan = json.loads(args.source_plan.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    payload = build(
        source_plan,
        protocol,
        source_plan_path=args.source_plan,
        protocol_path=args.protocol,
    )
    atomic_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "records": payload["record_count"]}))


if __name__ == "__main__":
    main()
