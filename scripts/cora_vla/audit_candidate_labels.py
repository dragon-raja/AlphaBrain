from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


LABELS = ("action_compatible", "effect_compatible", "joint_compatible", "teacher_success")


def confusion(rows: Sequence[Mapping[str, object]], label: str) -> dict[str, int | float]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in rows:
        predicted = bool(row[label])
        physical = bool(row["physical_compatible"])
        key = "tp" if predicted and physical else "fp" if predicted else "fn" if physical else "tn"
        counts[key] += 1
    total = len(rows)
    return {
        **counts,
        "total": total,
        "agreement": float((counts["tp"] + counts["tn"]) / total),
        "physical_success_recall": float(counts["tp"] / max(counts["tp"] + counts["fn"], 1)),
    }


def balanced_examples(rows: Sequence[Mapping[str, object]], count: int) -> list[dict[str, object]]:
    by_seed_group = defaultdict(list)
    for row in rows:
        if not bool(row["joint_compatible"]) and bool(row["physical_compatible"]):
            by_seed_group[(int(row["checkpoint_seed"]), str(row["pair_id"]))].append(row)
    selected = []
    keys = sorted(by_seed_group)
    offset = 0
    while len(selected) < count and keys:
        next_keys = []
        for key in keys:
            candidates = by_seed_group[key]
            if offset < len(candidates):
                selected.append(dict(candidates[offset]))
                next_keys.append(key)
                if len(selected) == count:
                    break
        keys = next_keys
        offset += 1
    for row in selected:
        row["visualization_reason"] = "joint_false_physical_success"
    if len(selected) < count:
        used = {
            (row["checkpoint_seed"], row["pair_id"], row["candidate_index"])
            for row in selected
        }
        fallback = [
            row
            for row in rows
            if bool(row["physical_compatible"])
            and (row["checkpoint_seed"], row["pair_id"], row["candidate_index"]) not in used
        ]
        for row in fallback:
            value = dict(row)
            value["visualization_reason"] = "physical_success_context"
            selected.append(value)
            if len(selected) == count:
                break
    if len(selected) < count:
        raise ValueError(f"only found {len(selected)} physical-success examples")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CORA Gate 1 candidate labels against physical outcomes")
    parser.add_argument("--inputs", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--examples-per-outcome", type=int, default=20)
    args = parser.parse_args()
    flat = []
    for path in args.inputs:
        payload = json.loads(path.read_text())
        seed = int(payload["checkpoint_seed"])
        for group_row in payload["rows"]:
            candidate_file = str(group_row["candidate_file"])
            for candidate in group_row["candidate_metrics"]:
                flat.append(
                    {
                        **candidate,
                        "checkpoint_seed": seed,
                        "pair_id": str(group_row["pair_id"]),
                        "outcome": str(group_row["outcome"]),
                        "frame_index": int(group_row["frame_index"]),
                        "candidate_file": candidate_file,
                    }
                )
    by_outcome = {}
    examples = []
    for outcome in ("attached", "slipped"):
        selected = [row for row in flat if row["outcome"] == outcome]
        by_outcome[outcome] = {
            "candidate_count": len(selected),
            "physical_success_rate": float(np.mean([row["physical_compatible"] for row in selected])),
            "teacher_completion_rate": float(np.mean([row["teacher_success"] for row in selected])),
            "labels_vs_physical": {label: confusion(selected, label) for label in LABELS},
            "joint_false_physical_success_count": int(
                sum(not row["joint_compatible"] and row["physical_compatible"] for row in selected)
            ),
        }
        examples.extend(balanced_examples(selected, args.examples_per_outcome))
    result = {
        "experiment": "cora_gate1_candidate_label_audit",
        "physical_label_is_reference_only_for_audit": True,
        "candidate_count": len(flat),
        "by_outcome": by_outcome,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.examples.write_text(json.dumps({"examples": examples}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "examples": len(examples)}, sort_keys=True))


if __name__ == "__main__":
    main()
