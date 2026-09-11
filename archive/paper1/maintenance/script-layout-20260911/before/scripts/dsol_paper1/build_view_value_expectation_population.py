#!/usr/bin/env python3
"""Freeze source-disjoint calibration and held-out LIBERO state populations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py


SOURCE_PATTERN = re.compile(
    rb"libero_(?:goal|object|spatial|10)::[^\"\s]+::demo_[0-9]+"
)
STAGE_FRACTIONS = {
    "calibration": (0.25, 0.55),
    "heldout_test": (0.15, 0.15, 0.25, 0.25, 0.55, 0.55),
}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(label: str, root_seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{root_seed}::{label}".encode()).digest()[:4], "little"
    )


def collect_legacy_sources(root: Path) -> list[str]:
    sources = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        with path.open("rb") as handle:
            carry = b""
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                payload = carry + chunk
                sources.update(match.decode() for match in SOURCE_PATTERN.findall(payload))
                carry = payload[-1024:]
    return sorted(sources)


def task_construction_spec(config_root: Path, task_id: str) -> Path:
    path = config_root / f"libero_constructed_{task_id}_v1.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing construction specification: {path}")
    return path.resolve()


def source_group(suite: str, hdf5_path: Path, demo_name: str) -> str:
    return f"{suite}::{hdf5_path.stem}::{demo_name}"


def task_states(
    task: Mapping[str, Any],
    *,
    hdf5_root: Path,
    config_root: Path,
    legacy_sources: set[str],
    root_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    hdf5_path = (hdf5_root / str(task["source"])).resolve()
    suite = hdf5_path.parent.name
    with h5py.File(hdf5_path, "r") as handle:
        demos = sorted(handle["data"].keys())
        inventory = [
            {
                "demo_name": demo_name,
                "demo_index": demo_index,
                "frame_count": int(handle["data"][demo_name]["states"].shape[0]),
                "source_group": source_group(suite, hdf5_path, demo_name),
            }
            for demo_index, demo_name in enumerate(demos)
        ]
    eligible = [row for row in inventory if row["source_group"] not in legacy_sources]
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"{root_seed}::{task['task_id']}::{row['source_group']}".encode()
        ).hexdigest()
    )
    if len(eligible) < 8:
        raise ValueError(f"{task['task_id']} has only {len(eligible)} untouched sources")
    construction_spec = task_construction_spec(config_root, str(task["task_id"]))
    split_sources = {
        "calibration": eligible[:2],
        "heldout_test": eligible[2:8],
    }
    result = {}
    for split, selected in split_sources.items():
        if len(selected) != len(STAGE_FRACTIONS[split]):
            raise AssertionError(f"state/stage count mismatch for {split}")
        states = []
        for row, stage_fraction in zip(selected, STAGE_FRACTIONS[split]):
            frame = max(1, min(row["frame_count"] - 2, round(stage_fraction * (row["frame_count"] - 1))))
            actual_fraction = frame / (row["frame_count"] - 1)
            pair_key = (
                f"expectation-v1::{split}::{task['task_id']}::"
                f"{row['demo_name']}::frame-{frame:05d}"
            )
            states.append(
                {
                    "pair_key": pair_key,
                    "source_group": row["source_group"],
                    "task_id": str(task["task_id"]),
                    "diagnostic_role": str(task["diagnostic_role"]),
                    "suite": suite,
                    "hdf5": str(hdf5_path),
                    "demo_name": row["demo_name"],
                    "demo_index": row["demo_index"],
                    "source_state_index": frame,
                    "frame_count": row["frame_count"],
                    "stage_fraction": actual_fraction,
                    "split": split,
                    "environment_seed": stable_seed(pair_key, root_seed + 17),
                    "construction_spec": str(construction_spec),
                    "construction_spec_sha256": sha256_file(construction_spec),
                }
            )
        result[split] = states
    return result


def build(
    *,
    collection_plan: Path,
    hdf5_root: Path,
    config_root: Path,
    legacy_root: Path,
    root_seed: int,
) -> dict[str, Any]:
    plan = json.loads(collection_plan.read_text(encoding="utf-8"))
    tasks = list(plan["tasks"])
    if len(tasks) != 8:
        raise ValueError("formal population requires exactly eight preregistered tasks")
    legacy_sources = collect_legacy_sources(legacy_root)
    population = {"calibration": [], "heldout_test": []}
    for task in tasks:
        selected = task_states(
            task,
            hdf5_root=hdf5_root,
            config_root=config_root,
            legacy_sources=set(legacy_sources),
            root_seed=root_seed,
        )
        for split in population:
            population[split].extend(selected[split])
    for split in population:
        population[split].sort(key=lambda row: (row["task_id"], row["source_group"]))
    calibration_sources = {row["source_group"] for row in population["calibration"]}
    test_sources = {row["source_group"] for row in population["heldout_test"]}
    calibration_states = {row["pair_key"] for row in population["calibration"]}
    test_states = {row["pair_key"] for row in population["heldout_test"]}
    task_counts = {
        split: {
            task["task_id"]: sum(row["task_id"] == task["task_id"] for row in population[split])
            for task in tasks
        }
        for split in population
    }
    status = "PASS" if (
        len(population["calibration"]) == 16
        and len(population["heldout_test"]) == 48
        and not calibration_sources.intersection(test_sources)
        and not calibration_states.intersection(test_states)
        and all(value == 2 for value in task_counts["calibration"].values())
        and all(value == 6 for value in task_counts["heldout_test"].values())
        and not (calibration_sources | test_sources).intersection(legacy_sources)
    ) else "FAIL"
    return {
        "schema": "dsol_view_value_expectation_population_v1",
        "status": status,
        "root_seed": root_seed,
        "selection_policy": "sha256_ordered_untouched_source_demonstrations_with_preregistered_stage_strata",
        "collection_plan": str(collection_plan.resolve()),
        "collection_plan_sha256": sha256_file(collection_plan),
        "legacy_root": str(legacy_root.resolve()),
        "legacy_source_count": len(legacy_sources),
        "legacy_source_ids_sha256": hashlib.sha256(
            "\n".join(legacy_sources).encode()
        ).hexdigest(),
        "legacy_sources_embedded": False,
        "task_count": len(tasks),
        "task_counts": task_counts,
        "source_disjoint": not calibration_sources.intersection(test_sources),
        "state_disjoint": not calibration_states.intersection(test_states),
        "legacy_source_disjoint": not (
            calibration_sources | test_sources
        ).intersection(legacy_sources),
        "population": {
            "calibration": {
                "state_count": len(population["calibration"]),
                "source_group_count": len(calibration_sources),
                "states": population["calibration"],
            },
            "heldout_test": {
                "state_count": len(population["heldout_test"]),
                "source_group_count": len(test_sources),
                "states": population["heldout_test"],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--root-seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        collection_plan=args.collection_plan,
        hdf5_root=args.hdf5_root,
        config_root=args.config_root,
        legacy_root=args.legacy_root,
        root_seed=args.root_seed,
    )
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "calibration": result["population"]["calibration"]["state_count"],
                "heldout_test": result["population"]["heldout_test"]["state_count"],
                "legacy_source_disjoint": result["legacy_source_disjoint"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
