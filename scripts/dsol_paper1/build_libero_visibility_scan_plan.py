#!/usr/bin/env python3
"""Build a split-preserving, stage-stratified M0 visibility scan ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}::{value}".encode()).hexdigest()


def load_episode_splits(pair_root: Path) -> dict[str, str]:
    root_manifest = json.loads((pair_root / "manifest.json").read_text())
    result = {}
    for shard in root_manifest["shards"]:
        shard_root = pair_root / shard["path"]
        shard_manifest = json.loads((shard_root / "manifest.json").read_text())
        for line in (shard_root / shard_manifest["records"]).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = str(row["episode_id"])
            split = str(row["split"])
            previous = result.setdefault(episode_id, split)
            if previous != split:
                raise ValueError(f"episode crosses splits: {episode_id}")
    return result


def stage_frames(frame_count: int, fractions: list[float]) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    return sorted(
        {
            min(max(int(round(fraction * (frame_count - 1))), 0), frame_count - 1)
            for fraction in fractions
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--collection-plan", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--calibration-episodes-per-task", type=int, default=2)
    parser.add_argument("--test-episodes-per-task", type=int, default=3)
    parser.add_argument(
        "--all-test-episodes",
        action="store_true",
        help="Use every eligible test episode instead of a fixed per-task limit.",
    )
    parser.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated split subset to emit (default: val,test).",
    )
    parser.add_argument(
        "--exclude-plan",
        type=Path,
        help="Exclude every source episode already present in another scan plan.",
    )
    parser.add_argument(
        "--allow-empty-task-splits",
        action="store_true",
        help="Permit an incremental plan to have no new episodes for a task/split.",
    )
    parser.add_argument("--stage-fractions", default="0.05,0.35,0.65,0.90")
    args = parser.parse_args()

    fractions = [float(value) for value in args.stage_fractions.split(",")]
    if not fractions or any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("stage fractions must lie in [0, 1]")
    requested_splits = tuple(
        value.strip() for value in args.splits.split(",") if value.strip()
    )
    if not requested_splits or any(
        value not in {"val", "test"} for value in requested_splits
    ):
        raise ValueError("splits must be a nonempty subset of val,test")
    excluded_episode_ids: set[str] = set()
    excluded_plan_sha256 = None
    if args.exclude_plan is not None:
        excluded_plan_path = args.exclude_plan.resolve()
        excluded_plan = json.loads(excluded_plan_path.read_text())
        excluded_episode_ids = {
            str(row["episode_id"]) for row in excluded_plan.get("records", [])
        }
        excluded_plan_sha256 = hashlib.sha256(
            excluded_plan_path.read_bytes()
        ).hexdigest()
    episode_splits = load_episode_splits(args.pair_root.resolve())
    collection = json.loads(args.collection_plan.read_text())
    records = []
    counts = defaultdict(int)
    for task in collection["tasks"]:
        hdf5_path = args.hdf5_root.resolve() / task["source"]
        suite = hdf5_path.parent.name
        stem = hdf5_path.stem
        with h5py.File(hdf5_path, "r") as handle:
            demos = sorted(handle["data"].keys())
            candidates = defaultdict(list)
            for demo_index, demo_name in enumerate(demos):
                episode_id = f"{suite}::{stem}::{demo_name}"
                split = episode_splits.get(episode_id)
                if split in requested_splits and episode_id not in excluded_episode_ids:
                    candidates[split].append((demo_index, demo_name, episode_id))
            for split, episode_limit in (
                ("val", args.calibration_episodes_per_task),
                ("test", args.test_episodes_per_task),
            ):
                if split not in requested_splits:
                    continue
                ranked_all = sorted(
                    candidates[split],
                    key=lambda row: stable_key(args.seed, row[2]),
                )
                ranked = (
                    ranked_all
                    if split == "test" and args.all_test_episodes
                    else ranked_all[:episode_limit]
                )
                if not ranked and args.allow_empty_task_splits:
                    continue
                if not ranked:
                    raise ValueError(
                        f"{task['task_id']} has no eligible {split} episodes after exclusions"
                    )
                if not (split == "test" and args.all_test_episodes) and len(
                    ranked
                ) < episode_limit:
                    raise ValueError(
                        f"{task['task_id']} has {len(ranked)} {split} episodes; "
                        f"requires {episode_limit}"
                    )
                for demo_index, demo_name, episode_id in ranked:
                    frame_count = int(handle["data"][demo_name]["states"].shape[0])
                    for stage_index, frame in enumerate(stage_frames(frame_count, fractions)):
                        records.append(
                            {
                                "scan_id": (
                                    f"{task['task_id']}::{split}::{demo_name}::"
                                    f"stage-{stage_index:02d}::frame-{frame:05d}"
                                ),
                                "task_id": task["task_id"],
                                "diagnostic_role": task["diagnostic_role"],
                                "suite": suite,
                                "hdf5": str(hdf5_path),
                                "episode_id": episode_id,
                                "split": split,
                                "demo_index": demo_index,
                                "demo_name": demo_name,
                                "frame": frame,
                                "frame_count": frame_count,
                                "stage_fraction": frame / max(frame_count - 1, 1),
                            }
                        )
                        counts[f"{split}::{task['task_id']}"] += 1

    payload = {
        "schema": "dsol_libero_visibility_scan_plan_v1",
        "seed": args.seed,
        "source_pair_manifest": str(args.pair_root.resolve() / "manifest.json"),
        "source_pair_manifest_sha256": hashlib.sha256(
            (args.pair_root.resolve() / "manifest.json").read_bytes()
        ).hexdigest(),
        "collection_plan": str(args.collection_plan.resolve()),
        "requested_splits": list(requested_splits),
        "all_test_episodes": bool(args.all_test_episodes),
        "excluded_source_episode_count": len(excluded_episode_ids),
        "exclude_plan": (
            None if args.exclude_plan is None else str(args.exclude_plan.resolve())
        ),
        "exclude_plan_sha256": excluded_plan_sha256,
        "stage_fractions": fractions,
        "record_count": len(records),
        "counts": dict(sorted(counts.items())),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"record_count": len(records), "counts": payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()
