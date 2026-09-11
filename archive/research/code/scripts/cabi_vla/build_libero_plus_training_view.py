from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INTRINSICS_224 = [
    [270.3919189857867, 0.0, 112.0],
    [0.0, 270.3919189857867, 112.0],
    [0.0, 0.0, 1.0],
]
SPLITS = ("train", "val", "test")


def scan_tfrecord_offsets(path: Path) -> list[tuple[int, int]]:
    """Return byte offset and total encoded length for every TFRecord record."""

    rows: list[tuple[int, int]] = []
    size = path.stat().st_size
    with path.open("rb") as stream:
        while stream.tell() < size:
            offset = stream.tell()
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"truncated TFRecord length at byte {offset}: {path}")
            (payload_length,) = struct.unpack("<Q", length_bytes)
            total_length = 8 + 4 + int(payload_length) + 4
            if offset + total_length > size:
                raise ValueError(f"record at byte {offset} exceeds TFRecord size: {path}")
            stream.seek(total_length - 8, os.SEEK_CUR)
            rows.append((offset, total_length))
        if stream.tell() != size:
            raise ValueError(f"TFRecord scan did not end at file boundary: {path}")
    return rows


def camera_pose_group_id(matrix: Sequence[Sequence[float]], *, decimals: int = 4) -> str:
    value = np.asarray(matrix, dtype="<f8")
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("camera pose must be a finite 4x4 matrix")
    rounded = np.round(value, decimals=decimals)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:20]


def _stable_bucket(seed: int, namespace: str, value: str, modulus: int) -> int:
    payload = f"{seed}:{namespace}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulus


def assign_pose_group_splits(
    episodes: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, str]:
    """Split exact camera-pose groups and repair task coverage deterministically."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 < val_fraction < 1.0 - train_fraction:
        raise ValueError("val_fraction must leave a non-empty test split")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        groups[str(episode["camera_pose_group_id"])].append(index)

    train_cut = round(train_fraction * 10_000)
    val_cut = round((train_fraction + val_fraction) * 10_000)
    assignments: dict[str, str] = {}
    for group_id in groups:
        bucket = _stable_bucket(seed, "pose_split", group_id, 10_000)
        assignments[group_id] = (
            "train" if bucket < train_cut else "val" if bucket < val_cut else "test"
        )

    tasks = sorted({str(row["language_instruction"]) for row in episodes})

    def task_counts() -> dict[str, Counter[str]]:
        counts = {split: Counter() for split in SPLITS}
        for group_id, indices in groups.items():
            split = assignments[group_id]
            counts[split].update(
                str(episodes[index]["language_instruction"]) for index in indices
            )
        return counts

    # Exact pose groups remain atomic. Move the smallest useful train group to
    # each missing holdout split while preserving at least one train episode.
    for target_split in ("val", "test"):
        while True:
            counts = task_counts()
            missing_tasks = [task for task in tasks if counts[target_split][task] == 0]
            if not missing_tasks:
                break
            missing_task = missing_tasks[0]
            candidates = []
            missing_set = set(missing_tasks)
            for group_id, indices in groups.items():
                if assignments[group_id] != "train":
                    continue
                group_counts = Counter(
                    str(episodes[index]["language_instruction"]) for index in indices
                )
                if missing_task not in group_counts:
                    continue
                if any(
                    counts["train"][task] - amount <= 0
                    for task, amount in group_counts.items()
                ):
                    continue
                covered = len(missing_set.intersection(group_counts))
                candidates.append(
                    (len(indices) / max(covered, 1), len(indices), -covered, group_id)
                )
            if not candidates:
                raise ValueError(
                    f"cannot give task {missing_task!r} {target_split} coverage "
                    "without breaking train coverage"
                )
            assignments[min(candidates)[-1]] = target_split

    final_counts = task_counts()
    for split in SPLITS:
        missing = [task for task in tasks if final_counts[split][task] == 0]
        if missing:
            raise ValueError(f"split {split} is missing task coverage: {missing}")
    return assignments


def add_budget_percentiles(episodes: list[dict[str, Any]], *, seed: int) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[(episode["split"], episode["language_instruction"])].append(episode)
    for key, rows in grouped.items():
        rows.sort(
            key=lambda row: _stable_bucket(
                seed,
                f"budget:{key[0]}:{key[1]}",
                row["episode_id"],
                2**63 - 1,
            )
        )
        count = len(rows)
        for index, row in enumerate(rows):
            row["budget_percentile"] = (index + 0.5) / count


def _split_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in SPLITS:
        rows = [row for row in episodes if row["split"] == split]
        summary[split] = {
            "episode_count": len(rows),
            "step_count": sum(int(row["step_count"]) for row in rows),
            "task_count": len({row["language_instruction"] for row in rows}),
            "camera_pose_group_count": len({row["camera_pose_group_id"] for row in rows}),
            "minimum_episodes_per_task": min(
                Counter(row["language_instruction"] for row in rows).values()
            ),
        }
    return summary


def build_training_view(
    *,
    dataset_root: Path,
    audit_path: Path,
    output: Path,
    seed: int,
    max_shards: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training view: {output}")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "complete":
        raise ValueError("camera dataset audit is not complete")
    audit_rows = list(audit["episodes"])
    shard_names = sorted({str(row["shard"]) for row in audit_rows})
    if max_shards is not None:
        if max_shards <= 0:
            raise ValueError("max_shards must be positive")
        shard_names = shard_names[:max_shards]
        allowed = set(shard_names)
        audit_rows = [row for row in audit_rows if str(row["shard"]) in allowed]

    rows_by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        rows_by_shard[str(row["shard"])].append(row)

    episodes: list[dict[str, Any]] = []
    for shard_index, shard_name in enumerate(shard_names, start=1):
        shard_path = dataset_root / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        offsets = scan_tfrecord_offsets(shard_path)
        shard_rows = sorted(rows_by_shard[shard_name], key=lambda row: int(row["record_index"]))
        if len(offsets) != len(shard_rows):
            raise ValueError(
                f"audit/TFRecord count mismatch for {shard_name}: "
                f"audit={len(shard_rows)} records={len(offsets)}"
            )
        for expected_index, (row, (offset, total_length)) in enumerate(
            zip(shard_rows, offsets, strict=True)
        ):
            record_index = int(row["record_index"])
            if record_index != expected_index:
                raise ValueError(
                    f"non-contiguous record index in {shard_name}: "
                    f"expected={expected_index} actual={record_index}"
                )
            matrix = row["external_camera_to_world_opencv"]
            episodes.append(
                {
                    "episode_id": f"{Path(shard_name).name}::{record_index:05d}",
                    "shard": shard_name,
                    "record_index": record_index,
                    "record_offset": offset,
                    "record_total_bytes": total_length,
                    "source_basename": row["source_basename"],
                    "language_instruction": row["language_instruction"],
                    "step_count": int(row["step_count"]),
                    "terminal_reward": float(row["terminal_reward"]),
                    "camera_pose_group_id": camera_pose_group_id(matrix),
                    "camera_to_world_opencv": matrix,
                }
            )
        print(
            f"indexed shard {shard_index}/{len(shard_names)}: "
            f"episodes={len(episodes)} path={Path(shard_name).name}",
            flush=True,
        )

    assignments = assign_pose_group_splits(episodes, seed=seed)
    for row in episodes:
        row["split"] = assignments[row["camera_pose_group_id"]]
    add_budget_percentiles(episodes, seed=seed)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "study": "pi05_libero_plus_multiview_training_view",
        "dataset_root": str(dataset_root),
        "source_audit": str(audit_path),
        "seed": seed,
        "split_policy": {
            "unit": "external_camera_pose_rounded_1e4",
            "initial_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
            "task_coverage_repair": True,
            "leakage_rule": "one exact pose group belongs to exactly one split",
        },
        "image_schema": {
            "source_resolution": [256, 256],
            "training_resolution": [224, 224],
            "views": ["agentview", "wrist"],
            "external_camera_intrinsics_224": DEFAULT_INTRINSICS_224,
        },
        "action_schema": {
            "action_dim": 7,
            "action_horizon": 10,
            "padding": "zeros_at_episode_end",
            "mean": audit["summary"]["action_mean"],
            "std": audit["summary"]["action_std"],
        },
        "summary": {
            "episode_count": len(episodes),
            "step_count": sum(row["step_count"] for row in episodes),
            "task_count": len({row["language_instruction"] for row in episodes}),
            "camera_pose_group_count": len({row["camera_pose_group_id"] for row in episodes}),
            "shard_count": len(shard_names),
            "splits": _split_summary(episodes),
        },
        "episodes": episodes,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build indexed LIBERO-Plus Pi0.5 training view")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--max-shards", type=int)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    manifest = build_training_view(
        dataset_root=args.dataset_root,
        audit_path=args.audit,
        output=args.output,
        seed=args.seed,
        max_shards=args.max_shards,
    )
    print(json.dumps(manifest["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
