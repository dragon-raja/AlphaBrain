from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from build_libero_plus_training_view import (
    DEFAULT_INTRINSICS_224,
    SPLITS,
    camera_pose_group_id,
    scan_tfrecord_offsets,
)


CANONICAL_CAMERA_TO_WORLD_OPENCV = [
    [-1.7233905013069872e-06, 0.5287697435529835, -0.8487653140297038, 0.6065773716836134],
    [0.9999999999985034, 7.823149652530503e-07, -1.5430955956352577e-06, 0.0],
    [-1.5194045527300304e-07, -0.848765314031093, -0.5287697435535403, 0.96],
    [0.0, 0.0, 0.0, 1.0],
]


def _stable_value(seed: int, namespace: str, value: str) -> int:
    payload = f"{seed}:{namespace}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def task_id_from_basename(value: str) -> str:
    name = PurePosixPath(value).name
    for suffix in ("_demo.hdf5", ".hdf5"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def source_category(value: str) -> str:
    path = PurePosixPath(value)
    if len(path.parts) < 3:
        return "unknown"
    # Public LIBERO-Plus paths end in <factor>/<suite>/<task>_demo.hdf5.
    return path.parent.parent.name.lower()


def _scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError("empty text feature")
    item = array.reshape(-1)[0]
    if isinstance(item, (bytes, np.bytes_)):
        return bytes(item).decode("utf-8", errors="replace")
    return str(item)


def assign_episode_splits(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, str]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    assignments: dict[str, str] = {}
    for task_id, task_rows in sorted(by_task.items()):
        ordered = sorted(
            task_rows,
            key=lambda row: _stable_value(
                seed, f"factor_split:{task_id}", str(row["episode_id"])
            ),
        )
        if len(ordered) < 3:
            raise ValueError(f"task {task_id!r} has fewer than three episodes")
        val_count = max(1, round(len(ordered) * 0.1))
        test_count = max(1, round(len(ordered) * 0.1))
        train_count = len(ordered) - val_count - test_count
        if train_count <= 0:
            raise ValueError(f"task {task_id!r} has no train episodes")
        for index, row in enumerate(ordered):
            split = (
                "train"
                if index < train_count
                else "val"
                if index < train_count + val_count
                else "test"
            )
            assignments[str(row["episode_id"])] = split
    return assignments


def add_factor_budget_percentiles(rows: list[dict[str, Any]], *, seed: int) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["factor_class"], row["task_id"])].append(row)
    for key, group in groups.items():
        group.sort(
            key=lambda row: _stable_value(
                seed, f"factor_budget:{':'.join(key)}", str(row["episode_id"])
            )
        )
        for index, row in enumerate(group):
            row["budget_percentile"] = (index + 0.5) / len(group)


def scan_goal_background_rows(
    *,
    dataset_root: Path,
    allowed_tasks: set[str],
    category: str = "env",
    max_shards: int | None = None,
) -> list[dict[str, Any]]:
    try:
        from tfrecord.reader import tfrecord_loader
    except ImportError as error:
        raise RuntimeError("tfrecord==1.14.6 is required") from error

    shards = sorted(dataset_root.rglob("*.tfrecord-*"))
    if max_shards is not None:
        shards = shards[:max_shards]
    if not shards:
        raise ValueError(f"no TFRecord shards found under {dataset_root}")
    description = {
        "episode_metadata/file_path": "byte",
        "steps/language_instruction": "byte",
        "steps/action": "float",
        "steps/reward": "float",
    }
    rows: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(shards, start=1):
        offsets = scan_tfrecord_offsets(shard)
        records = tfrecord_loader(str(shard), None, description)
        record_count = 0
        for record_index, (record, (offset, total_bytes)) in enumerate(
            zip(records, offsets, strict=True)
        ):
            record_count += 1
            source_path = _scalar_text(record["episode_metadata/file_path"])
            if source_category(source_path) != category:
                continue
            task_id = task_id_from_basename(source_path)
            if task_id not in allowed_tasks:
                continue
            action = np.asarray(record["steps/action"], dtype=np.float32)
            if action.size % 7:
                raise ValueError(f"invalid action shape in {shard} record {record_index}")
            step_count = int(action.size // 7)
            rewards = np.asarray(record["steps/reward"], dtype=np.float32).reshape(-1)
            if len(rewards) != step_count:
                raise ValueError(f"reward/action mismatch in {shard} record {record_index}")
            relative_shard = shard.relative_to(dataset_root).as_posix()
            rows.append(
                {
                    "episode_id": f"background::{relative_shard}::{record_index:05d}",
                    "shard": f"background_data/{relative_shard}",
                    "record_index": record_index,
                    "record_offset": offset,
                    "record_total_bytes": total_bytes,
                    "source_basename": PurePosixPath(source_path).name,
                    "source_category": category,
                    "task_id": task_id,
                    "language_instruction": _scalar_text(
                        record["steps/language_instruction"]
                    ),
                    "step_count": step_count,
                    "terminal_reward": float(rewards[-1]),
                    "factor_class": "background_only",
                    "camera_pose_group_id": camera_pose_group_id(
                        CANONICAL_CAMERA_TO_WORLD_OPENCV
                    ),
                    "camera_to_world_opencv": CANONICAL_CAMERA_TO_WORLD_OPENCV,
                }
            )
        if record_count != len(offsets):
            raise ValueError(
                f"TFRecord loader/offset mismatch for {shard}: "
                f"records={record_count} offsets={len(offsets)}"
            )
        print(
            f"scanned Goal shard {shard_index}/{len(shards)}: "
            f"selected_background_episodes={len(rows)}",
            flush=True,
        )
    return rows


def _camera_rows(
    manifest: Mapping[str, Any], *, allowed_tasks: set[str]
) -> list[dict[str, Any]]:
    result = []
    for source in manifest["episodes"]:
        task_id = task_id_from_basename(str(source["source_basename"]))
        if task_id not in allowed_tasks:
            continue
        row = dict(source)
        row.update(
            {
                "episode_id": f"camera::{source['episode_id']}",
                "shard": f"camera_data/{source['shard']}",
                "task_id": task_id,
                "source_category": "camera_view",
                "factor_class": "camera_only",
            }
        )
        result.append(row)
    return result


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    factors = {}
    for factor in ("camera_only", "background_only"):
        selected = [row for row in rows if row["factor_class"] == factor]
        factors[factor] = {
            "episode_count": len(selected),
            "step_count": sum(int(row["step_count"]) for row in selected),
            "task_count": len({row["task_id"] for row in selected}),
        }
    splits = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        splits[split] = {
            "episode_count": len(selected),
            "step_count": sum(int(row["step_count"]) for row in selected),
            "task_count": len({row["task_id"] for row in selected}),
            "factor_counts": dict(Counter(row["factor_class"] for row in selected)),
        }
    return {
        "episode_count": len(rows),
        "step_count": sum(int(row["step_count"]) for row in rows),
        "task_count": len({row["task_id"] for row in rows}),
        "factors": factors,
        "splits": splits,
    }


def build_factor_separated_view(
    *,
    camera_view: Path,
    background_dataset_root: Path,
    protocol: Path,
    output: Path,
    seed: int,
    max_background_shards: int | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training view: {output}")
    camera_manifest_path = camera_view / "manifest.json"
    camera_manifest = json.loads(camera_manifest_path.read_text())
    protocol_payload = json.loads(protocol.read_text())
    goal_tasks = {
        str(row["base_task"])
        for row in protocol_payload["composition_tasks"]
        if row["suite"] == "libero_goal"
    }
    if not goal_tasks:
        raise ValueError("protocol contains no LIBERO-Goal composition tasks")

    camera_rows = _camera_rows(camera_manifest, allowed_tasks=goal_tasks)
    background_rows = scan_goal_background_rows(
        dataset_root=background_dataset_root,
        allowed_tasks=goal_tasks,
        max_shards=max_background_shards,
    )
    if {row["task_id"] for row in camera_rows} != goal_tasks:
        raise ValueError("camera training data does not cover every Goal task")
    if {row["task_id"] for row in background_rows} != goal_tasks:
        missing = sorted(goal_tasks - {row["task_id"] for row in background_rows})
        raise ValueError(f"background training data is missing Goal tasks: {missing}")

    background_assignments = assign_episode_splits(background_rows, seed=seed)
    for row in background_rows:
        row["split"] = background_assignments[row["episode_id"]]
    rows = camera_rows + background_rows
    add_factor_budget_percentiles(rows, seed=seed)
    rows.sort(key=lambda row: (row["factor_class"], row["episode_id"]))

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "study": "pi05_libero_plus_factor_separated_composition_training_view",
        "dataset_root": str(output),
        "sources": {
            "camera_view": str(camera_manifest_path),
            "background_rlds": str(background_dataset_root),
            "evaluation_protocol": str(protocol),
        },
        "seed": seed,
        "split_policy": {
            "camera_rows": "reuse exact-pose grouped source splits",
            "background_rows": "deterministic episode split within task",
            "budget_unit": "split x factor x task",
        },
        "composition_claim": {
            "label": "FACTOR_SEPARATED_CATEGORY_COMPOSITION",
            "camera_only_training": True,
            "background_only_training": True,
            "joint_camera_background_training": False,
            "joint_training_episode_count": 0,
            "exact_camera_texture_identity_available": False,
            "exact_held_pair_claim_allowed": False,
            "reason": (
                "The public Goal RLDS preserves perturbation category in source identity "
                "but not per-episode camera or texture identifiers."
            ),
        },
        "image_schema": {
            "source_resolution": [256, 256],
            "training_resolution": [224, 224],
            "views": ["agentview", "wrist"],
            "external_camera_intrinsics_224": DEFAULT_INTRINSICS_224,
            "background_camera_assumption": "canonical LIBERO agentview",
        },
        "action_schema": dict(camera_manifest["action_schema"]),
        "summary": _summary(rows),
        "episodes": rows,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    staging.mkdir()
    try:
        os.symlink(camera_manifest["dataset_root"], staging / "camera_data")
        os.symlink(background_dataset_root, staging / "background_data")
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build factor-separated LIBERO-Plus camera/background training data"
    )
    parser.add_argument("--camera-view", type=Path, required=True)
    parser.add_argument("--background-dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--max-background-shards", type=int)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    manifest = build_factor_separated_view(
        camera_view=args.camera_view,
        background_dataset_root=args.background_dataset_root,
        protocol=args.protocol,
        output=args.output,
        seed=args.seed,
        max_background_shards=args.max_background_shards,
    )
    print(json.dumps(manifest["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
