from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from scripts.dsol_paper1.libero_pair_records import read_record, sha256_file


def _action_chunk(actions: np.ndarray, frame: int, horizon: int) -> np.ndarray:
    chunk = np.asarray(actions[frame : frame + horizon], dtype=np.float32)
    if len(chunk) < horizon:
        chunk = np.concatenate(
            [chunk, np.zeros((horizon - len(chunk), actions.shape[1]), np.float32)],
            axis=0,
        )
    return chunk


def _sample_indices(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, count, dtype=int).tolist()))


def _montage(rows: list[dict], output: Path) -> None:
    names = ("canonical", "broad_a", "broad_b", "wrist")
    image_size = 224
    label_height = 30
    canvas = Image.new(
        "RGB",
        (len(names) * image_size, len(rows) * (image_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        top = row_index * (image_size + label_height)
        for column, name in enumerate(names):
            left = column * image_size
            canvas.paste(Image.fromarray(row["images"][name]), (left, top))
            label = name if column else f"{row['task_id']} | {name}"
            draw.text((left + 5, top + image_size + 7), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def audit(args: argparse.Namespace) -> dict:
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "VERIFIED":
        raise ValueError("collection manifest is not VERIFIED")

    episode_splits: dict[str, set[str]] = defaultdict(set)
    sample_ids = set()
    duplicate_sample_ids = 0
    pose_a = Counter()
    pose_b = Counter()
    split_counts = Counter()
    task_counts = Counter()
    shard_audits = []
    sampled_rows = []
    hdf5_cache = {}

    try:
        for shard_row in manifest["shards"]:
            shard_root = root / shard_row["path"]
            shard_manifest = json.loads(
                (shard_root / "manifest.json").read_text(encoding="utf-8")
            )
            shard_path = shard_root / shard_manifest["shard"]
            records_path = shard_root / shard_manifest["records"]
            shard_sha_ok = sha256_file(shard_path) == shard_manifest["shard_sha256"]
            records_sha_ok = sha256_file(records_path) == shard_manifest["records_sha256"]
            rows = [
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) != int(shard_manifest["record_count"]):
                raise ValueError(f"record count mismatch: {shard_root}")
            for row in rows:
                sample_id = str(row["sample_id"])
                if sample_id in sample_ids:
                    duplicate_sample_ids += 1
                sample_ids.add(sample_id)
                episode_splits[str(row["episode_id"])].add(str(row["split"]))
                split_counts[str(row["split"])] += 1
                task_counts[str(shard_row["task_id"])] += 1
                pose_a[str(row["pose_a_id"])] += 1
                pose_b[str(row["pose_b_id"])] += 1

            source_path = Path(shard_manifest["source_hdf5"])
            source = hdf5_cache.setdefault(source_path, h5py.File(source_path, "r"))
            shard_samples = []
            with shard_path.open("rb") as handle:
                for index in _sample_indices(len(rows), args.samples_per_shard):
                    row = rows[index]
                    record = read_record(handle, offset=int(row["offset"]))
                    header = record["header"]
                    images = record["images"]
                    demo = source["data"][header["demo"]]
                    frame = int(header["frame"])
                    source_state = np.asarray(demo["states"][frame])
                    state_sha = hashlib.sha256(source_state.tobytes()).hexdigest()
                    source_actions = np.asarray(demo["actions"], dtype=np.float32)
                    expected_actions = _action_chunk(
                        source_actions,
                        frame,
                        int(shard_manifest["action_horizon"]),
                    )
                    expected_robot_state = np.concatenate(
                        [
                            np.asarray(demo["obs/ee_pos"][frame]),
                            np.asarray(demo["obs/ee_ori"][frame]),
                            np.asarray(demo["obs/gripper_states"][frame]),
                        ]
                    ).astype(np.float32)
                    shard_samples.append(
                        {
                            "sample_id": header["sample_id"],
                            "state_sha_match": state_sha == header["source_state_sha256"],
                            "action_match": bool(
                                np.array_equal(
                                    expected_actions,
                                    np.asarray(header["action_chunk"], dtype=np.float32),
                                )
                            ),
                            "robot_state_match": bool(
                                np.array_equal(
                                    expected_robot_state,
                                    np.asarray(header["robot_state"], dtype=np.float32),
                                )
                            ),
                            "pose_ids_distinct": row["pose_a_id"] != row["pose_b_id"],
                            "canonical_a_mae": float(
                                np.abs(
                                    images["canonical"].astype(np.int16)
                                    - images["broad_a"].astype(np.int16)
                                ).mean()
                            ),
                            "a_b_mae": float(
                                np.abs(
                                    images["broad_a"].astype(np.int16)
                                    - images["broad_b"].astype(np.int16)
                                ).mean()
                            ),
                            "image_shapes_ok": all(
                                image.shape == (224, 224, 3)
                                for image in images.values()
                            ),
                        }
                    )
                    if len(shard_samples) == 1:
                        sampled_rows.append(
                            {
                                "task_id": shard_row["task_id"],
                                "images": images,
                            }
                        )
            shard_audits.append(
                {
                    "task_id": shard_row["task_id"],
                    "shard_sha256_match": shard_sha_ok,
                    "records_sha256_match": records_sha_ok,
                    "sample_count": len(shard_samples),
                    "samples": shard_samples,
                }
            )
    finally:
        for source in hdf5_cache.values():
            source.close()

    flat_samples = [sample for shard in shard_audits for sample in shard["samples"]]
    split_leakage = {
        episode: sorted(splits)
        for episode, splits in episode_splits.items()
        if len(splits) != 1
    }
    checks = {
        "collection_count_match": len(sample_ids) == int(manifest["record_count"]),
        "no_duplicate_sample_ids": duplicate_sample_ids == 0,
        "no_episode_split_leakage": not split_leakage,
        "split_counts_match": dict(split_counts) == manifest["counts_by_split"],
        "all_shard_hashes_match": all(
            row["shard_sha256_match"] and row["records_sha256_match"]
            for row in shard_audits
        ),
        "all_sample_states_match": all(row["state_sha_match"] for row in flat_samples),
        "all_sample_actions_match": all(row["action_match"] for row in flat_samples),
        "all_sample_robot_states_match": all(
            row["robot_state_match"] for row in flat_samples
        ),
        "all_sample_pose_ids_distinct": all(
            row["pose_ids_distinct"] for row in flat_samples
        ),
        "all_sample_image_shapes_ok": all(
            row["image_shapes_ok"] for row in flat_samples
        ),
        "all_sample_views_change_pixels": all(
            row["canonical_a_mae"] > 0.0 and row["a_b_mae"] > 0.0
            for row in flat_samples
        ),
    }
    report = {
        "schema": "dsol_libero_pair_collection_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "root": str(root),
        "record_count": len(sample_ids),
        "episode_count": len(episode_splits),
        "task_counts": dict(task_counts),
        "split_counts": dict(split_counts),
        "split_episode_counts": dict(
            Counter(next(iter(splits)) for splits in episode_splits.values())
        ),
        "pose_a_support": len(pose_a),
        "pose_b_support": len(pose_b),
        "pose_a_histogram": dict(pose_a),
        "pose_b_histogram": dict(pose_b),
        "sample_count": len(flat_samples),
        "sample_canonical_a_mae": {
            "min": min(row["canonical_a_mae"] for row in flat_samples),
            "mean": float(np.mean([row["canonical_a_mae"] for row in flat_samples])),
            "max": max(row["canonical_a_mae"] for row in flat_samples),
        },
        "sample_a_b_mae": {
            "min": min(row["a_b_mae"] for row in flat_samples),
            "mean": float(np.mean([row["a_b_mae"] for row in flat_samples])),
            "max": max(row["a_b_mae"] for row in flat_samples),
        },
        "checks": checks,
        "split_leakage": split_leakage,
        "shards": shard_audits,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _montage(sampled_rows, args.montage)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a DSOL LIBERO pair collection.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--montage", type=Path, required=True)
    parser.add_argument("--samples-per-shard", type=int, default=16)
    return parser.parse_args(argv)


def main() -> None:
    report = audit(parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "record_count": report["record_count"],
                "episode_count": report["episode_count"],
                "split_counts": report["split_counts"],
                "sample_count": report["sample_count"],
                "pose_a_support": report["pose_a_support"],
                "pose_b_support": report["pose_b_support"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
