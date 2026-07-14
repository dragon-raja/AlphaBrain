from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from libero_snapshot_collector import gripper_transition_horizon


def oracle_window_horizon(
    frame_index: int,
    *,
    feedback_reveal_time: int,
    action_divergence_time: int,
    horizon: int,
) -> int:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if frame_index >= feedback_reveal_time:
        return horizon
    return min(horizon, max(0, action_divergence_time - frame_index))


def assign_window_labels(
    records: Sequence[Mapping[str, Any]],
    *,
    horizon: int,
    seed: int,
) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["window_group_id"]].append(record)
    group_ids = sorted(grouped)
    oracle_by_group = {
        group_id: int(grouped[group_id][0]["oracle_feedback_horizon"]) for group_id in group_ids
    }
    for group_id, rows in grouped.items():
        if any(int(row["oracle_feedback_horizon"]) != oracle_by_group[group_id] for row in rows):
            raise ValueError(f"branches disagree on oracle horizon for {group_id}")

    rng = np.random.default_rng(seed)
    random_by_group = {group_id: int(rng.integers(0, horizon + 1)) for group_id in group_ids}
    shuffled_by_group = {}
    by_stratum: dict[tuple[str, int], list[str]] = defaultdict(list)
    for group_id in group_ids:
        split = str(grouped[group_id][0]["split"])
        by_stratum[(split, len(grouped[group_id]))].append(group_id)
    for (split, multiplicity), split_groups in sorted(by_stratum.items()):
        values = np.asarray([oracle_by_group[group_id] for group_id in split_groups], dtype=np.int64)
        stratum = f"{split}:{multiplicity}"
        split_rng = np.random.default_rng(
            seed + int(hashlib.sha256(stratum.encode()).hexdigest()[:8], 16)
        )
        values = values[split_rng.permutation(len(values))]
        shuffled_by_group.update(
            {group_id: int(value) for group_id, value in zip(split_groups, values, strict=True)}
        )

    labels = {}
    for record in records:
        sample_id = str(record["sample_id"])
        group_id = str(record["window_group_id"])
        labels[sample_id] = {
            "full_h": horizon,
            "random_feedback_horizon": random_by_group[group_id],
            "shuffled_oracle_horizon": shuffled_by_group[group_id],
            "gripper_transition_horizon": int(record["gripper_transition_horizon"]),
            "oracle_feedback_horizon": int(record["oracle_feedback_horizon"]),
            "short_h": min(5, horizon),
        }
    return labels


def _save_frame(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image)).save(path, format="JPEG", quality=quality, subsampling=0)


def build_records(
    episode_root: Path,
    output_dir: Path,
    *,
    horizon: int,
    jpeg_quality: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((episode_root / "manifest.json").read_text())
    records = []
    group_summaries = []
    for group in manifest["groups"]:
        pair_id = str(group["pair_id"])
        feedback_reveal = int(group["feedback_reveal_time"])
        action_divergence = int(group["action_divergence_time"])
        split = str(group["split"])
        group_records = []
        for branch in ("attached", "slipped"):
            episode_path = episode_root / group["episode_files"][branch]
            with np.load(episode_path, allow_pickle=False) as episode:
                actions = np.asarray(episode["actions"], dtype=np.float32)
                robot_states = np.asarray(episode["robot_state"], dtype=np.float32)
                agentviews = np.asarray(episode["agentview"], dtype=np.uint8)
                wrists = np.asarray(episode["wrist"], dtype=np.uint8)
                valid_windows = len(actions) - horizon + 1
                if valid_windows <= 0:
                    raise ValueError(f"episode shorter than horizon: {episode_path}")
                for frame_index in range(valid_windows):
                    sample_id = f"{pair_id}::{branch}::{frame_index:04d}"
                    window_group_id = f"{pair_id}::{frame_index:04d}"
                    frame_dir = output_dir / "frames" / pair_id / branch
                    agent_path = frame_dir / f"{frame_index:04d}-agent.jpg"
                    wrist_path = frame_dir / f"{frame_index:04d}-wrist.jpg"
                    _save_frame(agent_path, agentviews[frame_index], jpeg_quality)
                    _save_frame(wrist_path, wrists[frame_index], jpeg_quality)
                    action_chunk = actions[frame_index : frame_index + horizon]
                    oracle = oracle_window_horizon(
                        frame_index,
                        feedback_reveal_time=feedback_reveal,
                        action_divergence_time=action_divergence,
                        horizon=horizon,
                    )
                    row = {
                        "sample_id": sample_id,
                        "window_group_id": window_group_id,
                        "pair_id": pair_id,
                        "branch_id": branch,
                        "branch_outcome": branch,
                        "task": "grasp_slip_full_episode",
                        "split": split,
                        "frame_index": frame_index,
                        "observation": {
                            "agentview_path": str(agent_path.relative_to(output_dir)),
                            "wrist_path": str(wrist_path.relative_to(output_dir)),
                        },
                        "robot_state": robot_states[frame_index].round(8).tolist(),
                        "language_instruction": "put the cream cheese in the bowl",
                        "action_chunk": action_chunk.round(8).tolist(),
                        "oracle_feedback_horizon": oracle,
                        "gripper_transition_horizon": gripper_transition_horizon(action_chunk),
                        "is_post_feedback": frame_index >= feedback_reveal,
                    }
                    records.append(row)
                    group_records.append(row)
        pre = [row for row in group_records if not row["is_post_feedback"]]
        post = [row for row in group_records if row["is_post_feedback"]]
        group_summaries.append(
            {
                "pair_id": pair_id,
                "split": split,
                "window_count": len(group_records),
                "pre_feedback_window_count": len(pre),
                "post_feedback_window_count": len(post),
                "pre_feedback_oracle_histogram": {
                    str(value): sum(row["oracle_feedback_horizon"] == value for row in pre)
                    for value in sorted({row["oracle_feedback_horizon"] for row in pre})
                },
            }
        )
    return records, group_summaries


def build_quality_report(
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, int]],
    *,
    horizon: int,
) -> dict[str, Any]:
    pair_splits: dict[str, set[str]] = defaultdict(set)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        pair_splits[str(row["pair_id"])].add(str(row["split"]))
        grouped[str(row["window_group_id"])].append(row)
    marginal_by_split: dict[str, dict[str, Counter[int]]] = defaultdict(
        lambda: {"oracle": Counter(), "shuffled": Counter()}
    )
    missing_labels = False
    for row in records:
        split = str(row["split"])
        sample_id = str(row["sample_id"])
        sample_labels = labels.get(sample_id)
        if sample_labels is None or not {
            "oracle_feedback_horizon",
            "shuffled_oracle_horizon",
        } <= sample_labels.keys():
            missing_labels = True
            continue
        marginal_by_split[split]["oracle"][int(sample_labels["oracle_feedback_horizon"])] += 1
        marginal_by_split[split]["shuffled"][int(sample_labels["shuffled_oracle_horizon"])] += 1
    checks = {
        "group_preserving_split": all(len(values) == 1 for values in pair_splits.values()),
        "paired_windows_share_oracle": all(
            len({int(row["oracle_feedback_horizon"]) for row in rows}) == 1 for rows in grouped.values()
        ),
        "post_feedback_horizon_restored": all(
            int(row["oracle_feedback_horizon"]) == horizon
            for row in records
            if bool(row["is_post_feedback"])
        ),
        "oracle_is_not_policy_observation": all(
            set(row["observation"]) == {"agentview_path", "wrist_path"} for row in records
        ),
        "all_labels_present": not missing_labels,
        "shuffled_oracle_sample_marginal_preserved": not missing_labels and all(
            values["oracle"] == values["shuffled"] for values in marginal_by_split.values()
        ),
        "label_bounds": all(
            all(0 <= int(value) <= horizon for value in row.values()) for row in labels.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "record_count": len(records),
            "window_group_count": len(grouped),
            "pair_count": len(pair_splits),
            "pre_feedback_record_count": sum(not bool(row["is_post_feedback"]) for row in records),
            "post_feedback_record_count": sum(bool(row["is_post_feedback"]) for row in records),
            "oracle_downweighted_record_count": sum(
                int(row["oracle_feedback_horizon"]) < horizon for row in records
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Pi0.5 windows from complete LIBERO counterfactual episodes")
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 80 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [80, 100]")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    staging = args.output_dir.parent / f".{args.output_dir.name}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    records, groups = build_records(
        args.episode_root,
        staging,
        horizon=args.horizon,
        jpeg_quality=args.jpeg_quality,
    )
    labels = assign_window_labels(records, horizon=args.horizon, seed=args.seed)
    quality = build_quality_report(records, labels, horizon=args.horizon)
    manifest = {
        "schema_version": 1,
        "generator": "build_libero_episode_windows.py",
        "episode_root": str(args.episode_root),
        "horizon": args.horizon,
        "image_storage": {"format": "JPEG", "quality": args.jpeg_quality, "chroma_subsampling": 0},
        "policy_input_fields": ["agentview", "wrist", "robot_state", "language_instruction"],
        "loss_only_fields": ["oracle_feedback_horizon"],
        "groups": groups,
    }
    (staging / "records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    (staging / "training_labels.json").write_text(
        json.dumps({"schema_version": 1, "horizon": args.horizon, "records": labels}, indent=2, sort_keys=True) + "\n"
    )
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (staging / "quality_report.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
    staging.rename(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), **quality}, sort_keys=True))
    if not quality["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
