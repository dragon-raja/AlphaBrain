from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np


HF_REPO = "physical-intelligence/libero"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gripper_feedback_horizon(actions: np.ndarray, action_horizon: int) -> np.ndarray:
    """Distance to the next gripper transition, including the transition action."""
    if actions.ndim != 2 or actions.shape[1] < 1:
        raise ValueError(f"expected [frames, action_dim] actions, got {actions.shape}")
    binary_gripper = actions[:, -1] > 0
    transitions = np.flatnonzero(binary_gripper[1:] != binary_gripper[:-1]) + 1
    labels = np.full(actions.shape[0], action_horizon, dtype=np.int16)
    for frame_index in range(actions.shape[0]):
        future = transitions[transitions >= frame_index]
        if future.size:
            labels[frame_index] = min(action_horizon, int(future[0] - frame_index + 1))
    return labels


def modality_metadata() -> dict:
    state_names = ("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper")
    action_names = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    return {
        "state": {
            name: {"start": index, "end": index + 1, "dtype": "float32", "original_key": "state"}
            for index, name in enumerate(state_names)
        },
        "action": {
            name: {"start": index, "end": index + 1, "dtype": "float32", "original_key": "actions"}
            for index, name in enumerate(action_names)
        },
        "video": {
            "primary_image": {"original_key": "image"},
            "wrist_image": {"original_key": "wrist_image"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
            "feedback_horizon": {"original_key": "feedback_horizon"},
        },
    }


def build_manifest(args: argparse.Namespace) -> None:
    repo = json.loads(args.repo_metadata.read_text())
    info = json.loads(args.info_metadata.read_text())
    episodes = read_jsonl(args.episodes_metadata)
    tasks = read_jsonl(args.tasks_metadata)
    task_records = [record for record in tasks if record["task"] == args.task]
    if len(task_records) != 1:
        raise ValueError(f"expected one exact task match, found {len(task_records)}")

    file_by_episode = {}
    for sibling in repo["siblings"]:
        path = sibling["rfilename"]
        if not path.startswith("data/") or "episode_" not in path:
            continue
        episode_index = int(path.rsplit("episode_", 1)[1].split(".", 1)[0])
        file_by_episode[episode_index] = {
            "repo_path": path,
            "size": sibling.get("lfs", {}).get("size", sibling.get("size", 0)),
            "sha256": sibling.get("lfs", {}).get("sha256"),
            "url": f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{quote(path)}",
        }

    candidates = []
    for episode in episodes:
        if args.task not in episode.get("tasks", []):
            continue
        source = file_by_episode.get(episode["episode_index"])
        if source is None:
            continue
        candidates.append({**episode, **source})
    candidates.sort(key=lambda record: (record["size"], record["episode_index"]))
    selected = candidates[: args.num_episodes]
    if len(selected) != args.num_episodes:
        raise ValueError(f"requested {args.num_episodes} episodes, found {len(selected)}")

    payload = {
        "source_repo": HF_REPO,
        "task": task_records[0],
        "action_horizon": args.action_horizon,
        "selection": "smallest episodes for plumbing smoke",
        "total_bytes": sum(record["size"] for record in selected),
        "episodes": selected,
        "info": info,
        "modality": modality_metadata(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.download_list:
        args.download_list.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{record['size']}\t{record['repo_path']}\t{record['url']}" for record in selected]
        args.download_list.write_text("\n".join(lines) + "\n")
    print(f"selected={len(selected)} bytes={payload['total_bytes']} manifest={args.output}")


def materialize(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest = json.loads(args.manifest.read_text())
    output_root = args.output_root
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    selected_episodes = []

    for episode in manifest["episodes"]:
        source = args.raw_root / episode["repo_path"]
        if not source.exists():
            raise FileNotFoundError(f"missing downloaded episode: {source}")
        if source.stat().st_size != episode["size"]:
            raise ValueError(f"size mismatch for {source}")
        expected_sha256 = episode.get("sha256")
        if expected_sha256:
            digest = sha256_file(source)
            if digest != expected_sha256:
                raise ValueError(f"sha256 mismatch for {source}")
        destination = output_root / episode["repo_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(source)
        actions = np.stack(table["actions"].to_pylist()).astype(np.float32)
        labels = gripper_feedback_horizon(actions, int(manifest["action_horizon"]))
        label_array = pa.array(labels, type=pa.int16())
        if "feedback_horizon" in table.column_names:
            table = table.set_column(table.column_names.index("feedback_horizon"), "feedback_horizon", label_array)
        else:
            table = table.append_column("feedback_horizon", label_array)
        pq.write_table(table, destination, compression="zstd")
        selected_episodes.append(
            {"episode_index": episode["episode_index"], "tasks": episode["tasks"], "length": episode["length"]}
        )

    info = dict(manifest["info"])
    info["total_episodes"] = len(selected_episodes)
    info["total_frames"] = sum(episode["length"] for episode in selected_episodes)
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{len(selected_episodes)}"}
    info["features"] = dict(info["features"])
    info["features"]["feedback_horizon"] = {"dtype": "int16", "shape": [1], "names": None}

    (output_root / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    (output_root / "meta" / "modality.json").write_text(json.dumps(manifest["modality"], indent=2) + "\n")
    (output_root / "meta" / "tasks.jsonl").write_text(json.dumps(manifest["task"]) + "\n")
    episode_lines = [json.dumps(episode) for episode in selected_episodes]
    (output_root / "meta" / "episodes.jsonl").write_text("\n".join(episode_lines) + "\n")
    print(f"materialized={output_root} episodes={len(selected_episodes)} frames={info['total_frames']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a tiny FRESH-compatible subset of physical-intelligence/libero")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--repo-metadata", type=Path, required=True)
    manifest_parser.add_argument("--info-metadata", type=Path, required=True)
    manifest_parser.add_argument("--episodes-metadata", type=Path, required=True)
    manifest_parser.add_argument("--tasks-metadata", type=Path, required=True)
    manifest_parser.add_argument("--task", required=True)
    manifest_parser.add_argument("--num-episodes", type=int, default=2)
    manifest_parser.add_argument("--action-horizon", type=int, default=10)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--download-list", type=Path)
    manifest_parser.set_defaults(func=build_manifest)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--manifest", type=Path, required=True)
    materialize_parser.add_argument("--raw-root", type=Path, required=True)
    materialize_parser.add_argument("--output-root", type=Path, required=True)
    materialize_parser.set_defaults(func=materialize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
