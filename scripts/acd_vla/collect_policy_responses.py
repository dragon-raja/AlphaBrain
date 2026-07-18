from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.branch_vla.gate0_common import observation_feature


LANGUAGE = "put the cream cheese in the bowl"
DEVELOPMENT_SPLITS = ("train", "val")
FORBIDDEN_PARTS = {"test", "tests", "confirmation", "confirm", "sealed"}
REQUIRED_KEYS = ("agentview", "wrist", "robot_state")


def assert_unsealed(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if lowered & FORBIDDEN_PARTS or any("confirmation" in part for part in lowered):
        raise ValueError(f"refusing sealed path: {path}")


def validate_splits(splits: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(value) for value in splits))
    if not selected or any(value not in DEVELOPMENT_SPLITS for value in selected):
        raise ValueError(f"only development splits are allowed: {selected}")
    return selected


def stable_seed(base: int, pair_id: str) -> int:
    digest = hashlib.sha256(f"acd-vla:{base}:{pair_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_episode(path: Path) -> dict[str, np.ndarray]:
    assert_unsealed(path)
    with np.load(path, allow_pickle=False) as handle:
        return {key: np.asarray(handle[key]) for key in REQUIRED_KEYS}


def policy_example(episode: Mapping[str, np.ndarray], index: int) -> dict[str, Any]:
    return {
        "image": [
            np.asarray(episode["agentview"][index], dtype=np.uint8),
            np.asarray(episode["wrist"][index], dtype=np.uint8),
        ],
        "lang": LANGUAGE,
        "language": LANGUAGE,
        "state": np.asarray(episode["robot_state"][index], dtype=np.float32),
    }


class StoredObservationPolicy:
    def __init__(self, socket_path: Path) -> None:
        self.connection = Client(str(socket_path), family="AF_UNIX", authkey=b"fresh-vla-local")
        handshake = self.connection.recv()
        self.horizon = int(handshake["horizon"])
        self.identity = {
            "checkpoint_realpath": str(handshake.get("checkpoint_realpath", "")),
            "model_size_bytes": int(handshake.get("model_size_bytes", 0)),
            "torch_version": handshake.get("torch_version"),
            "cuda_version": handshake.get("cuda_version"),
            "device_name": handshake.get("device_name"),
        }

    def predict(self, example: Mapping[str, Any], seed: int) -> tuple[np.ndarray, float]:
        self.connection.send({"op": "predict", "seed": int(seed), "example": dict(example)})
        response = self.connection.recv()
        if "error" in response:
            raise RuntimeError(f"remote Pi0.5 inference failed: {response['error']}")
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.shape != (self.horizon, 7) or not np.all(np.isfinite(actions)):
            raise RuntimeError(f"invalid remote Pi0.5 actions: {actions.shape}")
        return actions, float(response.get("predict_action_wall_seconds", 0.0))

    def close(self) -> None:
        try:
            self.connection.send({"op": "close"})
        finally:
            self.connection.close()


def record_paths(output_root: Path, pair_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", pair_id):
        raise ValueError(f"unsafe pair id: {pair_id!r}")
    return output_root / "records" / f"{pair_id}.npz", output_root / "records" / f"{pair_id}.json"


def validate_existing_record(array_path: Path, metadata_path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not array_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"partial record cannot be resumed: {array_path.stem}")
    metadata = json.loads(metadata_path.read_text())
    for key in ("pair_id", "split", "source_id", "feedback_reveal_time", "inference_seed"):
        if metadata.get(key) != expected.get(key):
            raise RuntimeError(f"resume mismatch for {expected['pair_id']} key={key}")
    with np.load(array_path, allow_pickle=False) as handle:
        for key in (
            "pre_actions",
            "attached_actions",
            "slipped_actions",
            "pre_feature",
            "attached_post_feature",
            "slipped_post_feature",
        ):
            if key not in handle or array_hash(handle[key]) != metadata["array_hashes"][key]:
                raise RuntimeError(f"corrupt resume record {expected['pair_id']} array={key}")
    return metadata


def selected_groups(manifest: Mapping[str, Any], splits: Sequence[str]) -> list[dict[str, Any]]:
    allowed = set(validate_splits(splits))
    groups = [dict(group) for group in manifest["groups"] if str(group["split"]) in allowed]
    if any(group["split"] not in DEVELOPMENT_SPLITS for group in groups):
        raise AssertionError("non-development group selected")
    return sorted(groups, key=lambda group: str(group["pair_id"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect frozen Pi0.5 counterfactual policy responses")
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--splits", nargs="+", default=DEVELOPMENT_SPLITS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-groups", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = validate_splits(args.splits)
    assert_unsealed(args.episode_root)
    assert_unsealed(args.output_root)
    if args.max_groups is not None and args.max_groups < 1:
        raise ValueError("max-groups must be positive")
    if args.output_root.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume after inspection: {args.output_root}")

    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    groups = selected_groups(manifest, splits)
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    if not groups:
        raise ValueError("no development groups selected")

    split_sources = {
        split: {int(group["source_initial_state_index"]) for group in groups if group["split"] == split}
        for split in splits
    }
    if len(splits) == 2 and split_sources["train"] & split_sources["val"]:
        raise RuntimeError("train/val source initial states overlap")

    policy = StoredObservationPolicy(args.policy_socket)
    run_config = {
        "schema_version": 1,
        "experiment": "acd_vla_gate0_policy_response_collection",
        "episode_root": str(args.episode_root.resolve()),
        "policy_seed": int(args.policy_seed),
        "splits": list(splits),
        "policy_identity": policy.identity,
        "policy_horizon": policy.horizon,
        "image_feature_size": 8,
        "test_episode_files_opened": 0,
        "confirmation_paths_opened": 0,
    }
    config_path = args.output_root / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != run_config:
            raise RuntimeError("resume run_config mismatch")
    else:
        atomic_json(config_path, run_config)

    records = []
    total_inference_seconds = 0.0
    try:
        for index, group in enumerate(groups, start=1):
            pair_id = str(group["pair_id"])
            split = str(group["split"])
            source_id = int(group["source_initial_state_index"])
            reveal = int(group["feedback_reveal_time"])
            inference_seed = stable_seed(args.policy_seed, pair_id)
            expected = {
                "pair_id": pair_id,
                "split": split,
                "source_id": source_id,
                "feedback_reveal_time": reveal,
                "inference_seed": inference_seed,
            }
            array_path, metadata_path = record_paths(args.output_root, pair_id)
            if array_path.exists() or metadata_path.exists():
                metadata = validate_existing_record(array_path, metadata_path, expected)
                records.append(metadata)
                print(json.dumps({"completed": index, "total": len(groups), "pair_id": pair_id, "resumed": True}), flush=True)
                continue

            attached = load_episode(args.episode_root / str(group["episode_files"]["attached"]))
            slipped = load_episode(args.episode_root / str(group["episode_files"]["slipped"]))
            if reveal < 1:
                raise RuntimeError(f"invalid feedback time for {pair_id}: {reveal}")
            for key in REQUIRED_KEYS:
                if not np.array_equal(attached[key][reveal - 1], slipped[key][reveal - 1]):
                    raise RuntimeError(f"pre-feedback twin leakage for {pair_id}: {key}")
            post_image_delta = max(
                int(np.abs(attached[key][reveal].astype(np.int16) - slipped[key][reveal].astype(np.int16)).max())
                for key in ("agentview", "wrist")
            )
            if post_image_delta <= 0:
                raise RuntimeError(f"feedback is not visually observable for {pair_id}")

            pre_actions, pre_seconds = policy.predict(policy_example(attached, reveal - 1), inference_seed)
            attached_actions, attached_seconds = policy.predict(policy_example(attached, reveal), inference_seed)
            slipped_actions, slipped_seconds = policy.predict(policy_example(slipped, reveal), inference_seed)
            inference_seconds = pre_seconds + attached_seconds + slipped_seconds
            total_inference_seconds += inference_seconds
            arrays = {
                "pre_actions": pre_actions,
                "attached_actions": attached_actions,
                "slipped_actions": slipped_actions,
                "pre_feature": observation_feature(
                    attached["agentview"][reveal - 1], attached["wrist"][reveal - 1], attached["robot_state"][reveal - 1]
                ).astype(np.float32),
                "attached_post_feature": observation_feature(
                    attached["agentview"][reveal], attached["wrist"][reveal], attached["robot_state"][reveal]
                ).astype(np.float32),
                "slipped_post_feature": observation_feature(
                    slipped["agentview"][reveal], slipped["wrist"][reveal], slipped["robot_state"][reveal]
                ).astype(np.float32),
            }
            metadata = {
                **expected,
                "record_file": str(array_path.relative_to(args.output_root)),
                "post_feedback_image_max_delta": post_image_delta,
                "inference_wall_seconds": inference_seconds,
                "array_hashes": {key: array_hash(value) for key, value in arrays.items()},
            }
            atomic_npz(array_path, **arrays)
            atomic_json(metadata_path, metadata)
            records.append(metadata)
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(groups),
                        "pair_id": pair_id,
                        "split": split,
                        "inference_wall_seconds": round(inference_seconds, 3),
                    }
                ),
                flush=True,
            )
    finally:
        policy.close()

    result = {
        **run_config,
        "status": "complete",
        "group_count": len(records),
        "split_group_counts": {
            split: sum(record["split"] == split for record in records) for split in splits
        },
        "split_source_counts": {split: len(values) for split, values in split_sources.items()},
        "train_val_source_overlap": len(split_sources.get("train", set()) & split_sources.get("val", set())),
        "total_inference_wall_seconds_new_records": total_inference_seconds,
        "records": records,
    }
    atomic_json(args.output_root / "manifest.json", result)
    print(json.dumps({"status": "complete", "group_count": len(records), "output_root": str(args.output_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
