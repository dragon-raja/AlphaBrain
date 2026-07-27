from __future__ import annotations

import argparse
import hashlib
import json
import os
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from collect_libero_bind_teacher import (
    load_state_bank,
    object_grasped,
    robot_state,
    upright_image,
)


SOURCE_OBJECTS = (
    "red_coffee_mug_1",
    "porcelain_mug_1",
    "white_yellow_mug_1",
)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def policy_example(
    observation: Mapping[str, Any],
    language: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    example = {
        "image": [
            upright_image(observation["agentview_image"]),
            upright_image(observation["robot0_eye_in_hand_image"]),
        ],
        "lang": language,
        "language": language,
        "state": robot_state(observation),
    }
    if metadata is not None:
        example.update(metadata)
    return example


class RemotePolicy:
    def __init__(self, socket_path: Path) -> None:
        self.connection = Client(
            str(socket_path),
            family="AF_UNIX",
            authkey=b"fresh-vla-local",
        )
        handshake = self.connection.recv()
        self.horizon = int(handshake["horizon"])
        self.identity = dict(handshake)

    def predict(
        self,
        observation: Mapping[str, Any],
        language: str,
        *,
        seed: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        self.connection.send(
            {
                "op": "predict",
                "seed": int(seed),
                "example": policy_example(observation, language, metadata),
            }
        )
        response = self.connection.recv()
        if "error" in response:
            raise RuntimeError(f"remote Pi0.5 inference failed: {response['error']}")
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.shape != (self.horizon, 7) or not np.all(np.isfinite(actions)):
            raise ValueError(f"invalid Pi0.5 action chunk: {actions.shape}")
        return np.clip(actions, -1.0, 1.0)

    def close(self) -> None:
        try:
            self.connection.send({"op": "close"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        self.connection.close()


def subgoal_state(
    observation: Mapping[str, Any],
    *,
    source_object: str,
    target_object: str,
    initial_source_z: float,
    source_grasped: bool,
    wrong_source_grasped: bool,
) -> dict[str, bool]:
    source = np.asarray(observation[f"{source_object}_pos"], dtype=np.float64)
    target = np.asarray(observation[f"{target_object}_pos"], dtype=np.float64)
    lifted = bool(source[2] - initial_source_z >= 0.015)
    transported = bool(lifted and np.linalg.norm(source[:2] - target[:2]) <= 0.08)
    return {
        "source_grasp": bool(source_grasped),
        "wrong_source_grasp": bool(wrong_source_grasped),
        "lift": lifted,
        "transport": transported,
    }


def run_episode(
    env: Any,
    policy: RemotePolicy,
    initial_state: np.ndarray,
    edge: Mapping[str, Any],
    *,
    execution_horizon: int,
    max_steps: int,
    seed: int,
    record_frames: bool,
    environment_setup: Callable[[Any], Mapping[str, Any]] | None = None,
    episode_setup: (
        Callable[
            [Any, Mapping[str, Any]],
            tuple[Mapping[str, Any], Mapping[str, Any]],
        ]
        | None
    ) = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    setup_metadata: dict[str, Any] = {}
    env.reset()
    if environment_setup is not None:
        setup_metadata.update(environment_setup(env))
    observation = env.set_init_state(np.asarray(initial_state))
    for _ in range(8):
        observation, _, _, _ = env.step(np.asarray([0.0] * 6 + [-1.0], np.float32))
    if episode_setup is not None:
        observation, observation_metadata = episode_setup(env, observation)
        setup_metadata.update(observation_metadata)
    camera_policy_metadata = {
        key: setup_metadata[key]
        for key in (
            "camera_intrinsics",
            "camera_to_world_opencv",
            "camera_intrinsics_by_view",
            "camera_to_world_opencv_by_view",
        )
        if key in setup_metadata
    }
    source_object = str(edge["source_object"])
    target_object = str(edge["target_object"])
    initial_source_z = float(observation[f"{source_object}_pos"][2])
    source_grasped = object_grasped(env, source_object)
    wrong_source_grasped = False
    lifted = False
    transported = False
    frames = []
    if record_frames:
        frames.append(
            np.concatenate(
                [
                    upright_image(observation["agentview_image"]),
                    upright_image(observation["robot0_eye_in_hand_image"]),
                ],
                axis=1,
            )
        )

    success = bool(env.check_success())
    steps = 0
    inference_calls = 0
    while steps < max_steps and not success:
        chunk = policy.predict(
            observation,
            str(edge["language_instruction"]),
            seed=stable_seed(seed, edge["edge_id"], execution_horizon, inference_calls),
            metadata=camera_policy_metadata or None,
        )
        inference_calls += 1
        for action in chunk[:execution_horizon]:
            observation, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
            steps += 1
            source_grasped = source_grasped or object_grasped(env, source_object)
            wrong_source_grasped = wrong_source_grasped or any(
                object_grasped(env, object_name)
                for object_name in SOURCE_OBJECTS
                if object_name != source_object
            )
            current = subgoal_state(
                observation,
                source_object=source_object,
                target_object=target_object,
                initial_source_z=initial_source_z,
                source_grasped=source_grasped,
                wrong_source_grasped=wrong_source_grasped,
            )
            lifted = lifted or current["lift"]
            transported = transported or current["transport"]
            success = bool(env.check_success())
            if record_frames:
                frames.append(
                    np.concatenate(
                        [
                            upright_image(observation["agentview_image"]),
                            upright_image(observation["robot0_eye_in_hand_image"]),
                        ],
                        axis=1,
                    )
                )
            if success or steps >= max_steps:
                break

    source = np.asarray(observation[f"{source_object}_pos"], dtype=np.float64)
    target = np.asarray(observation[f"{target_object}_pos"], dtype=np.float64)
    completed = (source_grasped, lifted, transported, success)
    row = {
        "success": success,
        "source_selection_success": source_grasped,
        "wrong_source_grasp": wrong_source_grasped,
        "lift_success": lifted,
        "transport_success": transported,
        "target_placement_success": success,
        "progress": float(np.mean(completed)),
        "completion_steps": steps,
        "inference_calls": inference_calls,
        "final_source_target_xy_distance": float(np.linalg.norm(source[:2] - target[:2])),
        **setup_metadata,
    }
    frame_array = np.asarray(frames) if record_frames else None
    return row, frame_array


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_state_indices(value: str | None, manifest: Mapping[str, Any], split: str) -> list[int]:
    if value:
        indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        indices = [
            int(row["canonical_state_index"])
            for row in manifest["states"]
            if row["split"] == split
        ]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("state indices must be a non-empty unique list")
    return indices


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-K LIBERO-Bind closed-loop evaluation")
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--state-indices")
    parser.add_argument("--edges", default="all")
    parser.add_argument("--execution-horizons", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--frame-dir", type=Path)
    parser.add_argument("--frame-episodes-per-edge", type=int, default=0)
    return parser.parse_args(args)


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {args.output}")
    if any(value not in (1, 2, 3) for value in args.execution_horizons):
        raise ValueError("execution horizons must be selected from 1, 2, 3")
    manifest = json.loads((args.suite_root / "manifest.json").read_text())
    states = load_state_bank(Path(manifest["canonical_init_states"]))
    state_indices = parse_state_indices(args.state_indices, manifest, args.split)
    edges = manifest["edges"]
    if args.edges != "all":
        requested = {part.strip() for part in args.edges.split(",") if part.strip()}
        edges = [edge for edge in edges if edge["edge_id"] in requested]
        if {edge["edge_id"] for edge in edges} != requested:
            raise KeyError("requested an unknown LIBERO-Bind edge")
    policy = RemotePolicy(args.policy_socket)
    if max(args.execution_horizons) > policy.horizon:
        raise ValueError("execution horizon exceeds policy chunk")

    rows = []
    expected = len(edges) * len(state_indices) * len(args.execution_horizons)
    partial = args.output.with_name(f"{args.output.stem}.partial{args.output.suffix}")
    try:
        for edge in edges:
            env = OffScreenRenderEnv(
                bddl_file_name=edge["bddl"],
                camera_heights=224,
                camera_widths=224,
                horizon=args.max_steps + 16,
                ignore_done=True,
            )
            try:
                env.seed(args.seed)
                for execution_horizon in args.execution_horizons:
                    for state_position, state_index in enumerate(state_indices):
                        record = (
                            args.frame_dir is not None
                            and state_position < args.frame_episodes_per_edge
                        )
                        metrics, frames = run_episode(
                            env,
                            policy,
                            states[state_index],
                            edge,
                            execution_horizon=execution_horizon,
                            max_steps=args.max_steps,
                            seed=args.seed,
                            record_frames=record,
                        )
                        row = {
                            "edge_id": edge["edge_id"],
                            "source_id": edge["source_id"],
                            "target_id": edge["target_id"],
                            "action_supervised": bool(edge["action_supervised"]),
                            "canonical_state_index": state_index,
                            "split": args.split,
                            "execution_horizon": execution_horizon,
                            **metrics,
                        }
                        if frames is not None:
                            args.frame_dir.mkdir(parents=True, exist_ok=True)
                            frame_file = (
                                f"{edge['edge_id']}--state-{state_index:02d}--k{execution_horizon}.npz"
                            )
                            np.savez_compressed(args.frame_dir / frame_file, frames=frames)
                            row["frame_file"] = frame_file
                        rows.append(row)
                        payload = {
                            "status": "partial",
                            "expected_episode_count": expected,
                            "policy_identity": policy.identity,
                            "rows": rows,
                        }
                        _atomic_write(partial, payload)
                        print(json.dumps(row, sort_keys=True), flush=True)
            finally:
                env.close()
    finally:
        policy.close()

    payload = {
        "schema_version": 1,
        "status": "complete",
        "suite": str(args.suite_root),
        "split": args.split,
        "state_indices": state_indices,
        "execution_horizons": args.execution_horizons,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "expected_episode_count": expected,
        "policy_identity": policy.identity,
        "rows": rows,
    }
    _atomic_write(args.output, payload)
    partial.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "episode_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
