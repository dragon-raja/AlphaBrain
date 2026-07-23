from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from build_libero_bind_training_view import transport_anchor_index
from collect_libero_bind_teacher import (
    load_state_bank,
    object_grasped,
    robot_state,
    upright_image,
)
from evaluate_libero_bind_closed_loop import (
    SOURCE_OBJECTS,
    RemotePolicy,
    stable_seed,
    subgoal_state,
)


def select_prefix_edge(
    edges: Sequence[Mapping[str, Any]], heldout_edge: Mapping[str, Any]
) -> Mapping[str, Any]:
    candidates = [
        edge
        for edge in edges
        if bool(edge["action_supervised"])
        and str(edge["source_id"]) == str(heldout_edge["source_id"])
        and str(edge["target_id"]) != str(heldout_edge["target_id"])
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one supervised source-matched prefix for {heldout_edge['edge_id']}, "
            f"found {[edge['edge_id'] for edge in candidates]}"
        )
    return candidates[0]


def remaining_policy_budget(total_budget: int, replayed_actions: int) -> int:
    remaining = total_budget - replayed_actions
    if total_budget <= 0 or remaining <= 0:
        raise ValueError("teacher prefix must leave a positive policy action budget")
    return remaining


def _frame(observation: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            upright_image(observation["agentview_image"]),
            upright_image(observation["robot0_eye_in_hand_image"]),
        ],
        axis=1,
    )


def run_target_handoff(
    env: Any,
    policy: RemotePolicy,
    initial_state: np.ndarray,
    heldout_edge: Mapping[str, Any],
    prefix_episode: Mapping[str, np.ndarray],
    *,
    prefix_edge_id: str,
    execution_horizon: int,
    total_action_budget: int,
    seed: int,
    record_frames: bool,
) -> tuple[dict[str, Any], np.ndarray | None]:
    phases = np.asarray(prefix_episode["phase"]).astype(str)
    handoff_frame = transport_anchor_index(phases)
    prefix_actions = np.asarray(prefix_episode["actions"][:handoff_frame], np.float32)
    policy_budget = remaining_policy_budget(total_action_budget, len(prefix_actions))

    env.reset()
    observation = env.set_init_state(np.asarray(initial_state))
    for _ in range(8):
        observation, _, _, _ = env.step(
            np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
        )
    source_object = str(heldout_edge["source_object"])
    target_object = str(heldout_edge["target_object"])
    initial_source_z = float(observation[f"{source_object}_pos"][2])
    frames = [_frame(observation)] if record_frames else []

    wrong_source_grasped = False
    for action in prefix_actions:
        observation, _, _, _ = env.step(action)
        wrong_source_grasped = wrong_source_grasped or any(
            object_grasped(env, object_name)
            for object_name in SOURCE_OBJECTS
            if object_name != source_object
        )
        if record_frames:
            frames.append(_frame(observation))

    replay_state = robot_state(observation)
    recorded_state = np.asarray(prefix_episode["robot_state"][handoff_frame])
    replay_state_max_abs_error = float(np.max(np.abs(replay_state - recorded_state)))
    replay_agent_mse = float(
        np.square(
            upright_image(observation["agentview_image"]).astype(np.float32)
            - np.asarray(prefix_episode["agentview"][handoff_frame], np.float32)
        ).mean()
    )
    if replay_state_max_abs_error > 1e-4:
        raise RuntimeError(
            f"teacher prefix replay drifted at target handoff: {replay_state_max_abs_error}"
        )

    source_grasped = object_grasped(env, source_object)
    lifted = bool(
        float(observation[f"{source_object}_pos"][2]) - initial_source_z >= 0.015
    )
    transported = False
    success = bool(env.check_success())
    policy_steps = 0
    inference_calls = 0
    while policy_steps < policy_budget and not success:
        chunk = policy.predict(
            observation,
            str(heldout_edge["language_instruction"]),
            seed=stable_seed(
                seed,
                heldout_edge["edge_id"],
                "target_handoff",
                execution_horizon,
                inference_calls,
            ),
        )
        inference_calls += 1
        for action in chunk[:execution_horizon]:
            observation, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
            policy_steps += 1
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
                frames.append(_frame(observation))
            if success or policy_steps >= policy_budget:
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
        "completion_steps": len(prefix_actions) + policy_steps,
        "prefix_actions_replayed": len(prefix_actions),
        "policy_steps": policy_steps,
        "policy_action_budget": policy_budget,
        "inference_calls": inference_calls,
        "prefix_edge_id": prefix_edge_id,
        "handoff_decision_point": "target_select",
        "replay_state_max_abs_error": replay_state_max_abs_error,
        "replay_agent_mse": replay_agent_mse,
        "final_source_target_xy_distance": float(
            np.linalg.norm(source[:2] - target[:2])
        ),
    }
    return row, np.asarray(frames) if record_frames else None


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc teacher-prefix target handoff diagnostic"
    )
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--edges", default="all_action_free")
    parser.add_argument("--execution-horizon", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--total-action-budget", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--frame-dir", type=Path)
    return parser.parse_args(args)


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {args.output}")
    suite = json.loads((args.suite_root / "manifest.json").read_text())
    collection = json.loads((args.collection_root / "manifest.json").read_text())
    states = load_state_bank(Path(suite["canonical_init_states"]))
    all_edges = list(suite["edges"])
    if args.edges == "all_action_free":
        edges = [edge for edge in all_edges if not bool(edge["action_supervised"])]
    else:
        requested = {value.strip() for value in args.edges.split(",") if value.strip()}
        edges = [edge for edge in all_edges if edge["edge_id"] in requested]
        if {edge["edge_id"] for edge in edges} != requested:
            raise KeyError("requested an unknown LIBERO-Bind edge")
    collection_rows = {
        (str(row["edge_id"]), int(row["canonical_state_index"])): row
        for row in collection["rows"]
        if bool(row.get("success"))
    }
    policy = RemotePolicy(args.policy_socket)
    rows = []
    try:
        for edge in edges:
            prefix_edge = select_prefix_edge(all_edges, edge)
            env = OffScreenRenderEnv(
                bddl_file_name=edge["bddl"],
                camera_heights=224,
                camera_widths=224,
                horizon=args.total_action_budget + 16,
                ignore_done=True,
            )
            try:
                env.seed(args.seed)
                for state_index in args.state_indices:
                    source = collection_rows[(prefix_edge["edge_id"], state_index)]
                    with np.load(
                        args.collection_root / source["episode_file"], allow_pickle=False
                    ) as archive:
                        episode = {key: np.asarray(archive[key]) for key in archive.files}
                    metrics, frames = run_target_handoff(
                        env,
                        policy,
                        states[state_index],
                        edge,
                        episode,
                        prefix_edge_id=str(prefix_edge["edge_id"]),
                        execution_horizon=args.execution_horizon,
                        total_action_budget=args.total_action_budget,
                        seed=args.seed,
                        record_frames=args.frame_dir is not None,
                    )
                    row = {
                        "edge_id": edge["edge_id"],
                        "source_id": edge["source_id"],
                        "target_id": edge["target_id"],
                        "action_supervised": False,
                        "canonical_state_index": state_index,
                        "split": "diagnostic_train_state",
                        "execution_horizon": args.execution_horizon,
                        **metrics,
                    }
                    if frames is not None:
                        args.frame_dir.mkdir(parents=True, exist_ok=True)
                        frame_file = (
                            f"{edge['edge_id']}--state-{state_index:02d}"
                            f"--k{args.execution_horizon}.npz"
                        )
                        np.savez_compressed(args.frame_dir / frame_file, frames=frames)
                        row["frame_file"] = frame_file
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
            finally:
                env.close()
    finally:
        policy.close()

    payload = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic_only": True,
        "post_hoc": True,
        "claim_boundary": (
            "Teacher-prefix handoff tests target-stage physical execution only; "
            "it is not an end-to-end migration result."
        ),
        "suite": str(args.suite_root),
        "collection": str(args.collection_root),
        "state_indices": args.state_indices,
        "execution_horizons": [args.execution_horizon],
        "max_steps": args.total_action_budget,
        "seed": args.seed,
        "policy_identity": policy.identity,
        "rows": rows,
    }
    _atomic_write(args.output, payload)
    print(json.dumps({"output": str(args.output), "episode_count": len(rows)}))


if __name__ == "__main__":
    main()
