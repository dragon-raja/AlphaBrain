#!/usr/bin/env python3
"""Paired full closed-loop evaluation from official LIBERO HDF5 states."""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def configure_imports() -> None:
    scripts = Path(__file__).resolve().parents[1]
    for path in (scripts / "dsol_paper1", scripts / "cabi_vla"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def masked_policy_observation(
    observation: Mapping[str, Any],
    *,
    prompt: str,
    resize_size: int,
    eval_seed: int,
    camera_calibration: Mapping[str, Any],
    sensor_control: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    from evaluate_pi05_libero_plus_views import prepare_policy_observation

    example, agent, wrist = prepare_policy_observation(
        observation,
        prompt=prompt,
        resize_size=resize_size,
        eval_seed=eval_seed,
        camera_calibration=camera_calibration,
    )
    if sensor_control == "external_only":
        wrist = np.zeros_like(wrist)
        example["observation/wrist_image"] = wrist
    elif sensor_control == "wrist_only":
        agent = np.zeros_like(agent)
        example["observation/image"] = agent
    elif sensor_control != "both":
        raise ValueError(f"unsupported sensor control: {sensor_control}")
    return example, agent, wrist


def deployed_camera_names(sensor_control: str) -> tuple[str, ...]:
    if sensor_control == "both":
        return ("agentview", "robot0_eye_in_hand")
    if sensor_control == "external_only":
        return ("agentview",)
    if sensor_control == "wrist_only":
        return ("robot0_eye_in_hand",)
    raise ValueError(f"unsupported sensor control: {sensor_control}")


def run_episode(
    spec: Mapping[str, Any],
    *,
    runtime: Path,
    config_root: Path,
    client: Any,
    replan_steps: int,
    wait_steps: int,
    resize_size: int,
    seed: int,
    save_video: bool,
    video_dir: Path,
    render_gpu: int,
) -> tuple[dict[str, Any], Any]:
    import h5py

    configure_imports()
    from audit_libero_hdf5_restore import _configure_runtime, _decode, _rewrite_model_paths

    hdf5_path = Path(spec["hdf5"]).resolve()
    _configure_runtime(runtime, hdf5_path.parent.parent, config_root)
    from libero.libero.envs import OffScreenRenderEnv
    from libero_camera_pose import capture_camera_reference, install_camera_pose
    from libero_visibility import task_entity_visibility
    from scan_libero_hdf5_views import _install_look_away
    from evaluate_pi05_libero_plus_views import (
        agentview_camera_calibration,
        clean_task_prompt,
        encode_av1_video,
        initial_image_metrics,
        physics_state_sha256,
        stable_seed,
    )

    with h5py.File(hdf5_path, "r") as handle:
        data = handle["data"]
        demo = data[str(spec["demo_name"])]
        state = np.asarray(demo["states"][int(spec.get("source_state_index", 0))])
        model_xml, rewrites = _rewrite_model_paths(_decode(demo.attrs["model_file"]), runtime)
        bddl_name = Path(_decode(data.attrs["bddl_file_name"])).name
        problem_info = json.loads(_decode(data.attrs["problem_info"]))
        prompt = str(problem_info["language_instruction"])
    suite = str(spec["suite"])
    bddl = runtime / "libero" / "libero" / "bddl_files" / suite / bddl_name
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_names=("agentview", "robot0_eye_in_hand"),
        camera_heights=256,
        camera_widths=256,
        render_gpu_device_id=render_gpu,
    )
    frames: list[np.ndarray] = []
    action_plan: collections.deque[np.ndarray] = collections.deque()
    inference_calls = 0
    success = False
    pair_seed = stable_seed(str(spec["pair_key"]), seed=seed)
    sensor_control = str(spec["sensor_control"])
    try:
        env.seed(pair_seed)
        env.reset()
        env.reset_from_xml_string(model_xml)
        observation = env.set_init_state(state)
        catalog_path = Path(spec["catalog"]) if "catalog" in spec else None
        table_plane_z = 0.0
        if catalog_path is not None:
            table_plane_z = float(json.loads(catalog_path.read_text())["table_plane_z"])
        reference = capture_camera_reference(
            env,
            camera_name="agentview",
            table_plane_z=table_plane_z,
        )
        camera_metadata = None
        pose = spec.get("pose")
        if pose is not None:
            if pose.get("orientation_mode") == "relative_look_away":
                camera_metadata = _install_look_away(env, reference, pose)
            else:
                camera_metadata = install_camera_pose(env, reference, pose)
            env.env._update_observables(force=True)
            observation = env.env._get_observations()
        for _ in range(wait_steps):
            observation, _, done, _ = env.step(DUMMY_ACTION)
            if done:
                success = True
                break
        camera_calibration = agentview_camera_calibration(env)
        initial_example, initial_agent, initial_wrist = masked_policy_observation(
            observation,
            prompt=prompt,
            resize_size=resize_size,
            eval_seed=pair_seed,
            camera_calibration=camera_calibration,
            sensor_control=sensor_control,
        )
        physics_sha256, physics_size = physics_state_sha256(env)
        entities = list(env.env.obj_of_interest)
        initial_visibility = task_entity_visibility(
            env,
            entity_names=entities,
            camera_names=deployed_camera_names(sensor_control),
            height=resize_size,
            width=resize_size,
        )
        initial_metrics = {
            "agent": initial_image_metrics(initial_agent),
            "wrist": initial_image_metrics(initial_wrist),
            "task_entity_visibility": initial_visibility,
            "physics_state_sha256": physics_sha256,
            "physics_state_size": physics_size,
        }
        max_steps = MAX_STEPS_BY_SUITE[suite]
        step = 0
        while not success and step < max_steps:
            if not action_plan:
                call_seed = stable_seed(
                    f"{spec['pair_key']}::policy_call::{inference_calls}", seed=seed
                )
                example, agent, wrist = masked_policy_observation(
                    observation,
                    prompt=prompt,
                    resize_size=resize_size,
                    eval_seed=call_seed,
                    camera_calibration=camera_calibration,
                    sensor_control=sensor_control,
                )
                chunk = np.asarray(client.infer(example)["actions"], dtype=np.float32)
                if len(chunk) < replan_steps or chunk.shape[1] != 7:
                    raise ValueError(f"invalid action chunk shape: {chunk.shape}")
                action_plan.extend(chunk[:replan_steps])
                inference_calls += 1
                if save_video:
                    frames.append(np.concatenate([agent, wrist], axis=1))
            observation, _, done, _ = env.step(action_plan.popleft())
            success = bool(done)
            step += 1
        if save_video:
            suffix = "success" if success else "failure"
            encode_av1_video(video_dir / f"{spec['episode_id']}--{suffix}.webm", frames)
        return (
            {
                **dict(spec),
                "status": "complete",
                "success": success,
                "completion_steps": step,
                "max_steps": max_steps,
                "wait_steps": wait_steps,
                "replan_steps": replan_steps,
                "inference_calls": inference_calls,
                "policy_noise_seed": pair_seed,
                "language": prompt,
                "clean_language": clean_task_prompt(Path(bddl_name).stem),
                "bddl_file": str(bddl),
                "asset_path_rewrites": rewrites,
                "camera_metadata": camera_metadata,
                "initial_metrics": initial_metrics,
            },
            env,
        )
    except Exception:
        env.close()
        raise


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--video-episodes", type=int, default=8)
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--episode-index", type=int)
    return parser.parse_args()


def selected_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    protocol = json.loads(args.protocol.read_text())
    specs = []
    for index, spec in enumerate(protocol["specs"]):
        if index % args.num_shards != args.shard_index:
            continue
        specs.append({**spec, "catalog": protocol["catalog"]})
    return specs[: args.max_episodes] if args.max_episodes is not None else specs


def main() -> None:
    args = parse_args()
    if not 1 <= args.replan_steps <= 10:
        raise ValueError("replan-steps must be in [1, 10]")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    specs = selected_specs(args)
    output = args.output_dir / f"episodes-shard-{args.shard_index:02d}.jsonl"
    completed = {
        json.loads(line)["episode_id"]
        for line in output.read_text().splitlines()
        if output.exists() and line.strip()
    } if output.exists() else set()
    if args.episode_index is None:
        for index, spec in enumerate(specs):
            if spec["episode_id"] in completed:
                continue
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--episode-index", str(index)],
                check=True,
            )
        return
    spec = specs[args.episode_index]
    if spec["episode_id"] in completed:
        return
    configure_imports()
    from audit_libero_hdf5_restore import _configure_runtime

    _configure_runtime(args.runtime.resolve(), Path(spec["hdf5"]).resolve().parents[1], args.config_root.resolve())
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    row, _environment = run_episode(
        spec,
        runtime=args.runtime.resolve(),
        config_root=args.config_root.resolve(),
        client=client,
        replan_steps=args.replan_steps,
        wait_steps=args.wait_steps,
        resize_size=args.resize_size,
        seed=args.seed,
        save_video=args.episode_index < args.video_episodes,
        video_dir=args.output_dir / "videos_av1",
        render_gpu=args.render_gpu,
    )
    append_jsonl(output, row)
    print(json.dumps({"episode_id": row["episode_id"], "condition": row["condition"], "success": row["success"]}, sort_keys=True), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
