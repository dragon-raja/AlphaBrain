from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from build_libero_plus_view_protocol import synthetic_camera_task_name
from libero_camera_pose import mujoco_camera_calibration


DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
POLICY_CAMERA_RESOLUTION = 224


def clean_task_prompt(base_task: str) -> str:
    """Recover the original LIBERO instruction without Plus metadata."""
    name = str(base_task)
    if not name:
        raise ValueError("base task must be non-empty")
    if name[0].isupper():
        name = re.sub(r"^.*?SCENE\d+_", "", name)
    return name.replace("_", " ")


def stable_seed(value: str, *, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}::{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def initial_image_metrics(image: np.ndarray) -> dict[str, float]:
    values = np.asarray(image, dtype=np.uint8)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("image must be uint8 HxWx3")
    gray = values.astype(np.float64).mean(axis=2)
    histogram = np.bincount((gray / 8).astype(np.int64).reshape(-1), minlength=32)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    edge = float(
        0.5
        * (
            np.mean(np.abs(np.diff(gray, axis=0)))
            + np.mean(np.abs(np.diff(gray, axis=1)))
        )
    )
    return {
        "entropy_32bin_bits": entropy,
        "mean_edge_strength": edge,
        "rgb_std": float(np.std(values.astype(np.float64))),
        "clipped_fraction": float(np.mean((values <= 2) | (values >= 253))),
    }


def simulator_visibility_metrics(
    env: Any,
    *,
    camera_name: str = "agentview",
    width: int = 256,
    height: int = 256,
) -> dict[str, Any]:
    segmentation = np.asarray(
        env.env.sim.render(
            camera_name=camera_name,
            width=width,
            height=height,
            depth=False,
            segmentation=True,
        )
    )
    if segmentation.shape != (height, width, 2):
        raise ValueError(f"unexpected simulator segmentation shape: {segmentation.shape}")
    # MuJoCo segmentation stores object type and object id; type 5 is a geom.
    geom_ids = np.where(segmentation[..., 0] == 5, segmentation[..., 1], -1)
    geom_to_instance = {
        int(geom_id): str(instance)
        for geom_id, instance in env.env.model.geom_ids_to_instances.items()
    }
    interest = [str(name) for name in env.obj_of_interest]
    objects = {}
    for name in interest:
        matching_ids = [geom_id for geom_id, instance in geom_to_instance.items() if instance == name]
        mask = np.isin(geom_ids, matching_ids) if matching_ids else np.zeros_like(geom_ids, dtype=bool)
        pixel_count = int(np.count_nonzero(mask))
        border_touch = bool(
            np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1])
        )
        objects[name] = {
            "pixel_count": pixel_count,
            "frame_fraction": pixel_count / float(width * height),
            "visible": pixel_count > 0,
            "visible_at_least_16px": pixel_count >= 16,
            "border_touch": border_touch,
        }
    return {
        "camera_name": camera_name,
        "frame_width": width,
        "frame_height": height,
        "object_count": len(interest),
        "all_interest_visible": bool(objects) and all(row["visible"] for row in objects.values()),
        "all_interest_visible_at_least_16px": bool(objects)
        and all(row["visible_at_least_16px"] for row in objects.values()),
        "any_interest_border_touch": any(row["border_touch"] for row in objects.values()),
        "minimum_interest_pixel_count": min(
            (int(row["pixel_count"]) for row in objects.values()),
            default=0,
        ),
        "objects": objects,
    }


def _candidate_representatives(protocol: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    representatives: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in protocol["official_camera_tasks"]:
        representatives.setdefault((str(row["suite"]), str(row["base_task"])), row)
    return representatives


def build_episode_specs(
    protocol: Mapping[str, Any],
    *,
    modes: Sequence[str],
    suites: Sequence[str] | None = None,
    init_state_count: int = 1,
) -> list[dict[str, Any]]:
    if init_state_count <= 0:
        raise ValueError("init-state-count must be positive")
    unknown = set(modes) - {"gap", "camera_full", "candidates", "composition"}
    if unknown:
        raise ValueError(f"unsupported evaluation modes: {sorted(unknown)}")
    allowed_suites = set(suites or MAX_STEPS_BY_SUITE)
    specs: list[dict[str, Any]] = []
    if "camera_full" in modes:
        for task in protocol["official_camera_tasks"]:
            if str(task["suite"]) not in allowed_suites:
                continue
            for init_state_index in range(init_state_count):
                specs.append(
                    {
                        "pair_key": (
                            f"camera-full::{task['suite']}::{task['task_id']}::"
                            f"init{init_state_index}"
                        ),
                        "suite": str(task["suite"]),
                        "task_index": int(task["task_index"]),
                        "base_task": str(task["base_task"]),
                        "prompt": clean_task_prompt(str(task["base_task"])),
                        "difficulty_level": int(task["difficulty_level"]),
                        "official_task_id": int(task["task_id"]),
                        "official_camera_name": str(task["name"]),
                        "perturbation_family": str(task["perturbation_family"]),
                        "condition": "official_camera",
                        "camera_task_name": str(task["name"]),
                        "init_state_index": init_state_index,
                    }
                )
    if "gap" in modes:
        for task in protocol["official_camera_tasks"]:
            if str(task["suite"]) not in allowed_suites:
                continue
            for init_state_index in range(init_state_count):
                pair_key = f"gap::{task['suite']}::{task['task_id']}::init{init_state_index}"
                common = {
                    "pair_key": pair_key,
                    "suite": str(task["suite"]),
                    "task_index": int(task["task_index"]),
                    "base_task": str(task["base_task"]),
                    "prompt": clean_task_prompt(str(task["base_task"])),
                    "difficulty_level": int(task["difficulty_level"]),
                    "official_task_id": int(task["task_id"]),
                    "official_camera_name": str(task["name"]),
                    "perturbation_family": str(task["perturbation_family"]),
                    "init_state_index": init_state_index,
                }
                specs.extend(
                    [
                        {
                            **common,
                            "condition": "canonical",
                            "camera_task_name": str(task["base_task"]),
                        },
                        {
                            **common,
                            "condition": "official_camera",
                            "camera_task_name": str(task["name"]),
                        },
                    ]
                )
    if "candidates" in modes:
        representatives = _candidate_representatives(protocol)
        for task in protocol["candidate_matrix_base_tasks"]:
            suite = str(task["suite"])
            base_task = str(task["base_task"])
            if suite not in allowed_suites:
                continue
            representative = representatives.get((suite, base_task), task)
            if "task_index" not in representative:
                raise ValueError(f"missing representative task for {suite}::{base_task}")
            for init_state_index in range(init_state_count):
                for view in protocol["candidate_views"]:
                    specs.append(
                        {
                            "pair_key": (
                                f"candidate::{suite}::{base_task}::init{init_state_index}"
                            ),
                            "suite": suite,
                            "task_index": int(representative["task_index"]),
                            "base_task": base_task,
                            "prompt": clean_task_prompt(base_task),
                            "difficulty_level": None,
                            "official_task_id": None,
                            "official_camera_name": None,
                            "perturbation_family": "candidate_matrix",
                            "condition": f"candidate:{view['name']}",
                            "camera_task_name": (
                                base_task
                                if str(view["name"]) == "canonical"
                                else synthetic_camera_task_name(base_task, view)
                            ),
                            "candidate_view": dict(view),
                            "init_state_index": init_state_index,
                        }
                    )
    if "composition" in modes:
        for task in protocol["composition_tasks"]:
            suite = str(task["suite"])
            if suite not in allowed_suites:
                continue
            for init_state_index in range(init_state_count):
                pair_key = (
                    f"composition::{suite}::{task['base_task']}::init{init_state_index}"
                )
                common = {
                    "pair_key": pair_key,
                    "suite": suite,
                    "task_index": int(task["task_index"]),
                    "base_task": str(task["base_task"]),
                    "prompt": clean_task_prompt(str(task["base_task"])),
                    "difficulty_level": int(task["difficulty_level"]),
                    "camera_difficulty_level": int(task["camera_difficulty_level"]),
                    "background_difficulty_level": int(
                        task["background_difficulty_level"]
                    ),
                    "background_difficulty_distance": int(
                        task["background_difficulty_distance"]
                    ),
                    "official_task_id": int(task["task_id"]),
                    "official_camera_name": str(task["camera_task_name"]),
                    "perturbation_family": str(task["perturbation_family"]),
                    "background_task_id": int(task["background_task_id"]),
                    "background_task_name": str(task["background_task_name"]),
                    "background_kind": str(task["background_kind"]),
                    "background_texture_index": int(
                        task["background_texture_index"]
                    ),
                    "background_exact_difficulty_match": bool(
                        task["background_exact_difficulty_match"]
                    ),
                    "init_state_index": init_state_index,
                }
                specs.extend(
                    [
                        {
                            **common,
                            "condition": "canonical",
                            "camera_task_name": str(task["base_task"]),
                        },
                        {
                            **common,
                            "condition": "camera_only",
                            "camera_task_name": str(task["camera_task_name"]),
                        },
                        {
                            **common,
                            "condition": "background_only",
                            "camera_task_name": str(task["background_task_name"]),
                        },
                        {
                            **common,
                            "condition": "camera_background",
                            "camera_task_name": str(
                                task["camera_background_task_name"]
                            ),
                        },
                    ]
                )
    for spec in specs:
        spec["episode_id"] = hashlib.sha256(
            f"{spec['pair_key']}::{spec['condition']}".encode("utf-8")
        ).hexdigest()[:20]
    return specs


def prepare_policy_observation(
    observation: Mapping[str, Any],
    *,
    prompt: str,
    resize_size: int,
    eval_seed: int,
    camera_calibration: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    from openpi_client import image_tools

    agent = np.ascontiguousarray(np.asarray(observation["agentview_image"])[::-1, ::-1])
    wrist = np.ascontiguousarray(np.asarray(observation["robot0_eye_in_hand_image"])[::-1, ::-1])
    agent = image_tools.convert_to_uint8(image_tools.resize_with_pad(agent, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"]),
            quat_to_axis_angle(np.asarray(observation["robot0_eef_quat"])),
            np.asarray(observation["robot0_gripper_qpos"]),
        )
    )
    return (
        {
            "observation/image": agent,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": str(prompt),
            "_eval_seed": int(eval_seed),
            "camera_intrinsics": np.asarray(
                camera_calibration["camera_intrinsics"], dtype=np.float64
            ),
            "camera_to_world_opencv": np.asarray(
                camera_calibration["camera_to_world_opencv"], dtype=np.float64
            ),
        },
        agent,
        wrist,
    )


def agentview_camera_calibration(env: Any) -> dict[str, np.ndarray]:
    calibration = mujoco_camera_calibration(
        env,
        camera_name="agentview",
        height=POLICY_CAMERA_RESOLUTION,
        width=POLICY_CAMERA_RESOLUTION,
    )
    return {
        "camera_intrinsics": np.asarray(calibration["intrinsics"], dtype=np.float64),
        "camera_to_world_opencv": np.asarray(
            calibration["camera_to_world_opencv"], dtype=np.float64
        ),
    }


def quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(quat[3]) / denominator)


def physics_state_sha256(env: Any) -> tuple[str, int]:
    state = np.asarray(env.env.sim.get_state().flatten(), dtype=np.float64)
    return hashlib.sha256(state.tobytes()).hexdigest(), int(state.size)


def probe_action_variance(
    client: Any,
    example: Mapping[str, Any],
    *,
    sample_count: int,
    seed_key: str,
    seed: int,
    horizon: int,
) -> dict[str, Any]:
    if sample_count <= 0:
        return {"sample_count": 0, "mean_variance": None, "mean_pairwise_rms": None}
    actions = []
    for sample_index in range(sample_count):
        request = dict(example)
        request["_eval_seed"] = stable_seed(f"{seed_key}::probe::{sample_index}", seed=seed)
        actions.append(np.asarray(client.infer(request)["actions"], dtype=np.float32)[:horizon])
    values = np.stack(actions)
    pairwise = []
    for left in range(sample_count):
        for right in range(left + 1, sample_count):
            pairwise.append(float(np.sqrt(np.mean(np.square(values[left] - values[right])))))
    return {
        "sample_count": sample_count,
        "mean_variance": float(np.mean(np.var(values, axis=0))),
        "mean_pairwise_rms": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def encode_av1_video(path: Path, frames: Sequence[np.ndarray], *, fps: int = 10) -> None:
    import imageio.v2 as imageio

    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        path,
        list(frames),
        format="FFMPEG",
        fps=fps,
        codec="libaom-av1",
        ffmpeg_params=["-crf", "38", "-cpu-used", "8", "-pix_fmt", "yuv420p"],
    )


def run_episode(
    spec: Mapping[str, Any],
    *,
    task_suite: Any,
    bddl_root: Path,
    client: Any,
    replan_steps: int,
    wait_steps: int,
    resize_size: int,
    seed: int,
    probe_samples: int,
    probe_horizon: int,
    save_video: bool,
    video_dir: Path,
    render_gpu: int,
) -> tuple[dict[str, Any], Any]:
    from libero.libero.envs import OffScreenRenderEnv

    suite = str(spec["suite"])
    task = task_suite.get_task(int(spec["task_index"]))
    init_states = task_suite.get_task_init_states(int(spec["task_index"]))
    init_state_index = int(spec.get("init_state_index", 0))
    if init_state_index >= len(init_states):
        raise ValueError(
            f"task {spec['task_index']} only has {len(init_states)} initial states, "
            f"cannot select {init_state_index}"
        )
    bddl = bddl_root / suite / f"{spec['camera_task_name']}.bddl"
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        render_gpu_device_id=render_gpu,
    )
    action_plan: collections.deque[np.ndarray] = collections.deque()
    frames: list[np.ndarray] = []
    inference_calls = 0
    success = False
    initial_metrics: dict[str, Any] = {}
    episode_seed = stable_seed(str(spec["pair_key"]), seed=seed)
    env.seed(episode_seed)
    env.reset()
    observation = env.set_init_state(init_states[init_state_index])
    for _ in range(wait_steps):
        observation, _, done, _ = env.step(DUMMY_ACTION)
        if done:
            success = True
            break
    camera_calibration = agentview_camera_calibration(env)
    initial_example, initial_agent, initial_wrist = prepare_policy_observation(
        observation,
        prompt=str(spec["prompt"]),
        resize_size=resize_size,
        eval_seed=episode_seed,
        camera_calibration=camera_calibration,
    )
    camera_id = int(env.env.sim.model.camera_name2id("agentview"))
    env.env.sim.forward()
    state_sha256, state_size = physics_state_sha256(env)
    initial_metrics = {
        "agent": initial_image_metrics(initial_agent),
        "wrist": initial_image_metrics(initial_wrist),
        "agent_camera_position": np.asarray(env.env.sim.data.cam_xpos[camera_id]).tolist(),
        "agent_camera_rotation": np.asarray(env.env.sim.data.cam_xmat[camera_id]).reshape(3, 3).tolist(),
        "physics_state_sha256": state_sha256,
        "physics_state_size": state_size,
        "sim_visibility": simulator_visibility_metrics(env),
        "action_probe": probe_action_variance(
            client,
            initial_example,
            sample_count=(
                probe_samples if str(spec["pair_key"]).startswith("candidate::") else 0
            ),
            seed_key=str(spec["pair_key"]),
            seed=seed,
            horizon=probe_horizon,
        ),
    }
    max_steps = MAX_STEPS_BY_SUITE[suite]
    step = 0
    while not success and step < max_steps:
        if not action_plan:
            call_seed = stable_seed(
                f"{spec['pair_key']}::policy_call::{inference_calls}",
                seed=seed,
            )
            example, agent, wrist = prepare_policy_observation(
                observation,
                prompt=str(spec["prompt"]),
                resize_size=resize_size,
                eval_seed=call_seed,
                camera_calibration=camera_calibration,
            )
            output = client.infer(example)
            chunk = np.asarray(output["actions"], dtype=np.float32)
            if len(chunk) < replan_steps or chunk.shape[1] != 7:
                raise ValueError(f"invalid action chunk shape: {chunk.shape}")
            action_plan.extend(chunk[:replan_steps])
            inference_calls += 1
            if save_video:
                frames.append(np.concatenate([agent, wrist], axis=1))
        action = action_plan.popleft()
        observation, _, done, _ = env.step(np.asarray(action, dtype=np.float32))
        success = bool(done)
        step += 1
    if save_video:
        suffix = "success" if success else "failure"
        encode_av1_video(video_dir / f"{spec['episode_id']}--{suffix}.webm", frames)
    row = {
        **dict(spec),
        "status": "complete",
        "success": success,
        "completion_steps": step,
        "max_steps": max_steps,
        "wait_steps": wait_steps,
        "replan_steps": replan_steps,
        "inference_calls": inference_calls,
        "episode_seed": episode_seed,
        "language": str(spec["prompt"]),
        "raw_plus_language": str(task.language),
        "bddl_file": str(bddl),
        "initial_metrics": initial_metrics,
    }
    # LIBERO-Plus terminates the interpreter from env.close(). Keep the environment
    # alive until the result is durable, then let the one-episode worker process exit.
    return row, env


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text().splitlines():
        if line.strip():
            completed.add(str(json.loads(line)["episode_id"]))
    return completed


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired OpenPI Pi0.5 LIBERO-Plus view evaluation")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bddl-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("gap", "camera_full", "candidates", "composition"),
        default=["gap"],
    )
    parser.add_argument("--suites", nargs="+", choices=tuple(MAX_STEPS_BY_SUITE))
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--init-state-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--probe-samples", type=int, default=3)
    parser.add_argument("--probe-horizon", type=int, default=5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--video-episodes", type=int, default=4)
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--episode-index", type=int)
    return parser.parse_args(args)


def _sharded_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    protocol = json.loads(args.protocol.read_text())
    specs = build_episode_specs(
        protocol,
        modes=args.modes,
        suites=args.suites,
        init_state_count=args.init_state_count,
    )
    specs = [spec for index, spec in enumerate(specs) if index % args.num_shards == args.shard_index]
    if args.max_episodes is not None:
        specs = specs[: args.max_episodes]
    return specs


def video_episode_indices(
    specs: Sequence[Mapping[str, Any]],
    count: int,
) -> set[int]:
    if count <= 0:
        return set()
    by_mode = [
        [index for index, spec in enumerate(specs) if str(spec["pair_key"]).startswith(prefix)]
        for prefix in ("gap::", "camera-full::", "candidate::", "composition::")
    ]
    by_mode = [indices for indices in by_mode if indices]
    if not by_mode:
        return set()
    selected: list[int] = []
    for allocation_index in range(count):
        mode_index = allocation_index % len(by_mode)
        position = allocation_index // len(by_mode)
        if position < len(by_mode[mode_index]):
            selected.append(by_mode[mode_index][position])
    if len(selected) < count:
        remaining = [
            index
            for index in range(len(specs))
            if index not in selected
        ]
        selected.extend(remaining[: count - len(selected)])
    return set(selected)


def _run_parent(args: argparse.Namespace, specs: Sequence[Mapping[str, Any]]) -> None:
    output = args.output_dir / f"episodes-shard-{args.shard_index:02d}.jsonl"
    completed = _load_completed(output)
    pending_indices = [
        index for index, spec in enumerate(specs) if str(spec["episode_id"]) not in completed
    ]
    for completed_count, episode_index in enumerate(pending_indices, start=1):
        print(
            json.dumps(
                {
                    "worker": completed_count,
                    "pending": len(pending_indices),
                    "episode_index": episode_index,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--episode-index",
                str(episode_index),
            ],
            check=True,
        )


def main() -> None:
    args = parse_args()
    if not 1 <= args.replan_steps <= 10:
        raise ValueError("replan-steps must be in [1, 10]")
    if args.probe_samples < 0 or args.probe_horizon <= 0:
        raise ValueError("invalid action probe configuration")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    specs = _sharded_specs(args)
    if args.episode_index is None:
        _run_parent(args, specs)
        return
    if not 0 <= args.episode_index < len(specs):
        raise ValueError(f"episode-index {args.episode_index} is outside [0, {len(specs)})")

    output = args.output_dir / f"episodes-shard-{args.shard_index:02d}.jsonl"
    completed = _load_completed(output)
    spec = specs[args.episode_index]
    if str(spec["episode_id"]) in completed:
        return

    from libero.libero import benchmark
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    task_suite = benchmark.get_benchmark_dict()[str(spec["suite"])]()
    row, _environment = run_episode(
        spec,
        task_suite=task_suite,
        bddl_root=args.bddl_root,
        client=client,
        replan_steps=args.replan_steps,
        wait_steps=args.wait_steps,
        resize_size=args.resize_size,
        seed=args.seed,
        probe_samples=args.probe_samples,
        probe_horizon=args.probe_horizon,
        save_video=args.episode_index in video_episode_indices(specs, args.video_episodes),
        video_dir=args.output_dir / "videos_av1",
        render_gpu=args.render_gpu,
    )
    _append_jsonl(output, row)
    print(
        json.dumps(
            {
                "episode_index": args.episode_index,
                "episode_id": row["episode_id"],
                "condition": row["condition"],
                "success": row["success"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
