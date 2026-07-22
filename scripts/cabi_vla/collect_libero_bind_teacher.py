from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

FRESH_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "fresh_vla"
if str(FRESH_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(FRESH_SCRIPT_DIR))

@dataclass(frozen=True)
class ObjectCalibration:
    grasp_height: float
    release_height: float
    approach_height: float = 0.18
    carry_height: float = 0.24
    lift_delta: float = 0.20
    close_steps: int = 30
    release_steps: int = 25
    grasp_stall_steps: int = 12
    grasp_xy_offsets: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    placement_xy_bias: tuple[float, float] = (0.0, 0.0)


CALIBRATIONS = {
    "red_coffee_mug_1": ObjectCalibration(
        grasp_height=0.09,
        release_height=0.14,
        grasp_stall_steps=120,
        grasp_xy_offsets=((0.0, 0.0), (-0.03, 0.01), (-0.03, 0.015)),
    ),
    "porcelain_mug_1": ObjectCalibration(grasp_height=0.10, release_height=0.115),
    "white_yellow_mug_1": ObjectCalibration(
        grasp_height=0.09,
        release_height=0.115,
        placement_xy_bias=(0.0, 0.01),
    ),
}


def parse_indices(value: str) -> list[int]:
    indices: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, stop_text = part.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise ValueError(f"descending state range is invalid: {part}")
            indices.extend(range(start, stop + 1))
        else:
            indices.append(int(part))
    if not indices:
        raise ValueError("at least one state index is required")
    if len(indices) != len(set(indices)):
        raise ValueError("state indices must be unique")
    if min(indices) < 0 or max(indices) > 49:
        raise ValueError("state indices must stay inside [0, 49]")
    return indices


def action_toward(
    current_position: Sequence[float],
    target_position: Sequence[float],
    *,
    gripper: float,
    translation_scale: float = 0.05,
) -> np.ndarray:
    current = np.asarray(current_position, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    if current.shape != (3,) or target.shape != (3,):
        raise ValueError("current and target positions must be three-vectors")
    action = np.zeros(7, dtype=np.float64)
    action[:3] = np.clip((target - current) / translation_scale, -1.0, 1.0)
    action[-1] = np.clip(gripper, -1.0, 1.0)
    return action


def quat_to_axis_angle(quat: Sequence[float]) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64).copy()
    if value.shape != (4,):
        raise ValueError("quaternion must have four values in xyzw order")
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - value[3] ** 2))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return value[:3] * (2.0 * np.arccos(value[3]) / denominator)


def robot_state(observation: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(observation["robot0_eef_pos"], dtype=np.float64),
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64),
        ]
    ).astype(np.float32)


def upright_image(image: np.ndarray) -> np.ndarray:
    return np.flip(np.asarray(image), axis=(0, 1)).copy()


def _step(env: Any, action: Sequence[float]) -> Mapping[str, Any]:
    observation, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
    return observation


def object_geom_names(env: Any, object_name: str) -> list[str]:
    names = [name for name in env.sim.model.geom_names if name and object_name in name]
    if not names:
        raise KeyError(f"no MuJoCo geoms found for {object_name}")
    return names


def object_grasped(env: Any, object_name: str) -> bool:
    return bool(
        env.env._check_grasp(
            env.robots[0].gripper,
            object_geom_names(env, object_name),
        )
    )


class TraceRecorder:
    def __init__(
        self,
        env: Any,
        observation: Mapping[str, Any],
        *,
        source_object: str,
        target_object: str,
    ) -> None:
        self.source_object = source_object
        self.target_object = target_object
        self.observations: dict[str, list] = {
            "agentview": [],
            "wrist": [],
            "robot_state": [],
            "eef_pose": [],
            "source_pose": [],
            "target_pose": [],
            "grasped": [],
            "success": [],
            "phase": [],
        }
        self.actions: list[np.ndarray] = []
        self._append(env, observation, "episode_start")

    def _append(self, env: Any, observation: Mapping[str, Any], phase: str) -> None:
        self.observations["agentview"].append(upright_image(observation["agentview_image"]))
        self.observations["wrist"].append(
            upright_image(observation["robot0_eye_in_hand_image"])
        )
        self.observations["robot_state"].append(robot_state(observation))
        self.observations["eef_pose"].append(
            np.concatenate(
                [observation["robot0_eef_pos"], observation["robot0_eef_quat"]]
            ).astype(np.float32)
        )
        for key, object_name in (
            ("source_pose", self.source_object),
            ("target_pose", self.target_object),
        ):
            self.observations[key].append(
                np.concatenate(
                    [observation[f"{object_name}_pos"], observation[f"{object_name}_quat"]]
                ).astype(np.float32)
            )
        self.observations["grasped"].append(object_grasped(env, self.source_object))
        self.observations["success"].append(bool(env.check_success()))
        self.observations["phase"].append(phase)

    def step(
        self,
        env: Any,
        observation: Mapping[str, Any],
        action: Sequence[float],
        phase: str,
    ) -> Mapping[str, Any]:
        self.actions.append(np.asarray(action, dtype=np.float32))
        observation = _step(env, action)
        self._append(env, observation, phase)
        return observation

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {key: np.asarray(value) for key, value in self.observations.items()}
        arrays["actions"] = np.asarray(self.actions, dtype=np.float32)
        return arrays


def move_to(
    env: Any,
    recorder: TraceRecorder,
    observation: Mapping[str, Any],
    target: Sequence[float],
    *,
    gripper: float,
    phase: str,
    tolerance: float = 0.01,
    max_steps: int = 120,
    allow_contact_stall: bool = False,
    contact_stall_steps: int = 12,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    target_value = np.asarray(target, dtype=np.float64)
    distances: list[float] = []
    for step_index in range(max_steps):
        distance = float(
            np.linalg.norm(np.asarray(observation["robot0_eef_pos"]) - target_value)
        )
        if distance <= tolerance:
            return observation, {
                "phase": phase,
                "steps": step_index,
                "reached": True,
                "contact_stall": False,
                "final_distance": distance,
            }
        distances.append(distance)
        if allow_contact_stall and len(distances) >= contact_stall_steps:
            recent = distances[-contact_stall_steps:]
            if min(recent) >= recent[0] - 1e-3:
                return observation, {
                    "phase": phase,
                    "steps": step_index,
                    "reached": False,
                    "contact_stall": True,
                    "final_distance": distance,
                }
        action = action_toward(
            observation["robot0_eef_pos"],
            target_value,
            gripper=gripper,
        )
        observation = recorder.step(env, observation, action, phase)
    if allow_contact_stall:
        return observation, {
            "phase": phase,
            "steps": max_steps,
            "reached": False,
            "contact_stall": True,
            "final_distance": distances[-1],
        }
    raise RuntimeError(f"teacher failed to reach {phase} within {max_steps} steps")


def hold(
    env: Any,
    recorder: TraceRecorder,
    observation: Mapping[str, Any],
    *,
    gripper: float,
    phase: str,
    steps: int,
) -> Mapping[str, Any]:
    action = np.asarray([0.0] * 6 + [gripper], dtype=np.float64)
    for _ in range(steps):
        observation = recorder.step(env, observation, action, phase)
    return observation


def _collect_episode_once(
    env: Any,
    initial_state: np.ndarray,
    edge: Mapping[str, Any],
    *,
    grasp_xy_offset: tuple[float, float],
    grasp_attempt_index: int,
    settle_steps: int = 8,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source_object = str(edge["source_object"])
    target_object = str(edge["target_object"])
    calibration = CALIBRATIONS[source_object]
    env.reset()
    observation = env.set_init_state(np.asarray(initial_state))
    for _ in range(settle_steps):
        observation = _step(env, [0.0] * 6 + [-1.0])

    recorder = TraceRecorder(
        env,
        observation,
        source_object=source_object,
        target_object=target_object,
    )
    source = np.asarray(observation[f"{source_object}_pos"], dtype=np.float64)
    target = np.asarray(observation[f"{target_object}_pos"], dtype=np.float64)
    phases = []

    observation, result = move_to(
        env,
        recorder,
        observation,
        source + [0.0, 0.0, calibration.approach_height],
        gripper=-1.0,
        phase="approach_above",
    )
    phases.append(result)
    observation, result = move_to(
        env,
        recorder,
        observation,
        source
        + np.asarray(
            [grasp_xy_offset[0], grasp_xy_offset[1], calibration.grasp_height]
        ),
        gripper=-1.0,
        phase="approach_grasp",
        allow_contact_stall=True,
        contact_stall_steps=calibration.grasp_stall_steps,
    )
    phases.append(result)
    observation = hold(
        env,
        recorder,
        observation,
        gripper=1.0,
        phase="close_gripper",
        steps=calibration.close_steps,
    )
    if not object_grasped(env, source_object):
        raise RuntimeError(f"teacher failed to grasp {source_object}")

    grasped_source = np.asarray(
        observation[f"{source_object}_pos"], dtype=np.float64
    )
    grasped_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    object_minus_eef_xy = grasped_source[:2] - grasped_eef[:2]
    placement_eef_xy = (
        target[:2]
        - object_minus_eef_xy
        + np.asarray(calibration.placement_xy_bias, dtype=np.float64)
    )

    lift_target = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    lift_target[2] += calibration.lift_delta
    for phase, waypoint in (
        ("lift", lift_target),
        (
            "transport",
            np.asarray(
                [placement_eef_xy[0], placement_eef_xy[1], target[2] + calibration.carry_height]
            ),
        ),
        (
            "lower",
            np.asarray(
                [placement_eef_xy[0], placement_eef_xy[1], target[2] + calibration.release_height]
            ),
        ),
    ):
        observation, result = move_to(
            env,
            recorder,
            observation,
            waypoint,
            gripper=1.0,
            phase=phase,
            allow_contact_stall=phase == "lower",
        )
        phases.append(result)
    if not object_grasped(env, source_object):
        raise RuntimeError(f"teacher dropped {source_object} before release")

    observation = hold(
        env,
        recorder,
        observation,
        gripper=-1.0,
        phase="release",
        steps=calibration.release_steps,
    )
    observation, result = move_to(
        env,
        recorder,
        observation,
        np.asarray(
            [placement_eef_xy[0], placement_eef_xy[1], target[2] + calibration.carry_height]
        ),
        gripper=-1.0,
        phase="retract",
    )
    phases.append(result)
    observation = hold(
        env,
        recorder,
        observation,
        gripper=-1.0,
        phase="settle",
        steps=10,
    )
    success = bool(env.check_success())
    arrays = recorder.arrays()
    metadata = {
        "success": success,
        "steps": int(len(arrays["actions"])),
        "source_object": source_object,
        "target_object": target_object,
        "calibration": asdict(calibration),
        "grasp_attempt_index": grasp_attempt_index,
        "grasp_xy_offset": list(grasp_xy_offset),
        "object_minus_eef_xy_at_grasp": object_minus_eef_xy.round(8).tolist(),
        "phases": phases,
        "final_source_position": np.asarray(
            observation[f"{source_object}_pos"]
        ).round(8).tolist(),
        "final_target_position": np.asarray(
            observation[f"{target_object}_pos"]
        ).round(8).tolist(),
    }
    return arrays, metadata


def collect_episode(
    env: Any,
    initial_state: np.ndarray,
    edge: Mapping[str, Any],
    *,
    settle_steps: int = 8,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collect one successful expert trace using state-agnostic grasp candidates.

    Candidate order is fixed per object and never depends on task edge or state index.
    A failed candidate is discarded and the exact initial simulator state is restored,
    so failed exploratory motion cannot leak into the saved policy trajectory.
    """

    source_object = str(edge["source_object"])
    calibration = CALIBRATIONS[source_object]
    last_error: RuntimeError | None = None
    last_result: tuple[dict[str, np.ndarray], dict[str, Any]] | None = None
    for attempt_index, grasp_xy_offset in enumerate(calibration.grasp_xy_offsets):
        try:
            result = _collect_episode_once(
                env,
                initial_state,
                edge,
                grasp_xy_offset=grasp_xy_offset,
                grasp_attempt_index=attempt_index,
                settle_steps=settle_steps,
            )
            if bool(result[1]["success"]):
                return result
            last_result = result
        except RuntimeError as error:
            last_error = error
            retryable = "failed to grasp" in str(error) or "dropped" in str(error)
            if not retryable:
                raise
    if last_result is not None:
        return last_result
    assert last_error is not None
    raise last_error


def load_state_bank(path: Path) -> np.ndarray:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"expected a zipped LIBERO state bank: {path}")
    with zipfile.ZipFile(path) as archive:
        data_name = next(
            (name for name in archive.namelist() if name.endswith("/data.pkl")),
            None,
        )
        if data_name is None:
            raise ValueError(f"state archive has no data.pkl: {path}")
        states = np.asarray(pickle.loads(archive.read(data_name)))
    if states.shape != (50, 84):
        raise ValueError(f"unexpected LIBERO-Bind state shape: {states.shape}")
    return states


def select_edges(
    manifest: Mapping[str, Any],
    edge_ids: Sequence[str],
    *,
    purpose: str,
) -> list[dict[str, Any]]:
    by_id = {edge["edge_id"]: edge for edge in manifest["edges"]}
    unknown = sorted(set(edge_ids) - set(by_id))
    if unknown:
        raise KeyError(f"unknown LIBERO-Bind edges: {unknown}")
    selected = [by_id[edge_id] for edge_id in edge_ids]
    if purpose == "train" and any(not edge["action_supervised"] for edge in selected):
        raise ValueError("training collection cannot include action-withheld edges")
    return selected


def write_episode_video(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    from video_io import write_h264_video

    frames = np.concatenate([arrays["agentview"], arrays["wrist"]], axis=2)
    write_h264_video(path, frames, fps=10.0)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    from libero.libero.envs import OffScreenRenderEnv

    suite_manifest = json.loads((args.suite_root / "manifest.json").read_text())
    edge_ids = (
        [edge["edge_id"] for edge in suite_manifest["edges"]]
        if args.edges == "all"
        else [value.strip() for value in args.edges.split(",") if value.strip()]
    )
    edges = select_edges(suite_manifest, edge_ids, purpose=args.purpose)
    state_indices = parse_indices(args.state_indices)
    if args.purpose == "train" and any(index >= 40 for index in state_indices):
        raise ValueError("training collection cannot include sealed test states")
    states = load_state_bank(Path(suite_manifest["canonical_init_states"]))

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    staging = args.output_dir.parent / f".{args.output_dir.name}.staging-{os.getpid()}"
    episodes_dir = staging / "episodes"
    videos_dir = staging / "videos"
    episodes_dir.mkdir(parents=True, exist_ok=False)
    videos_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        for edge in edges:
            env = OffScreenRenderEnv(
                bddl_file_name=edge["bddl"],
                camera_heights=args.resolution,
                camera_widths=args.resolution,
                horizon=args.max_steps,
                ignore_done=True,
            )
            try:
                env.seed(args.seed)
                for state_index in state_indices:
                    sample_id = f"{edge['edge_id']}--state-{state_index:02d}"
                    try:
                        arrays, metadata = collect_episode(
                            env,
                            states[state_index],
                            edge,
                            settle_steps=args.settle_steps,
                        )
                        row = {
                            "sample_id": sample_id,
                            "edge_id": edge["edge_id"],
                            "source_id": edge["source_id"],
                            "target_id": edge["target_id"],
                            "language_instruction": edge["language_instruction"],
                            "canonical_state_index": state_index,
                            "split": suite_manifest["states"][state_index]["split"],
                            "action_supervised": bool(edge["action_supervised"]),
                            **metadata,
                        }
                        if not args.metadata_only:
                            np.savez_compressed(episodes_dir / f"{sample_id}.npz", **arrays)
                            row["episode_file"] = f"episodes/{sample_id}.npz"
                            if not args.skip_videos:
                                write_episode_video(videos_dir / f"{sample_id}.mp4", arrays)
                                row["video_file"] = f"videos/{sample_id}.mp4"
                    except RuntimeError as error:
                        row = {
                            "sample_id": sample_id,
                            "edge_id": edge["edge_id"],
                            "source_id": edge["source_id"],
                            "target_id": edge["target_id"],
                            "language_instruction": edge["language_instruction"],
                            "canonical_state_index": state_index,
                            "split": suite_manifest["states"][state_index]["split"],
                            "action_supervised": bool(edge["action_supervised"]),
                            "success": False,
                            "error": str(error),
                        }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
            finally:
                env.close()

        success_count = sum(bool(row.get("success")) for row in rows)
        report = {
            "schema_version": 1,
            "suite": str(args.suite_root),
            "purpose": args.purpose,
            "seed": args.seed,
            "resolution": args.resolution,
            "requested_episode_count": len(rows),
            "successful_episode_count": success_count,
            "success_rate": success_count / len(rows),
            "rows": rows,
        }
        (staging / "manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(args.output_dir)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect generic LIBERO-Bind teacher traces")
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=Path("/share/longjunyu/cabi-vla/libero-bind-v0"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purpose", choices=("qa", "train", "sealed_eval"), default="qa")
    parser.add_argument("--edges", default="all")
    parser.add_argument("--state-indices", default="0")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="run physics and quality checks without writing episode arrays or videos",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="write training arrays now and generate H.264 QA videos in a separate pass",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    report = collect(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "success_rate": report["success_rate"],
                "successful_episode_count": report["successful_episode_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
