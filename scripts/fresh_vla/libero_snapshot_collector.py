from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from counterfactual_data import (
    CounterfactualRecord,
    build_policy_inputs,
    estimate_branch_divergence,
    threshold_sensitivity,
    validate_record,
)


DEFAULT_BDDL = Path(
    "/projects/openpi/third_party/libero/libero/libero/bddl_files/libero_goal/"
    "put_the_cream_cheese_in_the_bowl.bddl"
)
DEFAULT_INIT_STATES = Path("/workspace/envs/fresh-libero/runtime/cream_cheese_init_states.npy")
LANGUAGE = "put the cream cheese in the bowl"
OBJECT_JOINT = "cream_cheese_1_joint0"


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
        raise ValueError("current_position and target_position must be three-vectors")
    if translation_scale <= 0:
        raise ValueError("translation_scale must be positive")
    action = np.zeros(7, dtype=np.float64)
    action[:3] = np.clip((target - current) / translation_scale, -1.0, 1.0)
    action[-1] = np.clip(gripper, -1.0, 1.0)
    return action


def gripper_transition_horizon(actions: np.ndarray) -> int:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("actions must be non-empty [H, D]")
    closed = values[:, -1] > 0
    transitions = np.flatnonzero(closed[1:] != closed[:-1]) + 1
    return len(values) if transitions.size == 0 else int(transitions[0] + 1)


def offset_free_joint_qpos(qpos: Sequence[float], offset: Sequence[float]) -> np.ndarray:
    value = np.asarray(qpos, dtype=np.float64).copy()
    delta = np.asarray(offset, dtype=np.float64)
    if value.shape != (7,) or delta.shape != (3,):
        raise ValueError("free-joint qpos must have 7 values and offset must have 3")
    value[:3] += delta
    return value


def quat_to_axis_angle(quat: Sequence[float]) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64).copy()
    if value.shape != (4,):
        raise ValueError("quat must be a four-vector in xyzw order")
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - value[3] * value[3]))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return value[:3] * (2.0 * math.acos(value[3]) / denominator)


def robot_state_from_observation(observation: Mapping[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(observation["robot0_eef_pos"], dtype=np.float64),
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64),
        ]
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def compact_policy_observation(observation: Mapping[str, Any], snapshot_key: str) -> dict[str, Any]:
    agent = np.asarray(observation["agentview_image"])
    wrist = np.asarray(observation["robot0_eye_in_hand_image"])
    return {
        "snapshot_key": snapshot_key,
        "agentview_shape": list(agent.shape),
        "agentview_sha256": _array_sha256(agent),
        "wrist_shape": list(wrist.shape),
        "wrist_sha256": _array_sha256(wrist),
        "eef_position": np.asarray(observation["robot0_eef_pos"]).round(8).tolist(),
        "cream_cheese_position": np.asarray(observation["cream_cheese_1_pos"]).round(8).tolist(),
        "bowl_position": np.asarray(observation["akita_black_bowl_1_pos"]).round(8).tolist(),
    }


def _step(env: Any, action: Sequence[float]) -> Mapping[str, Any]:
    observation, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
    return observation


def _move_to(
    env: Any,
    observation: Mapping[str, Any],
    target: Sequence[float],
    *,
    gripper: float,
    tolerance: float = 0.008,
    max_steps: int = 60,
) -> Mapping[str, Any]:
    target_array = np.asarray(target, dtype=np.float64)
    for _ in range(max_steps):
        if np.linalg.norm(target_array - observation["robot0_eef_pos"]) < tolerance:
            return observation
        observation = _step(
            env,
            action_toward(observation["robot0_eef_pos"], target_array, gripper=gripper),
        )
    raise RuntimeError(f"scripted controller failed to reach target within {max_steps} steps")


def _reset_to_initial_state(env: Any, initial_state: np.ndarray, settle_steps: int) -> Mapping[str, Any]:
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(settle_steps):
        observation = _step(env, [0.0] * 6 + [-1.0])
    return observation


SIM_DATA_RUNTIME_FIELDS = (
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "ctrl",
    "mocap_pos",
    "mocap_quat",
    "qacc",
    "qfrc_constraint",
    "efc_force",
    "efc_JT",
    "efc_AR",
    "efc_AR_colind",
    "efc_AR_rowadr",
    "efc_AR_rownnz",
    "dof_island",
    "dof_islandind",
    "efc_island",
    "island_dofind",
    "island_efcind",
    "cacc",
    "cfrc_ext",
    "cfrc_int",
    "sensordata",
)

# These buffers are sized by MuJoCo's active contact/constraint graph. A state
# restore can rebuild an equivalent graph with a different packed length, so a
# captured buffer is only assignable when its shape still matches.
DYNAMIC_CONSTRAINT_RUNTIME_FIELDS = frozenset(
    {
        "efc_force",
        "efc_JT",
        "efc_AR",
        "efc_AR_colind",
        "efc_AR_rowadr",
        "efc_AR_rownnz",
        "dof_island",
        "dof_islandind",
        "efc_island",
        "island_dofind",
        "island_efcind",
    }
)

ROBOT_BUFFER_FIELDS = (
    "recent_qpos",
    "recent_actions",
    "recent_torques",
    "recent_ee_forcetorques",
    "recent_ee_pose",
    "recent_ee_vel",
    "recent_ee_acc",
    "recent_ee_vel_buffer",
)


def _capture_robot_buffers(env: Any, state: dict[str, np.ndarray]) -> None:
    robot = env.robots[0]
    for name in ROBOT_BUFFER_FIELDS:
        buffer = getattr(robot, name, None)
        if buffer is None:
            continue
        prefix = f"robot_buffer_{name}__"
        if hasattr(buffer, "buf"):
            state[f"{prefix}buf"] = np.asarray(buffer.buf).copy()
            state[f"{prefix}ptr"] = np.asarray(buffer.ptr)
            state[f"{prefix}size"] = np.asarray(buffer._size)
        elif hasattr(buffer, "current") and hasattr(buffer, "last"):
            state[f"{prefix}current"] = np.asarray(buffer.current).copy()
            state[f"{prefix}last"] = np.asarray(buffer.last).copy()


def _restore_robot_buffers(env: Any, state: Mapping[str, np.ndarray]) -> None:
    robot = env.robots[0]
    for name in ROBOT_BUFFER_FIELDS:
        buffer = getattr(robot, name, None)
        if buffer is None:
            continue
        prefix = f"robot_buffer_{name}__"
        if f"{prefix}buf" in state and hasattr(buffer, "buf"):
            buffer.buf[:] = np.asarray(state[f"{prefix}buf"])
            buffer.ptr = int(np.asarray(state[f"{prefix}ptr"]))
            buffer._size = int(np.asarray(state[f"{prefix}size"]))
        elif f"{prefix}current" in state and hasattr(buffer, "current"):
            buffer.current = np.asarray(state[f"{prefix}current"]).copy()
            buffer.last = np.asarray(state[f"{prefix}last"]).copy()


def _restore_sim_data_runtime_fields(
    sim_data: Any,
    state: Mapping[str, np.ndarray],
) -> tuple[str, ...]:
    skipped = []
    for name in SIM_DATA_RUNTIME_FIELDS:
        key = f"sim_data_{name}"
        if key not in state or sim_data is None or not hasattr(sim_data, name):
            continue
        target = np.asarray(getattr(sim_data, name))
        source = np.asarray(state[key])
        if target.shape != source.shape:
            if name in DYNAMIC_CONSTRAINT_RUNTIME_FIELDS:
                skipped.append(name)
                continue
            raise ValueError(
                f"fixed MuJoCo runtime field shape mismatch for {name}: "
                f"captured={source.shape}, current={target.shape}"
            )
        target[...] = source
    return tuple(skipped)


def _capture_controller_state(env: Any) -> dict[str, np.ndarray]:
    state = {"gripper_action": np.asarray(env.robots[0].gripper.current_action).copy()}
    sim_data = getattr(env.sim, "data", None)
    for name in SIM_DATA_RUNTIME_FIELDS:
        if sim_data is not None and hasattr(sim_data, name):
            state[f"sim_data_{name}"] = np.asarray(getattr(sim_data, name)).copy()
    object_ids = _object_geom_ids(env)
    state["object_friction"] = env.sim.model.geom_friction[object_ids].copy()
    # LIBERO randomizes fixed-body positions during reset; these positions are
    # model state and are not included in MuJoCo's flattened simulation state.
    state["model_body_pos"] = env.sim.model.body_pos.copy()
    if hasattr(env.sim.model, "site_rgba"):
        state["model_site_rgba"] = env.sim.model.site_rgba.copy()
    controller = env.robots[0].controller
    if hasattr(controller, "new_update"):
        state["controller_new_update"] = np.asarray(controller.new_update, dtype=bool)
    for name in ("initial_joint", "initial_ee_pos", "initial_ee_ori_mat"):
        if hasattr(controller, name):
            state[f"controller_{name}"] = np.asarray(getattr(controller, name)).copy()
    for name in ("goal_pos", "goal_ori", "relative_ori"):
        if hasattr(controller, name):
            state[f"controller_{name}"] = np.asarray(getattr(controller, name)).copy()
    ori_ref = getattr(controller, "ori_ref", None)
    state["controller_ori_ref_valid"] = np.asarray(ori_ref is not None, dtype=bool)
    if ori_ref is not None:
        state["controller_ori_ref"] = np.asarray(ori_ref).copy()
    for name in ("interpolator_pos", "interpolator_ori"):
        interpolator = getattr(controller, name, None)
        if interpolator is None:
            continue
        for field in ("start", "goal", "step"):
            state[f"controller_{name}_{field}"] = np.asarray(
                getattr(interpolator, field)
            ).copy()
    runtime_env = env.env
    state["runtime_cur_time"] = np.asarray(runtime_env.cur_time)
    state["runtime_timestep"] = np.asarray(runtime_env.timestep)
    state["runtime_done"] = np.asarray(runtime_env.done, dtype=bool)
    for name, value in runtime_env._obs_cache.items():
        state[f"runtime_obs_cache__{name}"] = np.asarray(value).copy()
    for name, observable in runtime_env._observables.items():
        prefix = f"runtime_observable__{name}__"
        state[f"{prefix}time_since_last_sample"] = np.asarray(
            observable._time_since_last_sample
        )
        state[f"{prefix}current_delay"] = np.asarray(observable._current_delay)
        state[f"{prefix}current_observed_value"] = np.asarray(
            observable._current_observed_value
        ).copy()
        state[f"{prefix}sampled"] = np.asarray(observable._sampled, dtype=bool)
    _capture_robot_buffers(env, state)
    return state


def _restore_snapshot(env: Any, sim_state: np.ndarray, controller_state: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
    # A flattened MuJoCo state omits robosuite controller / observable state.
    # Legacy snapshots need a wrapper reset. Complete runtime snapshots restore
    # those fields directly and must not rebuild contact/controller state.
    if "runtime_timestep" not in controller_state:
        env.reset()
    env.sim.model.body_pos[:] = controller_state["model_body_pos"]
    if "model_site_rgba" in controller_state and hasattr(env.sim.model, "site_rgba"):
        env.sim.model.site_rgba[:] = controller_state["model_site_rgba"]
    object_ids = _object_geom_ids(env)
    env.sim.model.geom_friction[object_ids] = controller_state["object_friction"]
    observation = env.regenerate_obs_from_state(sim_state)
    controller = env.robots[0].controller
    controller.update(force=True)
    for name in ("initial_joint", "initial_ee_pos", "initial_ee_ori_mat"):
        key = f"controller_{name}"
        if key in controller_state:
            setattr(controller, name, np.asarray(controller_state[key]).copy())
    if "controller_goal_pos" not in controller_state:
        controller.reset_goal()
    else:
        for name in ("goal_pos", "goal_ori", "relative_ori"):
            key = f"controller_{name}"
            if key in controller_state:
                setattr(controller, name, np.asarray(controller_state[key]).copy())
        if bool(np.asarray(controller_state.get("controller_ori_ref_valid", False))):
            controller.ori_ref = np.asarray(controller_state["controller_ori_ref"]).copy()
        else:
            controller.ori_ref = None
        for name in ("interpolator_pos", "interpolator_ori"):
            interpolator = getattr(controller, name, None)
            if interpolator is None:
                continue
            for field in ("start", "goal", "step"):
                key = f"controller_{name}_{field}"
                if key in controller_state:
                    value = np.asarray(controller_state[key]).copy()
                    if field == "step":
                        value = int(value)
                    setattr(interpolator, field, value)
    env.robots[0].gripper.current_action = controller_state["gripper_action"].copy()
    _restore_sim_data_runtime_fields(getattr(env.sim, "data", None), controller_state)
    if "controller_new_update" in controller_state:
        controller.new_update = bool(np.asarray(controller_state["controller_new_update"]))
    _restore_robot_buffers(env, controller_state)
    if "runtime_timestep" in controller_state:
        runtime_env = env.env
        runtime_env.cur_time = float(np.asarray(controller_state["runtime_cur_time"]))
        runtime_env.timestep = int(np.asarray(controller_state["runtime_timestep"]))
        runtime_env.done = bool(np.asarray(controller_state["runtime_done"]))
        runtime_env._obs_cache = {
            key.split("__", 1)[1]: np.asarray(value).copy()
            for key, value in controller_state.items()
            if key.startswith("runtime_obs_cache__")
        }
        for name, observable in runtime_env._observables.items():
            prefix = f"runtime_observable__{name}__"
            value_key = f"{prefix}current_observed_value"
            if value_key not in controller_state:
                continue
            observable._time_since_last_sample = float(
                np.asarray(controller_state[f"{prefix}time_since_last_sample"])
            )
            observable._current_delay = float(
                np.asarray(controller_state[f"{prefix}current_delay"])
            )
            observable._current_observed_value = np.asarray(
                controller_state[value_key]
            ).copy()
            observable._sampled = bool(
                np.asarray(controller_state[f"{prefix}sampled"])
            )
        observation = runtime_env._get_observations()
    return observation


def _set_object_offset(env: Any, offset: Sequence[float]) -> Mapping[str, Any]:
    qpos = env.sim.data.get_joint_qpos(OBJECT_JOINT)
    env.sim.data.set_joint_qpos(OBJECT_JOINT, offset_free_joint_qpos(qpos, offset))
    env.sim.data.set_joint_qvel(OBJECT_JOINT, np.zeros(6, dtype=np.float64))
    return env.regenerate_obs_from_state(env.get_sim_state())


def _object_geom_ids(env: Any) -> list[int]:
    names = [name for name in env.sim.model.geom_names if name and "cream_cheese_1" in name]
    if not names:
        raise RuntimeError("could not locate cream-cheese collision geometries")
    return [env.sim.model.geom_name2id(name) for name in names]


def _prepare_grasp_snapshot(
    env: Any,
    initial_state: np.ndarray,
    settle_steps: int,
    *,
    object_offset: Sequence[float] = (0.0, 0.0, 0.0),
    grasp_offset: Sequence[float] = (0.0, 0.0, 0.0),
    friction_scale: float = 1.0,
) -> tuple[Mapping[str, Any], np.ndarray, dict[str, np.ndarray]]:
    observation = _reset_to_initial_state(env, initial_state, settle_steps)
    if friction_scale <= 0:
        raise ValueError("friction_scale must be positive")
    if np.linalg.norm(np.asarray(object_offset, dtype=np.float64)) > 0:
        observation = _set_object_offset(env, object_offset)
        for _ in range(5):
            observation = _step(env, [0.0] * 6 + [-1.0])
    object_ids = _object_geom_ids(env)
    env.sim.model.geom_friction[object_ids] *= friction_scale
    object_position = np.asarray(observation["cream_cheese_1_pos"]).copy()
    grasp_target = object_position + np.asarray(grasp_offset, dtype=np.float64)
    observation = _move_to(env, observation, grasp_target + [0.0, 0.0, 0.12], gripper=-1.0)
    observation = _move_to(env, observation, grasp_target, gripper=-1.0)
    for _ in range(25):
        observation = _step(env, [0.0] * 6 + [1.0])
    object_geoms = [env.sim.model.geom_id2name(index) for index in _object_geom_ids(env)]
    if not env.env._check_grasp(env.robots[0].gripper, object_geoms):
        raise RuntimeError("scripted grasp calibration failed")
    return observation, env.get_sim_state().copy(), _capture_controller_state(env)


def _prepare_push_snapshot(
    env: Any, initial_state: np.ndarray, settle_steps: int
) -> tuple[Mapping[str, Any], np.ndarray, dict[str, np.ndarray]]:
    observation = _reset_to_initial_state(env, initial_state, settle_steps)
    object_position = np.asarray(observation["cream_cheese_1_pos"]).copy()
    approach = object_position + [-0.065, 0.0, 0.0]
    observation = _move_to(env, observation, approach + [0.0, 0.0, 0.10], gripper=-1.0)
    observation = _move_to(env, observation, approach, gripper=-1.0)
    for _ in range(25):
        observation = _step(env, [0.0] * 6 + [1.0])
    return observation, env.get_sim_state().copy(), _capture_controller_state(env)


def _prepare_reach_snapshot(
    env: Any, initial_state: np.ndarray, settle_steps: int
) -> tuple[Mapping[str, Any], np.ndarray, dict[str, np.ndarray]]:
    observation = _reset_to_initial_state(env, initial_state, settle_steps)
    return observation, env.get_sim_state().copy(), _capture_controller_state(env)


def _jitter_action(action: np.ndarray, rng: np.random.Generator, amount: float) -> np.ndarray:
    result = np.asarray(action, dtype=np.float64).copy()
    result[:6] += rng.normal(0.0, amount, size=6)
    return np.clip(result, -1.0, 1.0)


def validate_physical_branches(task: str, physics: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, float]:
    if task == "grasp_slip":
        attached = list(physics["attached"])
        slipped = list(physics["slipped"])
        attached_rate = float(np.mean([row["grasped_at_end"] for row in attached]))
        slipped_rate = float(np.mean([row["grasped_at_end"] for row in slipped]))
        attached_displacement = float(np.mean([row["object_displacement"] for row in attached]))
        if attached_rate != 1.0 or slipped_rate != 0.0 or attached_displacement < 0.1:
            raise RuntimeError(
                "grasp/slip intervention failed physical validation: "
                f"attached_rate={attached_rate}, slipped_rate={slipped_rate}, "
                f"attached_displacement={attached_displacement}"
            )
        return {
            "attached_grasp_rate": attached_rate,
            "slipped_grasp_rate": slipped_rate,
            "attached_mean_displacement": attached_displacement,
        }
    if task == "blocked_push":
        free_displacement = float(
            np.mean([row["object_displacement"] for row in physics["free_slide"]])
        )
        blocked_displacement = float(
            np.mean([row["object_displacement"] for row in physics["blocked"]])
        )
        ratio = free_displacement / max(blocked_displacement, 1e-9)
        if free_displacement < 0.03 or ratio < 3.0:
            raise RuntimeError(
                "free/blocked push intervention failed physical validation: "
                f"free_displacement={free_displacement}, blocked_displacement={blocked_displacement}, "
                f"ratio={ratio}"
            )
        return {
            "free_mean_displacement": free_displacement,
            "blocked_mean_displacement": blocked_displacement,
            "free_over_blocked_ratio": ratio,
        }
    if task == "deterministic_reach":
        first = float(np.mean([row["target_error"] for row in physics["repeat_a"]]))
        second = float(np.mean([row["target_error"] for row in physics["repeat_b"]]))
        delta = abs(first - second)
        first_trajectory = np.mean(
            [np.asarray(row["eef_trajectory"], dtype=np.float64) for row in physics["repeat_a"]], axis=0
        )
        second_trajectory = np.mean(
            [np.asarray(row["eef_trajectory"], dtype=np.float64) for row in physics["repeat_b"]], axis=0
        )
        max_effect_distance = float(np.max(np.linalg.norm(first_trajectory - second_trajectory, axis=-1)))
        if delta > 0.005 or max_effect_distance > 0.002:
            raise RuntimeError(
                "deterministic reach drift is too large: "
                f"target_error_delta={delta}, max_effect_distance={max_effect_distance}"
            )
        return {
            "repeat_target_error_delta": delta,
            "max_effect_distance": max_effect_distance,
        }
    raise ValueError(f"unknown task: {task}")


def _run_grasp_branch(
    env: Any,
    snapshot: np.ndarray,
    controller_state: Mapping[str, np.ndarray],
    branch: str,
    *,
    horizon: int,
    event_time: int,
    rng: np.random.Generator,
    slip_offset: Sequence[float] = (0.055, 0.0, -0.005),
    reference_observation: Mapping[str, Any] | None = None,
    video_frames: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any], float]:
    observation = _restore_snapshot(env, snapshot, controller_state)
    restored_error = float(np.max(np.abs(env.get_sim_state() - snapshot)))
    if reference_observation is None:
        image_error = 0.0
        state_error = 0.0
    else:
        image_error = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(observation["agentview_image"], dtype=np.int16)
                        - np.asarray(reference_observation["agentview_image"], dtype=np.int16)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.int16)
                        - np.asarray(reference_observation["robot0_eye_in_hand_image"], dtype=np.int16)
                    )
                )
            ),
        )
        state_error = float(
            np.max(
                np.abs(
                    robot_state_from_observation(observation)
                    - robot_state_from_observation(reference_observation)
                )
            )
        )
    if video_frames is not None:
        video_frames.append(
            np.concatenate(
                [
                    np.asarray(observation["agentview_image"]),
                    np.asarray(observation["robot0_eye_in_hand_image"]),
                ],
                axis=1,
            )
        )
    actions = []
    object_trajectory = [np.asarray(observation["cream_cheese_1_pos"]).copy()]
    robot_state_trajectory = [robot_state_from_observation(observation).copy()]
    for step_index in range(horizon):
        if step_index == event_time and branch == "slipped":
            observation = _set_object_offset(env, slip_offset)
        if step_index < event_time:
            action = np.asarray([0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 1.0])
        elif branch == "attached":
            target = np.asarray(observation["akita_black_bowl_1_pos"]) + [0.0, 0.0, 0.16]
            action = action_toward(observation["robot0_eef_pos"], target, gripper=1.0)
        elif step_index < event_time + 3:
            action = np.asarray([0.0, 0.0, -0.45, 0.0, 0.0, 0.0, -1.0])
        else:
            target = np.asarray(observation["cream_cheese_1_pos"])
            action = action_toward(observation["robot0_eef_pos"], target, gripper=-1.0)
        if step_index >= event_time:
            action = _jitter_action(action, rng, 0.004)
        observation = _step(env, action)
        if video_frames is not None:
            video_frames.append(
                np.concatenate(
                    [
                        np.asarray(observation["agentview_image"]),
                        np.asarray(observation["robot0_eye_in_hand_image"]),
                    ],
                    axis=1,
                )
            )
        actions.append(action)
        object_trajectory.append(np.asarray(observation["cream_cheese_1_pos"]).copy())
        robot_state_trajectory.append(robot_state_from_observation(observation).copy())
    object_geoms = [env.sim.model.geom_id2name(index) for index in _object_geom_ids(env)]
    return (
        np.asarray(actions),
        {
            "final_object_position": np.asarray(observation["cream_cheese_1_pos"]).round(8).tolist(),
            "final_eef_position": np.asarray(observation["robot0_eef_pos"]).round(8).tolist(),
            "object_displacement": float(np.linalg.norm(object_trajectory[-1] - object_trajectory[0])),
            "grasped_at_end": bool(env.env._check_grasp(env.robots[0].gripper, object_geoms)),
            "object_trajectory": np.asarray(object_trajectory).round(8).tolist(),
            "robot_state_trajectory": np.asarray(robot_state_trajectory).round(8).tolist(),
            "pre_branch_image_max_abs_error": image_error,
            "pre_branch_state_max_abs_error": state_error,
        },
        restored_error,
    )


def _run_push_branch(
    env: Any,
    snapshot: np.ndarray,
    controller_state: Mapping[str, np.ndarray],
    branch: str,
    *,
    horizon: int,
    event_time: int,
    rng: np.random.Generator,
    geom_friction: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], float]:
    geom_ids = _object_geom_ids(env)
    env.sim.model.geom_friction[geom_ids] = geom_friction
    observation = _restore_snapshot(env, snapshot, controller_state)
    restored_error = float(np.max(np.abs(env.get_sim_state() - snapshot)))
    actions = []
    object_trajectory = [np.asarray(observation["cream_cheese_1_pos"]).copy()]
    for step_index in range(horizon):
        if step_index == event_time and branch == "blocked":
            blocked = geom_friction.copy()
            blocked[:, 0] = 25.0
            blocked[:, 1:] = np.maximum(blocked[:, 1:], 1.0)
            env.sim.model.geom_friction[geom_ids] = blocked
        if step_index < event_time or branch == "free_slide":
            action = np.asarray([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        elif step_index < event_time + 3:
            action = np.asarray([-0.5, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0])
        else:
            target = np.asarray(observation["cream_cheese_1_pos"]) + [0.0, 0.06, 0.0]
            action = action_toward(observation["robot0_eef_pos"], target, gripper=1.0)
        if step_index >= event_time:
            action = _jitter_action(action, rng, 0.004)
        observation = _step(env, action)
        actions.append(action)
        object_trajectory.append(np.asarray(observation["cream_cheese_1_pos"]).copy())
    env.sim.model.geom_friction[geom_ids] = geom_friction
    return (
        np.asarray(actions),
        {
            "final_object_position": np.asarray(observation["cream_cheese_1_pos"]).round(8).tolist(),
            "final_eef_position": np.asarray(observation["robot0_eef_pos"]).round(8).tolist(),
            "object_displacement": float(np.linalg.norm(object_trajectory[-1] - object_trajectory[0])),
            "object_trajectory": np.asarray(object_trajectory).round(8).tolist(),
        },
        restored_error,
    )


def _run_reach_branch(
    env: Any,
    snapshot: np.ndarray,
    controller_state: Mapping[str, np.ndarray],
    *,
    horizon: int,
    reference_observation: Mapping[str, Any] | None = None,
    video_frames: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any], float]:
    observation = _restore_snapshot(env, snapshot, controller_state)
    restored_error = float(np.max(np.abs(env.get_sim_state() - snapshot)))
    if reference_observation is None:
        image_error = 0.0
        state_error = 0.0
    else:
        image_error = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(observation["agentview_image"], dtype=np.int16)
                        - np.asarray(reference_observation["agentview_image"], dtype=np.int16)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.int16)
                        - np.asarray(reference_observation["robot0_eye_in_hand_image"], dtype=np.int16)
                    )
                )
            ),
        )
        state_error = float(
            np.max(
                np.abs(
                    robot_state_from_observation(observation)
                    - robot_state_from_observation(reference_observation)
                )
            )
        )
    start = np.asarray(observation["robot0_eef_pos"]).copy()
    target = start + [0.08, 0.05, -0.03]
    move_action = action_toward(start, target, gripper=-1.0)
    actions = []
    eef_trajectory = [start.copy()]
    robot_state_trajectory = [robot_state_from_observation(observation).copy()]
    if video_frames is not None:
        video_frames.append(
            np.concatenate(
                [
                    np.asarray(observation["agentview_image"]),
                    np.asarray(observation["robot0_eye_in_hand_image"]),
                ],
                axis=1,
            )
        )
    for step_index in range(horizon):
        # The deterministic control compares an identical current-state plan,
        # so its two counterfactual labels must not create different actions.
        action = move_action if step_index < horizon // 2 else np.asarray([0.0] * 6 + [-1.0])
        observation = _step(env, action)
        actions.append(action)
        eef_trajectory.append(np.asarray(observation["robot0_eef_pos"]).copy())
        robot_state_trajectory.append(robot_state_from_observation(observation).copy())
        if video_frames is not None:
            video_frames.append(
                np.concatenate(
                    [
                        np.asarray(observation["agentview_image"]),
                        np.asarray(observation["robot0_eye_in_hand_image"]),
                    ],
                    axis=1,
                )
            )
    return (
        np.asarray(actions),
        {
            "final_eef_position": np.asarray(observation["robot0_eef_pos"]).round(8).tolist(),
            "target_error": float(np.linalg.norm(np.asarray(observation["robot0_eef_pos"]) - target)),
            "eef_trajectory": np.asarray(eef_trajectory).round(8).tolist(),
            "robot_state_trajectory": np.asarray(robot_state_trajectory).round(8).tolist(),
            "pre_branch_image_max_abs_error": image_error,
            "pre_branch_state_max_abs_error": state_error,
        },
        restored_error,
    )


def _conditioning_hash(observation: Mapping[str, Any], robot_state: Sequence[float]) -> str:
    payload = json.dumps(
        {
            "observation": observation,
            "robot_state": list(robot_state),
            "language_instruction": LANGUAGE,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def collect(args: argparse.Namespace) -> tuple[list[CounterfactualRecord], dict[str, Any], dict[str, np.ndarray]]:
    from libero.libero.envs import OffScreenRenderEnv

    initial_states = np.load(args.init_states)
    if args.num_pairs <= 0 or args.num_pairs > len(initial_states):
        raise ValueError(f"num_pairs must be in [1, {len(initial_states)}]")
    if args.repeats < 2:
        raise ValueError("repeats must be at least 2")
    if args.horizon < 8:
        raise ValueError("horizon must be at least 8")

    env = OffScreenRenderEnv(
        bddl_file_name=str(args.bddl),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )
    env.seed(args.seed)
    records = []
    pair_metadata = []
    snapshots = {}
    max_restore_error = 0.0
    event_time = 2
    try:
        for task in ("grasp_slip", "blocked_push", "deterministic_reach"):
            for pair_index in range(args.num_pairs):
                initial_state = initial_states[pair_index]
                pair_id = f"libero-{task}-{pair_index:03d}"
                if task == "grasp_slip":
                    observation, snapshot, controller_state = _prepare_grasp_snapshot(
                        env, initial_state, args.settle_steps
                    )
                    branch_names = ("attached", "slipped")
                elif task == "blocked_push":
                    observation, snapshot, controller_state = _prepare_push_snapshot(
                        env, initial_state, args.settle_steps
                    )
                    branch_names = ("free_slide", "blocked")
                else:
                    observation, snapshot, controller_state = _prepare_reach_snapshot(
                        env, initial_state, args.settle_steps
                    )
                    branch_names = ("repeat_a", "repeat_b")

                snapshot_key = pair_id
                snapshots[f"{snapshot_key}_agentview"] = np.asarray(observation["agentview_image"])
                snapshots[f"{snapshot_key}_wrist"] = np.asarray(observation["robot0_eye_in_hand_image"])
                policy_observation = compact_policy_observation(observation, snapshot_key)
                robot_state = robot_state_from_observation(observation).round(8).tolist()
                conditioning_hash = _conditioning_hash(policy_observation, robot_state)
                branch_rollouts = {}
                branch_physics = {}
                default_friction = env.sim.model.geom_friction[_object_geom_ids(env)].copy()

                for branch_index, branch_name in enumerate(branch_names):
                    rollout_actions = []
                    rollout_physics = []
                    for repeat_index in range(args.repeats):
                        rng = np.random.default_rng(
                            args.seed + pair_index * 1009 + branch_index * 101 + repeat_index
                        )
                        if task == "grasp_slip":
                            actions, physics, restore_error = _run_grasp_branch(
                                env,
                                snapshot,
                                controller_state,
                                branch_name,
                                horizon=args.horizon,
                                event_time=event_time,
                                rng=rng,
                            )
                        elif task == "blocked_push":
                            actions, physics, restore_error = _run_push_branch(
                                env,
                                snapshot,
                                controller_state,
                                branch_name,
                                horizon=args.horizon,
                                event_time=event_time,
                                rng=rng,
                                geom_friction=default_friction,
                            )
                        else:
                            actions, physics, restore_error = _run_reach_branch(
                                env,
                                snapshot,
                                controller_state,
                                horizon=args.horizon,
                            )
                        max_restore_error = max(max_restore_error, restore_error)
                        rollout_actions.append(actions)
                        rollout_physics.append(physics)
                    branch_rollouts[branch_name] = np.stack(rollout_actions)
                    branch_physics[branch_name] = rollout_physics

                estimate = estimate_branch_divergence(branch_rollouts, persistence=2)
                sensitivity = threshold_sensitivity(branch_rollouts, persistence=2)
                trajectory_key = "eef_trajectory" if task == "deterministic_reach" else "object_trajectory"
                effect_rollouts = {
                    branch_name: np.stack(
                        [np.asarray(row[trajectory_key], dtype=np.float64)[1:] for row in branch_physics[branch_name]]
                    )
                    for branch_name in branch_names
                }
                effect_estimate = estimate_branch_divergence(effect_rollouts, persistence=2)
                physical_validation = validate_physical_branches(task, branch_physics)
                deterministic = task == "deterministic_reach"
                physical_effect_divergence_time = (
                    args.horizon if deterministic else effect_estimate.action_divergence_time
                )
                expected_horizon = args.horizon if deterministic else event_time
                if estimate.oracle_feedback_horizon != expected_horizon:
                    raise RuntimeError(
                        f"unstable boundary for {pair_id}: expected {expected_horizon}, "
                        f"got {estimate.oracle_feedback_horizon}"
                    )

                for branch_name in branch_names:
                    mean_actions = branch_rollouts[branch_name].mean(axis=0)
                    gripper_horizon = gripper_transition_horizon(mean_actions)
                    record = CounterfactualRecord(
                        pair_id=pair_id,
                        branch_id=branch_name,
                        branch_outcome=branch_name,
                        observation=policy_observation,
                        robot_state=robot_state,
                        language_instruction=LANGUAGE,
                        action_chunk=mean_actions.round(7).tolist(),
                        event_time=args.horizon if deterministic else event_time,
                        feedback_reveal_time=physical_effect_divergence_time,
                        action_divergence_time=estimate.action_divergence_time,
                        gripper_transition_horizon=gripper_horizon,
                        oracle_feedback_horizon=estimate.oracle_feedback_horizon,
                        per_step_branch_divergence=estimate.per_step_branch_divergence,
                        is_deterministic_control=deterministic,
                    )
                    validate_record(record)
                    build_policy_inputs(record)
                    records.append(record)

                pair_metadata.append(
                    {
                        "pair_id": pair_id,
                        "task": task,
                        "conditioning_sha256": conditioning_hash,
                        "snapshot_state_dim": int(snapshot.size),
                        "within_branch_threshold": estimate.within_branch_threshold,
                        "event_time": args.horizon if deterministic else event_time,
                        "feedback_reveal_time": physical_effect_divergence_time,
                        "action_divergence_time": estimate.action_divergence_time,
                        "oracle_feedback_horizon": estimate.oracle_feedback_horizon,
                        "physical_effect_divergence_time": physical_effect_divergence_time,
                        "physical_effect_per_step_divergence": effect_estimate.per_step_branch_divergence,
                        "physical_validation": physical_validation,
                        "gripper_transition_horizons": {
                            name: gripper_transition_horizon(branch_rollouts[name].mean(axis=0))
                            for name in branch_names
                        },
                        "threshold_sensitivity": {
                            multiplier: asdict(value) for multiplier, value in sensitivity.items()
                        },
                        "physics": branch_physics,
                    }
                )
    finally:
        env.close()

    task_summary = {}
    for task in ("grasp_slip", "blocked_push", "deterministic_reach"):
        rows = [row for row in pair_metadata if row["task"] == task]
        task_summary[task] = {
            "pair_count": len(rows),
            "mean_oracle_horizon": float(np.mean([row["oracle_feedback_horizon"] for row in rows])),
            "oracle_horizons": [row["oracle_feedback_horizon"] for row in rows],
            "feedback_reveal_times": [row["feedback_reveal_time"] for row in rows],
            "physical_effect_divergence_times": [row["physical_effect_divergence_time"] for row in rows],
        }
    manifest = {
        "generator": "libero_snapshot_collector.py",
        "seed": args.seed,
        "bddl": str(args.bddl),
        "horizon": args.horizon,
        "repeats": args.repeats,
        "pair_count": len(pair_metadata),
        "record_count": len(records),
        "max_snapshot_restore_abs_error": max_restore_error,
        "policy_input_fields": ["observation", "robot_state", "language_instruction"],
        "counterfactual_interventions": {
            "grasp_slip": "same grasped snapshot and lift prefix; forced object slip after the event",
            "blocked_push": "same closed-gripper push prefix with no transition; hidden friction increase after event",
            "deterministic_reach": "same free-space target with no intervention",
        },
        "task_summary": task_summary,
        "pairs": pair_metadata,
    }
    return records, manifest, snapshots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a small real-LIBERO snapshot counterfactual pilot")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/counterfactual-libero-snapshot"),
    )
    parser.add_argument("--bddl", type=Path, default=DEFAULT_BDDL)
    parser.add_argument("--init-states", type=Path, default=DEFAULT_INIT_STATES)
    parser.add_argument("--num-pairs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, manifest, snapshots = collect(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records)
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.output_dir / "policy_observation_snapshots.npz", **snapshots)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "pair_count": manifest["pair_count"],
                "record_count": manifest["record_count"],
                "max_snapshot_restore_abs_error": manifest["max_snapshot_restore_abs_error"],
                "task_summary": manifest["task_summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
