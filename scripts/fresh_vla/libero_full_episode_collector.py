from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from libero_snapshot_collector import (
    _capture_controller_state,
    _object_geom_ids,
    _reset_to_initial_state,
    _restore_snapshot,
    _set_object_offset,
    _step,
    action_toward,
    robot_state_from_observation,
)


@dataclass(frozen=True)
class TeacherConfig:
    position_tolerance: float = 0.012
    lift_height: float = 0.16
    carry_height: float = 0.17
    release_height: float = 0.065
    recovery_approach_height: float = 0.12
    recovery_grasp_offset: float = 0.007
    release_steps: int = 18
    recovery_open_steps: int = 12
    recovery_close_steps: int = 30
    retract_steps: int = 12
    max_regrasp_attempts: int = 3
    max_place_attempts: int = 3


@dataclass(frozen=True)
class TeacherDecision:
    action: np.ndarray
    phase: str
    recovering: bool


class FullEpisodeTeacher:
    """One feedback-driven teacher used for both attached and slipped outcomes."""

    def __init__(self, observation: Mapping[str, Any], config: TeacherConfig | None = None):
        self.config = config or TeacherConfig()
        self.phase = "lift"
        self.phase_steps = 0
        self.regrasp_attempts = 0
        self.place_attempts = 0
        self.done = False
        self.initial_eef_xy = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)[:2].copy()
        self.initial_object_z = float(observation["cream_cheese_1_pos"][2])

    @property
    def recovering(self) -> bool:
        return self.phase.startswith("recover_") or self.regrasp_attempts > 0

    def _transition(self, phase: str) -> None:
        if phase != self.phase:
            self.phase = phase
            self.phase_steps = 0

    def _at(self, current: Sequence[float], target: Sequence[float]) -> bool:
        return bool(
            np.linalg.norm(np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64))
            <= self.config.position_tolerance
        )

    @staticmethod
    def _hold(gripper: float) -> np.ndarray:
        return np.asarray([0.0] * 6 + [gripper], dtype=np.float64)

    def decide(
        self,
        observation: Mapping[str, Any],
        *,
        grasped: bool,
        success: bool,
    ) -> TeacherDecision:
        if self.done:
            return TeacherDecision(self._hold(-1.0), "done", self.recovering)

        eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        obj = np.asarray(observation["cream_cheese_1_pos"], dtype=np.float64)
        bowl = np.asarray(observation["akita_black_bowl_1_pos"], dtype=np.float64)

        if success and self.phase not in {"retract", "done"}:
            self._transition("retract")

        while True:
            if self.phase in {"lift", "transport", "lower"} and not grasped:
                self._transition("recover_open")
                continue

            if self.phase == "lift":
                target = np.asarray(
                    [self.initial_eef_xy[0], self.initial_eef_xy[1], max(self.initial_object_z + self.config.lift_height, bowl[2] + self.config.carry_height)]
                )
                if self._at(eef, target):
                    self._transition("transport")
                    continue
                action = action_toward(eef, target, gripper=1.0)
                break

            if self.phase == "transport":
                target = bowl + np.asarray([0.0, 0.0, self.config.carry_height])
                if self._at(eef, target):
                    self._transition("lower")
                    continue
                action = action_toward(eef, target, gripper=1.0)
                break

            if self.phase == "lower":
                target = bowl + np.asarray([0.0, 0.0, self.config.release_height])
                if self._at(eef, target):
                    self._transition("release")
                    continue
                action = action_toward(eef, target, gripper=1.0)
                break

            if self.phase == "release":
                if self.phase_steps >= self.config.release_steps:
                    self.place_attempts += 1
                    self._transition("retract")
                    continue
                action = self._hold(-1.0)
                break

            if self.phase == "recover_open":
                if self.phase_steps >= self.config.recovery_open_steps:
                    self._transition("recover_above")
                    continue
                action = self._hold(-1.0)
                break

            if self.phase == "recover_above":
                target = obj + np.asarray([0.0, 0.0, self.config.recovery_approach_height])
                if self._at(eef, target):
                    self._transition("recover_descend")
                    continue
                action = action_toward(eef, target, gripper=-1.0)
                break

            if self.phase == "recover_descend":
                target = obj + np.asarray([0.0, 0.0, self.config.recovery_grasp_offset])
                if self._at(eef, target):
                    self._transition("recover_close")
                    continue
                action = action_toward(eef, target, gripper=-1.0)
                break

            if self.phase == "recover_close":
                if grasped and self.phase_steps >= 8:
                    self.regrasp_attempts += 1
                    self.initial_eef_xy = eef[:2].copy()
                    self.initial_object_z = float(obj[2])
                    self._transition("lift")
                    continue
                if self.phase_steps >= self.config.recovery_close_steps:
                    self.regrasp_attempts += 1
                    if self.regrasp_attempts >= self.config.max_regrasp_attempts:
                        self.done = True
                        return TeacherDecision(self._hold(-1.0), "failed_regrasp", True)
                    self._transition("recover_open")
                    continue
                action = self._hold(1.0)
                break

            if self.phase == "retract":
                target = np.asarray([eef[0], eef[1], bowl[2] + self.config.carry_height])
                if self.phase_steps >= self.config.retract_steps and self._at(eef, target):
                    if success:
                        self.done = True
                        return TeacherDecision(self._hold(-1.0), "done", self.recovering)
                    if self.place_attempts >= self.config.max_place_attempts:
                        self.done = True
                        return TeacherDecision(self._hold(-1.0), "failed_place", self.recovering)
                    self._transition("recover_open")
                    continue
                action = action_toward(eef, target, gripper=-1.0)
                break

            raise RuntimeError(f"unknown teacher phase: {self.phase}")

        decision = TeacherDecision(action=np.asarray(action, dtype=np.float64), phase=self.phase, recovering=self.recovering)
        self.phase_steps += 1
        return decision


def upright_image(image: np.ndarray) -> np.ndarray:
    return np.flip(np.asarray(image), axis=(0, 1)).copy()


def object_gripper_contact(env: Any) -> bool:
    object_ids = set(_object_geom_ids(env))
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        if contact.geom1 not in object_ids and contact.geom2 not in object_ids:
            continue
        other = contact.geom2 if contact.geom1 in object_ids else contact.geom1
        name = env.sim.model.geom_id2name(other) or ""
        if "gripper" in name:
            return True
    return False


def object_grasped(env: Any) -> bool:
    object_geoms = [env.sim.model.geom_id2name(index) for index in _object_geom_ids(env)]
    return bool(env.env._check_grasp(env.robots[0].gripper, object_geoms))


def object_finger_contacts(env: Any) -> dict[str, bool]:
    gripper = env.robots[0].gripper
    object_geoms = [env.sim.model.geom_id2name(index) for index in _object_geom_ids(env)]
    return {
        side: bool(env.env.check_contact(gripper.important_geoms[f"{side}_fingerpad"], object_geoms))
        for side in ("left", "right")
    }


class TraceRecorder:
    def __init__(self, env: Any, observation: Mapping[str, Any], phase: str):
        self.observations: dict[str, list[np.ndarray | bool | str]] = {
            "agentview": [],
            "wrist": [],
            "robot_state": [],
            "eef_pose": [],
            "object_pose": [],
            "gripper_qpos": [],
            "gripper_action": [],
            "grasped": [],
            "contact": [],
            "success": [],
            "sim_state": [],
            "teacher_phase": [],
        }
        self.actions: list[np.ndarray] = []
        self.action_phases: list[str] = []
        self._append_observation(env, observation, phase)

    def _append_observation(self, env: Any, observation: Mapping[str, Any], phase: str) -> None:
        self.observations["agentview"].append(upright_image(observation["agentview_image"]))
        self.observations["wrist"].append(upright_image(observation["robot0_eye_in_hand_image"]))
        self.observations["robot_state"].append(robot_state_from_observation(observation).astype(np.float32))
        self.observations["eef_pose"].append(
            np.concatenate([observation["robot0_eef_pos"], observation["robot0_eef_quat"]]).astype(np.float32)
        )
        self.observations["object_pose"].append(
            np.concatenate([observation["cream_cheese_1_pos"], observation["cream_cheese_1_quat"]]).astype(np.float32)
        )
        self.observations["gripper_qpos"].append(np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32))
        self.observations["gripper_action"].append(
            np.asarray(env.robots[0].gripper.current_action, dtype=np.float32).reshape(-1)
        )
        self.observations["grasped"].append(object_grasped(env))
        self.observations["contact"].append(object_gripper_contact(env))
        self.observations["success"].append(bool(env.check_success()))
        self.observations["sim_state"].append(np.asarray(env.get_sim_state(), dtype=np.float64).copy())
        self.observations["teacher_phase"].append(phase)

    def replace_current(self, env: Any, observation: Mapping[str, Any], phase: str) -> None:
        for values in self.observations.values():
            values.pop()
        self._append_observation(env, observation, phase)

    def step(self, env: Any, action: Sequence[float], phase: str) -> Mapping[str, Any]:
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.action_phases.append(phase)
        observation = _step(env, action)
        self._append_observation(env, observation, phase)
        return observation

    def arrays(self) -> dict[str, np.ndarray]:
        arrays = {key: np.asarray(values) for key, values in self.observations.items()}
        arrays["actions"] = np.asarray(self.actions, dtype=np.float32)
        arrays["action_phases"] = np.asarray(self.action_phases, dtype="U32")
        return arrays


def _record_move_to(
    env: Any,
    recorder: TraceRecorder,
    observation: Mapping[str, Any],
    target: Sequence[float],
    *,
    gripper: float,
    phase: str,
    tolerance: float = 0.008,
    max_steps: int = 80,
) -> Mapping[str, Any]:
    target_array = np.asarray(target, dtype=np.float64)
    for _ in range(max_steps):
        if np.linalg.norm(target_array - np.asarray(observation["robot0_eef_pos"])) <= tolerance:
            return observation
        action = action_toward(observation["robot0_eef_pos"], target_array, gripper=gripper)
        observation = recorder.step(env, action, phase)
    raise RuntimeError(f"teacher failed to reach {phase} target within {max_steps} steps")


def collect_grasp_prefix(
    env: Any,
    initial_state: np.ndarray,
    *,
    settle_steps: int,
    object_offset: Sequence[float],
    grasp_offset: Sequence[float],
    friction_scale: float,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any], np.ndarray, dict[str, np.ndarray]]:
    if friction_scale <= 0:
        raise ValueError("friction_scale must be positive")
    observation = _reset_to_initial_state(env, initial_state, settle_steps)
    if np.linalg.norm(np.asarray(object_offset, dtype=np.float64)) > 0:
        observation = _set_object_offset(env, object_offset)
        for _ in range(5):
            observation = _step(env, [0.0] * 6 + [-1.0])
    object_ids = _object_geom_ids(env)
    env.sim.model.geom_friction[object_ids] *= friction_scale

    recorder = TraceRecorder(env, observation, "episode_start")
    obj = np.asarray(observation["cream_cheese_1_pos"], dtype=np.float64)
    target = obj + np.asarray(grasp_offset, dtype=np.float64)
    observation = _record_move_to(
        env,
        recorder,
        observation,
        target + np.asarray([0.0, 0.0, 0.12]),
        gripper=-1.0,
        phase="approach_above",
    )
    observation = _record_move_to(
        env,
        recorder,
        observation,
        target,
        gripper=-1.0,
        phase="approach_grasp",
    )
    for _ in range(25):
        observation = recorder.step(env, [0.0] * 6 + [1.0], "close_gripper")
    if not object_grasped(env):
        raise RuntimeError("full-episode scripted grasp calibration failed")
    return recorder.arrays(), observation, env.get_sim_state().copy(), _capture_controller_state(env)


def collect_branch_continuation(
    env: Any,
    snapshot: np.ndarray,
    controller_state: Mapping[str, np.ndarray],
    *,
    outcome: str,
    slip_offset: Sequence[float],
    max_steps: int,
    lift_trigger_height: float = 0.015,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if outcome not in {"attached", "slipped"}:
        raise ValueError(f"unknown outcome: {outcome}")
    observation = _restore_snapshot(env, snapshot, controller_state)
    initial_object_z = float(observation["cream_cheese_1_pos"][2])
    teacher = FullEpisodeTeacher(observation)
    recorder = TraceRecorder(env, observation, teacher.phase)
    event_time = None
    feedback_reveal_time = None
    recovery_action_time = None
    regrasp_diagnostics = []

    for _ in range(max_steps):
        grasped = object_grasped(env)
        if teacher.phase == "recover_close" and teacher.phase_steps in {0, teacher.config.recovery_close_steps - 1}:
            regrasp_diagnostics.append(
                {
                    "attempt": teacher.regrasp_attempts,
                    "close_step": teacher.phase_steps,
                    "grasped": grasped,
                    "contact": object_gripper_contact(env),
                    "finger_contacts": object_finger_contacts(env),
                    "eef_minus_object": (
                        np.asarray(observation["robot0_eef_pos"])
                        - np.asarray(observation["cream_cheese_1_pos"])
                    ).round(5).tolist(),
                    "gripper_qpos": np.asarray(observation["robot0_gripper_qpos"]).round(5).tolist(),
                }
            )
        object_lift = float(observation["cream_cheese_1_pos"][2]) - initial_object_z
        if event_time is None and teacher.phase == "lift" and grasped and object_lift >= lift_trigger_height:
            event_time = len(recorder.actions)
            if outcome == "slipped":
                observation = _set_object_offset(env, slip_offset)
                recorder.replace_current(env, observation, teacher.phase)
                feedback_reveal_time = event_time

        success = bool(env.check_success())
        decision = teacher.decide(observation, grasped=object_grasped(env), success=success)
        if decision.recovering and recovery_action_time is None:
            recovery_action_time = len(recorder.actions)
        if teacher.done:
            break
        observation = recorder.step(env, decision.action, decision.phase)
    else:
        raise RuntimeError(
            f"{outcome} teacher exceeded max_steps={max_steps}: "
            f"phase={teacher.phase}, regrasp_attempts={teacher.regrasp_attempts}, "
            f"place_attempts={teacher.place_attempts}, grasped={object_grasped(env)}, "
            f"eef={np.asarray(observation['robot0_eef_pos']).round(5).tolist()}, "
            f"object={np.asarray(observation['cream_cheese_1_pos']).round(5).tolist()}, "
            f"object_quat={np.asarray(observation['cream_cheese_1_quat']).round(5).tolist()}, "
            f"eef_quat={np.asarray(observation['robot0_eef_quat']).round(5).tolist()}, "
            f"bowl={np.asarray(observation['akita_black_bowl_1_pos']).round(5).tolist()}, "
            f"regrasp_diagnostics={regrasp_diagnostics}"
        )

    final_success = bool(env.check_success())
    if not final_success:
        raise RuntimeError(
            f"{outcome} teacher ended without LIBERO task success: "
            f"phase={teacher.phase}, regrasp_attempts={teacher.regrasp_attempts}, "
            f"place_attempts={teacher.place_attempts}, grasped={object_grasped(env)}, "
            f"eef={np.asarray(observation['robot0_eef_pos']).round(5).tolist()}, "
            f"object={np.asarray(observation['cream_cheese_1_pos']).round(5).tolist()}, "
            f"object_quat={np.asarray(observation['cream_cheese_1_quat']).round(5).tolist()}, "
            f"eef_quat={np.asarray(observation['robot0_eef_quat']).round(5).tolist()}, "
            f"bowl={np.asarray(observation['akita_black_bowl_1_pos']).round(5).tolist()}, "
            f"regrasp_diagnostics={regrasp_diagnostics}"
        )
    if event_time is None:
        raise RuntimeError(f"{outcome} branch never reached the contact-triggered lift event")
    if outcome == "attached":
        feedback_reveal_time = event_time
    return recorder.arrays(), {
        "outcome": outcome,
        "event_time": event_time,
        "feedback_reveal_time": feedback_reveal_time,
        "recovery_action_time": recovery_action_time,
        "final_success": final_success,
        "steps": len(recorder.actions),
        "regrasp_attempts": teacher.regrasp_attempts,
        "regrasp_diagnostics": regrasp_diagnostics,
    }


def merge_prefix_and_continuation(
    prefix: Mapping[str, np.ndarray], continuation: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    result = {}
    for key in prefix:
        if key in {"actions", "action_phases"}:
            result[key] = np.concatenate([prefix[key], continuation[key]], axis=0)
        else:
            result[key] = np.concatenate([prefix[key][:-1], continuation[key]], axis=0)
    if len(result["agentview"]) != len(result["actions"]) + 1:
        raise RuntimeError("merged episode does not satisfy observation/action alignment")
    return result


def first_persistent_action_divergence(
    attached_actions: np.ndarray,
    slipped_actions: np.ndarray,
    *,
    threshold: float = 1e-6,
    persistence: int = 2,
) -> int:
    common = min(len(attached_actions), len(slipped_actions))
    distances = np.max(
        np.abs(np.asarray(attached_actions[:common]) - np.asarray(slipped_actions[:common])), axis=1
    )
    for index in range(max(0, common - persistence + 1)):
        if np.all(distances[index : index + persistence] > threshold):
            return index
    return common


def first_visual_reveal(
    attached: Mapping[str, np.ndarray],
    slipped: Mapping[str, np.ndarray],
    *,
    start: int,
) -> int:
    common = min(len(attached["agentview"]), len(slipped["agentview"]))
    for index in range(start, common):
        agent_diff = np.max(
            np.abs(attached["agentview"][index].astype(np.int16) - slipped["agentview"][index].astype(np.int16))
        )
        wrist_diff = np.max(
            np.abs(attached["wrist"][index].astype(np.int16) - slipped["wrist"][index].astype(np.int16))
        )
        if max(agent_diff, wrist_diff) > 0:
            return index
    return common
