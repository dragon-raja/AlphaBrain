"""Bounded camera-acquisition execution, separate from the archived evaluator.

This is an engineering primitive, not a policy evaluator or an execution release.
Only an endpoint observation is exposed to the caller. Intermediate simulator
images may be rendered internally but are never supplied to a selector here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .motion import BudgetLedger, CameraTrajectory, Pose


def _pose_error(a: Pose, b: Pose) -> tuple[float, float]:
    position = math.sqrt(sum((x - y) ** 2 for x, y in zip(a.position, b.position)))
    dot = abs(sum(x * y for x, y in zip(a.quaternion_wxyz, b.quaternion_wxyz)))
    angle = 2.0 * math.acos(min(1.0, max(0.0, dot)))
    return position, angle


@dataclass(frozen=True)
class StepFeedback:
    sim_time: float
    actual_pose: Pose
    calibration: Mapping[str, Any]
    terminated: bool = False
    success: bool = False


@dataclass
class AcquisitionResult:
    status: str
    completed_env_steps: int = 0
    endpoint_queries: int = 0
    observation: Optional[Mapping[str, Any]] = field(default=None, repr=False)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    terminal_success: bool = False

    def audit_record(self) -> dict[str, Any]:
        """No images or privileged intermediate observations in the audit summary."""
        return {
            "status": self.status,
            "completed_env_steps": self.completed_env_steps,
            "endpoint_queries": self.endpoint_queries,
            "events": self.events,
            "error": self.error,
            "terminal_success": self.terminal_success,
        }


def execute_acquisition(
    trajectory: CameraTrajectory,
    backend: Any,
    ledger: BudgetLedger,
    *,
    initial_observation: Optional[Mapping[str, Any]] = None,
    position_tolerance: float = 1e-7,
    angle_tolerance: float = 1e-6,
    clock_tolerance: float = 1e-7,
) -> AcquisitionResult:
    """Execute a prevalidated trajectory through a small backend interface.

    Required backend methods: camera_pose(), sim_time(), step(pose), observe().
    Step budget is charged before attempted mutation; on an exception the run
    terminates, and completed_env_steps separately identifies known completions.
    No policy calls or RNG are consumed in this stage. A no-acquisition option
    performs no backend operations and returns the same initial observation.
    """
    for tolerance in (position_tolerance, angle_tolerance, clock_tolerance):
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("execution tolerances must be finite and nonnegative")
    result = AcquisitionResult(status="not_started")
    if ledger.termination is not None:
        result.status = "failed"
        result.error = "ledger already terminated: " + ledger.termination
        return result
    if trajectory.env_steps == 0:
        result.status = "continued_without_acquisition"
        result.observation = initial_observation
        return result
    try:
        start_error = _pose_error(backend.camera_pose(), trajectory.samples[0].pose)
        if start_error[0] > position_tolerance or start_error[1] > angle_tolerance:
            raise ValueError("trajectory start does not match current camera pose")
        start_time = float(backend.sim_time())
        if not math.isfinite(start_time):
            raise ValueError("nonfinite simulator clock")
        for sample in trajectory.samples[1:]:
            ledger.charge(env_steps=1)
            feedback = backend.step(sample.pose)
            result.completed_env_steps += 1
            position_error, angle_error = _pose_error(feedback.actual_pose, sample.pose)
            expected_time = start_time + result.completed_env_steps * trajectory.dt
            if not math.isfinite(feedback.sim_time):
                raise ValueError("nonfinite simulator clock after acquisition step")
            if abs(feedback.sim_time - expected_time) > clock_tolerance:
                raise ValueError("acquisition did not advance the expected simulator time")
            if position_error > position_tolerance or angle_error > angle_tolerance:
                raise ValueError("actual camera pose differs from acquisition command")
            calibration_text = json.dumps(feedback.calibration, sort_keys=True, allow_nan=False)
            result.events.append({
                "step": result.completed_env_steps,
                "sim_time": feedback.sim_time,
                "position": list(feedback.actual_pose.position),
                "quaternion_wxyz": list(feedback.actual_pose.quaternion_wxyz),
                "position_error": position_error,
                "angle_error_rad": angle_error,
                "calibration": dict(feedback.calibration),
                "calibration_sha256": hashlib.sha256(calibration_text.encode()).hexdigest(),
            })
            if feedback.terminated or feedback.success:
                result.status = "terminated_during_acquisition"
                result.terminal_success = bool(feedback.success)
                ledger.terminate("environment_terminated_during_acquisition")
                return result
        ledger.charge(queries=1)
        result.observation = backend.observe()
        result.endpoint_queries = 1
        result.status = "acquired"
    except Exception as exc:
        result.status = "failed"
        result.error = "%s: %s" % (type(exc).__name__, exc)
        # Preserve the failed attempt; never silently resume as a successful hold.
        try:
            ledger.terminate("acquisition_error")
        except Exception:
            pass
    return result


class PersistentOSCGoalHold:
    """Locally pin OSC targets while preserving Panda gripper command state.

    No global monkeypatch and no change to the archived evaluation code. The
    Original set_goal and grip_action bindings are restored on context exit,
    including exceptions. Gripper actuator controls are pinned explicitly: a
    zero normalized gripper command is NOT a reliable runtime-state hold.
    Only the inspected fixed-impedance, one-arm Panda OSC interface is supported.
    This holds a control target, not the exact physical state of the robot.
    """

    def __init__(self, env: Any, *, reference: str = "existing_targets") -> None:
        import numpy as np

        self.env = env
        self.robot = env.env.robots[0]
        self.controller = self.robot.controller
        if len(env.env.robots) != 1 or int(env.env.action_dim) != 7:
            raise ValueError("hold backend supports one 7-action Panda arm only")
        if self.controller.impedance_mode != "fixed":
            raise ValueError("hold backend requires fixed OSC impedance")
        if type(self.robot.gripper).__name__ != "PandaGripper":
            raise ValueError("gripper actuator hold only audited for PandaGripper")
        self.sim = env.env.sim
        self.gripper_actuator_ids = np.asarray(self.robot._ref_joint_gripper_actuator_indexes, dtype=int)
        if self.gripper_actuator_ids.shape != (2,) or len(set(self.gripper_actuator_ids)) != 2:
            raise ValueError("expected two distinct Panda gripper actuators")
        self.gripper_controls = np.asarray(self.sim.data.ctrl[self.gripper_actuator_ids]).copy()
        if not np.all(np.isfinite(self.gripper_controls)):
            raise ValueError("nonfinite gripper actuator controls")
        parameters = inspect.signature(self.controller.set_goal).parameters
        if "set_pos" not in parameters or "set_ori" not in parameters:
            raise ValueError("OSC controller lacks explicit target overrides")
        self.controller.update(force=True)
        if reference == "existing_targets":
            position, orientation = self.controller.goal_pos, self.controller.goal_ori
        elif reference == "current_pose":
            position, orientation = self.controller.ee_pos, self.controller.ee_ori_mat
        else:
            raise ValueError("unknown hold reference")
        self.position = np.asarray(position, dtype=float).copy()
        self.orientation = np.asarray(orientation, dtype=float).copy()
        if self.position.shape != (3,) or self.orientation.shape != (3, 3):
            raise ValueError("missing or malformed OSC hold targets")
        if not np.all(np.isfinite(self.position)) or not np.all(np.isfinite(self.orientation)):
            raise ValueError("nonfinite OSC hold targets")
        self.action = np.zeros(7, dtype=float)
        self.reference = reference
        self._original = None

    def __enter__(self) -> "PersistentOSCGoalHold":
        if self._original is not None:
            raise RuntimeError("hold context already active")
        self._had_instance_override = "set_goal" in vars(self.controller)
        self._had_grip_override = "grip_action" in vars(self.robot)
        self._original_grip = self.robot.grip_action
        self._original = self.controller.set_goal
        original = self._original

        def set_fixed_goal(action: Any, *args: Any, **kwargs: Any) -> Any:
            if args:
                raise ValueError("unexpected positional OSC target override")
            kwargs.update(set_pos=self.position.copy(), set_ori=self.orientation.copy())
            return original(action, **kwargs)

        def fixed_gripper_controls(gripper: Any, gripper_action: Any) -> None:
            if gripper is not self.robot.gripper:
                raise ValueError("unexpected gripper in hold context")
            self.sim.data.ctrl[self.gripper_actuator_ids] = self.gripper_controls

        self.controller.set_goal = set_fixed_goal
        self.robot.grip_action = fixed_gripper_controls
        return self

    def step(self) -> Any:
        if self._original is None:
            raise RuntimeError("hold context is not active")
        return self.env.step(self.action.copy())

    def __exit__(self, *_args: Any) -> None:
        if self._had_instance_override:
            self.controller.set_goal = self._original
        else:
            del self.controller.set_goal
        if self._had_grip_override:
            self.robot.grip_action = self._original_grip
        else:
            del self.robot.grip_action
        self._original = None


class LiberoKinematicCameraBackend:
    """World-body agentview adapter; not a head or wrist dynamics model.

    It samples a reference path at each real environment control step, refreshes
    calibration each step, and exposes only the requested endpoint observation.
    It does not simulate continuous camera actuator dynamics. Path feasibility
    remains the caller's contract: no collision proof here.
    """

    def __init__(self, env: Any, hold: PersistentOSCGoalHold) -> None:
        # LIBERO-plus applies optional image perturbations in its wrapper step,
        # whereas endpoint readout below uses raw cached observables. Do not
        # silently mix those preprocessing/RNG semantics in one experiment.
        if getattr(env, "noise", 0) != 0:
            raise ValueError("kinematic endpoint adapter requires LIBERO-plus noise=0")
        self.env = env
        self.hold = hold
        self.sim = env.env.sim
        self.camera_id = int(self.sim.model.camera_name2id("agentview"))
        if int(self.sim.model.cam_bodyid[self.camera_id]) != 0:
            raise ValueError("kinematic adapter requires a world-body agentview camera")
        if int(self.sim.model.cam_mode[self.camera_id]) != 0:
            raise ValueError("kinematic adapter requires fixed camera mode, not tracking")

    def camera_pose(self) -> Pose:
        from libero_camera_pose import rotation_matrix_to_wxyz

        return Pose(
            tuple(float(v) for v in self.sim.data.cam_xpos[self.camera_id]),
            tuple(float(v) for v in rotation_matrix_to_wxyz(
                self.sim.data.cam_xmat[self.camera_id].reshape(3, 3)
            )),
        )

    def sim_time(self) -> float:
        return float(self.sim.data.time)

    def calibration(self) -> dict[str, Any]:
        from evaluate_pi05_libero_plus_views import agentview_camera_calibration

        return {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in agentview_camera_calibration(self.env).items()
        }

    def step(self, pose: Pose) -> StepFeedback:
        self.sim.model.cam_pos[self.camera_id] = pose.position
        self.sim.model.cam_quat[self.camera_id] = pose.quaternion_wxyz
        self.sim.forward()
        _observation, _reward, done, _info = self.hold.step()
        return StepFeedback(
            sim_time=self.sim_time(),
            actual_pose=self.camera_pose(),
            calibration=self.calibration(),
            terminated=bool(done),
            success=bool(self.env.check_success()),
        )

    def observe(self) -> Mapping[str, Any]:
        self.env.env._update_observables(force=True)
        return self.env.env._get_observations()
