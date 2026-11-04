"""Actual task observation/action mapping after an explicit selected-view handoff.

Head tracking is supplied by the task's admitted substep controller. This adapter
checks it, never substitutes a camera teleport or a fabricated wrist image. The
same synchronous renderer contract serves search and operation observations.
"""

import numpy as np
from .operation_io import environment_action, model_input
from .task_backend import EgoFrame


class NativeEgoCapture:
    def __init__(self, env, io, geometry, *, guard, render_guard):
        self.env, self.io, self.geometry = env, io, geometry
        self.guard, self.render_guard = guard, render_guard

    def __call__(self, epoch, sequence):
        self.guard()
        self.render_guard()
        state = self.io.read_head_state()
        angles = np.rad2deg(state["positions"])
        scene = self.io.non_head_state_identity()
        pose = self.geometry.measured_pose(angles[[0, 2]], angles[1])
        pixels = self.env.sim.render(width=1280, height=800, camera_name="egoview", depth=False)
        after = self.io.read_head_state()
        if (
            state != after
            or scene != self.io.non_head_state_identity()
            or not np.array_equal(pose, self.geometry.measured_pose(angles[[0, 2]], angles[1]))
        ):
            raise ValueError("Rendering mutated head or scene")
        self.guard()
        return EgoFrame(np.asarray(pixels), epoch, sequence, state["observed_at"], pose)


class SelectedTaskBackend:
    def __init__(
        self,
        env,
        io,
        geometry,
        capture_frame,
        *,
        guard,
        head_check,
        step_controlled,
        position_tolerance_rad,
        input_builder=model_input,
        action_builder=environment_action,
        on_select=None,
    ):
        if not np.isfinite(position_tolerance_rad) or position_tolerance_rad <= 0:
            raise ValueError("Explicit head tracking tolerance required")
        self.env, self.io, self.geometry, self.capture = env, io, geometry, capture_frame
        self.guard, self.head_check, self.step_controlled = guard, head_check, step_controlled
        self.input_builder, self.action_builder = input_builder, action_builder
        self.tolerance = position_tolerance_rad
        self.status = "NEW"
        self.sequence = 0
        self.on_select = on_select

    def accept_selected_view(self, selected, *, instruction, epoch):
        if self.status != "NEW":
            raise RuntimeError("Selected view already bound")
        try:
            self.guard()
            state = self.io.read_head_state()
            q = np.asarray(state["positions"])
            actual = self.geometry.measured_pose(np.rad2deg(q[[0, 2]]), np.rad2deg(q[1]))
            if (
                not selected.settled
                or selected.physical_state_id != self.io.non_head_state_identity()
                or not np.array_equal(actual, selected.camera_to_world)
            ):
                raise ValueError("Selected view handoff changed scene/head")
            self.target = q.copy()
            self.epoch = epoch
            self.instruction = instruction
            if self.on_select is not None:
                self.on_select(self.target.copy())
            self.status = "BOUND"
            self.head_check(self.target.copy())
        except BaseException:
            self.status = "HOLD"
            raise

    def _live(self):
        if self.status not in ("BOUND", "RUNNING"):
            raise RuntimeError("Operation backend not live")
        self.guard()
        self.head_check(self.target.copy())
        q = np.asarray(self.io.read_head_state()["positions"])
        if (
            q.shape != (3,)
            or not np.isfinite(q).all()
            or np.max(np.abs(q - self.target)) > self.tolerance
        ):
            raise ValueError("Selected head joint tracking drift")

    def begin_selected_view(self, epoch):
        if self.status != "BOUND" or epoch != self.epoch:
            raise ValueError("Unbound operation epoch")
        self._live()
        self.status = "RUNNING"

    def capture_input(self, epoch):
        try:
            self._live()
            if epoch != self.epoch:
                raise ValueError("Wrong operation cache epoch")
            stamp = self.io.read_head_state()["observed_at"]
            scene = self.io.non_head_state_identity()
            q = np.asarray(self.io.read_head_state()["positions"])
            pose = self.geometry.measured_pose(np.rad2deg(q[[0, 2]]), np.rad2deg(q[1]))
            self.sequence += 1
            frame = self.capture(epoch, self.sequence)
            if (
                not isinstance(frame, EgoFrame)
                or frame.epoch != epoch
                or frame.sequence != self.sequence
                or frame.observed_at != stamp
                or not np.array_equal(frame.camera_to_world, pose)
            ):
                raise ValueError("Stale operation frame")
            example = self.input_builder(self.env, frame.pixels)
            final_q = np.asarray(self.io.read_head_state()["positions"])
            if (
                example["prompt"] != self.instruction
                or scene != self.io.non_head_state_identity()
                or self.io.read_head_state()["observed_at"] != stamp
                or not np.array_equal(q, final_q)
            ):
                raise ValueError("Operation input changed task or state")
            self._live()
            return example, epoch
        except BaseException:
            self.status = "HOLD"
            raise

    def step_absolute(self, action):
        try:
            self._live()
            result = self.step_controlled(self.action_builder(self.env, action))
            if not isinstance(result, tuple) or len(result) != 4:
                raise ValueError("Native task step result required")
            _, _, done, _ = result
            success = self.env._check_success()
            if not isinstance(success, (bool, np.bool_)) or not isinstance(done, (bool, np.bool_)):
                raise ValueError("Native boolean success and termination required")
            self._live()
            if done or success:
                self.status = "COMPLETE"
            return dict(success=bool(success), terminated=bool(done))
        except BaseException:
            self.status = "HOLD"
            raise
