"""Explicit GR1 task construction and reusable search/operation ports.

The caller supplies every scientific choice and the model process. This is not
a scheduler or an activation permit. The projected fixed-scene protocol remains
a named candidate until frozen; CPU fixture images are always labeled as such.
"""

import numpy as np
from .task_runtime import create_task
from .head_io import MujocoHeadIO
from .head_servo import HeadServo
from .head_geometry import HeadGeometry
from .head_control import ProjectedSceneAdvance, SubstepHeadHold
from .task_backend import HeadTaskBackend, EgoFrame
from .operation_backend import NativeEgoCapture, SelectedTaskBackend
from .search_runtime import ArticulatedHeadMount, GR1SearchSource


class TaskPorts:
    def __init__(
        self,
        spec,
        *,
        controller_config,
        allowed_empty_categories,
        servo_settings,
        search_config,
        initial_angles,
        roll_hold,
        scene_protocol,
        frame_mode,
        guard,
        render_guard,
        clock,
    ):
        if not spec.active_head:
            raise ValueError("Search requires explicit active head extension")
        if frame_mode not in ("CPU_SYNTHETIC", "NATIVE_EGO"):
            raise ValueError("Explicit frame mode required")
        if frame_mode == "NATIVE_EGO":
            render_guard()
        self.env = None
        self.frame_mode = frame_mode
        try:
            import mujoco

            self.env = create_task(
                spec,
                guard=guard,
                controller_config=controller_config,
                allowed_empty_categories=allowed_empty_categories,
            )
            env = self.env
            m, d = env.sim.model._model, env.sim.data._data
            self.io = MujocoHeadIO(
                mujoco, m, d, naming_prefix="robot0_", advance=lambda: self.advance(), guard=guard
            )
            self.advance = ProjectedSceneAdvance(
                mujoco, m, d, self.io, protocol=scene_protocol, guard=guard
            )
            self.servo = HeadServo(
                self.io,
                self.io.layout,
                servo_settings,
                roll_hold=roll_hold,
                guard=guard,
                clock=clock,
            )
            self.geometry = HeadGeometry(
                mujoco, m, d, camera_name="egoview", naming_prefix="robot0_", roll_hold=roll_hold
            )
            if frame_mode == "CPU_SYNTHETIC":

                def capture(epoch, sequence):
                    guard()
                    state = self.io.read_head_state()
                    q = np.rad2deg(state["positions"])
                    return EgoFrame(
                        np.zeros((800, 1280, 3), np.uint8),
                        epoch,
                        sequence,
                        state["observed_at"],
                        self.geometry.measured_pose(q[[0, 2]], q[1]),
                    )

            else:
                capture = NativeEgoCapture(
                    env, self.io, self.geometry, guard=guard, render_guard=render_guard
                )
            self.search = HeadTaskBackend(
                env, self.io, self.servo, self.geometry, capture, guard=guard
            )
            self.initial = self.search.move_head_degrees(initial_angles)
            self.mount = ArticulatedHeadMount(
                search_config,
                self.geometry,
                joint_limits_deg=self.geometry.yaw_pitch_limits_deg,
                measured_forward_kinematics=self.geometry.measured_pose,
            )
            self.control = SubstepHeadHold(
                self.io,
                servo_settings,
                target_rad=self.io.read_head_state()["positions"],
                guard=guard,
            )
            env.search_head_control = self.control
            self.operation = SelectedTaskBackend(
                env,
                self.io,
                self.geometry,
                capture,
                guard=guard,
                head_check=self.control.check,
                on_select=self.control.set_target,
                step_controlled=env.step,
                position_tolerance_rad=servo_settings.position_tolerance,
            )
        except BaseException:
            self.close()
            raise

    def source(self, runtime, *, noise_bank, episode_id, provenance, guard):
        return GR1SearchSource(
            self.mount,
            self.search,
            runtime,
            noise_bank=noise_bank,
            episode_id=episode_id,
            model_sha256=provenance["model_sha256"],
            producer_sha256=provenance["producer_sha256"],
            history_sha256=provenance["history_sha256"],
            guard=guard,
        )

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
