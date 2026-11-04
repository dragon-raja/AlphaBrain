"""Explicit native head control and a fixed-scene simulation candidate.

Neither class selects scientific defaults. ProjectedSceneAdvance is an explicit
simulation intervention: other integration fields are restored after each
head-dynamics step. It is NOT unmodified scene physics, a collision certificate,
or a real-robot controller. Its use must be named in the frozen experiment.
"""

import numpy as np


class SubstepHeadHold:
    """Native effort PD applied at every robosuite physics substep."""

    def __init__(self, io, settings, *, target_rad, guard):
        settings.validate()
        self.io, self.settings, self.guard = io, settings, guard
        self.set_target(target_rad)

    def set_target(self, target_rad):
        q = np.asarray(target_rad, dtype=float)
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError("Three finite native joint targets required")
        self.io.layout.named_targets((q[0], q[2]), roll_hold=q[1])
        self.target = q.copy()

    def apply(self):
        self.guard()
        state = self.io.read_head_state()
        q, v = np.asarray(state["positions"]), np.asarray(state["velocities"])
        if (
            q.shape != (3,)
            or v.shape != (3,)
            or not np.isfinite(q).all()
            or not np.isfinite(v).all()
        ):
            raise ValueError("Finite native head state required")
        cfg = self.settings
        efforts = np.clip(
            np.asarray(cfg.kp) * (self.target - q) - np.asarray(cfg.kd) * v,
            -np.asarray(cfg.effort_limit),
            np.asarray(cfg.effort_limit),
        )
        self.io.write_head_efforts(dict(zip(self.io.layout.actuator_names, efforts)))

    def check(self, expected_target):
        self.guard()
        if not np.array_equal(self.target, np.asarray(expected_target)):
            raise ValueError("Operation head controller target differs from selected pose")


class ProjectedSceneAdvance:
    """Named candidate: freeze non-head integration, advance only head + clock."""

    protocol = "PROJECT_NONHEAD_INTEGRATION_V1"

    def __init__(self, api, model, data, io, *, protocol, guard):
        if protocol != self.protocol:
            raise ValueError("Explicit projected-scene protocol required")
        self.api, self.model, self.data, self.io, self.guard = api, model, data, io, guard
        self.spec = api.mjtState.mjSTATE_INTEGRATION
        self.reference = np.empty(api.mj_stateSize(model, self.spec))
        api.mj_getState(model, data, self.reference, self.spec)
        self.scene = io.non_head_state_identity()
        self.steps = 0

    def __call__(self):
        self.guard()
        if self.io.non_head_state_identity() != self.scene:
            raise ValueError("Scene changed before fixed-scene head step")
        self.api.mj_step(self.model, self.data)
        q = self.data.qpos[self.io.qpos].copy()
        v = self.data.qvel[self.io.qvel].copy()
        warm = self.data.qacc_warmstart[self.io.qvel].copy()
        ctrl = self.data.ctrl[self.io.actuators].copy()
        stamp = float(self.data.time)
        self.api.mj_setState(self.model, self.data, self.reference, self.spec)
        self.data.qpos[self.io.qpos] = q
        self.data.qvel[self.io.qvel] = v
        self.data.qacc_warmstart[self.io.qvel] = warm
        self.data.ctrl[self.io.actuators] = ctrl
        self.data.time = stamp
        # Refresh kinematics without silently changing integration warm starts.
        updated = np.empty_like(self.reference)
        self.api.mj_getState(self.model, self.data, updated, self.spec)
        self.api.mj_forward(self.model, self.data)
        self.api.mj_setState(self.model, self.data, updated, self.spec)
        if self.io.non_head_state_identity() != self.scene:
            raise ValueError("Projection failed to hold scene")
        self.steps += 1
        self.guard()
