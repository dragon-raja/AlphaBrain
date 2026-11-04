"""Actual MuJoCo head FK using scratch data, never stepping the live scene.

This is geometry, not a motor controller or collision/safety certificate.
Callers keep rendering, physical servo, and model admission separate.
"""

import numpy as np


class HeadGeometry:
    def __init__(self, api, model, data, *, camera_name, naming_prefix, roll_hold):
        self.api, self.model, self.data = api, model, data
        self.camera_id = api.mj_name2id(model, api.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id < 0:
            raise ValueError("Actual ego camera missing")
        self.joint_ids = [
            api.mj_name2id(model, api.mjtObj.mjOBJ_JOINT, naming_prefix + name)
            for name in ("head_yaw", "head_roll", "head_pitch")
        ]
        if min(self.joint_ids) < 0:
            raise ValueError("Articulated head joints missing")
        if any(model.jnt_type[j] != api.mjtJoint.mjJNT_HINGE for j in self.joint_ids):
            raise ValueError("Expected three native head hinges")
        if not all(model.jnt_limited[j] for j in self.joint_ids):
            raise ValueError("Explicit native mechanical limits required")
        self.addresses = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.limits = np.array(model.jnt_range[self.joint_ids], copy=True)
        if not np.isfinite(roll_hold) or not self.limits[1, 0] <= roll_hold <= self.limits[1, 1]:
            raise ValueError("Explicit legal roll hold required")
        self.roll_hold = float(roll_hold)
        self.scratch = api.MjData(model)
        self.spec = api.mjtState.mjSTATE_INTEGRATION
        self.size = api.mj_stateSize(model, self.spec)

    @property
    def yaw_pitch_limits_deg(self):
        return np.rad2deg(self.limits[[0, 2]])

    def __call__(self, yaw_pitch_deg):
        return self.measured_pose(yaw_pitch_deg, np.rad2deg(self.roll_hold))

    def measured_pose(self, yaw_pitch_deg, roll_deg):
        """FK for all measured joints, including residual held-roll error."""
        values = np.asarray(yaw_pitch_deg, dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError("Finite two-axis target required")
        yaw, pitch = np.deg2rad(values)
        if not np.isfinite(roll_deg):
            raise ValueError("Finite measured roll required")
        target = np.array([yaw, np.deg2rad(roll_deg), pitch])
        if np.any(target < self.limits[:, 0]) or np.any(target > self.limits[:, 1]):
            raise ValueError("FK target exceeds real mechanical limits")
        state = np.empty(self.size)
        self.api.mj_getState(self.model, self.data, state, self.spec)
        self.api.mj_setState(self.model, self.scratch, state, self.spec)
        self.scratch.qpos[self.addresses] = target
        self.api.mj_forward(self.model, self.scratch)
        pose = np.eye(4)
        pose[:3, 3] = self.scratch.cam_xpos[self.camera_id]
        # MuJoCo camera looks along -Z with +Y up; search uses OpenCV axes.
        pose[:3, :3] = self.scratch.cam_xmat[self.camera_id].reshape(3, 3) @ np.diag(
            [1.0, -1.0, -1.0]
        )
        after = np.empty(self.size)
        self.api.mj_getState(self.model, self.data, after, self.spec)
        if not np.array_equal(state, after):
            raise RuntimeError("Geometry modified live simulation")
        return pose
