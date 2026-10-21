"""Explicit continuous pan/tilt adapter for a fixed MuJoCo camera.

No environment or GPU is created here. Setpoints change camera model fields
only; forward kinematics is not a physical head-motion simulation.
"""
from __future__ import annotations

import hashlib
import numpy as np

from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount, rigid_pose
from scripts.vla_shared.libero_camera_pose import rotation_matrix_to_wxyz


def scene_identity(sim):
    h = hashlib.sha256()
    h.update(np.asarray(sim.get_state().flatten(), dtype=np.float64).tobytes())
    # Include rendered geometry/mocap as well as the integration state.
    for name in ('mocap_pos', 'mocap_quat', 'body_xpos', 'body_xmat', 'geom_xpos', 'geom_xmat'):
        h.update(np.asarray(getattr(sim.data, name), dtype=np.float64).tobytes())
    return h.hexdigest()


class ContinuousCamera:
    def __init__(self, sim, camera_id, reference_pose):
        self.sim, self.camera_id = sim, int(camera_id)
        if int(sim.model.cam_mode[self.camera_id]) != 0:
            raise ValueError('Continuous setpoints require fixed camera mode')
        self.mount = PanTiltMount(reference_pose)
        self.current = np.zeros(2)
        self.original_position = sim.model.cam_pos[self.camera_id].copy()
        self.original_quaternion = sim.model.cam_quat[self.camera_id].copy()
        self.physical = scene_identity(sim)

    def angles_for_pose(self, value):
        pose = rigid_pose(value)
        r = self.mount.reference[:3, :3].T @ pose[:3, :3]
        angles = self.mount.angles(np.rad2deg([np.arctan2(-r[2, 0], r[0, 0]),
                                             np.arctan2(-r[1, 2], r[1, 1])]))
        if not np.allclose(self.mount.pose(angles), pose, atol=1e-9, rtol=0):
            raise ValueError('Pose is not on the fixed-centre yaw/pitch mount')
        return angles

    def actual_pose(self):
        pose = np.eye(4)
        pose[:3, :3] = self.sim.data.cam_xmat[self.camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
        pose[:3, 3] = self.sim.data.cam_xpos[self.camera_id]
        return pose

    def receipt(self):
        if scene_identity(self.sim) != self.physical:
            raise ValueError('Continuous camera changed physical/rendered scene state')
        pose = self.actual_pose()
        if not np.allclose(pose, self.mount.pose(self.current), atol=1e-9, rtol=0):
            raise ValueError('Applied camera pose differs from requested mount pose')
        return dict(camera_to_world=pose, physical_state_id=self.physical, settled=True,
                    settling_scope='kinematic forward only; not physical head settling')

    def move(self, value):
        angles = self.angles_for_pose(value)
        if np.max(np.abs(angles - self.current)) > self.mount.config.step_limit_deg + 1e-9:
            raise ValueError('Setpoint exceeds per-axis movement bound')
        if scene_identity(self.sim) != self.physical:
            raise ValueError('Physical state changed before movement')
        cid = self.camera_id
        if np.max(np.abs(angles)) < 1e-12:
            # Preserve historical native zero-position bytes.
            self.sim.model.cam_pos[cid] = self.original_position
            self.sim.model.cam_quat[cid] = self.original_quaternion
            angles = np.zeros(2)
        else:
            pose = self.mount.pose(angles)
            parent = int(self.sim.model.cam_bodyid[cid])
            r_parent = self.sim.data.body_xmat[parent].reshape(3, 3)
            p_parent = self.sim.data.body_xpos[parent]
            local_rotation = r_parent.T @ pose[:3, :3] @ np.diag([1., -1., -1.])
            self.sim.model.cam_pos[cid] = r_parent.T @ (pose[:3, 3] - p_parent)
            self.sim.model.cam_quat[cid] = rotation_matrix_to_wxyz(local_rotation)
        self.sim.forward()
        self.current = angles.copy()
        return self.receipt()
