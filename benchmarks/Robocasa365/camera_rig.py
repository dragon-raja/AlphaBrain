"""MuJoCo adapter for linked exterior-camera calibration; no environment creation."""

import numpy as np
from scipy.spatial.transform import Rotation
from AlphaBrain.research.dsol.acquisition.camera_rig import orbit_rig


CAMERA_NAMES = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
CALIBRATION_POSES = [('canonical', 0, 0, 1.), ('yaw_m30', -30, 0, 1.), ('yaw_p30', 30, 0, 1.),
    ('pitch_m15', 0, -15, 1.), ('pitch_p15', 0, 15, 1.), ('near_080', 0, 0, .8), ('far_120', 0, 0, 1.2),
    *[(f'corner_{a}_{e}_{r}', a, e, r) for a in (-30, 30) for e in (-15, 15) for r in (.8, 1.2)]]


class ReferenceRig:
    """Capture at a fixed physical state; all applications are absolute offsets."""
    def __init__(self, sim):
        self.sim = sim
        self.ids = [sim.model.camera_name2id(n) for n in CAMERA_NAMES]
        self.local_positions = np.asarray(sim.model.cam_pos).copy()
        self.local_quaternions = np.asarray(sim.model.cam_quat).copy()
        sim.forward()
        self.positions = np.asarray(sim.data.cam_xpos)[self.ids[:2]].copy()
        self.rotations = np.asarray(sim.data.cam_xmat)[self.ids[:2]].reshape(2, 3, 3).copy()

    def apply(self, azimuth, elevation, radius):
        sim = self.sim
        self.restore()
        target = orbit_rig(self.positions, self.rotations, azimuth_deg=azimuth,
                           elevation_deg=elevation, radius_scale=radius)
        for i, camera_id in enumerate(self.ids[:2]):
            body = sim.model.cam_bodyid[camera_id]
            rotation = np.asarray(sim.data.body_xmat[body]).reshape(3, 3)
            position = np.asarray(sim.data.body_xpos[body])
            sim.model.cam_pos[camera_id] = rotation.T @ (target['positions'][i] - position)
            xyzw = Rotation.from_matrix(rotation.T @ target['rotations'][i]).as_quat()
            sim.model.cam_quat[camera_id] = xyzw[[3, 0, 1, 2]]
        sim.forward()
        return target

    def restore(self):
        self.sim.model.cam_pos[:] = self.local_positions
        self.sim.model.cam_quat[:] = self.local_quaternions
        self.sim.forward()

    def mounting(self):
        """Copy the current local installation, independent of current body pose."""
        return (np.asarray(self.sim.model.cam_pos).copy(),
                np.asarray(self.sim.model.cam_quat).copy())

    def install(self, mounting):
        """Install a saved local rig at any recorded trajectory frame.

        Unlike apply(), this does not re-anchor the orbit in world coordinates.
        Exterior views therefore follow the original support-body motion.
        """
        positions, quaternions = map(np.asarray, mounting)
        if positions.shape != self.local_positions.shape or quaternions.shape != self.local_quaternions.shape:
            raise ValueError('Incompatible camera installation shape')
        unchanged = [i for i in range(len(positions)) if i not in self.ids[:2]]
        if (not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
                or not np.array_equal(positions[unchanged], self.local_positions[unchanged])
                or not np.array_equal(quaternions[unchanged], self.local_quaternions[unchanged])):
            raise ValueError('Only the exterior stereo rig installation may change')
        self.sim.model.cam_pos[:] = positions
        self.sim.model.cam_quat[:] = quaternions
        self.sim.forward()

    def render(self, resolution=256):
        return [np.ascontiguousarray(self.sim.render(height=resolution, width=resolution, camera_name=n)[::-1])
                for n in CAMERA_NAMES]
