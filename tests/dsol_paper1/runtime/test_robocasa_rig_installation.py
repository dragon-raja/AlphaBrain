from types import SimpleNamespace
import unittest

import numpy as np

from benchmarks.Robocasa365.camera_rig import ReferenceRig


class RigInstallationTests(unittest.TestCase):
    def rig(self):
        rig = ReferenceRig.__new__(ReferenceRig)
        rig.ids = [0, 1, 2]
        rig.local_positions = np.zeros((4, 3))
        rig.local_quaternions = np.tile([1., 0., 0., 0.], (4, 1))
        rig.sim = SimpleNamespace(model=SimpleNamespace(cam_pos=rig.local_positions.copy(),
            cam_quat=rig.local_quaternions.copy()), forward=lambda: None)
        return rig

    def test_mount_is_local_and_copied(self):
        rig = self.rig()
        p, q = rig.mounting()
        p[0] = [1, 2, 3]
        rig.install((p, q))
        np.testing.assert_array_equal(rig.sim.model.cam_pos[0], [1, 2, 3])
        p[0] = [4, 5, 6]
        np.testing.assert_array_equal(rig.sim.model.cam_pos[0], [1, 2, 3])
        rig.restore()
        np.testing.assert_array_equal(rig.sim.model.cam_pos, rig.local_positions)

    def test_wrist_and_other_cameras_cannot_change(self):
        for camera in (2, 3):
            rig = self.rig()
            p, q = rig.mounting()
            p[camera, 0] = .001
            with self.assertRaises(ValueError):
                rig.install((p, q))
