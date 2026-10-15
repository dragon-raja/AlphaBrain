import unittest

import numpy as np

from benchmarks.Robocasa365.render_schedule import PolicyRenderSchedule


class Sim:
    def __init__(self):
        self.calls = 0

    def render(self, **kwargs):
        self.calls += 1
        return np.full((2, 2, 3), self.calls, dtype=np.uint8)


class RenderScheduleTests(unittest.TestCase):
    def test_fresh_before_every_replan_and_no_depth_reuse(self):
        sim = Sim()
        schedule = PolicyRenderSchedule(sim)
        first = sim.render(camera_name='left')
        first[:] = 0
        schedule.before_step(0, 25)
        self.assertEqual(sim.render(camera_name='left')[0, 0, 0], 1)
        self.assertEqual(sim.calls, 1)
        schedule.before_step(24, 25)
        self.assertEqual(sim.render(camera_name='left')[0, 0, 0], 2)
        schedule.before_step(25, 25)
        sim.render(camera_name='left', depth=True)
        sim.render(camera_name='left', depth=True)
        self.assertEqual(sim.calls, 4)
        schedule.close()
        self.assertNotIn('render', vars(sim))
        sim.render(camera_name='left')
        self.assertEqual(sim.calls, 5)
