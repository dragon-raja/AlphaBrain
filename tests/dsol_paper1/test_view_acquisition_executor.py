from __future__ import annotations

import unittest
from unittest.mock import patch
import types
import numpy as np

from AlphaBrain.research.dsol.acquisition.executor import (
    LiberoKinematicCameraBackend, PersistentOSCGoalHold, StepFeedback, execute_acquisition,
)
from AlphaBrain.research.dsol.acquisition.motion import (
    BudgetLedger, MotionLimits, Pose, WorldAABB, no_acquisition, plan_camera_motion, plan_matched_hold,
)


START = Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
GOAL = Pose((0.1, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
LIMITS = MotionLimits(1.0, 2.0, 1.0, 2.0, WorldAABB((-1, -1, -1), (1, 1, 1)))


class FakeBackend:
    def __init__(self, *, fail_at=None, terminate_at=None, bad_clock=False):
        self.pose = START
        self.time = 3.0
        self.steps = 0
        self.observations = 0
        self.fail_at = fail_at
        self.terminate_at = terminate_at
        self.bad_clock = bad_clock

    def camera_pose(self):
        return self.pose

    def sim_time(self):
        return self.time

    def step(self, pose):
        self.steps += 1
        if self.steps == self.fail_at:
            raise RuntimeError("simulated movement failure")
        self.time += 0.0 if self.bad_clock else 0.1
        self.pose = pose
        return StepFeedback(
            self.time, pose, {"position": list(pose.position)},
            terminated=self.steps == self.terminate_at,
        )

    def observe(self):
        self.observations += 1
        return {"current_image": "endpoint-only"}


def make_ledger(steps=10, queries=1):
    return BudgetLedger(limit_env_steps=steps, limit_queries=queries, limit_policy_calls=0)


def hold_fixture():
    class Controller:
        impedance_mode = "fixed"
        goal_pos = ee_pos = np.zeros(3)
        goal_ori = ee_ori_mat = np.eye(3)

        def update(self, force=False):
            pass

        def set_goal(self, action, set_pos=None, set_ori=None):
            self.last_position = set_pos

    class PandaGripper:
        current_action = np.array([0.0])

    class Robot:
        controller = Controller()
        gripper = PandaGripper()
        _ref_joint_gripper_actuator_indexes = [1, 2]

        def grip_action(self, gripper, gripper_action):
            raise AssertionError("normalized command must not execute during hold")

    robot = Robot()
    sim = types.SimpleNamespace(data=types.SimpleNamespace(ctrl=np.array([9.0, 0.01, -0.01])))

    def step(action):
        robot.controller.set_goal(action[:6])
        robot.grip_action(gripper=robot.gripper, gripper_action=action[-1:])
        return "stepped"

    env = types.SimpleNamespace(env=types.SimpleNamespace(robots=[robot], action_dim=7, sim=sim), step=step)
    return env, robot, sim


class PersistentHoldTests(unittest.TestCase):
    def test_entry_is_noop_and_step_pins_existing_actuator_targets(self):
        env, robot, sim = hold_fixture()
        before = sim.data.ctrl.copy()
        with PersistentOSCGoalHold(env) as hold:
            np.testing.assert_array_equal(sim.data.ctrl, before)
            sim.data.ctrl[1:] = 0.4
            self.assertEqual(hold.step(), "stepped")
            np.testing.assert_array_equal(sim.data.ctrl, before)
            np.testing.assert_array_equal(robot.gripper.current_action, [0.0])
        self.assertNotIn("set_goal", vars(robot.controller))
        self.assertNotIn("grip_action", vars(robot))

    def test_exception_restores_existing_instance_overrides(self):
        env, robot, _sim = hold_fixture()
        original_goal, original_grip = robot.controller.set_goal, robot.grip_action
        robot.controller.set_goal, robot.grip_action = original_goal, original_grip
        with self.assertRaisesRegex(RuntimeError, "body failure"):
            with PersistentOSCGoalHold(env):
                raise RuntimeError("body failure")
        self.assertIs(robot.controller.set_goal, original_goal)
        self.assertIs(robot.grip_action, original_grip)

    def test_reject_duplicate_gripper_actuator_mapping(self):
        env, robot, _sim = hold_fixture()
        robot._ref_joint_gripper_actuator_indexes = [1, 1]
        with self.assertRaisesRegex(ValueError, "distinct"):
            PersistentOSCGoalHold(env)


class AcquisitionExecutorTests(unittest.TestCase):
    def test_move_advances_time_and_exposes_only_endpoint(self):
        backend = FakeBackend()
        trajectory = plan_camera_motion(START, GOAL, duration=1.0, dt=0.1, limits=LIMITS)
        result = execute_acquisition(trajectory, backend, make_ledger())
        self.assertEqual(result.status, "acquired")
        self.assertEqual((result.completed_env_steps, result.endpoint_queries), (10, 1))
        self.assertEqual(backend.observations, 1)
        self.assertAlmostEqual(result.events[-1]["sim_time"], 4.0)
        self.assertNotEqual(result.events[0]["calibration_sha256"], result.events[-1]["calibration_sha256"])
        self.assertNotIn("observation", result.audit_record())

    def test_continue_is_exact_noop(self):
        class NoBackendCalls:
            def __getattr__(self, _name):
                raise AssertionError("no-acquisition must not call backend")

        initial = {"same": object()}
        result = execute_acquisition(
            no_acquisition(START, dt=0.1, limits=LIMITS), NoBackendCalls(), make_ledger(0, 0),
            initial_observation=initial,
        )
        self.assertEqual(result.status, "continued_without_acquisition")
        self.assertIs(result.observation, initial)
        self.assertEqual(result.completed_env_steps, 0)

    def test_hold_consumes_same_steps(self):
        result = execute_acquisition(
            plan_matched_hold(START, duration=1.0, dt=0.1, limits=LIMITS),
            FakeBackend(), make_ledger(),
        )
        self.assertEqual(result.status, "acquired")
        self.assertEqual(result.completed_env_steps, 10)
        self.assertTrue(all(e["position"] == [0.0, 0.0, 0.0] for e in result.events))

    def test_continue_cannot_bypass_terminated_ledger(self):
        ledger = make_ledger()
        ledger.terminate("previous_failure")
        result = execute_acquisition(no_acquisition(START, dt=0.1, limits=LIMITS), object(), ledger)
        self.assertEqual(result.status, "failed")
        self.assertIn("previous_failure", result.error)

    def test_real_backend_calibration_is_json_compatible(self):
        import json

        module = types.ModuleType("evaluate_pi05_libero_plus_views")
        module.agentview_camera_calibration = lambda _env: {"extrinsic": np.eye(4)}
        backend = LiberoKinematicCameraBackend.__new__(LiberoKinematicCameraBackend)
        backend.env = object()
        with patch.dict("sys.modules", {module.__name__: module}):
            calibration = backend.calibration()
        self.assertEqual(calibration["extrinsic"], np.eye(4).tolist())
        json.dumps(calibration, allow_nan=False)

    def test_endpoint_adapter_refuses_wrapper_image_noise(self):
        env = types.SimpleNamespace(noise=1)
        with self.assertRaisesRegex(ValueError, "noise=0"):
            LiberoKinematicCameraBackend(env, None)

    def test_insufficient_step_budget_stops_before_extra_mutation(self):
        backend = FakeBackend()
        result = execute_acquisition(
            plan_camera_motion(START, GOAL, duration=1.0, dt=0.1, limits=LIMITS),
            backend, make_ledger(3),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(backend.steps, 3)
        self.assertEqual(backend.observations, 0)

    def test_no_query_budget_does_not_leak_observation(self):
        backend = FakeBackend()
        result = execute_acquisition(
            plan_matched_hold(START, duration=1.0, dt=0.1, limits=LIMITS),
            backend, make_ledger(10, 0),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(backend.observations, 0)

    def test_failed_attempt_is_not_converted_to_hold(self):
        backend = FakeBackend(fail_at=2)
        result = execute_acquisition(
            plan_camera_motion(START, GOAL, duration=1.0, dt=0.1, limits=LIMITS),
            backend, make_ledger(),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.completed_env_steps, 1)
        self.assertIn("simulated movement failure", result.error)

    def test_environment_termination_has_no_extra_query(self):
        backend = FakeBackend(terminate_at=2)
        result = execute_acquisition(
            plan_camera_motion(START, GOAL, duration=1.0, dt=0.1, limits=LIMITS),
            backend, make_ledger(),
        )
        self.assertEqual(result.status, "terminated_during_acquisition")
        self.assertEqual(backend.steps, 2)
        self.assertEqual(backend.observations, 0)

    def test_clock_must_advance(self):
        backend = FakeBackend(bad_clock=True)
        result = execute_acquisition(
            plan_matched_hold(START, duration=1.0, dt=0.1, limits=LIMITS),
            backend, make_ledger(),
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("simulator time", result.error)

    def test_start_mismatch_prevents_mutation(self):
        backend = FakeBackend()
        backend.pose = GOAL
        result = execute_acquisition(
            plan_matched_hold(START, duration=1.0, dt=0.1, limits=LIMITS),
            backend, make_ledger(),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(backend.steps, 0)


if __name__ == "__main__":
    unittest.main()
