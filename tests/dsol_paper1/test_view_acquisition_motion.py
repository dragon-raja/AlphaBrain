from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "dsol_paper1"
sys.path.insert(0, str(SCRIPT_ROOT))

from AlphaBrain.research.dsol.acquisition.motion import BudgetExceeded, BudgetLedger, LedgerTerminated, MotionLimits, MotionValidationError, Pose, WorldAABB, no_acquisition, plan_camera_motion, plan_matched_hold


def limits(**overrides: object) -> MotionLimits:
    values = {
        "max_linear_speed": 100.0,
        "max_linear_acceleration": 100.0,
        "max_angular_speed": 100.0,
        "max_angular_acceleration": 100.0,
        "world_aabb": WorldAABB((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)),
    }
    values.update(overrides)
    return MotionLimits(**values)


ORIGIN = Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


class CameraMotionTests(unittest.TestCase):
    def test_endpoints_normalized_and_shared_quintic_progress(self) -> None:
        goal = Pose((1.0, 0.5, -1.0), (0.0, 0.0, 0.0, 1.0))
        trajectory = plan_camera_motion(ORIGIN, goal, duration=2.0, dt=0.1, limits=limits())
        self.assertEqual(trajectory.env_steps, 20)
        self.assertEqual(len(trajectory.samples), 21)
        self.assertEqual(trajectory.samples[0].pose.position, ORIGIN.position)
        self.assertEqual(trajectory.samples[-1].pose.position, goal.position)
        self.assertEqual(trajectory.samples[-1].time_s, 2.0)
        for sample in trajectory.samples:
            self.assertAlmostEqual(math.hypot(*sample.pose.quaternion_wxyz), 1.0)
            u = sample.time_s / trajectory.duration
            progress = 10 * u**3 - 15 * u**4 + 6 * u**5
            self.assertAlmostEqual(sample.pose.position[0], progress)
            self.assertAlmostEqual(
                2 * math.atan2(sample.pose.quaternion_wxyz[3], sample.pose.quaternion_wxyz[0]), math.pi * progress
            )
        for sample in (trajectory.samples[0], trajectory.samples[-1]):
            self.assertEqual(sample.linear_velocity_world, (0.0, 0.0, 0.0))
            self.assertEqual(sample.linear_acceleration_world, (0.0, 0.0, 0.0))
            self.assertEqual(sample.angular_velocity_world, (0.0, 0.0, 0.0))
            self.assertEqual(sample.angular_acceleration_world, (0.0, 0.0, 0.0))

    def test_quaternion_sign_and_180_degree_tie_are_deterministic(self) -> None:
        positive = Pose((0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
        negative = Pose((0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0))
        negative_start = Pose(ORIGIN.position, (-1.0, 0.0, 0.0, 0.0))
        first = plan_camera_motion(ORIGIN, positive, duration=2.0, dt=0.1, limits=limits())
        second = plan_camera_motion(negative_start, negative, duration=2.0, dt=0.1, limits=limits())
        self.assertEqual(first.samples, second.samples)
        self.assertAlmostEqual(first.rotation_angle, math.pi)
        self.assertAlmostEqual(first.samples[10].pose.quaternion_wxyz[0], math.sqrt(0.5))

    def test_shortest_arc_crossing_180_has_no_component_sign_jump(self) -> None:
        def rotated(degrees: float) -> Pose:
            half = math.radians(degrees) / 2.0
            return Pose(ORIGIN.position, (math.cos(half), 0.0, 0.0, math.sin(half)))

        trajectory = plan_camera_motion(rotated(170), rotated(190), duration=1.0, dt=0.05, limits=limits())
        self.assertAlmostEqual(trajectory.rotation_angle, math.radians(20))
        for before, after in zip(trajectory.samples, trajectory.samples[1:]):
            self.assertGreater(sum(a * b for a, b in zip(before.pose.quaternion_wxyz, after.pose.quaternion_wxyz)), 0.99)

    def test_equivalent_quaternions_have_no_rotation(self) -> None:
        goal = Pose(ORIGIN.position, (-1.0, 0.0, 0.0, 0.0))
        trajectory = plan_camera_motion(ORIGIN, goal, duration=1.0, dt=0.1, limits=limits())
        self.assertEqual(trajectory.rotation_angle, 0.0)
        self.assertEqual(trajectory.peaks.angular_speed, 0.0)

    def test_continuous_speed_and_acceleration_limits_even_with_only_two_samples(self) -> None:
        goal = Pose((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        for key, bound in (("max_linear_speed", 1.87), ("max_linear_acceleration", 5.77)):
            with self.subTest(key=key), self.assertRaisesRegex(MotionValidationError, "continuous"):
                plan_camera_motion(ORIGIN, goal, duration=1.0, dt=1.0, limits=limits(**{key: bound}))
        trajectory = plan_camera_motion(ORIGIN, goal, duration=1.0, dt=1.0, limits=limits())
        self.assertAlmostEqual(trajectory.peaks.linear_speed, 1.875)
        self.assertAlmostEqual(trajectory.peaks.linear_acceleration, 10 / math.sqrt(3))

    def test_angular_speed_and_acceleration_limits(self) -> None:
        goal = Pose(ORIGIN.position, (0.0, 0.0, 0.0, 1.0))
        for key in ("max_angular_speed", "max_angular_acceleration"):
            with self.subTest(key=key), self.assertRaisesRegex(MotionValidationError, "angular"):
                plan_camera_motion(ORIGIN, goal, duration=1.0, dt=1.0, limits=limits(**{key: 1.0}))

    def test_bad_pose_and_non_unit_quaternion_rejected(self) -> None:
        for position, quaternion in (
            ((math.nan, 0, 0), (1, 0, 0, 0)),
            ((0, 0, 0), (math.inf, 0, 0, 0)),
            ((0, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 0), (2, 0, 0, 0)),
            ((0, 0), (1, 0, 0, 0)),
            ((0, 0, 0), (1, 0, 0)),
        ):
            with self.subTest(position=position, quaternion=quaternion), self.assertRaises(MotionValidationError):
                Pose(position, quaternion)

    def test_input_sequences_are_copied(self) -> None:
        position = [0.0, 0.0, 0.0]
        quaternion = [1.0, 0.0, 0.0, 0.0]
        pose = Pose(position, quaternion)
        position[0], quaternion[0] = 99.0, 0.0
        self.assertEqual(pose, ORIGIN)

    def test_schedule_requires_finite_integer_steps(self) -> None:
        for duration, dt in ((1.0, 0.3), (0.0, 0.1), (-1, 0.1), (1, 0), (math.nan, 0.1), (1, math.inf), (1, 1e-9)):
            with self.subTest(duration=duration, dt=dt), self.assertRaises(MotionValidationError):
                plan_camera_motion(ORIGIN, ORIGIN, duration=duration, dt=dt, limits=limits())
        trajectory = plan_camera_motion(ORIGIN, ORIGIN, duration=0.3, dt=0.1, limits=limits())
        self.assertEqual(trajectory.env_steps, 3)

    def test_aabb_bounds_reject_outside_and_accept_boundary_segment(self) -> None:
        outside = Pose((2.0001, 0, 0), (1, 0, 0, 0))
        for start, goal in ((ORIGIN, outside), (outside, ORIGIN)):
            with self.assertRaisesRegex(MotionValidationError, "AABB"):
                plan_camera_motion(start, goal, duration=1, dt=0.1, limits=limits())
        boundary = Pose((2, 2, 2), (1, 0, 0, 0))
        trajectory = plan_camera_motion(ORIGIN, boundary, duration=1, dt=0.1, limits=limits())
        for sample in trajectory.samples:
            limits().world_aabb.validate(sample.pose)

    def test_bad_limits_and_aabb_rejected(self) -> None:
        for key in ("max_linear_speed", "max_linear_acceleration", "max_angular_speed", "max_angular_acceleration"):
            for value in (-1, math.nan, math.inf):
                with self.subTest(key=key, value=value), self.assertRaises(MotionValidationError):
                    limits(**{key: value})
        with self.assertRaises(MotionValidationError):
            WorldAABB((1, 0, 0), (0, 1, 1))
        with self.assertRaises(MotionValidationError):
            WorldAABB((math.nan, 0, 0), (1, 1, 1))

    def test_no_acquisition_is_zero_steps_but_hold_consumes_time(self) -> None:
        stationary_limits = limits(
            max_linear_speed=0, max_linear_acceleration=0, max_angular_speed=0, max_angular_acceleration=0
        )
        none = no_acquisition(ORIGIN, dt=0.1, limits=stationary_limits)
        hold = plan_matched_hold(ORIGIN, duration=2, dt=0.1, limits=stationary_limits)
        self.assertEqual((none.env_steps, none.duration, len(none.samples)), (0, 0.0, 1))
        self.assertEqual((hold.env_steps, hold.duration), (20, 2.0))
        self.assertTrue(all(sample.pose == ORIGIN for sample in hold.samples))
        self.assertEqual(none.peaks, hold.peaks)

    def test_arbitrary_orientation_hold_and_tiny_duration_stay_finite(self) -> None:
        pose = Pose((0.2, 0.3, 0.4), (0.5, 0.5, 0.5, 0.5))
        trajectory = plan_matched_hold(pose, duration=1e-200, dt=5e-201, limits=limits())
        self.assertEqual(trajectory.rotation_angle, 0.0)
        for sample in trajectory.samples:
            self.assertEqual(sample.time, sample.time_s)
            self.assertEqual(sample.pose, pose)
            self.assertEqual(sample.linear_acceleration_world, (0.0, 0.0, 0.0))
            self.assertEqual(sample.angular_acceleration_world, (0.0, 0.0, 0.0))


class BudgetLedgerTests(unittest.TestCase):
    def test_independent_counts_exact_limits_and_termination(self) -> None:
        ledger = BudgetLedger(limit_env_steps=10, limit_queries=2, limit_policy_calls=3)
        ledger.charge(env_steps=10, label="hold")
        ledger.charge(queries=2, policy_calls=3, label="observe_then_infer")
        self.assertEqual((ledger.env_steps, ledger.queries, ledger.policy_calls), (10, 2, 3))
        self.assertEqual(ledger.remaining, {"env_steps": 0, "queries": 0, "policy_calls": 0})
        ledger.terminate("environment_done")
        self.assertEqual(ledger.as_dict()["termination"], "environment_done")
        with self.assertRaises(LedgerTerminated):
            ledger.charge(label="after_done")
        with self.assertRaises(LedgerTerminated):
            ledger.terminate("overwrite")

    def test_each_budget_excess_is_atomic_and_fail_closed(self) -> None:
        for dimension in ("env_steps", "queries", "policy_calls"):
            with self.subTest(dimension=dimension):
                ledger = BudgetLedger(limit_env_steps=2, limit_queries=2, limit_policy_calls=2)
                ledger.charge(env_steps=1, queries=1, policy_calls=1, label="first")
                reservation = {"env_steps": 1, "queries": 1, "policy_calls": 1, dimension: 2}
                with self.assertRaises(BudgetExceeded):
                    ledger.charge(**reservation, label="too_much")
                self.assertEqual((ledger.env_steps, ledger.queries, ledger.policy_calls), (1, 1, 1))
                self.assertEqual(ledger.termination, f"budget_exceeded:{dimension}")
                self.assertEqual(len(ledger.entries), 2)
                with self.assertRaises(LedgerTerminated):
                    ledger.charge(label="retry")

    def test_invalid_counts_are_not_silently_coerced(self) -> None:
        for value in (-1, 1.0, 0.5, math.nan, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    BudgetLedger(limit_env_steps=value, limit_queries=0, limit_policy_calls=0)
                ledger = BudgetLedger(limit_env_steps=5, limit_queries=5, limit_policy_calls=5)
                with self.assertRaises(ValueError):
                    ledger.charge(env_steps=value, label="invalid")
                self.assertEqual(ledger.env_steps, 0)
                self.assertIsNone(ledger.termination)

    def test_returned_snapshots_cannot_change_counts(self) -> None:
        ledger = BudgetLedger(limit_env_steps=2, limit_queries=1, limit_policy_calls=1)
        ledger.charge(env_steps=1, label="step")
        snapshot = ledger.as_dict()
        snapshot["used"]["env_steps"] = 200
        snapshot["limits"]["env_steps"] = 200
        ledger.remaining["env_steps"] = 200
        self.assertEqual(ledger.env_steps, 1)
        self.assertEqual(ledger.remaining["env_steps"], 1)


if __name__ == "__main__":
    unittest.main()
