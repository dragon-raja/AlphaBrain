"""Task-side head/capture composition; no implicit renderer or physics protocol.

The caller injects the admitted HeadServo, native IO, actual FK and synchronous
frame capture. CPU fixtures exercise this same port. In particular this class
does not freeze bodies or substitute commanded angles for measured positions.
"""

from dataclasses import dataclass
import numpy as np
from .search_runtime import MeasuredHeadReceipt
from .operation_io import model_input


@dataclass(frozen=True)
class EgoFrame:
    pixels: np.ndarray
    epoch: int
    sequence: int
    observed_at: float
    camera_to_world: np.ndarray


class HeadTaskBackend:
    def __init__(
        self, env, io, servo, geometry, capture_frame, *, guard, input_builder=model_input
    ):
        self.env, self.io, self.servo, self.geometry = env, io, servo, geometry
        self.capture_frame, self.guard, self.input_builder = capture_frame, guard, input_builder
        self.epoch = 0
        self.sequence = 0
        self.status = "READY"
        self.arrival = None

    def _live(self):
        if self.status != "READY":
            raise RuntimeError("Task backend held; explicit recovery required")
        self.guard()

    def _receipt(self):
        if self.arrival is None:
            raise ValueError("Explicit initial servo settlement required")
        state = self.io.read_head_state()
        q, v = np.asarray(state["positions"]), np.asarray(state["velocities"])
        cfg = self.servo.settings
        target = np.array([*self.arrival["measured_joint_rad"]])
        settled = (
            q.shape == v.shape == (3,)
            and np.isfinite(q).all()
            and np.isfinite(v).all()
            and np.isfinite(state["observed_at"])
            and np.max(np.abs(q[[0, 2]] - target)) <= cfg.position_tolerance
            and abs(q[1] - self.servo.roll_hold) <= cfg.position_tolerance
            and np.max(np.abs(v)) <= cfg.velocity_tolerance
            and state["observed_at"] >= self.arrival["settled_at"]
        )
        if not settled:
            raise ValueError("Head drifted since measured settlement")
        angles = np.rad2deg(q[[0, 2]])
        roll = float(np.rad2deg(q[1]))
        return (
            MeasuredHeadReceipt(
                self.geometry.measured_pose(angles, roll),
                self.io.non_head_state_identity(),
                True,
                tuple(angles),
                roll,
            ),
            state["observed_at"],
        )

    def pose_receipt(self):
        try:
            self._live()
            result, _ = self._receipt()
            self._live()
            return result
        except BaseException:
            self.status = "HOLD"
            raise

    def move_head_degrees(self, target):
        try:
            self._live()
            target = np.asarray(target, dtype=float)
            if target.shape != (2,) or not np.isfinite(target).all():
                raise ValueError("Finite head target required")
            scene = self.io.non_head_state_identity()
            self.arrival = self.servo.move_head(np.deg2rad(target))
            if self.io.non_head_state_identity() != scene:
                raise ValueError("Fixed scene changed during head motion")
            return self.pose_receipt()
        except BaseException:
            self.status = "HOLD"
            raise

    def reset_observation_cache(self, epoch):
        try:
            self._live()
            if type(epoch) is not int or epoch <= self.epoch:
                raise ValueError("Cache epoch must advance")
            self.epoch = epoch
        except BaseException:
            self.status = "HOLD"
            raise

    def capture_input(self, epoch):
        try:
            self._live()
            if epoch != self.epoch:
                raise ValueError("Wrong observation epoch")
            before, stamp = self._receipt()
            self.sequence += 1
            frame = self.capture_frame(epoch, self.sequence)
            after, after_stamp = self._receipt()
            if (
                not isinstance(frame, EgoFrame)
                or frame.epoch != epoch
                or frame.sequence != self.sequence
                or frame.observed_at != stamp
                or after_stamp != stamp
                or before.physical_state_id != after.physical_state_id
                or not np.array_equal(before.camera_to_world, after.camera_to_world)
                or not np.array_equal(frame.camera_to_world, before.camera_to_world)
            ):
                raise ValueError("Stale frame or changed scene/head during capture")
            example = self.input_builder(self.env, np.array(frame.pixels, copy=True))
            final, final_stamp = self._receipt()
            if (
                final_stamp != stamp
                or final.physical_state_id != before.physical_state_id
                or not np.array_equal(final.camera_to_world, before.camera_to_world)
            ):
                raise ValueError("Input preprocessing changed scene/head")
            self._live()
            return example, self.epoch
        except BaseException:
            self.status = "HOLD"
            raise
