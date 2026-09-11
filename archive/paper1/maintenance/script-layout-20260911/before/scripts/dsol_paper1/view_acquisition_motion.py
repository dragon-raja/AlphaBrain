"""Pure-Python kinematic primitives for independent-camera acquisition.

Positions are world-frame metres; quaternions are unit ``(w, x, y, z)``.
Translation and shortest-arc SLERP share one quintic time law. Limits bound
the continuous trajectory, not just sampled poses. The world AABB certifies
only the camera *centre*: this is NOT collision, IK, physical-head, cable,
camera-body clearance, or dynamic-obstacle certification. An executor must
provide those checks and advance the environment for every charged step.

Budget charges are reservations made BEFORE issuing operations. Failed
operations stay charged; their executor must explicitly terminate the ledger.
No simulation, rendering, policy, or hardware calls occur in this module.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Sequence


UNIT_QUATERNION_TOLERANCE = 1e-6
MAX_TRAJECTORY_STEPS = 100_000
_PEAK_FIRST_DERIVATIVE = 15.0 / 8.0
_PEAK_SECOND_DERIVATIVE = 10.0 / math.sqrt(3.0)


class MotionValidationError(ValueError):
    """A pose, schedule, continuous motion limit, or AABB check failed."""


class BudgetExceeded(RuntimeError):
    """A reservation exceeds a limit; the ledger is now terminated."""


class LedgerTerminated(RuntimeError):
    """No further reservations or termination changes are allowed."""


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise MotionValidationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MotionValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise MotionValidationError(f"{name} must be finite")
    return result


def _finite_vector(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise MotionValidationError(f"{name} must have {length} finite components")
    try:
        result = tuple(_finite_scalar(item, name) for item in value)
    except TypeError as exc:
        raise MotionValidationError(f"{name} must have {length} finite components") from exc
    if len(result) != length:
        raise MotionValidationError(f"{name} must have {length} finite components")
    return result


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        position = _finite_vector(self.position, 3, "position")
        quaternion = _finite_vector(self.quaternion_wxyz, 4, "quaternion_wxyz")
        norm = math.hypot(*quaternion)
        if abs(norm - 1.0) > UNIT_QUATERNION_TOLERANCE:
            raise MotionValidationError("quaternion_wxyz must be unit norm (tolerance 1e-6)")
        object.__setattr__(self, "position", position)
        # Remove round-off only, rather than silently accepting non-unit input.
        object.__setattr__(self, "quaternion_wxyz", tuple(value / norm for value in quaternion))


@dataclass(frozen=True)
class WorldAABB:
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def __post_init__(self) -> None:
        lower = _finite_vector(self.lower, 3, "AABB lower")
        upper = _finite_vector(self.upper, 3, "AABB upper")
        if any(low > high for low, high in zip(lower, upper)):
            raise MotionValidationError("AABB lower bounds must not exceed upper bounds")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def validate(self, pose: Pose) -> None:
        if any(not low <= value <= high for value, low, high in zip(pose.position, self.lower, self.upper)):
            raise MotionValidationError("camera centre is outside world AABB")


@dataclass(frozen=True)
class MotionLimits:
    max_linear_speed: float
    max_linear_acceleration: float
    max_angular_speed: float
    max_angular_acceleration: float
    world_aabb: WorldAABB

    def __post_init__(self) -> None:
        for name in (
            "max_linear_speed", "max_linear_acceleration", "max_angular_speed", "max_angular_acceleration"
        ):
            value = _finite_scalar(getattr(self, name), name)
            if value < 0.0:
                raise MotionValidationError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if not isinstance(self.world_aabb, WorldAABB):
            raise MotionValidationError("world_aabb must be an explicit WorldAABB")


@dataclass(frozen=True)
class MotionSample:
    time_s: float
    pose: Pose
    linear_velocity_world: tuple[float, float, float]
    linear_acceleration_world: tuple[float, float, float]
    angular_velocity_world: tuple[float, float, float]
    angular_acceleration_world: tuple[float, float, float]

    @property
    def time(self) -> float:
        """Elapsed seconds; alias for executors using the shorter field name."""
        return self.time_s


@dataclass(frozen=True)
class MotionPeaks:
    linear_speed: float
    linear_acceleration: float
    angular_speed: float
    angular_acceleration: float


@dataclass(frozen=True)
class CameraTrajectory:
    kind: str
    dt: float
    duration: float
    samples: tuple[MotionSample, ...]
    translation_distance: float
    rotation_angle: float
    peaks: MotionPeaks

    @property
    def env_steps(self) -> int:
        """Initial pose is not an action; execute samples[1:] only."""
        return len(self.samples) - 1


def _schedule(duration: float, dt: float) -> tuple[float, float, int]:
    duration = _finite_scalar(duration, "duration")
    dt = _finite_scalar(dt, "dt")
    if duration <= 0.0 or dt <= 0.0:
        raise MotionValidationError("duration and dt must be positive; use no_acquisition for zero steps")
    ratio = duration / dt
    if not math.isfinite(ratio) or ratio > MAX_TRAJECTORY_STEPS:
        raise MotionValidationError(f"trajectory exceeds {MAX_TRAJECTORY_STEPS} steps")
    steps = round(ratio)
    if steps < 1 or not math.isclose(ratio, steps, rel_tol=0.0, abs_tol=1e-9):
        raise MotionValidationError("duration must be an integer multiple of dt")
    return steps * dt, dt, steps


def _canonical_quaternion(quaternion: tuple[float, ...]) -> tuple[float, ...]:
    # Resolves the two equally short arcs at exactly 180 degrees consistently
    # for q and -q. Do not canonicalize every sample: that can cause sign jumps.
    for value in quaternion:
        if value != 0.0:
            return tuple(-item for item in quaternion) if value < 0.0 else quaternion
    raise MotionValidationError("zero quaternion")


def _quaternion_product(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _plan(start: Pose, goal: Pose, *, duration: float, dt: float, limits: MotionLimits, kind: str) -> CameraTrajectory:
    if not isinstance(start, Pose) or not isinstance(goal, Pose) or not isinstance(limits, MotionLimits):
        raise MotionValidationError("start/goal must be Pose and limits must be MotionLimits")
    duration, dt, steps = _schedule(duration, dt)
    limits.world_aabb.validate(start)
    limits.world_aabb.validate(goal)
    # An AABB is convex and quintic progress is monotone on [0, 1], so these
    # endpoint checks cover the entire continuous camera-centre line segment.
    displacement = tuple(end - begin for begin, end in zip(start.position, goal.position))
    distance = math.hypot(*displacement)
    start_quaternion = _canonical_quaternion(start.quaternion_wxyz)
    goal_quaternion = _canonical_quaternion(goal.quaternion_wxyz)
    dot = sum(a * b for a, b in zip(start_quaternion, goal_quaternion))
    if dot < 0.0:
        goal_quaternion = tuple(-value for value in goal_quaternion)
    relative = (
        (1.0, 0.0, 0.0, 0.0)
        if goal_quaternion == start_quaternion
        else _quaternion_product(goal_quaternion, (start_quaternion[0], *(-v for v in start_quaternion[1:])))
    )
    vector_norm = math.hypot(*relative[1:])
    angle = 2.0 * math.atan2(vector_norm, max(0.0, relative[0]))
    axis = tuple(value / vector_norm for value in relative[1:]) if vector_norm else (0.0, 0.0, 0.0)
    peaks = MotionPeaks(
        _PEAK_FIRST_DERIVATIVE * (distance / duration),
        _PEAK_SECOND_DERIVATIVE * (distance / duration) / duration,
        _PEAK_FIRST_DERIVATIVE * (angle / duration),
        _PEAK_SECOND_DERIVATIVE * (angle / duration) / duration,
    )
    for name in ("linear_speed", "linear_acceleration", "angular_speed", "angular_acceleration"):
        value = getattr(peaks, name)
        if not math.isfinite(value) or value > getattr(limits, f"max_{name}"):
            raise MotionValidationError(f"continuous {name} peak {value} exceeds limit {getattr(limits, f'max_{name}')}")
    samples = []
    for index in range(steps + 1):
        u = index / steps
        progress = min(1.0, max(0.0, u * u * u * (10.0 + u * (-15.0 + 6.0 * u))))
        first = 30.0 * u * u * (1.0 - u) ** 2
        second = 60.0 * u * (1.0 - u) * (1.0 - 2.0 * u)
        half_angle = progress * angle / 2.0
        delta = (math.cos(half_angle), *(value * math.sin(half_angle) for value in axis))
        quaternion = _quaternion_product(delta, start_quaternion)
        position = tuple((1.0 - progress) * a + progress * b for a, b in zip(start.position, goal.position))
        if index == 0:
            position, quaternion = start.position, start_quaternion
        elif index == steps:
            position, quaternion = goal.position, goal_quaternion
        pose = Pose(position, quaternion)
        samples.append(MotionSample(
            time_s=duration if index == steps else index * dt,
            pose=pose,
            linear_velocity_world=tuple((value / duration) * first for value in displacement),
            linear_acceleration_world=tuple((value / duration) / duration * second for value in displacement),
            angular_velocity_world=tuple(value * (angle / duration) * first for value in axis),
            angular_acceleration_world=tuple(value * ((angle / duration) / duration) * second for value in axis),
        ))
    return CameraTrajectory(kind, dt, duration, tuple(samples), distance, angle, peaks)


def plan_camera_motion(start: Pose, goal: Pose, *, duration: float, dt: float, limits: MotionLimits) -> CameraTrajectory:
    """Plan a centre-line/SLERP move, with quaternion endpoint equality up to sign.

    Linear units are m, m/s, m/s²; angular units are rad, rad/s, rad/s².
    Duration must be a positive integer number of dt steps (at most 100,000).
    """
    return _plan(start, goal, duration=duration, dt=dt, limits=limits, kind="acquisition")


def plan_matched_hold(pose: Pose, *, duration: float, dt: float, limits: MotionLimits) -> CameraTrajectory:
    """A stationary control consuming exactly the requested environment steps.

    This does not imply a query or policy call: charge those explicitly when
    executed. Its camera-motion energy is NOT matched to a moving trajectory.
    """
    return _plan(pose, pose, duration=duration, dt=dt, limits=limits, kind="matched_hold")


def no_acquisition(pose: Pose, *, dt: float, limits: MotionLimits) -> CameraTrajectory:
    """Return the initial pose only: zero duration, motion, and environment steps."""
    if not isinstance(pose, Pose) or not isinstance(limits, MotionLimits):
        raise MotionValidationError("pose must be Pose and limits must be MotionLimits")
    dt = _finite_scalar(dt, "dt")
    if dt <= 0.0:
        raise MotionValidationError("dt must be positive")
    limits.world_aabb.validate(pose)
    zero = (0.0, 0.0, 0.0)
    sample = MotionSample(0.0, pose, zero, zero, zero, zero)
    return CameraTrajectory("no_acquisition", dt, 0.0, (sample,), 0.0, 0.0, MotionPeaks(0.0, 0.0, 0.0, 0.0))


@dataclass(frozen=True)
class BudgetEntry:
    label: str
    env_steps: int
    queries: int
    policy_calls: int
    termination: str | None = None


class BudgetLedger:
    """Single-executor, fail-closed reservation ledger; not thread-safe.

    An over-limit reservation consumes no counts but records termination and
    raises BudgetExceeded. Nothing may be charged after any termination.
    Limits and public snapshots cannot be mutated through the public API.
    A query is one delivered observation bundle, not an individual RGB image or
    an internal simulator render. A1 bundles contain external and wrist RGB.
    This ledger does not measure internal rendering compute or sensor energy.
    """

    def __init__(self, *, limit_env_steps: int, limit_queries: int, limit_policy_calls: int) -> None:
        self._limits = tuple(_nonnegative_integer(value, name) for value, name in (
            (limit_env_steps, "limit_env_steps"), (limit_queries, "limit_queries"),
            (limit_policy_calls, "limit_policy_calls"),
        ))
        self._counts = (0, 0, 0)
        self._entries: list[BudgetEntry] = []
        self._termination: str | None = None

    @property
    def env_steps(self) -> int:
        return self._counts[0]

    @property
    def queries(self) -> int:
        return self._counts[1]

    @property
    def policy_calls(self) -> int:
        return self._counts[2]

    @property
    def termination(self) -> str | None:
        return self._termination

    @property
    def entries(self) -> tuple[BudgetEntry, ...]:
        return tuple(self._entries)

    @property
    def remaining(self) -> dict[str, int]:
        return dict(zip(("env_steps", "queries", "policy_calls"), (a - b for a, b in zip(self._limits, self._counts))))

    def _require_active(self) -> None:
        if self._termination is not None:
            raise LedgerTerminated(f"ledger terminated: {self._termination}")

    def charge(self, *, env_steps: int = 0, queries: int = 0, policy_calls: int = 0, label: str = "operation") -> None:
        self._require_active()
        if not isinstance(label, str) or not label.strip():
            raise ValueError("charge label must be a nonempty string")
        names = ("env_steps", "queries", "policy_calls")
        amounts = tuple(
            _nonnegative_integer(value, name) for value, name in zip((env_steps, queries, policy_calls), names)
        )
        updated = tuple(a + b for a, b in zip(self._counts, amounts))
        exceeded = [name for name, value, limit in zip(names, updated, self._limits) if value > limit]
        if exceeded:
            reason = "budget_exceeded:" + ",".join(exceeded)
            self.terminate(reason)
            raise BudgetExceeded(f"{reason}; rejected reservation {label!r}: {dict(zip(names, amounts))}")
        self._counts = updated
        self._entries.append(BudgetEntry(label, *amounts))

    def terminate(self, reason: str) -> None:
        self._require_active()
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("termination reason must be a nonempty string")
        self._termination = reason
        self._entries.append(BudgetEntry("termination", 0, 0, 0, reason))

    def as_dict(self) -> dict[str, object]:
        names = ("env_steps", "queries", "policy_calls")
        return {
            "query_unit": "observation_bundle",
            "limits": dict(zip(names, self._limits)),
            "used": dict(zip(names, self._counts)),
            "remaining": self.remaining,
            "termination": self.termination,
            "entries": [
                {"label": entry.label, "env_steps": entry.env_steps, "queries": entry.queries,
                 "policy_calls": entry.policy_calls, "termination": entry.termination}
                for entry in self._entries
            ],
        }
