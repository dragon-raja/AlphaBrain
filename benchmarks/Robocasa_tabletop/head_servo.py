"""Bounded, injected GR1 head servo; importing never constructs a simulator.

The IO implementation owns stepping, fixed-scene checks and local safe stopping.
Gains and effort bounds require separate physical admission. This module does
not claim that the full mechanical range is a safe search domain.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ServoSettings:
    kp: tuple
    kd: tuple
    effort_limit: tuple
    position_tolerance: float
    velocity_tolerance: float
    settled_samples: int
    max_steps: int
    max_elapsed_seconds: float

    def validate(self):
        for values in (self.kp, self.kd, self.effort_limit):
            if len(values) != 3 or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Three explicit positive finite gains/effort limits required")
        for v in (self.position_tolerance, self.velocity_tolerance, self.max_elapsed_seconds):
            if not math.isfinite(v) or v <= 0:
                raise ValueError("Positive tolerances and elapsed-time bound required")
        if (
            type(self.settled_samples) is not int
            or self.settled_samples < 2
            or type(self.max_steps) is not int
            or self.max_steps < self.settled_samples
        ):
            raise ValueError("Bounded steps and multiple settled samples required")


class HeadServo:
    """Track all three real joints; two-axis requests explicitly hold roll.

    IO methods: read_head_state() -> positions/velocities/observed_at (yaw,roll,
    pitch), non_head_state_identity(), write_head_efforts(named), advance(), and
    stop_head_motion(). The last method is an independently admitted local safety
    path, not another research action and not an inferred emergency-stop policy.
    Any failure permanently holds this instance; recovery requires explicit audit.
    """

    def __init__(self, io, layout, settings, *, roll_hold, guard, clock):
        settings.validate()
        layout.named_targets((0.0, 0.0), roll_hold=roll_hold)
        if layout.actuator_kind != "motor":
            raise ValueError("Servo requires the audited native motor layout")
        self.io, self.layout, self.settings = io, layout, settings
        self.roll_hold, self.guard, self.clock = roll_hold, guard, clock
        self.status = "READY"
        self.stop_failure = None

    def move_head(self, yaw_pitch):
        if self.status != "READY":
            raise RuntimeError("Head servo held; explicit recovery required")
        targets = tuple(self.layout.named_targets(yaw_pitch, roll_hold=self.roll_hold).values())
        cfg = self.settings
        try:
            self.guard()
            start = self.clock()
            if not math.isfinite(start):
                raise ValueError("Finite monotonic clock required")
            scene = self.io.non_head_state_identity()
            previous_time = None
            stable = 0
            for step in range(cfg.max_steps + 1):
                self.guard()
                elapsed = self.clock() - start
                if not math.isfinite(elapsed) or elapsed < 0 or elapsed > cfg.max_elapsed_seconds:
                    raise TimeoutError("Head motion elapsed-time limit reached")
                state = self.io.read_head_state()
                pos, vel = tuple(state["positions"]), tuple(state["velocities"])
                stamp = state["observed_at"]
                if (
                    len(pos) != 3
                    or len(vel) != 3
                    or any(not math.isfinite(v) for v in pos + vel)
                    or not math.isfinite(stamp)
                    or previous_time is not None
                    and stamp <= previous_time
                ):
                    raise ValueError("Fresh finite measured head position and velocity required")
                previous_time = stamp
                if self.io.non_head_state_identity() != scene:
                    raise ValueError("Non-head scene changed during acquisition")
                if any(p < low or p > high for p, (low, high) in zip(pos, self.layout.limits)):
                    raise ValueError("Measured head joint outside mechanical limits")
                error = tuple(t - p for t, p in zip(targets, pos))
                at_target = (
                    max(map(abs, error)) <= cfg.position_tolerance
                    and max(map(abs, vel)) <= cfg.velocity_tolerance
                )
                stable = stable + 1 if at_target else 0
                if stable >= cfg.settled_samples:
                    self.guard()
                    return {
                        "measured_joint_rad": (pos[0], pos[2]),
                        "measured_roll_rad": pos[1],
                        "settled": True,
                        "settled_at": stamp,
                        "steps": step,
                        "elapsed_seconds": elapsed,
                    }
                if step == cfg.max_steps:
                    raise TimeoutError("Head motion step limit reached")
                efforts = tuple(
                    max(-limit, min(limit, kp * e - kd * v))
                    for kp, kd, limit, e, v in zip(cfg.kp, cfg.kd, cfg.effort_limit, error, vel)
                )
                self.guard()
                self.io.write_head_efforts(dict(zip(self.layout.actuator_names, efforts)))
                self.io.advance()
        except BaseException:
            self.status = "HOLD"
            try:
                self.io.stop_head_motion()
            except BaseException as error:
                # Retain evidence of a failed local stop while preserving the
                # original failure; never imply that a HOLD physically stopped IO.
                self.stop_failure = type(error).__name__
            raise
