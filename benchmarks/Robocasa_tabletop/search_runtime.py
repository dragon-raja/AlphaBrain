"""GR1 composition ports for the existing native search/PPO core.

Backends supply admitted motion and capture; this module never constructs a
simulator, renderer, model, controller service, or an alternative PPO learner.
Forward kinematics must come from the actual articulated head. A CPU fake
backend tests plumbing only and cannot authorize a real backend.
"""

from dataclasses import dataclass
import hashlib
import numpy as np

from AlphaBrain.research.dsol.acquisition.local_view_search import (
    PanTiltMount,
    PoseReceipt,
    ViewMeasurement,
    rigid_pose,
)
from AlphaBrain.research.dsol.acquisition.native_search import NativeContinuousSearch, native_gym
from AlphaBrain.research.dsol.data.native_features import (
    NativeFeatures,
    NativeQueryResult,
    layout_spec,
)
from AlphaBrain.research.dsol.data.native_observation import NativeStream


STREAMS = (NativeStream("ego_visual", 1152, "per_view", ("ego",), ("movable",)),)


@dataclass(frozen=True)
class MeasuredHeadReceipt(PoseReceipt):
    joint_angles_deg: tuple = ()
    roll_deg: float | None = None


class MeasuredHeadSearch(NativeContinuousSearch):
    """Measured arrival adapter; shared network, queries and reward unchanged.

    This contract is explicit and GR1-specific. A return verifies the measured
    joint pose, unchanged scene and a fresh, charged native query. It records the
    new score; it does not claim bit-identical images at approximately equal poses
    or use the returned score to retrospectively rewrite search rewards.
    """

    def __init__(self, *args, settling_tolerance_deg, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            not np.isfinite(settling_tolerance_deg)
            or settling_tolerance_deg <= 0
            or settling_tolerance_deg >= self.config.step_limit_deg / 4
        ):
            raise ValueError("Explicit bounded measured-arrival tolerance required")
        self.settling_tolerance_deg = float(settling_tolerance_deg)
        self.return_measurement = None

    def _accept_measured(self, receipt):
        if (
            not isinstance(receipt, MeasuredHeadReceipt)
            or receipt.roll_deg is None
            or not np.isfinite(receipt.roll_deg)
            or abs(receipt.roll_deg) > self.settling_tolerance_deg
        ):
            raise ValueError("Measured held-roll position outside declared tolerance")
        self.mount.record_measured(receipt)

    def _move(self, target):
        start = self.current.copy()
        target = self.mount.angles(target)
        distance = float(np.max(np.abs(target - start)))
        if distance == 0:
            return
        # Leave margin for bounded servo error at the previous measured arrival.
        count = max(
            1,
            int(np.ceil(distance / (self.config.step_limit_deg - 2 * self.settling_tolerance_deg))),
        )
        for i in range(1, count + 1):
            command = start + (target - start) * (i / count)
            if np.max(np.abs(command - self.current)) > self.config.step_limit_deg:
                raise ValueError("Measured head would require an oversized motion")
            receipt = self.source.move(self.mount.pose(command))
            if not isinstance(receipt, MeasuredHeadReceipt):
                raise ValueError("Actual head joint measurement required")
            measured = self.mount.angles(receipt.joint_angles_deg)
            if np.max(np.abs(measured - command)) > self.settling_tolerance_deg:
                raise ValueError("Head did not reach explicit servo tolerance")
            self._accept_measured(receipt)
            self._check_receipt(receipt, self.mount.pose(measured))
            self.current = measured.copy()
            self.events.append(
                dict(kind="move", angles=measured.tolist(), commanded_angles=command.tolist())
            )

    def _verify_return(self, expected, actual):
        a, b = expected.features, actual.features
        if (a.layout_sha256, a.history_sha256, a.model_sha256, a.producer_sha256) != (
            b.layout_sha256,
            b.history_sha256,
            b.model_sha256,
            b.producer_sha256,
        ):
            raise ValueError("Return native provenance changed")
        self.return_measurement = dict(
            actual_angles_deg=self.current.tolist(),
            actual_accel_mean=actual.accel_mean,
            actual_accel_std=actual.accel_std,
            historical_accel_mean=expected.accel_mean,
            input_sha256=b.input_sha256,
            historical_input_sha256=a.input_sha256,
            rule="MEASURED_POSE_FRESH_RESCORE_V1",
        )
        self.events.append(dict(kind="measured_return_verification", **self.return_measurement))


class ArticulatedHeadMount(PanTiltMount):
    """Reuse bounded joint proposals, replacing fixed-centre geometry with FK.

    The explicit search box is a symmetric subset of independently verified joint
    limits. This does not certify collision safety or choose a scientific domain.
    """

    def __init__(
        self, config, forward_kinematics, *, joint_limits_deg, measured_forward_kinematics=None
    ):
        bounds = np.asarray(joint_limits_deg, dtype=float)
        requested = np.array([config.yaw_limit_deg, config.pitch_limit_deg])
        if (
            bounds.shape != (2, 2)
            or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] > -requested)
            or np.any(bounds[:, 1] < requested)
        ):
            raise ValueError("Search domain exceeds actual joint limits")
        self.config = config
        self.forward_kinematics = forward_kinematics
        self.measured_forward_kinematics = measured_forward_kinematics
        self.targets = {}
        self.measurements = {}

    def pose(self, angles):
        angles = self.angles(angles)
        pose = self.measurements.get(angles.tobytes())
        pose = rigid_pose(self.forward_kinematics(angles.copy()) if pose is None else pose)
        key = pose.tobytes()
        if key in self.targets and not np.array_equal(self.targets[key], angles):
            raise ValueError("Ambiguous head FK pose; cannot infer target from pose")
        self.targets[key] = angles.copy()
        return pose

    def record_measured(self, receipt):
        if self.measured_forward_kinematics is None:
            raise ValueError("Measured three-joint FK required for physical arrival")
        angles = self.angles(receipt.joint_angles_deg)
        actual = rigid_pose(receipt.camera_to_world)
        expected = rigid_pose(self.measured_forward_kinematics(angles, receipt.roll_deg))
        if not np.allclose(actual, expected, rtol=0, atol=1e-9):
            raise ValueError("Measured camera pose differs from actual three-joint FK")
        key = angles.tobytes()
        old = self.measurements.get(key)
        if old is not None and not np.array_equal(old, actual):
            raise ValueError("Same measured search angles have changed roll/extrinsics")
        self.measurements[key] = actual.copy()
        self.targets[actual.tobytes()] = angles.copy()

    def target_for_pose(self, pose):
        key = rigid_pose(pose).tobytes()
        if key not in self.targets:
            raise ValueError("Pose was not produced by this episode articulated mount")
        return self.targets[key].copy()


def _input_digest(example):
    h = hashlib.sha256()
    for name in ("observation/image", "observation/state"):
        value = np.ascontiguousarray(example[name])
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    h.update(example["prompt"].encode())
    return h.hexdigest()


class GR1SearchSource:
    """One episode, fresh capture and paired noise at every charged query.

    Backend methods: move_head_degrees, pose_receipt, reset_observation_cache,
    capture_input(epoch). Input capture returns the real ego image, active state,
    and actual task instruction. The real model runtime owns per-call authorization.
    Failure holds this source; it never refunds, retries, or replays a query.
    """

    def __init__(
        self,
        mount,
        backend,
        runtime,
        *,
        noise_bank,
        episode_id,
        model_sha256,
        producer_sha256,
        history_sha256,
        guard,
    ):
        noise = np.asarray(noise_bank)
        if (
            noise.shape != (mount.config.score_samples, 16, 32)
            or noise.dtype != np.float32
            or not np.isfinite(noise).all()
            or not isinstance(episode_id, str)
            or not episode_id
        ):
            raise ValueError("Explicit paired-noise bank and unique episode required")
        self.mount, self.backend, self.runtime = mount, backend, runtime
        self.noise = noise.copy()
        self.noise.flags.writeable = False
        self.episode_id, self.guard = episode_id, guard
        self.model_sha256, self.producer_sha256 = model_sha256, producer_sha256
        self.history_sha256 = history_sha256
        self.epoch = 0
        self.queries = 0
        self.status = "READY"
        self.attempts = []
        self.expected_instruction = None

    def _live(self):
        if self.status != "READY":
            raise RuntimeError("Explicit audit and new episode binding required")
        self.guard()

    def move(self, pose):
        try:
            self._live()
            target = self.mount.target_for_pose(pose)
            receipt = self.backend.move_head_degrees(target)
            if not isinstance(receipt, PoseReceipt):
                raise ValueError("Measured articulated head pose receipt required")
            self.epoch += 1
            self.backend.reset_observation_cache(self.epoch)
            self._live()
            return receipt
        except BaseException:
            self.status = "HOLD"
            raise

    def query(self):
        try:
            self._live()
            if self.queries >= self.mount.config.query_budget:
                raise ValueError("Episode query budget exhausted")
            self.queries += 1
            query_id = f"{self.episode_id}/query-{self.queries}"
            attempt = dict(query_id=query_id, epoch=self.epoch, status="UNKNOWN", members=0)
            self.attempts.append(attempt)
            example, epoch = self.backend.capture_input(self.epoch)
            if epoch != self.epoch:
                raise ValueError("Stale observation cache epoch")
            if (
                self.expected_instruction is not None
                and example["prompt"] != self.expected_instruction
            ):
                raise ValueError("Task instruction changed within the bound episode")
            before = self.backend.pose_receipt()
            rows = []
            for member, noise in enumerate(self.noise):
                self._live()
                rows.append(
                    self.runtime.infer(
                        example,
                        noise,
                        request=dict(query_id=query_id, epoch=self.epoch, member=member),
                    )
                )
                attempt["members"] += 1
            after = self.backend.pose_receipt()
            if (
                not before.settled
                or not after.settled
                or before.physical_state_id != after.physical_state_id
                or not np.array_equal(before.camera_to_world, after.camera_to_world)
            ):
                raise ValueError("Scene/head moved while acquiring paired scores")
            feature = np.asarray(rows[0]["feature"])
            if any(not np.array_equal(row["feature"], feature) for row in rows[1:]):
                raise ValueError("Pre-language visual tap unexpectedly depends on action noise")
            scores = np.asarray([row["score"] for row in rows], dtype=float)
            if not np.isfinite(scores).all() or np.any(scores < 0):
                raise ValueError("Invalid native scoring output")
            packet = NativeFeatures(
                layout_spec(STREAMS),
                query_id,
                _input_digest(example),
                self.history_sha256,
                self.model_sha256,
                (feature.copy(),),
                self.producer_sha256,
            )
            packet.validate()
            self._live()
            attempt["status"] = "COMPLETE"
            return ViewMeasurement(
                after,
                NativeQueryResult(
                    packet.packed(), float(scores.mean()), float(scores.std()), len(scores), packet
                ),
            )
        except BaseException:
            self.status = "HOLD"
            raise


def collector_session(
    source, *, initial_angles, physical_state_id, score_scale, seed, settling_tolerance_deg=None
):
    """Same native collector and STOP lifecycle used by other model adapters."""
    cls = NativeContinuousSearch if settling_tolerance_deg is None else MeasuredHeadSearch
    options = (
        {}
        if settling_tolerance_deg is None
        else dict(settling_tolerance_deg=settling_tolerance_deg)
    )
    core = cls(
        source.mount,
        source,
        streams=STREAMS,
        producer_sha256=source.producer_sha256,
        initial_angles=initial_angles,
        physical_state_id=physical_state_id,
        score_scale=score_scale,
        seed=seed,
        **options,
    )
    if settling_tolerance_deg is not None:
        receipt = source.backend.pose_receipt()
        core._accept_measured(receipt)
        if not np.array_equal(
            source.mount.angles(initial_angles), source.mount.angles(receipt.joint_angles_deg)
        ):
            raise ValueError("Initial search angles must be the measured starting pose")
    return native_gym(STREAMS, producer_sha256=source.producer_sha256), {"session": core}
