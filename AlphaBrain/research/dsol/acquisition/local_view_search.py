"""Continuous camera setpoints and budgeted dynamic proposals; no runtime imports.

The first mount is a fixed optical centre, NOT a humanoid or collision model.
Backends must settle at every setpoint and score only explicitly queried views.
No 97-view catalog, unseen images, outcome labels or privileged critic is used.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..data.search_replay import QueryResult
from .search_env import unit_vector


@dataclass(frozen=True)
class LocalViewConfig:
    yaw_limit_deg: float = 30.
    pitch_limit_deg: float = 20.
    step_limit_deg: float = 10.
    min_proposal_deg: float = 2.
    query_budget: int = 8
    query_cost: float = .01
    score_samples: int = 2

    def __post_init__(self):
        values = (self.yaw_limit_deg, self.pitch_limit_deg, self.step_limit_deg,
                  self.min_proposal_deg, self.query_cost)
        if (not np.isfinite(values).all() or min(values[:4]) <= 0
                or self.min_proposal_deg > self.step_limit_deg or self.query_cost < 0
                or type(self.query_budget) is not int or not 2 <= self.query_budget <= 16
                or type(self.score_samples) is not int or self.score_samples < 1):
            raise ValueError('Invalid local camera bounds or query budget')


def rigid_pose(value):
    pose = np.array(value, dtype=np.float64, copy=True)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-6, rtol=0)):
        raise ValueError('Finite rigid camera-to-world OpenCV pose required')
    return pose


class PanTiltMount:
    """OpenCV frame: +X right, +Y down, +Z optical forward.

    yaw>0 looks right about mount +Y; pitch>0 looks up about yawed +X.
    R = R_mount Ry(yaw) Rx(pitch). No target-dependent look-at rotation.
    Limits are relative to installation, NOT a supplied optimal-view target.
    """

    def __init__(self, camera_to_world, config=LocalViewConfig()):
        self.reference = rigid_pose(camera_to_world)
        self.config = config

    def angles(self, value):
        angles = np.array(value, dtype=float, copy=True)
        if (angles.shape != (2,) or not np.isfinite(angles).all()
                or np.any(np.abs(angles) > [self.config.yaw_limit_deg, self.config.pitch_limit_deg])):
            raise ValueError('Yaw/pitch outside mount limits')
        return angles

    def pose(self, angles):
        yaw, pitch = np.deg2rad(self.angles(angles))
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        pose = self.reference.copy()
        pose[:3, :3] = pose[:3, :3] @ ry @ rx
        return pose

    def route(self, start, target):
        start, target = self.angles(start), self.angles(target)
        distance = np.max(np.abs(target - start))
        if distance == 0:
            return []
        count = int(np.ceil(distance / self.config.step_limit_deg))
        return [start + (target - start) * (i / count) for i in range(1, count)] + [target.copy()]

    def propose(self, current, history, *, seed, decision):
        """Four axial and four random directions, continuous lengths; no query.

        IDs depend only on the caller's trial seed and decision counter. Invalid
        points are removed, never clipped/replaced or projected onto a catalog.
        """
        if any(type(x) is not int or x < 0 for x in (seed, decision)):
            raise ValueError('Explicit nonnegative trial seed and decision required')
        current = self.angles(current)
        history = [self.angles(x) for x in history]
        rng = np.random.default_rng(np.random.SeedSequence([seed, decision]))
        phi = rng.uniform(0, 2 * np.pi, 4)
        directions = np.vstack((np.eye(2), -np.eye(2), np.column_stack((np.cos(phi), np.sin(phi)))))
        lengths = rng.uniform(self.config.min_proposal_deg, self.config.step_limit_deg, 8)
        accepted = []
        for direction, length in zip(directions, lengths):
            target = current + direction * length
            try:
                self.angles(target)
            except ValueError:
                continue
            if any(np.allclose(target, x, atol=1e-8, rtol=0) for x in history + accepted):
                continue
            accepted.append(target)
        return np.asarray(accepted, dtype=float).reshape(-1, 2)


@dataclass(frozen=True)
class PoseReceipt:
    camera_to_world: np.ndarray
    physical_state_id: str
    settled: bool = True


@dataclass(frozen=True)
class ViewMeasurement:
    receipt: PoseReceipt
    value: QueryResult


class ContinuousViewSource(Protocol):
    """Backend owns real motion/rendering and a fixed paired scoring-noise bank."""

    def move(self, camera_to_world: np.ndarray) -> PoseReceipt: ...

    def query(self) -> ViewMeasurement: ...


class LocalViewSearch:
    """Single-use, fail-closed search session. STOP returns to historical best.

    Every attempted query is charged, including the initial and return query.
    One query slot is reserved for return verification. Intermediate motion
    setpoints do not query. Return discrepancy stops the session, never falls
    back, retries, or silently chooses another view.
    """

    def __init__(self, mount, source, *, initial_angles, physical_state_id, wrist,
                 score_scale, seed):
        self.mount, self.source, self.config = mount, source, mount.config
        self.initial = mount.angles(initial_angles)
        self.current = self.initial.copy()
        self.physical = physical_state_id
        self.wrist = np.array(wrist, dtype=float, copy=True)
        if (not isinstance(physical_state_id, str) or not physical_state_id
                or self.wrist.ndim != 1 or len(self.wrist) == 0 or not np.isfinite(self.wrist).all()
                or not np.isfinite(score_scale) or score_scale <= 0
                or type(seed) is not int or seed < 0):
            raise ValueError('Explicit physical identity, wrist features, scale and seed required')
        self.scale, self.seed = float(score_scale), seed
        self.records, self.events = [], []
        self.queries = self.decision = self.best = 0
        self.status = 'NEW'
        self.reason = None
        self.proposals = np.empty((0, 2))

    def _check_receipt(self, receipt, pose):
        if (receipt.settled is not True or receipt.physical_state_id != self.physical
                or not np.allclose(rigid_pose(receipt.camera_to_world), pose, atol=1e-9, rtol=0)):
            raise ValueError('Unsettled camera, pose mismatch or physical state changed')

    def _move(self, target):
        for angles in self.mount.route(self.current, target):
            pose = self.mount.pose(angles)
            self._check_receipt(self.source.move(pose.copy()), pose)
            self.current = angles.copy()
            self.events.append(dict(kind='move', angles=angles.tolist()))

    def _query(self, kind):
        if self.queries >= self.config.query_budget:
            raise ValueError('Query budget exhausted')
        self.queries += 1  # Attempt charged before backend execution, even on failure.
        self.events.append(dict(kind=kind, query=self.queries, angles=self.current.tolist()))
        measurement = self.source.query()
        self._check_receipt(measurement.receipt, self.mount.pose(self.current))
        value = measurement.value
        visual = np.array(value.visual, dtype=float, copy=True)
        if (visual.shape != self.wrist.shape or not np.isfinite(visual).all()
                or not np.isfinite([value.accel_mean, value.accel_std]).all()
                or min(value.accel_mean, value.accel_std) < 0
                or value.score_samples != self.config.score_samples):
            raise ValueError('Invalid actual view measurement')
        return QueryResult(visual, float(value.accel_mean), float(value.accel_std), value.score_samples)

    def _refresh(self):
        self.proposals = self.mount.propose(self.current, [r[0] for r in self.records],
                                           seed=self.seed, decision=self.decision)

    def start(self):
        if self.status != 'NEW':
            raise ValueError('Session cannot restart; create a new explicitly authorized session')
        try:
            value = self._query('initial')
            self.records.append((self.current.copy(), value))
            self.status = 'ACTIVE'
            self._refresh()
        except Exception:
            self.status, self.reason = 'FAILED', 'initial_query'
            raise
        return self.observation()

    def step(self, action):
        if self.status != 'ACTIVE':
            raise ValueError('Search session is not active')
        masks = self.observation()['action_mask']
        if (not isinstance(action, (int, np.integer)) or not 0 <= action < len(masks)
                or not masks[action]):
            raise ValueError('STOP or a legal current proposal required')
        before, old_queries = self.records[self.best][1].accel_mean, self.queries
        try:
            if action == 0:
                self._finish('stop')
            else:
                target = self.proposals[action - 1 - len(self.records)].copy()
                self._move(target)
                value = self._query('search')
                self.records.append((target, value))
                if value.accel_mean < self.records[self.best][1].accel_mean:
                    self.best = len(self.records) - 1
                self.decision += 1
                self._refresh()
                if self.queries >= self.config.query_budget - 1:
                    self._finish('budget')
                elif not len(self.proposals):
                    self._finish('no_candidates')
        except Exception:
            self.status, self.reason = 'FAILED', 'backend_or_verification'
            raise
        reward = (before - self.records[self.best][1].accel_mean) / self.scale
        reward -= self.config.query_cost * (self.queries - old_queries)
        return self.observation(), float(reward), self.status == 'COMPLETE', False, self.info()

    def _finish(self, reason):
        target, expected = self.records[self.best]
        if not np.array_equal(self.current, target):
            self._move(target)
            actual = self._query('return_verification')
            if (actual.accel_mean != expected.accel_mean or actual.accel_std != expected.accel_std
                    or not np.array_equal(actual.visual, expected.visual)):
                raise ValueError('Return query differs under fixed state/history/noise')
        self.status, self.reason = 'COMPLETE', reason

    def observation(self):
        """History rows + fresh proposal rows use the existing actor schema.

        No new weights or changed feature dimensions: old SharedSearchActor can
        initialize the next model. Limits constrain proposals, but are NOT yet
        learned joint-limit features. Old weights passing this interface is not
        evidence of effective search in the new domain.
        """
        if not self.records or self.status == 'FAILED':
            raise ValueError('No valid active/complete observation')
        history = len(self.records)
        angles = [r[0] for r in self.records] + list(self.proposals)
        poses = np.stack([self.mount.pose(a) for a in angles])
        origin = self.mount.pose(self.initial)
        geometry = np.zeros((len(angles), 12), dtype=np.float32)
        geometry[:, :3] = (poses[:, :3, 3] - origin[:3, 3]) @ origin[:3, :3] / .1
        geometry[:, 3:] = (origin[:3, :3].T @ poses[:, :3, :3]).reshape(-1, 9)
        visual = np.zeros((len(angles), len(self.wrist)), dtype=np.float32)
        scores = np.zeros((len(angles), 2), dtype=np.float32)
        flags = np.zeros((len(angles), 4), dtype=np.float32)
        for i, (angle, value) in enumerate(self.records):
            visual[i] = unit_vector(value.visual)
            scores[i] = np.array([value.accel_mean, value.accel_std]) / self.scale
            flags[i, :2] = [1, (i + 1) / 16.]
            flags[i, 2] = np.array_equal(angle, self.current)
        flags[self.best, 3] = 1
        mask = np.zeros(len(angles) + 1, dtype=np.int8)
        if self.status == 'ACTIVE':
            mask[0] = 1
            if self.queries < self.config.query_budget - 1:
                mask[1 + history:] = 1
        return dict(geometry=geometry, visual=visual, scores=scores, flags=flags,
                    wrist=unit_vector(self.wrist).astype(np.float32),
                    **{'global': np.array([self.queries / 16.,
                        (self.config.query_budget - self.queries) / 16., self.config.query_budget / 16.,
                        self.records[0][1].accel_mean / self.scale,
                        self.records[self.best][1].accel_mean / self.scale,
                        self.config.score_samples / 8.], dtype=np.float32)}, action_mask=mask)

    def info(self):
        return dict(status=self.status, reason=self.reason, queries=self.queries,
                    scoring_members=self.queries * self.config.score_samples,
                    initial_query_cost=self.config.query_cost,
                    selected_angles=self.records[self.best][0].tolist() if self.records else None,
                    current_angles=self.current.tolist(), visited=len(self.records))
