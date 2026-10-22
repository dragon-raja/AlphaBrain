"""Continuous increments + STOP on the existing fixed-state query lifecycle."""
from typing import Protocol
import time
import numpy as np
from .local_view_search import LocalViewSearch, rigid_pose


def validate_head_calibration(*, world_from_base, head_from_camera, intrinsics):
    """Required measured transforms; no identity-transform default for missing calibration."""
    base, camera = rigid_pose(world_from_base), rigid_pose(head_from_camera)
    intrinsics = np.array(intrinsics, dtype=float, copy=True)
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0
            or not np.array_equal(intrinsics[2], [0, 0, 1])):
        raise ValueError('Explicit calibrated pinhole intrinsics required')
    return base, camera, intrinsics


class CameraKinematics(Protocol):
    """Radians/metres for a future URDF backend; no guessed camera extrinsic.

    A real backend must explicitly supply world_from_base and head_from_camera,
    joint ordering, limits, collision/settling checks and measured pose receipt.
    PanTiltMount instead exposes virtual degrees and a fixed optical centre.
    Adapters must declare which coordinate convention they implement.
    """
    def camera_pose(self, joint_radians, *, world_from_base, head_from_camera): ...


class ContinuousSearch(LocalViewSearch):
    def _move(self, target):
        start = time.perf_counter()
        try:
            return super()._move(target)
        finally:
            self.events.append(dict(kind='movement_elapsed', seconds=time.perf_counter()-start))

    def _refresh(self):
        self.proposals = np.empty((0, 2))  # No proposal list is passed to the actor.

    def observation(self):
        obs = super().observation()
        bounds = np.array([self.config.yaw_limit_deg, self.config.pitch_limit_deg])
        obs['joint_state'] = np.concatenate((self.current / bounds,
            (self.current + bounds) / (2 * bounds), (bounds - self.current) / (2 * bounds))).astype(np.float32)
        obs['delta_low'] = np.maximum(-self.config.step_limit_deg, -bounds - self.current).astype(np.float64)
        obs['delta_high'] = np.minimum(self.config.step_limit_deg, bounds - self.current).astype(np.float64)
        return obs

    def step(self, action):
        if self.status != 'ACTIVE':
            raise ValueError('Search is not active')
        if set(action) != {'stop', 'delta_deg'} or type(action['stop']) is not bool:
            raise ValueError('Explicit STOP and two degree increments required')
        delta = np.asarray(action['delta_deg'], dtype=float)
        obs = self.observation()
        if delta.shape != (2,) or not np.isfinite(delta).all():
            raise ValueError('Finite two-axis increment required')
        if action['stop']:
            if np.any(delta): raise ValueError('STOP has no motion branch')
        elif (np.any(delta < obs['delta_low']) or np.any(delta > obs['delta_high'])
                or self.queries >= self.config.query_budget - 1):
            raise ValueError('Increment outside legal interval; never clip or snap')
        before, old_queries = self.records[self.best][1].accel_mean, self.queries
        start = time.perf_counter()
        try:
            if action['stop']:
                self._finish('stop')
            else:
                target = self.mount.angles(self.current + delta)
                self._move(target)
                value = self._query('search')
                self.records.append((target.copy(), value))
                if value.accel_mean < self.records[self.best][1].accel_mean:
                    self.best = len(self.records) - 1
                self.decision += 1
                if self.queries >= self.config.query_budget - 1:
                    self._finish('budget')
        except Exception:
            self.status, self.reason = 'FAILED', 'backend_or_verification'
            raise
        self.events.append(dict(kind='decision_elapsed', seconds=time.perf_counter()-start))
        reward = (before - self.records[self.best][1].accel_mean) / self.scale
        reward -= self.config.query_cost * (self.queries - old_queries)
        return self.observation(), float(reward), self.status == 'COMPLETE', False, self.info()
