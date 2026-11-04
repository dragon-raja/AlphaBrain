"""CPU-testable GR1 head observation port; no simulator or model is constructed.

The injected backend must be admitted separately. This port owns freshness and
head-only motion checks, not PPO, query selection, budgets or dispatch. Callers
use the existing continuous-search lifecycle and center authority/phase ledger.
"""

import numpy as np
from AlphaBrain.research.dsol.data.native_observation import ObservationBundle


class HeadObservationPort:
    def __init__(self, backend, *, joint_limits_rad, tolerance_rad, guard):
        limits = np.asarray(joint_limits_rad, dtype=float)
        if (
            limits.shape != (2, 2)
            or not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1])
            or not np.isfinite(tolerance_rad)
            or tolerance_rad <= 0
        ):
            raise ValueError("Explicit audited yaw/pitch limits and tolerance required")
        self.backend, self.guard = backend, guard
        self.limits, self.tolerance = limits.copy(), float(tolerance_rad)
        self.status = "READY"
        self.queries = set()
        self.epoch = 0
        self.last_settled = None

    def _live(self):
        if self.status != "READY":
            raise RuntimeError("Explicit audit required before a new port/session")
        try:
            self.guard()
        except BaseException:
            self.status = "HOLD"
            raise

    def move(self, target_rad):
        self._live()
        target = np.asarray(target_rad, dtype=float)
        if (
            target.shape != (2,)
            or not np.isfinite(target).all()
            or np.any(target < self.limits[:, 0])
            or np.any(target > self.limits[:, 1])
        ):
            raise ValueError("Illegal head target; never clip")
        try:
            before = self.backend.non_head_state_identity()
            receipt = self.backend.move_head(target.copy())
            actual = np.asarray(receipt["measured_joint_rad"], dtype=float)
            if (
                actual.shape != (2,)
                or not np.isfinite(actual).all()
                or not receipt["settled"]
                or np.max(np.abs(actual - target)) > self.tolerance
                or not np.isfinite(receipt["settled_at"])
                or self.backend.non_head_state_identity() != before
            ):
                raise ValueError("Motion changed fixed scene or did not settle at measured target")
            self.guard()
            self.epoch += 1
            self.backend.reset_observation_cache(self.epoch)
            self.last_settled = float(receipt["settled_at"])
            return dict(receipt, cache_epoch=self.epoch)
        except BaseException:
            self.status = "HOLD"
            raise

    def capture(self, query_id):
        self._live()
        if not isinstance(query_id, str) or not query_id or query_id in self.queries:
            raise ValueError("Unique explicit query identity required; no automatic replay")
        if self.last_settled is None:
            raise ValueError("Explicit settled initial pose required")
        self.queries.add(query_id)  # Failure leaves this query spent/unknown, never retried.
        try:
            self.guard()
            bundle, epoch = self.backend.capture(query_id)
            if not isinstance(bundle, ObservationBundle):
                raise ValueError("Native observation metadata required")
            bundle.validate()
            if (
                epoch != self.epoch
                or bundle.query_id != query_id
                or bundle.settled_at < self.last_settled
                or [(c.camera_id, c.role, c.valid) for c in bundle.cameras]
                != [("ego", "movable", True)]
            ):
                raise ValueError("Wrong epoch, stale frame or fabricated GR1 camera")
            self.guard()
            return bundle
        except BaseException:
            self.status = "HOLD"
            raise

    def stop(self):
        """End acquisition; final selected-view return remains the search core's job."""
        self._live()
        self.status = "STOPPED"
        return dict(status=self.status, cache_epoch=self.epoch, queries=len(self.queries))
