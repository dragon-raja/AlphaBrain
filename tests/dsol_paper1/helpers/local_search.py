"""Explicitly synthetic search source. No cached scientific labels or model."""
import numpy as np
from AlphaBrain.research.dsol.acquisition.local_view_search import (
    LocalViewSearch, PanTiltMount, PoseReceipt, ViewMeasurement)
from AlphaBrain.research.dsol.data.search_replay import QueryResult


def synthetic_session():
    mount = PanTiltMount(np.eye(4))
    class Source:
        pose = np.eye(4)
        calls = 0
        fail = False
        def move(self, pose):
            self.pose = pose.copy()
            return PoseReceipt(self.pose.copy(), 'synthetic')
        def query(self):
            self.calls += 1
            if self.fail:
                raise RuntimeError('injected failure')
            score = 1. if np.array_equal(self.pose, np.eye(4)) else 2.
            return ViewMeasurement(PoseReceipt(self.pose.copy(), 'synthetic'),
                QueryResult(np.ones(4), score, 0., 2))
    source = Source()
    return LocalViewSearch(mount, source, initial_angles=[0., 0.],
        physical_state_id='synthetic', wrist=np.ones(4), score_scale=1., seed=19), source
