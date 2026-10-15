import numpy as np

from AlphaBrain.research.dsol.analysis.rollout_replay import compare_replays
from AlphaBrain.research.dsol.data.rollout_replay import ReplayRecorder


def test_distinguishes_observation_change_from_physics_change(tmp_path):
    records = []
    for index in range(2):
        recorder = ReplayRecorder()
        recorder.record_plan(0, dict(prompt='task', image=np.array([index], np.uint8)),
                             np.zeros(2), np.zeros((2, 2)), 'same')
        recorder.record_step(0, np.zeros(2), 'after', False)
        records.append(recorder.save(tmp_path / str(index)))
    result = compare_replays(*records)
    assert result['first_plan_difference']['array_differences']['image']['max_abs'] == 1.
    assert result['first_plan_difference']['physical_equal']
    assert result['first_step_difference'] is None
    assert result['matched_step_physics']
