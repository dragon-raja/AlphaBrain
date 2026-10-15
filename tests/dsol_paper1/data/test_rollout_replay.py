import numpy as np
import pytest

from AlphaBrain.research.dsol.data.artifacts import read, sha
from AlphaBrain.research.dsol.data.rollout_replay import ReplayRecorder, ActionReplay, array_identity


def test_records_are_copies_and_cannot_overwrite(tmp_path):
    recorder = ReplayRecorder()
    state = np.array([1., 2.], dtype=np.float32)
    recorder.record_plan(0, {'prompt': 'task', 'observation/state': state}, state, state, 'physical')
    recorder.record_step(0, state, 'after', False)
    state[:] = 9
    target = tmp_path / 'replay'
    record = read(recorder.save(target))
    assert record['arrays_sha256'] == sha(target / 'arrays.npz')
    with np.load(target / 'arrays.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['plan_0000/observation/state'], [1., 2.])
    assert record['plans'][0]['arrays']['noise'] == array_identity(np.array([1., 2.], np.float32))
    with pytest.raises(FileExistsError):
        recorder.save(target)
    replay = ActionReplay(target / 'replay.json')
    np.testing.assert_array_equal(replay.plan(0, 0, np.array([1., 2.], np.float32)), [1., 2.])
    with pytest.raises(ValueError):
        replay.plan(0, 1, np.array([1., 2.], np.float32))
