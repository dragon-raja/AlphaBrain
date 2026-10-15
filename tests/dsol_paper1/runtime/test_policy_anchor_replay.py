import numpy as np
import pytest

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.evaluation_matrix import NUMERIC_FLAGS
from benchmarks.Robocasa365.policy_replay import replay_score_anchor


@pytest.mark.parametrize('mismatch', [False, True])
def test_host_admission_compares_actual_frozen_arrays(tmp_path, monkeypatch, mismatch):
    velocity = np.ones((8, 10, 50, 32)); actions = np.ones((8, 50, 12))
    np.savez(tmp_path/'anchor.npz', velocity_trace=velocity, actions=actions,
             **{'observation/state': np.zeros(12)})
    write(tmp_path/'anchor.json', dict(model='official', score_repeats=8, action_steps=0,
          arrays_sha256=sha(tmp_path/'anchor.npz'), prompt='task', initial='i', view='canonical'))
    write(tmp_path/'cfg.json', dict(output=str(tmp_path/'result.json'), checkpoint='checkpoint', model_config='config',
          anchors=[dict(path=str(tmp_path/'anchor.json'), sha256=sha(tmp_path/'anchor.json'))]))
    class Policy:
        def __init__(self, *_): pass
        def trace(self, request, noise):
            assert request['prompt']=='task' and noise.shape==(50,32)
            return dict(velocity=velocity[:1]+int(mismatch), actions=actions[0])
    monkeypatch.setattr('benchmarks.Robocasa365.policy_replay.ResidentPolicy', Policy)
    monkeypatch.setenv('XLA_FLAGS', NUMERIC_FLAGS)
    if mismatch:
        with pytest.raises(ValueError, match='no production admission'): replay_score_anchor(tmp_path/'cfg.json')
        assert read(tmp_path/'result.json')['status']=='HOLD'
    else:
        assert replay_score_anchor(tmp_path/'cfg.json')['status']=='PASS'
        assert len(read(tmp_path/'result.json')['rows'])==8
