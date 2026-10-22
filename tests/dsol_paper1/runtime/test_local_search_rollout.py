import json
from pathlib import Path
import numpy as np
import pytest
from benchmarks.LIBERO.local_search_rollout import rollout_local_episode, input_identity


@pytest.mark.parametrize('suite,horizon', [('libero_goal', 300), ('libero_spatial', 220), ('libero_object', 280), ('libero_10', 520)])
@pytest.mark.parametrize('success', [False, True])
def test_explicit_horizon_full_predictions_and_input_noise_identity(suite, horizon, success):
    plan = json.loads((Path(__file__).resolve().parents[3]/'configs/dsol_paper1/search/dynamic_local_v1.json').read_text())
    example = dict(image=[np.zeros((2,2,3), np.uint8)]*2, state=np.zeros(8),
                   camera_intrinsics=np.eye(3), camera_to_world_opencv=np.eye(4), lang='fixture')
    events, requests = [], []
    class Bank:
        def get(self, state, repeat, replan):
            assert state == 'fixed' and repeat == 3
            return dict(noise=np.zeros((10,7)), noise_sha256=str(replan))
    def request(command):
        requests.append(command)
        assert command['actions'].shape == (5, 7)
        return dict(example=example, executed=2 if success else 5, success=success)
    result, arrays = rollout_local_episode(request, lambda *a, **kw: dict(actions=np.full((10,7),2.)),
        example, Bank(), 'fixed', 3, {}, plan=plan, suite=suite, record=events.append)
    replans = 1 if success else horizon // 5
    assert result['steps'] == (2 if success else horizon) and result['replans'] == replans
    assert arrays['full_clipped_action_chunks'].shape == (replans, 10, 7)
    assert np.all(arrays['full_clipped_action_chunks'] == 1)
    assert len(events) == 3 * replans
    assert result['input_sha256'] == [input_identity(example)] * replans
    assert result['execution_noise_sha256'] == list(map(str, range(replans)))
    changed = dict(example, lang='different'); assert input_identity(changed) != input_identity(example)


@pytest.mark.parametrize('failure', ['nan', 'partial', 'camera', 'inference'])
def test_fail_closed_keeps_pre_failure_journal(failure):
    plan = json.loads((Path(__file__).resolve().parents[3]/'configs/dsol_paper1/search/dynamic_local_v1.json').read_text())
    example = dict(state=np.zeros(8), camera_intrinsics=np.eye(3), camera_to_world_opencv=np.eye(4))
    events = []
    class Bank:
        def get(self, *args): return dict(noise=np.zeros((10,7)), noise_sha256='fixed')
    def infer(*args, **kwargs):
        if failure == 'inference': raise RuntimeError('injected')
        return dict(actions=np.full((10,7), np.nan if failure == 'nan' else 0.))
    def request(command):
        changed = dict(example)
        if failure == 'camera': changed['camera_to_world_opencv'] = np.zeros((4,4))
        return dict(example=changed, executed=1 if failure == 'partial' else 5, success=False)
    with pytest.raises((RuntimeError, ValueError)):
        rollout_local_episode(request, infer, example, Bank(), 'fixed', 0, {},
                              plan=plan, suite='libero_goal', record=events.append)
    assert events[0]['kind'] == 'inference_attempt'
    assert sum(e['kind'] == 'inference_attempt' for e in events) == 1
