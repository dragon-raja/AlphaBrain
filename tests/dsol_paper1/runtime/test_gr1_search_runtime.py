"""CPU-only composition tests; features/scores are explicitly synthetic."""
from contextlib import contextmanager
import json
import numpy as np
import pytest

from benchmarks.Robocasa_tabletop.search_runtime import (
    ArticulatedHeadMount, GR1SearchSource, STREAMS, collector_session, MeasuredHeadReceipt,
)
from benchmarks.Robocasa_tabletop.model_runtime import ScoreContract
from AlphaBrain.research.dsol.acquisition.local_view_search import LocalViewConfig, PoseReceipt
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores


def fk(angles):
    # A translating head fixture deliberately differs from fixed optical centre.
    pose = np.eye(4); pose[:2, 3] = np.asarray(angles) / 100
    return pose


class Backend:
    def __init__(self):
        self.angles = np.zeros(2); self.epoch = 0; self.stale = False

    def pose_receipt(self):
        return PoseReceipt(fk(self.angles), 'cpu-scene')

    def move_head_degrees(self, angles):
        self.angles = angles.copy(); return self.pose_receipt()

    def reset_observation_cache(self, epoch):
        self.epoch = epoch

    def capture_input(self, epoch):
        state = np.zeros(29, np.float32); state[:2] = self.angles
        return {'observation/image': np.zeros((2, 2, 3), np.uint8),
                'observation/state': state, 'prompt': 'synthetic CPU fixture'}, self.epoch - self.stale


class Model:
    def __init__(self, fail_member=None):
        self.calls = []; self.fail_member = fail_member

    def infer(self, example, noise, *, request):
        self.calls.append(request.copy())
        if request['member'] == self.fail_member:
            raise ConnectionError('synthetic unknown model-call outcome')
        angle = example['observation/state'][:2]
        return dict(feature=np.ones(4608, np.float32), score=1 + float(np.square(angle-3).sum()) / 100)


def source(episode=0, model=None):
    mount = ArticulatedHeadMount(LocalViewConfig(), fk, joint_limits_deg=[[-40, 40], [-25, 25]])
    return GR1SearchSource(mount, Backend(), model or Model(),
        noise_bank=np.zeros((2, 16, 32), np.float32), episode_id=f'cpu/{episode}',
        model_sha256='a'*64, producer_sha256='b'*64, history_sha256='c'*64, guard=lambda: None)


def test_articulated_geometry_and_limits():
    s = source(); pose = s.mount.pose([5, 2])
    assert np.array_equal(pose[:2, 3], [.05, .02])
    assert np.array_equal(s.mount.target_for_pose(pose), [5, 2])
    with pytest.raises(ValueError, match='joint limits'):
        ArticulatedHeadMount(LocalViewConfig(), fk, joint_limits_deg=[[-5, 5], [-5, 5]])


def test_partial_query_unknown_never_retried():
    s = source(model=Model(fail_member=1))
    with pytest.raises(ConnectionError): s.query()
    assert s.queries == 1 and s.status == 'HOLD'
    assert s.attempts[0]['status'] == 'UNKNOWN' and s.attempts[0]['members'] == 1
    with pytest.raises(RuntimeError): s.query()
    assert len(s.runtime.calls) == 2


def test_stale_epoch_rejected_before_model():
    s = source(); s.backend.stale = True
    with pytest.raises(ValueError, match='epoch'): s.query()
    assert not s.runtime.calls and s.status == 'HOLD'


def test_score_uses_only_executed_actions_and_active_dimensions():
    trace = np.random.default_rng(4).normal(size=(1, 10, 16, 32)).astype(np.float32)
    contract = ScoreContract(10, 10, 8, 8)
    expected = compute_accel_scores(trace[:, :, :8, :29], prefixes=[10])[10]['score'][0]
    assert contract.score(trace) == expected
    trace[:, :, 8:] = 1e9; trace[:, :, :, 29:] = -1e9
    assert contract.score(trace) == expected
    with pytest.raises(ValueError): ScoreContract(10, 10, 8, 16).validate()


def test_cpu_loaded_model_cannot_bypass_production_gate():
    from benchmarks.Robocasa_tabletop.model_runtime import GR1ModelRuntime
    from AlphaBrain.research.dsol.execution.dispatch_policy import ProductionDispatchDisabled
    runtime = object.__new__(GR1ModelRuntime)
    with pytest.raises(ProductionDispatchDisabled):
        runtime.infer({},np.zeros((16,32),np.float32),request={})


def test_native_collector_move_stop_return():
    pytest.importorskip('gymnasium')
    s = source(); env, options = collector_session(s, initial_angles=[0, 0],
        physical_state_id='cpu-scene', score_scale=1., seed=2)
    obs, _ = env.reset(options=options)
    assert obs['feature_0'].shape == (7, 4608)
    assert obs['feature_mask'][0].tolist() == [True]
    assert 'wrist' not in obs
    env.step({'stop': 0, 'delta_deg': np.array([3., 3.])})
    _, _, terminated, _, _ = env.step({'stop': 1, 'delta_deg': np.zeros(2)})
    assert terminated and all(a['status'] == 'COMPLETE' for a in s.attempts)
    assert len(s.runtime.calls) == 2*s.queries


def test_measured_arrival_records_real_angles_and_charges_fresh_return():
    pytest.importorskip('gymnasium')
    class MeasuredBackend(Backend):
        def pose_receipt(self):
            return MeasuredHeadReceipt(fk(self.angles),'cpu-scene',True,tuple(self.angles),0.)
        def move_head_degrees(self,angles):
            self.angles=angles.copy()+.01
            return self.pose_receipt()
    s=source();s.backend=MeasuredBackend();s.mount.measured_forward_kinematics=lambda angles,roll:fk(angles)
    env,options=collector_session(s,initial_angles=[0,0],physical_state_id='cpu-scene',
                                 score_scale=1.,seed=2,settling_tolerance_deg=.02)
    env.reset(options=options)
    env.step({'stop':0,'delta_deg':np.array([3.,3.])})
    core=options['session']
    assert np.allclose(core.records[1][0],[3.01,3.01])
    before=core.records[1][1].accel_mean
    env.step({'stop':0,'delta_deg':np.array([3.,3.])})
    _,_,done,_,_=env.step({'stop':1,'delta_deg':np.zeros(2)})
    assert done and s.queries==4 and len(s.runtime.calls)==8
    assert np.allclose(core.current,[3.02,3.02])
    assert core.records[1][1].accel_mean==before
    assert core.return_measurement['actual_accel_mean']!=before
    assert core.return_measurement['historical_input_sha256']!=core.return_measurement['input_sha256']


def test_shared_ppo_complete_batch_and_resume(tmp_path):
    torch = pytest.importorskip('torch'); pytest.importorskip('gymnasium')
    from AlphaBrain.research.dsol.selectors.unified_search_policy import UnifiedActorCritic
    from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO, ContinuousPPOConfig
    from AlphaBrain.research.dsol.data.search_core import core_spec
    from AlphaBrain.research.dsol.data.reward_pilot import reward_config
    torch.set_num_threads(1)

    @contextmanager
    def backend(row, output):
        s = source(row['episode'])
        yield collector_session(s, initial_angles=[0, 0], physical_state_id='cpu-scene',
                                score_scale=1., seed=row['episode'])

    def trainer(path, resume=False):
        torch.manual_seed(7)
        policy = UnifiedActorCritic(core_spec('U12', width=16), streams=STREAMS,
                                   groups=('ego',), producer_sha256='b'*64)
        return ContinuousPPO(policy, ContinuousPPOConfig(updates=2),
            [dict(episode=i) for i in range(64)], backend, path,
            binding=dict(source_kind='SYNTHETIC_GR1_CPU', retained_checkpoint_updates=[1]),
            reward_config=reward_config('G'), resume=resume)

    def snapshot(t):
        return t.store.load(json.loads((t.store.root/'latest.json').read_text()))

    with trainer(tmp_path/'whole') as t:
        t.run(); expected = snapshot(t)
    with trainer(tmp_path/'split') as t:
        for _ in range(17): t.collect_one()
    with trainer(tmp_path/'split', resume=True) as t:
        t.run(); actual = snapshot(t)
    assert actual['cursor'] == 64 and actual['update'] == 2

    def same(a, b):
        if isinstance(a, torch.Tensor): assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a: same(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b): same(x, y)
        else: assert a == b
    same(expected, actual)
