"""Complete synthetic CPU searches and optimizer recovery; zero VLA calls."""
from contextlib import contextmanager, closing
import json
import numpy as np
import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('gymnasium')
from AlphaBrain.research.dsol.selectors.unified_search_policy import UnifiedActorCritic
from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO, ContinuousPPOConfig
from AlphaBrain.research.dsol.data.search_core import core_spec
from AlphaBrain.research.dsol.data.reward_pilot import reward_config
from AlphaBrain.research.dsol.data.continuous_training_store import ContinuousTrainingStore
from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from AlphaBrain.research.dsol.acquisition.continuous_search_gym import ContinuousSearchGym
from tests.dsol_paper1.helpers.local_search import synthetic_session


@contextmanager
def backend(row, output):
    old, source = synthetic_session()
    core = ContinuousSearch(old.mount, source, initial_angles=[0, 0],
        physical_state_id='synthetic', wrist=np.ones(4), score_scale=1., seed=row['episode'])
    yield ContinuousSearchGym(4), {'session': core}


def trainer(path, variant, *, resume=False):
    torch.set_num_threads(1); torch.manual_seed(7)
    p = UnifiedActorCritic(core_spec(variant, width=16), visual_dim=4)
    return ContinuousPPO(p, ContinuousPPOConfig(updates=2),
        [dict(episode=i) for i in range(64)], backend, path,
        binding=dict(source_kind='SYNTHETIC_CPU', retained_checkpoint_updates=[1]),
        reward_config=reward_config('G'), resume=resume)


@pytest.mark.parametrize('variant', ['U0', 'U1', 'U2', 'U3', 'U12', 'U13', 'U23', 'U123'])
def test_complete_onpolicy_batch_and_exact_recovery(variant, tmp_path):
    def snapshot(t):
        return t.store.load(json.loads((t.store.root/'latest.json').read_text()))
    with trainer(tmp_path/'whole', variant) as t:
        t.run(); expected = snapshot(t)
    with trainer(tmp_path/'split', variant) as t:
        for _ in range(17):t.collect_one()
    with trainer(tmp_path/'split', variant, resume=True) as t:
        t.run(); actual = snapshot(t)
        assert actual['cursor']==64 and actual['update']==2
        assert actual['reward_state']['batches']==2
    def same(a,b):
        if isinstance(a, torch.Tensor): assert torch.equal(a,b)
        elif isinstance(a, dict):
            assert a.keys()==b.keys()
            for k in a:same(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b):same(x,y)
        else:assert a==b
    same(expected, actual)


def test_retained_full_optimizer_checkpoints_survive_new_updates(tmp_path):
    with closing(ContinuousTrainingStore(tmp_path/'run', dict(retained_checkpoint_updates=[1]))) as store:
        for i in range(5):store.save_checkpoint(i, dict(policy={}, optimizer={'state':{}}, update=i))
        assert sorted(x.name for x in (store.root/'checkpoints').glob('*.pt')) == [
            'update-000.pt','update-001.pt','update-003.pt','update-004.pt']
    with pytest.raises(ValueError, match='retention'):
        ContinuousTrainingStore(tmp_path/'invalid', dict(retained_checkpoint_updates=[-1]))
