import numpy as np
import pytest
pytest.importorskip('gymnasium')
from AlphaBrain.research.dsol.acquisition.local_view_gym import LocalSearchGym, local_baseline_action
from tests.dsol_paper1.helpers.local_search import synthetic_session


def test_dynamic_rows_pad_without_changing_action_indices_or_budget():
    core, source = synthetic_session()
    env = LocalSearchGym(4)
    obs, info = env.reset(options={'session': core})
    assert env.observation_space.contains(obs) and info['queries'] == 1
    assert not obs['visual'][1:].any() and not obs['scores'][1:].any()
    assert not env.action_masks()[10:].any()
    with pytest.raises(ValueError): env.step(15)
    assert source.calls == 1
    done = False
    while not done:
        raw = core.observation()
        assert np.array_equal(env.action_masks()[:len(raw['action_mask'])], raw['action_mask'])
        obs, reward, done, truncated, info = env.step(local_baseline_action(core))
        assert env.observation_space.contains(obs) and not truncated
    assert core.decision == 6 and info['queries'] == source.calls == 8
    assert not env.action_masks().any() and info['reason'] == 'budget'
    with pytest.raises(ValueError): env.step(0)


def test_reset_requires_explicit_new_episode_and_never_discards_failure():
    core, source = synthetic_session(); env = LocalSearchGym(4)
    with pytest.raises(ValueError): env.reset()
    env.reset(options={'session': core})
    fresh, _ = synthetic_session()
    with pytest.raises(ValueError): env.reset(options={'session': fresh})
    source.fail = True
    with pytest.raises(RuntimeError): env.step(local_baseline_action(core))
    with pytest.raises(ValueError): env.reset(options={'session': fresh})
    assert source.calls == 2 and core.status == 'FAILED'


def test_stop_cost_and_explicit_next_episode():
    core, source = synthetic_session(); env = LocalSearchGym(4)
    env.reset(options={'session': core})
    _, reward, done, truncated, info = env.step(0)
    assert reward == 0 and done and not truncated and source.calls == 1
    assert info['initial_query_cost'] == .01
    fresh, _ = synthetic_session()
    obs, _ = env.reset(options={'session': fresh})
    assert env.observation_space.contains(obs)


def test_baseline_tie_order_is_explicit():
    core, _ = synthetic_session(); core.start()
    core.proposals = np.array([[3., 0.], [-3., 0.], [4., 0.]])
    assert local_baseline_action(core) == 2  # STOP + one history row.
