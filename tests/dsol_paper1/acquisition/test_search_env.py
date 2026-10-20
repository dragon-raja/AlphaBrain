"""Search accounting, no-lookahead, and optional RL adapter admission."""

import numpy as np
import pytest

from AlphaBrain.research.dsol.acquisition.discrete_search import search
from AlphaBrain.research.dsol.acquisition.search_env import (
    SearchConfig, SearchEnv, baseline_action, gymnasium_env, rollout_baseline,
    permute_observation, catalog_action,
)
from AlphaBrain.research.dsol.data.search_replay import CachedQueries, OutcomeTable, QueryResult


class CountingQueries(CachedQueries):
    def __init__(self, visual, scores):
        super().__init__(visual, scores)
        self.calls = []

    def query(self, candidate):
        self.calls.append(candidate)
        return super().query(candidate)


def make_env(scores=(4., 2., 3., 1.), budget=4):
    n = len(scores)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * .1
    source = CountingQueries(np.arange(n * 3).reshape(n, 3), np.repeat(np.array(scores)[:, None], 8, axis=1))
    return SearchEnv(poses, source, np.array([1., 2.]), 2.,
                     SearchConfig(query_budget=budget, neighbors=1))


def test_only_acquired_views_are_exposed_and_observations_are_copies():
    env = make_env()
    obs, info = env.reset(start=0)
    assert env.source.calls == [0]
    assert info["queries"] == 1 and info["scoring_members"] == 8
    assert not obs["visual"][1:].any() and not obs["scores"][1:].any()
    assert obs["action_mask"].tolist() == [1, 0, 1, 0, 0]
    obs["scores"][0] = -100
    assert env.values[0, 0] == 4
    other = make_env(scores=(4., 99., 7., 8.))
    other_obs, _ = other.reset(start=0)
    for key in obs:
        np.testing.assert_array_equal(env.observation()[key], other_obs[key])


def test_reward_telescope_and_return_cost():
    env = make_env(scores=(1., 3., 4., 5.))
    env.reset(start=0)
    rewards = []
    for action in (2, 3, 4):
        _, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        assert not truncated
    assert done and info["stop_reason"] == "budget"
    assert info["path"] == [0, 1, 2, 3, 2, 1, 0]
    assert info["selected"] == 0 and info["scoring_members"] == 32
    assert info["translation_m"] == pytest.approx(.6)
    assert sum(rewards) == pytest.approx(-3 * .01 - 6 * .001)
    assert env.source.calls == [0, 1, 2, 3]
    assert not env.action_masks().any()
    with pytest.raises(RuntimeError):
        env.step(0)


def test_improvement_and_stop_charge_no_new_query():
    env = make_env()
    env.reset(start=0)
    _, reward, _, _, _ = env.step(2)
    assert reward == pytest.approx((4 - 2) / 2 - .01 - .001)
    env.step(3)
    calls = list(env.source.calls)
    _, reward, done, _, info = env.step(0)
    assert done and reward == pytest.approx(-.001)
    assert env.source.calls == calls
    assert info["path"] == [0, 1, 2, 1]
    assert info["queries"] == 3


def test_ties_keep_start_and_immediate_stop_is_free():
    env = make_env(scores=(1., 1., 1., 1.))
    obs, _ = env.reset(start=1)
    _, reward, done, _, info = env.step(0)
    assert done and reward == 0 and info["selected"] == 1
    assert info["path"] == [1]
    env.reset(start=0)
    env.step(2)
    assert env.best == 0


@pytest.mark.parametrize("action", [-1, 1, 4, 5, .5, "stop"])
def test_invalid_actions_do_not_query_or_move(action):
    env = make_env()
    env.reset(start=0)
    before = env.info()
    with pytest.raises(ValueError):
        env.step(action)
    assert env.info() == before
    assert env.source.calls == [0]


def test_single_query_budget_and_reset():
    env = make_env(budget=1)
    _, info = env.reset(start=2)
    assert env.done and info["selected"] == 2 and info["queries"] == 1
    _, info = env.reset(start=1)
    assert info["visited"] == [1] and info["path"] == [1] and info["total_reward"] == 0


@pytest.mark.parametrize("method", ["local_accel", "random"])
def test_old_search_query_order_and_endpoint_are_preserved(method):
    env = make_env()
    env.reset(start=0)
    old = search(env.edges, env.distance, 0, [4],
                 lambda node: env.source.query(node).accel_mean,
                 random_seed=17 if method == "random" else None)[4]
    new = rollout_baseline(env, 0, method, seed=17)
    assert new["visited"] == old["visited"] and new["selected"] == old["selected"]


@pytest.mark.parametrize("method", ["stay", "scan", "random", "local_accel"])
def test_baselines_reproducible_and_same_query_contract(method):
    one, two = make_env(), make_env()
    a, b = rollout_baseline(one, 0, method, 10), rollout_baseline(two, 0, method, 10)
    assert a == b
    assert one.source.calls == a["visited"]
    assert a["queries"] == (1 if method == "stay" else 4)
    assert a["path"][-1] == a["selected"]


def test_missing_labels_remain_missing_and_pairing_is_explicit():
    table = OutcomeTable(np.array([[[[1, 0, np.nan, 1], [0, 1, 1, np.nan], [np.nan] * 4]]]))
    paired = table.paired(0, 0, 0, 1)
    assert paired["paired_repeats"] == 2 and paired["gain"] == 0
    assert paired["rescued_fraction"] == paired["harmed_fraction"] == .5
    empty = table.paired(0, 0, 0, 2)
    assert empty["paired_repeats"] == 0 and empty["selected"] is None


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_invalid_scale_rejected(scale):
    env = make_env()
    with pytest.raises(ValueError):
        SearchEnv(env.poses, env.source, [1, 2], scale, env.config)


def test_no_nonfinite_or_negative_scores_and_invalid_pose():
    with pytest.raises(ValueError):
        CachedQueries([[1]], [[np.nan]])
    with pytest.raises(ValueError):
        CachedQueries([[1]], [[-1]])
    env = make_env()
    env.poses[0, 0, 0] = 2
    with pytest.raises(ValueError):
        SearchEnv(env.poses, env.source, [1, 2], 1, env.config)


def test_failed_query_terminates_without_retry():
    env = make_env()
    env.reset(start=0)
    def failed(node):
        return QueryResult(np.zeros(3), np.nan, 0, 8)
    env.source.query = failed
    with pytest.raises(ValueError):
        env.step(2)
    assert env.done and env.reason == "query_error"
    with pytest.raises(RuntimeError):
        env.step(2)


def test_candidate_slots_permuted_without_exposing_hidden_observations():
    env = make_env()
    obs, _ = env.reset(start=0)
    order = np.array([3, 0, 2, 1])
    mapped = permute_observation(obs, order)
    assert mapped["action_mask"].tolist() == [1, 0, 0, 0, 1]
    np.testing.assert_array_equal(mapped["visual"][1], obs["visual"][0])
    assert not mapped["visual"][[0, 2, 3]].any()
    assert catalog_action(0, order) == 0 and catalog_action(4, order) == 2
    env.step(catalog_action(4, order))
    assert env.visited == [0, 1]
    with pytest.raises(ValueError):
        permute_observation(obs, [0, 0, 1, 2])


def test_optional_gym_adapter_only_when_already_installed():
    pytest.importorskip("gymnasium")
    core = make_env()
    adapter = gymnasium_env(core, [0, 1])
    assert core.source.calls == []  # Constructing the bridge must not query.
    obs, info = adapter.reset(seed=9)
    assert adapter.observation_space.contains(obs)
    while not core.done:
        original_action = baseline_action(core, "local_accel")
        action = int(np.flatnonzero(adapter.order == original_action - 1)[0]) + 1
        assert adapter.action_masks()[action]
        obs, _, _, _, _ = adapter.step(action)
        assert adapter.observation_space.contains(obs)
