"""Synthetic CPU backend; no renderer, VLA inference or training."""
import numpy as np
import pytest

from AlphaBrain.research.dsol.acquisition.local_view_search import (
    LocalViewConfig, LocalViewSearch, PanTiltMount, PoseReceipt, ViewMeasurement,
)
from AlphaBrain.research.dsol.data.search_replay import QueryResult


class SyntheticSource:
    def __init__(self, mount, start=(0., 0.)):
        self.pose = mount.pose(start)
        self.initial = self.pose.copy()
        self.moves, self.calls = [], 0
        self.bad_state = self.bad_return = self.fail_query = False

    def receipt(self):
        return PoseReceipt(self.pose.copy(), 'changed' if self.bad_state else 'fixed')

    def move(self, pose):
        self.pose = pose.copy()
        self.moves.append(pose.copy())
        return self.receipt()

    def query(self):
        self.calls += 1
        if self.fail_query:
            raise RuntimeError('synthetic query failure')
        score = 1. if np.array_equal(self.pose, self.initial) else 2.
        if self.bad_return and self.calls > 1 and score == 1.:
            score += .1
        return ViewMeasurement(self.receipt(), QueryResult(np.ones(4), score, 0., 2))


def session(budget=8, start=(0., 0.)):
    mount = PanTiltMount(np.eye(4), LocalViewConfig(query_budget=budget))
    source = SyntheticSource(mount, start)
    env = LocalViewSearch(mount, source, initial_angles=start, physical_state_id='fixed',
                          wrist=np.ones(4), score_scale=1., seed=17)
    return env, source


def next_action(env):
    return int(np.flatnonzero(env.observation()['action_mask'][1:])[0]) + 1


def test_continuous_pose_conventions_and_fixed_centre():
    mount = PanTiltMount(np.eye(4))
    yaw = mount.pose([5.123, 0])
    pitch = mount.pose([0, 3.456])
    assert yaw[0, 2] > 0 and pitch[1, 2] < 0
    for p in [yaw, pitch, mount.pose([17.38, -9.73])]:
        np.testing.assert_array_equal(p[:3, 3], [0, 0, 0])
        np.testing.assert_allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-14)
        assert np.linalg.det(p[:3, :3]) == pytest.approx(1)


@pytest.mark.parametrize('value', [[31, 0], [0, 21], [np.nan, 0], [0], [0, np.inf]])
def test_reject_invalid_angles(value):
    with pytest.raises(ValueError):
        PanTiltMount(np.eye(4)).pose(value)


@pytest.mark.parametrize('angles', [[0, 0], [30, 20], [-30, -20], [29.9, -19.9]])
def test_proposals_are_seeded_bounded_and_not_clipped(angles):
    mount = PanTiltMount(np.eye(4))
    p = mount.propose(angles, [angles], seed=42, decision=0)
    np.testing.assert_array_equal(p, mount.propose(angles, [angles], seed=42, decision=0))
    other = mount.propose(angles, [angles], seed=43, decision=0)
    assert p.shape != other.shape or not np.array_equal(p, other)
    for target in p:
        mount.angles(target)
        length = np.linalg.norm(target - angles)
        assert 2 <= length <= 10
    assert len(np.unique(p, axis=0)) == len(p)


def test_return_route_respects_step_bounds_and_exact_endpoint():
    mount = PanTiltMount(np.eye(4))
    start, target = np.array([-29.9, -19.9]), np.array([29.8, 19.8])
    route = mount.route(start, target)
    assert np.max(np.abs(np.diff(np.vstack([start, route]), axis=0))) <= 10
    np.testing.assert_array_equal(route[-1], target)
    assert mount.route(target, target) == []


def test_stop_at_initial_costs_one_and_does_not_move():
    env, source = session()
    env.start()
    _, reward, done, _, info = env.step(0)
    assert done and reward == 0 and info['queries'] == 1 and source.calls == 1
    assert source.moves == [] and info['initial_query_cost'] == .01
    with pytest.raises(ValueError):
        env.start()


def test_return_query_is_charged_and_history_not_fabricated():
    env, source = session()
    env.start()
    env.step(next_action(env))
    assert len(env.records) == 2
    _, reward, done, _, info = env.step(0)
    assert done and reward == pytest.approx(-.01)
    assert info['queries'] == source.calls == 3 and info['visited'] == 2
    np.testing.assert_array_equal(source.pose, source.initial)
    assert env.events[-1]['kind'] == 'return_verification'


def test_budget_reserves_return_slot():
    env, source = session(budget=3)
    env.start()
    _, reward, done, _, info = env.step(next_action(env))
    assert done and info['reason'] == 'budget' and info['queries'] == 3
    assert reward == pytest.approx(-.02)
    with pytest.raises(ValueError):
        env.step(0)
    assert source.calls == 3


def test_best_current_does_not_add_unnecessary_return_query():
    env, source = session(budget=3)
    original = source.query
    def improving():
        out = original()
        return ViewMeasurement(out.receipt, QueryResult(out.value.visual, 1 / source.calls, 0., 2))
    source.query = improving
    env.start()
    _, reward, done, _, info = env.step(next_action(env))
    assert done and info['queries'] == 2 and info['selected_angles'] == info['current_angles']
    assert reward == pytest.approx(.49)


@pytest.mark.parametrize('failure', ['bad_state', 'bad_return', 'fail_query'])
def test_backend_or_return_failure_stops_without_retry(failure):
    env, source = session()
    env.start()
    env.step(next_action(env))
    setattr(source, failure, True)
    with pytest.raises((ValueError, RuntimeError)):
        env.step(0)
    assert env.status == 'FAILED'
    calls = source.calls
    with pytest.raises(ValueError):
        env.step(0)
    assert source.calls == calls


def test_failed_initial_query_is_charged_and_not_restartable():
    env, source = session()
    source.fail_query = True
    with pytest.raises(RuntimeError):
        env.start()
    assert env.queries == 1 and env.status == 'FAILED'
    with pytest.raises(ValueError):
        env.start()


@pytest.mark.parametrize('case', ['unsettled', 'wrong_pose', 'nan_score', 'wrong_noise_count'])
def test_initial_measurement_rejects_invalid_receipt_and_values(case):
    env, source = session()
    original = source.query
    def invalid():
        out = original()
        pose = out.receipt.camera_to_world.copy()
        if case == 'wrong_pose':
            pose[0, 3] += .001
        return ViewMeasurement(PoseReceipt(pose, 'fixed', case != 'unsettled'),
            QueryResult(out.value.visual, float('nan') if case == 'nan_score' else 1.,
                        0., 1 if case == 'wrong_noise_count' else 2))
    source.query = invalid
    with pytest.raises(ValueError):
        env.start()
    assert env.status == 'FAILED' and env.queries == 1


def test_two_query_budget_masks_search_and_invalid_action_does_not_move():
    env, source = session(budget=2)
    env.start()
    assert env.observation()['action_mask'].sum() == 1
    with pytest.raises(ValueError):
        env.step(2)
    assert env.queries == 1 and not source.moves
    env.step(0)
    assert env.status == 'COMPLETE'


def test_dynamic_rows_keep_history_and_hide_unqueried_information():
    env, _ = session()
    env.start()
    env.step(next_action(env))
    obs = env.observation()
    assert (obs['flags'][:2, 0] == 1).all()
    assert not obs['visual'][2:].any() and not obs['scores'][2:].any()
    assert not obs['action_mask'][1:3].any()
    assert obs['action_mask'][3:].all()
    assert set(obs) == {'geometry', 'visual', 'scores', 'flags', 'wrist', 'global', 'action_mask'}


def test_old_shared_actor_weights_accept_dynamic_observation():
    torch = pytest.importorskip('torch')
    from AlphaBrain.research.dsol.selectors.search_attention import SharedSearchActor
    torch.set_num_threads(1)
    env, _ = session()
    env.start()
    actor = SharedSearchActor(4, width=16).eval()
    def logits():
        obs = {k: torch.as_tensor(v[None]) for k, v in env.observation().items()}
        with torch.inference_mode():
            return actor(obs)
    first = logits()
    env.step(next_action(env))
    second = logits()
    assert first.shape[1] == 10 and second.shape[1] >= 4
    assert torch.isneginf(second[0, 1:3]).all()
    assert torch.isfinite(second[0, 0])
