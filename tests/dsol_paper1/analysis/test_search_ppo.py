"""Optional isolated-RL tests; no optimizer steps or private data required."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")

from AlphaBrain.research.dsol.acquisition.search_env import SearchConfig, SearchEnv, gymnasium_env
from AlphaBrain.research.dsol.analysis.search_training import assert_observation, compare_nested, run
from AlphaBrain.research.dsol.data.search_replay import CachedQueries
from AlphaBrain.research.dsol.selectors.search_ppo import build_model


def environment():
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, 0, 3] = np.arange(4) * .1
    core = SearchEnv(poses, CachedQueries(np.ones((4, 3)), np.ones((4, 8))),
                     [1., 2.], 1., SearchConfig(query_budget=4, neighbors=1))
    return gymnasium_env(core, [0, 1])


def test_actor_masks_critic_and_gradients(tmp_path):
    torch.set_num_threads(1)
    env = environment()
    model = build_model(env, 17, dict(n_steps=8, batch_size=8), tmp_path)
    obs, _ = env.reset(seed=3)
    assert_observation(obs)
    tensor, _ = model.policy.obs_to_tensor(obs)
    masks = env.action_masks()[None]
    distribution = model.policy.get_distribution(tensor, action_masks=masks)
    probabilities = distribution.distribution.probs
    assert torch.all(probabilities[~torch.tensor(masks)] == 0)
    assert torch.allclose(probabilities.sum(1), torch.ones(1))
    actions = distribution.get_actions()
    values, log_prob, _ = model.policy.evaluate_actions(tensor, actions, action_masks=masks)
    assert values.shape == (1, 1) and torch.isfinite(values).all()
    (values.square().mean() + values.mean() - log_prob.mean()).backward()
    assert model.policy.value_net.weight.grad.abs().sum() > 0
    assert model.policy.action_net.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.policy.parameters() if p.grad is not None)


def test_hidden_observation_guard():
    env = environment()
    obs, _ = env.reset(seed=3)
    unknown = np.flatnonzero(obs["flags"][:, 0] == 0)[0]
    obs["scores"][unknown, 0] = 1
    with pytest.raises(ValueError, match="leakage"):
        assert_observation(obs)


def test_import_and_no_execute_do_not_start_training(tmp_path):
    with pytest.raises(ValueError, match="execute"):
        run("absent-data", tmp_path / "new", "absent-config", execute=False)
    assert not (tmp_path / "new").exists()


def test_nested_optimizer_comparison():
    assert compare_nested({"a": [torch.ones(2), 3]}, {"a": [torch.ones(2), 3]})
    assert not compare_nested({"a": [torch.ones(2)]}, {"a": [torch.zeros(2)]})
