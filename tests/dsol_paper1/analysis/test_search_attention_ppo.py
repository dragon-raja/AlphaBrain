"""Toy-input integration only; never reads research data or invokes a VLA."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")

from stable_baselines3.common.vec_env import DummyVecEnv
from AlphaBrain.research.dsol.acquisition.search_budget import mixed_budget_env
from AlphaBrain.research.dsol.acquisition.search_env import SearchConfig, SearchEnv
from AlphaBrain.research.dsol.data.search_replay import CachedQueries
from AlphaBrain.research.dsol.selectors.search_attention_ppo import (
    AttentionPPO, build_attention_model,
)
from AlphaBrain.research.dsol.analysis.search_training import compare_nested


def toy_environment(privileged=True):
    count = 17
    poses = np.tile(np.eye(4), (count, 1, 1))
    poses[:, 0, 3] = np.arange(count) * .1
    scores = np.arange(1., count + 1)
    source = CachedQueries(np.ones((count, 4)), np.repeat(scores[:, None], 2, axis=1))
    core = SearchEnv(poses, source, np.ones(4), 1.,
                     SearchConfig(query_budget=4, neighbors=2, motion_cost=0.))
    return mixed_budget_env(core, range(count), full_accel=scores if privileged else None)


def test_budget_observation_and_no_privileged_gym_key():
    env = toy_environment()
    seen = set()
    for seed in range(30):
        obs, info = env.reset(seed=seed, options={"start": 16})
        budget = info["budget"]
        seen.add(budget)
        assert obs["global"][2] == budget / 16
        assert obs["global"][0] == 1 / 16
        assert "full_accel" not in obs
        assert obs["scores"][obs["flags"][:, 0] == 0].sum() == 0
        # Candidate permutation is reflected in both observed and privileged scores.
        visible = obs["flags"][:, 0].astype(bool)
        np.testing.assert_array_equal(env.critic_field()[visible], obs["scores"][visible, 0])
    assert seen == {4, 8, 16}
    deployed = toy_environment(False)
    deployed.reset(seed=1)
    with pytest.raises(RuntimeError, match="unavailable"):
        deployed.critic_field()


def test_toy_ppo_sidecar_alignment_mask_save_restore(tmp_path):
    torch.set_num_threads(1)
    env = DummyVecEnv([toy_environment, toy_environment])
    model = build_attention_model(env, 5, dict(n_steps=8, batch_size=8, n_epochs=1))
    actor_before = {k: v.clone() for k, v in model.policy.actor.state_dict().items()}
    critic_before = {k: v.clone() for k, v in model.policy.critic.state_dict().items()}
    # Synthetic 16 transitions validate the optimizer wiring, not scientific training.
    model.learn(total_timesteps=16)
    assert model.num_timesteps == 16
    assert not compare_nested(actor_before, model.policy.actor.state_dict())
    assert not compare_nested(critic_before, model.policy.critic.state_dict())
    buffer = model.rollout_buffer
    flags = buffer.observations["flags"][..., 0].astype(bool)
    scores = buffer.observations["scores"][..., 0]
    fields = buffer.swap_and_flatten(buffer.fields)
    np.testing.assert_array_equal(fields[flags], scores[flags])
    assert set(buffer.observations) == set(env.observation_space.spaces)
    assert "full_accel" not in buffer.observations
    mask = buffer.action_masks
    actions = buffer.actions.astype(int).reshape(-1)
    assert mask[np.arange(len(actions)), actions].all()
    path = tmp_path / "toy.zip"
    model.save(path)
    restored = AttentionPPO.load(path, device="cpu")
    assert compare_nested(model.policy.state_dict(), restored.policy.state_dict())
    assert compare_nested(model.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict())
    deployed = toy_environment(False)
    obs, _ = deployed.reset(seed=71)
    a, _ = model.predict(obs, deterministic=True, action_masks=deployed.action_masks())
    b, _ = restored.predict(obs, deterministic=True, action_masks=deployed.action_masks())
    assert int(a) == int(b)
    assert restored.policy.field_provider is None
    tensor, _ = restored.policy.obs_to_tensor(obs)
    with pytest.raises(RuntimeError, match="training"):
        restored.policy.predict_values(tensor)
    env.close()
