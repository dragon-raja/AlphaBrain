"""Architecture isolation tests; no training or policy/simulator invocation."""

import pytest

torch = pytest.importorskip("torch")

from AlphaBrain.research.dsol.selectors.search_attention import (
    PrivilegedSearchCritic, SharedSearchActor,
)


def observation(count=5):
    torch.manual_seed(19)
    flags = torch.zeros(2, count, 4)
    flags[:, :2, 0] = 1
    flags[:, :2, 1] = torch.tensor([1 / 16, 2 / 16])
    flags[:, 1, 2:] = 1
    masks = torch.ones(2, count + 1, dtype=torch.bool)
    masks[:, 1:3] = False
    return dict(
        geometry=torch.randn(2, count, 12), visual=torch.randn(2, count, 8),
        scores=torch.rand(2, count, 2), flags=flags, wrist=torch.randn(2, 8),
        **{"global": torch.rand(2, 6)}, action_mask=masks)


def test_candidate_permutation_and_variable_count():
    torch.set_num_threads(1)
    actor = SharedSearchActor(8, width=16, heads=4).eval()
    critic = PrivilegedSearchCritic(8, width=16, heads=4).eval()
    obs = observation()
    scores = torch.rand(2, 5)
    order = torch.tensor([3, 1, 4, 0, 2])
    shuffled = {k: v.clone() for k, v in obs.items()}
    for key in ("geometry", "visual", "scores", "flags"):
        shuffled[key] = obs[key][:, order]
    shuffled["action_mask"][:, 1:] = obs["action_mask"][:, 1:][:, order]
    actual, original = actor(shuffled), actor(obs)
    torch.testing.assert_close(actual[:, :1], original[:, :1])
    torch.testing.assert_close(actual[:, 1:], original[:, 1:][:, order])
    torch.testing.assert_close(critic(shuffled, scores[:, order]), critic(obs, scores))
    assert actor(observation(9)).shape == (2, 10)


def test_unseen_features_and_privileged_critic_cannot_change_actor():
    actor = SharedSearchActor(8, width=16).eval()
    critic = PrivilegedSearchCritic(8, width=16).eval()
    obs = observation()
    before = actor(obs).detach()
    obs["visual"][:, 2:] = 1e5
    obs["scores"][:, 2:] = 1e5
    torch.testing.assert_close(actor(obs), before)
    full = torch.ones(2, 5)
    v1 = critic(obs, full)
    v2 = critic(obs, full * 100)
    assert not torch.allclose(v1, v2)
    v1.sum().backward()
    assert all(p.grad is None for p in actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters())
    torch.testing.assert_close(actor(obs), before)
    with pytest.raises(ValueError, match="privileged"):
        actor(dict(obs, full_accel=full))


def test_mask_probabilities_and_actor_gradient():
    actor = SharedSearchActor(8, width=16)
    obs = observation()
    distribution = torch.distributions.Categorical(logits=actor(obs))
    assert (distribution.probs[~obs["action_mask"]] == 0).all()
    loss = -distribution.log_prob(torch.zeros(2, dtype=torch.long)).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)


def test_reject_empty_history_and_invalid_actions():
    actor = SharedSearchActor(8, width=16)
    obs = observation()
    obs["flags"].zero_()
    with pytest.raises(ValueError, match="initial"):
        actor(obs)
    obs = observation()
    obs["action_mask"][:, 1] = True
    with pytest.raises(ValueError, match="reacquire"):
        actor(obs)


def test_save_restore_logits(tmp_path):
    actor = SharedSearchActor(8, width=16).eval()
    torch.save(actor.state_dict(), tmp_path / "weights.pt")
    restored = SharedSearchActor(8, width=16).eval()
    restored.load_state_dict(torch.load(tmp_path / "weights.pt", weights_only=True))
    torch.testing.assert_close(actor(observation()), restored(observation()), rtol=0, atol=0)
