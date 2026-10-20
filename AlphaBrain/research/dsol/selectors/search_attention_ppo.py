"""SB3 MaskablePPO adapter with a critic sidecar, never an actor observation.

The base PPO objective, GAE, minibatching and rollout timing are unchanged.
Privileged fields follow the exact pre-action candidate permutation into the
buffer. No reward labels, policy download or training starts during import.
"""

import numpy as np
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

from .search_attention import SharedSearchActor, PrivilegedSearchCritic


class AttentionPolicy(MaskableMultiInputActorCriticPolicy):
    def _build(self, lr_schedule):
        visual_dim = self.observation_space["visual"].shape[-1]
        self.actor = SharedSearchActor(visual_dim)
        self.critic = PrivilegedSearchCritic(visual_dim)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.field_provider = None
        self.rollout_field = None
        self.minibatch_field = None

    def _field(self):
        if self.field_provider is None:
            raise RuntimeError("Critic requires explicit training environment fields")
        return torch.as_tensor(self.field_provider(), dtype=torch.float32, device=self.device)

    def get_distribution(self, obs, action_masks=None):
        distribution = MaskableCategoricalDistribution(self.action_space.n)
        distribution.proba_distribution(action_logits=self.actor(obs))
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def forward(self, obs, deterministic=False, action_masks=None):
        self.rollout_field = self._field()
        distribution = self.get_distribution(obs, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.critic(obs, self.rollout_field)[:, None], distribution.log_prob(actions)

    def predict_values(self, obs):
        return self.critic(obs, self._field())[:, None]

    def evaluate_actions(self, obs, actions, action_masks=None):
        if self.minibatch_field is None or len(self.minibatch_field) != len(actions):
            raise RuntimeError("Missing aligned critic minibatch")
        distribution = self.get_distribution(obs, action_masks)
        return (self.critic(obs, self.minibatch_field)[:, None],
                distribution.log_prob(actions), distribution.entropy())


class PrivilegedBuffer(MaskableDictRolloutBuffer):
    """Keep full fields outside observations and flatten using the SB3 order."""

    def __init__(self, *args, **kwargs):
        self.policy = None
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.fields = np.zeros((self.buffer_size, self.n_envs, self.action_space.n - 1), np.float32)

    def add(self, *args, **kwargs):
        if self.policy is None or self.policy.rollout_field is None:
            raise RuntimeError("Policy did not provide a pre-action critic field")
        field = self.policy.rollout_field.detach().cpu().numpy()
        if field.shape != self.fields[self.pos].shape or not np.isfinite(field).all():
            raise ValueError("Critic rollout field shape/value mismatch")
        self.fields[self.pos] = field
        super().add(*args, **kwargs)

    def _get_samples(self, batch_inds, env=None):
        samples = super()._get_samples(batch_inds, env)
        self.policy.minibatch_field = self.to_torch(self.swap_and_flatten(self.fields)[batch_inds])
        return samples


class AttentionPPO(MaskablePPO):
    def _setup_model(self):
        super()._setup_model()
        if not isinstance(self.rollout_buffer, PrivilegedBuffer):
            raise TypeError("Attention PPO requires the separate privileged buffer")
        self.rollout_buffer.policy = self.policy
        self._attach_field_provider()

    def _attach_field_provider(self):
        self.policy.field_provider = (
            (lambda: np.stack(self.env.env_method("critic_field"))) if self.env is not None else None)

    def set_env(self, env, force_reset=True):
        super().set_env(env, force_reset=force_reset)
        self._attach_field_provider()

    def train(self):
        super().train()
        for parameter in self.policy.parameters():
            if not torch.isfinite(parameter).all():
                raise ValueError("Nonfinite search parameter; stop")
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise ValueError("Nonfinite search gradient; stop")
        self.policy.minibatch_field = None


def build_attention_model(env, seed, ppo_config, *, device="cpu"):
    return AttentionPPO(AttentionPolicy, env, seed=seed, device=device,
                        rollout_buffer_class=PrivilegedBuffer, verbose=0, **ppo_config)
