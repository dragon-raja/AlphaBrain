"""Hybrid Bernoulli STOP + squashed Gaussian increments with exact Jacobian.

No training side effects; independent history critic, no candidate catalog.
Log densities are in degrees, and include the state-dependent affine scale.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .search_attention import SearchHistoryEncoder

BASE = {'geometry', 'visual', 'scores', 'flags', 'wrist', 'global', 'action_mask'}


class ContinuousHistory(nn.Module):
    def __init__(self, visual_dim, width=128, heads=4, layers=2):
        super().__init__()
        self.encoder = SearchHistoryEncoder(visual_dim, width, heads, layers)
        self.joints = nn.Sequential(nn.Linear(6, width), nn.Tanh())

    def forward(self, obs):
        if set(obs) != BASE | {'joint_state', 'delta_low', 'delta_high'}:
            raise ValueError('History and joint limits only; no privileged fields')
        if (obs['joint_state'].shape != (len(obs['global']), 6)
                or not torch.isfinite(obs['joint_state']).all()):
            raise ValueError('Finite normalized joint state required')
        _, history, _ = self.encoder({k: obs[k] for k in BASE})
        return history + self.joints(obs['joint_state'])


class BoundedStopNormal:
    def __init__(self, stop_logit, mean, log_std, low, high):
        if (mean.shape != low.shape or mean.shape != high.shape or mean.shape != log_std.shape
                or mean.shape != (*stop_logit.shape, 2)
                or any(not torch.isfinite(x).all() for x in (stop_logit, mean, log_std, low, high))
                or not (high > low).all()):
            raise ValueError('Finite aligned nonempty continuous action intervals required')
        self.stop = torch.distributions.Bernoulli(logits=stop_logit)
        self.normal = torch.distributions.Normal(mean.double(), log_std.double().exp())
        self.center, self.half = (high.double()+low.double())/2, (high.double()-low.double())/2

    def transform(self, latent):
        unit = torch.tanh(latent.double())
        if (unit.abs() >= 1).any():
            raise ValueError('Numerically saturated sample; fail rather than clip its probability')
        return self.center + self.half * unit

    def log_prob(self, stop, latent):
        if stop.dtype != torch.bool or stop.shape != self.stop.logits.shape or latent.shape != self.center.shape:
            raise ValueError('Stored branch and pre-transform latent required')
        if not torch.isfinite(latent).all(): raise ValueError('Finite stored latent required')
        branch = self.stop.log_prob(stop.to(self.stop.logits.dtype)).double()
        # STOP has no continuous density and gives zero gradient to motion parameters.
        motion = torch.zeros_like(branch)
        go = ~stop
        if go.any():
            z = latent[go].double()
            log_jac = self.half[go].log() + 2 * (math.log(2.) - z - F.softplus(-2*z))
            normal = torch.distributions.Normal(self.normal.loc[go], self.normal.scale[go])
            motion[go] = (normal.log_prob(z) - log_jac).sum(-1)
        return branch + motion

    def sample(self, deterministic=False):
        stop = self.stop.probs >= .5 if deterministic else self.stop.sample().bool()
        latent = torch.zeros_like(self.center)
        go = ~stop
        if go.any():
            normal = torch.distributions.Normal(self.normal.loc[go], self.normal.scale[go])
            latent[go] = normal.loc if deterministic else normal.sample()
        delta = torch.zeros_like(latent)
        if go.any():
            unit = latent[go].tanh()
            if (unit.abs() >= 1).any(): raise ValueError('Saturated Gaussian action; stop')
            delta[go] = self.center[go] + self.half[go] * unit
        return dict(stop=stop, latent=latent, delta_deg=delta, log_prob=self.log_prob(stop, latent))

    def entropy(self):
        # One reparameterized MC draw for H(transformed Normal), weighted by P(go).
        z = self.normal.rsample()
        jac = self.half.log() + 2*(math.log(2.) - z - F.softplus(-2*z))
        conditional = self.normal.entropy().sum(-1) + jac.sum(-1)
        return self.stop.entropy() + (1-self.stop.probs) * conditional


class ContinuousActorCritic(nn.Module):
    def __init__(self, visual_dim, width=128, heads=4, layers=2):
        super().__init__()
        self.actor_history = ContinuousHistory(visual_dim, width, heads, layers)
        self.critic_history = ContinuousHistory(visual_dim, width, heads, layers)
        self.stop_head = nn.Linear(width, 1)
        self.mean_head = nn.Linear(width, 2)
        self.log_std = nn.Parameter(torch.full((2,), -.5))
        self.value_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.stop_head.weight); nn.init.zeros_(self.stop_head.bias)
        nn.init.zeros_(self.mean_head.weight); nn.init.zeros_(self.mean_head.bias)

    def load_legacy_encoder(self, actor_state):
        encoder = {k.removeprefix('encoder.'): v for k,v in actor_state.items() if k.startswith('encoder.')}
        self.actor_history.encoder.load_state_dict(encoder, strict=True)

    def distribution(self, obs):
        history = self.actor_history(obs)
        return BoundedStopNormal(self.stop_head(history).squeeze(-1), self.mean_head(history),
            self.log_std.clamp(-5, 1).expand(len(history), -1), obs['delta_low'], obs['delta_high'])

    def value(self, obs):
        return self.value_head(self.critic_history(obs)).squeeze(-1)

    def objective(self, obs, stop, latent, old_log_prob, advantages, returns, *, with_metrics=False):
        distribution = self.distribution(obs)
        log_prob = distribution.log_prob(stop, latent)
        value = self.value(obs)
        if any(x.shape != value.shape or not torch.isfinite(x).all() for x in (old_log_prob, advantages, returns)):
            raise ValueError('Finite aligned PPO targets required')
        adv = advantages.detach()
        if len(adv)>1: adv = (adv-adv.mean())/(adv.std()+1e-8)
        ratio = (log_prob-old_log_prob.detach()).exp()
        policy = -torch.minimum(adv*ratio, adv*ratio.clamp(.8, 1.2)).mean()
        value_loss = (value-returns.detach()).square().mean()
        entropy = distribution.entropy().mean()
        loss = policy + .5*value_loss - .01*entropy
        if not torch.isfinite(loss): raise ValueError('Nonfinite PPO objective')
        if with_metrics:
            log_ratio = log_prob-old_log_prob.detach()
            return loss, dict(policy_loss=float(policy.detach()), value_loss=float(value_loss.detach()),
                entropy=float(entropy.detach()), approximate_kl=float(((ratio-1)-log_ratio).mean().detach()),
                clip_fraction=float(((ratio-1).abs()>.2).double().mean()))
        return loss
