"""Explicit score-feedback / STOP ablations over the shared actor and critic.

Rewards and final selection still use Accel. This is not an Accel-free method.
Fixed-query sampling has no Bernoulli branch in its probability or regularizer.
"""
import math
import torch
from torch.nn import functional as F
from .unified_search_policy import UnifiedActorCritic
from .continuous_search_policy import BoundedStopNormal
from ..data.core_mechanism import mechanism_spec
from ..execution.manifest import digest


def mask_score_feedback(obs):
    out=dict(obs)
    out['scores']=torch.zeros_like(obs['scores'])
    out['flags']=obs['flags'].clone();out['flags'][...,3]=0
    out['global']=obs['global'].clone();out['global'][...,3:5]=0
    return out


class FixedContinueNormal(BoundedStopNormal):
    def __init__(self, *args):
        super().__init__(*args)
        self.stop=torch.distributions.Bernoulli(probs=torch.zeros_like(self.stop.probs))

    def log_prob(self, stop, latent):
        if (stop.dtype!=torch.bool or stop.shape!=self.stop.probs.shape or stop.any()
                or latent.shape!=self.center.shape or not torch.isfinite(latent).all()):
            raise ValueError('Fixed-query actions must be finite motion samples')
        z=latent.double()
        jac=self.half.log()+2*(math.log(2.)-z-F.softplus(-2*z))
        return (self.normal.log_prob(z)-jac).sum(-1)

    def entropy(self):
        z=self.normal.rsample()
        jac=self.half.log()+2*(math.log(2.)-z-F.softplus(-2*z))
        return self.normal.entropy().sum(-1)+jac.sum(-1)

    def exploration_terms(self):
        z=self.normal.rsample()
        motion=(self.normal.entropy()+2*(math.log(2.)-z-F.softplus(-2*z))).sum(-1)
        return motion,torch.zeros_like(motion),motion


class MechanismActorCritic(UnifiedActorCritic):
    def __init__(self, spec, *, mechanism, **kwargs):
        if mechanism!=mechanism_spec(mechanism['name']):raise ValueError('Invalid mechanism identity')
        super().__init__(spec,**kwargs)
        self.mechanism=dict(mechanism)
        self.identity['mechanism']=dict(mechanism)
        self.search_core_sha256=digest(self.identity)

    def distribution(self, obs):
        x=obs if self.mechanism['explicit_score'] else mask_score_feedback(obs)
        if self.mechanism['learned_stop']:return super().distribution(x)
        h=self.actor_history(x)
        return FixedContinueNormal(self.stop_head(h).squeeze(-1),self.mean_head(h),
            self.log_std.clamp(-5,1).expand(len(h),-1),x['delta_low'],x['delta_high'])

    def value(self, obs):
        return super().value(obs if self.mechanism['explicit_score'] else mask_score_feedback(obs))
