from copy import deepcopy
import pytest

torch=pytest.importorskip('torch')
from AlphaBrain.research.dsol.selectors.mechanism_policy import MechanismActorCritic,FixedContinueNormal,mask_score_feedback
from AlphaBrain.research.dsol.data.core_mechanism import mechanism_spec
from AlphaBrain.research.dsol.data.search_core import core_spec


def observation():
    torch.manual_seed(7)
    x=dict(geometry=torch.randn(3,4,12),visual=torch.randn(3,4,8),wrist=torch.randn(3,8),
        scores=torch.randn(3,4,2),flags=torch.ones(3,4,4),joint_state=torch.randn(3,6),
        delta_low=torch.full((3,2),-5.),delta_high=torch.full((3,2),5.),action_mask=torch.ones(3,5,dtype=torch.bool))
    x['global']=torch.randn(3,6)
    return x


def test_both_branches_cannot_see_explicit_scores_and_input_unchanged():
    torch.set_num_threads(1)
    p=MechanismActorCritic(core_spec('U12',width=16),mechanism=mechanism_spec('no_score'),visual_dim=8).eval()
    torch.nn.init.normal_(p.mean_head.weight);torch.nn.init.normal_(p.stop_head.weight)
    obs=observation();old=deepcopy(obs);other=deepcopy(obs)
    other['scores']*=100;other['flags'][...,3]=0;other['global'][:,3:5]+=999
    torch.testing.assert_close(p.value(obs),p.value(other),rtol=0,atol=0)
    a,b=p.distribution(obs),p.distribution(other)
    torch.testing.assert_close(a.stop.logits,b.stop.logits,rtol=0,atol=0)
    torch.testing.assert_close(a.normal.loc,b.normal.loc,rtol=0,atol=0)
    for k in obs:assert torch.equal(obs[k],old[k])
    masked=mask_score_feedback(obs)
    assert torch.equal(masked['flags'][...,:3],obs['flags'][...,:3])
    assert torch.equal(masked['global'][:,:3],obs['global'][:,:3])
    assert torch.equal(masked['global'][:,5],obs['global'][:,5])


def test_fixed_query_exact_motion_density_gradient_and_sampling():
    p=MechanismActorCritic(core_spec('U12',width=16),mechanism=mechanism_spec('fixed_query'),visual_dim=8)
    obs=observation();d=p.distribution(obs);sample=d.sample()
    assert isinstance(d,FixedContinueNormal) and not sample['stop'].any()
    assert not d.sample(deterministic=True)['stop'].any()
    for _ in range(50):assert not d.sample()['stop'].any()
    z=sample['latent'];x=sample['delta_deg'];unit=x/d.half
    expected=(d.normal.log_prob(z)-d.half.log()-(1-unit.square()).log()).sum(-1)
    torch.testing.assert_close(sample['log_prob'],expected)
    loss=p.objective(obs,sample['stop'],z,sample['log_prob'].detach(),torch.tensor([1.,0.,-1.]),torch.zeros(3))
    loss.backward()
    assert p.stop_head.weight.grad is None
    assert p.mean_head.weight.grad.abs().sum()>0 and torch.isfinite(loss)
    assert not d.exploration_terms()[1].any()
    with pytest.raises(ValueError):d.log_prob(torch.ones(3,dtype=torch.bool),z)


def test_same_shape_wrong_mechanism_fails_before_loading():
    a=MechanismActorCritic(core_spec('U12',width=16),mechanism=mechanism_spec('no_score'),visual_dim=8)
    b=MechanismActorCritic(core_spec('U12',width=16),mechanism=mechanism_spec('fixed_query'),visual_dim=8)
    old=deepcopy(b.state_dict())
    with pytest.raises(ValueError):b.load_state_dict(a.state_dict())
    for k,v in b.state_dict().items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,old[k])
