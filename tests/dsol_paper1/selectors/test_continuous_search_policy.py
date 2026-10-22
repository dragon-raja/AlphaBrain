import math
import numpy as np
import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('gymnasium')
from AlphaBrain.research.dsol.selectors.continuous_search_policy import BoundedStopNormal, ContinuousActorCritic
from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from AlphaBrain.research.dsol.acquisition.continuous_search_gym import continuous_observation
from tests.dsol_paper1.helpers.local_search import synthetic_session


def obs():
    old, source = synthetic_session()
    core = ContinuousSearch(old.mount, source, initial_angles=[0,0], physical_state_id='synthetic', wrist=np.ones(4), score_scale=1., seed=19)
    return {k:torch.as_tensor(v[None]) for k,v in continuous_observation(core.start()).items()}


def test_probability_matches_transformed_distribution_and_stop_excludes_motion():
    logit=torch.tensor([.3,.3], requires_grad=True)
    mean=torch.zeros(2,2,requires_grad=True)
    low=torch.tensor([[-10.,-10.],[-3.,0.]])
    high=torch.tensor([[10.,10.],[10.,10.]])
    d=BoundedStopNormal(logit,mean,torch.zeros(2,2),low,high)
    z=torch.tensor([[.2,-.4],[.1,.8]],dtype=torch.float64)
    transform=torch.distributions.TransformedDistribution(d.normal,[torch.distributions.TanhTransform(cache_size=1),torch.distributions.AffineTransform(d.center,d.half)])
    expected=torch.nn.functional.logsigmoid(-logit).double()+transform.log_prob(d.transform(z)).sum(-1)
    torch.testing.assert_close(d.log_prob(torch.tensor([False,False]),z),expected)
    stop=d.log_prob(torch.tensor([True,True]),z)
    torch.testing.assert_close(stop,torch.nn.functional.logsigmoid(logit).double())
    stop.sum().backward(); assert mean.grad is None or not mean.grad.any()
    for _ in range(50):
        a=d.sample(); go=~a['stop']
        assert ((a['delta_deg'][go]>=low[go]) & (a['delta_deg'][go]<=high[go])).all()
        torch.testing.assert_close(a['log_prob'],d.log_prob(a['stop'],a['latent']))


def test_action_density_integrates_with_branch_mass():
    # Integrate physical-degree density via change of variables, asymmetric bounds.
    n=20001; z=torch.linspace(-8,8,n,dtype=torch.float64)
    # 2-D factored integral at identical axes, explicit tanh and affine Jacobian.
    normal=torch.distributions.Normal(torch.tensor(0.,dtype=torch.float64),torch.tensor(1.,dtype=torch.float64))
    jac=5*(1-z.tanh().square()); density=(normal.log_prob(z)-jac.log()).exp()
    integral=torch.trapezoid(density*jac,z)
    assert abs(float(integral**2)-1)<1e-7
    p=.37; assert abs(p+(1-p)*float(integral**2)-1)<1e-7


def test_critic_independent_save_restore_and_ppo_gradients_without_update(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(0)
    p=ContinuousActorCritic(4,width=16);o=obs()
    assert not ({id(v) for v in p.actor_history.parameters()} & {id(v) for v in p.critic_history.parameters()})
    a=p.distribution(o).sample();value=p.value(o)
    before={k:v.clone() for k,v in p.state_dict().items()}
    loss=p.objective(o,a['stop'],a['latent'],a['log_prob'].detach(),torch.ones(1),torch.ones(1));loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in p.parameters() if v.grad is not None)
    for k,v in p.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
    f=tmp_path/'state.pt';torch.save(p.state_dict(),f);q=ContinuousActorCritic(4,width=16)
    q.load_state_dict(torch.load(f,weights_only=True),strict=True)
    torch.testing.assert_close(q.value(o),value,atol=0,rtol=0)
    torch.testing.assert_close(q.distribution(o).log_prob(a['stop'],a['latent']),a['log_prob'],atol=0,rtol=0)
    bad=dict(o,full_accel=torch.ones(1,97))
    with pytest.raises(ValueError):p.value(bad)
