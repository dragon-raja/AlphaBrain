"""Q16 retains acquired rows; padding changes no acquired information."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from AlphaBrain.research.dsol.acquisition.continuous_search_gym import continuous_observation
from AlphaBrain.research.dsol.acquisition.local_view_gym import padded_observation
from AlphaBrain.research.dsol.selectors.unified_search_policy import UnifiedActorCritic
from AlphaBrain.research.dsol.data.search_core import core_spec


def observation(n):
    rng=np.random.default_rng(2026)
    obs={k:rng.normal(size=s).astype(np.float32) for k,s in
         dict(geometry=(n,12),visual=(n,8),wrist=(8,),scores=(n,2),flags=(n,4),joint_state=(6,)).items()}
    obs['global']=rng.normal(size=6).astype(np.float32)
    obs['flags'][:,0]=1;obs['flags'][:,2]=0;obs['flags'][-1,2]=1
    obs.update(action_mask=np.ones(n+1,dtype=np.int8),delta_low=np.full(2,-5.),delta_high=np.full(2,5.))
    return obs


@pytest.mark.parametrize('variant',['U12','U3'])
def test_actor_and_critic_padding_equivalence(variant):
    torch.set_num_threads(1);torch.manual_seed(9)
    policy=UnifiedActorCritic(core_spec(variant,width=16),visual_dim=8).eval()
    torch.nn.init.normal_(policy.stop_head.weight);torch.nn.init.normal_(policy.mean_head.weight)
    obs=observation(7);small=continuous_observation(obs);large=continuous_observation(obs,query_budget=16)
    historical=padded_observation({k:v for k,v in obs.items() if k not in ('joint_state','delta_low','delta_high')},7)
    for k,v in historical.items():np.testing.assert_array_equal(small[k],v)
    convert=lambda x:{k:torch.as_tensor(v[None]) for k,v in x.items()}
    a,b=convert(small),convert(large)
    with torch.inference_mode():
        torch.testing.assert_close(policy.value(a),policy.value(b),rtol=1e-5,atol=1e-6)
        da,db=policy.distribution(a),policy.distribution(b)
        torch.testing.assert_close(da.stop.logits,db.stop.logits,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(da.normal.loc,db.normal.loc,rtol=1e-5,atol=1e-6)


def test_long_history_preserves_every_row_and_old_cap_rejects_overflow():
    obs=observation(15);x=continuous_observation(obs,query_budget=16)
    for k,v in obs.items():np.testing.assert_array_equal(x[k],v)
    with pytest.raises(ValueError):continuous_observation(obs)
