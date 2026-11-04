from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.data.core_mechanism import mechanism_spec,reward_for
from AlphaBrain.research.dsol.data.online_reward import validate_config,assign_batch,empty_state
from AlphaBrain.research.dsol.data.reward_pilot import reward_config


def test_cost_sensitivity_requires_explicit_separate_contract():
    cfg=reward_config('G');cfg['query_cost']=.005
    with pytest.raises(ValueError):validate_config(cfg)
    raw=dict(policy_update=0,raw_reward=dict(schema='FORMAL_BATCH_REWARD_V1',initial_mean=1.,initial_std=.1,score_members=2,collector_scale=1.),transitions=[dict(best_previous=1.,best_current=.8,delta_queries=1,reward=(1.-.8)-.01)])
    original=deepcopy(raw)
    for name,cost in [('cost005',.005),('cost020',.02)]:
        cfg=reward_for(mechanism_spec(name));validate_config(cfg)
        out,_,receipt=assign_batch([raw],cfg,empty_state(),update=0)
        assert out[0]['transitions'][0]['reward']==pytest.approx(.2-cost)
        assert receipt['scale']==1 and receipt['additional_model_calls']==0
    assert raw==original
