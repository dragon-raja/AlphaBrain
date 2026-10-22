import json
from contextlib import closing
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from AlphaBrain.research.dsol.selectors.continuous_export import export_final
from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
from AlphaBrain.research.dsol.data.continuous_training_store import ContinuousTrainingStore,write_json
from AlphaBrain.research.dsol.acquisition.continuous_search_gym import continuous_observation
from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from tests.dsol_paper1.helpers.local_search import synthetic_session


def test_final_export_checks_complete_coverage_and_preserves_weights(tmp_path):
    torch.set_num_threads(1);policy=ContinuousActorCritic(4)
    old,source=synthetic_session()
    session=ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',
        wrist=np.ones(4),score_scale=1.,seed=0)
    session.start();obs=continuous_observation(session.observation())
    root=tmp_path/'training'
    with closing(ContinuousTrainingStore(root,{'source':'synthetic'})) as s:
        s.save_episode(0,dict(transitions=[dict(observation=obs)]),dict(queries=1))
        s.save_checkpoint(1,dict(policy=policy.state_dict(),update=1,cursor=1,queries=1))
    write_json(root/'status.json',dict(status='COMPLETE',episodes=1,updates=1,queries=1))
    receipt=export_final(root,tmp_path/'export',visual_dim=4,episodes=1,updates=1)
    payload=torch.load(tmp_path/'export/actor.pt',weights_only=True)
    assert all(torch.equal(v,payload['state_dict'][k]) for k,v in policy.state_dict().items())
    assert receipt['model_calls']==0
    with pytest.raises(ValueError):export_final(root,tmp_path/'again')
    with pytest.raises(ValueError):export_final(root,tmp_path/'export',visual_dim=4,episodes=1,updates=1)
    (root/'episodes/0000.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):export_final(root,tmp_path/'changed',visual_dim=4,episodes=1,updates=1)
    assert not (tmp_path/'changed').exists()
