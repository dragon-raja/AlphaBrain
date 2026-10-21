"""Synthetic full-runner test: no private data, VLA or simulator."""

import json
import numpy as np
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('gymnasium')
pytest.importorskip('sb3_contrib')

from AlphaBrain.research.dsol.data.search_replay import ReplayInputs
from AlphaBrain.research.dsol.analysis.search_attention_training import train_seed


def test_synthetic_runner_logs_budget_and_exports_actor(tmp_path):
    torch.set_num_threads(1)
    poses=np.tile(np.eye(4),(32,97,1,1));poses[:,:,0,3]=np.arange(97)*.1
    scores=np.broadcast_to(np.linspace(.01,.1,97)[None,None,:,None],(1,32,97,2)).copy()
    data=ReplayInputs(tuple(str(i) for i in range(97)),tuple({} for _ in range(32)),poses,
        np.ones((32,97,4)),np.ones((32,4)),scores,tuple(range(16)),tuple(range(16,32)),(.05,),{})
    plan={'training':dict(n_steps=4,batch_size=16,n_epochs=1,learning_rate=.0003,gamma=1.,
        gae_lambda=.95,clip_range=.2,ent_coef=.01,vf_coef=.5,max_grad_norm=.5)}
    output=tmp_path/'synthetic'
    receipt=train_seed(data,plan,output,0,128)
    assert receipt['steps']==receipt['checked_actions']==128
    assert receipt['budgets']==[4,8,16]
    updates=[json.loads(line) for line in (output/'updates.jsonl').read_text().splitlines()]
    assert [r['steps'] for r in updates]==[64,128]
    assert (output/'actor.pt').exists()
    assert len(json.loads((output/'deployment-controls.json').read_text())['controls'])==8
    assert receipt['checkpoint_admission']['no_deployment_privileged_provider']
