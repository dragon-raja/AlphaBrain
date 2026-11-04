import json
from contextlib import closing
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from AlphaBrain.research.dsol.selectors.continuous_export import export_final, export_stage
from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
from AlphaBrain.research.dsol.data.continuous_training_store import ContinuousTrainingStore,write_json
from AlphaBrain.research.dsol.acquisition.continuous_search_gym import continuous_observation
from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from tests.dsol_paper1.helpers.local_search import synthetic_session


def test_final_export_checks_complete_coverage_and_preserves_weights(tmp_path):
    torch.set_num_threads(1);policy=ContinuousActorCritic(4)
    with torch.no_grad():
        policy.mean_head.weight.normal_(std=.05);policy.stop_head.bias.fill_(-2.)
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
    assert receipt['control_cpu_threads']==2 and torch.get_num_threads()==1
    from benchmarks.LIBERO.continuous_evaluation import verify_actor
    deployed=ContinuousActorCritic(4).eval();deployed.load_state_dict(payload['state_dict']);deployed.requires_grad_(False)
    torch.set_num_threads(2)
    verify_actor(deployed,torch.load(tmp_path/'export/controls.pt',weights_only=True))
    torch.set_num_threads(1)
    with pytest.raises(ValueError):export_final(root,tmp_path/'again')
    with pytest.raises(ValueError):export_final(root,tmp_path/'export',visual_dim=4,episodes=1,updates=1)
    (root/'episodes/0000.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):export_final(root,tmp_path/'changed',visual_dim=4,episodes=1,updates=1)
    assert not (tmp_path/'changed').exists()


def test_stage_export_pins_optimizer_rng_without_changing_resume_binding(tmp_path):
    from AlphaBrain.research.dsol.data.artifacts import sha
    torch.set_num_threads(1);policy=ContinuousActorCritic(4)
    old,source=synthetic_session()
    session=ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',
        wrist=np.ones(4),score_scale=1.,seed=0)
    session.start();obs=continuous_observation(session.observation())
    root=tmp_path/'A-seed-0';plan=tmp_path/'plan.json';cfg=tmp_path/'config.json'
    write_json(plan,dict(training=dict(episodes_per_update=1,updates_per_run=3,episodes_per_run=3,
                                     arms=['A'],seeds=[0],query_cap=8)))
    write_json(cfg,dict(run=root.name,protocol=str(plan),output=str(root),updates=3,episodes_per_update=1))
    binding=dict(config_sha256=sha(cfg))
    with closing(ContinuousTrainingStore(root,binding)) as store:
        store.save_episode(0,dict(transitions=[dict(observation=obs)]),dict(queries=1))
        store.save_checkpoint(1,dict(policy=policy.state_dict(),optimizer={'state':{}},update=1,cursor=1,queries=1))
    target_policy=root/'policy.pt';torch.save({'state_dict':policy.state_dict()},target_policy)
    write_json(root/'publication.json',dict(status='PAUSED_AT_UPDATE_BOUNDARY',version=1,episodes=1,
        queries=1,run=root.name,policy=str(target_policy),policy_sha256=sha(target_policy)))
    stage=tmp_path/'stage.json'
    write_json(stage,dict(schema='PPO_FIXED_STAGE_V1',selection='USER_FIXED_BEFORE_ENDPOINT_NOT_OUTCOME_SELECTED',
        continuation='EXPLICIT_LATER_AUTHORIZATION_REQUIRED',evaluation_panel_changed=False,
        optimizer_schedule_or_reward_changed=False,preserve_full_checkpoint=True,
        original_plan=str(plan),original_plan_sha256=sha(plan),updates=1,episodes_per_run=1,episodes_per_update=1,
        training_calls_cap=16,runs={root.name:dict(path=str(cfg),sha256=sha(cfg))}))
    before={p:sha(root/p) for p in ('binding.json','latest.json','publication.json')}
    result=export_stage(root,tmp_path/'export',stage_path=stage,stage_sha256=sha(stage),visual_dim=4)
    assert result['status']=='FROZEN_STAGE_ENDPOINT'
    assert sha(tmp_path/'export/resume-checkpoint.pt')==json.loads((root/'latest.json').read_text())['sha256']
    pinned=torch.load(tmp_path/'export/resume-checkpoint.pt',weights_only=True)
    assert {'policy','optimizer','rng','cursor','update','queries'}.issubset(pinned)
    assert not (root/'status.json').exists()
    assert before=={p:sha(root/p) for p in before}
    with pytest.raises(ValueError):export_stage(root,tmp_path/'export',stage_path=stage,stage_sha256=sha(stage),visual_dim=4)
    write_json(root/'inflight.json',{'kind':'episode','index':1})
    with pytest.raises(ValueError,match='ambiguous'):
        export_stage(root,tmp_path/'blocked',stage_path=stage,stage_sha256=sha(stage),visual_dim=4)
