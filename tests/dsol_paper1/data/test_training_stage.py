import json
import pytest
from AlphaBrain.research.dsol.data.artifacts import sha
from AlphaBrain.research.dsol.data.training_stage import load_stage, evaluation_endpoint


@pytest.fixture
def stage(tmp_path):
    plan=tmp_path/'plan.json'
    plan.write_text(json.dumps({'training':dict(episodes_per_update=32,updates_per_run=256,
        episodes_per_run=8192,arms=['A','B'],seeds=[0,1,2],query_cap=8)}))
    configs={}
    for arm in ('A','B'):
        for seed in range(3):
            name=f'{arm}-seed-{seed}';p=tmp_path/(name+'.json')
            p.write_text(json.dumps(dict(run=name,protocol=str(plan),updates=256,episodes_per_update=32)))
            configs[name]=dict(path=str(p),sha256=sha(p))
    row=dict(schema='PPO_FIXED_STAGE_V1',selection='USER_FIXED_BEFORE_ENDPOINT_NOT_OUTCOME_SELECTED',
        continuation='EXPLICIT_LATER_AUTHORIZATION_REQUIRED',evaluation_panel_changed=False,
        optimizer_schedule_or_reward_changed=False,preserve_full_checkpoint=True,
        original_plan=str(plan),original_plan_sha256=sha(plan),runs=configs,
        updates=64,episodes_per_run=2048,episodes_per_update=32,training_calls_cap=196608)
    p=tmp_path/'stage.json';p.write_text(json.dumps(row))
    return p,row


def test_explicit_stage_preserves_legacy_and_requires_bound_amendment(stage):
    p,row=stage
    assert evaluation_endpoint({})==(256,8192)
    assert evaluation_endpoint(dict(training_stage=str(p),input_hashes={str(p):sha(p)}))==(64,2048)
    with pytest.raises(ValueError,match='Unbound'):evaluation_endpoint(dict(training_stage=str(p)))
    with pytest.raises(ValueError,match='hash'):load_stage(p,'bad')


@pytest.mark.parametrize('key,value',[('updates',65),('updates',True),('episodes_per_run',2049),
    ('episodes_per_update',16),('training_calls_cap',196609),('preserve_full_checkpoint',False),
    ('evaluation_panel_changed',True),('optimizer_schedule_or_reward_changed',True),
    ('selection','BEST_VALIDATION_RESULT'),('continuation','AUTOMATIC')])
def test_stage_rejects_inconsistent_or_unreviewed_changes(stage,key,value):
    p,row=stage;row[key]=value;p.write_text(json.dumps(row))
    with pytest.raises(ValueError):load_stage(p,sha(p))


def test_stage_requires_all_seeds_and_original_config_bytes(stage):
    p,row=stage;original=row['runs'].pop('B-seed-2');p.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='arms and seeds'):load_stage(p,sha(p))
    row['runs']['B-seed-2']=original;p.write_text(json.dumps(row))
    from pathlib import Path
    Path(original['path']).write_text('{}')
    with pytest.raises(ValueError,match='config changed'):load_stage(p,sha(p))
