import json
from pathlib import Path

import pytest

from AlphaBrain.research.dsol.data.continuous_evaluation_blocks import blocks,STAGES
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest
from AlphaBrain.research.dsol.execution.queue import MissingResult
from benchmarks.LIBERO.continuous_evaluation_fleet import build,inventory,request_for,validate_block,prior_stages
from reports.paper1.continuous_fleet_tracking import snapshot


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value));return str(path)


@pytest.fixture
def campaign(tmp_path):
    plan=json.loads((Path(__file__).resolve().parents[3]/'configs/dsol_paper1/search/continuous_search_v1.json').read_text())
    rows=[dict(cohort=t['cohort'],initial=i,task=t['task'],state_key=f"{t['cohort']}:{i}",start=start,
        method=method,angles_deg=angles,query_cap=1 if method=='stay' else 8,execution_repeats=[0,1,2,3])
        for t in plan['evaluation']['tasks'] for i in t['indices']
        for start,angles in enumerate(plan['camera']['evaluation_start_angles_deg']) for method in plan['evaluation']['methods']]
    design=write(tmp_path/'design.json',plan);schedule=write(tmp_path/'schedule.json',rows)
    conditions={}
    for model,host in [('canonical','fresh'),('broad','video'),('native_reference','gnu')]:
        paired=write(tmp_path/model/'paired.json',dict(design=design,schedule=schedule))
        nominal=write(tmp_path/model/'nominal.json',dict(paired_config=paired,paired_config_sha256=file_digest(paired)))
        runtime=write(tmp_path/model/'runtime.json',dict(host=host,source_artifacts=[],slots=[dict(host=host,id=str(i)) for i in range(8)]))
        conditions[model]=dict(host=host,paired=paired,paired_sha256=file_digest(paired),nominal=nominal,
                              nominal_sha256=file_digest(nominal),runtime=runtime,runtime_sha256=file_digest(runtime))
    return write(tmp_path/'campaign.json',dict(schema='continuous_evaluation_fleet_v1',run_id='synthetic',
        conditions=conditions,output=str(tmp_path/'results'),control=str(tmp_path/'control')))


def completed_nominal(campaign):
    cfg,_=inventory(campaign);req=request_for(campaign,'canonical',16)
    root=Path(cfg['output'])/'canonical/blocks/16';row=req['block']['rows'][0];tasks=[]
    for seed in range(4):
        asset=root/f"{row['cohort']}-{row['initial']:02d}-seed-{seed}"/'execution.npz'
        asset.parent.mkdir(parents=True,exist_ok=True);asset.write_bytes(b'synthetic trajectory')
        tasks.append(dict(row,repeat=seed,success=False,replans=1,execution_sha256=file_digest(asset)))
    write(root/'results.json',dict(tasks=tasks,searches=[],block_index=16))
    write(root/'status.json',dict(status='COMPLETE',config_sha256=file_digest(req['nominal']),calls=dict(execution=4)))
    write(root/'timing.json',{})
    (root/'model-attempts.jsonl').write_text(''.join(json.dumps(dict(index=i,stage_index=i,identity=dict(stage='execution')))+'\n' for i in range(1,5)))
    seal(root,req)
    return req,root


def seal(root,req):
    artifacts={str(p.relative_to(root)):file_digest(p) for p in root.rglob('*') if p.is_file() and p.name!='complete.json'}
    write(root/'complete.json',dict(status='COMPLETE',request_sha256=digest(req),artifacts=artifacts))


def test_disjoint_ownership_and_full_frozen_budget(campaign,tmp_path):
    cfg,plans=inventory(campaign)
    assert sorted(i for indices in STAGES.values() for i in indices)==list(range(24))
    all_addresses=[];total=0
    for model,work in plans.items():
        assert sum(b['tasks'] for b in work)==224
        assert sum(sum(b['limits'].values()) for b in work)==12864
        for stage in STAGES:
            queue=build(campaign,model,stage,tmp_path/'queues'/model/stage)
            assert len(queue['slots'])==(1 if stage=='admission' else 8)
            addresses=[digest(a) for t in queue['tasks'] for a in t['addresses']]
            all_addresses+=addresses;total+=len(addresses)
    assert total==672 and len(set(all_addresses))==672
    assert not Path(cfg['control']).exists() and not Path(cfg['output']).exists()
    with pytest.raises(MissingResult):prior_stages(campaign,'capacity')


def test_broken_seeds_and_method_order_rejected(campaign):
    cfg,_=inventory(campaign);paired=json.loads(Path(cfg['conditions']['canonical']['paired']).read_text())
    plan=json.loads(Path(paired['design']).read_text());rows=json.loads(Path(paired['schedule']).read_text())
    rows[0]['execution_repeats']=[0]
    with pytest.raises(ValueError):blocks(plan,rows)


def test_complete_negative_results_are_kept_missing_not_failed(campaign):
    empty=snapshot(campaign,'canonical')
    assert empty['scalars']['evaluation/completed_tasks']==0
    assert 'nominal/all/success_rate' not in empty['scalars']
    req,root=completed_nominal(campaign)
    assert validate_block(req,root)['calls']=={'execution':4}
    value=snapshot(campaign,'canonical')
    assert value['scalars']['nominal/all/success_rate']==0
    assert value['scalars']['nominal/all/tasks']==4
    assert value['scalars']['evaluation/missing_blocks']==23
    assert value['status']=='PARTIAL'


def test_corrupt_duplicate_budget_and_missing_evidence_fail_closed(campaign):
    req,root=completed_nominal(campaign)
    original=(root/'results.json').read_text()
    data=json.loads(original);data['tasks'][-1]=data['tasks'][0]
    write(root/'results.json',data);seal(root,req)
    with pytest.raises(ValueError,match='duplicate'):validate_block(req,root)
    (root/'results.json').write_text(original);write(root/'status.json',dict(status='COMPLETE',config_sha256=file_digest(req['nominal']),calls=dict(execution=241)));seal(root,req)
    with pytest.raises(ValueError,match='budget'):validate_block(req,root)
    write(root/'status.json',dict(status='COMPLETE',config_sha256=file_digest(req['nominal']),calls=dict(execution=4)));seal(root,req)
    (root/'timing.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='artifact'):validate_block(req,root)
    (root/'timing.json').unlink();seal(root,req)
    with pytest.raises(ValueError,match='missing'):validate_block(req,root)
