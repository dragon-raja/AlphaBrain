import json
from pathlib import Path
import pytest
from benchmarks.LIBERO import controlled_adaptation as a
from AlphaBrain.research.dsol.data import domain_adaptation as d
from AlphaBrain.research.dsol.execution.manifest import file_digest


def fixture(tmp_path):
    paths={};pubs={}
    for seed in range(3):
        p=tmp_path/f'schedule-{seed}.json'
        p.write_text(json.dumps([dict(episode=i,task='task',initial=i,state_key=f'state-{i}') for i in range(64)]))
        paths[str(seed)]=str(p)
        for arm in d.ARMS:
            p=tmp_path/f'{arm}-{seed}.pt';p.write_bytes(b'fake-publication')
            pubs[f'{arm}-seed-{seed}']=dict(version=0,episodes=0,policy=str(p),policy_sha256=file_digest(p))
    common=dict(operation_model='fastwam',observation_scale=.16535300100378683,input_hashes={})
    return common,dict(arms=list(d.ARMS),seeds=[0,1,2],schedules=paths),pubs


def test_complete_twelve_on_policy_batches(tmp_path):
    common,protocol,pubs=fixture(tmp_path);c=a.collection(common,protocol,pubs,0)
    assert c['limits']==dict(episodes=96,score=1536,paired_execution=0,nominal_execution=0)
    assert len(c['blocks'])==12 and all(b['host']=='fresh' and len(b['rows'])==8 for b in c['blocks'])
    assert not common['input_hashes']
    assert len({tuple(sorted(x.items())) for b in c['blocks'] for x in b['addresses']})==96
    for b in c['blocks']:assert c['input_hashes'][b['policy']]==b['policy_sha256']


@pytest.mark.parametrize('fault',['version','episodes','bytes'])
def test_stale_publication_rejected_before_dispatch(tmp_path,fault):
    common,protocol,pubs=fixture(tmp_path);p=next(iter(pubs.values()))
    if fault=='bytes':Path(p['policy']).write_bytes(b'changed')
    else:p[fault]=1
    with pytest.raises(ValueError,match='Mixed/stale'):a.collection(common,protocol,pubs,0)


def test_default_cannot_start(tmp_path):
    with pytest.raises(ValueError,match='Explicit bounded'):a.run(tmp_path/'missing.json')


def test_predeclared_endpoint_count_and_batch_dedup(tmp_path):
    from benchmarks.LIBERO.controlled_adaptation_evaluation import endpoint_actors
    runs=[f'{arm}-seed-{s}' for arm in d.ARMS for s in (0,1,2)]
    for version in (0,1,4,8):
        p=tmp_path/('initial-publications.json' if not version else f'endpoints/update-{version:02d}.json')
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps({r:dict(policy=f'{r}-{version}',version=version) for r in runs}))
    (tmp_path/'first-batch-gate.json').write_text(json.dumps(dict(status='PASS')))
    actors=endpoint_actors(tmp_path)
    assert len(actors)==33
    assert sum(x['version']==0 for x in actors.values())==3
    assert sum(x['version']==1 for x in actors.values())==6
    assert sum(x['version']==4 for x in actors.values())==12
    (tmp_path/'first-batch-gate.json').write_text(json.dumps(dict(status='FAIL')))
    with pytest.raises(ValueError):endpoint_actors(tmp_path)


def test_short_dual_host_blocks_preserve_all_science_and_budget(tmp_path):
    from benchmarks.LIBERO.controlled_adaptation_evaluation import campaign
    from AlphaBrain.research.dsol.data.operation_contract import fastwam_contract
    sim=tmp_path/'sim.json';sim.write_text(json.dumps(dict(operation_contract=fastwam_contract('a'*64))))
    common=dict(sim_config=str(sim),input_hashes={})
    rows=[dict(state_key=str(i),start=i,method='local',suite='libero_spatial') for i in range(32)]
    old=campaign(common,rows,{})
    new=campaign(common,rows,{},block_size=4,hosts=('fresh','video'))
    assert new['limits']==old['limits']
    assert [r for b in new['blocks'] for r in b['rows']]==rows
    assert [b['host'] for b in new['blocks']]==['fresh','video']*4
    assert max(b['limits']['episodes'] for b in new['blocks'])==8
    assert not common['input_hashes']
    # Host syntax is reusable; campaign_nodes enforces the admitted identities.
    for hosts in ((),('fresh','fresh'),('bad host',)):
        with pytest.raises(ValueError):campaign(common,rows,{},hosts=hosts)
    from benchmarks.LIBERO.controlled_campaign import campaign_nodes
    unbound=campaign(common,rows,{},hosts=('unbound',))
    with pytest.raises(ValueError):campaign_nodes({},unbound['blocks'],{})
    for size in (0,9,True):
        with pytest.raises(ValueError):campaign(common,rows,{},block_size=size)


def test_target_host_default_remains_fresh():
    from benchmarks.LIBERO.controlled_target_evaluation import execution_hosts
    assert execution_hosts({})==['fresh']
    assert execution_hosts(dict(execution_hosts=['video']))==['video']
