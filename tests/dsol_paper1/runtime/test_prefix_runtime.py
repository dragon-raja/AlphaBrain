"""Portable actual-entry prefix and native-sequence contract regression."""
import json
from pathlib import Path
import numpy as np
import pytest
from AlphaBrain.research.dsol.data.score_prefix import contract,validate
from AlphaBrain.research.dsol.metrics.score_prefix import score
from AlphaBrain.research.dsol.data.query_features import pool_four_query_groups
from benchmarks.LIBERO.score_prefix_runtime import PrefixRuntime
from benchmarks.LIBERO import prefix_pilot,controlled_training,controlled_target_recovery
from tests.dsol_paper1.helpers.dispatch_policy import allow_synthetic_boundaries


@pytest.mark.parametrize('family,steps,actions,execute', [('pi05',10,10,5),('groot_n15',4,10,10),('rldx',4,8,8)])
def test_native_axes_score_and_actual_decorator(monkeypatch,family,steps,actions,execute):
    allow_synthetic_boundaries(monkeypatch,'benchmarks/LIBERO/score_prefix_runtime.py:PrefixRuntime.infer')
    trace=np.random.default_rng(17).normal(size=(steps,actions,7)).astype(np.float32)
    native_actions=np.random.default_rng(1).normal(size=(actions,7))
    class Runtime:
        calls=0
        def infer(self,*args,**kw):self.calls+=1;return dict(trace=trace,actions=native_actions,score=score(trace,contract(family,3)))
    base=Runtime();a=PrefixRuntime(base,contract(family,3));b=PrefixRuntime(base,contract(family,steps))
    v=a.infer({},None,{},capture=True);vv=b.infer({},None,{},capture=True)
    assert v['actions'] is native_actions and vv['trace'] is trace and base.calls==2
    assert vv['score']==score(trace,contract(family,steps))
    assert contract(family,steps)['executed_actions']==execute
    b.output='new-block';assert base.output=='new-block'
    with pytest.raises(AttributeError):b.score_contract={}
    changed=contract(family,3);changed['scored_actions']=execute+1
    with pytest.raises(ValueError):validate(changed)
    with pytest.raises(ValueError):score(trace[:,:,:6],contract(family,3))


def test_ordered_query_groups_use_all_tokens_without_fake_grid():
    x=np.arange(64*7,dtype=np.float32).reshape(1,64,7);before=x.copy()
    out=pool_four_query_groups(x,queries=64,channels=7).reshape(4,7)
    assert np.array_equal(out,np.stack([x[0,i:i+16].mean(0) for i in range(0,64,16)]))
    assert np.array_equal(x,before)
    for invalid in [x.astype(np.float64),x[:,:63],x*np.nan]:
        with pytest.raises(ValueError):pool_four_query_groups(invalid,queries=64,channels=7)
    from benchmarks.LIBERO.rldx_search_runtime import feature_spec
    from AlphaBrain.research.dsol.data.native_features import read_layout
    spec=feature_spec();streams=read_layout(spec['layout'])
    assert len(streams)==1 and streams[0].fusion_scope=='joint' and streams[0].channels==4096
    assert spec['layout']['pooling']=='explicit_ordered_query_groups_4'


@pytest.mark.parametrize('schema',prefix_pilot.SCHEMAS)
@pytest.mark.parametrize('adapter',[controlled_training,controlled_target_recovery])
@pytest.mark.parametrize('failure',['none','disconnect','unknown'])
def test_actual_registered_entry_delegation_and_no_replay(tmp_path,monkeypatch,adapter,failure,schema):
    allow_synthetic_boundaries(monkeypatch,'benchmarks/LIBERO/'+adapter.__name__.split('.')[-1]+'.py:run','benchmarks/LIBERO/prefix_pilot.py:run')
    p=tmp_path/'campaign.json';p.write_text(json.dumps(dict(schema=schema,kind='admission',blocks=[dict(index=0,host='test')])));c=json.loads(p.read_text())
    monkeypatch.setattr(prefix_pilot,'validate_campaign',lambda _:c);monkeypatch.setattr(prefix_pilot,'cell_for',lambda _:(None,{}))
    attempts=[]
    def admission(c,cell,b,out,guard,pool):
        for i in range(3):
            guard();attempts.append(i)
            if failure=='unknown' and i==1:raise RuntimeError('unknown outcome')
        (out/'admission.json').write_text('{}')
    monkeypatch.setattr(prefix_pilot,'run_admission',admission)
    checks=[]
    def guard():
        checks.append(1)
        if failure=='disconnect' and len(checks)==2:raise ConnectionError('lost center')
    req=adapter.request_for(p,0);out=tmp_path/'out'
    if failure=='none':adapter.run(req,out,guard=guard);assert attempts==[0,1,2]
    else:
        with pytest.raises((RuntimeError,ConnectionError)):adapter.run(req,out,guard=guard)
        assert not (out/'admission.json').exists()
    with pytest.raises(FileExistsError):adapter.run(req,out,guard=guard)
    assert len(attempts)<=3


def test_direct_prefix_entry_is_closed(tmp_path):
    from AlphaBrain.research.dsol.execution.dispatch_policy import ProductionDispatchDisabled
    with pytest.raises(ProductionDispatchDisabled):prefix_pilot.run({},tmp_path/'out',guard=lambda:None)
    assert not (tmp_path/'out').exists()


def test_peer_golden_rejects_same_shape_different_numeric_runtime(tmp_path):
    from benchmarks.LIBERO.prefix_admission import peer_numerical_proof
    refs={}
    for host in ['a','b']:
        d=tmp_path/host;d.mkdir();p=d/'admission.json'
        p.write_text(json.dumps(dict(family='pi05',model_hashes={'suite':'a'*64},admitted_suite='suite')))
        np.savez(d/'numerical.npz',trace=np.zeros((10,10,7),np.float32),actions=np.zeros((10,7),np.float32));refs[host+'/suite']=str(p)
    assert peer_numerical_proof({'pi05':refs})['status']=='PASS_PEER_NATIVE_ACTION_TRACE_EQUALITY'
    np.savez(tmp_path/'b/numerical.npz',trace=np.ones((10,10,7),np.float32),actions=np.zeros((10,7),np.float32))
    with pytest.raises(ValueError,match='Peer native'):peer_numerical_proof({'pi05':refs})


def test_combination_evaluation_requires_exact_variant_proof(tmp_path):
    from benchmarks.LIBERO.prefix_campaign import PrefixCampaign,admission_identity
    from AlphaBrain.research.dsol.data.search_core import core_spec
    schedule=tmp_path/'dev.json';schedule.write_text('[{"suite":"s"}]')
    c=PrefixCampaign.__new__(PrefixCampaign)
    c.protocol={'cells':{'combo':{'family':'pi05','core_spec':core_spec('U123')}}}
    c.cfg={'hosts':{'pi05':['host']}}
    c.proofs={'pi05':{'host/s':'unrelated-core-proof'}};c.variant_proofs={}
    with pytest.raises(ValueError,match='No host admitted'):
        c.execute('combo','evaluation',str(schedule),version=16)
    from contextlib import contextmanager
    class Pool:
        @contextmanager
        def lease(self,key,hosts,**kw):
            assert hosts==['host'];raise RuntimeError('exact-variant-admitted')
            yield
    c.pool=Pool();c.variant_proofs={'U123':{'host/s':'legacy-name-only-proof'}}
    with pytest.raises(ValueError,match='No host admitted'):
        c.execute('combo','evaluation',str(schedule),version=16)
    c.variant_proofs={admission_identity(c.protocol['cells']['combo']):{'host/s':'correct-proof'}}
    with pytest.raises(RuntimeError,match='exact-variant-admitted'):
        c.execute('combo','evaluation',str(schedule),version=16)
