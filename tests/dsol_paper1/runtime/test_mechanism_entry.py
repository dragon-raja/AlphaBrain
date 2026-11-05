"""Exercise actual evaluation entry with synthetic sensor and actor backends."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
from tests.dsol_paper1.helpers.local_search import synthetic_session

@pytest.mark.parametrize('cap',[4,6,8,16])
def test_actual_search_enforces_cap_and_charges_return(tmp_path,monkeypatch,cap):
    from benchmarks.LIBERO import continuous_evaluation as entry,local_view_runtime
    from AlphaBrain.research.dsol.data.core_mechanism import mechanism_spec
    core,source=synthetic_session();source.mount=core.mount;source.physical='synthetic'
    source.current=np.array([0.,0.]);source.last_example={};source.last_values=[dict(wrist=np.ones(4))]
    original=source.query
    def query():
        result=original();source.queries=source.calls;return result
    source.query=query;source.queries=0
    source.move_angles=lambda angles:source.move(source.mount.pose(angles))
    monkeypatch.setattr(local_view_runtime,'RenderedViewSource',lambda *a:source)
    monkeypatch.setattr(entry,'actor_action',lambda *a,**kw:dict(stop=False,delta_deg=np.array([1.125,0.])))
    actor=SimpleNamespace(mechanism=mechanism_spec('Q'+str(cap)) if cap!=8 else {})
    info,_=entry.search(dict(method='continuous_ppo',angles_deg=[0.,0.],proposal_seed=19,query_cap=cap),None,None,None,tmp_path,actor,1.)
    assert info['queries']==source.calls==cap and info['reason']=='budget'
    np.testing.assert_array_equal(info['selected_angles'],[0.,0.])


def test_campaign_deployment_only_does_not_train(tmp_path):
    from benchmarks.LIBERO.prefix_campaign import PrefixCampaign
    from threading import RLock
    co=PrefixCampaign.__new__(PrefixCampaign);co.root=tmp_path;co.cfg={};co.lock=RLock();co.failures={}
    co.protocol=dict(design_stage='mechanism_screen',cells={'Q4':dict(evaluation_only=True,evaluation_schedules={'panel':dict(version=16)})})
    calls=[];co.execute=lambda n,k,s,**kw:calls.append((k,kw['version']));co.status=lambda *a,**kw:None
    co.cpu_update=lambda *a:pytest.fail('Must never update deployment-only actor')
    co.cell('Q4');assert calls==[('evaluation',16)]
