import copy
import hashlib
import json
from unittest.mock import patch

import pytest

from AlphaBrain.research.dsol.data.artifacts import read, sha, path_mapping
from AlphaBrain.research.dsol.data.official_view_space import validate, FORMAT


def test_mapping_preserves_raw_identity_and_prefix_boundaries(tmp_path):
    old='/retired/test-root'
    payload={'path':old+'/data','similar':old+'-other/data','nested':[old+'/weights']}
    p=tmp_path/'record.json';p.write_text(json.dumps(payload))
    expected=hashlib.sha256(p.read_bytes()).hexdigest()
    with path_mapping([[old,str(tmp_path)]]):
        assert read(old+'/record.json')['path']==str(tmp_path)+'/data'
        assert read(p)['similar']==old+'-other/data'
        assert sha(old+'/record.json')==expected
        assert json.loads(p.read_text())==payload
    assert read(p)==payload


def test_full_catalog_extension_rejects_scope_drift():
    views=['canonical']+[f'v{i}' for i in range(96)]
    reused=[views[i] for i in (0,2,3,4,5)]
    base=dict(initial_ids=list(range(40)),workers=[],numeric_flags='fixed',noise_version='fixed',
              checkpoint_repeats=[0,1,2,3],confirmation_repeats=[],first_input_source='frozen',
              p0_plan='p0',p0_plan_sha256='sha',panel_views=reused,
              models={'official':dict(checkpoint='/checkpoint',config='cfg',norm_sha256='sha',
                                      files={'weights':dict(sha256='sha',bytes=100,mtime_ns=0)})})
    plan=copy.deepcopy(base)
    plan.update(format=FORMAT,baseline_plan='baseline',baseline_plan_sha256='sha',
                baseline_audit='audit',baseline_audit_sha256='sha',all_views=views,reuse_views=reused,
                panel_views=[v for v in views if v not in reused],rollout_count=14720,reused_rollouts=800)
    ref={'catalog':{'views':[dict(view_id=v) for v in views]}}
    with patch('AlphaBrain.research.dsol.data.official_view_space.sha',return_value='sha'), \
         patch('AlphaBrain.research.dsol.data.official_view_space.read',side_effect=lambda p:base if p=='baseline' else {'status':'PASS'}), \
         patch('AlphaBrain.research.dsol.data.official_view_space.validate_baseline_panel',return_value=({},ref)):
        validate(plan)
        for key,value in [('rollout_count',15520),('panel_views',views),('initial_ids',list(range(39))),('checkpoint_repeats',[0,1])]:
            bad=copy.deepcopy(plan);bad[key]=value
            with pytest.raises(ValueError):validate(bad)
        bad=copy.deepcopy(plan);bad['models']['official']['files']['weights']['sha256']='bad'
        with pytest.raises(ValueError):validate(bad)


def test_runtime_module_import_is_cpu_safe():
    from benchmarks.Robocasa365 import evaluation_matrix
    assert callable(evaluation_matrix._rollout)
    assert callable(evaluation_matrix._score_cell)


def test_multisegment_runtime_paths_are_all_relocated():
    from benchmarks.Robocasa365.baseline_dispatch import relocated_environment
    original={'PYTHONPATH':'/new/source:/runtime:/old/project/src:/old/client/src',
              'LD_PRELOAD':'/old/library.so', 'UNCHANGED':'/old-extra/file'}
    result=relocated_environment(original,[['/old','/new']])
    assert result['PYTHONPATH']=='/new/source:/runtime:/new/project/src:/new/client/src'
    assert result['LD_PRELOAD']=='/new/library.so'
    assert result['UNCHANGED']=='/old-extra/file'
    assert original['LD_PRELOAD']=='/old/library.so'
