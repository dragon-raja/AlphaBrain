#!/usr/bin/env python3
"""Freeze a bounded rendering audit without changing any existing release."""

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))

from pathlib import Path
from scripts.dsol_paper1.render_bridge_common_v1 import ROOT, REPO, read, sha, write_new, cell_key
from scripts.dsol_paper1.evaluate_dsol_libero_hdf5_views import protocol_spec_at, protocol_spec_count
import scripts.dsol_paper1.run_full64_view_landscape_v2 as old_control

def identity(path): return {'path':str(path),'sha256':sha(path)}

def build():
    if ROOT.exists(): raise FileExistsError('No implicit overwrite or re-selection')
    old=old_control.preflight(); anchor=old['old_release_content']
    protocol=read(old['benchmark_protocol']['path'])
    specs=[protocol_spec_at(protocol,i) for i in range(protocol_spec_count(protocol))]
    states=[b['state'] for b in protocol['state_blocks']]
    assert len(specs)==len(set(map(cell_key,specs)))==192
    assert len(states)==len({s['task_id'] for s in states})==8
    assert {s['selected_candidate_id'] for s in specs}=={'canonical','broad_train_000','broad_heldout_000'}
    assert {s['policy_repeat_id'] for s in specs}==set(range(8))
    source_paths=[Path(old['benchmark_protocol']['path']),Path(anchor['catalog']['path']),Path(anchor['noise_bank']['path']),
        REPO/'scripts/dsol_paper1/evaluate_dsol_libero_hdf5_views.py',REPO/'scripts/dsol_paper1/trace_view_repeatability_v1.py',
        REPO/'scripts/dsol_paper1/run_view_value_expectation_accel_ensemble.py',REPO/'scripts/dsol_paper1/explicit_flow_noise.py',
        REPO/'scripts/dsol_paper1/libero_visibility.py',REPO/'scripts/cabi_vla/serve_alphabrain_pi05_websocket.py']
    source_paths += [REPO/'scripts/dsol_paper1'/name for name in (
        'render_bridge_common_v1.py', 'run_render_bridge_worker_v1.py',
        'protocols/build_render_protocol_bridge_v1.py',
        'operations/controllers/run_render_protocol_bridge_v1.py',
        'operations/controllers/queue_render_protocol_bridge_v1.py',
        'analysis/analyze_render_protocol_bridge_v1.py')]
    for state in states:
        for key in ('visibility_scan','policy_inputs','render_receipt'):
            item=state['static_assets']; assert sha(item[key])==item[key+'_sha256']
            source_paths.append(Path(item[key]))
        scan=read(state['static_assets']['visibility_scan'])
        assert scan['scene_construction']==next(b['scene_construction'] for b in protocol['state_blocks'] if b['state']['pair_key']==state['pair_key'])
        assert {r['visibility']['height'] for r in scan['records'] if 'visibility' in r}=={224}
    models={k:{field:anchor['canonical_checkpoint' if k=='canonical' else 'broad_reference_checkpoint'][field]
        for field in ('path','weights_sha256')} for k in ('canonical','broad')}
    source_paths += [REPO/p for p in anchor['critical_files']]
    for model in models.values():
        assert sha(Path(model['path'])/'model.safetensors')==model['weights_sha256']
        source_paths.append(Path(model['path'])/'framework_config.yaml')
    cells=[]
    for name,samples,workers,copies in [('aa4-a-32',4,32,2),('aa4-b-32',4,32,2),('aa0-a-32',0,32,2),('aa0-b-48',0,48,3)]:
        for model in ('canonical','broad'):
            cells.append({'name':model+'/'+name,'model':model,'offsamples':samples,'workers':workers,'copies':copies,'episodes':192})
    release={'status':'FROZEN_USER_AUTHORIZED_RENDER_BRIDGE','root':str(ROOT),
        'purpose':'Audit rendering repeatability and training-treatment sensitivity; no blanket historical-validity claim',
        'source_selection':'Reuse prior outcome-blind first-eight benchmark unchanged; no selection using renderer failures or success',
        'models':models,'states':states,'specs':specs,'catalog':anchor['catalog'],'noise_bank':anchor['noise_bank'],
        'sources':[identity(p) for p in sorted(set(source_paths))], 'cells':cells,
        'asset_cells':['aa4-a','aa4-b','aa0-a','aa0-b'], 'episode_count':1536,
        'asset_render_count':3104,'score_count':49664,'score_ensemble_size':8,'score_root_seed':20260921,
        'scientific_controls':{'replan_steps':5,'wait_steps':0,'flow_denoising_steps':10,'resize_size':224,
            'raw_rgb_size':256,'visibility_size':224,'wrapper_noise':0,'score_batch_size':16},
        'analysis':{'primary':['old-AA4 same-key success flip rate','AA0 full-trajectory bitwise agreement at 32 vs 48',
            'Broad-minus-canonical paired effect by renderer','difference of those training effects',
            '97-view Accel and visibility ranking repeatability and renderer sensitivity'],
            'bootstrap_unit':'source state (8 clusters); retain all views/noise/models jointly', 'bootstrap_draws':10000,'bootstrap_seed':20260908,
            'practical_review_margin_pp':5.0,'equivalence_rule':'95% paired source-cluster interval wholly within [-5,+5]pp for renderer main effect and training interaction; bounded subset only',
            'noaa_repetitions_not_independent_samples':True,'empirical_max_not_a_confirmed_oracle':True,
            'concurrency_rule':'All episode keys, full input/action/physical signatures and success exactly equal; >=5% same-instrumentation wall-time speedup. Candidate only, no automatic old-release change.',
            'limitations':'8 known, constructed-scene source states and only 3 closed-loop candidate views; not all legacy passive-camera tests, unseen tasks, training seeds or full97 oracle'},
        'base_port':29700,'max_execution_seconds':7200,'max_drain_seconds':7200,
        'automatic_retry':False,'overwrite_legacy':False,'automatic_full_rerun':False,'training':False,
        'authorization':'User approved matched old/new protocol verification and requested implementation on 2026-09-08'}
    ROOT.mkdir(); write_new(ROOT/'release.json',release)
    return release

if __name__=='__main__':
    r=build(); print({'status':r['status'],'episodes':r['episode_count'],'root':str(ROOT)})
