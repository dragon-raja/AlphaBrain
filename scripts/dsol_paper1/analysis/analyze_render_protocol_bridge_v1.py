#!/usr/bin/env python3
"""Paired source-cluster effects; no conflation of repeatability and validity."""

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))

import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scripts.dsol_paper1.runtime.render_bridge_common_v1 import ROOT, read, sha, write_new, cell_key, verify_release

def interval(values,seed=20260908,draws=10000):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or len(values)!=8:raise ValueError('Exactly eight source-cluster contrasts required')
    rng=np.random.default_rng(seed)
    boot=values[rng.integers(0,8,size=(draws,8))].mean(axis=1)
    return {'mean_pp':float(100*values.mean()),'ci95_pp':list(100*np.quantile(boot,[.025,.975])),
        'cluster_count':8,'source_contrasts_pp':list(100*values)}

def signatures_equal(a,b):
    return a['bridge']['calls']==b['bridge']['calls'] and a['bridge']['steps']==b['bridge']['steps'] and \
        a['success']==b['success'] and a['completion_steps']==b['completion_steps']

def rows(release,cell):
    output=ROOT/'rollouts'/cell['name'];completion=read(output/'completion.json')
    if completion['release_sha256']!=sha(ROOT/'release.json'):raise ValueError('Cell belongs to a different release')
    result=[]
    for i,spec in enumerate(release['specs']):
        path=output/f'{i:04d}.json'
        if sha(path)!=completion['rows'].get(str(path)):raise ValueError('Ledger changed')
        row=read(path)
        if cell_key(row)!=cell_key(spec) or row['bridge']['index']!=i:raise ValueError('Wrong paired key')
        if row['bridge']['release_sha256']!=sha(ROOT/'release.json') or set(row['bridge']['render_samples'])!={cell['offsamples']}:raise ValueError('Wrong renderer')
        result.append(row)
    if len(completion['rows'])!=192:raise ValueError('Cell includes extra or missing rows')
    return result,completion

def rank_comparison(left,right):
    a={r['candidate_id']:r['mean_accel_3'] for r in left['ranking']};b={r['candidate_id']:r['mean_accel_3'] for r in right['ranking']}
    if set(a)!=set(b) or len(a)!=97:raise ValueError('Candidate set mismatch')
    order=sorted(a); x=np.array([a[k] for k in order]);y=np.array([b[k] for k in order])
    rho=spearmanr(x,y).statistic
    return {'spearman':float(rho) if np.isfinite(rho) else None,'top1_same':left['selected_candidate_id']==right['selected_candidate_id'],
        'top10_overlap':len({r['candidate_id'] for r in left['ranking'][:10]} & {r['candidate_id'] for r in right['ranking'][:10]}),
        'max_accel_abs_difference':float(np.abs(x-y).max()),'member_scores_exact':all(
            next(r for r in left['ranking'] if r['candidate_id']==k)['member_accel_3']==
            next(r for r in right['ranking'] if r['candidate_id']==k)['member_accel_3'] for k in order)}

def main():
    release=verify_release(); all_rows={}; times={}
    for cell in release['cells']:all_rows[cell['name']],times[cell['name']]=rows(release,cell)
    grouped={}
    state_keys=[s['pair_key'] for s in release['states']]
    for cell,values in all_rows.items():
        grouped[cell]=np.array([np.mean([r['success'] for r in values if r['pair_key']==key]) for key in state_keys])
    report={'status':'PASS_COMPLETE_BRIDGE_ANALYSIS','release_sha256':sha(ROOT/'release.json'),'scope':release['analysis']['limitations'],
        'success_rates':{k:float(100*v.mean()) for k,v in grouped.items()},'repeatability':{},'effects':{},'metrics':{}}
    for model in ('canonical','broad'):
        for renderer,first,last in [('aa4','aa4-a-32','aa4-b-32'),('aa0','aa0-a-32','aa0-b-48')]:
            a=all_rows[model+'/'+first];b=all_rows[model+'/'+last]
            mismatches=[i for i,(x,y) in enumerate(zip(a,b)) if not signatures_equal(x,y)]
            flips=[i for i,(x,y) in enumerate(zip(a,b)) if x['success']!=y['success']]
            # Every common-index noise request must match even if horizon diverges.
            for x,y in zip(a,b):
                for u,v in zip(x['bridge']['calls'],y['bridge']['calls']):
                    if u['inputs']['_eval_noise']!=v['inputs']['_eval_noise']:raise ValueError('Noise alignment failed')
            report['repeatability'][model+'/'+renderer]={'episodes':192,'full_signature_mismatches':mismatches,
                'success_flip_indices':flips,'success_flip_rate':len(flips)/192,
                'speedup_including_startup':times[model+'/'+first]['wall_seconds_including_startup']/times[model+'/'+last]['wall_seconds_including_startup'],
                'rollout_only_speedup':times[model+'/'+first]['rollout_seconds']/times[model+'/'+last]['rollout_seconds']}
        old=(grouped[model+'/aa4-a-32']+grouped[model+'/aa4-b-32'])/2
        new=grouped[model+'/aa0-a-32'] # Repeat B is a determinism check, not extra independent data.
        report['effects'][model+'_renderer_change']=interval(new-old)
    old_gap=(grouped['broad/aa4-a-32']+grouped['broad/aa4-b-32']-grouped['canonical/aa4-a-32']-grouped['canonical/aa4-b-32'])/2
    new_gap=grouped['broad/aa0-a-32']-grouped['canonical/aa0-a-32']
    report['effects']['training_gap_old']=interval(old_gap);report['effects']['training_gap_new']=interval(new_gap)
    report['effects']['training_renderer_interaction']=interval(new_gap-old_gap)
    for key in ('canonical_renderer_change','broad_renderer_change','training_renderer_interaction'):
        lo,hi=report['effects'][key]['ci95_pp']
        report['effects'][key]['within_preregistered_5pp_margin']=lo>-5 and hi<5
    for model in ('canonical','broad'):
        metric=[]
        for i,state in enumerate(release['states']):
            ranks={cell:read(ROOT/'scores'/model/cell/f'{i:02d}'/'ranking.json') for cell in release['asset_cells']}
            for cell,rank in ranks.items():
                receipt=read(ROOT/'scores'/model/cell/f'{i:02d}'/'bridge-receipt.json')
                asset=read(ROOT/'assets'/cell/f'{i:02d}'/'render.json')
                if receipt['ranking_sha256']!=sha(ROOT/'scores'/model/cell/f'{i:02d}'/'ranking.json') or rank['render_artifact_sha256']!=asset['artifact_sha256']:raise ValueError('Metric provenance mismatch')
            metric.append({'state':state['pair_key'],'aa4_repeat':rank_comparison(ranks['aa4-a'],ranks['aa4-b']),
                'aa0_repeat':rank_comparison(ranks['aa0-a'],ranks['aa0-b']), 'renderer_change':rank_comparison(ranks['aa4-a'],ranks['aa0-a'])})
        report['metrics'][model]=metric
    report['asset_repeatability']=[]
    for i,state in enumerate(release['states']):
        row={'state':state['pair_key']}
        for label,a,b in [('aa4_repeat','aa4-a','aa4-b'),('aa0_repeat','aa0-a','aa0-b'),('renderer_change','aa4-a','aa0-a')]:
            ra=read(ROOT/'assets'/a/f'{i:02d}'/'render.json');rb=read(ROOT/'assets'/b/f'{i:02d}'/'render.json')
            with np.load(ra['artifact']) as x,np.load(rb['artifact']) as y:
                row[label]={'identical_arrays':{k:x[k].dtype==y[k].dtype and x[k].shape==y[k].shape and x[k].tobytes()==y[k].tobytes() for k in x.files},
                    'visibility_exact':ra['visibility']==rb['visibility'],
                    'max_visibility_difference':max(abs(u['score']-v['score']) for u,v in zip(ra['visibility'],rb['visibility']))}
        report['asset_repeatability'].append(row)
    deterministic=all(not report['repeatability'][m+'/aa0']['full_signature_mismatches'] for m in ('canonical','broad'))
    deterministic=deterministic and all(all(r['aa0_repeat']['identical_arrays'].values()) and r['aa0_repeat']['visibility_exact'] for r in report['asset_repeatability'])
    report['noaa_repeatability_pass']=deterministic
    report['speed_candidate_48']=deterministic and all(report['repeatability'][m+'/aa0']['speedup_including_startup']>=1.05 for m in ('canonical','broad'))
    report['migration_decision']='REQUIRES_REVIEW_NO_AUTOMATIC_FULL_REPLACEMENT'
    report['not_claimed']=['all historical results valid or invalid','renderer equivalence from nonsignificance','97-view oracle from 3-view closed loops','new render data reusable in old scientific union']
    write_new(ROOT/'analysis.json',report)
    lines=['# 渲染协议桥接验证结果','',f"新协议完整重复性检查：{deterministic}。48 并发候选通过：{report['speed_candidate_48']}。",'',
        '这是一组 8 个来源状态的协议审计，不是全部历史实验的有效性认证。','',
        '| 配对效应 | 均值（百分点） | 来源聚类 95% 区间 |','|---|---:|---|']
    for name,r in report['effects'].items():lines.append(f"| {name} | {r['mean_pp']:.2f} | [{r['ci95_pp'][0]:.2f}, {r['ci95_pp'][1]:.2f}] |")
    lines+=['','原始结果和冻结协议未覆盖；是否整体迁移需结合区间宽度、逐任务表现和指标稳定性复核。']
    with (ROOT/'summary_zh.md').open('x') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':report['status'],'noaa_repeatability_pass':deterministic,'speed_candidate_48':report['speed_candidate_48']}))

if __name__=='__main__':main()
