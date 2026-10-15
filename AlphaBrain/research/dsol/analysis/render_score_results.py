"""Summarize exact-input and native micro-pixel replay on frozen outcomes."""
from pathlib import Path
import argparse
import numpy as np
from ..data.artifacts import read,write,sha
from .robocasa_statistics import correlation


def summarize(root,matrix_root):
    root=Path(root);matrix_root=Path(matrix_root);plan=read(root/'plan.json')
    population=read(matrix_root/'population.json');original_ids={j['initial_id']:i for i,j in enumerate(population['jobs'])}
    initials=[j['initial_id'] for j in plan['jobs']];lookup={s:i for i,s in enumerate(initials)};views=plan['views'];view_lookup={v:i for i,v in enumerate(views)}
    population_views=[v['view_id'] if isinstance(v,dict) else v for v in population['views']]
    if views!=population_views:raise ValueError('View ordering changed')
    ids=[original_ids[i] for i in initials]
    with np.load(matrix_root/'matrix.npz') as z:
        old=z['accel_prefixes'][...,1][:,ids];q=z['success'][:,ids].mean(-1)
    # model x archived-image source x initial x view x scoring repeat
    scores=np.repeat(old[:,None],2,axis=1)
    controls=[];pixel_actions=[];receipts={};replayed=set()
    for worker in plan['workers']:
        path=root/'workers'/worker['worker_id'];status=read(path/'status.json')
        if status['status']!='COMPLETE' or status['plan_sha256']!=sha(root/'plan.json'):raise ValueError('Incomplete or mismatched worker')
        m=('canonical','broad').index(worker['arm']);receipts[str(path/'status.json')]=sha(path/'status.json')
        for index in worker['entry_indices']:
            entry=plan['entries'][index];i=lookup[entry['initial_id']];v=view_lookup[entry['view_id']]
            meta=read(path/f'cell-{index:04d}.json');file=path/f'cell-{index:04d}.npz'
            if sha(file)!=meta['arrays_sha256']:raise ValueError('Replay array changed')
            with np.load(file) as z:
                scores[m,:,i,v]=z['scores'];delta=z['actions'][1]-z['actions'][0]
            np.testing.assert_allclose(scores[m,m,i,v],old[m,i,v],atol=plan.get('runtime_replay_gate',{}).get('score_atol',1e-6),rtol=0)
            controls.append(dict(model=m,**meta['control']))
            pixel_actions.append(dict(model=m,task=entry['task'],different=entry['different'],
                max_abs=float(np.abs(delta).max()),ee_rms=float(np.sqrt((delta[:,:25,:6]**2).mean()))))
            replayed.add((m,index))
    expected={(m,k) for m in range(2) for k,e in enumerate(plan['entries']) if e['replay']}
    if replayed!=expected:raise ValueError('Missing paired replays')
    mean=scores.mean(-1);old_mean=old.mean(-1);choices=mean.argmin(-1)
    report=dict(scope='Two archived render versions crossed with two frozen models. Identical cells reused; changed cells freshly evaluated both ways. Existing endpoint outcomes only, no new rollouts.',
        counts=plan['counts'],tasks={},model_names=['canonical','broad'],source_names=['canonical_archive','broad_archive'],
        replay_controls=dict(cells=len(controls),max_old_score_abs=max(c['score_max_abs'] for c in controls),
            max_old_action_abs=max(c['action_max_abs'] for c in controls),
            exact_repeat_sentinels=sum('same_input_repeat_action_exact' in c for c in controls),
            all_repeat_sentinels_exact=all(c.get('same_input_repeat_action_exact',True) for c in controls)),
        provenance={str(root/'plan.json'):sha(root/'plan.json'),str(matrix_root/'matrix.npz'):sha(matrix_root/'matrix.npz'),str(Path(__file__)):sha(__file__),**receipts})
    for task in plan['tasks']:
        ti=np.array([i for i,j in enumerate(plan['jobs']) if j['task']==task]);rows=np.arange(len(ti));item=dict(models={})
        for m,name in enumerate(report['model_names']):
            first,second=mean[m,0,ti],mean[m,1,ti]
            d=np.abs(scores[m,1,ti]-scores[m,0,ti]);pix=[x for x in pixel_actions if x['model']==m and x['task']==task and x['different']]
            a0,a1=choices[m,0,ti],choices[m,1,ti]
            qq=q[m,ti]
            item['models'][name]=dict(pixel_score_abs=dict(mean=float(d.mean()),p99=float(np.quantile(d,.99)),maximum=float(d.max())),
                mean8_score_abs_change=dict(mean=float(np.abs(second-first).mean()),
                    p99=float(np.quantile(np.abs(second-first),.99)),maximum=float(np.abs(second-first).max())),
                pixel_action_max_abs=max((x['max_abs'] for x in pix),default=0),
                pixel_ee_rms_max=max((x['ee_rms'] for x in pix),default=0),
                initial_score_rank_rho=[correlation(x,y) for x,y in zip(first,second)],
                spatial_score_rank_rho=correlation(first[:,1:].mean(0),second[:,1:].mean(0)),
                winner_changes=int(np.sum(a0!=a1)),initials=len(ti),
                chosen_success_by_image_source=[float(qq[rows,c].mean()) for c in (a0,a1)],
                spatial_value_rho_by_image_source=[correlation(-x[:,1:].mean(0),qq[:,1:].mean(0)) for x in (first,second)],
                old_vs_native_replay_winner_changes=int(np.sum(old_mean[m,ti].argmin(-1)!=choices[m,m,ti])))
        item['crossover_by_common_image_source']={}
        for source,source_name in enumerate(report['source_names']):
            item['crossover_by_common_image_source'][source_name]=[[float(q[e,ti][rows,choices[s,source,ti]].mean()) for e in range(2)] for s in range(2)]
        item['same_image_cross_model_score_rms']=float(np.sqrt(np.square(mean[0,0,ti]-mean[1,0,ti]).mean()))
        item['pixel_mean_score_change_rms_by_model']=[float(np.sqrt(np.square(mean[m,1,ti]-mean[m,0,ti]).mean())) for m in range(2)]
        different_initials={e['initial_id'] for e in plan['entries'] if e['task']==task and e['different']}
        identical=[i for i in ti if initials[i] not in different_initials]
        item['fully_identical_initials']=len(identical)
        if identical:
            ii=np.array(identical);rr=np.arange(len(ii));item['fully_identical_crossover']=[[float(q[e,ii][rr,choices[s,0,ii]].mean()) for e in range(2)] for s in range(2)]
        differences=[e['pixel_difference'] for e in plan['entries'] if e['task']==task]
        item['pixel_difference_maxima']={k:dict(max_abs=max(x[k]['max_abs'] for x in differences),
            max_mean_abs=max(x[k]['mean_abs'] for x in differences)) for k in differences[0]}
        report['tasks'][task]=item
    write(root/'summary.json',report);np.savez_compressed(root/'scores-crossed.npz',scores=scores,old_scores=old,success=q,initial_ids=np.array(initials),views=np.array(views))
    print(report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--matrix-root',type=Path,required=True);a=p.parse_args();summarize(a.root,a.matrix_root)
