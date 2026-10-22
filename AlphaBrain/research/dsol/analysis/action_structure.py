"""Audit fixed historical winners and window switches using saved arrays only."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from ..data.artifacts import read, write, sha
from ..metrics.action_structure import action_structure, outcome_label, ordinal_concordance, FEATURES
from ..metrics.flow_structure import structure
from .flow_structure import checked_npz, choices
from .channel_phase import baseline_indices

PANELS = ('libero_canonical', 'libero_broad', 'plus_canonical', 'plus_broad')
CHANNELS = ('translation', 'rotation', 'gripper')
SCOPES = dict(near=5, full=10)


def clean(x):
    if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def summarize_pairs(rows):
    """Stratify before comparison; do not pool tasks/noises into fake trials."""
    groups = []
    for panel in PANELS:
        for mode in ('include', 'exclude'):
            for start,end in ((1,3),(3,5)):
                rr = [r for r in rows if (r['panel'],r['mode'],r['start'],r['end']) == (panel,mode,start,end)]
                changed = [r for r in rr if r['changed']]
                features = {}
                for key in rr[0]['added_delta']:
                    groups_by_label = {lab:[r['added_delta'][key] for r in changed if r['outcome']==lab and r['added_delta'][key] is not None] for lab in ('improved','degraded','equal','unknown')}
                    pos,neg = groups_by_label['improved'],groups_by_label['degraded']
                    leave_task = {}
                    for task in sorted({r['task'] for r in changed}):
                        leave_task[task] = ordinal_concordance(
                            [r['added_delta'][key] for r in changed if r['task']!=task and r['outcome']=='improved' and r['added_delta'][key] is not None],
                            [r['added_delta'][key] for r in changed if r['task']!=task and r['outcome']=='degraded' and r['added_delta'][key] is not None])
                    features[key] = dict(values=groups_by_label,concordance=ordinal_concordance(pos,neg),
                        ranges_overlap=(max(min(pos),min(neg)) <= min(max(pos),max(neg))) if pos and neg else None,
                        leave_one_task_out=leave_task)
                groups.append(dict(panel=panel,mode=mode,start=start,end=end,n=len(rr),changed=len(changed),
                    outcomes=dict(Counter(r['outcome'] for r in rr)),changed_outcomes=dict(Counter(r['outcome'] for r in changed)),
                    features=features,per_task={t:dict(Counter(r['outcome'] for r in changed if r['task']==t)) for t in sorted({r['task'] for r in rr})}))
    return groups


def run(config_path):
    cfg=read(config_path);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    write(out/'frozen-plan.json',cfg);sources={}
    def verify(p,digest=None):
        p=str(p);actual=sha(p)
        if digest is not None and actual!=digest: raise ValueError('Source changed: '+p)
        sources[p]=actual
        return p
    for p,h in cfg['input_sha256'].items(): verify(p,h)
    inventory=read(cfg['inventory'])['records'];previous=read(cfg['previous_sources'])
    golden_sources=read(cfg['structure_sources']);root=Path(cfg['structure_root'])
    cases=[];pairs=[];selections=[];coverage=[]
    for item in sorted(inventory,key=lambda r:(r['panel'],r['initial'])):
        panel,index=item['panel'],item['initial'];p=Path(item['path'])
        npz=root/panel/f'{index:03d}.npz';jp=npz.with_suffix('.json')
        verify(npz,previous[str(npz)]);meta=read(verify(jp,previous[str(jp)]))
        with checked_npz(p,golden_sources[str(p)]) as z:
            actions=z['actions'].copy();v=z['velocities'].copy();scores=z['scores'].copy();views=z['views'].tolist()
        sources[str(p)]=golden_sources[str(p)]
        recomputed=structure(v,'libero')
        with np.load(npz,allow_pickle=False) as old:
            if not np.array_equal(scores,old['baseline__score']) or views!=old['views'].tolist():raise ValueError('Original score/order mismatch')
            for k in recomputed:
                if not np.array_equal(recomputed[k],old[k],equal_nan=True):raise ValueError('Original structure mismatch: '+k)
            y=old['outcomes'].copy()
        f={scope+'_'+k:value for scope,h in SCOPES.items() for k,value in action_structure(actions,h).items()}
        channel={c:recomputed['baseline_'+c+'__accel_contribution'] for c in CHANNELS}
        if not np.allclose(sum(channel.values()),scores,atol=1e-12,rtol=1e-11):raise ValueError('Attribution mismatch')
        labels=[float(row.mean()) if np.isfinite(row).all() else None for row in y]
        base_ids=baseline_indices(panel,views,'disturbance_uniform')
        base_success=float(y[base_ids].mean()) if np.isfinite(y[base_ids]).all() else None
        common=dict(panel=panel,initial=index,task=meta['task'],split=meta['split'])
        ids_by_mode={};selected=set()
        for mode in ('include','exclude'):
            ids=choices(scores,mode=='exclude');ids_by_mode[mode]=ids;selected.update(ids)
            if [views[i] for i in ids]!=[r['view'] for r in meta['selection'][mode]['original']]:raise ValueError('Winner changed')
            for w,idx in zip((1,3,5),ids):
                delta=None if base_success is None or labels[idx] is None else labels[idx]-base_success
                selections.append(dict(**common,mode=mode,window=w,view=views[idx],success=labels[idx],baseline_success=base_success,
                    gain=delta,outcome=outcome_label(delta),features={k:float(x[idx,:,:w].mean()) for k,x in f.items()},
                    channel={c:float(x[idx,:,:w].mean()) for c,x in channel.items()}))
            for wi,(start,end) in enumerate(((1,3),(3,5))):
                a,b=ids[wi:wi+2];delta=None if labels[a] is None or labels[b] is None else labels[b]-labels[a]
                diff={k:x[b]-x[a] for k,x in f.items()}
                pairs.append(dict(**common,mode=mode,start=start,end=end,old_view=views[a],new_view=views[b],
                    old_success=labels[a],new_success=labels[b],success_delta=delta,outcome=outcome_label(delta),changed=a!=b,
                    added_delta={k:float(x[:,start:end].mean()) for k,x in diff.items()},
                    early_delta={k:float(x[:,:start].mean()) for k,x in diff.items()},
                    score_gap_by_stage=(scores[b]-scores[a]).mean(0).tolist(),
                    channel_gap_by_stage={c:(x[b]-x[a]).mean(0).tolist() for c,x in channel.items()},
                    feature_delta_by_stage={k:x.mean(0).tolist() for k,x in diff.items()}))
        candidates=[]
        for idx in sorted(selected):
            # All repeat curves retained; no best-noise selection.
            a=np.clip(actions[idx],-1,1)
            vv=v[idx].astype(float)
            energy=np.sum(np.diff(vv,axis=2)**2,axis=(-2,-1))
            candidates.append(dict(view=views[idx],success=labels[idx],scores=scores[idx],channels={c:x[idx] for c,x in channel.items()},
                features={k:x[idx] for k,x in f.items()},gripper=a[...,6],rotation=np.linalg.norm(a[...,3:6],axis=-1),
                rotation_xyz=a[...,3:6],denoise_change_norm=np.sqrt(energy),
                denoise_channel_energy={c:np.sum(np.diff(vv[...,sl],axis=2)**2,axis=(-2,-1)) for c,sl in [('translation',slice(0,3)),('rotation',slice(3,6)),('gripper',slice(6,7))]}))
        case=dict(**common,views=views,labels=labels,baseline_success=base_success,
            selections={m:[views[i] for i in ids] for m,ids in ids_by_mode.items()},candidates=candidates)
        filename=f'{panel}-{index:02d}.json';write(out/'cases'/filename,clean(case))
        cases.append(dict(**common,file=filename,candidates=len(candidates)))
        coverage.append(dict(**common,candidates=len(views),known_labels=sum(x is not None for x in labels),
            score_nodes=int(scores.size),raw_shape=list(actions.shape),velocity_shape=list(v.shape),
            selected_candidates=len(selected),original_exact=True,clip_fraction=float((np.abs(actions)>1).mean()),
            invalid_features={k:int((~np.isfinite(x)).sum()) for k,x in f.items()}))
        np.savez_compressed(out/'cases'/f'{panel}-{index:02d}-features.npz',**f,views=np.asarray(views))
        if (index+1)%8==0:print(panel,index+1,flush=True)
    pairs=clean(pairs);selections=clean(selections)
    write(out/'transitions.json',pairs);write(out/'selections.json',selections);write(out/'cases.json',cases)
    write(out/'coverage.json',coverage);write(out/'groups.json',summarize_pairs(pairs));write(out/'sources.json',sources)
    write(out/'summary.json',dict(status='COMPLETE',records=len(cases),candidates=sum(r['candidates'] for r in coverage),
        labeled_candidates=sum(r['known_labels'] for r in coverage),score_nodes=sum(r['score_nodes'] for r in coverage),
        selections=len(selections),transitions=len(pairs),changed_transitions=sum(r['changed'] for r in pairs),
        original_scores_features_and_selections_exact=True,new_model_calls=0,new_actions=0,new_simulation_steps=0,
        source_sha256=sha(__file__),sources_sha256=sha(out/'sources.json')))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    print(run(parser.parse_args().config))
