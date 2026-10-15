"""Audit immutable paired policy inputs and plan sparse render-score replay."""
from pathlib import Path
import argparse
from collections import Counter
import numpy as np
from .artifacts import read, write, sha
from .rollout_replay import array_identity
from .evaluation_matrix import flow_noise

IMAGE_KEYS=('observation/image','observation/right_image','observation/wrist_image')


def request_from_cell(cell,metadata):
    with np.load(Path(cell)/'scores.npz',allow_pickle=False) as z:
        request={k:z[k].copy() for k in metadata['input_arrays']}
    if {k:array_identity(v) for k,v in request.items()}!=metadata['input_arrays']:
        raise ValueError('Stored input array identity changed')
    return dict(request,prompt=metadata['prompt'])


def compare_requests(left,right):
    if left['prompt']!=right['prompt'] or not np.array_equal(left['observation/state'],right['observation/state']):
        raise ValueError('Non-image input differs; cannot attribute to rendering')
    metrics={}
    for key in IMAGE_KEYS:
        if left[key].dtype!=np.uint8 or left[key].shape!=right[key].shape:raise ValueError('Expected matched uint8 images')
        delta=np.abs(left[key].astype(np.int16)-right[key].astype(np.int16))
        metrics[key]=dict(max_abs=int(delta.max()),mean_abs=float(delta.mean()),changed_channels=int(np.count_nonzero(delta)))
    return metrics


def build(plan_path,population_path,output,tasks):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plan=read(plan_path);population=read(population_path)
    mapping={(w['arm'],i):w['worker_id'] for w in plan['workers'] for i in w['initials']}
    jobs=[j for j in population['jobs'] if j['split']=='target' and j['task'] in tasks]
    if set(j['task'] for j in jobs)!=set(tasks):raise ValueError('Missing task')
    entries=[];counts=Counter()
    for job in jobs:
        for view in plan['view_ids']:
            cells={arm:str(Path(plan_path).parent/'workers'/mapping[arm,job['initial_id']]/'cells'/job['initial_id']/view) for arm in ('canonical','broad')}
            meta={arm:read(Path(path)/'scores.json') for arm,path in cells.items()}
            for arm,m in meta.items():
                provenance=m['provenance']
                if any(provenance[k]!=v for k,v in dict(arm=arm,initial_id=job['initial_id'],view_id=view,snapshot_manifest_sha256=job['snapshot_manifest_sha256']).items()):
                    raise ValueError('Source provenance mismatch')
                if [row['noise'] for row in m['rows']]!=[array_identity(flow_noise(job['initial_id'],'score',i)) for i in range(8)]:
                    raise ValueError('Noise address mismatch')
            req={arm:request_from_cell(cells[arm],meta[arm]) for arm in cells}
            delta=compare_requests(req['canonical'],req['broad'])
            different=any(d['max_abs'] for d in delta.values())
            if max(d['max_abs'] for d in delta.values())>2:raise ValueError('Not a micro-pixel pair; inspect before inference')
            entries.append(dict(task=job['task'],initial_id=job['initial_id'],view_id=view,cells=cells,
                metadata_sha256={arm:sha(Path(path)/'scores.json') for arm,path in cells.items()},
                arrays_sha256={arm:m['arrays_sha256'] for arm,m in meta.items()},
                different=different,pixel_difference=delta,replay=different or view=='canonical'))
            counts[(job['task'],'different' if different else 'identical')]+=1
        print(job['task'],job['scene_seed'],'audited',flush=True)
    selected=[i for i,e in enumerate(entries) if e['replay']]
    workers=[dict(worker_id=f'{arm}-{shard}',arm=arm,gpu=m*4+shard,entry_indices=selected[shard::4])
        for m,arm in enumerate(('canonical','broad')) for shard in range(4)]
    result=dict(format='render_score_micro_pixel_v1',scope='Same snapshot/view paired archived inputs; all target initials of requested tasks. No simulator, training, downloads or new rollouts.',
        scoring_repeats=8,flow_steps=10,models=plan['models'],tasks=tasks,views=plan['view_ids'],jobs=jobs,entries=entries,workers=workers,
        replay_rule='All pixel-different cells, plus canonical-view sentinel per initial. Exact-identical cells reuse audited old scores. Both image versions evaluated by each model.',
        runtime_replay_gate=dict(score_atol=1e-6,action_atol=1e-5,action_rtol=1e-5),
        counts={t:{k:counts[t,k] for k in ('different','identical')} for t in tasks},
        sources={str(p):sha(p) for p in [plan_path,population_path,Path(__file__)]})
    write(output/'plan.json',result);print(result['counts']);print('Replayed cells per model',len(selected))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-plan',type=Path,required=True);p.add_argument('--population',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',nargs='+',required=True);a=p.parse_args()
    build(a.source_plan,a.population,a.output,a.tasks)
