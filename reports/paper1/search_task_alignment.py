"""Audit complete A8/B8 receipts against original replay; export static figures."""
import argparse
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.analysis.search_task_alignment import task_rows,paired_cluster_interval,cutoff_pair,METHODS
from AlphaBrain.research.dsol.data.artifacts import read,sha,write
from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank


def build(root,output):
    import json
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False)
    if read(root/'pipeline-status.json')['status']!='COMPLETE':raise ValueError('Full evaluation not complete')
    configs={c:read(root/f'config-{c}.json') for c in ('A','B')}
    replay_path=Path(configs['A']['training'])/'evaluation.jsonl'
    replay={}
    with replay_path.open() as stream:
        for line in stream:
            r=json.loads(line)
            if r['budget']==8 and r['seed']==0 and r['method'] in METHODS:
                key=(r['initial'],r['start'],r['method'])
                if key in replay:raise ValueError('Duplicate archived search identity')
                replay[key]=r
    with np.load(configs['A']['outcome_matrix'],allow_pickle=False) as z:
        labels=z['success'][0].copy();steps=z['completion_steps'][0].copy()
    states=read(Path(configs['A']['outcome_matrix']).parent/'states.json')
    banks={c:ExplicitFlowNoiseBank(Path(configs[c]['execution_noise_manifest' if c=='A' else 'domain_execution_noise_manifest']),verify_file=True)
           for c in ('A','B')}
    expected_noise={};rows=[];nominal=[];searches=[];receipts=[]
    for path in sorted(list((root/'first-block').glob('*/*/status.json'))+list((root/'remaining').glob('*/*/status.json'))):
        d=read(path)
        if d['status']!='COMPLETE':raise ValueError('Incomplete block')
        receipts.append(dict(path=str(path),sha256=sha(path)))
        for s in d['searches']:
            if s['cohort'] not in ('A','B'):continue
            entry={k:s[k] for k in ('cohort','initial','start','method','selected','queries','visited','path')}
            if s['cohort']=='A':
                old=replay[s['initial'],s['start'],s['method']]
                entry.update(replay_selected=old['selected_candidate'],
                    selection_exact=s['selected']==old['selected_candidate'],
                    queries_exact=s['queries']==old['queries'],path_exact=s['path']==old['path'],
                    replay32_success=old['success'])
            searches.append(entry)
        for r in d['tasks']:
            c=r['cohort'];bank=banks['A' if c=='A' else 'B']
            noise_key=(c,r['state_key'],r['repeat'])
            if noise_key not in expected_noise:
                expected_noise[noise_key]=[bank.get(r['state_key'],r['repeat'],j)['noise_sha256'] for j in range(60)]
            if r['execution_noise_sha256']!=expected_noise[noise_key][:r['replans']]:
                raise ValueError('Execution-noise address mismatch')
            row={k:r[k] for k in ('cohort','initial','start','method','repeat','success','selected','queries','steps')}
            row['task']=r['state_key'].rsplit('::',1)[0].split('::')[-1]
            row['replay_success']=None
            if c=='A':
                if states[r['initial']]['pair_key']!=r['state_key']:raise ValueError('Matrix initial identity mismatch')
                old=replay[r['initial'],r['start'],r['method']]
                value=int(labels[r['initial'],old['selected_candidate'],r['repeat']])
                row['replay_success']=bool(value) if value in (0,1) else None
                row['replay_completion_steps']=int(steps[r['initial'],old['selected_candidate'],r['repeat']])
                row['replay_selected']=old['selected_candidate']
            (nominal if c=='capability' else rows).append(row)
    if len(rows)!=15360 or len(nominal)!=128:raise ValueError('Full population count mismatch')
    # Explicit frozen support, not merely equal-sized method subsets.
    expected={(c,i,s,m,k) for c in ('A','B') for i in (list(range(16)) if c=='B' else [i for i in range(32) if i%4 in (2,3)])
              for s in (range(65,97,4) if c=='B' else range(65,97)) for m in METHODS for k in range(8)}
    if {(r['cohort'],r['initial'],r['start'],r['method'],r['repeat']) for r in rows}!=expected:
        raise ValueError('Frozen A/B trial population changed')
    table=task_rows(rows)
    summary=dict(status='COMPLETE',table=table,nominal=nominal,
        alignment=dict(A_searches=sum(s['cohort']=='A' for s in searches),
            selected_mismatch=sum(s.get('selection_exact') is False for s in searches),
            query_mismatch=sum(s.get('queries_exact') is False for s in searches),
            path_mismatch=sum(s.get('path_exact') is False for s in searches),
            matched_A_labels=sum(r['cohort']=='A' and r['replay_success'] is not None for r in rows),
            success_after_300=sum(r['cohort']=='A' and r['replay_success'] is True and r['replay_completion_steps']>300 for r in rows),
            B_replay_labels='not available; not counted as failure',execution_noise_exact=True),
        intervals={c:{ref:paired_cluster_interval([r for r in rows if r['cohort']==c],reference=ref)
                      for ref in ('stay','local_accel')} for c in ('A','B')},
        provenance=dict(receipts=receipts,replay_file=str(replay_path),replay_sha256=sha(replay_path),
            matrix_sha256=sha(configs['A']['outcome_matrix'])),
        limitation='Not an independent confirmation: inspected tasks/data; B tasks unseen by RL, not necessarily unseen by VLA')
    write(output/'summary.json',summary);write(output/'paired-rows.json',rows);write(output/'search-alignment.json',searches)
    return summary


def common_horizons(root,output):
    """Add, never replace, a common-horizon sidecar to the raw audit."""
    root,output=Path(root),Path(output)
    config=read(root/'config-A.json')
    states=read(Path(config['outcome_matrix']).parent/'states.json')
    limits={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}
    with np.load(config['outcome_matrix'],allow_pickle=False) as z:
        successes=z['success'][0];steps=z['completion_steps'][0]
        for i,state in enumerate(states):
            failed=steps[i][successes[i]==0]
            if len(failed) and not np.all(failed==limits[state['suite']]):
                raise ValueError('Historical failure-step limits do not match suite protocol')
    rows=[]
    for original in read(output/'paired-rows.json'):
        if original['cohort']!='A':continue
        limit=min(300,limits[states[original['initial']]['suite']])
        rows.append(cutoff_pair(original,limit))
    target=output/'common-horizon.json'
    if target.exists():raise FileExistsError(target)
    result=dict(table=task_rows(rows),limits=limits,formal_horizon=300,
        alignment_scope='Same eight seeds, same selections, cutoff=min(old suite horizon,300); residual differences retained',
        records=len(rows),remaining_disagreements=sum(r['success']!=r['replay_success'] for r in rows))
    write(target,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    summary=build(args.root,args.output);common=common_horizons(args.root,args.output)
    from AlphaBrain.research.dsol.plotting.search_task_alignment import figures
    figures(summary,common,args.output)
