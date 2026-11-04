"""CPU-only diagnostic builder over a closed, hash-verified target campaign."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.analysis.target_search_diagnostic import diagnose, training_diagnostic, spatial_diagnostic
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def compare_pi05(cases, report, output):
    """Match initial IDs AND angles; initialization protocols may still differ."""
    from AlphaBrain.research.dsol.analysis.target_search_diagnostic import pose_key
    report=Path(report);states={c['state_key'] for c in cases}
    rows=[x for x in csv.DictReader((report/'baseline-spatial.csv').open()) if x['state_key'] in states]
    indexed={pose_key(x['state_key'],[float(x['yaw_deg']),float(x['pitch_deg'])]):x for x in rows}
    comparison={}
    for method in ('stay','reference'):
        target=[c for c in cases if c['method']==method]
        pairs=[(c,indexed[pose_key(c['state_key'],c['selected_angles'])]) for c in target]
        comparison[method]=dict(states=len({c['state_key'] for c,_ in pairs}),episodes=2*len(pairs),
            pi05_success=float(np.mean([int(r['success_repeats'])/int(r['execution_repeats']) for _,r in pairs])),
            fastwam_success=float(np.mean([e['success'] for c,_ in pairs for e in c['executions']])))
    result=dict(scope='Same official initial-state IDs and angular positions; post-settling physical equality is NOT established. Model-specific initialization, noise, sampler and scoring differ. Not a model-only causal comparison.',
        matched=comparison,pi05_source=str(report),pi05_panel=rows,
        source_sha256=hashlib.sha256((report/'baseline-spatial.csv').read_bytes()).hexdigest(),
        excluded='Legacy PPO recovery panel is not complete on these state/start pairs; do not mix its denominator.')
    write(Path(output)/'matched-pi05-comparison.json',result)
    return result


def build(root, output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    repair=root/'evaluation-repair'
    assert read(repair/'completion.json')['status']=='COMPLETE'
    provenance={};cases=[];seen=set()
    recovery=read(repair/'explicit-continuation.json')
    old_cases={}
    for rec in recovery['records'].values():
        p=Path(rec['case'])
        if not (p/'search.json').exists():continue
        s=read(p/'search.json')
        old_cases[(s['method'],s['state_key'],s['start'])]=p
    for sub,phase in [('evaluation','endpoints'),('baseline-untouched','baseline'),
                       ('continuation-source-rebound','baseline'),('endpoints-next','endpoints')]:
        for block in sorted((repair/sub/'execution'/f'{phase}-real/results').glob('block-*')):
            complete=read(block/'complete.json');assert complete['status']=='COMPLETE'
            def checked(p):
                stamp=hashlib.sha256(p.read_bytes()).hexdigest()
                if p.is_relative_to(block):
                    assert complete['artifacts'][str(p.relative_to(block))]==stamp,str(p)
                else:
                    assert recovery['artifacts'][str(p)]==stamp,str(p)
                provenance[str(p)]=stamp
            checked(block/'results.json');result=read(block/'results.json')
            for i,s in enumerate(result['searches']):
                identity=(s['method'],s['state_key'],s['start'])
                if identity in seen:raise ValueError('Duplicate scientific search')
                seen.add(identity);folder=block/f'case-{i:04d}'
                executions=[t for t in result['tasks'] if (t['method'],t['state_key'],t['start'])==identity]
                assert sorted(t['repeat'] for t in executions)==[0,1]
                qrows=[]
                for event in s['events']:
                    if 'query' not in event:continue
                    name=f"query-{event['query']:02d}.npz";p=folder/name
                    if not p.exists():p=old_cases[identity]/name
                    checked(p)
                    with np.load(p,allow_pickle=False) as z:
                        values=[]
                        for j in (0,1):
                            scores=compute_accel_scores(z[f'trace_{j}'][None],prefixes=[3,9])
                            assert not any(scores[k]['degenerate'][0] for k in (3,9))
                            a=float(scores[3]['score'][0]);assert abs(a-float(z[f'score_{j}']))<1e-12
                            values.append((a,float(scores[9]['score'][0])))
                    qrows.append(dict(query=event['query'],kind=event['kind'],angles=event['angles'],
                                      p3=[v[0] for v in values],p9=[v[1] for v in values],path=str(p)))
                if s['method'] not in ('stay','reference'):assert len(qrows)==s['queries']
                cases.append(dict(method=s['method'],state_key=s['state_key'],task=s['task'],
                    start=s['start'],angles_deg=s['angles_deg'],selected_angles=s['selected_angles'],
                    queries=s['queries'],reason=s['reason'],queries_data=qrows,
                    selected_input_sha256=s['selected_input_sha256'],
                    executions=[dict(repeat=e['repeat'],success=e['success']) for e in executions]))
    assert len(cases)*2==2336
    episodes=[];updates=[]
    for run in sorted((root/'training/runs').iterdir()):
        for p in sorted((run/'episodes').glob('*.json')):
            provenance[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            episodes.append(dict(read(p)['metrics'],run=run.name))
        for p in sorted((run/'updates').glob('*.json')):
            if '-retention' in p.name:continue
            provenance[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            updates.append(dict(read(p),run=run.name))
    assert len(episodes)==768 and len(updates)==96
    training=training_diagnostic(episodes,updates)
    write(output/'cases.json',cases);write(output/'training-diagnostic.json',training)
    write(output/'diagnostic.json',diagnose(cases));write(output/'input-manifest.json',provenance)
    write(output/'spatial-diagnostic.json',spatial_diagnostic(cases))
    write(output/'status.json',dict(status='CPU_ANALYSIS_COMPLETE',cases=len(cases),
        episodes=2336,training_episodes=len(episodes),model_calls=0))
    return cases,training


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--font',required=True);p.add_argument('--pi05-report');a=p.parse_args()
    cases,training=build(a.root,a.output)
    from AlphaBrain.research.dsol.plotting.target_search_diagnostic import figures
    figures(cases,training,a.output,a.font)
    if a.pi05_report:
        comparison=compare_pi05(cases,a.pi05_report,a.output)
        from AlphaBrain.research.dsol.plotting.target_search_diagnostic import model_comparison_figures
        model_comparison_figures(cases,comparison,a.output,a.font)


if __name__=='__main__':main()
