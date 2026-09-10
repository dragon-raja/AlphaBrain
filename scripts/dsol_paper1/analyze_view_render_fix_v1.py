#!/usr/bin/env python3
"""Strict, full-length acceptance of the opt-in no-AA diagnostic protocol."""
import hashlib
import json
from pathlib import Path
import numpy as np
import run_full64_view_landscape_v2 as control
from analyze_view_repeatability_traces_v1 import compare, difference

ROOT = control.ROOT/'repeatability-root-cause-v1/render-fix-window-v1'

def main():
    control.require((ROOT/'noaa-validation-complete.json').exists(), 'Validation driver is incomplete')
    report = {'utc':control.stamp(), 'status':'PASS_BOUNDED_NOAA_REPEATABILITY',
        'formal_protocol_changed':False, 'old_render_protocol_equivalence_claimed':False,
        'scope':'Two previously divergent task-state-view-noise keys, two render GPUs, identical canonical checkpoint',
        'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'tasks':{}, 'static':{}}
    for task in ('goal_wine_rack','libero10_mug_microwave'):
        paths=[ROOT/f'{task}-noaa-gpu{gpu}' for gpu in (0,1)]
        for path in paths:
            control.require((path/'completion.json').exists(), 'Trace is incomplete')
            protocol=control.read_json(path/'render-protocol.json')
            control.require(protocol['offsamples'] and set(protocol['offsamples'])=={0}, 'Render protocol mismatch')
        result=compare(*paths)
        result['episodes']=[control.read_json(path/'episode.json') for path in paths]
        result['repeated_request_comparisons']=sum(len([p for p in c['repeated_request_comparisons'] if p['probe']>0])
            for path in paths for c in control.read_json(path/'calls.json'))
        result['repeated_request_failures']=sum(not p['exact'] for path in paths for c in control.read_json(path/'calls.json')
            for p in c['repeated_request_comparisons'] if p['probe']>0)
        result['pass']=all(v['lengths'][0]==v['lengths'][1] and v['lengths'][0]>0 and not v['first_differences']
            for v in (result['call'],result['step'])) and result['repeated_request_failures']==0
        result['pass']=result['pass'] and result['episodes'][0]['success']==result['episodes'][1]['success']
        report['tasks'][task]=result
    for task, labels in {
        'goal_wine_rack':['noaa300-goal_wine_rack-gpu0','static-noaa-gpu1'],
        'libero10_mug_microwave':['noaa300-libero10_mug_microwave-gpu0','noaa300-libero10_mug_microwave-gpu1']}.items():
        receipts=[control.read_json(ROOT/label/'completion.json') for label in labels]
        hashes=[r['results']['baseline']['sha256'] for r in receipts]
        report['static'][task]={'renders':sum(map(len,hashes)), 'cross_gpu_unique_images':len(set(h for values in hashes for h in values)),
            'sources':labels, 'pass':len(set(h for values in hashes for h in values))==1}
    with np.load(ROOT/'static-flags-per-render-gpu0/baseline.npz') as old, np.load(ROOT/'noaa300-goal_wine_rack-gpu0/baseline.npz') as new:
        report['observation_protocol_change_example']=difference(old['images'][0],new['images'][0])
    if not all(r['pass'] for r in list(report['tasks'].values())+list(report['static'].values())):
        report['status']='FAIL_BOUNDED_NOAA_REPEATABILITY'
    control.write_new_json(ROOT/'noaa-validation-analysis-v1.json',report)
    print(json.dumps({'status':report['status'],'tasks':{t:{'pass':r['pass'],'calls':r['call']['lengths'],'steps':r['step']['lengths']} for t,r in report['tasks'].items()},'static':report['static']}))
    control.require(report['status']=='PASS_BOUNDED_NOAA_REPEATABILITY', 'Repeatability validation failed')

if __name__=='__main__': main()
