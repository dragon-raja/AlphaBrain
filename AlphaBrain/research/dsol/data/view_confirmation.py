"""Prepare bounded fresh-noise confirmation targets; never launch workers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .artifacts import read, write, sha


def frozen_targets(outcomes, accel, splits, initials, views, repeats):
    """Freeze four controls and deduplicate identical initial/view trials.

    The empirical winner intentionally uses old execution labels; its fresh
    evaluation tests repeatability, not a deployable label-free method.
    """
    y, a, splits = np.asarray(outcomes), np.asarray(accel), np.asarray(splits)
    if y.ndim != 3 or y.shape[:2] != a.shape or a.shape != (len(initials), len(views)):
        raise ValueError('Input axes differ')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(a).all():
        raise ValueError('Invalid source values')
    if splits.shape != (len(initials),) or not np.isin(splits, ['pretrain','target']).all() or not (splits=='pretrain').any():
        raise ValueError('Missing/invalid seen split')
    if views[0]!='canonical' or len(set(initials))!=len(initials) or len(set(views))!=len(views):
        raise ValueError('Invalid catalog identities')
    if not repeats or len(set(repeats))!=len(repeats) or any(type(r) is not int or r < y.shape[-1] for r in repeats):
        raise ValueError('New execution addresses must not overlap original repeats')
    q=y.mean(-1)
    choices=dict(canonical=np.zeros(len(initials),int), minimum_accel=a.argmin(1),
        seen_fixed=np.full(len(initials),q[splits=='pretrain'].mean(0).argmax(),int),
        old_empirical_winner=q.argmax(1))
    selected=[]; trials=[]
    for i,initial in enumerate(initials):
        aliases={}
        for method,index in choices.items(): aliases.setdefault(int(index[i]),[]).append(method)
        selected.append(dict(initial=initial, split=str(splits[i]),
            methods={name:views[int(index[i])] for name,index in choices.items()}))
        for view_index,methods in sorted(aliases.items()):
            for repeat in repeats:
                identity=['xr1_paired_v1', initial, views[view_index], 'execution',repeat]
                trial_id=hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest()
                trials.append(dict(trial_id=trial_id, initial=initial, view=views[view_index],
                    repeat=repeat, methods=methods))
    return selected,trials


def prepare(config_path):
    cfg=read(config_path);root=Path(cfg['output'])
    root.mkdir(parents=True,exist_ok=False)
    source=Path(cfg['analysis'])/'panel_arrays.npz';selection=Path(cfg['panel'])/'selection.json'
    jobs={r['initial_id']:r for r in read(selection)['jobs']}
    with np.load(source,allow_pickle=False) as z:
        model=z['models'].tolist().index('xr1')
        initials,views=z['initials'].tolist(),z['views'].tolist()
        selected,trials=frozen_targets(z['success'][model],z['accel'][model],
            [jobs[i]['split'] for i in initials],initials,views,cfg['execution_repeats'])
    if len(initials)!=40 or len(views)!=97: raise ValueError('Unexpected original panel')
    shutil.copy2(Path(__file__).resolve(),root/'preparation-source.py')
    protocol=dict(status='DRAFT_NOT_LAUNCHED_REQUIRES_PROTOCOL_AND_RUNTIME_ADMISSION',
        sources={str(p):sha(p) for p in (source,selection,Path(config_path))},
        preparation_sha256=sha(root/'preparation-source.py'), model='xr1',
        initial_count=len(initials), source_catalog_count=len(views),
        execution_repeats=cfg['execution_repeats'], trials=trials, selections=selected,
        planned_rollouts=len(trials), undeduplicated_rollouts=len(initials)*4*len(cfg['execution_repeats']),
        scoring='Reuse old eight-draw scores; do not reselect from new outcomes',
        primary_comparisons=['old_empirical_winner - canonical','minimum_accel - canonical','seen_fixed - canonical'],
        reporting='All/seen/heldout and all ten tasks; old seen-fixed fitting scores are not independent performance',
        limitations=['New noise on old initials, not new-scene or unseen-task generalization',
            'No new random-perturbation average; do not subtract old stay outcomes as fresh paired confirmation',
            'Old empirical winner is outcome-informed; new result is not the full-space oracle',
            'Configuration change, not physically executed camera search'],
        execution_gate='Standard queue backend admission required; this file cannot launch GPU work')
    write(root/'protocol.json',protocol)
    return protocol


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    prepare(parser.parse_args().config)
