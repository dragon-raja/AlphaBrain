"""Audit native-window results and reuse absolute97 recovery endpoints.

This consumes frozen short probes and independently-noised full task outcomes;
it neither runs simulations nor selects a preferred window after seeing labels.
"""
from pathlib import Path
import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from .plus_window_queue import validate_block
from .plus_evaluation_queue import validate_result


def choices(scores, *, include_canonical):
    if scores.ndim != 3 or scores.shape[1:] != (8, 5) or not np.isfinite(scores).all():
        raise ValueError('Expected candidate x8 score members x5 windows')
    start = 0 if include_canonical else 1
    prefixes = np.cumsum(scores.mean(axis=1), axis=1)/np.arange(1, 6)
    return np.argmin(prefixes[start:], axis=0)+start


def audit_and_analyze(config_file):
    cfg = read(config_file); worker = read(cfg['worker_config']); root = Path(worker['root'])
    production = read(cfg['full_config']); full_root = root/'plus-evaluation'
    original = read(worker['inputs'])['records']; rows = []; receipts = []
    if sha(cfg['old_matrix']) != cfg['old_matrix_sha256']:
        raise ValueError('Original97 outcome matrix changed')
    with np.load(cfg['old_matrix'], allow_pickle=False) as z:
        success = z['success'].copy(); candidate_ids = z['candidate_ids'].copy()
        accel = z['accel_members'].copy()
    if success.shape != (2,32,97,32) or accel.shape != (2,32,97,8):
        raise ValueError('Original97 dimensions changed')
    for mi, model in enumerate(('canonical', 'broad')):
        for initial in range(32):
            native = root/'plus-probes'/model/f'{initial:02d}'
            validate_block(cfg['worker_config'], model, initial, native)
            full = full_root/model/f'{initial:02d}'
            validate_result(cfg['full_config'], initial, model, full)
            custom = root/'probes'/model/f'{initial:02d}'
            custom_config = cfg['libero_worker_config']
            if cfg.get('libero_ownership'):
                item = next(r for r in read(cfg['libero_ownership'])['records'] if r['model']==model and r['initial']==initial)
                custom = Path(item['output']); custom_config = item['config']
            from .window_queue import validate_block as validate_custom
            validate_custom(custom_config, model, initial, custom)
            with np.load(native/'probes.npz', allow_pickle=False) as z:
                scores = z['scores'].copy(); native_views = z['views'].copy()
            outcomes = np.full((len(native_views),4), np.nan)
            outcomes[0] = success[mi,initial,0,:4]
            with np.load(full/'probes.npz', allow_pickle=False) as z:
                if not np.array_equal(scores[:,:,0],z['scores']) or not np.array_equal(native_views,z['views']):
                    raise ValueError('Native window1 != original Plus initial scores')
                outcomes[z['execution_view_indices']] = z['success']
            if initial == 0:
                admission = Path(production['admission_root'])/model/'00'
                validate_result(production['admission_config'],initial,model,admission)
                with np.load(admission/'probes.npz',allow_pickle=False) as z:
                    for index, view in enumerate(z['execution_view_indices']):
                        if view in production['reused_rollout_indices'].get('0',[]):
                            if np.isfinite(outcomes[view]).any():
                                raise ValueError('Duplicate native outcome ownership')
                            outcomes[view] = z['success'][index]
            if not np.isfinite(outcomes).all():
                raise ValueError('Native outcome coverage incomplete')
            with np.load(custom/'probes.npz',allow_pickle=False) as z:
                custom_scores=z['scores'].copy()
                if not np.array_equal(z['views'],candidate_ids) or not np.array_equal(custom_scores[:,:,0],accel[mi,initial]):
                    raise ValueError('Custom97 original window1 changed')
            row = dict(model=model,initial=initial,task=original[initial]['state']['task_id'],
                native_perturb_no_adjust=float(outcomes[1:].mean()),
                canonical=float(outcomes[0].mean()))
            for allow in (False,True):
                suffix='with_canonical' if allow else 'without_canonical'
                selected=choices(scores,include_canonical=allow)
                selected97=choices(custom_scores,include_canonical=allow)
                row['native_'+suffix]=dict(views=native_views[selected].tolist(), success=outcomes[selected].mean(axis=1).tolist())
                row['custom97_'+suffix]=dict(views=candidate_ids[selected97].tolist(),
                    success=success[mi,initial,selected97,:4].mean(axis=1).tolist())
            rows.append(row)
            receipts.append(dict(model=model,initial=initial,native=sha(native/'status.json'),
                full=sha(full/'status.json'),custom=sha(custom/'status.json')))
    aggregate={}
    for model in ('canonical','broad'):
        subset=[r for r in rows if r['model']==model]
        # Exactly four initials per task, but keep equal-task weighting explicit.
        tasks=sorted({r['task'] for r in subset})
        metrics=['native_perturb_no_adjust','canonical','native_with_canonical','native_without_canonical',
                 'custom97_with_canonical','custom97_without_canonical']
        per_task={}
        for task in tasks:
            group=[r for r in subset if r['task']==task]
            if len(group)!=4: raise ValueError('Expected four physical initials per task')
            per_task[task]={key: np.mean([r[key]['success'] if isinstance(r[key],dict) else r[key] for r in group],axis=0).tolist()
                            for key in metrics}
        aggregate[model]=dict(per_task=per_task,overall={key:np.mean([v[key] for v in per_task.values()],axis=0).tolist() for key in metrics})
    result=dict(status='PASS',window1_native_original_exact=True,window1_custom_original_exact=True,
        score_repeats=8,execution_repeats=4,windows=[1,2,3,4,5],models=aggregate,rows=rows,receipts=receipts,
        scope='Same8tasks/32initials; official camera distribution and absolute97 endpoints. Offline probe-reset, not physical online search. All windows reported; no best-window confirmation claim.',
        endpoint_scope='All32initials x97 cached endpoints checked from first official start per initial; all1364 official starts have prior physical/wrist receipts. No complete rollout rerun.',
        old_matrix_sha256=cfg['old_matrix_sha256'],config_sha256=sha(config_file))
    destination=root/'analysis'/'plus-windows-and-custom97.json'
    if destination.exists(): raise ValueError('Refuse to overwrite published extension analysis')
    write(destination,result)
    return result
