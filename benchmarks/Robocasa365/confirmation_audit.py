"""CPU-only final acceptance of frozen probes and independent endpoints.

Uses the production endpoint validator without constructing a backend or starting
workers. Original ownership, including pilots, is authoritative after migration.
"""
import argparse
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.analysis.recovery_confirmation import selected_conditions
from AlphaBrain.research.dsol.analysis.short_view_windows import read_cells
from .confirmation_queue import ConfirmationBackend
from .snapshot_bundle import load_snapshot_bundle, snapshot_numeric_fingerprint


def candidate_fingerprint(snapshot, positions, quaternions):
    """The runtime fingerprint includes installation: replace only those fields."""
    return snapshot_numeric_fingerprint(dataclasses.replace(snapshot,
        camera_positions=positions,camera_quaternions=quaternions))


def run(config_file):
    cfg = read(config_file); out = Path(cfg['output']); out.mkdir(exist_ok=False, parents=True)
    started = time.time(); sources = {str(config_file): sha(config_file)}
    def checked(path, digest=None):
        value = sha(path)
        if digest is not None and value != digest:
            raise ValueError('Source changed: ' + str(path))
        sources[str(path)] = value
        return read(path)
    def progress(stage, **kw):
        write(out/'status.json', dict(status='RUNNING', stage=stage, started_unix=started, **kw))
    try:
        arrays = {}; metadata = {}; totals = {}; receipts = []
        for model, entry in cfg['models'].items():
            progress('score_audit', model=model)
            plan = checked(entry['plan'])
            checked(plan['authorization'], plan['authorization_sha256'])
            probe_cfg = checked(plan['probe_config'], plan['probe_config_sha256'])
            panel = checked(probe_cfg['panel_plan'], plan['probe_plan_sha256'])
            reference = checked(panel['reference_plan'], panel['reference_plan_sha256'])
            render = checked(panel['reference_release'], panel['reference_release_sha256'])
            rendered = {r['initial_id']:r for r in render['records']}
            jobs = {j['initial_id']: j for j in reference['jobs']}
            initials, views = plan['initial_ids'], panel['all_views']
            extra = [v['view_id'] for v in reference['catalog']['views'] if v['role']=='extra_eval']
            ownership = checked(Path(entry['probes'])/'ownership.json')['records']
            scores, first, times, score_receipts = read_cells(ownership, initials, views)
            selections = {}
            for i, initial in enumerate(initials):
                ref = plan['selections'][model][initial]
                saved = checked(ref['path'], ref['sha256'])
                rebuilt = selected_conditions(views, extra, scores[i])
                for key in rebuilt:
                    if saved[key] != rebuilt[key]:
                        raise ValueError('Choice differs from frozen scores: '+key)
                for source in saved['sources']:
                    if sha(source['path']) != source['sha256']:
                        raise ValueError('Frozen score evidence changed: '+source['path'])
                if not saved['window1_exact'] or not saved['all_97_candidates_verified']:
                    raise ValueError('Missing first-inference acceptance')
                for j, view in enumerate(views):
                    anchor = saved['anchors'][view]
                    a = checked(anchor['path'], anchor['sha256'])
                    payload = Path(anchor['path']).with_suffix('.npz')
                    if sha(payload) != a['arrays_sha256']:
                        raise ValueError('Anchor arrays changed')
                    with np.load(payload, allow_pickle=False) as z:
                        if not np.array_equal(z['scores'], scores[i,j,:,0]):
                            raise ValueError('Window one differs from independent initial scoring')
                selections[initial] = saved
            write(out/(model+'-probe-audit.json'), dict(status='PASS', cells=len(ownership),
                  branches=len(ownership)*8, records=score_receipts))
            progress('endpoint_audit', model=model)
            owners = []; tasks = []
            for root in entry['endpoints']:
                root = Path(root)
                owner = checked(root/'ownership.json')
                if owner['plan_sha256'] != sha(entry['plan']) or owner['model'] != model:
                    raise ValueError('Ownership bound to another scientific plan')
                owners.extend(owner['records'])
                for name in ('manifest.json','pilot-manifest.json'):
                    manifest = checked(root/name)
                    tasks.extend(manifest['tasks'])
            expected = {(i,v,r) for i in initials for v in selections[i]['evaluation_views'] for r in range(8)}
            actual = [(r['initial'],r['view'],k) for r in owners for k in range(8)]
            addresses = [(a['initial'],a['view'],a['repeat']) for t in tasks for a in t['addresses']]
            if len(actual)!=len(set(actual)) or set(actual)!=expected or len(addresses)!=len(set(addresses)) or set(addresses)!=expected:
                raise ValueError('Missing/duplicate ownership or task addresses')
            source_candidates = [Path(p) for p in entry['worker_sources']]
            source_hashes = {p: (sha(p/'benchmarks/Robocasa365/confirmation_execution.py'),
                              sha(p/'benchmarks/Robocasa365'/('evaluation_matrix.py' if model=='official' else 'xr1_evaluation.py')))
                             for p in source_candidates}
            def validate(task):
                identity = read(Path(task['output'])/'identity.json')
                matched = [p for p,h in source_hashes.items() if h==(identity['source_sha256'],identity['rollout_source_sha256'])]
                if not matched: raise ValueError('Unregistered execution source')
                backend = object.__new__(ConfirmationBackend)
                backend.runtime = dict(model=model, plan=entry['plan'], source=str(matched[0]))
                return backend.validate(task)
            with ThreadPoolExecutor(max_workers=cfg.get('workers',8)) as pool:
                for k, receipt in enumerate(pool.map(validate,tasks)):
                    receipts.append(receipt)
                    if k%40==0: progress('endpoint_audit', model=model, blocks=k+1, total_blocks=len(tasks))
            owner_map = {(r['initial'],r['view']):Path(r['cell']) for r in owners}
            y = np.full((len(initials),len(views),8),np.nan)
            trial_receipts = []
            for i, initial in enumerate(initials):
                checked(Path(jobs[initial]['path'])/'manifest.json',jobs[initial]['snapshot_manifest_sha256'])
                snapshot=load_snapshot_bundle(Path(jobs[initial]['path'])).snapshot
                rr=rendered[initial];cr=Path(rr['candidate_root'])
                cs=checked(cr/'summary.json',rr['candidate_summary_sha256'])
                if sha(cr/'geometry.npz')!=cs['files']['geometry.npz']['sha256']:
                    raise ValueError('Released geometry changed')
                with np.load(cr/'geometry.npz',allow_pickle=False) as geom:
                    geometry_views=geom['view_ids'].tolist()
                    expected_physical={v:candidate_fingerprint(snapshot,
                        geom['camera_positions'][geometry_views.index(v)],geom['camera_quaternions'][geometry_views.index(v)])
                        for v in selections[initial]['evaluation_views']}
                for view in selections[initial]['evaluation_views']:
                    for r in range(8):
                        path = owner_map[initial,view]/f'execution_{r}.json'; row=read(path)
                        if row['initial_physical_sha256']!=expected_physical[view]:
                            raise ValueError('Execution differs from snapshot plus released camera installation')
                        y[i,views.index(view),r] = row['success']
                        trial_receipts.append(dict(path=str(path),sha256=sha(path)))
            if np.isfinite(y).sum()!=plan['rollout_count']: raise ValueError('Endpoint count mismatch')
            metadata[model] = [dict(initial=i,task=jobs[i]['task'],split=jobs[i]['split'],
                layout_id=jobs[i]['layout_id'],style_id=jobs[i]['style_id'],scene_seed=jobs[i]['scene_seed']) for i in initials]
            arrays.update({model+'__scores':scores,model+'__success':y,model+'__first_success':first,
                           model+'__choices':np.array([selections[i]['chosen_indices'] for i in initials]),
                           model+'__initials':np.array(initials)})
            totals[model] = dict(initials=len(initials),probe_cells=len(ownership),
                score_branches=len(ownership)*8,execution_trials=len(expected),blocks=len(tasks),
                window1_exact=True,choices_recomputed=True,paired_execution_noise_verified=True,
                original_initial_restored=True,source_versions=[str(p) for p in source_candidates])
            write(out/(model+'-endpoint-audit.json'),dict(status='PASS',records=trial_receipts))
        np.savez_compressed(out/'accepted-arrays.npz',**arrays,views=views,extra_views=extra)
        write(out/'population.json',metadata)
        write(out/'audit.json',dict(status='PASS',models=totals,source_files=sources,
            arrays_sha256=sha(out/'accepted-arrays.npz'),population_sha256=sha(out/'population.json'),
            validator_sha256=sha(Path(__file__).with_name('confirmation_queue.py')),
            auditor_sha256=sha(__file__),started_unix=started,finished_unix=time.time(),
            scientific_failures_retained=True,new_rollouts=0,
            caveat='Verifies stored evidence and original protocol, not bitwise reproducibility of native rendering.'))
        write(out/'status.json',dict(status='PASS',finished_unix=time.time(),models=totals))
    except BaseException as exc:
        write(out/'status.json',dict(status='FAILED',error=repr(exc),started_unix=started))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    run(p.parse_args().config)
