"""Publish disjoint replacement queues after the old coordinator has exited.

Workers need not stop: running handles remain on their original host/slot.
Only never-started tasks may move. Original output and scientific identities stay
unchanged, so existing analysis ownership remains valid.
"""
from copy import deepcopy
from pathlib import Path
import shutil

from .manifest import verify_artifacts
from .queue import exclusive, load, prepare, _durable_write


def repartition(old_root, destinations, evidence_root):
    """Caller proves controller exit; its exclusive lock is checked here too."""
    old_root = Path(old_root)
    with exclusive(old_root):
        old, state = load(old_root)
        verify_artifacts(old)
        from .resources import controls, device, set_drain
        old_drains = controls(old_root, old)['drained']
        if (old_root/'retired.json').exists(): raise ValueError('Queue already handed off')
        if state['status'] not in ('RUNNING', 'PREPARED'):
            raise ValueError('Only healthy queues can transfer ownership')
        tasks = {t['key']: t for t in old['tasks']}
        keys = [k for d in destinations for k in d['task_keys']]
        if len(keys) != len(set(keys)) or set(keys) != set(tasks):
            raise ValueError('Replacement queues overlap or omit tasks')
        evidence_root=Path(evidence_root)
        evidence_root.mkdir(parents=True,exist_ok=False)
        for name in ('manifest.json','state.json'):
            shutil.copy2(old_root/name,evidence_root/name)
        prepared = []
        for d in destinations:
            if Path(d['root']).exists(): raise FileExistsError(d['root'])
            new = deepcopy(old)
            new['tasks'] = [tasks[k] for k in d['task_keys']]
            new['slots'] = d['slots']; new['runtime'] = d['runtime']
            # Resource-only host admission is separately fingerprinted. It does
            # not replace the scientific worker runtime or its original receipt.
            if 'resource_admission' in d:
                new['resource_admission'] = deepcopy(d['resource_admission'])
            if 'host_bootstrap' in d:
                new['host_bootstrap'] = deepcopy(d['host_bootstrap'])
            # Runtime migration may not change workers, inputs, noise or inference flags.
            allowed = {'host', 'workers_per_gpu'}
            if {k:v for k,v in new['runtime'].items() if k not in allowed} != {
                    k:v for k,v in old['runtime'].items() if k not in allowed}:
                raise ValueError('Repartition cannot change scientific worker runtime')
            for t in new['tasks']:
                row = state['tasks'][t['key']]
                if row['status']=='LAUNCHING': raise ValueError('Unresolved launch; do not transfer')
                if row['status']=='RUNNING' and row['slot'] not in new['slots']:
                    raise ValueError('Running worker cannot change host or slot')
                if new['runtime']['host'] != old['runtime']['host'] and (
                        row['status']!='PENDING' or row['attempt']!=0 or Path(t['output']).exists()):
                    raise ValueError('Only never-started tasks can move to another host')
            assets = {a['path']:a['sha256'] for a in old['artifacts']}
            assets.update({a['path']:a['sha256'] for a in d['controller_artifacts']})
            from .manifest import file_digest
            for path in (evidence_root/'manifest.json', evidence_root/'state.json'):
                assets[str(path)] = file_digest(path)
            new['artifacts'] = [dict(path=p,sha256=h) for p,h in sorted(assets.items())]
            verify_artifacts(new)
            for field in ('resource_admission', 'host_bootstrap'):
                if field in new:
                    binding = new[field]
                    if assets.get(binding['path']) != binding['sha256']:
                        raise ValueError('Resource admission must be an immutable artifact')
            prepared.append((d,new))
        # Tombstone before publishing: failure is fail-closed, never resume old queue.
        _durable_write(old_root/'retired.json',dict(status='RETIRED_FOR_DISJOINT_HANDOFF',
            replacements=[d['root'] for d in destinations], original_state_unchanged=True))
        for d,new in prepared:
            fresh = prepare(new,d['root'])
            fresh['tasks'] = {k:deepcopy(state['tasks'][k]) for k in d['task_keys']}
            fresh['events'].append(dict(event='adopted_original_worker_handles', old_root=str(old_root)))
            _durable_write(Path(d['root'])/'state.json',fresh)
            devices = {(s['host'], device(s)) for s in new['slots']}
            drains = [r for r in old_drains if (r['host'], r['gpu']) in devices]
            drains += [dict(host=new['runtime']['host'], gpu=str(g), reason='Finish-time rebalance: let in-flight work finish')
                       for g in d.get('initial_drained_gpus', [])]
            for row in drains:
                set_drain(d['root'], row['host'], row['gpu'], reason=row['reason'])
        return dict(status='PREPARED',blocks={d['runtime']['host']:len(d['task_keys']) for d in destinations},
                    scientific_addresses_unchanged=True, existing_outputs_unchanged=True)
