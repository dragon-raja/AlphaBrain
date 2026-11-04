"""Local proof of sibling workers for the existing GPU ownership guard.

No device is probed or process launched during import. A same-GPU sibling is
recognized only from this bound store, current epoch and a live birth identity.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time

from . import gpu_guard
from .authority_wire import identity
from .manifest import digest
from .queue import _durable_write


@contextmanager
def launch_lock(root, slot):
    gpu = str(slot['gpu_id'])
    if not gpu.isdigit():
        raise ValueError('Physical GPU index required')
    with (Path(root)/('gpu-'+gpu+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def owned_siblings(root, binding, launch, node, *, process=None):
    root = Path(root)
    process = process or gpu_guard.process
    capacity = node.get('workers_per_gpu', 1)
    if type(capacity) is not int or capacity not in (1, 2, 3):
        raise ValueError('Only explicit one/two/three-worker capacity is supported')
    store = json.loads((root/'store.json').read_text())
    here = identity()
    if store['identity'] != here or store['binding_sha256'] != digest(binding):
        raise RuntimeError('Unbound worker store')
    devices = {str(g['index']):g['uuid'] for g in node['gpu_guard']['devices']}
    gpu = str(launch['slot']['gpu_id'])
    if gpu not in devices:
        raise ValueError('Unknown GPU')
    owned = []
    resident = binding.get('resident_workers', False)
    own_slot = False
    for path in (root/('residents' if resident else 'tasks')).glob('*.json'):
        row = json.loads(path.read_text())
        other = row['launch']
        if (not resident and other['key'] == launch['key']) or str(other['slot']['gpu_id']) != gpu:
            continue
        if row['status'] == ('CLOSED' if resident else 'COMPLETE'):
            continue  # Surviving unowned CUDA children are rejected by preflight.
        same_slot = other['slot']['id'] == launch['slot']['id']
        reusing = resident and same_slot and row['process']['pid'] == os.getpid()
        if reusing and row['status']!='IDLE':
            raise RuntimeError('Only an idle completed resident may continue')
        if (row['status'] not in (('RUNNING','IDLE') if resident else ('RUNNING',))
                or row['identity'] != here or row['store_id'] != store['store_id']
                or row['starts'] != 1 or row['launch_sha256'] != digest(other)
                or other['binding_sha256'] != digest(binding)
                or other['operation'] != 'LIBERO_BLOCK'
                or other['owner'] != launch['owner'] or other['epoch'] != launch['epoch']
                or other['permit']['stage'] != launch['permit']['stage']
                or other['slot']['host'] != launch['slot']['host']
                or (same_slot and not reusing)
                or (resident and (row.get('resident') is not True or digest(other['permit'])!=digest(launch['permit'])))):
            raise RuntimeError('Unsettled or foreign same-GPU worker')
        spec = binding['tasks'].get(other['key'])
        if spec is None or other['slot'] not in spec['eligible_slots']:
            raise RuntimeError('Sibling slot is outside binding')
        handle = row['process']
        actual = process(handle['pid'])
        args = actual['argv']
        if (str(actual['start_ticks']) != str(handle['start_ticks']) or actual['pgid'] != handle['pgid']
                or actual['pid'] != handle['pgid'] or actual['sid'] != handle['pgid']
                or actual['state'] in ('Z', 'X')
                or actual['executable'] != node['python_resolved']
                or actual['executable_sha256'] != node['python_sha256']
                or args.count('-m') != 1 or args[args.index('-m')+1] != 'benchmarks.LIBERO.continuous_authority'
                or args.count('--root') != 1 or args[args.index('--root')+1] != str(root)):
            raise RuntimeError('Sibling live process identity differs')
        owned.append(dict(pid=actual['pid'], start=actual['start_ticks'], boot=here['boot'],
                          group=actual['pgid'], gpu_uuid=devices[gpu]))
        own_slot = own_slot or reusing
    if len({r['pid'] for r in owned}) != len(owned):
        raise RuntimeError('Duplicate resident GPU ownership')
    if len(owned)-int(own_slot) >= capacity:
        raise RuntimeError('Physical GPU worker capacity exhausted')
    return owned


def preflight(root, binding, launch, node):
    owned = owned_siblings(root, binding, launch, node)
    # owned_siblings proves current PID, birth, binding, slot and IDLE state.
    # Cold starts still reserve a new peak; warm continuations retain all
    # existing peaks without adding a third model to a two-model allocation.
    continuing=os.getpid() if any(o['pid']==os.getpid() for o in owned) else None
    target=next(g['uuid'] for g in node['gpu_guard']['devices'] if str(g['index'])==str(launch['slot']['gpu_id']))
    kwargs=dict(owned=owned,target_gpu_uuid=target)
    if continuing is not None:kwargs['continuing_pid']=continuing
    result=gpu_guard.preflight(node['gpu_guard'], **kwargs)
    evidence=Path(root)/'preflights';evidence.mkdir(exist_ok=True)
    _durable_write(evidence/f"{launch['key']}-{time.time_ns()}.json",result)
    return result
