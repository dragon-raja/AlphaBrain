"""Outcome-independent resource controls; never alter a scientific manifest.

A drain stops *new* dispatch to a physical GPU, including all its worker slots.
Existing workers finish normally. This is not cancellation or task recovery.
"""
import fcntl
from contextlib import contextmanager
import json
from pathlib import Path
import time

from .manifest import digest
from .queue import _durable_write, load


def device(slot):
    return str(slot.get('gpu_id', slot['id']))


def controls(root, plan):
    path = Path(root) / 'resource-control.json'
    if not path.exists():
        return dict(manifest_sha256=digest(plan), drained=[])
    value = json.loads(path.read_text())
    if value['manifest_sha256'] != digest(plan):
        raise ValueError('Resource control belongs to another manifest')
    known = {(s['host'], device(s)) for s in plan['slots']}
    if any((r['host'], r['gpu']) not in known or not r.get('reason') for r in value['drained']):
        raise ValueError('Unknown or unexplained drained GPU')
    return value


def drained_slots(root, plan):
    return {(r['host'], r['gpu']) for r in controls(root, plan)['drained']}


@contextmanager
def dispatch_guard(root, plan, slot):
    """Serialize drain acknowledgement with dispatch intent + launch receipt."""
    with (Path(root)/'resource-control.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield (slot['host'],device(slot)) not in drained_slots(root,plan)


def set_drain(root, host, gpu, *, reason, enabled=True):
    root = Path(root)
    if not reason.strip():
        raise ValueError('An audit reason is required')
    with (root / 'resource-control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan, _ = load(root)
        if (root / 'retired.json').exists():
            raise ValueError('Cannot control a retired queue')
        if (host, str(gpu)) not in {(s['host'], device(s)) for s in plan['slots']}:
            raise ValueError('Unknown physical GPU')
        value = controls(root, plan)
        value['drained'] = [r for r in value['drained'] if (r['host'], r['gpu']) != (host, str(gpu))]
        if enabled:
            value['drained'].append(dict(host=host, gpu=str(gpu), reason=reason))
        value.setdefault('events', []).append(dict(host=host, gpu=str(gpu), enabled=enabled,
                                                  reason=reason, unix=time.time()))
        _durable_write(root / 'resource-control.json', value)
    return resource_status(root)


def resource_status(root):
    plan, state = load(root)
    drained = drained_slots(root, plan)
    rows = []
    pending = sum(r['status'] == 'PENDING' for r in state['tasks'].values())
    for host, gpu in sorted({(s['host'], device(s)) for s in plan['slots']}):
        slots = [s for s in plan['slots'] if (s['host'], device(s)) == (host, gpu)]
        active = [r for r in state['tasks'].values() if r['status'] in ('LAUNCHING', 'RUNNING')
                  and (r['slot']['host'], device(r['slot'])) == (host, gpu)]
        held = (host, gpu) in drained
        rows.append(dict(host=host, gpu=gpu, slots=len(slots), active=len(active),
                         drained=held, quiescent=held and not active,
                         idle_with_pending=bool(pending and not held and len(active) < len(slots))))
    return dict(status=state['status'], pending_blocks=pending, devices=rows)


def choose_capacity(rows, *, minimum_gain=0.05):
    """Choose measured throughput, not the largest count that happened to fit.

    Caller must bind the observations to model/checkpoint/runtime/hardware.
    Inference-only evidence cannot certify simulator or training capacity.
    """
    eligible = [r for r in rows if r['status'] == 'PASS' and r['quality_pass']
                and r['headroom_mib'] >= r['required_headroom_mib'] and r['throughput'] > 0]
    if not eligible:
        return None
    best = min(eligible, key=lambda r: r['workers'])
    for row in sorted(eligible, key=lambda r: r['workers']):
        if row['throughput'] > best['throughput'] * (1 + minimum_gain):
            best = row
    return best['workers']


def training_allocation(devices, *, world_size):
    """DDP/FSDP gang reservation: one rank per exclusive physical GPU.

    This creates a resource request, not a training launcher. Evaluation slot
    multiplication must never be used to split ranks into independent jobs.
    """
    identities = [(r['host'], str(r['gpu'])) for r in devices]
    if type(world_size) is not int or world_size < 1 or len(identities) != world_size or len(set(identities)) != world_size:
        raise ValueError('Training requires world_size distinct physical GPUs')
    return dict(mode='exclusive_gang', world_size=world_size,
                ranks=[dict(rank=i, host=h, gpu=g) for i, (h, g) in enumerate(identities)])
