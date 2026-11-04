"""CPU-only remote agent. Reads local intents and a live channel, never custody.

There is no arbitrary command, model, GPU, training or production backend.
Losing the controller before GO cannot start work. After GO a single bounded
synthetic action may complete; an absent ACK is reconciled, never replayed.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import uuid

from .authority_wire import Wire, fingerprint, identity, require_local
from .manifest import digest
from .queue import _durable_write, exclusive


def prepare(root):
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    require_local(root.parent)
    root.mkdir(exist_ok=False)
    _durable_write(root/'store.json', dict(schema='REMOTE_CPU_ONLY_V1', store_id=uuid.uuid4().hex,
                                         identity=identity()))
    (root/'tasks').mkdir()


def serve(root, wire):
    root = Path(root)
    require_local(root)
    store = json.loads((root/'store.json').read_text())  # never auto-recreate
    if store['schema'] != 'REMOTE_CPU_ONLY_V1' or store['identity'] != identity():
        raise RuntimeError('Worker store/container changed; explicit reconciliation required')
    session = uuid.uuid4().hex
    wire.send(dict(kind='HELLO', identity=identity(), store_id=store['store_id'], source=fingerprint(), session=session))
    request = wire.receive()
    if request == dict(kind='PROBE', session=session):
        wire.send(dict(kind='HEALTHY', session=session)); return
    key = request.get('key', '')
    if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('Invalid task key')
    path = root/'tasks'/(key+'.json')
    with exclusive(root):
        if request.get('kind') == 'INSPECT':
            if request.get('session') != session:
                raise RuntimeError('Stale inspection session')
            wire.send(dict(kind='STATE', row=json.loads(path.read_text()) if path.exists() else None)); return
        if request.get('kind') != 'PREPARE' or request.get('session') != session:
            raise RuntimeError('A fresh preparation handshake is required')
        launch = request['launch']
        if (launch.get('operation') != 'CPU_SYNTHETIC_ONLY' or launch.get('key') != key
                or type(launch.get('epoch')) is not int or launch['epoch'] < 1
                or type(launch.get('attempt')) is not int or launch['attempt'] < 1
                or not launch.get('limits') or any(type(v) is not int or v < 0 for v in launch['limits'].values())):
            raise ValueError('Invalid synthetic launch request')
        if path.exists():
            raise RuntimeError('Duplicate or unknown task; never replay')
        stamp = digest(launch)
        row = dict(status='PREPARED', launch=deepcopy(launch), launch_sha256=stamp,
                   session=session, store_id=store['store_id'], identity=identity(), starts=0)
        _durable_write(path, row)
        wire.send(dict(kind='READY', session=session, launch_sha256=stamp))
        # A file, expired session, old GO or EOF cannot substitute for this
        # live channel response. The ten-second receive bound is independent
        # of the controller's clock or a shared filesystem's cache.
        go = wire.receive()
        if go != dict(kind='GO', session=session, launch_sha256=stamp):
            raise RuntimeError('Grant does not match this live worker session')
        row.update(status='INTENT', starts=1)
        _durable_write(path, row)
        receipt = dict(simulation_only=True, task_sha256=launch['task_sha256'],
                       budget_consumed=deepcopy(launch['limits']), scientific_success=False)
        row.update(status='COMPLETE', receipt=receipt, finished_at=time.time())
        _durable_write(path, row)
        wire.send(dict(kind='ACK', session=session, launch_sha256=stamp, row=row))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.execute or not args.simulate:
        parser.error('Production worker disabled; explicit CPU simulation required')
    wire = Wire(sys.stdin.fileno(), sys.stdout.fileno())
    try:
        if args.prepare:
            prepare(args.root)
        serve(args.root, wire)
    except Exception as error:
        try:
            wire.send(dict(kind='ERROR', error=type(error).__name__+': '+str(error)))
        except Exception:
            pass
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
