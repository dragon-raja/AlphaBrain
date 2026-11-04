"""Single-center authorization of bounded remote CPU fakes via live SSH.

Remote agents cannot authorize themselves from shared JSON. Mutable custody,
permits and backend state stay on the pinned controller's local filesystem;
immutable checkpoints are fsynced to the configured persistent directory
before GO. No automatic controller failover or production backend exists.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

from .authority_wire import Channel, checkpoint, identity, require_local
from .manifest import digest, verify_artifacts
from .queue import MissingResult, _durable_write, exclusive
from .worker_fence import check_permit


def prepare_target(target):
    """Explicit empty-store preparation; never run as part of a launch/retry."""
    with Channel(target, prepare=True) as channel:
        hello = deepcopy(channel.hello)
        channel.wire.send(dict(kind='PROBE', session=hello['session']))
        if channel.wire.receive() != dict(kind='HEALTHY', session=hello['session']):
            raise RuntimeError('Worker preparation failed')
    return {k: hello[k] for k in ('identity', 'store_id', 'source')}


class AuthoritySimulationBackend:
    config_schema = 'CPU_AUTHORITY_V1'
    production = False

    def __init__(self, root, limits, *, config=None):
        self.root, self.limits = Path(root), deepcopy(limits)
        self.root.mkdir(parents=True, exist_ok=True)
        require_local(self.root)
        path = self.root/'authority-config.json'
        existing = path.exists()
        if path.exists():
            self.config = json.loads(path.read_text())
            if config is not None and config != self.config:
                raise ValueError('Authority configuration changed')
        else:
            if config is None:
                raise ValueError('Explicit authority configuration required')
            self.config = deepcopy(config)
            _durable_write(path, self.config)
        if self.config.get('schema') != self.config_schema or self.config.get('authority') != identity():
            raise RuntimeError('Controller node changed; no automatic failover')
        self.path = self.root/'launches.json'
        if not self.path.exists():
            if existing:
                raise RuntimeError('Authority launch records missing; explicit reconciliation required')
            _durable_write(self.path, dict(attempts={}))

    def bind_authority(self, ledger, permit=None):
        require_local(ledger.root)
        if permit is not None:
            require_local(Path(permit).parent)
        if self.config['authority'] != identity():
            raise RuntimeError('Controller identity changed')
        mirror = Path(self.config['checkpoint_directory'])
        if not mirror.is_absolute() or not mirror.is_dir():
            raise ValueError('Existing persistent checkpoint directory required')
        pin = ledger.root/'authority-node.json'
        expected = dict(identity=identity(), backend=str(self.root.resolve()), config_sha256=digest(self.config))
        if pin.exists():
            if json.loads(pin.read_text()) != expected:
                raise RuntimeError('Authority custody binding changed')
        else:
            _durable_write(pin, expected)
        ledger.authority_mirror = mirror
        checkpoint(ledger)

    def _target(self, host):
        return self.config['peers'][host]

    def channel(self, target):
        return Channel(target)

    def launch_channel(self, target, request):
        return self.channel(target)

    def launch_fields(self, task, request):
        return {}

    def accept_ack(self, channel, row, handle, launch):
        self._validate_row(row, handle, launch)

    def receive_launch_response(self, channel):
        return channel.wire.receive()

    def _hello(self, channel, target):
        if any(channel.hello[k] != target['expected'][k] for k in ('identity', 'store_id', 'source')):
            raise RuntimeError('Worker node, source or store identity changed')

    def probe(self, peer):
        target = self._target(peer)
        with self.channel(target) as channel:
            self._hello(channel, target)
            session = channel.hello['session']
            channel.wire.send(dict(kind='PROBE', session=session))
            return channel.wire.receive() == dict(kind='HEALTHY', session=session)

    def admit(self, slot):
        return self.probe(slot['host'])

    def start(self, *args):
        raise RuntimeError('Remote CPU launch requires the live center authority boundary')

    def start_authorized(self, ledger, plan, permit_path, request, *, clock=time.time):
        if ledger.failed or ledger.session != (request['owner'], request['epoch']):
            raise RuntimeError('Live authority session requires explicit recovery')
        self.bind_authority(ledger, permit_path)
        key = request['key']
        with exclusive(ledger.root):
            def check():
                if ledger.failed or ledger.session != (request['owner'], request['epoch']):
                    raise RuntimeError('Live authority session requires explicit recovery')
                budget, state = ledger.read()
                permit = json.loads(Path(permit_path).read_text())
                check_permit(state, budget, plan, permit, request['owner'], request['epoch'], clock(), production=self.production)
                row = state['tasks'][key]
                if (row.get('worker_request') != request or row['status'] != 'RESERVED'
                        or row['launch_status'] != 'RESERVED' or row['reserved'] != self.limits[key]
                        or row['epoch'] != request['epoch'] or row['slot'] != request['slot']
                        or row['attempt'] != request['attempt'] or row['queue_sha256'] != digest(plan)
                        or request['permit_sha256'] != digest(permit)):
                    raise RuntimeError('Authority reservation/permit changed')
                for peer in budget['peers']:
                    if self.probe(peer) is not True:
                        raise ConnectionError('Authority peer disconnected')
                return state, row
            state, row = check()
            if 'worker_claim' in row:
                raise RuntimeError('Duplicate or unknown grant; explicit reconciliation required')
            task = next(t for t in plan['tasks'] if t['key'] == key)
            if request['task_sha256'] != digest(task) or request['slot'] not in plan['slots']:
                raise ValueError('Unbound remote launch')
            verify_artifacts(plan)
            target = self._target(request['slot']['host'])
            launch = dict(operation='CPU_SYNTHETIC_ONLY', key=key, task_sha256=digest(task),
                          request_sha256=digest(request), epoch=request['epoch'], owner=request['owner'],
                          attempt=request['attempt'], limits=deepcopy(row['reserved']), slot=request['slot'])
            launch.update(self.launch_fields(task, request))
            with self.launch_channel(target, request) as channel:
                self._hello(channel, target)
                session = channel.hello['session']
                handle = dict(key=key, host=request['slot']['host'], attempt=request['attempt'],
                              launch_sha256=digest(launch), session=session, store_id=channel.hello['store_id'])
                local = json.loads(self.path.read_text())
                if key in local['attempts']:
                    raise RuntimeError('Duplicate authority transport record')
                local['attempts'][key] = dict(handle=handle, launch=launch)
                _durable_write(self.path, local)
                row['worker_claim'] = dict(outcome='INTENT', request_sha256=digest(request), handle=handle)
                ledger._save(state, dict(event='authority_prepare', key=key))
                channel.wire.send(dict(kind='PREPARE', session=session, key=key, launch=launch))
                ready = self.receive_launch_response(channel)
                if ready != dict(kind='READY', session=session, launch_sha256=digest(launch)):
                    raise RuntimeError('Worker preparation does not match authority request')
                state, row = check()  # fresh center decision; no remote NFS read
                row['worker_claim']['outcome'] = 'GRANTED'
                ledger._save(state, dict(event='authority_grant_before_go', key=key))
                # A slow persistent checkpoint may outlive the permit. Check
                # the local clock/permit again after fsync and before GO.
                budget, current = ledger.read()
                current_permit = json.loads(Path(permit_path).read_text())
                check_permit(current, budget, plan, current_permit, request['owner'], request['epoch'], clock(), production=self.production)
                if digest(current_permit) != request['permit_sha256']:
                    raise RuntimeError('Permit changed during persistent grant')
                channel.wire.send(dict(kind='GO', session=session, launch_sha256=digest(launch)))
                ack = self.receive_launch_response(channel)
                if (ack.get('kind') != 'ACK' or ack.get('session') != session
                        or ack.get('launch_sha256') != digest(launch)):
                    raise RuntimeError('Unknown authority acknowledgement')
                self.accept_ack(channel, ack['row'], handle, launch)
                row['worker_claim'].update(outcome='ACKNOWLEDGED', handle=handle)
                ledger._save(state, dict(event='worker_start_ack', key=key))
                return handle

    def _validate_row(self, row, handle, launch):
        expected = self._target(handle['host'])['expected']
        if (row['launch'] != launch or row['launch_sha256'] != handle['launch_sha256']
                or row['session'] != handle['session'] or row['store_id'] != handle['store_id']
                or row['identity'] != expected['identity']):
            raise RuntimeError('Remote receipt identity changed')
        if row['status'] != 'COMPLETE' or row['starts'] != 1:
            raise MissingResult('Remote task not proved complete; keep reservation')
        expected_receipt = dict(simulation_only=True, task_sha256=launch['task_sha256'],
                                budget_consumed=launch['limits'], scientific_success=False)
        if row['receipt'] != expected_receipt:
            raise ValueError('Remote synthetic budget/receipt changed')
        return deepcopy(row['receipt'])

    def _inspect(self, host, key):
        target = self._target(host)
        with self.channel(target) as channel:
            self._hello(channel, target)
            channel.wire.send(dict(kind='INSPECT', session=channel.hello['session'], key=key))
            value = channel.wire.receive()
            if value.get('kind') != 'STATE':
                raise RuntimeError('Invalid remote inspection')
            return value['row']

    def locate(self, task, slot, attempt):
        entry = json.loads(self.path.read_text())['attempts'].get(task['key'])
        remote = self._inspect(slot['host'], task['key'])
        if entry is None:
            if remote is not None:
                raise RuntimeError('Unledgered remote worker intent')
            return None
        if entry['launch']['slot'] != slot or entry['handle']['attempt'] != attempt:
            raise RuntimeError('Remote attempt changed')
        return deepcopy(entry['handle'])

    def alive(self, handle):
        # Taking the worker store lock proves no handshake for this store is
        # active. Missing/INTENT records are not treated as safe to relaunch.
        self._inspect(handle['host'], handle['key'])
        return False

    def poll(self, handle):
        entry = json.loads(self.path.read_text())['attempts'][handle['key']]
        row = self._inspect(handle['host'], handle['key'])
        if row is None:
            raise MissingResult('Remote store lost the launch; keep reservation')
        self._validate_row(row, handle, entry['launch'])
        return 0

    def cancel(self, handle):
        raise RuntimeError('Remote uncertainty is never cleared by killing transport')

    def validate(self, task):
        entry = json.loads(self.path.read_text())['attempts'].get(task['key'])
        if entry is None:
            raise MissingResult('No authority launch record')
        row = self._inspect(entry['handle']['host'], task['key'])
        if row is None:
            raise MissingResult('No remote receipt; keep reservation')
        return self._validate_row(row, entry['handle'], entry['launch'])
