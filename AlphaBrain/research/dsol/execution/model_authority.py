"""Live, non-replaying model transport; scientific execution stays in adapters.

CPU contract mode uses the identical persistent channel lifecycle without model
imports. Production requires a separately bound, explicit stage permit.
"""
from copy import deepcopy
from contextlib import contextmanager
import json
import select
import time
from pathlib import Path

from .authority_backend import AuthoritySimulationBackend
from .authority_wire import Channel
from .manifest import digest
from .queue import MissingResult, _durable_write, exclusive
from .worker_fence import check_permit


class ModelAuthorityBackend(AuthoritySimulationBackend):
    config_schema = 'LIVE_MODEL_AUTHORITY_V1'

    def __init__(self, root, limits, *, config=None):
        super().__init__(root, limits, config=config)
        if self.config.get('mode') not in ('cpu_contract', 'libero'):
            raise ValueError('Explicit CPU contract or LIBERO mode required')
        self.production = self.config['mode'] == 'libero'
        self.channels = {}
        self.contexts = {}
        self.peer_channels = {}
        self.inspection_channels = {}
        self.inspection_last_probe = {}
        self.failed_inspection_hosts = set()
        self.persistent_inspection = self.config.get('persistent_inspection', False)
        if type(self.persistent_inspection) is not bool:
            raise ValueError('Persistent inspection requires an explicit boolean')
        self.idle_channels = {}
        self.resident = self.config.get('resident_workers', False)
        self.launch_timeout = self.config.get('launch_timeout_seconds', 60)
        if type(self.launch_timeout) not in (int, float) or not 60 <= self.launch_timeout <= 600:
            raise ValueError('Launch validation timeout must be bounded to 60..600 seconds')
        if type(self.resident) is not bool:
            raise ValueError('Resident workers requires an explicit boolean')
        self.parallel_launches = self.config.get('parallel_launches', 1)
        if type(self.parallel_launches) is not int or not 1 <= self.parallel_launches <= 32:
            raise ValueError('Launch concurrency must be an integer in 1..32')
        self.pending_launches = None
        self.metrics = dict(poll_rounds=0, ledger_checks=0, heartbeat_rounds=0,
                            heartbeat_messages=0, continuation_grants=0, max_grants_per_round=0, poll_seconds=0.,
                            worker_processes=0, resident_reuses=0, idle_heartbeats=0,
                            completion_inspections=0, completion_inspection_seconds=0.,
                            inspection_connections=0,
                            resident_done_alive_checks=0, max_pending_launches=0)

    def begin_authorized(self, ledger, plan, permit_path, request, *, queue_root, clock=time.time):
        if self.parallel_launches <= 1:
            raise RuntimeError('Parallel preparation was not enabled')
        if plan.get('binding_sha256') != self.config['binding_sha256']:
            raise ValueError('Queue binding differs from transport')
        if self.pending_launches is None:
            from .parallel_launch import ParallelLaunches
            self.pending_launches = ParallelLaunches(self)
        self.contexts[request['key']] = (ledger, plan, Path(permit_path), deepcopy(request), clock)
        return self.pending_launches.begin(request, queue_root)

    def poll_authorized(self, ticket):
        if self.pending_launches is None:
            raise RuntimeError('No pending launch authority')
        return self.pending_launches.poll(ticket)

    def cancel_pending(self, ticket):
        if self.pending_launches is not None:
            self.pending_launches.cancel(ticket)
        raise RuntimeError('Launch stopped; retain reservation for explicit reconciliation')

    def locate(self, task, slot, attempt):
        entry = json.loads(self.path.read_text())['attempts'].get(task['key'])
        if entry is not None and entry.get('handle') is None:
            raise RuntimeError('Unacknowledged connection intent requires explicit audit; no replay')
        return super().locate(task, slot, attempt)

    @contextmanager
    def launch_channel(self, target, request):
        slot = digest(request['slot'])
        client = self.idle_channels.pop(slot, None) if self.resident else None
        if client is None:
            with self.channel(target) as channel:
                self.metrics['worker_processes'] += 1
                yield channel
            return
        channel = client['channel']
        try:
            old = client['launch']
            if (old['owner'] != request['owner'] or old['epoch'] != request['epoch']
                    or old['slot'] != request['slot'] or digest(old['permit']) != request['permit_sha256']
                    or channel.closed or channel.process.poll() is not None):
                raise RuntimeError('Resident session lost/changed; no automatic reconnect')
            self.metrics['resident_reuses'] += 1
            yield channel
        except BaseException:
            channel.close()
            raise

    def channel(self, target):
        allowed = {'benchmarks.LIBERO.continuous_authority'}
        if not self.production:
            allowed.add('benchmarks.Robocasa_tabletop.continuous_authority')
        if target.get('agent') not in allowed:
            raise ValueError('Only reviewed agents are registered; GR1 remains CPU-contract only')
        channel = Channel(target, agent=target['agent'], expected_source=target['expected']['source'],
                          environment=target.get('environment', {}))
        channel.wire.timeout = 60
        return channel

    def receive_launch_response(self, channel):
        # PREPARE and post-GO validation hash large immutable assets. Their
        # bounded IO allowance must not lengthen execution heartbeats/grants.
        previous = channel.wire.timeout
        deadline = time.monotonic() + self.launch_timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Bounded launch validation deadline expired')
                channel.wire.timeout = min(10., remaining)
                try:
                    return channel.wire.receive()
                except TimeoutError:
                    # PREPARE may hash tens of GB; keep the independent pinned
                    # peer channels live, without granting model computation.
                    # A failed peer still aborts and the overall deadline never
                    # resets. Wire retains partial bytes between bounded reads.
                    for peer in self.config['peers']:
                        if self.probe(peer) is not True:
                            raise ConnectionError('Peer lost during launch validation')
        finally:
            channel.wire.timeout = previous

    def _node_health(self,peer):
        config=self.config.get('node_health',{}).get(peer)
        if self.production and config is not None:
            from .node_health import require_fresh_health
            require_fresh_health(config)

    def probe(self, peer):
        self._node_health(peer)
        channel = self.peer_channels.get(peer)
        if channel is None:
            target = self._target(peer)
            channel = self.channel(target).__enter__()
            try:
                self._hello(channel, target)
            except BaseException:
                channel.close()
                raise
            self.peer_channels[peer] = channel
        try:
            session = channel.hello['session']
            channel.wire.send(dict(kind='PROBE', session=session))
            if channel.wire.receive() != dict(kind='HEALTHY', session=session):
                raise RuntimeError('Heartbeat response changed')
            self.probe_idle()
            self.probe_inspections()
            return True
        except BaseException:
            channel.close()
            # Do not reconnect and silently resume after a lost authority link.
            raise ConnectionError('Pinned peer heartbeat disconnected') from None

    def probe_idle(self):
        # Idle heartbeats keep a retained process alive, never grant compute.
        # Busy workers and all node peers still use fresh checks for every grant.
        now=time.monotonic()
        waiting = [c for c in self.channels.values() if self.resident and c.get('done')]
        clients=[c for c in [*self.idle_channels.values(), *waiting] if now-c.get('last_idle_probe',0.)>=10.]
        for client in clients:
            client['channel'].wire.send(dict(kind='IDLE_PROBE',session=client['handle']['session']))
        for client in clients:
            if client['channel'].wire.receive()!=dict(kind='IDLE',session=client['handle']['session']):
                raise RuntimeError('Resident idle connection changed')
            client['last_idle_probe']=now
        self.metrics['idle_heartbeats']+=len(clients)

    def close_channels(self):
        if self.pending_launches is not None:
            self.pending_launches.close()
        for channel in self.inspection_channels.values():
            channel.close()
        self.inspection_channels.clear()
        self.inspection_last_probe.clear()
        for channel in self.peer_channels.values():
            channel.close()
        self.peer_channels.clear()
        for client in self.channels.values():
            client['channel'].close()
        for client in self.idle_channels.values():
            client['channel'].close()
        self.idle_channels.clear()
        try:
            _durable_write(self.root/'transport-metrics.json', self.metrics)
        except OSError:
            pass  # Missing telemetry never authorizes a restart.

    def probe_many(self, peers):
        """Fresh fan-out/fan-in, no cached health or reconnect on failure."""
        pending = []
        try:
            for peer in peers:
                self._node_health(peer)
                channel = self.peer_channels.get(peer)
                if channel is None:
                    target = self._target(peer)
                    channel = self.channel(target).__enter__()
                    try:
                        self._hello(channel, target)
                    except BaseException:
                        channel.close()
                        raise
                    self.peer_channels[peer] = channel
                session = channel.hello['session']
                channel.wire.send(dict(kind='PROBE', session=session))
                pending.append((channel, session))
            for channel, session in pending:
                if channel.wire.receive() != dict(kind='HEALTHY', session=session):
                    raise RuntimeError('Heartbeat response changed')
            self.probe_idle()
            self.probe_inspections()
        except BaseException:
            for channel in self.peer_channels.values():
                channel.close()
            raise ConnectionError('Pinned peer heartbeat disconnected') from None
        self.metrics['heartbeat_rounds'] += 1
        self.metrics['heartbeat_messages'] += len(pending)

    def poll_many(self, handles):
        """One locked live decision for this round's individual CHECK messages.

        No credits or future grants: every worker still blocks for its exact
        next sequence. Re-read local custody/permit after collecting requests,
        probe every peer freshly, and hold the lock through current replies.
        """
        started = time.monotonic()
        if not handles:
            return []
        if len({h['key'] for h in handles}) != len(handles):
            raise ValueError('Duplicate worker in polling batch')
        clients = []
        for handle in handles:
            client = self.channels.get(handle['key'])
            if client is None or client['handle'] != handle:
                raise RuntimeError('Lost or mismatched live worker channel')
            clients.append(client)
        ready = set(select.select([c['channel'].wire.reader for c in clients if not c.get('done')], [], [], 0)[0])
        messages = []
        for handle, client in zip(handles, clients):
            wire = client['channel'].wire
            value = (wire.receive_ready() if self.parallel_launches > 1 else wire.receive()) if not client.get('done') and (wire.buffer or wire.reader in ready) else None
            if value is not None:
                if value.get('session') != handle['session']:
                    raise RuntimeError('Wrong live worker session')
                if value.get('kind') == 'CHECK':
                    if value.get('sequence') != client['sequence'] + 1:
                        raise RuntimeError('Repeated or missing worker checkpoint')
                elif value.get('kind') == 'DONE':
                    self._validate_row(value['row'], handle, client['launch'])
                else:
                    raise RuntimeError('Unexpected live model worker response')
            messages.append(value)
        ledger, plan, permit_path, request, clock = self.contexts[handles[0]['key']]
        for handle in handles:
            other = self.contexts[handle['key']]
            if (other[0] is not ledger or other[1] is not plan or other[2] != permit_path
                    or any(other[3][k] != request[k] for k in ('owner', 'epoch', 'permit_sha256', 'queue_sha256'))):
                raise RuntimeError('Mixed authority sessions in polling batch')
        self.probe_many(self.config['peers'])
        with exclusive(ledger.root):
            budget, state = ledger.read()
            if ledger.failed or ledger.session != (request['owner'], request['epoch']):
                raise RuntimeError('Controller continuation session lost')
            permit = json.loads(permit_path.read_text())
            if digest(permit) != request['permit_sha256']:
                raise RuntimeError('Running worker permit changed')
            check_permit(state, budget, plan, permit, request['owner'], request['epoch'], clock(),
                         dispatch=False, production=self.production)
            self.metrics['ledger_checks'] += 1
            if set(budget['peers']) != set(self.config['peers']):
                raise RuntimeError('Heartbeat peer coverage differs from ledger')
            result = []
            grants = 0
            for handle, client, value in zip(handles, clients, messages):
                if clock() >= permit['finish_until']:
                    raise RuntimeError('Worker finish deadline expired during polling batch')
                if value and value['kind'] == 'CHECK':
                    client['channel'].wire.send(dict(kind='CONTINUE', session=handle['session'], sequence=value['sequence']))
                    client['sequence'] = value['sequence']
                    self.metrics['continuation_grants'] += 1
                    grants += 1
                if value and value['kind'] == 'DONE':
                    client['done'] = True
                    if not self.resident:
                        client['channel'].close()
                if client.get('done'):
                    result.append(0)
                else:
                    result.append(None)
        self.metrics['poll_rounds'] += 1
        self.metrics['max_grants_per_round'] = max(self.metrics['max_grants_per_round'], grants)
        self.metrics['poll_seconds'] += time.monotonic() - started
        return result

    def launch_fields(self, task, request):
        # The task contains only an immutable request and binding reference.
        if task['payload']['binding_sha256'] != self.config['binding_sha256']:
            raise ValueError('Task is outside this reviewed model binding')
        permit = json.loads(self.contexts[request['key']][2].read_text())
        return dict(operation='LIBERO_BLOCK' if self.production else 'CPU_BLOCK_CONTRACT',
                    task=deepcopy(task), binding_sha256=self.config['binding_sha256'], permit=permit,
                    resident_workers=self.resident)

    def start_authorized(self, ledger, plan, permit_path, request, *, clock=time.time):
        if plan.get('binding_sha256') != self.config['binding_sha256']:
            raise ValueError('Queue binding differs from transport')
        self.contexts[request['key']] = (ledger, plan, Path(permit_path), deepcopy(request), clock)
        return super().start_authorized(ledger, plan, permit_path, request, clock=clock)

    def verify_previous(self, budget, state, previous):
        for task in budget['tasks']:
            if task['stage'] in previous:
                if self.validate(task) != state['tasks'][task['key']]['receipt']:
                    raise ValueError('Previous stage receipt changed')

    def _identity(self, row, handle, launch):
        expected = self._target(handle['host'])['expected']
        if (row['launch'] != launch or row['launch_sha256'] != handle['launch_sha256']
                or row['session'] != handle['session'] or row['store_id'] != handle['store_id']
                or row['identity'] != expected['identity'] or row['starts'] != 1):
            raise ValueError('Model worker receipt identity changed')

    def accept_ack(self, channel, row, handle, launch):
        self._identity(row, handle, launch)
        if row['status'] != 'RUNNING':
            raise RuntimeError('Worker did not acknowledge the one-time start')
        channel.detached = True
        self.channels[handle['key']] = dict(channel=channel, handle=deepcopy(handle), launch=deepcopy(launch), sequence=0)

    def _continue(self, key, *, probe=True):
        ledger, plan, permit_path, request, clock = self.contexts[key]
        with exclusive(ledger.root):
            budget, state = ledger.read()
            if ledger.failed or ledger.session != (request['owner'], request['epoch']):
                raise RuntimeError('Controller continuation session lost')
            permit = json.loads(permit_path.read_text())
            if digest(permit) != request['permit_sha256']:
                raise RuntimeError('Running worker permit changed')
            check_permit(state, budget, plan, permit, request['owner'], request['epoch'], clock(),
                         dispatch=False, production=self.production)
            for peer in budget['peers'] if probe else ():
                if not self.probe(peer):
                    raise ConnectionError('Peer disconnect stops further model calls')

    def poll(self, handle):
        key = handle['key']
        client = self.channels.get(key)
        if client is None:
            raise RuntimeError('Lost live model channel; explicit reconciliation required')
        channel = client['channel']
        # Finish deadlines and peer failures apply even while the model loads.
        self._continue(key, probe=False)
        if channel.wire.buffer or select.select([channel.wire.reader], [], [], 0)[0]:
            value = channel.wire.receive()
            if value.get('session') != handle['session']:
                raise RuntimeError('Wrong live worker session')
            if value.get('kind') == 'CHECK':
                sequence = client['sequence']+1
                if value.get('sequence') != sequence:
                    raise RuntimeError('Repeated or missing worker checkpoint')
                self._continue(key)
                client['sequence'] = sequence
                channel.wire.send(dict(kind='CONTINUE', session=handle['session'], sequence=sequence))
                return None
            if value.get('kind') == 'DONE':
                self._validate_row(value['row'], handle, client['launch'])
                client['done'] = True
                if not self.resident:
                    channel.close()
                return 0
            raise RuntimeError('Unexpected live model worker response')
        return None

    def alive(self, handle):
        client = self.channels.get(handle['key'])
        if self.resident and client is not None:
            if client['handle'] != handle:
                raise RuntimeError('Mismatched live completion handle')
            if client.get('done'):
                # The pinned agent emits DONE only after releasing the task
                # lock. Its model/simulator processes intentionally remain
                # resident. validate() still performs a fresh INSPECT, checks
                # the lock and rehashes every scientific result before billing.
                # This is never inferred from a shared status file, reused
                # across sessions, or used by locate()/explicit recovery.
                self.metrics['resident_done_alive_checks'] += 1
                return False
        return super().alive(handle)

    def validate(self, task):
        receipt = super().validate(task)
        client = self.channels.get(task['key'])
        if self.resident and client is not None:
            if not client.get('done'):
                raise RuntimeError('Resident task has no live completion')
            slot = digest(client['launch']['slot'])
            if slot in self.idle_channels:
                raise RuntimeError('Duplicate idle resident slot')
            self.idle_channels[slot] = self.channels.pop(task['key'])
        return receipt

    def cancel(self, handle):
        client = self.channels.get(handle['key'])
        if client is not None:
            client['channel'].close()
        # EOF stops the next guard. Never claim a running CUDA call was killed,
        # and never release its full reservation without remote reconciliation.
        raise RuntimeError('Transport closed; model consumption requires reconciliation')

    def _inspect(self, host, key):
        started = time.monotonic()
        self.metrics['completion_inspections'] += 1
        try:
            return self._inspect_remote(host, key)
        finally:
            self.metrics['completion_inspection_seconds'] += time.monotonic() - started

    def _inspect_remote(self, host, key):
        if self.persistent_inspection:
            return self._inspect_persistent(host, key)
        target = self._target(host)
        with self.channel(target) as channel:
            self.metrics['inspection_connections'] += 1
            self._hello(channel, target)
            channel.wire.send(dict(kind='INSPECT', session=channel.hello['session'], key=key))
            value = channel.wire.receive()
            if value.get('kind') != 'STATE':
                raise RuntimeError('Invalid model worker inspection')
            if value['active']:
                raise RuntimeError('Worker still holds its task lock')
            return value['row']

    def _inspect_persistent(self, host, key):
        """Reuse only the read-only transport, never a previous receipt or PASS.

        A lost/invalid response poisons this host for this backend instance.
        There is no reconnect, fallback to one-shot inspection, or model grant.
        """
        if host in self.failed_inspection_hosts:
            raise ConnectionError('Inspection session failed; explicit recovery required')
        channel = self.inspection_channels.get(host)
        try:
            self._node_health(host)
            if channel is None:
                target = self._target(host)
                channel = self.channel(target).__enter__()
                self.inspection_channels[host] = channel
                self.metrics['inspection_connections'] += 1
                self._hello(channel, target)
                session = channel.hello['session']
                channel.wire.send(dict(kind='INSPECT_SESSION', session=session))
                if channel.wire.receive() != dict(kind='INSPECTION_READY', session=session):
                    raise RuntimeError('Invalid inspection handshake')
            session = channel.hello['session']
            channel.wire.send(dict(kind='INSPECT', session=session, key=key))
            value = channel.wire.receive()
            if (value.get('kind') != 'STATE' or value.get('session') != session
                    or value.get('key') != key):
                raise RuntimeError('Invalid inspection response identity')
            if value['active']:
                raise RuntimeError('Worker still holds its task lock')
            self.inspection_last_probe[host] = time.monotonic()
            return value['row']
        except BaseException:
            self.failed_inspection_hosts.add(host)
            if channel is not None:
                channel.close()
            raise

    def probe_inspections(self):
        """Keep idle read-only sessions live without granting any computation."""
        now = time.monotonic()
        for host, channel in self.inspection_channels.items():
            if host in self.failed_inspection_hosts:
                raise ConnectionError('Inspection session failed; explicit recovery required')
            if now - self.inspection_last_probe.get(host, 0.) < 15.:
                continue
            session = channel.hello['session']
            try:
                channel.wire.send(dict(kind='INSPECTION_PROBE', session=session))
                if channel.wire.receive() != dict(kind='INSPECTION_ALIVE', session=session):
                    raise RuntimeError('Inspection heartbeat changed')
                self.inspection_last_probe[host] = now
            except BaseException:
                self.failed_inspection_hosts.add(host)
                channel.close()
                raise

    def _validate_row(self, row, handle, launch):
        self._identity(row, handle, launch)
        if row['status'] != 'COMPLETE':
            raise MissingResult('Incomplete model block; reservation is retained')
        receipt = row['receipt']
        if (receipt['simulation_only'] is not (not self.production) or receipt['task_sha256'] != launch['task_sha256']
                or receipt['binding_sha256'] != self.config['binding_sha256']
                or receipt['scientific_success'] is not None):
            raise ValueError('Unbound or success-selected model receipt')
        counts = receipt['budget_consumed']
        if set(counts) != set(launch['limits']) or any(type(v) is not int or not 0 <= v <= launch['limits'][k]
                                                     for k, v in counts.items()):
            raise ValueError('Worker budget exceeded')
        if counts['episodes'] != launch['limits']['episodes'] or not receipt['result_sha256']:
            raise ValueError('Incomplete scientific block coverage')
        return deepcopy(receipt)
