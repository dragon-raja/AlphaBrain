"""Bounded concurrent preparation with single-threaded live GO authorization.

Only connection setup runs in threads. Each connection has one owner; ledger,
peer channels, PREPARE/GO/ACK and model continuations stay on the queue thread.
No worker can compute before durable intent, a fresh GO and live CHECK grants.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import time
import uuid

from .manifest import digest, verify_artifacts
from .queue import _durable_write, exclusive
from .resources import device, dispatch_guard
from .worker_fence import check_permit


class ParallelLaunches:
    def __init__(self, backend):
        self.backend = backend
        self.pool = ThreadPoolExecutor(max_workers=backend.parallel_launches,
                                       thread_name_prefix='authority-connect')
        self.pending = {}
        self.closed = False

    def _check(self, item, outcome):
        """Caller owns the short local ledger transaction; no network IO."""
        b = self.backend
        ledger, plan, path, request, clock = b.contexts[item['ticket']['key']]
        if ledger.failed or ledger.session != (request['owner'], request['epoch']):
            raise RuntimeError('Launch authority session lost')
        budget, state = ledger.read()
        permit = json.loads(path.read_text())
        check_permit(state, budget, plan, permit, request['owner'], request['epoch'],
                     clock(), production=b.production)
        row = state['tasks'][request['key']]
        if (digest(permit) != request['permit_sha256']
                or set(budget['peers']) != set(b.config['peers'])
                or row.get('worker_request') != request or row['status'] != 'RESERVED'
                or row['launch_status'] != 'RESERVED' or row['reserved'] != b.limits[request['key']]
                or row['epoch'] != request['epoch'] or row['slot'] != request['slot']
                or row['attempt'] != request['attempt'] or row['queue_sha256'] != digest(plan)):
            raise RuntimeError('Parallel launch reservation/permit changed')
        claim = row.get('worker_claim')
        if outcome is None:
            if claim is not None:
                raise RuntimeError('Duplicate or uncertain launch intent')
        elif (claim is None or claim.get('outcome') != outcome
              or claim.get('ticket') != item['ticket']
              or claim.get('request_sha256') != digest(request)
              or claim.get('handle') != item.get('handle')):
            raise RuntimeError('Parallel launch claim changed')
        return ledger, state, row

    def _save_transport(self, item):
        records = json.loads(self.backend.path.read_text())
        key = item['ticket']['key']
        old = records['attempts'].get(key)
        if old is not None and old.get('ticket') != item['ticket']:
            raise RuntimeError('Duplicate transport attempt')
        records['attempts'][key] = dict(ticket=item['ticket'], launch=item['launch'],
                                        handle=item.get('handle'), phase=item['phase'])
        _durable_write(self.backend.path, records)

    def _open(self, target):
        # This thread only creates and verifies its own connection. It cannot
        # mutate custody, probe shared wires, send PREPARE or authorize GO.
        channel = self.backend.channel(target).__enter__()
        try:
            self.backend._hello(channel, target)
            return channel
        except BaseException:
            channel.close()
            raise

    def begin(self, request, queue_root):
        b = self.backend
        if self.closed or len(self.pending) >= b.parallel_launches:
            raise RuntimeError('Parallel preparation is closed or full')
        key = request['key']
        if key in self.pending:
            raise RuntimeError('Duplicate pending task')
        ledger, plan, _, _, _ = b.contexts[key]
        task = next(t for t in plan['tasks'] if t['key'] == key)
        if request['task_sha256'] != digest(task) or request['slot'] not in task.get('eligible_slots', plan['slots']):
            raise ValueError('Unbound parallel launch')
        verify_artifacts(plan)
        item = dict(ticket=dict(kind='PENDING_AUTHORITY_LAUNCH_V1', key=key, nonce=uuid.uuid4().hex),
                    phase='CONNECTING', deadline=time.monotonic()+b.launch_timeout,
                    queue_root=Path(queue_root))
        b.probe_many(b.config['peers'])
        with exclusive(ledger.root):
            _, state, row = self._check(item, None)
            launch = dict(operation='CPU_SYNTHETIC_ONLY', key=key, task_sha256=digest(task),
                          request_sha256=digest(request), epoch=request['epoch'], owner=request['owner'],
                          attempt=request['attempt'], limits=deepcopy(row['reserved']), slot=request['slot'])
            launch.update(b.launch_fields(task, request));item['launch'] = launch
            self._save_transport(item)
            row['worker_claim'] = dict(outcome='INTENT', request_sha256=digest(request),
                                      ticket=item['ticket'], handle=None)
            ledger._save(state, dict(event='parallel_intent_before_connect', key=key))
        self.pending[key] = item  # cleanup owns it before any connection work
        client = b.idle_channels.pop(digest(request['slot']), None) if b.resident else None
        if client is not None:
            item['channel'] = client['channel']
            old = client['launch']
            if (old['owner'] != request['owner'] or old['epoch'] != request['epoch']
                    or old['slot'] != request['slot'] or digest(old['permit']) != request['permit_sha256']
                    or item['channel'].closed or item['channel'].process.poll() is not None):
                raise RuntimeError('Resident connection changed; no reconnect')
            b.metrics['resident_reuses'] += 1
        else:
            item['future'] = self.pool.submit(self._open, b._target(request['slot']['host']))
            b.metrics['worker_processes'] += 1
        b.metrics['max_pending_launches'] = max(b.metrics['max_pending_launches'], len(self.pending))
        return deepcopy(item['ticket'])

    def poll(self, ticket):
        b = self.backend
        item = self.pending.get(ticket['key'])
        if self.closed or item is None or item['ticket'] != ticket:
            raise RuntimeError('Unknown parallel launch ticket')
        if time.monotonic() >= item['deadline']:
            raise TimeoutError('Parallel launch deadline expired; retain reservation')
        ledger, plan, _, request, _ = b.contexts[ticket['key']]
        if item['phase'] == 'CONNECTING':
            if 'channel' not in item:
                if not item['future'].done():
                    return None
                item['channel'] = item['future'].result()
            # The remote GPU ownership lock is deliberately nonblocking. Only
            # one preparation per physical device may reach it; different
            # devices/hosts connect and prepare concurrently. Computation by an
            # acknowledged sibling remains allowed up to the bound capacity.
            slot = request['slot']
            if any(other is not item and other['phase'] in ('PREPARING', 'WAIT_ACK')
                   and other['launch']['slot']['host'] == slot['host']
                   and device(other['launch']['slot']) == device(slot)
                   for other in self.pending.values()):
                return None
            channel = item['channel'];session = channel.hello['session']
            b.probe_many(b.config['peers'])
            with exclusive(ledger.root):
                _, state, row = self._check(item, 'INTENT')
                item['handle'] = dict(key=ticket['key'], host=request['slot']['host'], attempt=request['attempt'],
                    launch_sha256=digest(item['launch']), session=session, store_id=channel.hello['store_id'])
                row['worker_claim']['handle'] = deepcopy(item['handle'])
                item['phase'] = 'PREPARING';self._save_transport(item)
                ledger._save(state, dict(event='parallel_prepare', key=ticket['key']))
            channel.wire.send(dict(kind='PREPARE', session=session, key=ticket['key'], launch=item['launch']))
            return None
        channel = item['channel'];handle = item['handle'];stamp = handle['launch_sha256']
        response = channel.wire.receive_ready()
        if response is None:
            return None
        if item['phase'] == 'PREPARING':
            if response != dict(kind='READY', session=handle['session'], launch_sha256=stamp):
                raise RuntimeError('Parallel preparation response changed')
            b.probe_many(b.config['peers'])
            with dispatch_guard(item['queue_root'], plan, request['slot']) as allowed:
                if not allowed:
                    raise RuntimeError('GPU drained during preparation; no GO')
                with exclusive(ledger.root):
                    _, state, row = self._check(item, 'INTENT')
                    row['worker_claim']['outcome'] = 'GRANTED'
                    ledger._save(state, dict(event='authority_grant_before_go', key=ticket['key']))
                    # Persistent checkpoint/fsync may outlive the permit.
                    self._check(item, 'GRANTED')
                    channel.wire.send(dict(kind='GO', session=handle['session'], launch_sha256=stamp))
            item['phase'] = 'WAIT_ACK';self._save_transport(item)
            return None
        if (item['phase'] != 'WAIT_ACK' or response.get('kind') != 'ACK'
                or response.get('session') != handle['session'] or response.get('launch_sha256') != stamp):
            raise RuntimeError('Unknown parallel launch acknowledgement')
        b.probe_many(b.config['peers'])
        with exclusive(ledger.root):
            _, state, row = self._check(item, 'GRANTED')
            b.accept_ack(channel, response['row'], handle, item['launch'])
            row['worker_claim']['outcome'] = 'ACKNOWLEDGED'
            row.update(launch_status='RUNNING', handle=deepcopy(handle))
            ledger._save(state, dict(event='backend_start_ack', key=ticket['key']))
        item['phase'] = 'RUNNING';self._save_transport(item)
        del self.pending[ticket['key']]
        return deepcopy(handle)

    @staticmethod
    def _close_future(future):
        if future.cancelled():
            return
        try:
            future.result().close()
        except BaseException:
            pass  # No late completion can authorize PREPARE or GO.

    def cancel(self, ticket):
        item = self.pending.get(ticket['key'])
        if item is None or item['ticket'] != ticket:
            raise RuntimeError('Unknown cancellation ticket')
        if 'channel' in item:
            item['channel'].close()
        elif 'future' in item:
            item['future'].cancel()
            item['future'].add_done_callback(self._close_future)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for item in self.pending.values():
            self.cancel(item['ticket'])
        self.pool.shutdown(wait=False, cancel_futures=True)
