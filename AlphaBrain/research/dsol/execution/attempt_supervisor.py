"""Finite, audit-gated recovery orchestration; no backend or task replay logic."""
import json
from pathlib import Path
import time

from .authority_wire import require_local
from .queue import exclusive, _durable_write


class RecoverySupervisor:
    """A durable once-only claim around explicit audit and resume callbacks.

    A crashed/unknown launch is never invoked again. The adapter must audit
    exact process exit, inputs, checkpoints and retained consumption, and launch
    through the ordinary queue with a new binding. This layer grants no GPU or
    model-call permission. Limits are fixed at creation, not reset on restart.
    """
    def __init__(self, root, *, deadline, max_recoveries=2, clock=time.time):
        if type(max_recoveries) is not int or not 1 <= max_recoveries <= 3:
            raise ValueError('One to three explicitly bounded recoveries required')
        if type(deadline) not in (int, float) or not 0 < deadline < float('inf'):
            raise ValueError('Finite deadline required')
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        require_local(self.root)
        self.clock, self.path = clock, self.root / 'state.json'
        contract = dict(deadline=deadline, max_recoveries=max_recoveries)
        with exclusive(self.root):
            if self.path.exists():
                if self.read()['contract'] != contract:
                    raise ValueError('Recovery contract cannot change')
            else:
                _durable_write(self.path, dict(contract=contract, attempts={}))

    def read(self):
        return json.loads(self.path.read_text())

    def attempt(self, failure_id, *, audit, resume):
        if not isinstance(failure_id, str) or not failure_id:
            raise ValueError('Exact failed execution identity required')
        with exclusive(self.root):
            state = self.read(); contract = state['contract']
            if self.clock() >= contract['deadline']:
                raise RuntimeError('Recovery deadline expired')
            if failure_id in state['attempts']:
                raise RuntimeError('Existing recovery claim requires explicit reconciliation')
            if len(state['attempts']) >= contract['max_recoveries']:
                raise RuntimeError('Recovery limit exhausted')
            row = dict(status='AUDITING', started=self.clock())
            state['attempts'][failure_id] = row
            _durable_write(self.path, state)
            try:
                proof = audit()
                required = ('all_workers_stopped', 'same_node_generation',
                            'inputs_verified', 'checkpoint_verified',
                            'unknown_budget_retained', 'within_original_budget')
                if (proof.get('status') != 'READY' or proof.get('failure_id') != failure_id
                        or any(proof.get(k) is not True for k in required)
                        or proof.get('automatic_refund') is not False):
                    raise RuntimeError('Recovery audit did not authorize this exact attempt')
                if self.clock() >= contract['deadline']:
                    raise RuntimeError('Deadline expired during audit')
                row.update(status='LAUNCH_INTENT', proof=proof)
                _durable_write(self.path, state)
                process = resume(proof)
                if not isinstance(process, dict) or not all(k in process for k in ('pid','start_ticks','boot_id')):
                    raise RuntimeError('Missing new process birth identity')
                row.update(status='RESUMED', process=process, finished=self.clock())
                _durable_write(self.path, state)
                return row
            except BaseException as error:
                row.update(status='UNKNOWN' if row['status']=='LAUNCH_INTENT' else 'HOLD',
                           error=type(error).__name__, finished=self.clock())
                _durable_write(self.path, state)
                raise
