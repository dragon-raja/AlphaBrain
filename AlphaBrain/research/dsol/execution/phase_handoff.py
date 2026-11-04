"""Audited phase-boundary handoff, without replaying or refunding unknown work.

Reservation barriers use the *existing* PhaseBudget locks. They stop new phases,
not the model-call authority of already reserved phases. The caller must retire
all old outer owners before releasing a barrier and granting their resources.
No signals, GPU operations or budget mutations occur in this module.
"""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import time

from .authority_wire import require_local
from .controller_ledger import ControllerLedger
from .manifest import file_digest
from ..data.artifacts import read


@contextmanager
def reservation_barrier(roots, *, timeout=30):
    """Acquire all existing budget locks; release partial acquisitions on retry."""
    paths = sorted({Path(p).resolve() for p in roots})
    if not paths or timeout <= 0:
        raise ValueError('Explicit existing budget roots and bounded timeout required')
    for p in paths:
        require_local(p)
        if not p.is_dir() or not (p/'coordinator.lock').is_file():
            raise ValueError('Missing authoritative budget lock')
    deadline = time.monotonic() + timeout
    handles = []
    try:
        while True:
            try:
                for p in paths:
                    h = (p/'coordinator.lock').open('a')
                    handles.append(h)
                    fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                for h in reversed(handles): h.close()
                handles.clear()
                if time.monotonic() >= deadline:
                    raise TimeoutError('Reservation barrier remained occupied')
                time.sleep(.05)
        yield
    finally:
        for h in reversed(handles): h.close()


def completed_phase(row, execution_config, phase):
    """Validate a phase receipt against its authoritative local task ledger.

    Missing completion is pending, never evidence of zero consumption. The
    caller may settle a reservation only after its owners have been fenced.
    """
    if Path(phase).name != phase or phase in ('', '.', '..'):
        raise ValueError('Single phase component required')
    campaign = read(row['campaign'])
    if file_digest(row['campaign']) != row['campaign_sha256']:
        raise ValueError('Changed reserved campaign')
    evidence = Path(execution_config['output'])/'execution'/(phase+'-real')
    receipt = evidence/'completion.json'
    if not receipt.exists():
        return None
    result = read(receipt)
    local = Path(execution_config['local_root'])/(phase+'-real')/'center'/'custody'
    require_local(local)
    plan, state = ControllerLedger(local).read()
    counts = {k: sum(t['consumed'][k] for t in state['tasks'].values())
              for k in campaign['limits']}
    calls = sum(counts[k] for k in ('score', 'paired_execution', 'nominal_execution'))
    if (plan['campaign_sha256'] != row['campaign_sha256']
            or set(state['tasks']) != {t['key'] for t in plan['tasks']}
            or not state['tasks'] or state['dispatch_enabled'] is not False
            or any(t['status'] != 'COMPLETE' for t in state['tasks'].values())
            or result['status'] != 'COMPLETE' or result['dispatch_enabled'] is not False
            or result['cpu'] is not False or result['phase'] != phase
            or result['counts'] != counts or result['actual_model_calls'] != calls
            or result['actual_episodes'] != counts['episodes']
            or calls > row['reserved_model_calls']):
        raise ValueError('Unsettled or inconsistent phase evidence')
    return dict(result=result, receipt=str(receipt), receipt_sha256=file_digest(receipt),
                local_ledger=str(local/'ledger.json'),
                local_ledger_sha256=file_digest(local/'ledger.json'))


def remaining_schedules(expected, completed, attempted):
    """Partition frozen schedule IDs; an attempted but unsettled ID blocks all."""
    if len(expected) != len(set(expected)):
        raise ValueError('Duplicate expected schedule')
    expected_set, completed, attempted = set(expected), set(completed), set(attempted)
    if not completed <= attempted <= expected_set:
        raise ValueError('Unknown or inconsistent schedule identity')
    if attempted - completed:
        raise ValueError('Unsettled attempt prohibits continuation')
    return [s for s in expected if s not in completed]
