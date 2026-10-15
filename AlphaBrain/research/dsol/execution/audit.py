"""Bounded, read-only scientific receipt audit while production continues."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import time

from .queue import load


def audit_results(root, backend, *, per_status=2, include_running=False):
    if type(per_status) is not int or per_status < 1:
        raise ValueError('Positive per-status sample size required')
    plan, state = load(root)
    task_map = {t['key']: t for t in plan['tasks']}
    rows = []; started = time.monotonic()
    states = ('COMPLETE', 'RUNNING') if include_running else ('COMPLETE',)
    if include_running and 'allow_partial' not in inspect.signature(backend.validate).parameters:
        raise ValueError('Backend does not support committed partial-result validation')
    for status in states:
        keys = sorted(k for k,r in state['tasks'].items() if r['status']==status)[:per_status]
        for key in keys:
            task=deepcopy(task_map[key]); old=state['tasks'][key]
            if status=='RUNNING':
                status_path=Path(task['output'])/'status.json'
                if not status_path.exists() or json.loads(status_path.read_text()).get('status')=='STARTING':
                    rows.append(dict(key=key,task_status=status,validation='NOT_READY',
                                     reason='Worker startup has not committed a result; not a numerical failure'))
                    continue
            try:
                receipt=backend.validate(task,allow_partial=True) if status=='RUNNING' else backend.validate(task)
                if status=='COMPLETE' and receipt != old['receipt']:
                    raise ValueError('Completed receipt changed')
                rows.append(dict(key=key,task_status=status,validation='PASS',receipt=receipt))
            except Exception as exc:
                rows.append(dict(key=key,task_status=status,validation='HOLD',error=str(exc),error_type=type(exc).__name__))
    passed=bool(rows) and all(r['validation']=='PASS' for r in rows)
    return dict(status='PASS' if passed else ('HOLD' if any(r['validation']=='HOLD' for r in rows) else 'INCOMPLETE'),
                manifest_sha256=state['manifest_sha256'],rows=rows,seconds=time.monotonic()-started,
                scope='Deterministic task-key sample of committed outputs; not every trial or bitwise renderer certification')
