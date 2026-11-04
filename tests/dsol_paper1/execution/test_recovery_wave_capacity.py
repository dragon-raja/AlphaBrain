"""Explicit CPU integration at the planned larger resident-wave task count."""
import json
import os
import pytest
from AlphaBrain.research.dsol.execution.controlled_queue import run_controlled_queue
from tests.dsol_paper1.helpers.controller_cases import OWNER
from tests.dsol_paper1.helpers.model_cases import make_model_case


@pytest.mark.skipif(os.environ.get('DSOL_LARGE_CPU_QUEUE') != '1', reason='Explicit larger CPU integration required')
def test_288_blocks_settle_once_with_resident_process_reuse(tmp_path):
    # Six local fake slots keep this verification cheap on center. The live
    # campaign retains 48 admitted GPU slots; this is a queue-length test.
    slots = [dict(host='fresh', id=f'cpu-{i}') for i in range(6)]
    case = make_model_case(tmp_path, slots=slots, checks=2, resident=True, task_count=288)
    result = run_controlled_queue(case['queue'], case['ledger'], OWNER, 1, case['permit'],
                                 case['backend'], simulate=True, poll_seconds=.01, timeout_seconds=180)
    assert result['status'] == 'COMPLETE'
    _, state = case['ledger'].read()
    assert len(state['tasks']) == 288 and state['dispatch_enabled'] is False
    assert all(r['status']=='COMPLETE' and r['consumed']==r['reserved'] for r in state['tasks'].values())
    rows=[json.loads(p.read_text()) for p in (case['worker']/'tasks').glob('*.json')]
    assert len(rows)==288 and all(r['starts']==1 for r in rows)
    assert case['backend'].metrics['worker_processes'] <= 6
    assert case['backend'].metrics['resident_reuses'] >= 282
    assert all(json.loads(p.read_text())['status']=='CLOSED' for p in (case['worker']/'residents').glob('*.json'))
