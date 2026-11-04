"""Actual GR1 CPU remote-agent entry with the existing controller and ledger."""
import json
import pytest
from AlphaBrain.research.dsol.execution import authority_wire
from AlphaBrain.research.dsol.execution.controlled_queue import recover_controlled_queue
from tests.dsol_paper1.helpers.model_cases import make_model_case,run_model_case
from tests.dsol_paper1.helpers.controller_cases import OWNER,NEXT
from benchmarks.Robocasa_tabletop import continuous_authority as gr1


@pytest.fixture
def case(tmp_path):
    value=make_model_case(tmp_path,benchmark='gr1',task_count=1)
    try:yield value
    finally:value['backend'].close_channels()


def test_gr1_actual_remote_entry_uses_shared_ledger_and_native_query(case):
    run_model_case(case)
    rows=list((case['worker']/'tasks').glob('*.json'))
    row=json.loads(rows[0].read_text())
    assert row['status']=='COMPLETE' and row['starts']==1
    assert row['receipt']['simulation_only'] is True
    assert row['receipt']['budget_consumed']=={'calls':2,'episodes':1}
    spec=case['binding']['tasks'][row['launch']['key']]
    from pathlib import Path
    proof=json.loads((Path(spec['output'])/'cpu-receipt.json').read_text())
    assert proof['real_model_calls']==0 and proof['feature_shape']==[4608]
    gr1.verify_completed(case['binding'],row)
    with pytest.raises(ValueError,match='explicit recovery'):
        run_model_case(case)  # Completed queue cannot be launched again.
    assert json.loads(rows[0].read_text())['starts']==1


@pytest.mark.parametrize('fault',['disconnect_after_ack','stale_epoch','lost_done','partial_result'])
def test_gr1_remote_unknown_requires_explicit_audit(case,monkeypatch,fault):
    original=authority_wire.Wire.receive
    def receive(self):
        value=original(self)
        if value.get('kind')=='ACK' and fault=='disconnect_after_ack':raise ConnectionError('lost ACK')
        if value.get('kind')=='CHECK' and fault=='stale_epoch':
            permit=json.loads(case['permit'].read_text());permit['epoch']=0
            case['permit'].write_text(json.dumps(permit))
        if value.get('kind')=='DONE' and fault in ('lost_done','partial_result'):
            if fault=='partial_result':
                p=case['worker']/'tasks'/(value['row']['launch']['key']+'.json')
                row=json.loads(p.read_text());row['status']='UNKNOWN';row.pop('receipt')
                p.write_text(json.dumps(row))
            raise ConnectionError('lost completion')
        return value
    monkeypatch.setattr(authority_wire.Wire,'receive',receive)
    with pytest.raises((RuntimeError,OSError)):run_model_case(case)
    monkeypatch.setattr(authority_wire.Wire,'receive',original)
    state=next(iter(case['ledger'].read()[1]['tasks'].values()))
    assert state['status']=='RESERVED' and state['reserved']['calls']==2
    with pytest.raises(RuntimeError):run_model_case(case)
    if fault=='lost_done':
        assert recover_controlled_queue(case['queue'],case['ledger'],OWNER,1,NEXT,
            case['backend'],simulate=True)==2
    else:
        with pytest.raises(RuntimeError):
            recover_controlled_queue(case['queue'],case['ledger'],OWNER,1,NEXT,
                case['backend'],simulate=True)
    assert len(list((case['worker']/'tasks').glob('*.json')))==1


def test_gr1_adapter_never_accepts_production_mode(case,tmp_path):
    value=dict(case['binding'],mode='libero');path=tmp_path/'wrong.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='CPU contract'):gr1.load_binding(path)
    with pytest.raises(RuntimeError,match='not registered'):gr1.execute_block()
    case['backend'].production=True
    target=next(iter(case['backend'].config['peers'].values()))
    with pytest.raises(ValueError,match='CPU-contract'):case['backend'].channel(target)


def test_gr1_actual_agent_composes_actor_child_search_and_operation(tmp_path):
    pytest.importorskip('torch');pytest.importorskip('gymnasium')
    c=make_model_case(tmp_path,benchmark='gr1_trial',task_count=1)
    try:
        run_model_case(c)
        row=json.loads(next((c['worker']/'tasks').glob('*.json')).read_text())
        assert row['status']=='COMPLETE' and row['receipt']['budget_consumed']==dict(calls=4,episodes=1)
        from pathlib import Path
        p=Path(c['binding']['tasks'][row['launch']['key']]['output'])/'cpu-receipt.json'
        result=json.loads(p.read_text())
        assert result['real_model_calls']==0 and result['child_exited']
        assert result['trial']['operation']['actions']==17 and result['trial']['search']['queries']==1
        gr1.verify_completed(c['binding'],row)
    finally:c['backend'].close_channels()


def test_gr1_actual_agent_runs_two_trial_block(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('gymnasium')
    c=make_model_case(tmp_path, benchmark='gr1_block', task_count=1)
    try:
        run_model_case(c)
        row=json.loads(next((c['worker']/'tasks').glob('*.json')).read_text())
        assert row['status']=='COMPLETE'
        assert row['receipt']['budget_consumed']==dict(calls=8,episodes=2)
        from pathlib import Path
        output=Path(c['binding']['tasks'][row['launch']['key']]['output'])
        receipt=json.loads((output/'cpu-receipt.json').read_text())
        assert receipt['real_model_calls']==0 and receipt['completed_episodes']==2
        assert len(receipt['block']['episodes'])==2
        gr1.verify_completed(c['binding'],row)
    finally:
        c['backend'].close_channels()


def test_gr1_actual_agent_preflights_bound_plans_and_settles_only_actual_calls(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('gymnasium')
    c=make_model_case(tmp_path, benchmark='gr1_prepared', task_count=1)
    try:
        run_model_case(c)
        row=json.loads(next((c['worker']/'tasks').glob('*.json')).read_text())
        assert row['launch']['limits']==dict(calls=36,episodes=2)
        assert row['receipt']['budget_consumed']==dict(calls=8,episodes=2)
        assert row['status']=='COMPLETE' and row['receipt']['simulation_only'] is True
        gr1.verify_completed(c['binding'],row)
    finally:
        c['backend'].close_channels()
