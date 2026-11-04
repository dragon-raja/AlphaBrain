import pytest
from AlphaBrain.research.dsol.execution.attempt_supervisor import RecoverySupervisor


def ready(key='failed'):
    return dict(status='READY', failure_id=key, all_workers_stopped=True,
                same_node_generation=True, inputs_verified=True, checkpoint_verified=True,
                unknown_budget_retained=True, within_original_budget=True,
                automatic_refund=False)


def process(_):
    return dict(pid=123, start_ticks=456, boot_id='new')


def test_resume_only_once_and_contract_survives_restart(tmp_path):
    s=RecoverySupervisor(tmp_path,deadline=100,clock=lambda:1)
    assert s.attempt('failed',audit=ready,resume=process)['status']=='RESUMED'
    s=RecoverySupervisor(tmp_path,deadline=100,clock=lambda:2)
    with pytest.raises(RuntimeError,match='Existing'):s.attempt('failed',audit=ready,resume=process)
    with pytest.raises(ValueError):RecoverySupervisor(tmp_path,deadline=200)


@pytest.mark.parametrize('field', ['all_workers_stopped','same_node_generation',
    'inputs_verified','checkpoint_verified','unknown_budget_retained','within_original_budget'])
def test_any_unresolved_gate_never_launches(tmp_path,field):
    s=RecoverySupervisor(tmp_path,deadline=100,clock=lambda:1);p=ready();p[field]=False
    def forbidden(_):pytest.fail('unsafe launch')
    with pytest.raises(RuntimeError):s.attempt('failed',audit=lambda:p,resume=forbidden)
    assert s.read()['attempts']['failed']['status']=='HOLD'


def test_unknown_launch_is_not_repeated(tmp_path):
    s=RecoverySupervisor(tmp_path,deadline=100,clock=lambda:1)
    def lost(_):raise ConnectionError('after process spawn')
    with pytest.raises(ConnectionError):s.attempt('failed',audit=ready,resume=lost)
    assert s.read()['attempts']['failed']['status']=='UNKNOWN'
    with pytest.raises(RuntimeError,match='Existing'):s.attempt('failed',audit=ready,resume=process)


def test_deadline_during_audit_and_budget_limit(tmp_path):
    now=[1];s=RecoverySupervisor(tmp_path,deadline=100,max_recoveries=1,clock=lambda:now[0])
    def expire():now[0]=100;return ready()
    with pytest.raises(RuntimeError,match='Deadline'):s.attempt('failed',audit=expire,resume=process)
    now[0]=2
    with pytest.raises(RuntimeError,match='limit'):s.attempt('different',audit=lambda:ready('different'),resume=process)
