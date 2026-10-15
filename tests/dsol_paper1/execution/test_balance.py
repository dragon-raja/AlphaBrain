import pytest
from AlphaBrain.research.dsol.execution.balance import plan_allocation


def test_balance_moves_towards_equal_finish_not_fixed_model_hosts():
    result=plan_allocation(dict(pi=120*3600,xr=100*3600),dict(pi=16,xr=8),24,
                           qualified=dict(pi=True,xr=True))
    assert result['allocation']==dict(pi=13,xr=11)
    assert result['change_recommended']
    assert result['proposed_remaining_hours']<result['current_remaining_hours']


def test_cold_pilot_cannot_trigger_migration():
    result=plan_allocation(dict(pi=120,xr=100),dict(pi=16,xr=8),24,
                           qualified=dict(pi=False,xr=True))
    assert not result['change_recommended']
    assert result['status']=='WAITING_FOR_REPRESENTATIVE_TIMING'


def test_balanced_layout_unchanged_and_finished_model_releases_cards():
    result=plan_allocation(dict(pi=160,xr=80),dict(pi=16,xr=8),24,qualified=dict(pi=True,xr=True))
    assert not result['change_recommended']
    result=plan_allocation(dict(pi=0,xr=80),dict(pi=16,xr=8),24,qualified=dict(pi=True,xr=True))
    assert result['allocation']==dict(pi=0,xr=24)
    assert result['change_recommended']


def test_invalid_resource_budget_rejected():
    with pytest.raises(ValueError):
        plan_allocation(dict(pi=10),dict(pi=8),24,qualified=dict(pi=True))
