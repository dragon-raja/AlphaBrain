import pytest
from AlphaBrain.research.dsol.execution.balance import plan_allocation
from AlphaBrain.research.dsol.execution.balance import assign_pending_blocks


def test_cost_first_balances_long_tail_preserving_science():
    blocks=[dict(rows=[dict(trial_id=str(i))],weight=w) for i,w in enumerate([8,7,6,5,4])]
    result=assign_pending_blocks(blocks,{'a':1,'b':1},cost=lambda b:b['weight'])
    assert max(result['estimated_finish'].values())==17
    assert sorted(b['rows'][0]['trial_id'] for b in result['blocks'])==list('01234')
    assert all('host' not in b for b in blocks)
    assert result['dispatch_authority'] is False


def test_heterogeneous_capacity_and_live_work_rejected():
    blocks=[dict(rows=[i]) for i in range(12)]
    result=assign_pending_blocks(blocks,{'a':1,'b':2},cost=lambda _:1)
    assert result['estimated_work']=={'a':4,'b':8}
    with pytest.raises(ValueError):assign_pending_blocks([dict(status='PENDING')],{'a':1},cost=lambda _:1)
    with pytest.raises(ValueError):assign_pending_blocks(blocks,{'a':0},cost=lambda _:1)
    with pytest.raises(ValueError):assign_pending_blocks(blocks,{'a':1},cost=lambda _:float('nan'))


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
