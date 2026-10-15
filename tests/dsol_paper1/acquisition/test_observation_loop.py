from dataclasses import replace
import pytest

from AlphaBrain.research.dsol.acquisition.observation_loop import EpisodeCursor
from AlphaBrain.research.dsol.data.active_observation import AcquisitionReceipt
from AlphaBrain.research.dsol.selectors.observation_planning import choose_observation
from tests.dsol_paper1.helpers.active_observation import (
    FakeEvaluator, FakeFeasibility, FakePredictor, context, observation, options,
)


def cursor(**changes):
    return replace(EpisodeCursor(context(), 0, 20, 2), **changes)


def receipt(status="arrived", tick=2):
    return AcquisitionReceipt("move", 0, tick, status, observation(tick))


def test_cpu_plan_acquire_operate_cycle():
    before = cursor()
    plan = choose_observation(before.context, options(), predictor=FakePredictor(),
                              evaluator=FakeEvaluator(), feasibility=FakeFeasibility(),
                              max_prediction_calls=2)
    acquired = before.acquire(plan.selected, receipt())
    assert acquired.execution_noise_address == before.execution_noise_address
    assert acquired.context.current.features == observation(2).features
    after = acquired.operation(observation(7), commands=((0.1,),), terminated=False)
    assert after.replan_index == 1
    assert after.context.current.tick == 7
    assert len(after.acquisitions) == 1
    assert after.context.executed_operations[0].commands == ((0.1,),)
    assert len(before.context.history) == 1


def test_failed_move_is_retained_charged_and_not_continue():
    after = cursor().acquire(options()[1], receipt("failed", 1))
    assert len(after.acquisitions) == 1
    assert after.acquisitions[0].status == "failed"
    assert after.context.current.tick == 1
    assert after.replan_index == 0


def test_noop_uses_same_operation_noise_and_history():
    left, right = cursor(), cursor()
    assert left.execution_noise_address == right.execution_noise_address
    assert left.operation(observation(5), commands=((0.1,),), terminated=False) == right.operation(observation(5), commands=((0.1,),), terminated=False)


def test_continue_is_not_a_camera_query():
    with pytest.raises(ValueError, match="continue"):
        cursor().acquire(options()[0], receipt())


@pytest.mark.parametrize("change", [{"terminated": True}, {"deadline_tick": 0}, {"max_acquisitions": 0}])
def test_budget_and_terminal_gates(change):
    with pytest.raises(ValueError):
        cursor(**change).acquire(options()[1], receipt())


def test_arrival_deadline_stops_future_actions():
    after = cursor(deadline_tick=2).acquire(options()[1], receipt())
    assert after.terminated
    with pytest.raises(ValueError):
        after.operation(observation(5), commands=((0.1,),), terminated=False)


@pytest.mark.parametrize("change", [{"option_id": "other"}, {"start_tick": 1}])
def test_receipt_binding(change):
    with pytest.raises(ValueError):
        cursor().acquire(options()[1], replace(receipt(), **change))


def test_success_cannot_arrive_early():
    with pytest.raises(ValueError, match="duration"):
        cursor().acquire(options()[1], receipt(tick=1))
