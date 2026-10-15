from dataclasses import replace
import pytest

from AlphaBrain.research.dsol.selectors.observation_planning import choose_observation
from tests.dsol_paper1.helpers.active_observation import (
    FakeEvaluator, FakeFeasibility, FakePredictor, context, options,
)


def plan(**changes):
    arguments = dict(predictor=FakePredictor(), evaluator=FakeEvaluator(),
                     feasibility=FakeFeasibility(), max_prediction_calls=2)
    arguments.update(changes)
    return choose_observation(context(), options(), **arguments)


def test_model_assisted_plan_only_predicts_legal_options():
    predictor = FakePredictor()
    result = plan(predictor=predictor, feasibility=FakeFeasibility(("hold",)))
    assert result.selected.option_id == "move"
    assert predictor.calls == ["move"]
    assert result.rejected == ("hold",)


def test_no_prediction_ablation():
    assert plan(predictor=None, max_prediction_calls=0).selected.option_id == "move"


def test_tie_and_margin_preserve_continue():
    assert plan(evaluator=FakeEvaluator({"continue": .7, "move": .7, "hold": .3})).selected.kind == "continue"
    assert plan(min_advantage=.4).selected.kind == "continue"


def test_budget_is_not_silent_truncation():
    predictor = FakePredictor()
    with pytest.raises(ValueError, match="budget"):
        plan(predictor=predictor, max_prediction_calls=1)
    assert predictor.calls == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_bad_value_rejected(bad):
    with pytest.raises(ValueError, match="evaluator"):
        plan(evaluator=FakeEvaluator({"continue": bad, "move": .7, "hold": .3}))


def test_prediction_space_must_match():
    class WrongPredictor(FakePredictor):
        def predict(self, context, option):
            return replace(super().predict(context, option), encoder_id="wrong")
    with pytest.raises(ValueError, match="identity"):
        plan(predictor=WrongPredictor())


def test_continue_cannot_disappear():
    with pytest.raises(ValueError, match="continue"):
        plan(feasibility=FakeFeasibility(("continue",)))
