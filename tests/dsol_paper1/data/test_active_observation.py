from dataclasses import asdict, replace
import json

import pytest

from AlphaBrain.research.dsol.data.active_observation import (
    ExecutedOperation, OutcomeExample, PredictedObservation, training_batch,
)
from tests.dsol_paper1.helpers.active_observation import context, observation, options


def test_history_is_immutable_and_serializable():
    initial = context()
    updated = initial.observe(observation(2))
    assert len(initial.history) == 1
    assert len(updated.history) == 2
    assert json.loads(json.dumps(asdict(updated), allow_nan=False))["versions"]["operation"] == "operation-v1"


def test_future_executed_commands_cannot_enter_context():
    with pytest.raises(ValueError, match="past"):
        context(executed_operations=(ExecutedOperation(0, 2, ((0.1,),)),))


@pytest.mark.parametrize("new", [
    observation(0), observation(2, episode_id="another"),
    observation(2, encoder_id="another"), observation(2, features=(0.0,)),
    PredictedObservation("encoder-v1", "predictor-v1", (0.1, 0.2)),
])
def test_illegal_history_rejected(new):
    with pytest.raises(ValueError):
        context().observe(new)


@pytest.mark.parametrize("features", [(float("nan"),), (float("inf"),), [0.1], (True,)])
def test_invalid_features_rejected(features):
    with pytest.raises(ValueError):
        observation(features=features)


@pytest.mark.parametrize("change", [
    {"duration_ticks": -1}, {"duration_ticks": True}, {"duration_ticks": 0},
    {"kind": "move_with_execution"}, {"target": []},
])
def test_invalid_option_rejected(change):
    with pytest.raises(ValueError):
        replace(options()[1], **change)


def example(ctx=None):
    return OutcomeExample(context() if ctx is None else ctx, options()[1], "execution-0", True, 50, 0.8)


def batch(items):
    return training_batch(items, operation_id="operation-v1", encoder_id="encoder-v1",
                          continuation_policy_id="continuation-v1")


def test_training_admission():
    assert batch((example(),)) == (example(),)
    with pytest.raises(ValueError, match="duplicate"):
        batch((example(), example()))


@pytest.mark.parametrize("split", ["development", "test"])
def test_evaluation_cannot_train(split):
    with pytest.raises(ValueError, match="cannot enter training"):
        batch((example(context(split=split)),))


@pytest.mark.parametrize("field", ["operation", "encoder", "continuation"])
def test_stale_value_labels_rejected(field):
    ctx = context()
    if field == "continuation":
        ctx = replace(ctx, continuation_policy_id="other")
    elif field == "encoder":
        ctx = replace(ctx, versions=replace(ctx.versions, encoder="other"),
                      history=(observation(encoder_id="other"),))
    else:
        ctx = replace(ctx, versions=replace(ctx.versions, operation="other"))
    with pytest.raises(ValueError, match="mixed"):
        batch((example(ctx),))
