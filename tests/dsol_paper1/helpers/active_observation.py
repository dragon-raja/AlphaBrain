"""Synthetic fixtures only; not observations, outcomes or weights from a robot."""
from dataclasses import replace

from AlphaBrain.research.dsol.data.active_observation import (
    AcquiredObservation, DecisionContext, ModelVersions, ObservationOption,
    PredictedObservation,
)


def observation(tick=0, **changes):
    return replace(AcquiredObservation("episode", f"rgb-{tick}", tick, "encoder-v1",
                                      (0.1, 0.2), (0.0,), (0.0,)), **changes)


def context(**changes):
    return replace(DecisionContext("synthetic task", "train",
                   ModelVersions("operation-v1", "encoder-v1", "predictor-v1", "value-v1", "actor-v1"),
                   "continuation-v1", "synthetic-rig", (observation(),)), **changes)


def options():
    return (
        ObservationOption("continue", "continue", "synthetic-rig", (), 0),
        ObservationOption("move", "move_and_hold", "synthetic-rig", (0.1,), 2),
        ObservationOption("hold", "matched_hold", "synthetic-rig", (), 2),
    )


class FakePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, context, option):
        self.calls.append(option.option_id)
        return PredictedObservation(context.versions.encoder, context.versions.predictor, (0.3, 0.4))


class FakeEvaluator:
    def __init__(self, scores=None):
        self.scores = {"continue": 0.4, "move": 0.7, "hold": 0.3} if scores is None else scores

    def value(self, context, option, prediction):
        return self.scores[option.option_id]


class FakeFeasibility:
    def __init__(self, blocked=()):
        self.blocked = blocked

    def allows(self, context, option):
        return option.option_id not in self.blocked
