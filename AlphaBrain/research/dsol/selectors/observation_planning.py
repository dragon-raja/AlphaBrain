"""Replaceable short-horizon observation planner; no learned weights included.

This is model-assisted option ranking, NOT an implemented RL trainer. Values
must include the declared continuation and acquisition costs. Proposal sources
may be rules or a learned actor; the planner never queries candidate RGB caches.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

from ..data.active_observation import DecisionContext, ObservationOption, PredictedObservation


class ObservationPredictor(Protocol):
    def predict(self, context: DecisionContext, option: ObservationOption) -> PredictedObservation: ...


class OutcomeEvaluator(Protocol):
    def value(self, context: DecisionContext, option: ObservationOption,
              prediction: PredictedObservation | None) -> float: ...


class OptionProposer(Protocol):
    def propose(self, context: DecisionContext) -> tuple[ObservationOption, ...]: ...


class FeasibilityChecker(Protocol):
    def allows(self, context: DecisionContext, option: ObservationOption) -> bool: ...


@dataclass(frozen=True)
class Plan:
    selected: ObservationOption
    values: tuple[tuple[str, float], ...]
    rejected: tuple[str, ...]


def choose_observation(context: DecisionContext, options: tuple[ObservationOption, ...], *,
                       predictor: ObservationPredictor | None, evaluator: OutcomeEvaluator,
                       feasibility: FeasibilityChecker, max_prediction_calls: int,
                       min_advantage: float = 0.0) -> Plan:
    """Compare net expected returns against continue; tie favors no movement.

    A missing predictor is an explicit direct-value baseline. No silent
    truncation, no inferred reachability, no calibrated-confidence claim.
    """
    if type(max_prediction_calls) is not int or max_prediction_calls < 0:
        raise ValueError("invalid prediction budget")
    if not math.isfinite(min_advantage) or min_advantage < 0:
        raise ValueError("invalid advantage margin")
    if not options or len({o.option_id for o in options}) != len(options):
        raise ValueError("options must be nonempty and uniquely identified")
    if any(o.embodiment_contract != context.embodiment_contract for o in options):
        raise ValueError("option embodiment mismatch")
    baseline = [o for o in options if o.kind == "continue"]
    if len(baseline) != 1:
        raise ValueError("exactly one continue baseline is required")
    allowed, rejected = [], []
    for option in options:
        legal = feasibility.allows(context, option)
        if type(legal) is not bool:
            raise ValueError("feasibility must return an explicit bool")
        (allowed if legal else rejected).append(option)
    if baseline[0] not in allowed:
        raise ValueError("continue is unavailable; runtime must handle termination/safety")
    if predictor is not None and sum(o.kind != "continue" for o in allowed) > max_prediction_calls:
        raise ValueError("proposal set exceeds prediction budget; no implicit truncation")
    values = []
    for option in allowed:
        prediction = None
        if option.kind != "continue" and predictor is not None:
            prediction = predictor.predict(context, option)
            if type(prediction) is not PredictedObservation or (
                prediction.encoder_id != context.versions.encoder or
                prediction.predictor_id != context.versions.predictor or
                len(prediction.features) != len(context.current.features)
            ):
                raise ValueError("prediction identity/dimension mismatch")
        value = evaluator.value(context, option, prediction)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("evaluator returned a nonfinite/non-scalar value")
        values.append((option.option_id, float(value)))
    scores = dict(values)
    selected = baseline[0]
    best = max(allowed, key=lambda option: scores[option.option_id])
    if scores[best.option_id] > scores[selected.option_id] + min_advantage:
        selected = best
    return Plan(selected, tuple(values), tuple(o.option_id for o in rejected))
