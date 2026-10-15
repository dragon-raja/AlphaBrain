"""CPU-only contracts for outcome-driven observation learning (not a release).

Only acquired features enter history. Predicted features and outcome labels have
different types. Provenance is a caller declaration, not proof of sensor access;
real adapters must audit encoding, timestamps, coordinate frames and permissions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math


SCHEMA = "dsol-active-observation/v1"


def _name(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("identity must be a nonempty string")


def _count(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("count must be a nonnegative integer")


def _vector(value: tuple[float, ...]) -> None:
    if not isinstance(value, tuple) or not value or any(
        type(x) not in (int, float) or not math.isfinite(x) for x in value
    ):
        raise ValueError("expected a nonempty immutable finite numeric vector")


@dataclass(frozen=True)
class ModelVersions:
    operation: str
    encoder: str
    predictor: str
    evaluator: str
    actor: str

    def __post_init__(self):
        for value in vars(self).values():
            _name(value)


@dataclass(frozen=True)
class AcquiredObservation:
    episode_id: str
    observation_id: str
    tick: int
    encoder_id: str
    features: tuple[float, ...]
    proprioception: tuple[float, ...]
    camera_configuration: tuple[float, ...]

    def __post_init__(self):
        for value in (self.episode_id, self.observation_id, self.encoder_id):
            _name(value)
        _count(self.tick)
        for value in (self.features, self.proprioception, self.camera_configuration):
            _vector(value)


@dataclass(frozen=True)
class ExecutedOperation:
    """Only already executed commands, not future realized robot trajectories."""
    start_tick: int
    end_tick: int
    commands: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        _count(self.start_tick)
        _count(self.end_tick)
        if self.end_tick <= self.start_tick or not isinstance(self.commands, tuple) or not self.commands:
            raise ValueError("invalid executed operation segment")
        for command in self.commands:
            _vector(command)
        if len({len(command) for command in self.commands}) != 1:
            raise ValueError("mixed action dimensions")


@dataclass(frozen=True)
class DecisionContext:
    task: str
    split: str
    versions: ModelVersions
    continuation_policy_id: str
    embodiment_contract: str
    history: tuple[AcquiredObservation, ...]
    executed_operations: tuple[ExecutedOperation, ...] = ()

    def __post_init__(self):
        for value in (self.task, self.continuation_policy_id, self.embodiment_contract):
            _name(value)
        if self.split not in ("train", "development", "test"):
            raise ValueError("unknown split")
        if not isinstance(self.versions, ModelVersions):
            raise ValueError("missing model versions")
        if not isinstance(self.history, tuple) or not self.history or any(
            type(item) is not AcquiredObservation for item in self.history
        ):
            raise ValueError("history accepts acquired observations only")
        first = self.history[0]
        if any(item.episode_id != first.episode_id or
               item.encoder_id != self.versions.encoder or
               len(item.features) != len(first.features) for item in self.history):
            raise ValueError("mixed episode, encoder or feature dimension")
        if len({item.observation_id for item in self.history}) != len(self.history):
            raise ValueError("duplicate observation")
        if any(a.tick >= b.tick for a, b in zip(self.history, self.history[1:])):
            raise ValueError("history must advance physical time")
        if not isinstance(self.executed_operations, tuple) or any(
            type(segment) is not ExecutedOperation or segment.end_tick > self.current.tick
            for segment in self.executed_operations
        ):
            raise ValueError("only past executed operations may enter history")
        if any(a.end_tick > b.start_tick for a, b in zip(self.executed_operations, self.executed_operations[1:])):
            raise ValueError("overlapping executed operations")

    @property
    def current(self) -> AcquiredObservation:
        return self.history[-1]

    def observe(self, observation: AcquiredObservation) -> DecisionContext:
        return replace(self, history=self.history + (observation,))


@dataclass(frozen=True)
class ObservationOption:
    option_id: str
    kind: str
    embodiment_contract: str
    target: tuple[float, ...]
    duration_ticks: int

    def __post_init__(self):
        _name(self.option_id)
        _name(self.embodiment_contract)
        _count(self.duration_ticks)
        if self.kind not in ("continue", "move_and_hold", "matched_hold"):
            raise ValueError("unsupported option; concurrent operation needs a new contract")
        if not isinstance(self.target, tuple):
            raise ValueError("target must be immutable")
        if self.kind == "move_and_hold":
            _vector(self.target)
        elif self.target:
            raise ValueError("continue/hold cannot command a camera target")
        if (self.kind == "continue") != (self.duration_ticks == 0):
            raise ValueError("continue has zero acquisition time; movement/hold must take time")


@dataclass(frozen=True)
class PredictedObservation:
    encoder_id: str
    predictor_id: str
    features: tuple[float, ...]

    def __post_init__(self):
        _name(self.encoder_id)
        _name(self.predictor_id)
        _vector(self.features)


@dataclass(frozen=True)
class AcquisitionReceipt:
    option_id: str
    start_tick: int
    end_tick: int
    status: str
    observation: AcquiredObservation

    def __post_init__(self):
        _name(self.option_id)
        _count(self.start_tick)
        _count(self.end_tick)
        if self.end_tick <= self.start_tick or self.status not in ("arrived", "failed"):
            raise ValueError("receipt requires elapsed time and explicit arrival/failure")
        if type(self.observation) is not AcquiredObservation or self.observation.tick != self.end_tick:
            raise ValueError("receipt must contain the actual endpoint observation")


@dataclass(frozen=True)
class OutcomeExample:
    """Training-only branch outcome. Never an input to the inference planner.

    A return belongs to its frozen continuation policy, not an optimal dynamic
    Oracle. Separate examples are required for each independent execution repeat.
    """
    context: DecisionContext
    option: ObservationOption
    noise_id: str
    success: bool
    remaining_ticks: int
    return_value: float

    def __post_init__(self):
        _name(self.noise_id)
        _count(self.remaining_ticks)
        if type(self.success) is not bool or type(self.return_value) not in (int, float) or not math.isfinite(self.return_value):
            raise ValueError("invalid outcome")
        if self.option.embodiment_contract != self.context.embodiment_contract:
            raise ValueError("outcome embodiment mismatch")


def training_batch(examples: tuple[OutcomeExample, ...], *, operation_id: str,
                   encoder_id: str, continuation_policy_id: str) -> tuple[OutcomeExample, ...]:
    """Conservative first version: reject evaluation data and stale value labels."""
    if not examples:
        raise ValueError("empty training batch")
    seen = set()
    for item in examples:
        context = item.context
        if context.split != "train":
            raise ValueError("development/test outcomes cannot enter training")
        if (context.versions.operation, context.versions.encoder, context.continuation_policy_id) != (
            operation_id, encoder_id, continuation_policy_id
        ):
            raise ValueError("mixed operation, encoder or continuation policy")
        key = (context.current.episode_id, context.current.observation_id,
               item.option.option_id, item.noise_id)
        if key in seen:
            raise ValueError("duplicate branch/noise outcome")
        seen.add(key)
    return tuple(examples)
