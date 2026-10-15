"""Pure state transitions for observation/operation alternation.

No simulator, actuator, policy inference or parameter update is implemented.
Receipts must come from an independently audited adapter. Failed movement is
retained and charged; it is never rewritten as a successful continue action.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..data.active_observation import (
    AcquisitionReceipt, AcquiredObservation, DecisionContext, ExecutedOperation, ObservationOption,
)


@dataclass(frozen=True)
class EpisodeCursor:
    context: DecisionContext
    execution_repeat: int
    deadline_tick: int
    max_acquisitions: int
    replan_index: int = 0
    acquisitions: tuple[AcquisitionReceipt, ...] = ()
    terminated: bool = False

    def __post_init__(self):
        for value in (self.execution_repeat, self.deadline_tick, self.max_acquisitions, self.replan_index):
            if type(value) is not int or value < 0:
                raise ValueError("invalid episode counter")
        if type(self.terminated) is not bool:
            raise ValueError("terminated must be bool")

    @property
    def execution_noise_address(self) -> tuple[str, int, int, str]:
        # Intentionally independent of view, actor, option and scoring calls.
        return (self.context.current.episode_id, self.execution_repeat, self.replan_index, "execution")

    def _ready(self):
        if self.terminated or self.context.current.tick >= self.deadline_tick:
            raise ValueError("episode terminated or time budget exhausted")

    def acquire(self, option: ObservationOption, receipt: AcquisitionReceipt) -> EpisodeCursor:
        self._ready()
        if option.kind == "continue":
            raise ValueError("continue does not acquire; execute the next operation segment")
        if option.embodiment_contract != self.context.embodiment_contract:
            raise ValueError("embodiment mismatch")
        if len(self.acquisitions) >= self.max_acquisitions:
            raise ValueError("acquisition budget exhausted")
        if receipt.option_id != option.option_id or receipt.start_tick != self.context.current.tick:
            raise ValueError("receipt is not for this decision")
        elapsed = receipt.end_tick - receipt.start_tick
        if elapsed > option.duration_ticks or (receipt.status == "arrived" and elapsed != option.duration_ticks):
            raise ValueError("receipt duration differs from bounded option")
        context = self.context.observe(receipt.observation)
        # Arrival after deadline is recorded but cannot lead to another action.
        return replace(self, context=context, acquisitions=self.acquisitions + (receipt,),
                       terminated=receipt.end_tick >= self.deadline_tick)

    def operation(self, endpoint: AcquiredObservation, *, commands: tuple[tuple[float, ...], ...],
                  terminated: bool) -> EpisodeCursor:
        self._ready()
        if type(terminated) is not bool:
            raise ValueError("terminated must be bool")
        segment = ExecutedOperation(self.context.current.tick, endpoint.tick, commands)
        context = replace(self.context, history=self.context.history + (endpoint,),
                          executed_operations=self.context.executed_operations + (segment,))
        return replace(self, context=context, replan_index=self.replan_index + 1,
                       terminated=terminated or endpoint.tick >= self.deadline_tick)
