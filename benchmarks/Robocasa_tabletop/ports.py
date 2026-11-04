"""Structural ports for GR1 composition; no model/simulator construction.

Implementations may use different processes or libraries, but must preserve the
observable contract. Runtime validators remain authoritative; Protocol does not
replace provenance, numeric or real-device admission.
"""
from typing import Any, Mapping, Protocol
import numpy as np
from AlphaBrain.research.dsol.acquisition.local_view_search import PoseReceipt


class ModelPort(Protocol):
    contract: Any

    def infer(self, example: Mapping, noise: np.ndarray, *, request: Mapping) -> Mapping:
        """Exactly one explicit-noise model call under a live grant."""
        ...


class HeadPort(Protocol):
    def pose_receipt(self) -> PoseReceipt: ...
    def move_head_degrees(self, angles: np.ndarray) -> PoseReceipt: ...
    def reset_observation_cache(self, epoch: int) -> None: ...
    def capture_input(self, epoch: int) -> tuple[Mapping, int]: ...


class OperationPort(Protocol):
    def accept_selected_view(self, pose: PoseReceipt, *, instruction: str, epoch: int) -> None: ...
    def begin_selected_view(self, epoch: int) -> None: ...
    def capture_input(self, epoch: int) -> tuple[Mapping, int]: ...
    def step_absolute(self, action: np.ndarray) -> Mapping[str, bool]: ...
