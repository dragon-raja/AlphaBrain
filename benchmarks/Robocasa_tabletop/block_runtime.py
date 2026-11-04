"""Bounded GR1 episode blocks beneath the existing controller/ledger.

This module owns neither dispatch nor retries. An adapter supplies an episode
executor and a live guard. All intent/result records are durable; any ambiguous
call or partial block stays UNKNOWN for controller-side audit. Plans carry no
implicit task, model, core, or scientific defaults.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import re

from AlphaBrain.research.dsol.execution.manifest import digest, file_digest
from AlphaBrain.research.dsol.execution.queue import _durable_write


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    calls: int
    request: Mapping

    def validate(self):
        if (
            not isinstance(self.episode_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.episode_id)
            or self.episode_id in (".", "..")
            or type(self.calls) is not int
            or self.calls < 1
            or not isinstance(self.request, dict)
        ):
            raise ValueError("Explicit safe episode identity, call cap and request required")
        digest(self.request)


class EpisodeExecutor(Protocol):
    def __call__(self, spec: EpisodeSpec, output: Path, grant: Callable[[], None]) -> Mapping:
        """Execute once; grant must be called immediately BEFORE each model call.

        Return status=COMPLETE and completed_model_calls. Never resume inside
        this method after unknown consumption; the controller reconciles it.
        """
        ...


def run_block(
    episodes,
    output,
    *,
    binding_sha256,
    call_limit,
    guard,
    execute: EpisodeExecutor,
    binding_manifest=None,
):
    episodes = tuple(episodes)
    if (
        not episodes
        or not re.fullmatch("[0-9a-f]{64}", binding_sha256)
        or type(call_limit) is not int
        or call_limit < 1
    ):
        raise ValueError("Frozen binding and bounded nonempty block required")
    if binding_manifest is not None and digest(binding_manifest) != binding_sha256:
        raise ValueError("Block manifest differs from bound identity")
    for episode in episodes:
        episode.validate()
    if len({e.episode_id for e in episodes}) != len(episodes):
        raise ValueError("Duplicate episode in block")
    if sum(e.calls for e in episodes) > call_limit:
        raise ValueError("Episode reservations exceed block budget")
    guard()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    state = dict(
        status="INTENT",
        binding_sha256=binding_sha256,
        attempted_model_calls=0,
        completed_model_calls=0,
        completed_episodes=0,
        episodes={},
        current=None,
    )
    path = output / "block-state.json"
    _durable_write(path, state)
    try:
        if binding_manifest is not None:
            _durable_write(output / "binding.json", binding_manifest)
        for spec in episodes:
            guard()
            directory = output / spec.episode_id
            directory.mkdir(exist_ok=False)
            entry = dict(
                status="INTENT", attempted_model_calls=0, request_sha256=digest(spec.request)
            )
            state["current"] = spec.episode_id
            state["episodes"][spec.episode_id] = entry
            _durable_write(path, state)

            def grant():
                guard()
                if entry["attempted_model_calls"] >= spec.calls:
                    raise RuntimeError("Episode model-call cap reached")
                entry["attempted_model_calls"] += 1
                state["attempted_model_calls"] += 1
                entry["status"] = "CALL_UNKNOWN"
                state["status"] = "UNKNOWN"
                _durable_write(path, state)

            result = dict(execute(spec, directory, grant))
            calls = result.get("completed_model_calls")
            if (
                result.get("status") != "COMPLETE"
                or type(calls) is not int
                or calls != entry["attempted_model_calls"]
            ):
                raise ValueError("Episode completion differs from charged calls")
            guard()
            _durable_write(directory / "result.json", result)
            entry.update(status="COMPLETE", result_sha256=file_digest(directory / "result.json"))
            state["completed_model_calls"] += calls
            state["completed_episodes"] += 1
            _durable_write(path, state)
        guard()
        state.update(status="COMPLETE", current=None)
        _durable_write(path, state)
        return state
    except BaseException:
        state["status"] = "UNKNOWN_REQUIRES_AUDIT"
        _durable_write(path, state)
        raise
