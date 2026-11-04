"""Bounded selected-view operation entry with injected, authorized GR1 ports.

No worker launches, retries, budget refunds or search-policy updates. Durable
receipts must be supplied by the existing task/result store. Real operation is
blocked unless both the caller guard and runtime's per-call authority permit it.
"""

import math
import numpy as np
from .ports import ModelPort, OperationPort


def validate_operation(runtime, *, episode_id, epoch, noises, max_actions, max_seconds):
    if (
        type(max_actions) is not int
        or max_actions < 1
        or type(epoch) is not int
        or epoch < 0
        or not isinstance(episode_id, str)
        or not episode_id
        or not math.isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("Explicit operation identity, action/time caps required")
    noises = np.asarray(noises)
    expected_chunks = math.ceil(max_actions / runtime.contract.execute_actions)
    if (
        noises.shape != (expected_chunks, 16, 32)
        or noises.dtype != np.float32
        or not np.isfinite(noises).all()
    ):
        raise ValueError("Frozen operation noise for every possible replan required")
    return noises


def run_selected_operation(
    runtime: ModelPort, backend: OperationPort, *, episode_id, epoch, noises, max_actions, max_seconds, guard, clock, persist
):
    noises = validate_operation(
        runtime,
        episode_id=episode_id,
        epoch=epoch,
        noises=noises,
        max_actions=max_actions,
        max_seconds=max_seconds,
    )
    receipt = dict(
        episode_id=episode_id,
        epoch=epoch,
        status="INTENT",
        actions=0,
        completed_model_calls=0,
        attempted_model_calls=0,
        success=None,
    )
    persist(dict(receipt))
    start = clock()
    if not math.isfinite(start):
        raise ValueError("Finite monotonic clock required")

    def live():
        guard()
        elapsed = clock() - start
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed > max_seconds:
            raise TimeoutError("Operation elapsed-time bound reached")

    try:
        live()
        backend.begin_selected_view(epoch)
        for chunk, noise in enumerate(noises):
            live()
            example, observed_epoch = backend.capture_input(epoch)
            if observed_epoch != epoch:
                raise ValueError("Stale selected-view observation")
            receipt["attempted_model_calls"] += 1
            receipt["status"] = "MODEL_CALL_UNKNOWN"
            persist(dict(receipt))
            output = runtime.infer(
                example,
                noise,
                request=dict(query_id=f"{episode_id}/operation-{chunk}", epoch=epoch, member=0),
            )
            receipt["completed_model_calls"] += 1
            actions = np.asarray(output["actions"])
            if (
                actions.shape != (runtime.contract.execute_actions, 29)
                or not np.isfinite(actions).all()
            ):
                raise ValueError("Native action chunk differs from frozen execution prefix")
            for action in actions:
                if receipt["actions"] == max_actions:
                    break
                live()
                receipt["status"] = "ACTION_UNKNOWN"
                persist(dict(receipt))
                outcome = backend.step_absolute(action)
                receipt["actions"] += 1
                if set(outcome) != {"success", "terminated"} or any(
                    type(x) is not bool for x in outcome.values()
                ):
                    raise ValueError("Explicit benchmark termination/success required")
                receipt["status"] = "RUNNING"
                persist(dict(receipt))
                if outcome["success"] or outcome["terminated"]:
                    receipt.update(status="COMPLETE", success=outcome["success"])
                    live()
                    persist(dict(receipt))
                    return receipt
        receipt.update(status="COMPLETE", success=False)
        live()
        persist(dict(receipt))
        return receipt
    except BaseException:
        # Keep the exact last durable phase for recovery: a timed-out call/action
        # may have completed remotely. Never infer unused budget from this error.
        receipt["outcome"] = "UNKNOWN_REQUIRES_AUDIT"
        persist(dict(receipt))
        raise
