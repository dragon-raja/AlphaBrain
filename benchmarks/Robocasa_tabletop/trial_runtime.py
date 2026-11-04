"""Single-owner GR1 search-to-operation composition, with no implicit replay.

The controller supplies frozen policy/noise/provenance and owns the durable
sink. Search and operation use the same runtime and selected task. Real model
execution remains independently gated; synthetic ports exercise this entry.
"""

from .episode_runtime import run_search_episode
from .operation_rollout import run_selected_operation, validate_operation


def run_search_and_operation(
    source,
    policy,
    binding,
    operation_backend,
    *,
    instruction,
    session_options,
    search_decisions,
    search_seconds,
    operation_noises,
    operation_actions,
    operation_seconds,
    guard,
    clock,
    persist,
):
    if source.status != "READY" or source.queries:
        raise ValueError("Fresh owned source required; unknown or completed trial cannot replay")
    validate_operation(
        source.runtime,
        episode_id=source.episode_id,
        epoch=source.epoch,
        noises=operation_noises,
        max_actions=operation_actions,
        max_seconds=operation_seconds,
    )
    last = {"stage": "VALIDATING", "status": "INTENT", "episode_id": source.episode_id}
    persist(dict(last))

    def sink(stage):
        def write(row):
            last.update(stage=stage, status=row["status"], detail=row)
            persist(dict(last))

        return write

    try:
        guard()
        search = run_search_episode(
            source,
            policy,
            binding,
            instruction=instruction,
            session_options=session_options,
            max_decisions=search_decisions,
            max_seconds=search_seconds,
            guard=guard,
            clock=clock,
            persist=sink("SEARCH"),
        )
        guard()
        selected = source.backend.pose_receipt()
        # No reset-to-zero or state reconstruction is hidden in this handoff.
        # An admitted operation adapter must check this exact scene/head pose.
        last.update(stage="HANDOFF", status="UNKNOWN")
        persist(dict(last))
        operation_backend.accept_selected_view(
            selected, instruction=instruction, epoch=source.epoch
        )
        operation = run_selected_operation(
            source.runtime,
            operation_backend,
            episode_id=source.episode_id,
            epoch=source.epoch,
            noises=operation_noises,
            max_actions=operation_actions,
            max_seconds=operation_seconds,
            guard=guard,
            clock=clock,
            persist=sink("OPERATION"),
        )
        guard()
        result = dict(
            episode_id=source.episode_id,
            status="COMPLETE",
            search=search,
            operation=operation,
            completed_model_calls=search["completed_model_calls"]
            + operation["completed_model_calls"],
        )
        last.update(stage="TRIAL", status="COMPLETE", detail=result)
        persist(dict(last))
        source.status = "COMPLETE"
        return result
    except BaseException:
        source.status = "HOLD"
        last["outcome"] = "UNKNOWN_REQUIRES_AUDIT"
        persist(dict(last))
        raise
