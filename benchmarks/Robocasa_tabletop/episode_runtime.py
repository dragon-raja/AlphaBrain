"""Bounded GR1 search entry; uses shared policy/collector and injected authority.

No simulator/model construction, dispatch, retries or scientific defaults. The
durable sink and live guard belong to the existing owning controller. Real
operation-model infer remains independently gated by its registered authority.
"""

import math
from .policy_binding import bind_collector, validate_policy


def run_search_episode(
    source,
    policy,
    binding,
    *,
    instruction,
    session_options,
    max_decisions,
    max_seconds,
    guard,
    clock,
    persist,
):
    if (
        type(max_decisions) is not int
        or max_decisions < 1
        or not math.isfinite(max_seconds)
        or max_seconds <= 0
    ):
        raise ValueError("Explicit search action and elapsed-time bounds required")
    from AlphaBrain.research.dsol.selectors.continuous_ppo import batch_observations
    import torch

    # Validate semantic binding, including actor identity, before acquisition.
    validate_policy(policy, binding)
    receipt = dict(
        episode_id=source.episode_id,
        status="INTENT",
        decisions=0,
        queries=0,
        completed_model_calls=0,
        outcome=None,
    )
    persist(dict(receipt))
    start = clock()

    def live():
        guard()
        elapsed = clock() - start
        if not math.isfinite(elapsed) or not 0 <= elapsed <= max_seconds:
            raise TimeoutError("Search elapsed-time bound reached")

    def sync():
        receipt.update(
            queries=source.queries, completed_model_calls=sum(a["members"] for a in source.attempts)
        )

    try:
        live()
        env, options = bind_collector(source, binding, instruction=instruction, **session_options)
        receipt["status"] = "QUERY_UNKNOWN"
        persist(dict(receipt))
        obs, _ = env.reset(options=options)
        for _ in range(max_decisions):
            live()
            sync()
            with torch.inference_mode():
                action = policy.distribution(batch_observations([obs])).sample(deterministic=True)
            command = dict(
                stop=bool(action["stop"][0]), delta_deg=action["delta_deg"][0].cpu().numpy()
            )
            receipt["status"] = "DECISION_UNKNOWN"
            persist(dict(receipt))
            obs, _, done, truncated, info = env.step(command)
            receipt["decisions"] += 1
            sync()
            if truncated:
                raise RuntimeError("Truncation requires audit, not a completed search")
            if done:
                live()
                receipt.update(status="COMPLETE", outcome=info)
                persist(dict(receipt))
                return receipt
            receipt["status"] = "RUNNING"
            persist(dict(receipt))
        raise TimeoutError("Search decision bound reached; no implicit STOP or return")
    except BaseException:
        sync()
        source.status = "HOLD"
        receipt["outcome"] = "UNKNOWN_REQUIRES_AUDIT"
        persist(dict(receipt))
        raise
