"""Compose bound GR1 trials under the existing durable block accounting.

Resource factories own model/task construction and independent admission. This
entry adds no scheduler, real-model allowance, retry or scientific defaults.
The same entry accepts CPU synthetic ports for end-to-end/fault testing.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from AlphaBrain.research.dsol.execution.manifest import digest
from .block_runtime import EpisodeSpec, run_block
from .policy_binding import validate_policy, policy_state_identity
from .trial_plan import TrialPlan
from .trial_runtime import run_search_and_operation


def run_prepared_block(
    plans,
    output,
    *,
    policy,
    policy_binding,
    calls_per_episode,
    call_limit,
    resources,
    guard,
    clock=time.monotonic,
):
    """Validate ALL trials before opening resources, then run each exactly once.

    resources(plan, prepared, directory, guard) is a context manager yielding
    (source, operation_backend). It must use prepared noise and model identity.
    Actual call intent is charged here, independently of runtime callbacks.
    """
    binding = validate_policy(policy, policy_binding)
    weights = policy_state_identity(policy)
    plans = tuple(plans)
    if not plans or len(plans) != len(calls_per_episode):
        raise ValueError("One explicit reservation for every trial required")
    prepared = [p.prepare(binding, cap) for p, cap in zip(plans, calls_per_episode)]
    # Physical task/controller/runtime identity is part of the block, not just
    # feature shapes and noise. Factories expose the same data they instantiate.
    resource_binding = resources.binding()
    resource_stamp = digest(resource_binding)
    resources.validate(plans, prepared)
    specs = [
        EpisodeSpec(p.episode_id, cap, p.request()) for p, cap in zip(plans, calls_per_episode)
    ]
    # run_block checks duplicate IDs and total reservation before execute.
    by_id = {p.episode_id: (p, data) for p, data in zip(plans, prepared)}

    def execute(spec, directory, grant):
        plan, data = by_id[spec.episode_id]
        guard()
        if digest(resources.binding()) != resource_stamp:
            raise ValueError("Resource binding changed after preflight")
        if policy_state_identity(policy) != weights:
            raise ValueError("Policy weights changed after preflight")
        with resources(plan, data, directory, guard) as (source, operation):
            if (
                source.episode_id != plan.episode_id
                or asdict(source.mount.config) != plan.search_config
                or asdict(source.runtime.contract) != binding["score"]
            ):
                raise ValueError("Constructed resources differ from the bound trial")
            import numpy as np

            if not np.array_equal(source.noise, data["score_noise"]):
                raise ValueError("Resource changed paired scoring noise")
            # No reuse/retry after an unknown model call, including the same request.
            runtime = source.runtime

            class ChargedRuntime:
                contract = runtime.contract
                seen = set()

                def infer(self, example, noise, *, request):
                    key = digest(request)
                    if key in self.seen:
                        raise RuntimeError("Repeated model call requires audit, not replay")
                    self.seen.add(key)
                    grant()
                    return runtime.infer(example, noise, request=request)

            source.runtime = ChargedRuntime()
            receipt = source.backend.pose_receipt()
            # A measured head can arrive slightly away from the commanded position.
            # Never substitute a guessed zero or nominal/reference view here.
            angles = (
                receipt.joint_angles_deg
                if hasattr(receipt, "joint_angles_deg")
                else source.mount.target_for_pose(receipt.camera_to_world)
            )
            options = dict(
                initial_angles=angles,
                physical_state_id=receipt.physical_state_id,
                score_scale=plan.score_scale,
                seed=plan.seed,
                settling_tolerance_deg=plan.settling_tolerance_deg,
            )
            with (directory / "trial-events.jsonl").open("x") as events:

                def persist(row):
                    events.write(
                        json.dumps(row, default=lambda x: x.tolist(), allow_nan=False) + "\n"
                    )
                    events.flush()
                    os.fsync(events.fileno())

                return run_search_and_operation(
                    source,
                    policy,
                    binding,
                    operation,
                    instruction=plan.instruction,
                    session_options=options,
                    search_decisions=plan.search_decisions,
                    search_seconds=plan.search_seconds,
                    operation_noises=data["operation_noise"],
                    operation_actions=plan.operation_actions,
                    operation_seconds=plan.operation_seconds,
                    guard=guard,
                    clock=clock,
                    persist=persist,
                )

    manifest = dict(
        policy=binding,
        policy_state_sha256=weights,
        resources=resource_binding,
        trials=[p.request() for p in plans],
        caps=list(calls_per_episode),
        total=call_limit,
    )
    return run_block(
        specs,
        Path(output),
        binding_sha256=digest(manifest),
        binding_manifest=manifest,
        call_limit=call_limit,
        guard=guard,
        execute=execute,
    )
