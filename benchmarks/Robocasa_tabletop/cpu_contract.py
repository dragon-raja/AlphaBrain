"""Explicit synthetic native-query fixture for remote GR1 CPU admission only.

No real environment, operation model, renderer, training or GPU is constructed.
Every synthetic model member still requests a live center continuation grant.
"""

import numpy as np
from AlphaBrain.research.dsol.acquisition.local_view_search import LocalViewConfig, PoseReceipt
from .search_runtime import ArticulatedHeadMount, GR1SearchSource


def run_query(guard, *, episode_id):
    class Backend:
        def pose_receipt(self):
            return PoseReceipt(np.eye(4), "synthetic-fixed-scene")

        def capture_input(self, epoch):
            return {
                "observation/image": np.zeros((2, 2, 3), np.uint8),
                "observation/state": np.zeros(29, np.float32),
                "prompt": "SYNTHETIC_GR1_CPU",
            }, epoch

    class Model:
        calls = 0

        def infer(self, example, noise, *, request):
            guard()
            self.calls += 1
            return dict(feature=np.ones(4608, np.float32), score=1.0)

    runtime = Model()
    source = GR1SearchSource(
        ArticulatedHeadMount(
            LocalViewConfig(), lambda angles: np.eye(4), joint_limits_deg=[[-40, 40], [-25, 25]]
        ),
        Backend(),
        runtime,
        noise_bank=np.zeros((2, 16, 32), np.float32),
        episode_id=episode_id,
        model_sha256="a" * 64,
        producer_sha256="b" * 64,
        history_sha256="c" * 64,
        guard=guard,
    )
    result = source.query()
    return dict(
        status="CPU_CONTRACT_COMPLETE",
        episode_id=episode_id,
        queries=source.queries,
        synthetic_calls=runtime.calls,
        real_model_calls=0,
        gpu_used=False,
        feature_shape=list(result.value.features.values[0].shape),
        packet=result.value.features.metadata(),
        attempts=source.attempts,
    )


def run_trial(guard, *, episode_id, output, model_grant=None):
    """Actual shared actor/search/operation entry, all inputs explicitly fake."""
    import json
    import os
    from pathlib import Path
    import sys
    import time
    import torch
    from AlphaBrain.research.dsol.data.search_core import core_spec
    from .model_process import CPUModelProcess
    from .model_runtime import ScoreContract
    from .policy_binding import make_binding, build_policy
    from .trial_runtime import run_search_and_operation

    output = Path(output)
    torch.set_num_threads(1)

    def fk(angles):
        pose = np.eye(4)
        pose[:2, 3] = np.asarray(angles) / 100
        return pose

    class Backend:
        angles = np.zeros(2)
        epoch = 0

        def pose_receipt(self):
            return PoseReceipt(fk(self.angles), "CPU_SYNTHETIC_SCENE")

        def move_head_degrees(self, angles):
            self.angles = angles.copy()
            return self.pose_receipt()

        def reset_observation_cache(self, epoch):
            self.epoch = epoch

        def capture_input(self, epoch):
            return {
                "observation/image": np.zeros((2, 2, 3), np.uint8),
                "observation/state": np.zeros(29, np.float32),
                "prompt": "SYNTHETIC_GR1_CPU",
            }, self.epoch

    backend = Backend()

    class Operation:
        steps = 0

        def accept_selected_view(self, pose, *, instruction, epoch):
            if pose.physical_state_id != "CPU_SYNTHETIC_SCENE":
                raise ValueError("fixture scene changed")
            self.epoch = epoch

        def begin_selected_view(self, epoch):
            if epoch != self.epoch:
                raise ValueError("fixture epoch")

        def capture_input(self, epoch):
            return backend.capture_input(epoch)

        def step_absolute(self, action):
            self.steps += 1
            return dict(success=self.steps == 17, terminated=False)

    provenance = {
        k: "a" * 64
        for k in (
            "model_sha256",
            "producer_sha256",
            "history_sha256",
            "processor_sha256",
            "normalization_sha256",
            "source_sha256",
        )
    }
    score = ScoreContract(10, 10, 16, 16)
    binding = make_binding(core_spec("U12", width=16), provenance, score)
    torch.manual_seed(7)
    policy = build_policy(binding)
    with torch.no_grad():
        policy.stop_head.bias.fill_(100.0)
    calls = []

    def grant(request):
        guard()
        if model_grant is not None:
            model_grant()
        calls.append(dict(request))

    with (output / "trial-events.jsonl").open("x") as events:

        def persist(row):
            events.write(json.dumps(row, default=lambda v: v.tolist()) + "\n")
            events.flush()
            os.fsync(events.fileno())

        with CPUModelProcess(
            sys.executable,
            Path(__file__).resolve().parents[2],
            guard=grant,
            log_path=output / "synthetic-model.log",
        ) as runtime:
            runtime.contract = (
                score  # Explicit CPU fixture identity, not native numerical admission.
            )
            source = GR1SearchSource(
                ArticulatedHeadMount(
                    LocalViewConfig(), fk, joint_limits_deg=[[-40, 40], [-25, 25]]
                ),
                backend,
                runtime,
                noise_bank=np.zeros((2, 16, 32), np.float32),
                episode_id=episode_id,
                model_sha256="a" * 64,
                producer_sha256="a" * 64,
                history_sha256="a" * 64,
                guard=guard,
            )
            result = run_search_and_operation(
                source,
                policy,
                binding,
                Operation(),
                instruction="SYNTHETIC_GR1_CPU",
                session_options=dict(
                    initial_angles=[0, 0],
                    physical_state_id="CPU_SYNTHETIC_SCENE",
                    score_scale=1.0,
                    seed=7,
                ),
                search_decisions=8,
                search_seconds=30.0,
                operation_noises=np.zeros((2, 16, 32), np.float32),
                operation_actions=17,
                operation_seconds=30.0,
                guard=guard,
                clock=time.monotonic,
                persist=persist,
            )
            child = runtime.process
    if len(calls) != 4 or child.poll() is None:
        raise RuntimeError("CPU trial did not close exact owned child/budget")
    return dict(
        status="CPU_TRIAL_COMPLETE",
        synthetic_calls=4,
        real_model_calls=0,
        gpu_used=False,
        child_exited=True,
        trial=result,
    )


def run_trial_block(guard, *, episode_id, output):
    """Two complete synthetic trials through the real block implementation."""
    from .block_runtime import EpisodeSpec, run_block
    from AlphaBrain.research.dsol.execution.manifest import digest

    episodes = [EpisodeSpec(f"trial-{i}", 4, {"fixture": "synthetic_trial"}) for i in range(2)]

    def execute(spec, directory, grant):
        result = run_trial(guard, episode_id=spec.episode_id, output=directory, model_grant=grant)
        return dict(
            status="COMPLETE", completed_model_calls=result["synthetic_calls"], trial=result
        )

    result = run_block(
        episodes,
        output,
        binding_sha256=digest({"fixture": episode_id}),
        call_limit=8,
        guard=guard,
        execute=execute,
    )
    return dict(
        status="CPU_BLOCK_COMPLETE",
        synthetic_calls=result["completed_model_calls"],
        real_model_calls=0,
        gpu_used=False,
        completed_episodes=2,
        block=result,
    )
