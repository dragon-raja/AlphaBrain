"""Bound-plan fixture for the actual GR1 CPU authority entry; no real physics."""

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np

from AlphaBrain.research.dsol.acquisition.local_view_search import LocalViewConfig, PoseReceipt
from AlphaBrain.research.dsol.data.search_core import core_spec
from AlphaBrain.research.dsol.execution.manifest import file_digest
from .model_process import CPUModelProcess
from .model_runtime import ScoreContract
from .policy_binding import PROVENANCE, make_binding, build_policy
from .prepared_execution import run_prepared_block
from .search_runtime import ArticulatedHeadMount, GR1SearchSource
from .trial_plan import NoiseFile, TrialPlan


@contextmanager
def synthetic_resources(plan, prepared, directory, guard):
    """Actual model IPC; fake image/state/action ports clearly labeled synthetic."""

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
                "prompt": plan.instruction,
            }, self.epoch

    backend = Backend()

    class Operation:
        steps = 0

        def accept_selected_view(self, pose, *, instruction, epoch):
            if pose.physical_state_id != "CPU_SYNTHETIC_SCENE" or instruction != plan.instruction:
                raise ValueError("Synthetic scene/instruction changed")
            self.epoch = epoch

        def begin_selected_view(self, epoch):
            if epoch != self.epoch:
                raise ValueError("Synthetic epoch changed")

        def capture_input(self, epoch):
            return backend.capture_input(epoch)

        def step_absolute(self, action):
            self.steps += 1
            return dict(success=self.steps == 17, terminated=False)

    with CPUModelProcess(
        sys.executable,
        Path(__file__).resolve().parents[2],
        guard=lambda request: guard(),
        log_path=directory / "synthetic-model.log",
    ) as runtime:
        runtime.contract = ScoreContract(**prepared["policy_binding"]["score"])
        mount = ArticulatedHeadMount(
            prepared["config"], fk, joint_limits_deg=[[-40, 40], [-25, 25]]
        )
        mount.pose(backend.angles)
        p = prepared["policy_binding"]["provenance"]
        yield GR1SearchSource(
            mount,
            backend,
            runtime,
            noise_bank=prepared["score_noise"],
            episode_id=plan.episode_id,
            model_sha256=p["model_sha256"],
            producer_sha256=p["producer_sha256"],
            history_sha256=p["history_sha256"],
            guard=guard,
        ), Operation()


synthetic_resources.binding = lambda: {"mode": "CPU_SYNTHETIC_ONLY", "fixture": "GR1_PREPARED_V1"}
synthetic_resources.validate = lambda plans, prepared: None


def fixture(output, *, count=2):
    """Explicit test-only assets and policy, never a scientific initialization."""
    import torch

    torch.set_num_threads(1)
    output = Path(output)
    score = ScoreContract(10, 10, 16, 16)
    binding = make_binding(core_spec("U12", width=16), {k: "a" * 64 for k in PROVENANCE}, score)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        policy = build_policy(binding)
    with torch.no_grad():
        policy.stop_head.bias.fill_(100.0)
    files = []
    for name in ("score", "operation"):
        path = output / (name + ".npy")
        with path.open("xb") as stream:
            np.save(stream, np.zeros((2, 16, 32), np.float32), allow_pickle=False)
        files.append(NoiseFile(str(path.resolve()), file_digest(path)))
    plans = [
        TrialPlan(
            f"trial-{i}",
            "SYNTHETIC_GR1_CPU",
            asdict(LocalViewConfig()),
            *files,
            8,
            30.0,
            17,
            30.0,
            1.0,
            7,
            None,
        )
        for i in range(count)
    ]
    return plans, policy, binding


def run(guard, *, output):
    plans, policy, binding = fixture(output)
    block = run_prepared_block(
        plans,
        Path(output) / "block",
        policy=policy,
        policy_binding=binding,
        calls_per_episode=[18, 18],
        call_limit=36,
        resources=synthetic_resources,
        guard=guard,
    )
    return dict(
        status="CPU_PREPARED_BLOCK_COMPLETE",
        block=block,
        synthetic_calls=block["completed_model_calls"],
        completed_episodes=block["completed_episodes"],
        real_model_calls=0,
        gpu_used=False,
    )
