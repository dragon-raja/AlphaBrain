"""Explicit task/model factories for the prepared GR1 block entry.

CPU mode constructs real task physics with synthetic images and model output.
Native mode stays closed until separate GPU admission/registration; its resource
wiring is expressed here without enabling real model calls or EGL contexts.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import time
import numpy as np

from .head_servo import ServoSettings
from .task_runtime import TaskSpec


@dataclass(frozen=True)
class TaskResources:
    spec: TaskSpec
    controller_config: dict
    allowed_empty_categories: tuple
    servo_settings: ServoSettings
    initial_angles: tuple
    roll_hold: float
    scene_protocol: str
    expected_process_fingerprint: dict

    def validate(self, plan, prepared):
        self.spec.validate()
        self.servo_settings.validate()
        config = prepared["config"]
        if (
            not self.spec.active_head
            or self.spec.horizon < plan.operation_actions
            or self.controller_config.get("type") != "BASIC"
            or self.controller_config.get("control_delta") is not False
            or self.scene_protocol != "PROJECT_NONHEAD_INTEGRATION_V1"
            or self.roll_hold != 0.0
        ):
            raise ValueError(
                "Explicit active-head absolute control and held-scene contract required"
            )
        angles = np.asarray(self.initial_angles)
        if (
            angles.shape != (2,)
            or not np.isfinite(angles).all()
            or np.any(np.abs(angles) > [config.yaw_limit_deg, config.pitch_limit_deg])
        ):
            raise ValueError("Initial head angles outside explicit search domain")
        if plan.settling_tolerance_deg is None:
            raise ValueError("Real task requires a measured head-arrival tolerance")
        if np.rad2deg(self.servo_settings.position_tolerance) > plan.settling_tolerance_deg:
            raise ValueError("Servo can settle outside search acceptance tolerance")
        from .task_runtime import process_fingerprint

        if self.expected_process_fingerprint != process_fingerprint():
            raise ValueError("Task interpreter/hash seed differs before resource construction")


class NativeResources:
    """Owned per-trial contexts; no caches or implicit resume.

    Task settings are indexed by immutable trial ID, not selected by host. A
    synthetic model child uses the same process IPC but cannot provide native
    numeric evidence. Production currently refuses before constructing anything.
    """

    def __init__(
        self,
        tasks,
        *,
        model_python,
        source_root,
        mode,
        model_config=None,
        import_paths=(),
        clock=time.monotonic,
    ):
        if mode not in ("CPU_TASK_SYNTHETIC_MODEL", "NATIVE"):
            raise ValueError("Explicit resource mode required")
        if not Path(model_python).is_absolute() or not Path(source_root).is_absolute():
            raise ValueError("Explicit owned interpreter and source root required")
        self.tasks, self.model_python, self.source_root = tasks, model_python, source_root
        self.mode, self.model_config, self.import_paths = mode, model_config, import_paths
        self.clock = clock

    def binding(self):
        from AlphaBrain.research.dsol.execution.manifest import file_digest

        return dict(
            mode=self.mode,
            tasks={k: asdict(v) for k, v in self.tasks.items()},
            model_python=str(Path(self.model_python).resolve()),
            interpreter_sha256=file_digest(Path(self.model_python).resolve()),
            source_root=str(Path(self.source_root).resolve()),
            model_config=self.model_config,
            import_paths=list(self.import_paths),
        )

    def validate(self, plans, prepared):
        if self.mode == "NATIVE":
            from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production

            reject_production("benchmarks/Robocasa_tabletop/native_resources.py:NativeResources")
        if set(self.tasks) != {p.episode_id for p in plans}:
            raise ValueError("Exact task coverage required")
        for plan, data in zip(plans, prepared):
            self.tasks[plan.episode_id].validate(plan, data)

    @contextmanager
    def __call__(self, plan, prepared, directory, guard):
        if self.mode == "NATIVE":
            # This is intentionally not on the production entry allowlist.
            from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production

            reject_production("benchmarks/Robocasa_tabletop/native_resources.py:NativeResources")
        settings = self.tasks[plan.episode_id]
        settings.validate(plan, prepared)
        from .task_composition import TaskPorts
        from .model_process import CPUModelProcess
        from .model_runtime import ScoreContract

        def no_render():
            raise PermissionError("CPU contract cannot render native frames")

        with TaskPorts(
            settings.spec,
            controller_config=settings.controller_config,
            allowed_empty_categories=settings.allowed_empty_categories,
            servo_settings=settings.servo_settings,
            search_config=prepared["config"],
            initial_angles=settings.initial_angles,
            roll_hold=settings.roll_hold,
            scene_protocol=settings.scene_protocol,
            frame_mode="CPU_SYNTHETIC",
            expected_process_fingerprint=settings.expected_process_fingerprint,
            guard=guard,
            render_guard=no_render,
            clock=self.clock,
        ) as ports:
            example, _ = ports.search.capture_input(0)
            if example["prompt"] != plan.instruction:
                raise ValueError("Actual task instruction differs before model construction")
            with CPUModelProcess(
                self.model_python,
                self.source_root,
                guard=lambda request: guard(),
                continuation_guard=guard,
                log_path=directory / "synthetic-model.log",
            ) as runtime:
                runtime.contract = ScoreContract(**prepared["policy_binding"]["score"])
                if runtime.contract.execute_actions != 16:
                    raise ValueError("CPU child fixture emits 16 actions; no silent slicing")
                source = ports.source(
                    runtime,
                    noise_bank=prepared["score_noise"],
                    episode_id=plan.episode_id,
                    provenance=prepared["policy_binding"]["provenance"],
                    guard=guard,
                )
                yield source, ports.operation
