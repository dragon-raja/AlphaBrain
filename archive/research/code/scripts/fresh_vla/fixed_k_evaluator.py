from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


class FixedKEnvironment(Protocol):
    def reset_to(self, initial_state: Any, branch_outcome: str) -> Mapping[str, Any]: ...

    def step(self, action: np.ndarray) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]: ...

    def close(self) -> None: ...


class ChunkPolicy(Protocol):
    def predict_action(self, observation: Mapping[str, Any]) -> np.ndarray: ...


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    initial_state: Any
    branch_outcome: str
    is_deterministic_control: bool = False


def evaluate_fixed_k(
    env_factory: Callable[[], FixedKEnvironment],
    policy: ChunkPolicy,
    episodes: Sequence[EpisodeSpec],
    *,
    execution_horizons: Sequence[int] = (1, 3),
    max_steps: int = 200,
) -> dict[str, Any]:
    results = {}
    for execution_horizon in execution_horizons:
        if execution_horizon <= 0:
            raise ValueError(f"execution horizon must be positive, got {execution_horizon}")
        rows = []
        env = env_factory()
        try:
            for episode in episodes:
                observation = env.reset_to(episode.initial_state, episode.branch_outcome)
                completion_steps = 0
                success = False
                premature_commitment = False
                failure_continuation = False
                recovery_success = False
                done = False
                while completion_steps < max_steps and not done:
                    chunk = np.asarray(policy.predict_action(observation), dtype=np.float32)
                    if chunk.ndim != 2 or chunk.shape[0] < execution_horizon:
                        raise ValueError(
                            f"policy chunk must be [H, D] with H >= {execution_horizon}, got {chunk.shape}"
                        )
                    for action in chunk[:execution_horizon]:
                        observation, _, done, info = env.step(action)
                        completion_steps += 1
                        success = success or bool(info.get("success", False))
                        premature_commitment = premature_commitment or bool(
                            info.get("premature_commitment", False)
                        )
                        failure_continuation = failure_continuation or bool(
                            info.get("failure_continuation", False)
                        )
                        recovery_success = recovery_success or bool(info.get("recovery_success", False))
                        if done or completion_steps >= max_steps:
                            break
                rows.append(
                    {
                        "episode_id": episode.episode_id,
                        "branch_outcome": episode.branch_outcome,
                        "is_deterministic_control": episode.is_deterministic_control,
                        "success": success,
                        "premature_commitment": premature_commitment,
                        "failure_continuation": failure_continuation,
                        "recovery_success": recovery_success,
                        "completion_steps": completion_steps,
                    }
                )
        finally:
            env.close()

        failures = [row for row in rows if row["branch_outcome"] not in {"success", "attached"}]
        deterministic = [row for row in rows if row["is_deterministic_control"]]
        results[str(execution_horizon)] = {
            "execution_horizon": execution_horizon,
            "episode_count": len(rows),
            "success_rate": statistics.mean(row["success"] for row in rows) if rows else 0.0,
            "failure_branch_success_rate": statistics.mean(row["success"] for row in failures) if failures else None,
            "premature_commitment_rate": statistics.mean(row["premature_commitment"] for row in rows) if rows else 0.0,
            "failure_continuation_rate": statistics.mean(row["failure_continuation"] for row in failures)
            if failures
            else None,
            "recovery_success_rate": statistics.mean(row["recovery_success"] for row in failures) if failures else None,
            "deterministic_success_rate": statistics.mean(row["success"] for row in deterministic)
            if deterministic
            else None,
            "mean_completion_steps": statistics.mean(row["completion_steps"] for row in rows) if rows else 0.0,
            "episodes": rows,
        }
    return results
