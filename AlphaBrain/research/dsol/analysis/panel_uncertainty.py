"""Paired uncertainty for shared multi-task policies, not independent runs/task.

Inputs are seed x task x initial means: keep paired starts/execution repeats
inside each initial before calling. All methods must use exactly the same
panel. Small-seed bootstrap intervals are descriptive sensitivity estimates,
not selection-adjusted significance tests or guaranteed coverage statements.
"""

import numpy as np


def paired_panel_intervals(left, right, *, draws=20000, random_seed=20261104):
    """Return fixed-policy and crossed seed/task/initial percentile intervals.

    Resample policy seeds globally, not independently for each task: one
    trained multi-task policy produces all tasks' outcomes. Resample task and
    initial identities jointly for both methods and all selected seeds.
    Tasks and initials have equal weight; ragged/missing panels are rejected.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.ndim != 3 or left.shape != right.shape or min(left.shape) < 2:
        raise ValueError("Need matching seed x task x initial panels, each axis >= 2")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Panels must be finite and complete")
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 100:
        raise ValueError("draws must be an integer >= 100")
    delta = left - right
    seeds, tasks, initials = delta.shape
    rng = np.random.default_rng(random_seed)
    fixed, crossed = np.empty(draws), np.empty(draws)
    # Bounded memory: panels can grow without allocating draws x full panel.
    for start in range(0, draws, 256):
        count = min(256, draws - start)
        task_index = rng.integers(tasks, size=(count, tasks))
        initial_index = rng.integers(initials, size=(count, tasks, initials))
        seed_index = rng.integers(seeds, size=(count, seeds))
        sampled = delta[:, task_index[:, :, None], initial_index]
        fixed[start:start+count] = sampled.mean(axis=(0, 2, 3))
        by_draw = sampled.transpose(1, 0, 2, 3)
        crossed[start:start+count] = by_draw[
            np.arange(count)[:, None], seed_index
        ].mean(axis=(1, 2, 3))
    return {
        "mean_difference": float(delta.mean()),
        "per_seed_difference": delta.mean(axis=(1, 2)).tolist(),
        "seed_sample_sd": float(delta.mean(axis=(1, 2)).std(ddof=1)),
        "per_task_difference": delta.mean(axis=(0, 2)).tolist(),
        "fixed_observed_policies_task_initial_ci95": np.quantile(fixed, [.025, .975]).tolist(),
        "crossed_seed_task_initial_ci95_sensitivity": np.quantile(crossed, [.025, .975]).tolist(),
        "seeds": seeds, "tasks": tasks, "initials_per_task": initials,
        "draws": draws, "bootstrap_seed": random_seed,
        "selection_adjusted": False,
        "scope": "Descriptive development panel; few-seed coverage not guaranteed",
    }
