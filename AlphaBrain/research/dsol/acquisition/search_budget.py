"""Mixed query budgets with a separate train-only critic field accessor."""

from dataclasses import replace

import numpy as np

from .search_env import catalog_action, permute_observation


def attention_observation(core, observation):
    """Absolute counts preserve the distinction between 4/8/16 budgets."""
    result = {k: v.copy() for k, v in observation.items()}
    result["flags"][:, 1] *= core.config.query_budget / 16.
    count = len(core.visited)
    result["global"] = np.asarray([
        count / 16., (core.config.query_budget - count) / 16.,
        core.config.query_budget / 16., core.initial_score / core.scale,
        core.values[core.best, 0] / core.scale, core.source.score_samples / 8.,
    ], np.float32)
    return result


def mixed_budget_env(core, starts, budgets=(4, 8, 16), *, full_accel=None):
    import gymnasium as gym

    starts, budgets = tuple(starts), tuple(budgets)
    if (not starts or not budgets or any(s < 0 or s >= core.n for s in starts)
            or any(b < 2 or b > core.n or b > 16 for b in budgets)
            or core.config.motion_cost != 0):
        raise ValueError("Valid starts/budgets and query-only cost required")
    if full_accel is not None:
        full_accel = np.asarray(full_accel, np.float32).copy()
        if full_accel.shape != (core.n,) or not np.isfinite(full_accel).all() or (full_accel < 0).any():
            raise ValueError("Full critic field must match candidate axis")
    shapes = {"geometry": (core.n, 12), "visual": (core.n, core.source.feature_dim),
              "scores": (core.n, 2), "flags": (core.n, 4), "wrist": core.wrist.shape,
              "global": (6,), "action_mask": (core.n + 1,)}

    class Adapter(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.order = None
            self.action_space = gym.spaces.Discrete(core.n + 1)
            self.observation_space = gym.spaces.Dict({
                k: gym.spaces.MultiBinary(core.n + 1) if k == "action_mask"
                else gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
                for k, shape in shapes.items()})

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            options = options or {}
            start = options.get("start", starts[int(self.np_random.integers(len(starts)))])
            budget = options.get("budget", budgets[int(self.np_random.integers(len(budgets)))])
            if start not in starts or budget not in budgets:
                raise ValueError("Start/budget outside frozen distribution")
            core.config = replace(core.config, query_budget=int(budget))
            self.order = self.np_random.permutation(core.n)
            obs, info = core.reset(start=int(start))
            return self._observation(obs), dict(info, budget=budget)

        def _observation(self, obs):
            return permute_observation(attention_observation(core, obs), self.order)

        def step(self, action):
            obs, reward, done, truncated, info = core.step(catalog_action(int(action), self.order))
            return self._observation(obs), reward, done, truncated, dict(info, budget=core.config.query_budget)

        def action_masks(self):
            mask = core.action_masks()
            mask[1:] = mask[1:][self.order]
            return mask

        def critic_field(self):
            if full_accel is None or self.order is None:
                raise RuntimeError("Privileged field is unavailable outside admitted training")
            return (full_accel[self.order] / core.scale).copy()

    return Adapter()
