"""Explicit-session Gym adapter for STOP/degree increments, not a candidate ID."""
import gymnasium as gym
import numpy as np
from .local_view_gym import padded_observation


def continuous_observation(obs):
    extra = {'joint_state', 'delta_low', 'delta_high'}
    out = padded_observation({k:v for k,v in obs.items() if k not in extra}, 7)
    out.update({k:np.array(obs[k], copy=True) for k in extra})
    return out


class ContinuousSearchGym(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, visual_dim):
        from .local_view_gym import LocalSearchGym
        base = LocalSearchGym(visual_dim)
        spaces = {}
        for k, space in base.observation_space.spaces.items():
            if k == 'action_mask': spaces[k] = gym.spaces.MultiBinary(8)
            elif k in ('wrist', 'global'): spaces[k] = space
            else: spaces[k] = gym.spaces.Box(-np.inf, np.inf, (7, *space.shape[1:]), np.float32)
        spaces.update(joint_state=gym.spaces.Box(-1, 1, (6,), np.float32),
            delta_low=gym.spaces.Box(-10, 10, (2,), np.float64),
            delta_high=gym.spaces.Box(-10, 10, (2,), np.float64))
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = gym.spaces.Dict(stop=gym.spaces.Discrete(2),
            delta_deg=gym.spaces.Box(-10, 10, (2,), np.float64))
        self.visual_dim, self.core = visual_dim, None

    def reset(self, *, seed=None, options=None):
        from .continuous_search import ContinuousSearch
        if self.core is not None and self.core.status != 'COMPLETE':
            raise ValueError('Cannot discard an active/failed search')
        core = (options or {}).get('session')
        if (not isinstance(core, ContinuousSearch) or core.status != 'NEW'
                or core.config.query_budget != 8 or len(core.wrist) != self.visual_dim):
            raise ValueError('Explicit fresh continuous session required')
        super().reset(seed=seed); self.core = core
        return continuous_observation(core.start()), core.info()

    def step(self, action):
        if self.core is None or not self.action_space.contains(action):
            raise ValueError('Reset first and provide a valid continuous action')
        obs, reward, done, truncated, info = self.core.step(dict(stop=bool(action['stop']), delta_deg=action['delta_deg']))
        return continuous_observation(obs), reward, done, truncated, info
