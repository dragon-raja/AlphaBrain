"""Fixed-shape Gym boundary for dynamic proposals; no implicit episode factory.

The caller must position the source and bind the state/noise before reset.
Reset never retries a failed episode or advances a hidden training schedule.
"""
import gymnasium as gym
import numpy as np


def padded_observation(observation, rows):
    count = len(observation['geometry'])
    if count > rows:
        raise ValueError('Dynamic observation exceeds declared row capacity')
    result = {}
    for name, value in observation.items():
        value = np.asarray(value)
        if name in ('wrist', 'global'):
            result[name] = value.astype(np.float32, copy=True)
        else:
            shape = (rows + 1,) if name == 'action_mask' else (rows, *value.shape[1:])
            result[name] = np.zeros(shape, dtype=np.int8 if name == 'action_mask' else np.float32)
            result[name][:len(value)] = value
    return result


class LocalSearchGym(gym.Env):
    """Bind one explicitly supplied LocalViewSearch per reset; no auto-reset."""
    metadata = {'render_modes': []}

    def __init__(self, visual_dim, query_budget=8):
        if type(visual_dim) is not int or visual_dim < 1 or query_budget != 8:
            raise ValueError('First dynamic design requires positive visual dimension and budget eight')
        self.visual_dim = visual_dim
        self.rows = query_budget - 1 + 8  # Seven acquired rows + eight proposals.
        self.action_space = gym.spaces.Discrete(self.rows + 1)
        shapes = dict(geometry=(self.rows, 12), visual=(self.rows, visual_dim),
                      scores=(self.rows, 2), flags=(self.rows, 4), wrist=(visual_dim,))
        shapes['global'] = (6,)
        self.observation_space = gym.spaces.Dict({
            **{k: gym.spaces.Box(-np.inf, np.inf, shape=v, dtype=np.float32) for k, v in shapes.items()},
            'action_mask': gym.spaces.MultiBinary(self.rows + 1)})
        self.core = None

    def reset(self, *, seed=None, options=None):
        if self.core is not None and self.core.status != 'COMPLETE':
            raise ValueError('Cannot discard an active/failed episode; stop and record it')
        core = (options or {}).get('session')
        if (core is None or core.status != 'NEW' or core.config.query_budget != 8
                or len(core.wrist) != self.visual_dim):
            raise ValueError('Explicit fresh compatible session required')
        super().reset(seed=seed)  # Does not change the session proposal/noise seed.
        self.core = core
        return padded_observation(core.start(), self.rows), core.info()

    def action_masks(self):
        if self.core is None:
            raise ValueError('Reset with an explicit session first')
        return padded_observation(self.core.observation(), self.rows)['action_mask'].astype(bool)

    def step(self, action):
        if self.core is None:
            raise ValueError('Reset with an explicit session first')
        obs, reward, terminated, truncated, info = self.core.step(action)
        return padded_observation(obs, self.rows), reward, terminated, truncated, info


def local_baseline_action(core):
    """Nearest current proposal to the best acquired view; frozen tie order."""
    if core.status != 'ACTIVE':
        raise ValueError('Active search required')
    legal = np.flatnonzero(core.observation()['action_mask'][1:]) + 1
    if not len(legal):
        return 0
    history = len(core.records)
    best = core.records[core.best][0]
    def key(action):
        i = action - 1 - history
        point = core.proposals[i]
        return (float(np.sum((point - best)**2)), float(np.sum((point - core.current)**2)), i)
    return int(min(legal, key=key))
