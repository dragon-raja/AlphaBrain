"""Static text context around any admitted search environment; no model calls."""
import numpy as np
import gymnasium as gym


class TaskConditionedEnv(gym.Wrapper):
    def __init__(self,env,vector):
        super().__init__(env)
        self.vector=np.asarray(vector,dtype=np.float32).copy()
        if self.vector.ndim!=1 or not len(self.vector) or not np.isfinite(self.vector).all():raise ValueError('Finite task vector required')
        self.observation_space=gym.spaces.Dict(dict(env.observation_space.spaces,task_features=gym.spaces.Box(-np.inf,np.inf,self.vector.shape,np.float32)))

    @property
    def core(self):return self.env.core

    def _condition(self,obs):
        if 'task_features' in obs:raise ValueError('Task feature injected twice')
        return dict(obs,task_features=self.vector.copy())

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs);return self._condition(obs),info

    def step(self,action):
        obs,r,done,truncated,info=self.env.step(action)
        return self._condition(obs),r,done,truncated,info
