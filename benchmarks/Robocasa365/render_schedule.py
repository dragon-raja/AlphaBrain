"""Skip only RGB renders whose observations will not reach the policy.

Environment stepping, controllers, sensors, success checks, and horizons still
run at their original rates. The evaluation caller must require fresh rendering
on the step immediately preceding each policy call. Validate with action replay
before using this optimization in a released runtime.
"""
import numpy as np


class PolicyRenderSchedule:
    def __init__(self, sim):
        self.sim = sim
        self.original = sim.render
        self.had_instance_method = 'render' in vars(sim)
        self.required = True
        self.cache = {}
        self.rendered = self.reused = 0
        sim.render = self.render

    def render(self, *args, **kwargs):
        # Do not suppress depth, segmentation, or unfamiliar positional APIs.
        eligible = not args and not kwargs.get('depth', False) and not kwargs.get('segmentation', False)
        key = tuple(sorted(kwargs.items())) if eligible else None
        if eligible and not self.required and key in self.cache:
            self.reused += 1
            return self.cache[key].copy()
        result = self.original(*args, **kwargs)
        self.rendered += 1
        if eligible and isinstance(result, np.ndarray):
            self.cache[key] = result.copy()
        return result

    def before_step(self, step, replan_steps):
        if step < 0 or replan_steps <= 0:
            raise ValueError('Invalid rendering schedule')
        self.required = (step + 1) % replan_steps == 0

    def close(self):
        if self.had_instance_method:
            self.sim.render = self.original
        else:
            del self.sim.render
