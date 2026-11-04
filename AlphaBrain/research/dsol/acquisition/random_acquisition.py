"""Independent legal continuous acquisition, followed by the shared minimum.

The sampler sees only motion bounds and a private seeded RNG. It cannot inspect
images, scores, Local candidates or PPO decisions. The shared search session
owns scoring, minimum-visited selection and charged return verification.
"""
import numpy as np


class UniformIncrementAcquisition:
    def __init__(self,seed):
        if type(seed) is not int or seed<0:raise ValueError('Explicit independent acquisition seed required')
        self.rng=np.random.default_rng(seed)

    def action(self,delta_low,delta_high):
        low,high=np.asarray(delta_low,dtype=np.float64),np.asarray(delta_high,dtype=np.float64)
        if (low.shape!=(2,) or high.shape!=(2,) or not np.isfinite([low,high]).all()
                or np.any(low>high)):
            raise ValueError('Finite ordered legal motion intervals required')
        return dict(stop=False,delta_deg=self.rng.uniform(low,high))
