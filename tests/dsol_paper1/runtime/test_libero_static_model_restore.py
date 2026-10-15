import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from benchmarks.LIBERO.initialization import initialize,capture_model_reference


class StaticModelRestoreTest(unittest.TestCase):
    def test_restore_between_reset_and_official_state(self):
        model=SimpleNamespace(body_pos=np.zeros((2,3)),body_quat=np.tile([1.,0,0,0],(2,1)))
        class Env:
            def __init__(self):self.env=SimpleNamespace(sim=SimpleNamespace(model=model));self.seen=None
            def seed(self,seed):pass
            def reset(self):model.body_pos[:]=99
            def set_init_state(self,state):self.seen=model.body_pos.copy();return {}
            def check_success(self):return False
            def step(self,action):return {},0,False,{}
        env=Env();ref=capture_model_reference(env)
        spec=dict(environment_seed=41,initialization=dict(official_file='f',official_file_sha256='x',init_state_index=0,settle_steps=10))
        modules={'scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views':SimpleNamespace(DUMMY_ACTION=np.zeros(7)),
                 'evaluate_pi05_libero_plus_views':SimpleNamespace(physics_state_sha256=lambda env:('hash',1))}
        with patch.dict(sys.modules,modules):
            initialize(env,spec,np.zeros(3),model_reference=ref)
        np.testing.assert_array_equal(env.seen,np.zeros((2,3)))
        np.testing.assert_array_equal(ref['body_pos'],np.zeros((2,3)))

