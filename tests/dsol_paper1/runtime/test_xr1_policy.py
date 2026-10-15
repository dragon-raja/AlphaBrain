import numpy as np
import pytest
from benchmarks.Robocasa365.xr1_policy import recorded_observation
from benchmarks.Robocasa365.xr1_evaluation import noise_seed


def test_recorded_raw_state_and_native_images():
    state=np.arange(16,dtype=np.float32)
    images=[np.zeros((256,256,3),dtype=np.uint8) for _ in range(3)]
    obs=recorded_observation(state,images)
    assert obs['state.gripper_qpos'].tolist()==[14,15]
    assert obs['state.base_rotation'].tolist()==[10,11,12,13]
    assert obs['state.end_effector_rotation_relative'].tolist()==[3,4,5,6]
    with pytest.raises(ValueError,match='Native'):
        recorded_observation(state,[np.zeros((224,224,3))]*3)


def test_explicit_noise_address_not_call_order():
    before=noise_seed('initial','execution',0,2)
    noise_seed('initial','score',7,0)
    assert noise_seed('initial','execution',0,2)==before
    assert noise_seed('initial','execution',1,2)!=before
    assert noise_seed('initial','execution',0,3)!=before
    assert noise_seed('initial','score',0,2)!=before
