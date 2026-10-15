import pytest

from benchmarks.Robocasa365.render_preflight import validate_render_device


@pytest.mark.parametrize('visible', ['GPU-example', '', '-1', '0,0'])
def test_unsupported_selector_is_rejected_without_loading_renderer(visible):
    env={'CUDA_VISIBLE_DEVICES':visible}
    with pytest.raises(ValueError):
        validate_render_device(0,env)
    assert env=={'CUDA_VISIBLE_DEVICES':visible}


def test_explicit_physical_device_selection_and_conflict():
    env={'CUDA_VISIBLE_DEVICES':'0','MUJOCO_EGL_DEVICE_ID':'0','MUJOCO_GL':'egl'}
    assert validate_render_device(0,env)['physical_device']==0
    with pytest.raises(ValueError):
        validate_render_device(1,env)
    with pytest.raises(ValueError):
        validate_render_device(0,dict(env,MUJOCO_EGL_DEVICE_ID='1'))
    with pytest.raises(ValueError):
        validate_render_device(0,dict(env,MUJOCO_GL='osmesa'))
