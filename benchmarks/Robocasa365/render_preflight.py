"""Validate native Robosuite EGL device selection before importing a renderer.

This adapter uses physical numeric device IDs, including CUDA_VISIBLE_DEVICES.
GPU UUIDs remain useful inventory identities but this renderer cannot parse
them as device selectors. No environment variables are silently rewritten.
"""


def validate_render_device(device, environ):
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise ValueError('Renderer device must be a nonnegative physical GPU index')
    selected = environ.get('CUDA_VISIBLE_DEVICES')
    if selected is not None:
        entries = selected.split(',')
        if not entries or any(not x.isascii() or not x.isdigit() for x in entries):
            raise ValueError('Native Robosuite EGL requires numeric CUDA_VISIBLE_DEVICES; GPU UUID selectors are unsupported')
        indices = [int(x) for x in entries]
        if len(indices) != len(set(indices)) or device not in indices:
            raise ValueError('Renderer device is not uniquely included in visible physical GPU indices')
    egl = environ.get('MUJOCO_EGL_DEVICE_ID')
    if egl is not None and (not egl.isascii() or not egl.isdigit() or int(egl) != device):
        raise ValueError('MUJOCO_EGL_DEVICE_ID must equal the intended physical GPU index')
    for name in ('MUJOCO_GL', 'PYOPENGL_PLATFORM'):
        if environ.get(name, 'egl').lower() != 'egl':
            raise ValueError(name+' differs from the frozen native EGL renderer')
    return dict(physical_device=device, cuda_visible_devices=selected,
                mujoco_egl_device_id=egl, renderer='native EGL', environment_modified=False)
