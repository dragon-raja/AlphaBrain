"""Isolated XR-1 adapter using the pinned upstream observation processor.

No model/environment is constructed on import. The original flow integrator is
unchanged; optional instrumentation observes its native velocity calls.
"""
from collections import deque
import importlib.util
from pathlib import Path

import numpy as np


class XR1Policy:
    def __init__(self, checkpoint, upstream, *, attention='flash_attention_2'):
        import torch
        from transformers import AutoModel, AutoProcessor

        checkpoint, upstream = Path(checkpoint), Path(upstream)
        if not (checkpoint/'IMPORT_RECEIPT.json').is_file() or checkpoint.name.endswith('.partial'):
            raise ValueError('Only a completed byte-verified checkpoint may execute')
        spec = importlib.util.spec_from_file_location('dsol_xr1_upstream_entry', upstream/'eval_robocasa365/entry.py')
        entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
        self.entry = entry
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True,
                                                      local_files_only=True, use_fast=False)
        # Reuse the upstream pure message builder without opening its socket.
        self.message_builder = object.__new__(entry.EvalClient)
        self.message_builder.crop_ratio = 0.95
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True,
            local_files_only=True, attn_implementation=attention, dtype=torch.bfloat16).cuda().eval()
        self.model = self.model.to(torch.bfloat16)
        self.history = deque(maxlen=7)

    def reset(self):
        self.history.clear()

    def observe(self, observation):
        self.history.append(observation)

    def inputs(self, instruction):
        if not self.history:
            raise ValueError('No observation history')
        entry = self.entry
        states = entry.sample_history(deque(entry.observation_to_state(o) for o in self.history), 4, 2)
        images = {key: entry.sample_history(deque(o[key] for o in self.history), 4, 2)
                  for key in entry.CAMERA_KEYS}
        state = np.zeros((1,4,60), dtype=np.float32); state[0,:,:14] = states
        return dict(self.processor.apply_chat_template(
            self.message_builder._build_messages(images, instruction), tokenize=True,
            return_dict=True, return_tensors='pt', do_resize=False, state=state,
            robot_type='robocasa365'))

    def infer(self, instruction, seed, *, capture=False):
        import torch

        inputs = self.inputs(instruction)
        inputs = {k: (v.to(device=self.model.device, dtype=self.model.dtype) if v.is_floating_point()
                      else v.to(self.model.device)) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        velocities = []
        original = self.model.dit_forward
        def observed(*args, **kwargs):
            value = original(*args, **kwargs)
            velocities.append(value.detach())
            return value
        if capture:
            self.model.dit_forward = observed
        try:
            with torch.random.fork_rng(devices=[self.model.device.index]), torch.inference_mode():
                torch.manual_seed(int(seed))
                result = self.model(**inputs, num_steps=5)
        finally:
            if capture:
                self.model.dit_forward = original
        native = result.actions.cpu()
        actions = self.processor.decode_action(native, robot_type='robocasa365')[0,:,:12].float().numpy()
        trace = torch.stack(velocities,dim=1).float().cpu().numpy() if capture else None
        return dict(actions=actions, normalized_actions=native.float().numpy(), velocity_trace=trace)


def recorded_observation(state, images):
    """Decode the existing Pi05 request's lossless 16D raw state packing."""
    state = np.asarray(state)
    if state.shape != (16,) or len(images) != 3:
        raise ValueError('Expected 16D raw state and three native RGB arrays')
    keys = ('end_effector_position_relative','end_effector_rotation_relative',
            'base_position','base_rotation','gripper_qpos')
    boundaries = (0,3,7,10,14,16)
    obs = {'state.'+key: state[boundaries[i]:boundaries[i+1]].copy() for i,key in enumerate(keys)}
    for name, image in zip(('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'),images):
        if np.asarray(image).shape != (256,256,3):
            raise ValueError('Native 256 RGB required; do not reuse resized policy input')
        obs['video.'+name] = image
    return obs
