"""Resident OpenPI adapter for explicit-noise RoboCasa evaluation.

Imports do not load a model, initialize a GPU, or construct an environment.
"""
import dataclasses

import numpy as np


def policy_request(observation):
    from openpi_client import image_tools

    request = {'prompt': observation['annotation.human.task_description']}
    for target, source in (('image', 'robot0_agentview_left'),
                           ('right_image', 'robot0_agentview_right'),
                           ('wrist_image', 'robot0_eye_in_hand')):
        request['observation/' + target] = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(observation['video.' + source]), 224, 224))
    request['observation/state'] = np.concatenate([observation['state.' + key] for key in
        ('end_effector_position_relative', 'end_effector_rotation_relative',
         'base_position', 'base_rotation', 'gripper_qpos')])
    return request


class ResidentPolicy:
    def __init__(self, checkpoint, config):
        from openpi.policies import policy_config
        from openpi.training import config as training_config
        from openpi.shared import normalize
        from AlphaBrain.research.dsol.metrics.pi05_jax import make_velocity_sampler

        cfg = training_config.get_config(config)
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, data_dirs=[]))
        self.policy = policy_config.create_trained_policy(cfg, checkpoint,
            norm_stats=normalize.load(checkpoint / 'assets'), sample_kwargs={'num_steps': 10})
        self.sampler = make_velocity_sampler(self.policy._model)

    def infer(self, request, noise):
        return np.asarray(self.policy.infer(request, noise=noise)['actions'])

    def trace(self, request, noise):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_api
        from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores

        inputs = self.policy._input_transform(jax.tree.map(lambda x: x, request))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        observation = model_api.Observation.from_dict(inputs)
        traced = self.sampler(jax.random.key(0), observation, noise=jnp.asarray(noise)[None])
        velocity = np.asarray(traced['velocity_trace'])
        actions = self.policy._output_transform({'state': np.asarray(inputs['state'][0]),
            'actions': np.asarray(traced['actions'][0])})['actions']
        scores = compute_accel_scores(velocity)
        return dict(actions=np.asarray(actions), velocity=velocity,
                    times=np.asarray(traced['times']), steps=int(traced['step_count']),
                    scores={str(k): dict(score=float(v['score'][0]), degenerate=bool(v['degenerate'][0]))
                            for k, v in scores.items()})
