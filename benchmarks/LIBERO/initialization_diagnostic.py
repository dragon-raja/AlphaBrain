"""Opt-in LIBERO initialization diagnostics; never modifies saved checkpoints.

Run inside the frozen parent's policy runtime. Native-32 noise extends the
existing seven coordinates deterministically; it is NOT identical 32D noise
to the legacy 7D policy. This is an attribution probe, not a published panel.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def extend_noise(noise, seed, dimension):
    noise = np.asarray(noise, dtype=np.float32)
    if noise.shape != (10, 7) or not np.isfinite(noise).all():
        raise ValueError('Expected finite legacy noise of shape (10, 7)')
    if dimension == 7:
        return noise.copy()
    if dimension != 32:
        raise ValueError('Unsupported diagnostic action dimension')
    # Do not reuse the original PCG64 stream: float32 and float64 normal draws
    # from the same seed share random bits and can correlate across coordinates.
    # A domain-separated seed gives the added coordinates their own stream.
    extra_seed = int.from_bytes(hashlib.sha256(
        f'libero-native32-extra25-v1::{int(seed)}'.encode()).digest()[:16], 'little')
    result = np.empty((10, 32), dtype=np.float32)
    result[:, :7] = noise
    result[:, 7:] = np.random.Generator(np.random.PCG64(extra_seed)).standard_normal((10, 25), dtype=np.float32)
    return result


class DiagnosticPolicy:
    def __init__(self, args):
        import torch
        self.args = args
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        self.last_raw = None
        self.last_trace = {}
        norm = json.loads((args.native_checkpoint/'assets/physical-intelligence/libero/norm_stats.json').read_text())
        self.norm = norm.get('norm_stats', norm)['actions']
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.alpha_checkpoint/'framework_config.yaml')
        self.mean = np.asarray(cfg.framework.normalization.action_mean, dtype=np.float32)
        self.std = np.asarray(cfg.framework.normalization.action_std, dtype=np.float32)
        self.dimension = 7 if args.variant.startswith('alpha') or args.variant.startswith('native7') else 32
        if args.variant.startswith('native'):
            from openpi.training.config import get_config
            from openpi.policies.policy import Policy
            from openpi.shared.normalize import load as load_norm
            from openpi import transforms
            config = get_config('pi05_libero')
            config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
            # Same native builder components as create_trained_policy. Avoid its
            # import of unrelated training data loaders (LeRobot is not needed
            # for inference); do not replace any model or transform algorithm.
            model = config.model.load_pytorch(config, str(args.native_checkpoint/'model.safetensors'))
            model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
            data = config.data.create(config.assets_dirs, config.model)
            stats = load_norm(args.native_checkpoint/'assets'/data.asset_id)
            self.policy = Policy(model,
                transforms=[transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs,
                            transforms.Normalize(stats, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs],
                output_transforms=[*data.model_transforms.outputs,
                                   transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm), *data.data_transforms.outputs],
                sample_kwargs={'num_steps': 10}, metadata=config.policy_metadata,
                is_pytorch=True, pytorch_device='cuda:0')
            model = self.policy._model
            if self.dimension == 7:
                inp = model.action_in_proj
                out = model.action_out_proj
                new_in = torch.nn.Linear(7, inp.out_features, device=inp.weight.device, dtype=inp.weight.dtype)
                new_out = torch.nn.Linear(out.in_features, 7, device=out.weight.device, dtype=out.weight.dtype)
                with torch.no_grad():
                    new_in.weight.copy_(inp.weight[:, :7]); new_in.bias.copy_(inp.bias)
                    new_out.weight.copy_(out.weight[:7]); new_out.bias.copy_(out.bias[:7])
                model.action_in_proj, model.action_out_proj = new_in, new_out
            sample = self.policy._sample_actions
            def capture(*pos, **kwargs):
                raw = sample(*pos, **kwargs)
                self.last_raw = raw[0].detach().cpu().float().numpy()
                return raw
            self.policy._sample_actions = capture
            # Native output transform also returns quantile-denormalized actions;
            # retain raw samples so each normalization intervention is explicit.
        else:
            from AlphaBrain.model.framework.base_framework import BaseFramework
            self.model = BaseFramework.from_pretrained(str(args.alpha_checkpoint), strict_checkpoint=True)
            self.model = self.model.to(torch.bfloat16).to('cuda:0').eval()
            self.model.use_action_norm = False
            self.model.gripper_remap = False
        print(json.dumps({'ready': args.variant, 'normalization': args.normalization, 'dimension': self.dimension}), flush=True)

    @property
    def metadata(self):
        return {'diagnostic': True, 'variant': self.args.variant, 'normalization': self.args.normalization,
                'action_dim': 7, 'internal_action_dim': self.dimension, 'action_horizon': 10,
                'explicit_flow_noise': True, 'noise_extension': 'sha256-domain-separated-PCG64-extra25-v1' if self.dimension == 32 else None}

    def infer(self, observation):
        import torch
        noise7 = np.asarray(observation['_eval_noise'], dtype=np.float32)
        if digest(noise7) != observation['_eval_noise_sha256']:
            raise ValueError('Noise identity mismatch')
        noise = extend_noise(noise7, observation['_eval_seed'], self.dimension)
        torch.manual_seed(int(observation['_eval_seed']))
        start = time.perf_counter()
        with torch.inference_mode():
            if self.args.variant.startswith('native'):
                obs = {key: observation[key] for key in ('observation/image', 'observation/wrist_image', 'observation/state', 'prompt')}
                self.policy.infer(obs, noise=noise)
            else:
                example = {'image': [observation['observation/image'], observation['observation/wrist_image']],
                           'lang': observation['prompt'], 'state': observation['observation/state']}
                out = self.model.predict_action(examples=[example], noise=noise[None], return_flow_trace=True)
                self.last_raw = np.asarray(out['normalized_actions'][0], dtype=np.float32)
                self.last_trace = {key: value for key, value in out.items() if key.startswith('flow_')}
        raw = self.last_raw[:, :7]
        if self.args.normalization == 'quantile':
            low = np.asarray(self.norm['q01'], dtype=np.float32)[:7]
            high = np.asarray(self.norm['q99'], dtype=np.float32)[:7]
            actions = (raw+1)/2*(high-low+1e-6)+low
        else:
            actions = raw*self.std+self.mean
        actions = np.ascontiguousarray(np.clip(actions, -1, 1), dtype=np.float32)
        if not np.isfinite(actions).all():
            raise ValueError('Nonfinite actions')
        return {'actions': actions, 'predict_action_wall_seconds': time.perf_counter()-start,
                'explicit_flow_noise': True, 'noise_sha256': digest(noise7), 'full_noise_sha256': digest(noise),
                'action_chunk_sha256': digest(actions)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--alpha-checkpoint', type=Path, required=True)
    p.add_argument('--native-checkpoint', type=Path, required=True)
    p.add_argument('--variant', choices=['alpha', 'native', 'native7'], required=True)
    p.add_argument('--normalization', choices=['quantile', 'meanstd'], required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--port', type=int)
    args = p.parse_args()
    policy = DiagnosticPolicy(args)
    if args.port:
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port, metadata=policy.metadata).serve_forever()
        return
    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
    bank = ExplicitFlowNoiseBank(args.parent/'noise-banks/bank_INITIAL32.manifest.json')
    rows = []
    for state_index in range(0, 32, 4):
        folder = args.parent/'assets/aa0-a'/f'{state_index:02d}'
        asset = np.load(folder/'policy_inputs.npz')
        render = json.loads((folder/'render.json').read_text())
        pair_key = render.get('pair_key')
        if pair_key is None:
            pair_key = json.loads((args.parent/'release.json').read_text())['states'][state_index]['pair_key']
        n = bank.get(pair_key, 0, 0)
        obs = {'observation/image': asset['external_images'][0], 'observation/wrist_image': asset['wrist_images'][0],
               'observation/state': asset['robot_states'][0], 'prompt': render.get('language', render.get('clean_language')),
               '_eval_noise': n['noise'], '_eval_seed': n['noise_seed'], '_eval_noise_sha256': n['noise_sha256']}
        result = policy.infer(obs)
        rows.append({'state_index': state_index, 'pair_key': pair_key, 'prompt': obs['prompt'],
                     'raw': policy.last_raw.tolist(), 'actions': result['actions'].tolist(),
                     'noise7_sha256': result['noise_sha256'], 'full_noise_sha256': result['full_noise_sha256'],
                     'image_sha256': digest(obs['observation/image']), 'wrist_sha256': digest(obs['observation/wrist_image'])})
        print(json.dumps({'state_index': state_index, 'raw_abs_mean': float(np.abs(policy.last_raw).mean()), 'first_action': result['actions'][0].tolist()}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'metadata': policy.metadata, 'rows': rows}, indent=2)+'\n')


if __name__ == '__main__':
    main()
