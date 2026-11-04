"""Bounded native GR00T runtime candidate; no import-time model or GPU access.

This adapter supplies raw actions and native flow observations, not fabricated
pi0.5 visual features. Every entry requires a live registered controller scope
and a frozen runtime contract; import or CPU tests grant no execution permit.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.execution.manifest import file_digest
from .groot_flow_capture import (UPSTREAM_SHA256, graph_generation,
                                  install_graph_observer, read_graph_trace)

SCHEMA = 'GROOT_N15_LIBERO_RUNTIME_V1'
NOISE_RULE = 'GROOT_N15_NATIVE_CUDA_SEED_FROM_F32_10X7_V1'
SAMPLER = dict(flow_steps=4, predicted_actions=10, effective_dimensions=7,
               padded_dimensions=32, rtc=False, replan_steps=10)


def noise_seed(noise):
    value = np.ascontiguousarray(noise, dtype='<f4')
    if value.shape != (10, 7) or not np.isfinite(value).all():
        raise ValueError('Finite scientific (10,7) noise address required')
    return int.from_bytes(hashlib.sha256(NOISE_RULE.encode()+b'\0'+value.tobytes()).digest()[:8],
                          'little') & ((1 << 63)-1)


def validate_contract(path, expected_sha256):
    if file_digest(path) != expected_sha256:
        raise ValueError('GR00T runtime contract identity changed')
    c = read(path)
    if (c['schema'] != SCHEMA or c['sampler'] != SAMPLER
            or c['noise_rule'] != NOISE_RULE
            or c['suite'] not in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')):
        raise ValueError('Unreviewed native GR00T contract')
    source = Path(c['source'])
    expected_config = source/f"configs/gr00tn15/gr00tn15_eagle_3b_{c['suite']}_full_finetune.py"
    if Path(c['model_config']) != expected_config:
        raise ValueError('Model/suite config identity differs')
    required = [c[k] for k in ('checkpoint', 'model_config', 'statistics', 'import_manifest')]
    required += [str(Path(c['tokenizer'])/x) for x in
                 ('vocab.json', 'merges.txt', 'added_tokens.json',
                  'special_tokens_map.json', 'tokenizer_config.json')]
    required += [str(p) for p in Path(c['tokenizer']).rglob('*') if p.is_file()]
    required += [str(source/'fluxvla/models/third_party_models/eagle2_hg_model/config.json')]
    if any(p not in c['input_hashes'] for p in required):
        raise ValueError('Unbound GR00T dependency')
    for p, stamp in c['input_hashes'].items():
        if file_digest(p) != stamp: raise ValueError('Changed GR00T asset: '+p)
    manifest = read(c['import_manifest'])
    for name, stamp in manifest['preserved_python'].items():
        if file_digest(source/name) != stamp: raise ValueError('Changed native implementation: '+name)
    for name, stamps in manifest['changed_initializers'].items():
        if file_digest(source/name) != stamps['candidate']: raise ValueError('Changed import surface: '+name)
    if file_digest(source/'fluxvla/models/heads/flow_matching_inference_head.py') != UPSTREAM_SHA256:
        raise ValueError('Unreviewed native inference head')
    kernels = c['kernel_hashes']
    actual_kernels = {str(p.relative_to(source)) for p in source.rglob('*.so')}
    if (len(kernels) != 3 or set(kernels) != actual_kernels
            or any(file_digest(source/p) != h for p,h in kernels.items())):
        raise ValueError('Three pinned native CUDA extensions required')
    return c


class NativeProcessor:
    """Compose pinned upstream transforms; collate like its single-frame dataset."""
    def __init__(self, contract, config):
        from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
        reject_production('benchmarks/LIBERO/groot_runtime.py:NativeProcessor.__init__')
        from fluxvla.tokenizers import PretrainedTokenizer  # registry registration
        from fluxvla.transforms.transform_inputs import ProcessLiberoEvalInputs
        from fluxvla.transforms.transform_images import TransformImage
        from fluxvla.transforms.transform_prompts import ProcessPromptsWithImage
        from fluxvla.transforms.normalize import LiberoProprioFromInputs, DenormalizeLiberoAction
        from fluxvla.engines import build_transform_from_cfg
        transforms = deepcopy(config.eval.dataset.transforms)
        if [x['type'] for x in transforms] != [
                'ProcessLiberoEvalInputs', 'TransformImage',
                'ProcessPromptsWithImage', 'LiberoProprioFromInputs']:
            raise ValueError('Unreviewed GR00T input pipeline')
        transforms[2]['tokenizer']['model_path'] = contract['tokenizer']
        self.transforms = [build_transform_from_cfg(x) for x in transforms]
        self.key = contract['suite']+'_no_noops'
        self.stats = read(contract['statistics'])
        self.denormalize = DenormalizeLiberoAction(norm_stats=self.stats, norm_type='mean_std')

    def inputs(self, example):
        from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
        reject_production('benchmarks/LIBERO/groot_runtime.py:NativeProcessor.inputs')
        import torch
        from .simulation_inputs import raw_observation
        # The legacy transport field contains raw sensor data, not FastWAM features.
        raw = raw_observation(example['fastwam_raw'])
        language = example['lang']
        if not isinstance(language, str) or not language.strip():
            raise ValueError('Explicit task language required')
        data = dict(raw, task_description=language, norm_stats=self.stats[self.key])
        for transform in self.transforms: data = transform(data)
        return dict(images=data['pixel_values'].cuda().unsqueeze(0),
                    img_masks=torch.tensor([data['img_masks']]).cuda(),
                    lang_tokens=torch.tensor(data['lang_tokens']).unsqueeze(0).cuda(),
                    lang_masks=torch.tensor(data['lang_masks'].tolist()).unsqueeze(0).cuda(),
                    states=torch.from_numpy(data['states']).bfloat16().cuda().unsqueeze(0),
                    embodiment_ids=torch.from_numpy(data['embodiment_ids']).int().cuda().unsqueeze(0))


class GrootRuntime:
    def __init__(self, contract_path, contract_sha256, output, limit, *, instrument,native_features=None):
        from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
        reject_production('benchmarks/LIBERO/groot_runtime.py:GrootRuntime.__init__')
        if type(limit) is not int or limit < 1 or type(instrument) is not bool:
            raise ValueError('Explicit bounded runtime required')
        self.contract = c = validate_contract(contract_path, contract_sha256)
        self.native_adapter=None;self.native_needs_warmup=native_features is not None
        if native_features is not None:
            from .native_feature_runtime import NativeRuntimeFeatures
            self.native_adapter=NativeRuntimeFeatures(native_features,family='groot_n15',model_sha256=c['input_hashes'][c['checkpoint']])
        self.output, self.limit, self.calls, self.failed = Path(output), limit, 0, False
        if not self.output.is_dir() or (self.output/'calls.jsonl').exists():
            raise ValueError('New accounting directory required')
        import torch
        from mmengine.config import Config
        from safetensors.torch import load_file
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
        sys.path.insert(0, c['source'])
        import fluxvla
        if not Path(fluxvla.__file__).resolve().is_relative_to(Path(c['source']).resolve()):
            raise ValueError('A different FluxVLA package is already imported')
        from fluxvla.engines import build_vla_from_cfg
        config = Config.fromfile(c['model_config'])
        model_config = deepcopy(config.inference_model)
        model_config['pretrained_name_or_path'] = None
        model_config['vlm_backbone']['vlm_path'] = str(
            Path(c['source'])/'fluxvla/models/third_party_models/eagle2_hg_model')
        self.model = build_vla_from_cfg(model_config)
        self.model.load_state_dict(load_file(c['checkpoint'], device='cpu'), strict=True)
        self.model.eval().requires_grad_(False).to(device='cuda:0', dtype=torch.bfloat16)
        self.processor = NativeProcessor(c, config)
        self.instrument = instrument
        if instrument: install_graph_observer(self.model.vla_head)

    def infer(self, example, noise, identity, *, capture=False, warmup=False,native_request=None,feature_capture=True):
        from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
        reject_production('benchmarks/LIBERO/groot_runtime.py:GrootRuntime.infer')
        import torch
        from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
        if self.failed or self.calls >= self.limit:
            raise ValueError('Failed/unknown call or exhausted GR00T budget')
        if capture and (not self.instrument or warmup):
            raise ValueError('Only an initialized observed model can produce scores')
        native_mode=getattr(self,'native_adapter',None) is not None
        native=self.native_adapter if capture and native_mode and feature_capture else None
        if not feature_capture and not native_mode:raise ValueError('Native capture control requires native runtime')
        if native is not None:native.validate_request(example,native_request)
        elif native_request is not None:raise ValueError('Unbound native feature request')
        previous = graph_generation(self.model.vla_head) if capture else None
        batch = self.processor.inputs(example)
        seed = noise_seed(noise)
        self.calls += 1; self.failed = True
        with (self.output/'calls.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(identity, index=self.calls, noise_rule=NOISE_RULE,
                                         seed=str(seed), warmup=warmup))+'\n')
            stream.flush(); os.fsync(stream.fileno())
        reject_production('benchmarks/LIBERO/groot_runtime.py:GrootRuntime.infer')
        torch.cuda.synchronize(); begin = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            if native is None:normalized=self.model.predict_action(**batch,seed=seed)
            else:normalized,native_packet=native.call(self.model,lambda:self.model.predict_action(**batch,seed=seed),example=example,request=native_request)
        torch.cuda.synchronize()
        normalized = normalized[0].float().cpu().numpy().copy()
        if normalized.shape != (10,7) or not np.isfinite(normalized).all():
            raise ValueError('Invalid native GR00T action chunk')
        actions = self.processor.denormalize(dict(action=normalized.copy(),norm_stats_key=self.processor.key))
        if actions.shape != (10,7) or not np.isfinite(actions).all():
            raise ValueError('Invalid decoded GR00T action chunk')
        result = dict(actions=actions, normalized_actions=normalized,
                      seconds=time.perf_counter()-begin, seed=seed,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        if capture:
            trace = read_graph_trace(self.model.vla_head, previous_generation=previous)
            result.update(trace=trace['effective_velocities'][:,0], initial_noise=trace['initial_noise'],
                          timesteps=trace['times'])
            scores = compute_accel_scores(result['trace'][None], prefixes=[3,4])
            if any(x['degenerate'].any() for x in scores.values()):
                raise ValueError('Degenerate GR00T score')
            result['scores'] = {str(k):float(v['score'][0]) for k,v in scores.items()}
            if native_mode:result['score']=result['scores']['3']
            if native is not None:result['native_features']=native_packet
        if warmup:self.native_needs_warmup=False
        self.failed = False
        return result
