"""Bound FastWAM LIBERO adapter; no model work or downloads at import time.

Input/decoder conventions follow the pinned FastWAM LIBERO implementation.
The operation model sees its native pixels; the transferred actor sees the
separately admitted source visual bridge. Neither is substituted for the other.
This candidate is not GPU/numerically admitted merely by passing CPU tests.
"""
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.execution.manifest import file_digest as sha

SCHEMA = 'FASTWAM_LIBERO_RUNTIME_V1'
NOISE_RULE = 'fastwam-libero-tensor-address-torch-cpu-seed63-v1'
from ..simulation_inputs import RAW_OBSERVATION
PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
SAMPLER = dict(steps=10, horizon=32, dimensions=7, sigma_shift=5., replan_steps=10,
               compile=False, tiled=False, rand_device='cpu', binarize_gripper=True)


def noise_seed(noise):
    """Bind native CPU RNG to an existing scientific noise *address*.

    The input tensor is not FastWAM's latent. This explicit domain-separated
    mapping preserves pairing across poses/hosts, not cross-model latent values.
    No mutable global RNG, worker id or outcome participates in the address.
    """
    value = np.ascontiguousarray(noise, dtype='<f4')
    if value.shape != (10, 7) or not np.isfinite(value).all():
        raise ValueError('Expected finite (10, 7) scientific noise address')
    return int.from_bytes(hashlib.sha256(NOISE_RULE.encode()+b'\0'+value.tobytes()).digest()[:8],
                          'little') & ((1 << 63)-1)


from ..simulation_inputs import raw_observation


def model_inputs(example, processor, *, device, dtype):
    """Official rotate → PIL bilinear resize → horizontal concat → bf16 scale."""
    import torch
    from PIL import Image
    raw = example['fastwam_raw']
    if raw.get('schema') != RAW_OBSERVATION:
        raise ValueError('Explicit raw FastWAM observations required')
    raw = raw_observation(raw)
    images = []
    for key in ('agentview_image','robot0_eye_in_hand_image'):
        image = Image.fromarray(np.ascontiguousarray(raw[key][::-1,::-1]))
        # This adapter admits only the square 256px LIBERO renderer. No hidden
        # crop/interpolation of the source-policy's already-resized 224px view.
        images.append(np.asarray(image.resize((224,224), resample=Image.Resampling.BILINEAR),
                                 dtype=np.uint8))
    pixels = torch.tensor(np.concatenate(images,axis=1)).permute(2,0,1).unsqueeze(0)
    pixels = pixels.to(device=device,dtype=dtype) * (2.0/255.0) - 1.0
    quat = raw['robot0_eef_quat'].copy()
    quat[3] = np.clip(quat[3],-1.,1.)
    den = np.sqrt(1.-quat[3]*quat[3])
    axis = np.zeros(3) if math.isclose(den,0.) else quat[:3]*2.*math.acos(quat[3])/den
    state = np.concatenate((raw['robot0_eef_pos'],axis,raw['robot0_gripper_qpos'])).astype(np.float32)
    batch = {'state': {'default': torch.as_tensor(state,dtype=torch.float32).unsqueeze(0)}}
    batch = processor.normalizer.forward(processor.action_state_transform(batch))
    proprio = batch['state']['default']
    if tuple(proprio.shape) != (1,8) or not torch.isfinite(proprio).all():
        raise ValueError('Invalid native FastWAM proprioception')
    return pixels, proprio


def decode_actions(action, processor):
    """Keep all 32 decoded native actions; caller executes the first 10."""
    import torch
    if tuple(action.shape) != (32,7) or not torch.isfinite(action).all():
        raise ValueError('Invalid native FastWAM action chunk')
    action = action.unsqueeze(0).to(dtype=torch.float32,device='cpu')
    value = processor.normalizer.normalizers['action']['default'].backward(action).numpy()[0].copy()
    # Upstream: [0 close,1 open] → [-1 close,+1 open] → LIBERO sign → binarize.
    value[:,-1] = np.sign(-(value[:,-1]*2-1))
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite decoded FastWAM actions')
    return value


def source_pixels(example, *, device):
    """Source Pi0.5 uint8 224px inputs, independently of FastWAM's raw views."""
    import torch
    if len(example['image']) != 2:
        raise ValueError('Both source visual views required')
    out = {}
    for name,image in zip(('external','wrist'),example['image']):
        value = np.asarray(image)
        if value.shape != (224,224,3) or value.dtype != np.uint8:
            raise ValueError('Exact source-policy 224px uint8 pixels required')
        # Matches openpi.models.model.Observation.from_dict uint8 conversion.
        # Preserve Policy.infer's batching order/strides as well as values.
        # Unsqueezing after CHW permutation gives batch stride 3, unlike the
        # native 150528 stride; CUDA convolutions may then choose other kernels.
        batch=torch.from_numpy(np.array(value)).to(device)[None,...]
        out[name] = batch.to(torch.float32).permute(0,3,1,2)/255.*2.-1.
    return out


SCORE_WINDOWS = {
    'accel_prefix3_native_velocity_all32x7_v1': 32,
    'accel_prefix3_native_velocity_first10x7_v2': 10,
}


def score_trace(trace, definition):
    """Score an explicit action window; never change the native sampler.

    The prefix indexes denoising velocities, the window indexes future actions.
    Both definitions preserve all ten native denoising steps and 32 predictions.
    A changed definition must be bound into a new runtime/actor protocol hash.
    """
    from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
    value = np.asarray(trace)
    if (definition not in SCORE_WINDOWS or value.shape != (10, 32, 7)
            or not np.isfinite(value).all()):
        raise ValueError('Explicit FastWAM score definition and full native trace required')
    score = compute_accel_scores(value[None, :, :SCORE_WINDOWS[definition]], prefixes=[3])[3]
    if score['degenerate'].any():
        raise ValueError('Degenerate FastWAM score')
    return float(score['score'][0])


def validate_contract(path, expected_sha256):
    if sha(path) != expected_sha256:
        raise ValueError('FastWAM runtime contract changed')
    c = read(path)
    if (c['schema'] != SCHEMA or c['sampler'] != SAMPLER or c['noise_rule'] != NOISE_RULE
            or c['score_definition'] not in SCORE_WINDOWS
            or c['raw_observation'] != RAW_OBSERVATION):
        raise ValueError('Unsupported FastWAM runtime contract')
    required = {c['checkpoint'], c['statistics'], c['source_manifest'], c['model_config'],
                c['processor_config'], c['visual_checkpoint'], c['visual_config']}
    if not required.issubset(c['input_hashes']):
        raise ValueError('Unbound FastWAM assets')
    for filename, stamp in c['input_hashes'].items():
        if sha(filename) != stamp:
            raise ValueError('FastWAM asset changed: '+filename)
    manifest = read(c['source_manifest'])
    for relative, stamp in manifest['files'].items():
        if sha(Path(c['source'])/relative) != stamp:
            raise ValueError('FastWAM source changed: '+relative)
    cfg = read(c['model_config'])
    if (cfg.get('skip_dit_load_from_pretrain') is not True or cfg.get('load_text_encoder') is not True
            or cfg.get('action_dit_pretrained_path') is not None or cfg.get('proprio_dim') != 8):
        raise ValueError('FastWAM full-checkpoint offline loader contract required')
    return c


def load_components(c,*,load_visual=True):
    """Only called from the authority-guarded constructor; strict weights."""
    from AlphaBrain.research.dsol.execution import dispatch_policy
    dispatch_policy.reject_production('benchmarks/LIBERO/legacy/fastwam_runtime.py:load_components')
    import sys
    import torch
    from AlphaBrain.model.pi05_features import load_standalone_siglip
    source = Path(c['source']).resolve()/'src'
    sys.path.insert(0,str(source))
    os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = c['asset_root']
    os.environ['DIFFSYNTH_SKIP_DOWNLOAD'] = 'true'
    os.environ['HF_HUB_OFFLINE'] = '1'
    from fastwam.runtime import create_fastwam
    import fastwam
    if not Path(fastwam.__file__).resolve().is_relative_to(source):
        raise ValueError('A different FastWAM package is already imported')
    model = create_fastwam(**read(c['model_config']),model_dtype=torch.bfloat16,device='cuda:0')
    payload = torch.load(c['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    if set(payload) != {'mot','proprio_encoder','step','torch_dtype'}:
        raise ValueError('Unrecognized FastWAM checkpoint layout')
    model.mot.load_state_dict(payload['mot'],strict=True)
    model.proprio_encoder.load_state_dict(payload['proprio_encoder'],strict=True)
    model.eval().requires_grad_(False)
    processor = load_processor(c['processor_config'],c['statistics'])
    visual = load_standalone_siglip(c['visual_checkpoint'],
        expected_sha256=c['input_hashes'][c['visual_checkpoint']],
        vision_config=read(c['visual_config']),device='cuda:0') if load_visual else None
    return model,processor,visual


def load_processor(config,statistics):
    """Construct the official processor without dataset/model instantiation."""
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    kwargs=read(config)
    if kwargs['action_state_merger']!={'_target_':'fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign'}:
        raise ValueError('Only the pinned LIBERO state merger is supported')
    kwargs['action_state_merger']=ConcatLeftAlign()
    processor=FastWAMProcessor(**kwargs).eval()
    # Preserve upstream float64 statistics, not torch.tensor(list)'s float32.
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(statistics))
    return processor


class FastWAMRuntime:
    def __init__(self, config, output, limit):
        from AlphaBrain.research.dsol.execution import dispatch_policy
        dispatch_policy.reject_production('benchmarks/LIBERO/legacy/fastwam_runtime.py:FastWAMRuntime.__init__')
        checkpoint = config['models'][config['operation_model']]
        if checkpoint.get('implementation') != 'fastwam_native' or type(limit) is not int or limit < 1:
            raise ValueError('Explicit FastWAM binding and bounded budget required')
        self.contract = validate_contract(checkpoint['runtime_contract'],checkpoint['runtime_contract_sha256'])
        if (checkpoint['path'] != self.contract['checkpoint']
                or checkpoint['weights_sha256'] != self.contract['input_hashes'][checkpoint['path']]):
            raise ValueError('FastWAM operation identity differs')
        self.output,self.limit,self.calls,self.failed = Path(output),limit,0,False
        if not self.output.is_dir() or (self.output/'calls.jsonl').exists():
            raise ValueError('New accounting output directory required')
        self.native_adapter=None
        if config.get('native_features') is not None:
            from ..native_feature_runtime import NativeRuntimeFeatures
            self.native_adapter=NativeRuntimeFeatures(config['native_features'],family='fastwam',model_sha256=checkpoint['weights_sha256'])
        self.model,self.processor,self.visual = (load_components(self.contract) if self.native_adapter is None
                                                 else load_components(self.contract,load_visual=False))

    def infer(self, example, noise, identity, capture=True, *, instrument=True,native_request=None,feature_capture=True):
        from AlphaBrain.research.dsol.execution import dispatch_policy
        dispatch_policy.reject_production('benchmarks/LIBERO/legacy/fastwam_runtime.py:FastWAMRuntime.infer')
        import torch
        from AlphaBrain.research.dsol.acquisition.flow_trace import SchedulerTrace
        if self.failed or self.calls >= self.limit:
            raise ValueError('Failed/unknown invocation or exhausted FastWAM budget')
        if capture and not instrument:
            raise ValueError('Uninstrumented control cannot return a trace or features')
        seed = noise_seed(noise)
        pixels,proprio = model_inputs(example,self.processor,device=self.model.device,dtype=self.model.torch_dtype)
        native_mode=getattr(self,'native_adapter',None) is not None
        native=self.native_adapter if capture and native_mode and feature_capture else None
        if not feature_capture and not native_mode:raise ValueError('Native capture control requires a native runtime')
        if native is not None:native.validate_request(example,native_request)
        elif native_request is not None:raise ValueError('Native request cannot fall back to source visual features')
        actor_pixels = source_pixels(example,device=self.model.device) if capture and not native_mode else None
        if not isinstance(example['lang'],str) or not example['lang'].strip():
            raise ValueError('Nonempty task language required')
        trace = SchedulerTrace(steps=10,horizon=32,dimensions=7)
        self.calls += 1
        self.failed = True  # Cleared only after all outputs validate; never retry unknown calls.
        with (self.output/'calls.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(identity,index=self.calls,noise_rule=NOISE_RULE,seed=str(seed)))+'\n')
            stream.flush();os.fsync(stream.fileno())
        dispatch_policy.reject_production('benchmarks/LIBERO/legacy/fastwam_runtime.py:FastWAMRuntime.infer')
        if str(self.model.device).startswith('cuda'):torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.inference_mode(), (trace.capture(self.model.infer_action_scheduler) if instrument else nullcontext()):
            def infer():return self.model.infer_action(prompt=PROMPT.format(task=example['lang']),input_image=pixels,
                action_horizon=32,proprio=proprio,negative_prompt='',text_cfg_scale=1.,
                num_inference_steps=10,sigma_shift=5.,seed=seed,rand_device='cpu',tiled=False,
                compile_action_infer=False)
            if native is None:pred=infer()
            else:pred,native_features=native.call(self.model,infer,example=example,request=native_request)
            full_actions = decode_actions(pred['action'],self.processor)
            features = self.visual(actor_pixels) if capture and not native_mode else None
        if str(self.model.device).startswith('cuda'):torch.cuda.synchronize()
        result = dict(actions=full_actions[:10].copy(),full_actions=full_actions,
                      seconds=time.perf_counter()-begin,noise_seed=seed)
        if str(self.model.device).startswith('cuda'):
            result['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
            result['peak_reserved_mib']=torch.cuda.max_memory_reserved()/1024**2
        if instrument:
            arrays = trace.arrays()
            result.update(trace=arrays['velocities'][:,0],full_noise=arrays['initial_noise'],
                          timesteps=arrays['timesteps'],deltas=arrays['deltas'])
        if capture:
            result['score'] = score_trace(result['trace'], self.contract['score_definition'])
            result['score_definition'] = self.contract['score_definition']
            if native is not None:result['native_features']=native_features
            elif not native_mode:
                for name in ('external','wrist'):
                    value = features[name].detach().float().cpu().numpy().reshape(-1)
                    if value.shape != (4608,) or not np.isfinite(value).all():
                        raise ValueError('Invalid source visual bridge output')
                    result[name] = value
        self.failed = False
        return result
