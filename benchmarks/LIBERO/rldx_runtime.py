"""Bounded, eager RLDX LIBERO runtime; imports never load a model or touch CUDA.

Native four-frame inputs and the full 16x64 Gaussian sampler are preserved.
Accel observes only the eight executed actions and seven real dimensions.
Joint cognition tokens are not mislabeled as independent camera features.
"""
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.execution.manifest import file_digest

SCHEMA='RLDX_LIBERO_RUNTIME_V1'
SAMPLER=dict(timesteps=[0.,.25,.5,.75],terminal_time=1.,horizon=16,
             dimensions=64,executed_actions=8,valid_dimensions=7,rtc=False,memory=False)
HISTORY=[-6,-4,-2,0]


def validate_contract(path,stamp,*,weights=False):
    if file_digest(path)!=stamp:raise ValueError('RLDX contract identity changed')
    c=read(path)
    if (c['schema']!=SCHEMA or c['sampler']!=SAMPLER or c['video_deltas']!=HISTORY
            or c['attention']!='sdpa' or c['embodiment']!='general_embodiment'):
        raise ValueError('Unreviewed RLDX native sampler/observation contract')
    for p,h in c['input_hashes'].items():
        if file_digest(p)!=h:raise ValueError('Changed RLDX dependency: '+p)
    manifest=read(c['source_manifest'])
    for item in manifest['files']:
        if file_digest(Path(c['source'])/item['path'])!=item['sha256']:
            raise ValueError('Changed pinned RLDX source')
    cfg=read(Path(c['checkpoint'])/'config.json')
    if (cfg.get('use_memory',False) or cfg.get('rtc_inference_mode','none')!='none'
            or cfg.get('action_horizon',16)!=16 or cfg.get('max_action_dim',64)!=64):
        raise ValueError('Memory/RTC/action contract requires another adapter')
    shards=read(Path(c['checkpoint'])/'model.safetensors.index.json')['weight_map']
    if set(shards.values())!={Path(p).name for p in c['checkpoint_shards']}:
        raise ValueError('Incomplete checkpoint shard identity')
    if weights:
        for p,h in c['checkpoint_shards'].items():
            if file_digest(p)!=h:raise ValueError('Changed checkpoint shard')
    return c


class LiberoHistory:
    """Native reset padding, then real per-action sensor history; no fake wrist."""
    def __init__(self,example):
        self.frames=deque([self.frame(example)]*7,maxlen=7)
        self.language=example['lang']

    @staticmethod
    def frame(example):
        from .simulation_inputs import raw_observation
        raw=raw_observation(example['fastwam_raw'])
        q=np.asarray(raw['robot0_eef_quat'],dtype=np.float64).copy();q[3]=np.clip(q[3],-1.,1.)
        den=np.sqrt(1.-q[3]*q[3])
        rot=np.zeros(3) if np.isclose(den,0.,rtol=1e-9,atol=0.) else q[:3]*2.*np.arccos(q[3])/den
        return dict(front_view=raw['agentview_image'][::-1,::-1].copy(),
            left_wrist_view=raw['robot0_eye_in_hand_image'][::-1,::-1].copy(),
            eef_pos_absolute=np.asarray(raw['robot0_eef_pos'],dtype=np.float32).copy(),
            eef_rot_absolute=rot.astype(np.float32),
            gripper_close=np.asarray(raw['robot0_gripper_qpos'],dtype=np.float32).copy())

    def append(self,example):
        if example['lang']!=self.language:raise ValueError('Task changed within an episode')
        self.frames.append(self.frame(example))

    def observation(self):
        frames=list(self.frames)
        return dict(video={k:np.stack([frames[d-1][k] for d in HISTORY])[None]
                           for k in ('front_view','left_wrist_view')},
                    state={k:frames[-1][k][None,None].astype(np.float32)
                           for k in ('eef_pos_absolute','eef_rot_absolute','gripper_close')},
                    language={'annotation.human.action.task_description':[[self.language]]})


def simulator_actions(decoded):
    keys=('eef_pos_delta','eef_rot_delta','gripper_close')
    arrays=[np.asarray(decoded[k],dtype=np.float32) for k in keys]
    if [x.shape for x in arrays]!=[(1,16,3),(1,16,3),(1,16,1)]:
        raise ValueError('Native LIBERO action decoding shape changed')
    action=np.concatenate(arrays,axis=-1)[0].copy()
    # Native SimPolicyWrapper first converts close -> open (1-close),
    # then LiberoEnv maps [0,1] to +/-1 and inverts for robosuite.
    action[:,-1]=1.0-action[:,-1]
    action[:,-1]=-np.sign(2.*action[:,-1]-1.)
    if not np.isfinite(action).all():raise ValueError('Nonfinite decoded actions')
    return action


def accel_scores(trace):
    from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
    v=np.asarray(trace)
    if v.shape!=(4,1,16,64) or not np.isfinite(v).all():raise ValueError('Invalid native velocity trace')
    value=compute_accel_scores(v[:,:,:8,:7].transpose(1,0,2,3),prefixes=(3,4))
    if any(bool(x['degenerate'][0]) for x in value.values()):raise ValueError('Degenerate native Accel')
    return {str(p):float(x['score'][0]) for p,x in value.items()}


class RLDXRuntime:
    def __init__(self,contract_path,stamp,output,limit):
        from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
        reject_production('benchmarks/LIBERO/rldx_runtime.py:RLDXRuntime.__init__')
        if type(limit) is not int or limit<1:raise ValueError('Finite explicit call budget required')
        self.contract=c=validate_contract(contract_path,stamp,weights=True)
        self.output=Path(output);self.limit=limit;self.calls=0;self.failed=False
        if not self.output.is_dir() or (self.output/'model-attempts.jsonl').exists():raise ValueError('New runtime accounting required')
        os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                          RLDX_ATTN_IMPL='sdpa',NO_ALBUMENTATIONS_UPDATE='1')
        sys.path.insert(0,c['source'])
        import torch
        import rldx
        if not Path(rldx.__file__).resolve().is_relative_to(Path(c['source']).resolve()):raise ValueError('Wrong RLDX source imported')
        from rldx import RLDX,RLDXConfig,RLDXProcessor,EmbodimentTag
        from rldx.policy.policy_runtime import PolicyRuntime
        cfg=RLDXConfig.from_pretrained(c['checkpoint'],local_files_only=True)
        cfg.model_name=c['vlm_metadata']  # local-path binding only, original config is immutable
        started=time.monotonic()
        self.model,info=RLDX.from_pretrained(c['checkpoint'],config=cfg,torch_dtype=torch.bfloat16,
            output_loading_info=True,local_files_only=True,
            transformers_loading_kwargs={'local_files_only':True,'skip_pretrained_weights':True})
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Checkpoint does not strictly cover native model: '+repr(info))
        if self.model.config.action_horizon!=16 or self.model.config.max_action_dim!=64:
            raise ValueError('Loaded native action shape differs')
        self.model.eval().requires_grad_(False).to('cuda:0')
        pc=deepcopy(read(Path(c['checkpoint'])/'processor_config.json')['processor_kwargs'])
        pc.update(model_name=c['vlm_metadata'],statistics=read(Path(c['checkpoint'])/'statistics.json'),
                  embodiment_id_mapping=read(Path(c['checkpoint'])/'embodiment_id.json'))
        processor=RLDXProcessor(**pc,transformers_loading_kwargs={'local_files_only':True})
        processor.eval()
        modalities=processor.get_modality_configs()[c['embodiment']]
        if (modalities['video'].delta_indices!=HISTORY or
                modalities['video'].modality_keys!=['front_view','left_wrist_view']):
            raise ValueError('Real camera roles or temporal contract changed')
        if getattr(self.model,'use_memory',False) or self.model.action_model._rtc.enabled_inference():
            raise ValueError('Live model memory/RTC differs from contract')
        self.pipeline=PolicyRuntime(model=self.model,processor=processor,modality_configs=modalities,
            embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,collate_fn=processor.collator,
            language_key=modalities['language'].modality_keys[0],registry=None,infer_lock=threading.Lock(),
            use_memory=False,rtc_inference_mode='none',rtc_inference_delay=0,rtc_exec_horizon=8,rtc_enabled=False)
        (self.output/'loading.json').write_text(json.dumps(dict(strict=True,loading_info=info,
            load_seconds=time.monotonic()-started,attention='sdpa',torch=torch.__version__,
            parameter_dtypes=sorted({str(p.dtype) for p in self.model.parameters()})))+'\n')

    def bind_task(self,output,limit):
        """Only accounting moves across sequential successful resident leases."""
        from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
        reject_production('benchmarks/LIBERO/rldx_runtime.py:RLDXRuntime.bind_task')
        output=Path(output)
        if (self.failed or type(limit) is not int or limit<1 or not output.is_dir()
                or not 0<=self.calls<=self.limit or (output/'model-attempts.jsonl').exists()
                or (output/'calls.jsonl').exists()):
            raise RuntimeError('Unknown or previously used RLDX block cannot be rebound')
        self.output,self.limit,self.calls=output,limit,0

    def _append(self,name,row):
        with (self.output/name).open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())

    def infer(self,history,seed,identity,*,instrument=True,features=False):
        from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
        reject_production('benchmarks/LIBERO/rldx_runtime.py:RLDXRuntime.infer')
        if self.failed or self.calls>=self.limit:raise RuntimeError('RLDX budget exhausted or prior unknown outcome')
        if type(seed) is not int or not 0<=seed<2**63:raise ValueError('Explicit native RNG seed required')
        import torch
        from rldx.policy.step_request import StepRequest
        from AlphaBrain.research.dsol.acquisition.flow_trace import TokenTimeFlowTrace
        request=StepRequest(obs=history.observation(),batch_size=1)
        _,states,collated=self.pipeline._prepare_inputs(request)
        trace=TokenTimeFlowTrace(timesteps=SAMPLER['timesteps'],terminal_time=1.,horizon=16,dimensions=64)
        head=self.model.action_model;started=time.monotonic();self.calls+=1
        row=dict(index=self.calls,seed=seed,identity=identity,instrument=instrument,features=features)
        self._append('model-attempts.jsonl',row)
        try:
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed);torch.cuda.manual_seed(seed)
                with trace.capture(head.action_encoder,head.action_decoder) if instrument else nullcontext():
                    pred=self.pipeline._forward(collated)
                normalized=pred['action_pred'].detach().float().cpu().numpy().copy()
                decoded=self.pipeline._decode(pred['action_pred'].float(),states)
                rng_cpu=torch.random.get_rng_state().numpy().copy()
                rng_cuda=torch.cuda.get_rng_state().cpu().numpy().copy()
                result=dict(normalized_actions=normalized,actions=simulator_actions(decoded),rng_cpu=rng_cpu,rng_cuda=rng_cuda)
                if features:
                    for k in ('backbone_features','state_features'):
                        x=pred[k].detach().float().cpu().numpy().copy()
                        if not np.isfinite(x).all():raise ValueError('Nonfinite native features')
                        result[k]=x
                if instrument:
                    result.update(trace.arrays());result['scores']=accel_scores(result['velocities'])
            result.update(seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            self._append('calls.jsonl',dict(row,seconds=result['seconds'],complete=True))
            return result
        except BaseException:
            self.failed=True
            raise
