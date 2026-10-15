"""Explicit cached-initial XR-1 inference/trace admission; no task rollout."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--attention',default='flash_attention_2',choices=('flash_attention_2','sdpa'))
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:
        p.error('Explicit --execute required')
    from AlphaBrain.research.dsol.data.artifacts import path_mapping, read, sha, write
    from AlphaBrain.research.dsol.data.evaluation_admission import resolve_initial, load_candidate
    from AlphaBrain.research.dsol.analysis.accel_subspace import panel_cell
    from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
    from benchmarks.Robocasa365.xr1_policy import XR1Policy, recorded_observation
    import torch

    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'status.json',dict(status='PREPARING',new_rollouts=0))
    try:
        with path_mapping([('/share/longjunyu','/share/personal/longjunyu')]):
            plan=read(a.panel/'plan.json'); old=read(plan['baseline_plan']); p0=read(plan['p0_plan'])
            initial=plan['initial_ids'][0]
            admitted=resolve_initial(p0['reference_plan'],p0['reference_release'],initial)
            candidate=load_candidate(admitted,'canonical')
            cell=panel_cell(a.panel,plan,old,initial,'canonical')
            meta=read(cell/'scores.json')
            if sha(cell/'scores.npz')!=meta['arrays_sha256']:
                raise ValueError('Initial score archive changed')
            with np.load(cell/'scores.npz',allow_pickle=False) as z:
                state=z['observation/state'].copy()
        obs=recorded_observation(state,candidate['images'])
        write(a.output/'status.json',dict(status='LOADING_MODEL',initial=initial,new_rollouts=0))
        policy=XR1Policy(a.checkpoint,a.upstream,attention=a.attention)
        policy.observe(obs)
        started=time.monotonic()
        plain=policy.infer(meta['prompt'],1701)
        plain_seconds=time.monotonic()-started
        started=time.monotonic()
        traced=policy.infer(meta['prompt'],1701,capture=True)
        trace_seconds=time.monotonic()-started
        error=float(np.max(np.abs(plain['actions']-traced['actions'])))
        if not np.allclose(plain['actions'],traced['actions'],rtol=1e-5,atol=1e-5):
            raise ValueError('Instrumented forward changed actions')
        v=traced['velocity_trace']
        if v.shape[1]!=5 or traced['actions'].shape[0]<16 or not np.isfinite(v).all():
            raise ValueError('Invalid native action/flow trace')
        scores=compute_accel_scores(v[:,:,:16,:12],prefixes=(3,))[3]
        if scores['degenerate'].any():
            raise ValueError('Degenerate executed-window score')
        np.savez_compressed(a.output/'probe.npz',**traced)
        result=dict(status='INFERENCE_TRACE_SMOKE_PASS',not_rollout_validation=True,new_rollouts=0,
            initial=initial,view='canonical',attention=a.attention,denoise_steps=5,executed_window=16,
            active_dims=12,action_shape=list(traced['actions'].shape),trace_shape=list(v.shape),
            trace_action_max_error=error,accel_p3=float(scores['score'][0]),
            plain_seconds=plain_seconds,trace_seconds=trace_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            checkpoint_receipt_sha256=sha(a.checkpoint/'IMPORT_RECEIPT.json'),
            upstream_entry_sha256=sha(a.upstream/'eval_robocasa365/entry.py'),
            adapter_sha256=sha(Path(__file__).parents[3]/'benchmarks/Robocasa365/xr1_policy.py'),
            historical_input_sha256=meta['arrays_sha256'])
        write(a.output/'status.json',result)
        print(json.dumps(result,ensure_ascii=False))
    except BaseException as exc:
        write(a.output/'status.json',dict(status='FAILED',error=str(exc),new_rollouts=0))
        raise


if __name__=='__main__':
    main()
