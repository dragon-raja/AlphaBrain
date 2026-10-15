"""Fixed-panel forward diagnostic orchestration, with injected model adapter.

No GPU/model work at import time. This is not a closed-loop evaluation.
"""
import hashlib
from pathlib import Path
import time
import numpy as np

from ..data.artifacts import read,sha,write
from ..metrics.accel import compute_accel_scores


def paired_noise(task,repeat,shape,seed):
    address=f'observation-evidence-v1::{seed}::{task}::{repeat}'
    value=int.from_bytes(hashlib.sha256(address.encode()).digest()[:8],'little')
    return np.random.Generator(np.random.PCG64(value)).standard_normal(shape,dtype=np.float32)


def run_probe(panel_file,benchmark,output,identity,predict,shape):
    panel=read(panel_file);spec=panel['benchmarks'][benchmark];out=Path(output);out.mkdir(parents=True,exist_ok=False)
    if sha(spec['payload'])!=spec['payload_sha256']:raise ValueError('Frozen input panel changed')
    arrays=np.load(spec['payload']);repeats=panel['protocol']['noise_repeats'];seed=panel['protocol']['noise_seed']
    actions=[];velocities=[];scores=[];started=time.time()
    receipt=dict(status='RUNNING',benchmark=benchmark,identity=identity,panel_sha256=sha(panel_file),
        source_sha256=sha(__file__),no_rollouts=True,no_training=True,completed_states=0)
    write(out/'status.json',receipt)
    try:
        for i,record in enumerate(spec['records']):
            va=[];vv=[];vs=[]
            for view in range(3):
                na=[];nv=[];ns=[]
                for repeat in range(repeats):
                    noise=paired_noise(record['task'],repeat,shape,seed)
                    a,v=predict(arrays['images'][i,view],arrays['states'][i],record,noise)
                    a=np.asarray(a,dtype=np.float32);v=np.asarray(v,dtype=np.float32)
                    assert a.shape==arrays['actions'][i].shape and v.shape==(10,*shape)
                    assert np.isfinite(a).all() and np.isfinite(v).all()
                    score=compute_accel_scores(v[None],prefixes=(3,))[3]
                    assert not score['degenerate'][0]
                    na.append(a);nv.append(v);ns.append(score['score'][0])
                va.append(np.stack(na));vv.append(np.stack(nv));vs.append(ns)
            actions.append(np.stack(va));velocities.append(np.stack(vv));scores.append(vs)
            receipt.update(completed_states=i+1,total_states=len(spec['records']),elapsed_seconds=time.time()-started)
            write(out/'status.json',receipt);print(f'{i+1}/{len(spec["records"])} states {time.time()-started:.1f}s',flush=True)
        payload=out/'predictions.npz'
        np.savez_compressed(payload,actions=np.stack(actions),velocity_trace=np.stack(velocities),accel=np.array(scores))
        receipt.update(status='COMPLETE',payload_sha256=sha(payload),elapsed_seconds=time.time()-started)
    except Exception as exc:
        receipt.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:write(out/'status.json',receipt)

