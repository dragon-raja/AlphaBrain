"""Read-only snapshots of already available observations and simulator arrays.

No simulator imports, rendering, stepping, RNG calls or policy inference.
"""
from pathlib import Path
import numpy as np
from .artifacts import write, sha


def frame(sim, observation):
    result={}
    for key,value in observation.items():
        if isinstance(value,(np.ndarray,float,int,bool,np.number)):
            a=np.asarray(value)
            if a.dtype.kind in 'buif':result['obs/'+key]=a.copy()
    for key in ('qpos','qvel','ctrl','qacc_warmstart','body_xpos','body_xquat','site_xpos','site_xmat'):
        if hasattr(sim.data,key):result['sim/'+key]=np.asarray(getattr(sim.data,key)).copy()
    result['sim/time']=np.asarray(sim.data.time)
    result['sim/contacts']=np.asarray([(c.geom1,c.geom2,c.dist) for c in sim.data.contact[:sim.data.ncon]],dtype=float).reshape(-1,3)
    return result


def names(sim):
    result={}
    for key in ('body_names','geom_names','site_names','joint_names'):
        if hasattr(sim.model,key):
            result[key]=[v.decode() if isinstance(v,bytes) else str(v) for v in getattr(sim.model,key)]
    for key in ('jnt_qposadr','jnt_dofadr','jnt_type','geom_bodyid','site_bodyid'):
        if hasattr(sim.model,key):result[key]=np.asarray(getattr(sim.model,key)).tolist()
    return result


def exact(actual, expected):
    a,b=np.asarray(actual),np.asarray(expected)
    return dict(exact=bool(np.array_equal(a,b)),shape_actual=list(a.shape),shape_expected=list(b.shape),
                max_abs=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.shape==b.shape and a.size else None)


def save_branch(root, arrays, records, identity, complete):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(root/'telemetry.npz',**arrays)
    write(root/'receipt.json',dict(identity=identity,complete=bool(complete),records=records,
        arrays_sha256=sha(root/'telemetry.npz'),scope='Short scoring-noise prefix only; not a new full-task outcome.'))
