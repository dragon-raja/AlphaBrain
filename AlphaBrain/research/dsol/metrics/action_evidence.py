"""Read-only diagnostics of stored commands; these are not task-progress scores."""
import numpy as np

from .accel import compute_accel_scores


def flow_components(velocity, model):
    """Input repeat x observation-stage x denoising-step x horizon x dimension."""
    v=np.asarray(velocity)
    if v.ndim!=5 or not np.isfinite(v).all(): raise ValueError('Invalid saved flow trace')
    if model=='xr1': v=v[:,:,:,:16,:12]
    flat=v.reshape((-1,*v.shape[2:]))
    m=compute_accel_scores(flat,prefixes=(3,))[3]
    if m['degenerate'].any(): raise ValueError('Degenerate trace')
    raw={k:m[k].reshape(v.shape[:2]) for k in ('score','numerator','denominator')}
    return raw, {k:a.mean(0).tolist() for k,a in raw.items()}


def libero_commands(actions):
    """Controller-normalized, clipped commands, not realized robot motion.

    Five commands are executed before the next observation; stage5's commands
    are only proposed. Overlap compares the same scheduled future five steps.
    """
    a=np.asarray(actions,float)
    if a.ndim!=4 or a.shape[1:]!=(5,10,7) or not np.isfinite(a).all():
        raise ValueError('Expected repeat x 5 x 10 x 7 LIBERO chunks')
    c=np.clip(a,-1,1);x=c[:,:,:5,:3];r=c[:,:,:5,3:6]
    overlap=np.linalg.norm(c[:,1:,:5,:3]-c[:,:-1,5:10,:3],axis=-1).mean((0,2))
    return dict(translation_command_norm=np.linalg.norm(x,axis=-1).mean((0,2)).tolist(),
                rotation_command_norm=np.linalg.norm(r,axis=-1).mean((0,2)).tolist(),
                translation_temporal_variation=np.linalg.norm(np.diff(x,axis=2),axis=-1).mean((0,2)).tolist(),
                translation_noise_std=x.std(0).mean((1,2)).tolist(),
                translation_clipped_fraction=(np.abs(a[:,:,:5,:3])>1).mean((0,2,3)).tolist(),
                gripper_positive_fraction=(c[:,:,:5,6]>0).mean((0,2)).tolist(),
                overlapping_replan_translation_change=[None,*overlap.tolist()])
