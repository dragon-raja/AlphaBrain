"""Separate view invariance, natural-state response and demonstration error."""
import numpy as np


def response_metrics(prediction,target,scale,*,execute_steps,dims):
    p=np.asarray(prediction,float);y=np.asarray(target,float);s=np.asarray(scale,float)
    if p.ndim!=5 or y.shape!=(p.shape[0],p.shape[3],p.shape[4]) or p.shape[0]%2:
        raise ValueError('Expected ordered state pairs: state,view,noise,time,channel')
    if not np.isfinite(p).all() or not np.isfinite(y).all() or np.any(s[np.asarray(dims)]<=0):
        raise ValueError('Finite observations and positive fixed scales required')
    dims=np.asarray(dims);p=p[:,:,:,:execute_steps,dims]/s[dims];y=y[:,:execute_steps,dims]/s[dims]
    mean=p.mean(axis=2)
    error=np.mean((mean-y[:,None])**2,axis=(2,3))
    per_noise_error=np.mean((p-y[:,None,None])**2,axis=(2,3,4))
    a,b=mean[::2,0],mean[1::2,0];ya,yb=y[::2],y[1::2]
    dp,dy=b-a,yb-ya
    matched=((a-ya)**2+(b-yb)**2).mean((1,2))/2
    swapped=((a-yb)**2+(b-ya)**2).mean((1,2))/2
    cosine=[];gain=[]
    for x,t in zip(dp,dy):
        norm=np.linalg.norm(x)*np.linalg.norm(t)
        cosine.append(float(np.sum(x*t)/norm) if norm>1e-12 else None)
        gain.append(float(np.sum(x*t)/np.sum(t*t)) if np.sum(t*t)>1e-12 else None)
    return dict(view_response_rms=float(np.sqrt(np.var(p,axis=1).mean())),
        action_rms=float(np.sqrt(np.square(p).mean())),
        canonical_demonstration_mse=float(error[:,0].mean()),
        perturbed_demonstration_mse=float(error[:,1:].mean()),
        canonical_expected_noise_mse=float(per_noise_error[:,0].mean()),
        perturbed_expected_noise_mse=float(per_noise_error[:,1:].mean()),
        per_state_view_mse=error.tolist(),
        canonical_state_response_rms=float(np.sqrt(np.square(dp).mean())),
        target_state_response_rms=float(np.sqrt(np.square(dy).mean())),
        matched_state_mse=matched.tolist(),swapped_state_mse=swapped.tolist(),
        state_assignment_margin=(swapped-matched).tolist(),
        state_change_cosine=cosine,state_change_projection_gain=gain,
        target_state_change_rms=np.sqrt(np.square(dy).mean((1,2))).tolist(),
        mean_prediction_error_only=True,not_closed_loop_success=True)
