"""Paired 2x2 state/history-by-view contrasts; no fitting or outcome imputation."""
import numpy as np


def factorial(values):
    """Last axes are origin branch (good,bad), then installed view (good,bad)."""
    y=np.asarray(values,dtype=float)
    if y.shape[-2:]!=(2,2) or not np.isfinite(y).all():
        raise ValueError('Complete finite 2x2 observations required')
    gg,gb,bg,bb=y[...,0,0],y[...,0,1],y[...,1,0],y[...,1,1]
    return dict(state=.5*((gg-bg)+(gb-bb)),view=.5*((gg-gb)+(bg-bb)),
                interaction=(gg-gb)-(bg-bb),diagonal=gg-bb,
                view_at_good_state=gg-gb,view_at_bad_state=bg-bb)


def summarize(values, delta):
    """Descriptive eight-noise evidence, not independent task-level inference."""
    result={}
    for name,x in factorial(values).items():
        # Conservative common envelope for all contrasts, including interaction.
        positive=int(np.sum(x>4*delta));negative=int(np.sum(x < -4*delta))
        result[name]=dict(mean=float(x.mean()),median=float(np.median(x)),
            values=x.tolist(),positive_beyond_noise=positive,negative_beyond_noise=negative,
            consistent_7_of_8=max(positive,negative)>=7,control_delta=float(delta))
    return result
