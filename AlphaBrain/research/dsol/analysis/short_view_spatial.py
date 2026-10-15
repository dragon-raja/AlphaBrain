"""Spatial diagnostics of already completed matched-window probes; CPU only."""
import argparse
from pathlib import Path
import numpy as np

from ..data.artifacts import read, write, sha
from .view_landscape import load_panel, association


def spatial_statistics(means, success):
    """Exclude canonical from geometry; keep all 97 options for actual selection."""
    a,q=np.asarray(means,float),np.asarray(success,float)
    if a.ndim!=3 or a.shape[-1]!=5 or a.shape[:2]!=q.shape or q.shape[1]!=97:
        raise ValueError('Expected initial x 97 views x 5 windows and matching success')
    if not np.isfinite(a).all() or not np.isfinite(q).all() or (q<0).any() or (q>1).any():
        raise ValueError('Finite scores and valid probabilities required')
    rows=[]
    for k in range(5):
        aa,qq=a[:,1:,k],q[:,1:]
        order=np.argsort(aa,axis=1,kind='stable')
        quartiles=[float(np.take_along_axis(qq,ix,axis=1).mean()) for ix in np.split(order,4,axis=1)]
        rows.append(dict(window=k+1,association=association(aa,qq),quartile_success=quartiles,
            selected_success=float(q[np.arange(len(q)),a[:,:,k].argmin(1)].mean())))
    return rows


def run(config_path):
    cfg=read(config_path);root=Path(cfg['completed_analysis']);summary=read(root/'summary.json')
    if sha(root/'windows.npz')!=summary['arrays_sha256']:raise ValueError('Completed analysis payload changed')
    panel=load_panel(cfg['panel']);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    result=dict(scope='96 measured rig-parameter candidates, excluding canonical, no interpolation; selector still uses all 97',
        temporal_scope='Every point is the initial camera installation. Later Accel comes from that candidate-specific probe trajectory, not a shared intermediate robot state.',
        success_scope='Original four independent complete-task executions from the initial state; same success field for every scoring window.',
        color_scope='Independent full-range colorbars for each Accel panel; compare numerical bars for absolute scale.',
        sources={str(root/'windows.npz'):sha(root/'windows.npz'),str(Path(cfg['endpoint_arrays'])):sha(cfg['endpoint_arrays']),str(Path(__file__)):sha(__file__)},models={})
    with np.load(cfg['endpoint_arrays'],allow_pickle=False) as z:
        if z['initials'].tolist()!=panel['initials'] or z['views'].tolist()!=panel['views']:raise ValueError('Endpoint axes mismatch')
        y=z['success'].copy();models=z['models'].tolist()
    with np.load(root/'windows.npz',allow_pickle=False) as z:
        if z['initials'].tolist()!=panel['initials'] or z['views'].tolist()!=panel['views']:raise ValueError('Probe axes mismatch')
        for model in summary['models']:
            a=z[model+'__means'];q=y[models.index(model)].mean(-1);splits={}
            for label,split in [('seen','pretrain'),('heldout','target')]:
                keep=panel['splits']==split
                rows=spatial_statistics(a[keep],q[keep]);per_task={}
                for task in np.unique(panel['tasks'][keep]):
                    mask=keep&(panel['tasks']==task)
                    per_task[str(task)]=dict(canonical=float(q[mask,0].mean()),stay=float(q[mask][:,panel['extra']].mean()),
                        windows=spatial_statistics(a[mask],q[mask]))
                splits[label]=dict(initials=int(keep.sum()),windows=rows,per_task=per_task)
                from ..plotting.short_view_spatial import plot
                plot(panel['coordinates'],a[keep,1:].mean(0),q[keep,1:].mean(0),rows,model,label,out)
            result['models'][model]=splits
    write(out/'summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
