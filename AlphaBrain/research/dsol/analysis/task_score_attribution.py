"""Read-only scorer/executor crossovers on a frozen paired evaluation matrix.

Exploratory diagnostics, not an independently evaluated new selector.
"""
from pathlib import Path
import argparse
import numpy as np
from ..data.artifacts import read, write, sha
from .robocasa_statistics import correlation


def crossover(success, scores):
    """Average scoring noise before selection; never choose from success labels."""
    y=np.asarray(success,float);a=np.asarray(scores,float)
    if y.ndim!=4 or a.ndim!=4 or y.shape[:3]!=a.shape[:3] or y.shape[0]!=2:
        raise ValueError('Expected two models x matched initials x views x repeats')
    if not np.isfinite(y).all() or not np.isfinite(a).all():raise ValueError('Nonfinite matrix')
    choices=a.mean(-1).argmin(-1);rows=np.arange(y.shape[1])
    outcomes=np.stack([np.stack([y[e,rows,choices[s]] for e in range(2)]) for s in range(2)])
    return choices,outcomes


def paired_interval(delta):
    rng=np.random.default_rng(20260924);delta=np.asarray(delta)
    boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
    return dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        positive=int((delta>0).sum()),zero=int((delta==0).sum()),negative=int((delta<0).sum()),
        scope='Exploratory initial bootstrap; fixed four execution repeats, no multiple-comparison correction')


def analyze(matrix_root,panel_root,output,tasks):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    matrix_root=Path(matrix_root);panel_root=Path(panel_root)
    population=read(matrix_root/'population.json')
    with np.load(matrix_root/'matrix.npz') as z:y=z['success'];a=z['accel_prefixes'][...,1]
    with np.load(matrix_root/'method-arrays.npz') as z:methods={k:z[k] for k in z.files}
    report=dict(scope='Target/heldout scenes only; 97 candidates including canonical; existing complete rollouts. No new inference, movement or training.',
        models=population['models'],tasks={},source_hashes={str(p):sha(p) for p in [matrix_root/'matrix.npz',matrix_root/'population.json',matrix_root/'method-arrays.npz',panel_root/'response-summary.json',Path(__file__)]})
    training=read(panel_root/'response-summary.json')['benchmarks']['robocasa']['models']
    for task in tasks:
        ids=np.array([i for i,j in enumerate(population['jobs']) if j['split']=='target' and j['task']==task])
        if len(ids)==0:raise ValueError('Missing target task: '+task)
        Y,A=y[:,ids],a[:,ids];Q=Y.mean(-1);S=A.mean(-1)
        choices,outcomes=crossover(Y,A)
        item=dict(initials=len(ids),initial_ids=[population['jobs'][i]['initial_id'] for i in ids],
            crossover_rows_scorer_columns_executor=outcomes.mean((2,3)).tolist(),
            same_selected_view_fraction=float(np.mean(choices[0]==choices[1])),
            selected_view_ids=[[population['views'][v] for v in row] for row in choices],
            scorer_change_broad_executor=paired_interval(outcomes[0,1].mean(-1)-outcomes[1,1].mean(-1)),
            per_initial_scorer_change=(outcomes[0,1].mean(-1)-outcomes[1,1].mean(-1)).tolist(),
            execution_split_scorer_change=[float((outcomes[0,1,:,k:k+2]-outcomes[1,1,:,k:k+2]).mean()) for k in [0,2]],
            models={},spatial_cross_rho=[[correlation(-S[s,:,1:].mean(0),Q[e,:,1:].mean(0)) for e in range(2)] for s in range(2)])
        for m,name in enumerate(('canonical','broad')):
            np.testing.assert_allclose(outcomes[m,m].mean(-1),methods[name+'__min_accel'][ids])
            model={key:float(methods[name+'__'+key][ids].mean()) for key in ('canonical','stay_extra32','stay_noncanonical96','min_accel','rgb_accel','empirical_oracle')}
            model['spatial_rho']=correlation(-S[m,:,1:].mean(0),Q[m,:,1:].mean(0))
            model['score_field_split_half_rho']=correlation(A[m,:,1:,:4].mean((0,2)),A[m,:,1:,4:].mean((0,2)))
            model['success_field_split_half_rho']=correlation(Y[m,:,1:,:2].mean((0,2)),Y[m,:,1:,2:].mean((0,2)))
            model['score_halves_selected_success']=[float(crossover(Y,A[...,k:k+4])[1][m,m].mean()) for k in [0,4]]
            # Rank within each initial, not across task difficulty. Exclude canonical.
            order=np.argsort(S[m,:,1:],axis=1,kind='stable')
            ordered=np.take_along_axis(Q[m,:,1:],order,axis=1)
            model['per_initial_accel_quartile_success']=[float(q.mean()) for q in np.split(ordered,4,axis=1)]
            item['models'][name]=model
        item['cross_model_accel_field_rho']=correlation(S[0,:,1:].mean(0),S[1,:,1:].mean(0))
        item['cross_model_success_field_rho']=correlation(Q[0,:,1:].mean(0),Q[1,:,1:].mean(0))
        item['training_sample_not_same_population']={}
        for name in ('canonical_8250','broad_8250'):
            t=training[name];i=t['tasks'].index(task);errors=np.array(t['per_state_view_mse'])[2*i:2*i+2]
            item['training_sample_not_same_population'][name]=dict(per_frame_view_mse=errors.tolist(),
                canonical_mse=float(errors[:,0].mean()),perturbed_mse=float(errors[:,1:].mean()))
        report['tasks'][task]=item
    write(output/'summary.json',report)
    print({t:{k:v for k,v in d.items() if k in ('crossover_rows_scorer_columns_executor','scorer_change_broad_executor','execution_split_scorer_change','models')} for t,d in report['tasks'].items()})
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--matrix-root',type=Path,required=True);p.add_argument('--panel-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',nargs='+',required=True);args=p.parse_args()
    analyze(args.matrix_root,args.panel_root,args.output,args.tasks)
