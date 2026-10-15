"""Post-P0 diagnostics, explicitly exploratory and not a new method selection."""
from itertools import combinations
from pathlib import Path
import argparse

import numpy as np
from scipy.stats import rankdata

from ..data.artifacts import read, write
from ..data.diagnostic_panel import frozen_choices
from .robocasa_statistics import interval, correlation


def split_noise_diagnostic(y):
    """All six directed 2+2 splits; repeats/partitions are not new initials.

    Report both deterministic catalog-order ties and uniform-among-ties
    expected held-out outcome, so preference for canonical is visible.
    """
    y = np.asarray(y, float)
    if y.ndim != 3 or y.shape[-1] != 4:
        raise ValueError('Expected initial x view x four noise repetitions')
    rows = np.arange(len(y)); records = []
    outcomes = []; tie_outcomes = []
    for train in combinations(range(4), 2):
        test = tuple(i for i in range(4) if i not in train)
        dev, held = y[...,train].mean(-1), y[...,test].mean(-1)
        ties = dev == dev.max(1,keepdims=True)
        selected = dev.argmax(1)
        outcome = held[rows, selected]
        tie_outcome = (held*ties).sum(1)/ties.sum(1)
        records.append(dict(selection_noises=train, evaluation_noises=test,
                            selected_success=float(outcome.mean()), tie_mean_success=float(tie_outcome.mean()),
                            mean_number_tied=float(ties.sum(1).mean())))
        outcomes.append(outcome); tie_outcomes.append(tie_outcome)
    return dict(partitions=records, selected_by_initial=np.mean(outcomes,0),
                tie_mean_by_initial=np.mean(tie_outcomes,0))


def summarize(analysis, output):
    analysis, output = Path(analysis), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    population = read(analysis/'population.json')
    with np.load(analysis/'matrix.npz') as z:
        y, acc = z['success'].astype(float), z['accel_prefixes'][...,1]
    jobs, views = population['jobs'], population['views']
    masks = {split: np.array([j['split']==split for j in jobs]) for split in ('pretrain','target')}
    tasks = np.array([j['task'] for j in jobs]); xyz = np.array([v['parameters'] for v in views])
    geometry = dict(reference=np.arange(len(views))==0, inner_noncanonical=(np.abs(xyz[:,0])<20)&(np.arange(len(views))>0),
                    outer_noncanonical=np.abs(xyz[:,0])>=20,
                    train64=np.array([v['role']=='train' for v in views]),
                    extra32=np.array([v['role'] not in ('train','reference') for v in views]))
    result = dict(scope='Post-outcome exploratory diagnostics; not a fresh heldout claim',
                  caveat='2+2 uses fewer selection noises than original 4-noise winner; neither estimates the exact population Oracle.',
                  models={}, training_geometry={}, selection_reproduction={})
    q = y.mean(-1)
    with np.load(analysis/'method-arrays.npz') as methods:
        for m, arm in enumerate(('canonical','broad')):
            choices = frozen_choices(y[m], methods[arm+'__score_accel'], methods[arm+'__score_rgb_accel'])
            rows = np.arange(len(jobs))
            for method, key in (('canonical','canonical'),('min_accel','min_accel'),('rgb_accel','rgb_accel'),('empirical_winner','empirical_oracle')):
                if not np.array_equal(q[m,rows,choices[method]], methods[arm+'__'+key]):
                    raise ValueError('Frozen confirmation selection differs from published method: '+arm+'/'+method)
            result['selection_reproduction'][arm]='PASS'
            result['models'][arm] = {}
            for split, mask in masks.items():
                local_y, local_q, local_acc = y[m,mask], q[m,mask], acc[m,mask]
                local_tasks = tasks[mask]
                noise = split_noise_diagnostic(local_y)
                costs = local_acc.mean(-1)
                ranks = (rankdata(costs,axis=1)-1)/(len(views)-1)
                winners = local_q == local_q.max(1,keepdims=True)
                half_rank_r = [correlation(a,b) for a,b in zip(local_acc[...,:4].mean(-1),local_acc[...,4:].mean(-1))]
                half_rank_r = [x for x in half_rank_r if x is not None]
                rho = [correlation(-a,b) for a,b in zip(costs,local_q)]
                rho = [x for x in rho if x is not None]
                order = np.argsort(costs,axis=1,kind='stable')
                quartiles = [float(np.take_along_axis(local_q,order[:,ids],axis=1).mean()) for ids in np.array_split(np.arange(len(views)),4)]
                result['models'][arm][split] = dict(
                    initials=int(mask.sum()), canonical=float(local_q[:,0].mean()), extra32=float(local_q[:,geometry['extra32']].mean()),
                    empirical_oracle=float(local_q.max(1).mean()),
                    crossfit_catalog_ties=interval(noise['selected_by_initial'],local_tasks),
                    crossfit_uniform_ties=interval(noise['tie_mean_by_initial'],local_tasks), partitions=noise['partitions'],
                    oracle_mean_tied_views=float(winners.sum(1).mean()),
                    oracle_unique_fraction=float(np.mean(winners.sum(1)==1)),
                    oracle_winner_mean_accel_percentile=float(np.mean((ranks*winners).sum(1)/winners.sum(1))),
                    oracle_winner_low_accel_quartile_fraction=float(np.mean(((ranks<=.25)*winners).sum(1)/winners.sum(1))),
                    score_half_noise_rank_median=float(np.median(half_rank_r)),
                    score_half_noise_winner_agreement=float(np.mean(local_acc[...,:4].mean(-1).argmin(1)==local_acc[...,4:].mean(-1).argmin(1))),
                    score_success_rank_median=float(np.median(rho)), defined_rank_initials=len(rho),
                    success_by_accel_quartile=quartiles,
                    min_accel=interval(local_q[np.arange(len(local_q)), costs.argmin(1)],local_tasks))
    for split, mask in masks.items():
        result['training_geometry'][split] = {}
        for group, vm in geometry.items():
            before, after = q[0,mask][:,vm].mean(1), q[1,mask][:,vm].mean(1)
            result['training_geometry'][split][group] = dict(views=int(vm.sum()), canonical_training=float(before.mean()),
                broad_training=float(after.mean()), paired_gain=interval(after-before,tasks[mask]))
    write(output/'diagnostics.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.analysis,args.output)
    from ..plotting.robocasa_attribution import plot_diagnostics
    plot_diagnostics(result,args.output/'attribution.png')


if __name__=='__main__':
    main()
