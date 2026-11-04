"""Compose a paired development uncertainty report from a complete outcome CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from AlphaBrain.research.dsol.analysis.panel_uncertainty import paired_panel_intervals


def build(source, output):
    source, output = Path(source), Path(output)
    rows = list(csv.DictReader(source.open()))
    variants = sorted({r['variant'] for r in rows})
    seeds = sorted({int(r['seed']) for r in rows})
    tasks = sorted({(r['suite'], r['task']) for r in rows})
    panel = {}
    for r in rows:
        key = (r['variant'], int(r['seed']), (r['suite'], r['task']), r['state_key'], r['view_id'], r['repeat'])
        if key in panel:
            raise ValueError('Duplicate paired identity')
        if any(r[c] not in ('True', 'False') for c in ('success', 'stay_success')):
            raise ValueError('Invalid success value')
        panel[key] = (float(r['success'] == 'True'), float(r['stay_success'] == 'True'))
    reference = {(k[2], k[3], k[4], k[5]) for k in panel if k[0] == variants[0] and k[1] == seeds[0]}
    arrays, stay = {}, None
    for variant in variants:
        data, baselines = [], []
        for seed in seeds:
            keys = {(k[2], k[3], k[4], k[5]) for k in panel if k[:2] == (variant, seed)}
            if keys != reference:
                raise ValueError('Incomplete or mismatched panel')
            task_values, task_stay = [], []
            for task in tasks:
                states = sorted({k[1] for k in reference if k[0] == task})
                initial_values, initial_stay = [], []
                for state in states:
                    pairs = sorted(k for k in reference if k[:2] == (task, state))
                    values = [panel[(variant, seed, *k)] for k in pairs]
                    initial_values.append(np.mean([v[0] for v in values]))
                    initial_stay.append(np.mean([v[1] for v in values]))
                task_values.append(initial_values); task_stay.append(initial_stay)
            data.append(task_values); baselines.append(task_stay)
        arrays[variant] = np.asarray(data)
        if stay is not None and not np.array_equal(stay, baselines):
            raise ValueError('Stay labels changed across methods')
        stay = np.asarray(baselines)
    if not np.all(stay == stay[0]):
        raise ValueError('Shared Stay labels changed across policy seeds')
    arrays['Stay'] = stay
    results = {}
    for a, b in [('U12','U3'), ('U12','U0'), ('U12T','U12'), ('U12','Stay'), ('U3','Stay')]:
        result = paired_panel_intervals(arrays[a], arrays[b])
        per_seed = np.asarray(result['per_seed_difference'])
        radius = t.ppf(.975, len(seeds)-1)*per_seed.std(ddof=1)/np.sqrt(len(seeds))
        result['fixed_panel_seed_mean_t_ci95_assumption_sensitive'] = [float(per_seed.mean()-radius), float(per_seed.mean()+radius)]
        results[a+' minus '+b] = result
    output.mkdir(parents=True, exist_ok=True)
    report = {'status':'DESCRIPTIVE_NOT_SELECTION_ADJUSTED', 'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'input':str(source),'variants':variants,'seeds':seeds,'tasks':tasks,'comparisons':results,
              'limitations':['Selected development comparisons: not confirmatory tests.',
                             'Three policy seeds and eight tasks provide limited distributional information.',
                             'Task bootstrap assumes exchangeable task clusters; fixed designed starts/repeats stay bundled.',
                             'Seed t interval assumes independent approximately normal seed-level differences; n=3 cannot verify this.',
                             'All curves use pointwise intervals, not simultaneous or sequential-testing bounds.']}
    (output/'intervals.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['Paired development uncertainty (percentage points)','Intervals are diagnostic; not corrected for architecture selection.','']
    for name, r in results.items():
        fmt=lambda vals: '['+', '.join(f'{v*100:.2f}' for v in vals)+']'
        lines.append(f"{name}: mean={r['mean_difference']*100:.2f}; fixed-policies task/initial={fmt(r['fixed_observed_policies_task_initial_ci95'])}; crossed sensitivity={fmt(r['crossed_seed_task_initial_ci95_sensitivity'])}; fixed-panel seed t={fmt(r['fixed_panel_seed_mean_t_ci95_assumption_sensitive'])}")
    (output/'README.txt').write_text('\n'.join(lines)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); result=build(args.input,args.output)
    print(json.dumps({'status':result['status'],'comparisons':len(result['comparisons'])}))


if __name__ == '__main__':
    main()
