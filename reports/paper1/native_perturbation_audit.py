"""Audit existing matched outcomes and native first-inference traces on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.analysis.continuous_stop_audit import distribution, paired_recovery
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_trace(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Trajectory digest mismatch: {path}')
    with np.load(path, allow_pickle=False) as arrays:
        trace = arrays['native_full_velocity_traces'][0, :, :, :7]
        noise = arrays['native_full_noises'][0]
    metric = compute_accel_scores(trace[None], prefixes=[3])[3]
    if metric['degenerate'].any():
        raise ValueError('Degenerate first-inference metric')
    return float(metric['score'][0]), noise


def build(binding_path):
    binding = read(binding_path)
    models = {}
    provenance = {str(binding_path): digest(binding_path)}
    for model in sorted({s['model'] for s in binding['tasks'].values()}):
        nominal, paired = {}, {}
        for spec in binding['tasks'].values():
            if spec['model'] != model:
                continue
            root = Path(spec['output'])
            provenance[str(root/'results.json')] = digest(root/'results.json')
            for task in read(root/'results.json')['tasks']:
                key = (task['state_key'], task['repeat'])
                if task['method'] == 'nominal':
                    if key in nominal:
                        raise ValueError('Duplicate nominal reference')
                    nominal[key] = dict(task, root=str(root))
                else:
                    address = (*key, task['start'], task['method'])
                    if address in paired:
                        raise ValueError('Duplicate paired result')
                    paired[address] = task
        rows = []
        for key, stay in paired.items():
            if key[-1] != 'stay':
                continue
            reference = nominal[key[:2]]
            common = min(len(reference['execution_noise_sha256']), len(stay['execution_noise_sha256']))
            if reference['execution_noise_sha256'][:common] != stay['execution_noise_sha256'][:common]:
                raise ValueError('Nominal/stay execution noise mismatch')
            if reference['horizon'] != stay['horizon']:
                raise ValueError('Nominal/stay horizon mismatch')
            row = {k: stay[k] for k in ('state_key','repeat','start','task','initial')}
            row.update(nominal=reference['success'], stay=stay['success'],
                       angles=stay['selected_angles'], matched_noise_replans=common)
            for method in ('local_near_best_accel', 'continuous_ppo'):
                selected = paired[(*key[:3], method)]
                common_selected = min(len(stay['execution_noise_sha256']), len(selected['execution_noise_sha256']))
                if stay['execution_noise_sha256'][:common_selected] != selected['execution_noise_sha256'][:common_selected]:
                    raise ValueError('Selected/stay noise mismatch')
                row[method] = selected['success']
            if model == 'native_reference':
                path = Path(reference['root'])/f"{reference['cohort']}-{reference['initial']:02d}-seed-{reference['repeat']}"/'execution.npz'
                a0, noise0 = first_trace(path, reference['execution_sha256'])
                ap, noisep = first_trace(stay['trajectory'], stay['trajectory_sha256'])
                if not np.array_equal(noise0, noisep):
                    raise ValueError('Native 32D initial execution noise mismatch')
                row.update(nominal_first_accel=a0, perturbed_first_accel=ap, accel_delta=ap-a0)
            rows.append(row)
        methods = {}
        for method in ('local_near_best_accel', 'continuous_ppo'):
            selected = [dict(r, selected=r[method]) for r in rows]
            methods[method] = dict(overall=paired_recovery(selected),
                by_task={t: paired_recovery([r for r in selected if r['task']==t]) for t in sorted({r['task'] for r in rows})},
                by_start={str(s): paired_recovery([r for r in selected if r['start']==s]) for s in sorted({r['start'] for r in rows})})
        result = dict(rows=rows, methods=methods)
        if model == 'native_reference':
            result['first_execution_accel'] = {
                label: dict(n=len(group), nominal=distribution(r['nominal_first_accel'] for r in group),
                    perturbed=distribution(r['perturbed_first_accel'] for r in group),
                    delta=distribution(r['accel_delta'] for r in group),
                    positive_delta=sum(r['accel_delta']>0 for r in group))
                for label, group in [('all',rows), ('stay_failed',[r for r in rows if not r['stay']]),
                                      ('stay_succeeded',[r for r in rows if r['stay']])]}
        models[model] = result
    return dict(models=models, source_sha256=provenance, scientific_calls=0,
        interpretation='First execution inference at identical initial state and 32D noise; not the two-noise search score. Shared nominal references and repeated seeds are not independent tasks.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(args.binding), indent=2)+'\n')


if __name__ == '__main__':
    main()
