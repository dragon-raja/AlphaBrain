"""Read-only analysis of a frozen model/initial/view/repeat panel."""
from pathlib import Path
import json

import numpy as np
from scipy.stats import rankdata

from ..data.artifacts import sha
from .robocasa_statistics import correlation, interval


def association(accel, success):
    """Positive means lower Accel accompanies higher success; both axes ranked."""
    a, q = np.asarray(accel), np.asarray(success)
    if a.shape != q.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(q).all():
        raise ValueError('Expected finite matching initial x view matrices')
    x, y = rankdata(-a, axis=1), rankdata(q, axis=1)
    x -= x.mean(1, keepdims=True)
    y -= y.mean(1, keepdims=True)
    denominator = np.sqrt(np.sum(x*x)*np.sum(y*y))
    individual = [correlation(-aa, qq) for aa, qq in zip(a, q)]
    valid = [v for v in individual if v is not None]
    return dict(spatial_mean_spearman=correlation(-a.mean(0), q.mean(0)),
                pooled_within_initial_rank_correlation=float(np.sum(x*y)/denominator) if denominator else None,
                mean_defined_initial_spearman=float(np.mean(valid)) if valid else None,
                defined_initials=len(valid), total_initials=len(a))


def model_statistics(y, a, tasks, extra):
    q = y.mean(-1)
    candidate_a, candidate_q = a[:, 1:], q[:, 1:]
    order = np.argsort(candidate_a, axis=1, kind='stable')
    quartiles = np.stack([np.take_along_axis(candidate_q, ix, axis=1).mean(1)
                          for ix in np.array_split(order, 4, axis=1)], axis=1)
    # Match the existing panel selector: all 97 options, including reference.
    chosen = a.argmin(1)
    selected = q[np.arange(len(q)), chosen]
    gain = selected-q[:, extra].mean(1)
    low_high = quartiles[:, 0]-quartiles[:, -1]
    rows = []
    for task in np.unique(tasks):
        keep = tasks == task
        rows.append(dict(task=str(task), initials=int(keep.sum()),
                         accel_mean=float(candidate_a[keep].mean()),
                         quartiles=quartiles[keep].mean(0).tolist(),
                         gain=float(gain[keep].mean()), reference=float(q[keep, 0].mean()),
                         stay=float(q[keep][:, extra].mean()), selected=float(selected[keep].mean()),
                         association=association(candidate_a[keep], candidate_q[keep])))
    leave_one = []
    for task in np.unique(tasks):
        keep = tasks != task
        leave_one.append(dict(removed=str(task), gain=float(gain[keep].mean()),
                              low_high_gap=float(low_high[keep].mean())))
    return dict(association=association(candidate_a, candidate_q),
                accel_mean=float(candidate_a.mean()), quartiles=quartiles.mean(0).tolist(),
                low_high_gap=interval(low_high, tasks), gain=interval(gain, tasks),
                reference=float(q[:, 0].mean()), stay=float(q[:, extra].mean()),
                selected=float(selected.mean()), reference_selected_count=int((chosen == 0).sum()),
                per_task=rows, leave_one_task_out=leave_one,
                execution_repeat_gains=(y[np.arange(len(q)), chosen]-y[:, extra].mean(1)).mean(0).tolist())


def load_panel(root):
    root = Path(root)
    selection = json.loads((root/'selection.json').read_text())
    with np.load(root/'analysis/panel_arrays.npz', allow_pickle=False) as z:
        y, a = z['success'].copy(), z['accel'].copy()
        models, initials, views = [z[k].astype(str).tolist() for k in ('models', 'initials', 'views')]
    by_initial = {j['initial_id']: j for j in selection['jobs']}
    by_view = {v['view_id']: v for v in selection['views']}
    jobs = [by_initial[i] for i in initials]
    catalog = [by_view[v] for v in views]
    if y.shape != (*a.shape, 4) or a.shape != (len(models), len(initials), len(views)):
        raise ValueError('Unexpected panel axes')
    if views[0] != 'canonical' or len(views) != 97 or not np.isfinite(a).all():
        raise ValueError('Expected finite full 97-view panel')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Expected binary rollout outcomes')
    return dict(y=y, accel=a, models=models, initials=initials, views=views,
                tasks=np.array([j['task'] for j in jobs]), splits=np.array([j['split'] for j in jobs]),
                coordinates=np.asarray([v['parameters'] for v in catalog[1:]], float),
                extra=np.array([i for i,v in enumerate(catalog) if v['role'] == 'extra_eval']))


def population_sensitivity(root, panel):
    """Verify panel is a true subset, then repeat identical statistics on P0."""
    root = Path(root)
    population = json.loads((root/'population.json').read_text())
    jobs, views = population['jobs'], population['views']
    if list(population['models']) != ['canonical', 'broad']:
        raise ValueError('Unexpected P0 model order')
    with np.load(root/'matrix.npz', allow_pickle=False) as z:
        y = z['success'].copy()
        a = z['accel_prefixes'][..., 1].mean(-1)  # prefixes are 2..10; index 1 is 3
    indices = [next(i for i,j in enumerate(jobs) if j['initial_id'] == key) for key in panel['initials']]
    if not np.array_equal(y[:, indices], panel['y'][1:]) or not np.allclose(
            a[:, indices], panel['accel'][1:], rtol=0, atol=1e-12):
        raise ValueError('Panel/P0 matched cell identity failed')
    tasks, splits = [np.array([j[k] for j in jobs]) for k in ('task', 'split')]
    extra = np.array([i for i,v in enumerate(views) if v['role'] == 'extra_eval'])
    result = dict(matched_panel_verified=True, sources={str(root/k): sha(root/k)
                  for k in ('population.json', 'matrix.npz')}, splits={})
    for split in ('all', 'pretrain', 'target'):
        keep = np.ones(len(tasks), bool) if split == 'all' else splits == split
        result['splits'][split] = {name: model_statistics(y[m, keep], a[m, keep], tasks[keep], extra)
                                  for m,name in enumerate(('canonical', 'broad'))}
    return result


def run(root, output, *, population=None):
    root, output = Path(root), Path(output)
    panel = load_panel(root)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(scope='40-initial diagnostic, 97 views, 4 execution repeats, 8 score repeats; no new rollouts',
                  correlation='Spearman(-Accel, success); spatial means and within-initial ranks reported separately',
                  space='3D rig candidate parameters, not Cartesian camera XYZ; 96 measured noncanonical poses; no interpolation',
                  selector='Per-initial minimum of eight-noise mean Accel over all 97 options; fixed through full task',
                  uncertainty='Task/initial bootstrap fixes observed four-repeat means; not exhaustive noise uncertainty',
                  sources={str(p): sha(p) for p in [root/'selection.json', root/'analysis/panel_arrays.npz',
                                                    root/'analysis/summary.json', Path(__file__)]}, splits={})
    for split in ('all', 'pretrain', 'target'):
        keep = np.ones(len(panel['initials']), bool) if split == 'all' else panel['splits'] == split
        result['splits'][split] = {name: model_statistics(panel['y'][m, keep], panel['accel'][m, keep],
                                  panel['tasks'][keep], panel['extra']) for m,name in enumerate(panel['models'])}
    from ..plotting.view_landscape import plot
    if population is not None:
        result['population_sensitivity'] = population_sensitivity(population, panel)
    result['sources'][str(Path(__file__).parents[1]/'plotting/view_landscape.py')] = sha(Path(__file__).parents[1]/'plotting/view_landscape.py')
    plot(panel, result, output)
    (output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return result
