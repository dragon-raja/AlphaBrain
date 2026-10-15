"""CPU-only exploratory static-rule checks over a frozen four-model matrix.

Does not publish a report or change prior selectors. All rules are declared in
an explicit config before execution; no heldout-based family selection.
"""
from pathlib import Path
import argparse
import gzip
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import rankdata

from ..acquisition.discrete_search import camera_graph
from ..data.artifacts import read, write, sha
from ..selectors.static_view import metric_choices, designs, calibrate_seen
from .rig_geometry import audit_mounts, camera_opencv
from .robocasa_statistics import interval
from .view_landscape import load_panel
from .xr1_fullview import seen_fixed_choices


def score_stability(a, *, draws=1000, seed=20260930):
    """Resample paired scoring addresses across views, never select by outcome."""
    a = np.asarray(a, float)
    if a.ndim != 3 or a.shape[-1] != 8 or not np.isfinite(a).all():
        raise ValueError('Requires eight finite scoring draws per view')
    n, v, _ = a.shape
    rng = np.random.default_rng(seed)
    picks = np.empty((n, draws), int)
    original = a.mean(-1).argmin(1)
    split = [a[..., :4].mean(-1), a[..., 4:].mean(-1)]
    rank = np.stack([rankdata(x, axis=1) for x in split])
    rank -= np.mean(rank, axis=-1, keepdims=True)
    numerator = (rank[0]*rank[1]).sum(1)
    denominator = np.sqrt((rank[0]**2).sum(1)*(rank[1]**2).sum(1))
    correlations = np.divide(numerator, denominator, out=np.full(n, np.nan), where=denominator > 0)
    top = [np.argsort(x, axis=1, kind='stable')[:, :10] for x in split]
    for i in range(n):
        indices = rng.integers(8, size=(draws, 8))
        picks[i] = np.take(a[i], indices, axis=1).mean(-1).argmin(0)
    frequencies = [np.bincount(row, minlength=v).max()/draws for row in picks]
    return picks, dict(half_score_rank_correlation_mean=float(np.nanmean(correlations)),
        half_minimum_same_fraction=float(np.mean(split[0].argmin(1) == split[1].argmin(1))),
        half_top10_overlap_fraction=float(np.mean([len(set(x)&set(y))/10 for x,y in zip(*top)])),
        original_minimum_bootstrap_frequency=float(np.mean(picks == original[:, None])),
        bootstrap_modal_view_frequency=float(np.mean(frequencies)),
        bootstrap_draws=draws, paired_scoring_address_resampling=True)


def noise_split_outcome_selection(y):
    """Outcome-informed selector sensitivity, not a deployment rule/true bound."""
    y = np.asarray(y)
    if y.ndim != 3 or y.shape[-1] % 2 or not np.isin(y,[0,1]).all():
        raise ValueError('Expected binary initial x view x even execution-repeat array')
    half = y.shape[-1]//2
    values = []
    for select, evaluate in [(slice(0,half),slice(half,None)),(slice(half,None),slice(0,half))]:
        winner = y[...,select].mean(-1).argmax(1)
        values.append(y[...,evaluate].mean(-1)[np.arange(len(y)),winner])
    return float(np.mean(values))


def run_supplement(config_path, libero_analysis):
    """Derived comparisons and same-rule LIBERO checks; no rule retuning."""
    cfg = read(config_path);root = Path(cfg['output']);saved = read(root/'summary.json')
    for path,expected in saved['sources'].items():
        if sha(path) != expected:raise ValueError('Source changed after primary diagnostic')
    panel = load_panel(cfg['panel'])
    target = np.flatnonzero(panel['splits'] == 'target')
    with np.load(Path(cfg['analysis'])/'panel_arrays.npz',allow_pickle=False) as z:
        q = z['success'].mean(-1);models = z['models'].tolist()
    comparisons = {}
    pairs = [('calibrated_accel_pose','calibrated_pose'),('calibrated_accel_pose','seen_global_fixed'),
             ('calibrated_accel_task','calibrated_accel'),('min_neighbor_mean','min_mean')]
    with np.load(root/'decisions.npz',allow_pickle=False) as z:
        for m,name in enumerate(models):
            rows = {}
            for left,right in pairs:
                a,b = [z[name+'__'+k][target] for k in (left,right)]
                delta = q[m,target,a]-q[m,target,b]
                rows[left+'_minus_'+right] = dict(interval=interval(delta,panel['tasks'][target]),
                    different_choices=int(np.sum(a != b)), per_initial_delta=delta.tolist())
            comparisons[name] = rows
    libero = Path(libero_analysis)
    provenance = read(libero/'provenance.json')
    for name in ('matrix.npz','states.json'):
        if sha(libero/name) != provenance['artifacts'][name]:raise ValueError('LIBERO audited input changed')
    with np.load(libero/'matrix.npz',allow_pickle=False) as z:
        outcomes, a, poses = z['success'].copy(),z['accel_members'].copy(),z['camera'].copy()
    states = read(libero/'states.json')
    test = np.array([i for i,s in enumerate(states) if s['initialization']['init_state_index'] >= 2])
    tasks = np.array([s['task_id'] for s in states])
    distance = np.array([camera_graph(p)[1] for p in poses])
    lib_results = {}
    for m,name in enumerate(('canonical','broad')):
        q = outcomes[m].mean(-1);stay = q[test,65:].mean(1)
        decisions = metric_choices(a[m],distance,neighborhood=cfg['neighborhood'],se_penalty=cfg['se_penalty'])
        lib_results[name] = dict(stability=score_stability(a[m,test],draws=cfg['bootstrap_draws'],seed=cfg['seed'])[1],
            stay=float(stay.mean()),canonical=float(q[test,0].mean()),empirical_oracle=float(q[test].max(1).mean()),
            outcome_selected_split_noise_crosscheck=noise_split_outcome_selection(outcomes[m,test]),
            methods={k:dict(success=float(q[test,c[test]].mean()),
                           gain=interval(q[test,c[test]]-stay,tasks[test])) for k,c in decisions.items()})
    sources = {str(p):sha(p) for p in (root/'decisions.npz',root/'summary.json',libero/'matrix.npz',libero/'states.json',libero/'provenance.json')}
    target = root/'supplement-source.py'
    shutil.copy2(Path(__file__).resolve(),target)
    result = dict(status='COMPLETE_EXPLORATORY_SUPPLEMENT',paired_comparisons=comparisons,libero=lib_results,
        sources=sources,source_snapshot={str(target):sha(target)},new_rollouts=0,
        scope='No rule retuning; RoboCasa heldout20 four repeats; LIBERO evaluation16 thirty-two repeats; not matched cross-benchmark causality')
    write(root/'diagnostic_supplement.json',result)
    return result


def load_geometry(panel_root, panel):
    selection = read(Path(panel_root)/'selection.json')
    by_initial = {j['initial_id']:j for j in selection['jobs']}
    parameters = np.asarray([v['parameters'] for v in selection['views']], float)
    distances, sources, layouts = [], {}, []
    for initial in panel['initials']:
        job = by_initial[initial]
        root = Path(job['source_cache'])
        summary = read(root/'summary.json')
        if sha(root/'summary.json') != job['source_summary_sha256']:
            raise ValueError('Candidate cache summary differs from the selected source')
        geometry = root/'geometry.npz'
        if sha(geometry) != summary['files']['geometry.npz']['sha256']:
            raise ValueError('Saved geometry hash changed')
        xml_path = Path(job['path'])/'compiled_model.xml.gz'
        xml = ET.fromstring(gzip.decompress(xml_path.read_bytes()))
        names = [c.attrib['name'] for c in xml.find('worldbody').iter('camera')]
        ids = [names.index(c) for c in ('robot0_agentview_left', 'robot0_agentview_right')]
        with np.load(geometry, allow_pickle=False) as g:
            if g['view_ids'].tolist() != panel['views']:
                raise ValueError('Geometry candidate order differs')
            rig = audit_mounts(g['camera_positions'], g['camera_quaternions'], ids, parameters)
        poses = np.stack([camera_opencv(p,r) for p,r in zip(rig['midpoint'],rig['rotations'][:,0])])
        distances.append(camera_graph(poses)[1])
        sources.update({str(p):sha(p) for p in (geometry, root/'summary.json', xml_path)})
        layouts.append(job['layout_id'])
    return np.array(distances), parameters, np.array(layouts), sources


def run(config_path):
    cfg = read(config_path)
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parents[1]
    snapshot = output/'source'; snapshot.mkdir()
    for name in ('analysis/static_view_decisions.py', 'selectors/static_view.py',
                 'analysis/robocasa_statistics.py', 'analysis/xr1_fullview.py',
                 'acquisition/discrete_search.py', 'analysis/rig_geometry.py',
                 'analysis/view_landscape.py', 'data/artifacts.py', 'plotting/static_view_decisions.py'):
        target = snapshot/name;target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package/name, target)
    write(output/'protocol.json', dict(config=cfg, config_sha256=sha(config_path),
        status='EXPLORATORY: heldout aggregates previously inspected, not a blind confirmation',
        scoring='All97 queried before one view selection; paired eight scoring addresses',
        calibration='Seen labels only; leave-one-layout-out alpha CV; no target-based family selection',
        feature_permission='Known rig control offsets and acquired candidate Accel; pose is calibrated-mount prior',
        noise_uncertainty='Bootstrap measures scoring-selection sensitivity; task/initial CI conditions on four execution outcomes',
        new_rollouts=0, physical_camera_search=False))
    panel = load_panel(cfg['panel'])
    root = Path(cfg['analysis'])
    with np.load(root/'panel_arrays.npz', allow_pickle=False) as z:
        y, mean = z['success'].copy(), z['accel'].copy()
        models = z['models'].tolist()
        if z['initials'].tolist() != panel['initials'] or z['views'].tolist() != panel['views']:
            raise ValueError('Analysis panel identities differ')
        xr = z['score_repeats'].copy()
    with np.load(cfg['pi05_score_repeats'], allow_pickle=False) as z:
        if z['initials'].tolist() != panel['initials'] or z['views'].tolist() != panel['views']:
            raise ValueError('Score identities differ')
        pi = z['scores'][..., z['variants'].tolist().index('full32_h50'), :].copy()
    repeats = np.concatenate([pi, xr[None]])
    if y.shape != (4, 40, 97, 4) or not np.allclose(repeats.mean(-1), mean, rtol=0, atol=1e-12):
        raise ValueError('Raw scores do not reproduce the frozen selector scores')
    distance, parameters, layouts, sources = load_geometry(cfg['panel'], panel)
    sources.update({str(p):sha(p) for p in (root/'panel_arrays.npz', Path(cfg['panel'])/'selection.json',
                                          Path(cfg['pi05_score_repeats']), Path(config_path))})
    train = np.flatnonzero(panel['splits'] == 'pretrain')
    test = np.flatnonzero(panel['splits'] == 'target')
    result = dict(status='COMPLETE_EXPLORATORY_CPU', sources=sources, models={}, new_rollouts=0,
                  limitations=['Known ten tasks, two seen/two heldout initials per task',
                               'All97 catalog reused; not unseen-catalog, unseen-task or physical search evidence',
                               'Oracle maximum after four execution-noise means is finite-sample optimistic',
                               'Metric and calibrated rules all require full score acquisition except fixed-view baselines'])
    choices_all, predictions = {}, {}
    for m,name in enumerate(models):
        q = y[m].mean(-1)
        picks, stability = score_stability(repeats[m], draws=cfg['bootstrap_draws'], seed=cfg['seed'])
        choices = metric_choices(repeats[m], distance, neighborhood=cfg['neighborhood'], se_penalty=cfg['se_penalty'])
        fixed = seen_fixed_choices(q, panel['tasks'], panel['splits'])
        choices['seen_global_fixed'] = np.full(len(q), fixed['global_view'])
        choices['seen_task_fixed'] = fixed['per_initial_task_choices']
        fit = {}
        for family, interaction in cfg['calibration_families']:
            key = 'calibrated_'+family+('_task' if interaction else '')
            x = designs(repeats[m], parameters, panel['tasks'], family=family, task_interaction=interaction)
            estimator, info = calibrate_seen(x, q[train], train, layouts, panel['tasks'], alphas=cfg['alphas'])
            predicted = estimator.predict(x)
            choices[key] = predicted.argmax(1)
            predictions[name+'__'+key+'__predicted_relative_value'] = predicted
            fit[key] = info
        choices_all.update({name+'__'+k:v for k,v in choices.items()})
        rows = {}
        for split, indices in [('seen', train), ('heldout', test)]:
            stay = q[indices][:,panel['extra']].mean(1)
            methods = {}
            for key,c in choices.items():
                value = q[indices,c[indices]]
                methods[key] = dict(success=float(value.mean()), gain=interval(value-stay,panel['tasks'][indices]),
                    view_ids=[panel['views'][v] for v in c[indices]],
                    same_view_as_min_mean_fraction=float(np.mean(c[indices] == choices['min_mean'][indices])),
                    execution_repeat_success=y[m,indices,c[indices]].mean(0).tolist())
            resampled_outcomes = q[indices[:, None],picks[indices]].mean(0)
            cross_noise = []
            for select_noise, evaluate_noise in [(slice(0,2),slice(2,4)),(slice(2,4),slice(0,2))]:
                select_q = y[m,indices,:,select_noise].mean(-1)
                evaluate_q = y[m,indices,:,evaluate_noise].mean(-1)
                winner = select_q.argmax(1)
                cross_noise.append(evaluate_q[np.arange(len(indices)),winner])
            rows[split] = dict(stay=float(stay.mean()), canonical=float(q[indices,0].mean()),
                empirical_oracle=float(q[indices].max(1).mean()), methods=methods,
                outcome_selected_two_noise_crosscheck=float(np.mean(cross_noise)),
                outcome_crosscheck_definition='Select using two execution repeats, evaluate the other two, average both directions; NOT a deployable selector or precise population Oracle',
                scoring_bootstrap_min_success=dict(mean=float(resampled_outcomes.mean()),
                    range95=np.quantile(resampled_outcomes,[.025,.975]).tolist()),
                stability=score_stability(repeats[m,indices], draws=cfg['bootstrap_draws'], seed=cfg['seed'])[1])
        result['models'][name] = dict(fit=fit, splits=rows, all_stability=stability)
    np.savez_compressed(output/'decisions.npz', **choices_all, **predictions)
    from ..plotting.static_view_decisions import plot
    plot(result, output)
    if any(sha(path) != value for path,value in sources.items()):
        raise ValueError('Input changed during analysis')
    result['source_snapshot'] = {str(p.relative_to(output)):sha(p) for p in snapshot.rglob('*.py')}
    result['artifacts'] = {p.name:sha(p) for p in output.iterdir() if p.suffix in ('.png','.npz')}
    write(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run(parser.parse_args().config)
