"""Retrospective early-rollout view scores with disjoint-noise evaluation.

These scores require executing candidate branches. They are not available to
an initial-state selector and are not evidence of physical camera movement.
"""
import argparse
from itertools import combinations
from pathlib import Path
import shutil

import numpy as np

from ..data.artifacts import read, sha, write
from .robocasa_statistics import interval
from .view_landscape import association, load_panel
from .xr1_fullview import seen_fixed_choices


def audit_saved_prefixes(analysis_root, manifest_path, output):
    """Check first/last manifest view of every initial against raw executions.

    This is an explicit stratified audit, not a claim to re-audit all rollouts.
    It writes a separate receipt without changing the analysis publication.
    """
    analysis_root, manifest_path, output = map(Path, (analysis_root, manifest_path, output))
    with np.load(analysis_root/'panel_arrays.npz', allow_pickle=False) as z:
        initials, views = z['initials'].tolist(), z['views'].tolist()
        y = z['success'][z['models'].tolist().index('xr1')]
        temporal = z['rollout_early_accel'].copy()
    records = read(manifest_path)['records']; files = {}; cells = 0
    for i,initial in enumerate(initials):
        rows = sorted([r for r in records if r['initial']==initial], key=lambda r:views.index(r['view']))
        if len(rows)<2: raise ValueError('Every initial must have at least two manifest cells')
        for row in (rows[0],rows[-1]):
            v=views.index(row['view']); cells+=1
            if len(row['executions'])!=4: raise ValueError('Expected four raw executions')
            for repeat,path in enumerate(row['executions']):
                raw=read(path); plans=raw['plans'][:5]
                if raw['repeat']!=repeat or [p['step'] for p in plans]!=[0,16,32,48,64]:
                    raise ValueError('Execution identity or native replan addresses differ')
                expected=[np.mean([p['accel_p3'] for p in plans[:k]]) for k in (1,3,5)]
                if not np.array_equal(temporal[i,v,repeat],expected) or bool(y[i,v,repeat])!=raw['success']:
                    raise ValueError('Saved prefix or outcome differs from raw execution')
                files[str(path)]=sha(path)
    receipt=dict(status='PASS', initials=len(initials), cells=cells, executions=len(files),
        selection='First and last view in explicit source manifest for every initial',
        sources={str(manifest_path):sha(manifest_path),str(analysis_root/'panel_arrays.npz'):sha(analysis_root/'panel_arrays.npz')},
        raw_files=files, source_sha256=sha(__file__),
        scope='Stratified raw-record audit; not a fresh rollout or full raw-file audit')
    if output.exists(): raise FileExistsError(output)
    write(output,receipt)
    return receipt


def cross_noise_choices(scores, outcomes):
    """All six 2/2 splits; one view per initial and split, never per outcome.

    Scores have axes initial, view, execution repeat. Evaluation outcomes have
    the same axes. A split's two scoring repeats never evaluate its choice.
    Splits overlap: average within initial before any uncertainty calculation.
    """
    scores, outcomes = np.asarray(scores, float), np.asarray(outcomes)
    if scores.ndim != 3 or scores.shape != outcomes.shape or scores.shape[-1] != 4:
        raise ValueError('Requires matching initial x view x four-repeat arrays')
    if not np.isfinite(scores).all() or not np.isin(outcomes, [0, 1]).all():
        raise ValueError('Nonfinite score or nonbinary outcome')
    choices, values, associations, splits = [], [], [], []
    for train in combinations(range(4), 2):
        test = tuple(i for i in range(4) if i not in train)
        a = scores[..., list(train)].mean(-1)
        q = outcomes[..., list(test)].mean(-1)
        choice = a.argmin(1)
        choices.append(choice)
        values.append(q[np.arange(len(q)), choice])
        associations.append(association(a[:, 1:], q[:, 1:]))
        splits.append(dict(score_repeats=list(train), evaluate_repeats=list(test)))
    return dict(choices=np.asarray(choices), values=np.asarray(values),
                per_initial=np.mean(values, axis=0), associations=associations, splits=splits)


def run(config_path):
    cfg = read(config_path)
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=False)
    source = Path(cfg['analysis'])/'panel_arrays.npz'
    panel = load_panel(cfg['panel'])
    with np.load(source, allow_pickle=False) as z:
        if z['initials'].tolist() != panel['initials'] or z['views'].tolist() != panel['views']:
            raise ValueError('Initial/view identities differ')
        m = z['models'].tolist().index('xr1')
        y, a, temporal = z['success'][m], z['accel'][m], z['rollout_early_accel'].copy()
    if y.shape != (40, 97, 4) or temporal.shape != (40, 97, 4, 3):
        raise ValueError('Unexpected frozen panel shape')
    if not np.isfinite(temporal).all():
        raise ValueError('Missing early prefixes: do not silently change population')
    paths = [source, Path(cfg['panel'])/'selection.json', Path(config_path)]
    sources = {str(p): sha(p) for p in paths}
    write(output/'protocol.json', dict(config=cfg, sources=sources,
        status='EXPLORATORY_REUSE_NOT_FRESH_CONFIRMATION',
        candidate_count=97, execution_repeats=4, prefixes=[1, 3, 5],
        action_step_at_last_score=[0, 32, 64], native_replan_steps=16,
        folds='All six disjoint 2-score/2-outcome splits; average within initial',
        score_access='Executed all candidate branches; not an initial deployable selector',
        selection='Minimum mean score, one view per initial per split; no outcome-based choice',
        new_rollouts=0, physical_camera_motion=False,
        uncertainty='Task/initial bootstrap, conditional on saved four execution repeats'))
    q = y.mean(-1)
    fixed = seen_fixed_choices(q, panel['tasks'], panel['splits'])
    initial_choice = a.argmin(1)
    baseline = dict(stay=q[:, panel['extra']].mean(1), canonical=q[:, 0],
        initial_accel8=q[np.arange(len(q)), initial_choice], seen_fixed=q[:, fixed['global_view']])
    results, arrays = {}, dict(**baseline, initials=np.asarray(panel['initials']))
    for k, prefix in enumerate((1, 3, 5)):
        cross = cross_noise_choices(temporal[..., k], y)
        # Retrospective only: selection on a trajectory's score, same trajectory's
        # outcome. Deliberately separate from disjoint-repeat diagnostic.
        same_choice = temporal[..., k].argmin(1)
        same = y[np.arange(len(y))[:, None], same_choice, np.arange(4)[None, :]].mean(1)
        arrays[f'prefix{prefix}_cross_choices'] = cross['choices']
        arrays[f'prefix{prefix}_cross_values'] = cross['values']
        arrays[f'prefix{prefix}_same_values'] = same
        results[str(prefix)] = dict(splits=cross['splits'], populations={})
        for name, mask in [('all', np.ones(len(y), bool)), ('seen', panel['splits']=='pretrain'),
                           ('heldout', panel['splits']=='target')]:
            rows = np.flatnonzero(mask); tasks = panel['tasks'][mask]
            value = cross['per_initial'][mask]
            assocs = []
            for split in cross['splits']:
                aa = temporal[mask, :, :, k][..., split['score_repeats']].mean(-1)
                qq = y[mask][..., split['evaluate_repeats']].mean(-1)
                assocs.append(association(aa[:, 1:], qq[:, 1:]))
            correlations = [r['pooled_within_initial_rank_correlation'] for r in assocs]
            valid = [r for r in correlations if r is not None]
            results[str(prefix)]['populations'][name] = dict(
                initials=len(rows), success=float(value.mean()),
                same_trajectory_retrospective_success=float(same[mask].mean()),
                baseline={key: float(v[mask].mean()) for key,v in baseline.items()},
                gain_vs_stay=interval(value-baseline['stay'][mask], tasks),
                gain_vs_initial_accel8=interval(value-baseline['initial_accel8'][mask], tasks),
                gain_vs_prefix1=interval(value-cross_noise_choices(temporal[mask, :, :, 0], y[mask])['per_initial'], tasks),
                fold_association=assocs,
                mean_fold_within_initial_rank_correlation=float(np.mean(valid)) if valid else None,
                per_task=[dict(task=str(t), success=float(value[tasks==t].mean()),
                              stay=float(baseline['stay'][mask][tasks==t].mean()),
                              initial_accel8=float(baseline['initial_accel8'][mask][tasks==t].mean()))
                          for t in np.unique(tasks)])
    snapshot = output/'source'; snapshot.mkdir()
    package = Path(__file__).resolve().parents[1]
    for relative in ('analysis/temporal_view_diagnostic.py', 'analysis/robocasa_statistics.py',
                     'analysis/view_landscape.py', 'analysis/xr1_fullview.py',
                     'plotting/temporal_view_diagnostic.py', 'data/artifacts.py'):
        target = snapshot/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package/relative, target)
    result = dict(status='COMPLETE_RETROSPECTIVE_DIAGNOSTIC', model='xr1', results=results,
        sources=sources, source_snapshot={str(p.relative_to(output)): sha(p) for p in snapshot.rglob('*.py')},
        new_rollouts=0, limitations=[
            'Prefixes use actual candidate-specific robot trajectories, not held-fixed-state repeated inference',
            'Cross-noise reuse is not new independent confirmation; six folds are not independent observations',
            'Early trajectory scores differ in information access and draw count from initial eight-draw scores',
            'Four repeats and twenty heldout initials limit precision; all ten tasks retained',
            'No target-based choice of prefix; all three prefixes reported'])
    np.savez_compressed(output/'decisions.npz', **arrays)
    from ..plotting.temporal_view_diagnostic import plot
    plot(result, output)
    if any(sha(p) != expected for p,expected in sources.items()):
        raise ValueError('Inputs changed during diagnostic')
    result['artifacts'] = {p.name: sha(p) for p in (output/'decisions.npz', output/'temporal_diagnostic_zh.png')}
    write(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(parser.parse_args().config)
