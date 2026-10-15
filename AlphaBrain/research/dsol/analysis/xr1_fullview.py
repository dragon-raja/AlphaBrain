"""Matched XR-1/π0.5 97-view analysis; consume completed immutable cells only."""
import argparse
from pathlib import Path

import numpy as np

from ..data.artifacts import read, sha, write
from .robocasa_statistics import empirical_hierarchy, interval
from .view_landscape import load_panel, model_statistics, association


def seen_fixed_choices(q, tasks, splits):
    """Simple calibration baselines; heldout labels never choose a view."""
    q, tasks, splits = np.asarray(q), np.asarray(tasks), np.asarray(splits)
    if q.ndim != 2 or tasks.shape != (len(q),) or splits.shape != (len(q),):
        raise ValueError('Expected initial x view outcomes with task/split labels')
    if not np.isfinite(q).all() or not np.isin(splits, ['pretrain', 'target']).all():
        raise ValueError('Invalid outcome values or split labels')
    train = splits == 'pretrain'
    test = splits == 'target'
    if not train.any() or not test.any() or not set(tasks[test]) <= set(tasks[train]):
        raise ValueError('Requires seen calibration and heldout initials of known tasks')
    global_view = int(q[train].mean(0).argmax())
    by_task = {str(t):int(q[train & (tasks == t)].mean(0).argmax()) for t in np.unique(tasks[train])}
    return dict(global_view=global_view, task_views=by_task,
                per_initial_task_choices=np.array([by_task[str(t)] for t in tasks]))


def run_seen_rule_diagnostic(analysis_root, panel_root, output):
    """Exploratory known-task calibration diagnostic; no additional rollouts."""
    analysis_root, output = Path(analysis_root), Path(output)
    panel = load_panel(panel_root)
    with np.load(analysis_root/'panel_arrays.npz', allow_pickle=False) as data:
        if data['initials'].tolist() != panel['initials'] or data['views'].tolist() != panel['views']:
            raise ValueError('Mismatched panel identities')
        outcomes, models = data['success'].copy(), data['models'].tolist()
    test = panel['splits'] == 'target'
    result = dict(status='COMPLETE', models={},
                  scope='Exploratory seen-calibrated fixed-view baselines; same known10 tasks, heldout20 initials; no new rollouts',
                  limitation='Target outcomes have already been inspected; not a fresh blind confirmation or physical camera search',
                  sources={str(analysis_root/'panel_arrays.npz'):sha(analysis_root/'panel_arrays.npz'),
                           str(Path(panel_root)/'selection.json'):sha(Path(panel_root)/'selection.json'),
                           str(Path(__file__).resolve()):sha(__file__)})
    for m,name in enumerate(models):
        q = outcomes[m].mean(-1)
        choices = seen_fixed_choices(q, panel['tasks'], panel['splits'])
        stay = q[test][:,panel['extra']].mean(1)
        global_q = q[test,choices['global_view']]
        task_q = q[np.flatnonzero(test),choices['per_initial_task_choices'][test]]
        result['models'][name] = dict(stay=float(stay.mean()), canonical=float(q[test,0].mean()),
            global_view=panel['views'][choices['global_view']],
            task_views={t:panel['views'][v] for t,v in choices['task_views'].items()},
            global_fixed=float(global_q.mean()), task_fixed=float(task_q.mean()),
            global_gain=interval(global_q-stay,panel['tasks'][test]),
            task_gain=interval(task_q-stay,panel['tasks'][test]))
    output.mkdir(parents=True,exist_ok=False)
    import shutil
    shutil.copy2(Path(__file__).resolve(), output/'analysis-source.py')
    result['source_snapshot']={str(output/'analysis-source.py'):sha(output/'analysis-source.py')}
    write(output/'summary.json',result)
    return result


def render_completed(analysis_root, panel_root, output):
    """Finish plotting in a separate CPU environment, without re-estimation.

    Preserve the failed original analysis directory and consume its admitted
    arrays/statistics verbatim. The completion receipt is a new provenance link.
    """
    analysis_root, output = Path(analysis_root), Path(output)
    result = read(analysis_root / 'summary.json')
    for path, expected in result['sources'].items():
        if sha(path) != expected:
            raise ValueError('Analysis source changed: ' + path)
    panel = load_panel(panel_root)
    with np.load(analysis_root / 'panel_arrays.npz', allow_pickle=False) as data:
        if data['initials'].tolist() != panel['initials'] or data['views'].tolist() != panel['views']:
            raise ValueError('Completed analysis axes differ from the original panel')
        models = data['models'].tolist()
        y, a = data['success'].copy(), data['accel'].copy()
        if models != [*panel['models'], 'xr1'] or y.shape != (4, 40, 97, 4) or a.shape != (4, 40, 97):
            raise ValueError('Expected complete matched four-model panel')
        if not np.isin(y, [0, 1]).all() or not np.isfinite(a).all():
            raise ValueError('Invalid completed array values')
        if not np.array_equal(y[:3], panel['y']) or not np.array_equal(a[:3], panel['accel']):
            raise ValueError('Reused π0.5 outcomes/scores changed')
    sources = {str(analysis_root / name): sha(analysis_root / name)
               for name in ('summary.json', 'panel_arrays.npz')}
    panel.update(y=y, accel=a, models=models)
    output.mkdir(parents=True, exist_ok=False)
    from ..plotting.view_landscape import plot
    plot(panel, result, output)
    if any(sha(path) != expected for path, expected in sources.items()):
        raise ValueError('Input changed during rendering')
    figures = {p.name: sha(p) for p in output.glob('*.png')}
    if len(figures) != 4:
        raise ValueError('Expected three split landscapes and a regional diagnostic')
    import shutil
    frozen = output / 'renderer-source'
    frozen.mkdir()
    for source in (Path(__file__).resolve(), Path(__file__).parents[1] / 'plotting/view_landscape.py'):
        shutil.copy2(source, frozen / source.name)
    write(output / 'completion.json', dict(status='COMPLETE', sources=sources,
          source_artifacts=result['sources'], figures=figures,
          renderer={str(Path(__file__).resolve()): sha(__file__),
                    str(Path(__file__).parents[1] / 'plotting/view_landscape.py'):
                    sha(Path(__file__).parents[1] / 'plotting/view_landscape.py')},
          renderer_snapshot={str(p):sha(p) for p in frozen.iterdir()},
          xr1_rollouts=15520, new_rollouts=0, statistics_recomputed=False,
          original_failure_preserved=True))
    return output


def assemble(panel_root, roots, output, pi05_subspaces=None, cell_manifest=None):
    panel_root, output = Path(panel_root), Path(output)
    old = load_panel(panel_root)
    initials, views = old['initials'], old['views']
    y = np.full((40, 97, 4), np.nan)
    acc = np.full((40, 97, 8), np.nan)
    time_acc = np.full((40, 97, 4, 3), np.nan)
    files = {}
    identities = []
    cells = {}
    for root in map(Path, roots):
        plan = read(root / 'plan.json')
        identities.append(plan)
        files[str(root / 'plan.json')] = sha(root / 'plan.json')
        for worker in read(root / 'launch.json')['workers']:
            location = Path(worker['output'])
            if read(location / 'status.json')['status'] != 'COMPLETE' and cell_manifest is None:
                raise ValueError('Unfinished phase cannot enter full analysis')
            identity = read(location / 'identity.json')
            for initial in identity['initials']:
                for view in identity['views']:
                    if (initial, view) in cells:
                        raise ValueError('Duplicate source cell')
                    cells[initial, view] = location / 'cells' / initial / view
    overrides = {}
    if cell_manifest is not None:
        files[str(cell_manifest)] = sha(cell_manifest)
        for row in read(cell_manifest)['records']:
            key = row['initial'], row['view']
            if key in overrides or key not in cells or len(row['executions']) != 4:
                raise ValueError('Invalid explicit source override')
            overrides[key] = row
    for plan in identities:
        for key in ('initial_ids', 'checkpoint_receipt_sha256', 'panel_sha256', 'noise_version',
                    'native_replan_steps', 'native_flow_steps', 'native_observation_history'):
            if plan[key] != identities[0][key]:
                raise ValueError(f'Cross-phase protocol mismatch: {key}')
    if set(cells) != {(i, v) for i in initials for v in views}:
        raise ValueError('The complete 40 x 97 matrix is required')
    for i, initial in enumerate(initials):
        horizon = None
        for v, view in enumerate(views):
            # Fingerprint contains camera extrinsics: only compare repeats
            # of the SAME view. Runtime checks the original non-camera state.
            physical_hash = None
            cell = cells[initial, view]
            override = overrides.get((initial, view))
            score_path = Path(override['scores']) if override else cell / 'scores.json'
            scores = read(score_path)
            if cell_manifest is not None and sha(score_path.with_name('scores.npz')) != scores['arrays_sha256']:
                raise ValueError('Scoring source hash mismatch')
            acc[i, v] = [r['accel_p3'] for r in scores['rows']]
            for r in range(4):
                execution_path = Path(override['executions'][r]) if override else cell / f'execution_{r}.json'
                if cell_manifest is not None:
                    from ..data.xr1_rebalance import validate_trial
                    row = validate_trial(execution_path, initial, r)
                else:
                    row = read(execution_path)
                if physical_hash is None:
                    physical_hash = row['initial_physical_sha256']
                if horizon is None:
                    horizon = row['horizon']
                if row['initial_physical_sha256'] != physical_hash or row['horizon'] != horizon:
                    raise ValueError('Same-view start hash or cross-view horizon differs')
                y[i, v, r] = row['success']
                for k, prefix in enumerate((1, 3, 5)):
                    records = row['plans'][:prefix]
                    # Missing early replans remain missing; never impute success as low Accel.
                    if len(records) == prefix:
                        time_acc[i, v, r, k] = np.mean([p['accel_p3'] for p in records])
    if not np.isfinite(acc).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid complete-matrix values')
    output.mkdir(parents=True, exist_ok=False)
    combined = dict(old, y=np.concatenate([old['y'], y[None]]),
                    accel=np.concatenate([old['accel'], acc.mean(-1)[None]]),
                    models=[*old['models'], 'xr1'])
    np.savez_compressed(output / 'panel_arrays.npz', success=combined['y'], accel=combined['accel'],
                        score_repeats=acc, rollout_early_accel=time_acc,
                        models=np.array(combined['models']), initials=np.array(initials), views=np.array(views))
    result = dict(splits={}, sources=files,
                  scope='Matched 40 initials x 97 views x 4 closed-loop outcomes per model',
                  xr1_metric='Prefix3 normalized flow Accel over native executed16 x active12; mean of8 independent score draws',
                  pi05_metric='Previously registered prefix3 over50 x32; native backend differs. Compare ranks/recovery, not raw scales.',
                  selector='Per-initial minimum eight-score mean over97, fixed installation throughout rollout',
                  oracle='Empirical maximum AFTER four-noise averaging per initial; optimistic finite-sample descriptive ceiling, no per-noise winner',
                  camera='Simulation installation candidate; not a certified physical action or online camera search',
                  temporal='First1/3/5 replan Accel is retained for retrospective diagnosis only; no success-conditioned imputation or deployed selector claim')
    for split in ('all', 'pretrain', 'target'):
        keep = np.ones(40, bool) if split == 'all' else old['splits'] == split
        result['splits'][split] = {}
        for m, model in enumerate(combined['models']):
            local_y, local_a = combined['y'][m, keep], combined['accel'][m, keep]
            stats = model_statistics(local_y, local_a, old['tasks'][keep], old['extra'])
            stats['empirical_hierarchy'] = empirical_hierarchy(local_y.mean(-1), old['tasks'][keep])
            result['splits'][split][model] = stats
        xrq, piq = y[keep].mean(-1), old['y'][0, keep].mean(-1)
        result['splits'][split]['xr1_vs_pi05_paired'] = {
            'canonical_gain': interval(xrq[:, 0]-piq[:, 0], old['tasks'][keep]),
            'stay_extra32_gain': interval(xrq[:, old['extra']].mean(1)-piq[:, old['extra']].mean(1), old['tasks'][keep])}
    if pi05_subspaces is not None:
        with np.load(pi05_subspaces, allow_pickle=False) as data:
            if data['initials'].tolist() != initials or data['views'].tolist() != views:
                raise ValueError('π0.5 subspace axes differ from matched panel')
            variants = data['variants'].tolist()
            active = data['scores'][:, :, :, variants.index('active12_h25')].mean(-1)
        result['sources'][str(pi05_subspaces)] = sha(pi05_subspaces)
        result['active_executed_window_sensitivity'] = {}
        for split in ('all', 'pretrain', 'target'):
            keep = np.ones(40, bool) if split == 'all' else old['splits'] == split
            result['active_executed_window_sensitivity'][split] = {
                name: model_statistics(old['y'][m, keep], active[m, keep], old['tasks'][keep], old['extra'])
                for m, name in enumerate(old['models'])}
        result['active_executed_window_note'] = 'π0.5 active25x12 vs XR-1 active16x12; each native executed window, not identical horizons/integration grids'
    write(output / 'summary.json', result)
    from ..plotting.view_landscape import plot
    plot(combined, result, output)
    write(output / 'status.json', dict(status='COMPLETE', initials=40, views=97,
                                     xr1_rollouts=15520, new_pi05_rollouts=0))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--roots', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pi05-subspaces', type=Path)
    parser.add_argument('--cell-manifest', type=Path)
    args = parser.parse_args()
    assemble(args.panel, args.roots, args.output, args.pi05_subspaces, args.cell_manifest)
