"""Audited cross-benchmark window analysis; no inference or outcome-based selection."""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from ..data.artifacts import read, sha, write


def select(scores, exclude=False):
    a = np.asarray(scores)
    if a.ndim != 3 or a.shape[1:] != (8, 5) or not np.isfinite(a).all():
        raise ValueError('Expected candidate x 8 scoring repeats x 5 windows')
    means = np.cumsum(a.mean(1), axis=-1) / np.arange(1, 6)
    return np.argmin(means[int(exclude):], axis=0) + int(exclude)


def rank_corr(a, b):
    a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None


def signal_diagnostics(scores, first, stride):
    chosen = select(scores)
    half = [np.cumsum(scores[:, part].mean(1), axis=-1)/np.arange(1, 6)
            for part in (slice(0, 4), slice(4, 8))]
    return dict(chosen=chosen.tolist(), canonical=(chosen == 0).tolist(),
                changed=(chosen != chosen[0]).tolist(),
                half_agreement=(half[0].argmin(0) == half[1].argmin(0)).tolist(),
                half_rank=[rank_corr(half[0][:, k], half[1][:, k]) for k in range(5)],
                post_success=[float(((first[chosen[k]] >= 0) &
                                    (first[chosen[k]] <= k*stride)).mean()) for k in range(5)])


def summarize(x, records, draws=4000):
    """Equal tasks, split-stratified initials, common execution resampling.

    Conditional on the measured candidate/scoring set. Descriptive intervals,
    not a new prespecified confirmation or a familywise best-window test.
    """
    x = np.asarray(x, float)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError('Do not report incomplete endpoint outcomes')
    tasks = np.array([r['task'] for r in records]); splits = np.array([r['split'] for r in records])
    unique = np.unique(tasks); rng = np.random.default_rng(20261013)
    means = []; boot_by_task = []
    for task in unique:
        split_means = []; split_boot = []
        for split in np.unique(splits[tasks == task]):
            ids = np.flatnonzero((tasks == task) & (splits == split))
            split_means.append(x[ids].mean((0, 2)))
            chosen = ids[rng.integers(len(ids), size=(draws, len(ids)))]
            noise = rng.integers(x.shape[-1], size=(*chosen.shape, x.shape[-1]))
            samples = np.take_along_axis(x[chosen].transpose(0, 1, 3, 2), noise[..., None], axis=2)
            split_boot.append(samples.mean((1, 2)))
        means.append(np.mean(split_means, axis=0)); boot_by_task.append(np.mean(split_boot, axis=0))
    means = np.array(means); all_boot = np.stack(boot_by_task, axis=1)
    task_draw = rng.integers(len(unique), size=(draws, len(unique)))
    boot = all_boot[np.arange(draws)[:, None], task_draw].mean(1)
    avg = means.mean(0)
    def ci(v): return np.quantile(v, [.025, .975], axis=0).T.tolist()
    return dict(n=len(x), execution_repeats=x.shape[-1], mean=avg.tolist(), ci95=ci(boot),
                gain=(avg[2:]-avg[0]).tolist(), gain_ci95=ci(boot[:, 2:]-boot[:, :1]),
                vs_w1=(avg[2:]-avg[2]).tolist(), vs_w1_ci95=ci(boot[:, 2:]-boot[:, 2:3]),
                per_task={str(t): v.tolist() for t, v in zip(unique, means)})


def panel(records, scores, outcomes, disturbances, first, stride):
    """Variable candidate counts are allowed (official Plus camera catalogs)."""
    arrays = {False: [], True: []}; diagnostics = []
    for r, s, y, d, f in zip(records, scores, outcomes, disturbances, first):
        diagnostics.append(signal_diagnostics(s, f, stride))
        for exclude in (False, True):
            choices = select(s, exclude)
            arrays[exclude].append(np.concatenate([y[d].mean(0)[None], y[:1], y[choices]], axis=0))
    result = dict(records=records, diagnostics=diagnostics, methods={})
    for exclude, values in arrays.items():
        x = np.asarray(values); coverage = np.isfinite(x[:, 2:]).all(-1)
        key = 'exclude' if exclude else 'include'
        result['methods'][key] = dict(coverage=coverage.sum(0).tolist(), scopes={})
        result['methods'][key]['complete_window_means'] = [
            float(x[:, k+2].mean()) if coverage[:, k].all() else None for k in range(5)]
        for scope in ['all', *sorted({r['split'] for r in records})]:
            ids = [i for i, r in enumerate(records) if scope == 'all' or r['split'] == scope]
            if np.isfinite(x[ids]).all():
                result['methods'][key]['scopes'][scope] = summarize(x[ids], [records[i] for i in ids])
        if np.isfinite(x).all():
            delta = x[:, 2:].mean(-1)-x[:, 2:3].mean(-1)
            result['methods'][key]['vs_w1_initials'] = dict(
                better=(delta > 1e-12).sum(0).tolist(), worse=(delta < -1e-12).sum(0).tolist(),
                equal=(abs(delta) <= 1e-12).sum(0).tolist())
    result['signal'] = {k: np.mean([d[k] for d in diagnostics], axis=0).tolist()
                        for k in ('canonical', 'changed', 'half_agreement', 'post_success')}
    result['signal']['half_rank'] = [float(np.mean([d['half_rank'][k] for d in diagnostics
                                                   if d['half_rank'][k] is not None])) for k in range(5)]
    return result


def run(config_file):
    cfg = read(config_file); out = Path(cfg['output'])
    if out.exists(): raise ValueError('Refuse to overwrite analysis')
    sources = {}
    def checked(path, digest=None):
        path = Path(path); h = sha(path)
        if digest is not None and h != digest: raise ValueError('Evidence changed: '+str(path))
        sources[str(path)] = h
        return path
    root = Path(cfg['libero_plus_root'])
    accepted = read(checked(root/'analysis/plus-windows-and-custom97.json'))
    if accepted['status'] != 'PASS': raise ValueError('Joint audit required')
    ownership = read(checked(root/'window-recovery-ownership.json'))
    records = read(checked(root/'inputs/libero-inputs.json'))['records']
    meta = [dict(initial=i, task=r['state']['task_id'],
                 split='development' if i % 4 < 2 else 'heldout_initial') for i, r in enumerate(records)]
    with np.load(checked(cfg['libero_matrix'], accepted['old_matrix_sha256'])) as z:
        yold = z['success'].copy(); old_accel = z['accel_members'].copy(); views = z['candidate_ids'].copy()
    extra = np.flatnonzero(np.char.startswith(views, 'broad_heldout_'))
    if len(extra) != 32: raise ValueError('Frozen heldout32 baseline required')
    panels = {}; bridge = {}
    production = read(checked(cfg['plus_production_config']))
    def probe(path):
        status = read(checked(path/'status.json'))
        if status['status'] != 'COMPLETE': raise ValueError('Incomplete probe')
        with np.load(checked(path/'probes.npz', status['arrays_sha256'])) as z:
            return z['scores'].copy(), z['first_success'].copy()
    for mi, model in enumerate(('canonical', 'broad')):
        custom_s = []; custom_f = []; native_s = []; native_f = []; native_y = []
        for i in range(32):
            receipt = next(r for r in accepted['receipts'] if r['model'] == model and r['initial'] == i)
            item = next(r for r in ownership['records'] if r['model'] == model and r['initial'] == i)
            custom = Path(item['output']); native = root/'plus-probes'/model/f'{i:02d}'
            full = root/'plus-evaluation'/model/f'{i:02d}'
            for p, key in ((custom, 'custom'), (native, 'native'), (full, 'full')):
                checked(p/'status.json', receipt[key])
            s, f = probe(custom)
            if not np.array_equal(s[:, :, 0], old_accel[mi, i]): raise ValueError('Custom W1 changed')
            custom_s.append(s); custom_f.append(f)
            s, f = probe(native); native_s.append(s); native_f.append(f)
            y = np.full((len(s), 4), np.nan); y[0] = yold[mi, i, 0, :4]
            status = read(full/'status.json')
            with np.load(checked(full/'probes.npz', status['arrays_sha256'])) as z:
                if not np.array_equal(s[:, :, 0], z['scores']): raise ValueError('Plus W1 changed')
                y[z['execution_view_indices']] = z['success']
            if i == 0:
                p = Path(production['admission_root'])/model/'00'; status = read(checked(p/'status.json'))
                with np.load(checked(p/'probes.npz', status['arrays_sha256'])) as z:
                    for j, v in enumerate(z['execution_view_indices']):
                        if v in production['reused_rollout_indices']['0']: y[v] = z['success'][j]
            if not np.isfinite(y).all(): raise ValueError('Missing Plus outcomes')
            native_y.append(y)
        panels['libero_'+model] = panel(meta, custom_s, yold[mi], [extra]*32, custom_f, 5)
        panels['plus_'+model] = panel(meta, native_s, native_y,
                                     [np.arange(1, len(y)) for y in native_y], native_f, 5)
        # Same Plus initial distribution and four execution repeats at both endpoints.
        bridge[model] = {}
        for exclude in (False, True):
            x = np.array([np.concatenate([native_y[i][1:].mean(0)[None], yold[mi, i, :1, :4],
                                          yold[mi, i, select(custom_s[i], exclude), :4]], axis=0) for i in range(32)])
            bridge[model]['exclude' if exclude else 'include'] = summarize(x, meta)
        # Golden comparison to the admitted aggregation, independently reconstructed.
        for mode in ('include', 'exclude'):
            key = 'with_canonical' if mode == 'include' else 'without_canonical'
            if not np.allclose(panels['plus_'+model]['methods'][mode]['scopes']['all']['mean'][2:],
                               accepted['models'][model]['overall']['native_'+key], rtol=0, atol=1e-15):
                raise ValueError('Native statistics differ from accepted aggregation')
    robo = Path(cfg['robocasa_accepted']); audit = read(checked(robo/'audit.json'))
    summary = read(checked(robo/'summary.json')); metric = read(checked(robo/'metric-audit.json'))
    if audit['status'] != 'PASS' or metric['status'] != 'PASS': raise ValueError('RoboCasa audit required')
    pop = read(checked(robo/'population.json', audit['population_sha256']))
    with np.load(checked(robo/'accepted-arrays.npz', audit['arrays_sha256'])) as z:
        extra = [z['views'].tolist().index(v) for v in z['extra_views']]
        for model in ('official', 'xr1'):
            s = z[model+'__scores']; y = z[model+'__success']; f = z[model+'__first_success']
            if not np.array_equal(np.array([select(v) for v in s]), z[model+'__choices']):
                raise ValueError('RoboCasa frozen choices changed')
            panels['robocasa_'+model] = panel(pop[model], s, y, [extra]*80, f, 25 if model == 'official' else 16)
            orig = summary['models'][model]['populations']['all']['methods']
            vals = [orig[k]['success']['mean'] for k in ['disturbance_uniform', 'canonical', *[f'window{k}' for k in range(1, 6)]]]
            if not np.allclose(panels['robocasa_'+model]['methods']['include']['scopes']['all']['mean'], vals, atol=1e-15, rtol=0):
                raise ValueError('RoboCasa published numbers changed')
            # Keep the already-published confirmation intervals; harmonizing
            # presentation is not permission to replace its inference protocol.
            for scope, original_scope in [('all','all'), ('pretrain','seen'), ('target','heldout')]:
                target = panels['robocasa_'+model]['methods']['include']['scopes'][scope]
                frozen = summary['models'][model]['populations'][original_scope]['methods']
                names = ['disturbance_uniform','canonical',*[f'window{k}' for k in range(1,6)]]
                if not np.allclose(target['mean'],[frozen[n]['success']['mean'] for n in names],rtol=0,atol=1e-15):
                    raise ValueError('RoboCasa split means changed')
                target['ci95'] = [frozen[n]['success']['ci95'] for n in names]
                target['gain_ci95'] = [frozen[n]['gain_vs_stay']['ci95'] for n in names[2:]]
                target['vs_w1_ci95'] = [frozen[n]['gain_vs_window1']['ci95'] for n in names[2:]]
                target['interval_source'] = 'Original accepted confirmation summary, unchanged'
    result = dict(status='PASS', panels=panels, bridge=bridge, robocasa_original=summary,
                  sources=sources, config_sha256=sha(config_file), source_sha256=sha(__file__),
                  conditions=['disturbance', 'canonical', 'W1', 'W2', 'W3', 'W4', 'W5'],
                  uncertainty='LIBERO/Plus:4000 paired task / split-stratified initial / execution bootstrap draws, descriptive only, frozen scoring choices fixed. RoboCasa include-canonical uses unchanged original10000-draw confirmation intervals; exclusion only where complete, descriptive. No training seed uncertainty; no best-window test.',
                  scope='No new rollouts; no online camera movement. LIBERO/Plus same 8 known tasks,32 previously inspected initials. RoboCasa 80 fresh initials/model of10 known tasks. Exclusion statistics omitted when any selected endpoint missing.')
    write(out/'summary.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    print(run(p.parse_args().config)['status'])
