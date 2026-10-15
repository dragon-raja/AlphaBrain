"""CPU current-image field diagnostic; immutable outputs and frozen selections."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read, write, sha
from AlphaBrain.research.dsol.analysis.relative_accel import paired_label_audit, selection_metrics
from AlphaBrain.research.dsol.selectors.accel_field import fit_accel_field
from AlphaBrain.research.dsol.plotting.view_space import candidate_coordinates


def build(config_path):
    cfg = read(config_path)
    root = Path(cfg['audited_analysis'])
    provenance = read(root / 'provenance.json')
    names = ['matrix.npz', 'embeddings.npz', 'states.json']
    hashes = {name: sha(root / name) for name in names}
    if any(hashes[n] != provenance['artifacts'][n] for n in names):
        raise ValueError('Frozen input changed')
    with np.load(root / 'matrix.npz') as z:
        a = z['accel_members'][:, :, 1:].copy()
        ids = z['candidate_ids'][1:].copy()
    with np.load(root / 'embeddings.npz') as z:
        images = z['external'][:, 1:].copy()
    if a.shape != (2, 32, 96, 8) or images.shape != (32, 96, 512):
        raise ValueError('Unexpected archive shape')
    states = read(root / 'states.json')
    tasks = np.array([s['task_ordinal'] for s in states])
    initials = np.array([s['initialization']['init_state_index'] for s in states])
    release = read(cfg['release'])
    catalog = release['protocol']['state_blocks'][0]['candidates']
    if list(ids) != [c['selected_candidate_id'] for c in catalog[1:]]:
        raise ValueError('Candidate identities differ')
    coords = candidate_coordinates(catalog)
    starts = np.arange(*cfg['fit_starts'])
    tests = np.arange(*cfg['evaluation_starts'])
    out = Path(cfg['output']) / 'history' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(config=cfg, input_hashes=hashes, candidate_count=96,
        permission='Only current external frozen ResNet18 feature and current task-reference camera coordinates; no target image, task ID, canonical image, success input',
        geometric_privilege='Known task-reference coordinates and fixed aligned candidate catalog. Not unknown-reference deployment or continuous SE(3).',
        target='Per-initial centered/std-normalized field; training-noise half only. Four opposite noises evaluate the predicted selected view.',
        splits='initial_holdout: init0/1 -> init2/3; task_holdout: remaining tasks init0/1 -> heldout task init2/3',
        noise='Both 4->4 directions; paired within each initial. Eight samples cannot calibrate full tails.',
        fixed_hyperparameters='PCA 8, ridge alpha 10; no test-driven selection',
        field_scope='All 96 target poses may provide training-initial labels; heldout32 describes source views, not unseen target coordinates.',
        success_scope='Archived full-task endpoint outcomes read only after score-only selections saved; no new acquisition or rollout; no success-based model tuning',
        independent_units='Initials/tasks, not source images, view pairs or noises',
        caveats=['Same-task initial holdout is not new-scene generalization', 'Task holdout still shares benchmark/camera family',
                 'Pairwise uniform squared error equals centered squared error up to scale; it is not an extra independent loss',
                 'This small linear baseline is not a test of all learned-field architectures'])
    write(out / 'protocol.json', protocol)
    audit = [{**paired_label_audit(a[m, s]), 'model': ['canonical', 'broad'][m], 'initial': s}
             for m in range(2) for s in range(32)]
    write(out / 'noise-audit.json', audit)
    rows, decisions, parameters = [], [], {}
    folds = [('initial_holdout', 'all', np.flatnonzero(initials < 2), np.flatnonzero(initials >= 2))]
    folds += [('task_holdout', str(t), np.flatnonzero((tasks != t) & (initials < 2)),
               np.flatnonzero((tasks == t) & (initials >= 2))) for t in np.unique(tasks)]
    for split, fold, train, evaluate in folds:
        x = images[train][:, starts].reshape(-1, 512)
        q = np.tile(coords[starts], (len(train), 1))
        for m, name in enumerate(['canonical', 'broad']):
            for direction in range(2):
                halves = [a[m, ..., :4].mean(-1), a[m, ..., 4:].mean(-1)]
                source, target = halves[direction], halves[1-direction]
                target_train = np.repeat(source[train], len(starts), axis=0)
                normalized = source[train] - source[train].mean(-1, keepdims=True)
                template = (normalized / np.maximum(normalized.std(-1, keepdims=True), 1e-8)).mean(0)
                predictions = {'template': [np.tile(template, (len(tests), 1)) for s in evaluate]}
                for family, use_image in [('pose_only', False), ('current_rgb_pose', True)]:
                    model = fit_accel_field(x, q, target_train, use_image=use_image,
                                            image_dim=cfg['image_dim'], alpha=cfg['alpha'])
                    predictions[family] = [model.predict(images[s, tests], coords[tests]) for s in evaluate]
                    prefix = f'{split}_{fold}_{name}_{direction}_{family}'
                    for k, value in vars(model).items():
                        parameters[prefix + '_' + k] = np.asarray(value)
                for family, fields in predictions.items():
                    for s, pred in zip(evaluate, fields):
                        choices = pred.argmin(-1)
                        meta = dict(split=split, fold=fold, model=name, noise_direction=direction, method=family,
                                    initial=int(s), task=int(tasks[s]))
                        rows.append({**meta, **selection_metrics(pred, target[s])})
                        decisions.append({**meta, 'training_initials': train.tolist(), 'source_indices': tests.tolist(),
                                          'selected_noncanonical_indices': choices.tolist()})
        print(f'completed {split} {fold}', flush=True)
    np.savez_compressed(out / 'fitted-models.npz', **parameters)
    write(out / 'score-only-selections.json', decisions)
    frozen_selections = sha(out / 'score-only-selections.json')
    # Deliberately open success only after every method and selection is frozen.
    with np.load(root / 'matrix.npz') as z:
        success = z['success'].mean(-1)[:, :, 1:]
    for row, decision in zip(rows, decisions):
        m = ['canonical', 'broad'].index(row['model'])
        s = row['initial']
        values = success[m, s, decision['selected_noncanonical_indices']]
        row['archived_endpoint_success'] = float(values.mean())
        row['stay_success'] = float(success[m, s, tests].mean())
    if sha(out / 'score-only-selections.json') != frozen_selections:
        raise ValueError('Selections changed after success access')
    write(out / 'per-initial.json', rows)
    summary = dict(new_rollouts=0, vla_training_steps=0, noise={}, models=[])
    for name in ['canonical', 'broad']:
        selected = [r for r in audit if r['model'] == name]
        summary['noise'][name] = {k: float(np.median([r[k] for r in selected])) for k in
            ['paired_to_independent_variance_median', 'centered_to_raw_noise_variance']}
        for split in ['initial_holdout', 'task_holdout']:
            for method in ['template', 'pose_only', 'current_rgb_pose']:
                r = [v for v in rows if v['model'] == name and v['split'] == split and v['method'] == method]
                summary['models'].append(dict(model=name, split=split, method=method,
                    **{key: float(np.mean([v[key] for v in r])) for key in
                       ['low_quartile_hit', 'normalized_regret', 'archived_endpoint_success', 'stay_success']}))
    write(out / 'summary.json', summary)
    from AlphaBrain.research.dsol.plotting.relative_accel import summary_figure
    fig = summary_figure(summary)
    fig.savefig(out / 'summary.png', dpi=160)
    fig.savefig(out / 'summary.svg')
    import matplotlib.pyplot as plt
    plt.close(fig)
    sources = [Path(config_path), Path(__file__), Path('AlphaBrain/research/dsol/selectors/accel_field.py'),
               Path('AlphaBrain/research/dsol/analysis/relative_accel.py'), Path('AlphaBrain/research/dsol/plotting/relative_accel.py')]
    write(out / 'provenance.json', dict(input_hashes=hashes, selections_sha256=frozen_selections,
        sources={str(p.resolve()): sha(p) for p in sources},
        artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    if any(sha(root / n) != hashes[n] for n in names):
        raise ValueError('Input changed during run')
    write(Path(cfg['output']) / 'latest.json', dict(archive=str(out), summary=summary))
    print(str(out), flush=True)
    print(summary, flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    build(parser.parse_args().config)
