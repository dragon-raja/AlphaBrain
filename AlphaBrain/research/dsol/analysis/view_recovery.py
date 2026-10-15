"""Run an explicit, CPU-only disturbed-view recovery study on frozen data."""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shutil
import numpy as np

from ..acquisition.discrete_search import camera_graph, search
from ..selectors.query_ranker import development_fit
from ..metrics.noise_rules import noise_rule_choices
from ..data.artifacts import read, write, sha
from .statistics import paired_interval


def outcome_summary(values, baseline, translation, rotation, scope):
    """Average start poses/replay seeds first; never count them as new initials."""
    v, b = values[:, scope], baseline[:, scope]
    delta = v - b
    ix = np.arange(len(v))
    return dict(success=paired_interval(v.mean(1), ix),
                gain_vs_stay=paired_interval(delta.mean(1), ix),
                improved_fraction=float(np.mean(delta > 1e-12)),
                harmed_fraction=float(np.mean(delta < -1e-12)),
                equal_fraction=float(np.mean(abs(delta) <= 1e-12)),
                mean_translation_m=float(translation[:, scope].mean()),
                mean_rotation_deg=float(rotation[:, scope].mean()),
                per_initial_success=v.mean(1).tolist(), per_task_success=v.mean(1).reshape(8, 2).mean(1).tolist())


def metric_reference_summary(success, accel, visibility, evaluation_indices):
    """Full-query metric-only reference from the same frozen endpoint matrix.

    Success is already averaged over execution noise. No fitting, label-based
    selection, search replay or VLA inference is performed here.
    """
    ix = np.asarray(evaluation_indices)
    if len(ix) != 16:
        raise ValueError('Expected the task-major 16 evaluation initials')
    result = {'models': {}, 'definition': {
        'single_accel': 'Each of 8 scoring seeds independently chooses min Accel; average the 8 chosen endpoint outcomes; no seed selection by success',
        'min_accel': 'Average 8 scores per candidate first, then select one candidate per initial',
        'visibility': 'External-camera simulator visibility, privileged segmentation reference',
        'accel10_visibility': 'Lowest 10 by mean Accel, then highest visibility within that set',
        'candidate_count': 97, 'evaluation_initial_count': 16, 'main_start_distribution': 'uniform heldout catalog 32'}}
    scopes = {'noncanonical96': slice(None), 'heldout32': slice(64, 96)}
    for m, name in enumerate(['canonical', 'broad']):
        q = np.asarray(success)[m, ix]
        decisions = noise_rule_choices(np.asarray(accel)[m, ix], np.asarray(visibility)[ix])
        values = {'stay': q[:, 1:], 'known_canonical': np.broadcast_to(q[:, :1], (16, 96))}
        for method, choices in decisions.items():
            picked = q[np.arange(16)[:, None], choices] if choices.ndim == 2 else q[np.arange(16), choices, None]
            values[method] = np.broadcast_to(picked.mean(axis=1, keepdims=True), (16, 96))
        rows = {}
        for method, value in values.items():
            rows[method] = {scope: outcome_summary(value, q[:, 1:], np.zeros_like(value), np.zeros_like(value), subset)
                            for scope, subset in scopes.items()}
            for row in rows[method].values():
                row['mean_translation_m'] = None
                row['mean_rotation_deg'] = None
        result['models'][name] = rows
    return result


def run(config_path):
    config_path = Path(config_path).resolve()
    cfg = read(config_path)
    source, output = Path(cfg['audited_analysis']), Path(cfg['output'])
    provenance = read(source / 'provenance.json')
    inputs = ['matrix.npz', 'embeddings.npz', 'states.json', 'selections.json']
    for name in inputs:
        if sha(source / name) != provenance['artifacts'][name]:
            raise ValueError('Audited input changed: ' + name)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = output / 'history' / stamp
    out.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parents[1]
    source_hashes = {}
    for relative in ['analysis/view_recovery.py', 'analysis/statistics.py', 'selectors/query_ranker.py',
                     'acquisition/discrete_search.py', 'data/artifacts.py']:
        p = package / relative
        dest = out / 'source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        source_hashes[relative] = sha(p)
    protocol = dict(
        config=cfg, config_sha256=sha(config_path), frozen_before_fit_utc=stamp,
        input_hashes={name: sha(source / name) for name in inputs}, source_hashes=source_hashes,
        status='Exploratory: earlier test aggregates inspected; no blind confirmation claim',
        population='8 known tasks; initial 0/1 development, 2/3 evaluation; same 32 execution noises',
        starts='All 96 noncanonical starts, equally weighted; heldout catalog 32 also reported separately',
        graph='Symmetric 6-NN relative SE(3) distance; no canonical target or candidate ID feature; no physical reachability claim',
        query='One distinct visited node includes its external RGB; Accel policies also use eight score noises; start counts as one query',
        visual_features='Frozen ResNet18 external 512D; wrist extracted only from context[512:]; canonical external context excluded',
        families=['queried_rgb_wrist_task', 'queried_rgb_wrist_task_accel_mean_std'],
        navigation_permission='Known relative candidate pose catalog for navigation only; no privileged canonical-pose command in search',
        baselines='Stay; same-budget random-frontier best-queried-score; full query; known-calibration return canonical separately',
        learned_fit='Development-only PCA/standardization/alpha CV; within-initial centered target and features; known task IDs, not unseen task language generalization',
        costs='Query counts plus sum of translation and rotation along catalog graph including return to selected node; not physical time or collision-free travel',
        caveats=['No simulator or VLA training; cached score/image features and measured endpoint outcomes only',
                 'Scene and robot unchanged while acquiring; routes not physically validated',
                 'Endpoint evaluation uses same finite rollout bank, not new independent confirmation',
                 'Search seed repeats/start poses are averaged within each initial for bootstrap',
                 'Visibility segmentation and canonical-image features excluded from deployable methods'])
    write(out / 'protocol.json', protocol)
    with np.load(source / 'matrix.npz') as z:
        q, acc, camera = z['success'].mean(-1), z['accel_members'], z['camera']
        ids = z['candidate_ids']
    with np.load(source / 'embeddings.npz') as z:
        images, context = z['external'], z['context']
    if images.shape != (32, 97, 512) or context.shape != (32, 1024):
        raise ValueError('Unexpected audited feature schema')
    if ids[0] != 'canonical' or not all(str(v).startswith('broad_heldout_') for v in ids[65:]):
        raise ValueError('Candidate partition mismatch')
    wrists = context[:, 512:].copy()  # Never pass the canonical-image half to a learner.
    states = read(source / 'states.json')
    initials = np.array([s['initialization']['init_state_index'] for s in states])
    tasks = np.array([s['task_ordinal'] for s in states])
    dev, test = np.flatnonzero(initials < 2), np.flatnonzero(initials >= 2)
    if len(test) != 16 or not np.array_equal(tasks[test], np.repeat(np.arange(8), 2)):
        raise ValueError('Expected task-major 16 evaluation initials')
    graphs = [camera_graph(camera[i], cfg['neighbors'], cfg['translation_scale_m'], cfg['rotation_scale_deg']) for i in test]
    scopes = {'noncanonical96': slice(None), 'heldout32': slice(64, 96), 'training_catalog64': slice(0, 64)}
    result = dict(protocol=protocol, models={}, archive=str(out))
    examples = []
    for m, name in enumerate(['canonical', 'broad']):
        learners, fit_info = {}, {}
        for family, use_accel in [('rgb', False), ('rgb_accel', True)]:
            model, info = development_fit(images[dev], wrists[dev], tasks[dev], acc[m, dev],
                                          q[m, dev], initials[dev], cfg, use_accel=use_accel)
            learners[family], fit_info[family] = model, info
            np.savez_compressed(out / f'{name}-{family}-weights.npz', **asdict(model))
        baseline = q[m, test, 1:]
        model_result = dict(fit=fit_info, methods={})
        # Precomputed per-candidate scores are a replay cache, not given to search.
        # The callback below is the only score access and each visit is charged.
        cache = {'accel': acc[m, test].mean(-1)}
        for family, model in learners.items():
            cache[family] = np.stack([model.score(images[i], wrists[i], tasks[i], acc[m, i]) for i in test])
        np.savez_compressed(out / f'{name}-candidate-scores.npz', **cache)
        for method, family, random in [('local_accel', 'accel', False), ('random_accel', 'accel', True),
                                       ('local_rgb', 'rgb', False), ('local_rgb_accel', 'rgb_accel', False),
                                       ('random_rgb_accel', 'rgb_accel', True)]:
            repeats = cfg['random_repeats'] if random else 1
            shape = (16, 96, repeats, len(cfg['budgets']))
            selected = np.empty(shape, dtype=np.int16)
            travel = np.empty(shape); turns = np.empty(shape)
            visited = np.empty((16, 96, repeats, max(cfg['budgets'])), dtype=np.int16)
            for ti, initial in enumerate(test):
                edges, distance, translation, rotation = graphs[ti]
                for start in range(1, 97):
                    for repeat in range(repeats):
                        seed = cfg['random_seed'] + int(initial) * 100000 + start * 100 + repeat if random else None
                        queried = []
                        def query(node):
                            queried.append(node)
                            return cache[family][ti, node]
                        traces = search(edges, distance, start, cfg['budgets'], query, random_seed=seed)
                        if queried != traces[max(cfg['budgets'])]['visited']:
                            raise ValueError('Unlogged query')
                        visited[ti, start - 1, repeat] = queried
                        for bi, budget in enumerate(cfg['budgets']):
                            record = traces[budget]
                            path = record['path']
                            ix = (ti, start - 1, repeat, bi)
                            selected[ix] = record['selected']
                            travel[ix] = sum(translation[a, b] for a, b in zip(path[:-1], path[1:]))
                            turns[ix] = sum(rotation[a, b] for a, b in zip(path[:-1], path[1:]))
                        if start == 65 and repeat == 0:
                            examples.append(dict(model=name, method=method, initial=int(initial), start=start, traces=traces))
            # Outcome lookup is deliberately after search has completed.
            outcomes = q[m, test[:, None, None, None], selected]
            np.savez_compressed(out / f'{name}-{method}-replay.npz', selected=selected, visited=visited,
                                translation_m=travel, rotation_deg=turns, success=outcomes)
            rows = {}
            for bi, budget in enumerate(cfg['budgets']):
                rows[str(budget)] = {scope: outcome_summary(outcomes[..., bi].mean(2), baseline,
                                                          travel[..., bi].mean(2), turns[..., bi].mean(2), subset)
                                    for scope, subset in scopes.items()}
            model_result['methods'][method] = rows
            print(f'{name}: {method} complete', flush=True)
        refs = {'stay': np.broadcast_to(np.arange(1, 97), (16, 96)),
                'known_canonical': np.zeros((16, 96), dtype=int)}
        for family, scores in cache.items():
            choices = np.broadcast_to(scores.argmin(1)[:, None], (16, 96)).copy()
            for i in range(16):
                # Keep the start on exact best-score ties; no canonical tie preference.
                tied = scores[i, 1:] == scores[i].min()
                choices[i, tied] = np.arange(1, 97)[tied]
            refs['full_' + family] = choices
        model_result['references'] = {}
        for method, choices in refs.items():
            values = q[m, test[:, None], choices]
            model_result['references'][method] = {scope: outcome_summary(values, baseline, np.zeros_like(values),
                                                                         np.zeros_like(values), subset)
                                                  for scope, subset in scopes.items()}
            for row in model_result['references'][method].values():
                # No acquisition route was simulated for these references.
                row['mean_translation_m'] = None
                row['mean_rotation_deg'] = None
        comparisons = {}
        for left, right in [('local_accel', 'random_accel'), ('local_rgb_accel', 'random_rgb_accel'),
                            ('local_rgb_accel', 'local_accel')]:
            key = left + '_minus_' + right
            comparisons[key] = {}
            for budget in cfg['budgets']:
                comparisons[key][str(budget)] = {}
                for scope in scopes:
                    values = [np.asarray(model_result['methods'][method][str(budget)][scope]['per_initial_success'])
                              for method in [left, right]]
                    comparisons[key][str(budget)][scope] = paired_interval(values[0] - values[1], np.arange(16))
        model_result['paired_method_comparisons'] = comparisons
        result['models'][name] = model_result
    write(out / 'trace-examples.json', examples)
    write(out / 'summary.json', result)
    receipt = dict(status='COMPLETE_EXPLORATORY_OFFLINE_RECOVERY', archive=str(out),
                   audited_source=str(source), new_closed_loops=0, vla_training_steps=0,
                   input_hashes=protocol['input_hashes'], source_hashes=source_hashes,
                   artifacts={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    write(out / 'provenance.json', receipt)
    write(output / 'latest.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config)))


if __name__ == '__main__':
    main()
