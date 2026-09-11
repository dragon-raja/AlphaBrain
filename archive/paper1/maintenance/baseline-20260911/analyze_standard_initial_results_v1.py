#!/usr/bin/env python3
"""Read-only evaluation audit, offline rankers and statistics for INITIAL32.

No policy service, simulator, evaluation launcher or checkpoint training is run.
Only a new, immutable derived-analysis directory and its latest pointer are written.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.stats import spearmanr

ROOT = Path('/share/longjunyu/alphabrain/experiments/dsol-standard-initial-aa0-v1-20260908')
OUTPUT = ROOT / 'analysis/research-v1'
MODELS = ['canonical', 'broad']
WEIGHTS = Path('/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
CONTRACT = {
    'population': 'all 32 official initial states; 8 known tasks, 4 initials per task',
    'development_initial_indices': [0, 1], 'test_initial_indices': [2, 3],
    'families': ['geometry_context_ridge', 'candidate_image_ridge'],
    'alpha_grid': [1.0, 10.0, 100.0], 'context_pca_dim': 4, 'candidate_pca_dim': 8,
    'cv': 'two development folds: train initial 0 validate 1, then reverse; task-balanced top1 success',
    'alpha_tie': 'larger alpha', 'candidate_tie': 'catalog order',
    'encoder': 'frozen ImageNet ResNet18 pool, CPU; no downloads',
    'encoder_sha256': 'f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec',
    'input_permissions': {
        'geometry_context_ridge': 'canonical external RGB + wrist RGB + task ID + candidate camera geometry; no candidate RGB or visibility',
        'candidate_image_ridge': 'same plus all candidate RGB; queryable-image condition, not free physical acquisition',
    },
    'outcome': 'mean of 32 full-task noise repeats, then one view per initial',
    'test_access_disclosure': 'aggregate test results were inspected before this exploratory learner design; no claim of fully blind confirmation',
    'no_test_hyperparameter_tuning': True, 'no_new_closed_loops': True,
    'no_vla_training': True, 'no_dynamic_camera': True,
    'bootstrap': '10000 paired hierarchical resamples: tasks then initials within task; conditional on measured noise means; exploratory with 8 tasks',
}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, p)


def hierarchy(q):
    """q = task x initial x candidate noise-averaged success."""
    if q.ndim != 3 or not np.isfinite(q).all():
        raise ValueError('Expected a finite task x initial x candidate matrix')
    return dict(canonical=float(q[:, :, 0].mean()),
                uniform=float(q.mean()),
                global_best=float(q.mean(axis=(0, 1)).max()),
                task_best=float(q.mean(axis=1).max(axis=1).mean()),
                initial_oracle=float(q.max(axis=2).mean()))


def metric_choices(acc, vis):
    top = np.argsort(acc, axis=1, kind='stable')[:, :10]
    return dict(min_accel=np.argmin(acc, axis=1),
                visibility=np.argmax(vis, axis=1),
                accel10_visibility=np.array([min(top[i], key=lambda c: (-vis[i, c], c))
                                            for i in range(len(acc))]))


def paired_interval(values, subset, seed=20260911):
    """Mean/CI of fixed per-initial outcomes or paired differences (not max bias correction)."""
    v = np.asarray(values)[subset].reshape(8, -1)
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 8, (10000, 8))
    i = rng.integers(0, v.shape[1], (10000, 8, v.shape[1]))
    samples = v[t[:, :, None], i].mean(axis=(1, 2))
    return {'mean': float(v.mean()), 'ci95': np.quantile(samples, [.025, .975]).tolist()}


def pca_fit(x, train, dimension):
    flat = x[train].reshape(-1, x.shape[-1]); mean = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    components = vt[:min(dimension, len(flat) - 1)]
    return (x - mean) @ components.T, mean, components


def feature_matrix(geometry, context, images, tasks, train, family):
    c, cm, cv = pca_fit(context, train, CONTRACT['context_pca_dim'])
    n, k, _ = geometry.shape
    task = np.eye(8)[tasks]
    broadcast = lambda x: np.broadcast_to(x[:, None, :], (n, k, x.shape[-1]))
    pieces = [geometry, broadcast(task), broadcast(c),
              (geometry[:, :, :, None] * task[:, None, None, :]).reshape(n, k, -1),
              (geometry[:, :, :, None] * c[:, None, None, :]).reshape(n, k, -1)]
    transforms = {'context_mean': cm, 'context_components': cv}
    if family == 'candidate_image_ridge':
        z, zm, zv = pca_fit(images, train, CONTRACT['candidate_pca_dim'])
        pieces.extend([z, z - z[:, :1],
                       (z[:, :, :, None] * task[:, None, None, :]).reshape(n, k, -1)])
        transforms.update(image_mean=zm, image_components=zv)
    elif family != 'geometry_context_ridge':
        raise ValueError(family)
    return np.concatenate(pieces, axis=-1), transforms


def ridge_predict(features, labels, train, alpha):
    x = features[train].reshape(-1, features.shape[-1]); y = labels[train].reshape(-1)
    mu = x.mean(axis=0); scale = x.std(axis=0); scale[scale < 1e-8] = 1
    x = (x - mu) / scale; ym = y.mean()
    w = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - ym))
    pred = ((features - mu) / scale) @ w + ym
    return pred, dict(feature_mean=mu, feature_scale=scale, weights=w, intercept=np.array(ym))


def fit_ranker(geometry, context, images, tasks, initials, q, family):
    dev = np.flatnonzero(initials < 2)
    trials = []
    for alpha in CONTRACT['alpha_grid']:
        scores = []
        for vi in [0, 1]:
            train = np.flatnonzero(initials == 1 - vi); valid = np.flatnonzero(initials == vi)
            x, _ = feature_matrix(geometry, context, images, tasks, train, family)
            pred, _ = ridge_predict(x, q, train, alpha)
            choices = pred[valid].argmax(axis=1)
            scores.extend(q[valid, choices].tolist())
        trials.append({'alpha': alpha, 'development_oof_success': float(np.mean(scores))})
    chosen = max(trials, key=lambda a: (a['development_oof_success'], a['alpha']))
    x, transform = feature_matrix(geometry, context, images, tasks, dev, family)
    pred, weights = ridge_predict(x, q, dev, chosen['alpha'])
    return pred, {**transform, **weights}, {'cv': trials, 'alpha': chosen['alpha'], 'feature_dimension': x.shape[-1]}


def audit_and_load(out):
    r = read(ROOT / 'release.json'); rh = sha(ROOT / 'release.json')
    states = r['states']; cids = [c['selected_candidate_id'] for c in r['protocol']['state_blocks'][0]['candidates']]
    assert len(states) == 32 and len(cids) == 97 and cids[0] == 'canonical'
    inputs = {str(ROOT / 'release.json'): rh}
    for h in r['hosts']:
        for path in [ROOT/f'hosts/{h}/dense-complete.json', ROOT/f'hosts/{h}/gate-pass.json',
                     ROOT/f'scheduling/dynamic-v1/{h}/cross-host-gate-pass.json']:
            inputs[str(path)] = sha(path)
        assert read(ROOT/f'hosts/{h}/dense-complete.json')['episodes'] == 99328
    vis = np.empty((32, 97)); acc = np.empty((2, 32, 97, 8)); camera = []; artifact_shas = []
    for i in range(32):
        path = ROOT/f'assets/aa0-a/{i:02d}/render.json'; a = read(path)
        inputs[str(path)] = sha(path)
        assert a['release_sha256'] == rh and a['candidate_count'] == 97
        artifact = Path(a['artifact']); assert sha(artifact) == a['artifact_sha256']
        # The rollout receipt hashes render.json (which in turn hashes the NPZ).
        inputs[str(artifact)] = a['artifact_sha256']; artifact_shas.append(sha(path))
        with np.load(artifact) as z:
            assert z['candidate_ids'].tolist() == cids
            camera.append(z['camera_to_world_opencv'])
        vd = {v['candidate_id']: v['per_camera']['agentview']['score'] for v in a['visibility']}
        vis[i] = [vd[c] for c in cids]
        for mi, model in enumerate(MODELS):
            path = ROOT/f'scores/{model}/aa0-a/{i:02d}/ranking.json'; sc = read(path)
            complete = read(path.parent/'completion.json')
            assert complete['ranking_sha256'] == sha(path) and complete['release_sha256'] == rh
            assert sc['pair_key'] == states[i]['pair_key'] and sc['physics_state_sha256'] == a['physics_state_sha256']
            inputs[str(path)] = sha(path)
            sd = {v['candidate_id']: v['member_accel_3'] for v in sc['ranking']}
            acc[mi, i] = [sd[c] for c in cids]
    def cell(key):
        mi, si = key; model = MODELS[mi]; s = states[si]; ii = s['initialization']['init_state_index']
        h = next(h for h, v in r['hosts'].items() if si in v['state_indices'])
        folder = ROOT/f'hosts/{h}/dense-init-{ii}/{model}/{si:02d}'
        files = sorted(folder.glob('*.json')); assert len(files) == 3104
        y = np.full((97, 32), -1, dtype=np.int8); hashes = []; lengths = np.empty((97, 32), dtype=np.int16)
        for f in files:
            raw = f.read_bytes(); a = json.loads(raw); idx = a['aa0']['index']; ss, rem = divmod(idx, 3104); c, n = divmod(rem, 32)
            assert ss == si and int(f.stem) == idx and y[c, n] == -1
            assert a['aa0']['release_sha256'] == rh and a['aa0']['model'] == model and a['aa0']['host'] == h
            assert a['aa0']['checkpoint_sha256'] == r['models'][model]['weights_sha256']
            assert a['pair_key'] == s['pair_key'] and a['selected_candidate_id'] == cids[c] and a['policy_repeat_id'] == n
            assert a['noise_bank_manifest_sha256'] == r['noise_bank']['manifest_sha256'] and a['explicit_flow_noise']
            assert a['initialization'] == s['initialization'] and not a['initialization_receipt']['demonstration_state_used']
            assert a['initialization_receipt']['matched_metric_asset_sha256'] == artifact_shas[si]
            assert set(a['aa0']['render_samples']) == {0} and a['status'] == 'complete'
            assert a['scene_construction'] is None and isinstance(a['success'], bool)
            y[c, n] = a['success']; lengths[c, n] = a['completion_steps']
            hashes.append((str(f.relative_to(ROOT)), hashlib.sha256(raw).hexdigest()))
        assert (y >= 0).all()
        return mi, si, y, lengths, hashes
    Y = np.empty((2, 32, 97, 32), dtype=np.int8); steps = np.empty_like(Y, dtype=np.int16); records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for k, (m, s, y, length, hashes) in enumerate(pool.map(cell, [(m, s) for m in range(2) for s in range(32)])):
            Y[m, s] = y; steps[m, s] = length; records.extend(hashes)
            if (k + 1) % 8 == 0: print('Audit cells', k + 1, '/64', flush=True)
    write(out/'record-hashes.json', records)
    write(out/'inputs.json', inputs)
    np.savez_compressed(out/'matrix.npz', success=Y, completion_steps=steps, accel_members=acc,
                        visibility=vis, camera=np.stack(camera), candidate_ids=np.array(cids))
    write(out/'states.json', states)
    return r, Y, acc, vis, np.stack(camera)


def encode_assets(out):
    import torch
    from torchvision.models import resnet18
    assert sha(WEIGHTS) == CONTRACT['encoder_sha256']
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    model = resnet18(weights=None); model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    model.fc = torch.nn.Identity(); model.eval()
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    def encode(images):
        output = []
        with torch.inference_mode():
            for start in range(0, len(images), 16):
                x = torch.from_numpy(images[start:start+16]).permute(0, 3, 1, 2).float()/255
                output.append(model((x - mean)/std).numpy())
        return np.concatenate(output)
    external, context = [], []
    for i in range(32):
        with np.load(ROOT/f'assets/aa0-a/{i:02d}/policy_inputs.npz') as z:
            e = encode(z['external_images']); w = encode(z['wrist_images'][:1])
        external.append(e); context.append(np.concatenate([e[0], w[0]]))
        if (i+1) % 8 == 0: print('CPU image features', i+1, '/32', flush=True)
    external, context = np.stack(external), np.stack(context)
    np.savez_compressed(out/'embeddings.npz', external=external, context=context)
    return external, context


def summarize(out, r, Y, acc, vis, camera, images, context):
    q = Y.mean(axis=-1); am = acc.mean(axis=-1)
    initials = np.array([s['initialization']['init_state_index'] for s in r['states']])
    tasks = np.array([s['task_ordinal'] for s in r['states']]); dev = np.flatnonzero(initials < 2); test = np.flatnonzero(initials >= 2)
    # Actual camera geometry: displacement and relative rotation from the canonical view.
    delta = camera[:, :, :3, 3] - camera[:, :1, :3, 3]
    rotation = np.einsum('sji,scjk->scik', camera[:, 0, :3, :3], camera[:, :, :3, :3])
    canonical_flag = np.broadcast_to((np.arange(97) == 0)[None, :, None], (32, 97, 1))
    geom = np.concatenate([delta, rotation.reshape(32, 97, 9), canonical_flag], axis=-1)
    results = {'contract': CONTRACT, 'episodes': int(Y.size), 'models': {}, 'task_ids': [r['states'][i*4]['task_id'] for i in range(8)]}
    selected_all = {}; per_initial = {}
    for m, model in enumerate(MODELS):
        choices = {'canonical': np.zeros(32, dtype=int), **metric_choices(am[m], vis)}
        global_c = q[m, dev].mean(axis=0).argmax()
        task_c = q[m, dev].reshape(8, 2, 97).mean(axis=1).argmax(axis=1)
        choices['dev_global_fixed'] = np.full(32, global_c)
        choices['dev_task_fixed'] = task_c[tasks]
        learning = {}
        for family in CONTRACT['families']:
            pred, weights, info = fit_ranker(geom, context, images, tasks, initials, q[m], family)
            choices[family] = pred.argmax(axis=1); learning[family] = info
            np.savez_compressed(out/f'{model}-{family}.npz', **weights, predictions=pred,
                                choices=choices[family], development_indices=dev, test_indices=test)
        methods = {name: q[m, np.arange(32), c] for name, c in choices.items()}
        methods['uniform'] = q[m].mean(axis=1)
        methods['initial_oracle'] = q[m].max(axis=1)
        task_upper = q[m].reshape(8, 4, 97).mean(axis=1).argmax(axis=1)
        methods['all_task_oracle'] = q[m, np.arange(32), task_upper[tasks]]
        scopes = {}
        for label, subset in [('all32', np.arange(32)), ('heldout16', test), ('development16', dev)]:
            scopes[label] = {'hierarchy': hierarchy(q[m, subset].reshape(8, -1, 97)), 'methods': {}}
            for key, value in methods.items():
                if key == 'all_task_oracle' and label != 'all32': continue
                scopes[label]['methods'][key] = {
                    'success': paired_interval(value, subset),
                    'gain_vs_canonical': paired_interval(value-methods['canonical'], subset),
                    'per_task_success': value[subset].reshape(8, -1).mean(axis=1).tolist(),
                }
        correlation = []
        stability = []
        for s in range(32):
            aa = spearmanr(-am[m, s], q[m, s]).statistic
            vv = spearmanr(vis[s], q[m, s]).statistic
            win = acc[m, s].argmin(axis=0)
            stability.append({'initial': s, 'distinct_single_noise_winners': int(len(set(win.tolist()))),
                              'agreement_with_ensemble_winner': float(np.mean(win == choices['min_accel'][s]))})
            correlation.append({'initial': s, 'minus_accel_spearman': float(aa) if np.isfinite(aa) else None,
                                'visibility_spearman': float(vv) if np.isfinite(vv) else None})
        results['models'][model] = {'scopes': scopes, 'learners': learning, 'rank_correlations': correlation,
                                    'score_noise_stability': stability,
                                    'candidate_groups': {'canonical': float(q[m, :, 0].mean()),
                                        'training_catalog_64': float(q[m, :, 1:65].mean()),
                                        'heldout_catalog_32': float(q[m, :, 65:].mean())},
                                    'fraction_within_5pp_of_canonical': float((q[m] >= q[m, :, :1]-.05).mean())}
        selected_all[model] = {k: v.tolist() for k, v in choices.items()}
        per_initial[model] = {k: v.tolist() for k, v in methods.items()}
    results['training_effect'] = {k: paired_interval(np.asarray(per_initial['broad'][k])-np.asarray(per_initial['canonical'][k]), np.arange(32))
                                  for k in ['canonical', 'uniform', 'initial_oracle']}
    write(out/'selections.json', selected_all); write(out/'per-initial.json', per_initial)
    write(out/'summary.json', results)
    return results


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); out = OUTPUT/'history'/stamp
    out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json', {**CONTRACT, 'frozen_utc': stamp, 'source_sha256': sha(__file__),
                                'release_sha256': sha(ROOT/'release.json')})
    r, Y, acc, vis, camera = audit_and_load(out)
    images, context = encode_assets(out)
    result = summarize(out, r, Y, acc, vis, camera, images, context)
    provenance = {'status': 'COMPLETE_OFFLINE_ANALYSIS', 'archive': str(out), 'episodes': int(Y.size),
                  'release_sha256': sha(ROOT/'release.json'), 'source_sha256': sha(__file__),
                  'encoder_sha256': sha(WEIGHTS), 'new_closed_loops': 0, 'vla_training_steps': 0,
                  'artifacts': {f.name: sha(f) for f in out.iterdir() if f.is_file()}}
    write(out/'provenance.json', provenance); write(OUTPUT/'latest.json', provenance)
    print(json.dumps({'archive': str(out), 'models': {m: result['models'][m]['scopes']['heldout16'] for m in MODELS}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
