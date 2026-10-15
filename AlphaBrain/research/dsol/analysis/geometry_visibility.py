"""CPU-only, post-hoc geometry and visibility strata on frozen LIBERO outcomes.

No outcomes determine visibility, strata, or metric-based selections. Filtering
starts and restricting destination candidates are separate estimands.
"""
from pathlib import Path
import numpy as np

from ..data.artifacts import read, sha, write
from ..metrics.noise_rules import noise_rule_choices
from .view_structure import camera_distance, correlation


def aggregate_pairs(values, baseline, mask, tasks, seed=20260920):
    """Equal starts within initial, equal initials within task, equal tasks.

    Bootstrap task/initial clusters; not independent start-pose or rollout rows.
    This descriptive interval conditions on the finite measured noise bank.
    """
    values, baseline, mask = np.asarray(values), np.asarray(baseline), np.asarray(mask, bool)
    tasks = np.asarray(tasks)
    if values.shape != baseline.shape or mask.shape != values.shape or values.ndim != 2:
        raise ValueError('Expected matching initial x start arrays')
    counts = mask.sum(1)
    means = np.divide((values * mask).sum(1), counts, out=np.full(len(values), np.nan), where=counts > 0)
    stays = np.divide((baseline * mask).sum(1), counts, out=np.full(len(values), np.nan), where=counts > 0)
    groups = [np.flatnonzero((tasks == t) & (counts > 0)) for t in np.unique(tasks)]
    groups = [g for g in groups if len(g)]
    if not groups:
        return dict(success=None, stay=None, gain=None, ci95_gain=None, initials=0, tasks=0, starts=0)
    avg = lambda a: float(np.mean([a[g].mean() for g in groups]))
    rng = np.random.default_rng(seed)
    task_draws = rng.integers(len(groups), size=(4000, len(groups)))
    samples = np.zeros_like(task_draws, dtype=float)
    for column in range(len(groups)):
        for gi, group in enumerate(groups):
            rows = np.flatnonzero(task_draws[:, column] == gi)
            chosen = rng.choice(group, size=(len(rows), len(group)))
            samples[rows, column] = (means - stays)[chosen].mean(1)
    return dict(success=avg(means), stay=avg(stays), gain=avg(means - stays),
                ci95_gain=np.quantile(samples.mean(1), [.025, .975]).tolist(),
                initials=int((counts > 0).sum()), tasks=len(groups), starts=int(mask.sum()),
                per_initial_success=[None if not np.isfinite(x) else float(x) for x in means],
                per_initial_stay=[None if not np.isfinite(x) else float(x) for x in stays],
                starts_per_initial=counts.tolist())


def constrained_choices(accel, visibility, allowed):
    """Prespecified rules within an explicitly privileged visible subset."""
    a, v, mask = np.asarray(accel), np.asarray(visibility), np.asarray(allowed, bool)
    if a.shape != v.shape or a.shape != mask.shape or not mask.any(1).all():
        raise ValueError('Every initial needs a nonempty eligible candidate set')
    top = [np.flatnonzero(row)[np.argsort(x[row], kind='stable')[:10]] for x, row in zip(a, mask)]
    return dict(min_accel=np.where(mask, a, np.inf).argmin(1),
                visibility=np.where(mask, v, -np.inf).argmax(1),
                accel10_visibility=np.array([min(c, key=lambda j: (-v[i, j], j)) for i, c in enumerate(top)]))


def visible_relations(q, accel, visible, visibility):
    """Ranks within each initial, with/without visibility strata; descriptive."""
    rows = []
    for y, a, ok, v in zip(q, accel, visible, visibility):
        idx = np.flatnonzero(ok)
        row = dict(all_rho=correlation(-a, y), visible_count=len(idx), visible_rho=None,
                   visible_low_quartile_success=None, visible_high_quartile_success=None,
                   within_visibility_bins_rho=[])
        if len(idx) >= 8:
            row['visible_rho'] = correlation(-a[idx], y[idx])
            ranked = idx[np.argsort(a[idx], kind='stable')]
            k = max(1, len(idx)//4)
            row['visible_low_quartile_success'] = float(y[ranked[:k]].mean())
            row['visible_high_quartile_success'] = float(y[ranked[-k:]].mean())
            # Visibility bands defined without success labels. Not equal information.
            bands = np.array_split(idx[np.argsort(v[idx], kind='stable')], 4)
            row['within_visibility_bins_rho'] = [correlation(-a[b], y[b]) for b in bands if len(b) >= 3]
        rows.append(row)
    return rows


def run(config_path):
    cfg = read(config_path)
    source, recovery, output = (Path(cfg[k]) for k in ('analysis', 'recovery', 'output'))
    output.mkdir(parents=True, exist_ok=False)
    provenance = read(source/'provenance.json')
    inputs = {}
    for name in ('matrix.npz', 'states.json'):
        if sha(source/name) != provenance['artifacts'][name]:
            raise ValueError('Frozen matrix identity changed')
        inputs[str(source/name)] = sha(source/name)
    with np.load(source/'matrix.npz') as z:
        q, accel, camera, ids, vis = z['success'].mean(-1), z['accel_members'], z['camera'], z['candidate_ids'], z['visibility']
    states = read(source/'states.json')
    tasks = np.array([s['task_ordinal'] for s in states])
    ev = np.array([i for i, s in enumerate(states) if s['initialization']['init_state_index'] >= 2])
    if q.shape != (2, 32, 97) or len(ev) != 16 or not all(str(x).startswith('broad_heldout_') for x in ids[65:]):
        raise ValueError('Expected original INITIAL32 evaluation population')
    visible, border, pixels, intrinsics = [], [], [], []
    for state in states:
        path = Path(state['initial_asset_record']); meta = read(path); inputs[str(path)] = sha(path)
        rows = {v['candidate_id']: v for v in meta['visibility']}
        entity_rows = [list(rows[str(cid)]['per_camera']['agentview']['entities'].values()) for cid in ids]
        minima = [min(e['visible_pixels'] for e in entries) for entries in entity_rows]
        pixels.append(minima); visible.append(np.array(minima) > 0)
        border.append([any(e['touches_border'] for e in entries) for entries in entity_rows])
        hashes = {v['inputs']['camera_intrinsics']['sha256'] for v in meta['input_identities']}
        intrinsics.append(len(hashes))
    visible, border, pixels = np.array(visible), np.array(border), np.array(pixels)
    distance, rotation = camera_distance(camera)
    starts = np.arange(65, 97)
    masks = dict(all=np.ones((16, 32), bool), visible=visible[ev][:, starts],
                 missing=~visible[ev][:, starts])
    result = dict(scope='Exploratory CPU reuse; zero new VLA inference, rendering or rollout; current PDF unchanged',
        definitions=dict(visible='Every registered task entity has >0 segmented pixels in external initial image; NOT full visibility or equal information',
            population='8 known tasks x 2 evaluation initials x 32 heldout start poses; 32 execution noises per endpoint',
            primary='Filter start poses only; keep original selected endpoints and query paths unchanged',
            restricted='Separate diagnostic: use privileged segmentation to restrict full-query destinations; NOT a deployed new method',
            weights='Equal starts per initial, equal initials per task, equal tasks; empty initials reported/excluded',
            uncertainty='Task/initial bootstrap; conditional on recorded finite noise bank; no multiple-comparison or max-bias correction'),
        geometry=dict(noncanonical_translation_min_m=float(distance[:, 1:].min()),
            noncanonical_translation_median_m=float(np.median(distance[:, 1:])),
            noncanonical_translation_max_m=float(distance[:, 1:].max()),
            all_intrinsics_constant=all(n == 1 for n in intrinsics),
            all_noncanonical_translate=bool(np.all(distance[:, 1:] > 1e-8)),
            maximum_position_difference_between_initials_m=float(np.abs(camera[:, :, :3, 3]-camera[:1, :, :3, 3]).max())),
        strata={name: dict(starts=int(mask.sum()), initials=int(mask.any(1).sum()),
                          per_initial_counts=mask.sum(1).tolist()) for name, mask in masks.items()}, models={})
    old = read(recovery/'summary.json'); inputs[str(recovery/'summary.json')] = sha(recovery/'summary.json')
    budgets = old['protocol']['config']['budgets']; bi = budgets.index(16)
    for m, arm in enumerate(('canonical', 'broad')):
        local_q, a = q[m, ev], accel[m, ev]
        baseline = local_q[:, starts]
        choices = noise_rule_choices(a, vis[ev])
        choices.update(known_canonical=np.zeros(16, int),
                       nearest_position_noncanonical=1+distance[ev, 1:].argmin(1),
                       nearest_rotation_noncanonical=1+rotation[ev, 1:].argmin(1))
        values = {'stay': baseline}
        destinations = {}
        for method, selected in choices.items():
            selected = np.asarray(selected).reshape(16, -1)
            values[method] = np.broadcast_to(local_q[np.arange(16)[:, None], selected].mean(1)[:, None], (16, 32))
            destinations[method] = visible[ev][np.arange(16)[:, None], selected].mean(1).tolist()
        for method in ('local_accel', 'random_accel', 'local_rgb', 'local_rgb_accel', 'random_rgb_accel'):
            file = recovery/f'{arm}-{method}-replay.npz'; inputs[str(file)] = sha(file)
            with np.load(file) as z:
                selected = z['selected'][:, 64:96, :, bi]
                measured = local_q[np.arange(16)[:, None, None], selected]
                np.testing.assert_allclose(measured, z['success'][:, 64:96, :, bi], rtol=0, atol=0)
            values[method+'_16'] = measured.mean(2)
        # All-start results must reproduce the archived primary references.
        for name, archived in [('min_accel', 'full_accel'), ('known_canonical', 'known_canonical')]:
            np.testing.assert_allclose(values[name].mean(), old['models'][arm]['references'][archived]['heldout32']['success']['mean'], rtol=0, atol=1e-12)
        main = {method: {s: aggregate_pairs(v, baseline, mask, tasks[ev]) for s, mask in masks.items()} for method, v in values.items()}
        ok = visible[ev]
        restricted = {}
        if ok.any(1).all():
            selections = constrained_choices(a.mean(-1), vis[ev], ok)
            uniform = (local_q*ok).sum(1)/ok.sum(1)
            vals = {name: local_q[np.arange(16), sel] for name, sel in selections.items()}
            vals['uniform_visible'] = uniform
            for name, val in vals.items():
                restricted[name] = aggregate_pairs(np.broadcast_to(val[:, None], baseline.shape), baseline, masks['visible'], tasks[ev])
        result['models'][arm] = dict(primary=main, destination_visible_fraction=destinations,
            restricted_destination=restricted,
            relations=visible_relations(local_q[:, 1:], a.mean(-1)[:, 1:], ok[:, 1:], vis[ev, 1:]))
    payload = output/'arrays.npz'
    np.savez_compressed(payload, camera=camera, success=q, accel=accel.mean(-1), visible=visible,
                        min_entity_pixels=pixels, border=border, candidate_ids=ids, evaluation_indices=ev, tasks=tasks)
    write(output/'summary.json', result)
    inputs[str(config_path)] = sha(config_path)
    write(output/'provenance.json', dict(inputs=inputs, output_arrays_sha256=sha(payload),
        analysis_source_sha256=sha(__file__), no_new_rollouts=True, no_gpu=True, no_pdf_change=True))
    return result
