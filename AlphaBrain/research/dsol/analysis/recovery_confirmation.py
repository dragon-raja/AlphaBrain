"""Freeze multi-window choices before independent confirmation outcomes exist.

This module consumes scoring records only. It neither launches a simulator nor
reads task-success labels from the confirmation execution stage.
"""
from pathlib import Path

import numpy as np

from ..data.artifacts import read, sha, write
from .short_view_windows import read_cells, window_choices


def selected_conditions(views, extra_views, scores):
    """Stable catalog-order ties; one shared evaluation per selected view."""
    if (len(views) != 97 or len(set(views)) != 97 or views[0] != 'canonical'
            or len(extra_views) != 32 or len(set(extra_views)) != 32
            or not set(extra_views) <= set(views) or 'canonical' in extra_views):
        raise ValueError('Original 97 candidates and 32 disturbance controls required')
    scores = np.asarray(scores, float)
    if scores.shape != (97, 8, 5):
        raise ValueError('All candidates and eight complete five-replan probes required')
    means, chosen = window_choices(scores[None])
    conditions = {'canonical': ['canonical'], 'disturbance_uniform': list(extra_views)}
    conditions.update({f'window{k+1}': [views[j]] for k, j in enumerate(chosen[0])})
    needed = {v for values in conditions.values() for v in values}
    evaluated = [v for v in views if v in needed]
    return dict(conditions=conditions, evaluation_views=evaluated,
                primary_window=3, windows=[1, 2, 3, 4, 5],
                means=means[0].tolist(), chosen_indices=chosen[0].tolist(),
                tie_rule='First minimum in frozen candidate catalog order',
                execution_repeats=list(range(8)), new_rollouts=len(evaluated)*8,
                no_new_oracle_claim=True)


def freeze_ready_initials(probe_root, output):
    """Publish immutable per-initial choices as all 97 score cells become ready.

    Partial populations remain explicitly pending. Corrupt completed cells fail
    closed; neither a replacement candidate nor a smaller candidate set is used.
    Existing selections are preserved, with their source fingerprints rechecked.
    """
    root, output = Path(probe_root), Path(output)
    manifest = read(root/'manifest.json')
    protocol = read(root/'protocol.json')
    for path in (root/'protocol.json', root/'ownership.json'):
        registered = [a for a in manifest['artifacts'] if a['path'] == str(path)]
        if len(registered) != 1 or sha(path) != registered[0]['sha256']:
            raise ValueError('Frozen probe ownership or protocol changed')
    runtime = manifest['runtime']
    config = read(runtime['config'])
    if sha(runtime['config']) != protocol['config_sha256']:
        raise ValueError('Probe configuration changed')
    plan = read(config['panel_plan'])
    if sha(config['panel_plan']) != protocol['panel_plan_sha256']:
        raise ValueError('Probe population changed')
    if (plan['initial_ids'] != protocol['initials'] or plan['all_views'] != protocol['views']
            or len(plan['initial_ids']) != 80 or protocol['score_repeats'] != 8
            or protocol['primary_window'] != 3 or protocol['windows'] != [1, 2, 3, 4, 5]):
        raise ValueError('Approved confirmation scope changed')
    reference = read(plan['reference_plan'])
    if sha(plan['reference_plan']) != plan['reference_plan_sha256']:
        raise ValueError('Reference initial/camera catalog changed')
    jobs = {j['initial_id']: j for j in reference['jobs']}
    views = protocol['views']
    extra = [v['view_id'] for v in reference['catalog']['views'] if v['role'] == 'extra_eval']
    rows = read(root/'ownership.json')['records']
    keys = [(r['initial'], r['view']) for r in rows]
    if (len(keys) != len(set(keys))
            or set(keys) != {(i, v) for i in protocol['initials'] for v in views}):
        raise ValueError('Incomplete or duplicate scoring ownership')
    by_initial = {i: [] for i in protocol['initials']}
    for row in rows:
        by_initial[row['initial']].append(row)
    output.mkdir(parents=True, exist_ok=True)
    binding = dict(format='recovery_confirmation_selection_v1', model=protocol['model'],
                   probe_manifest=str(root/'manifest.json'),
                   probe_manifest_sha256=sha(root/'manifest.json'),
                   protocol_sha256=sha(root/'protocol.json'),
                   ownership_sha256=sha(root/'ownership.json'),
                   source_sha256=sha(__file__), primary_window=3,
                   endpoint_access='No confirmation outcome files read',
                   interpretation='Probe then reset to original initial; not online camera correction')
    if (output/'binding.json').exists():
        if read(output/'binding.json') != binding:
            raise ValueError('Existing selection release has another source or protocol')
    else:
        write(output/'binding.json', binding)
    frozen, pending = [], []
    for initial in protocol['initials']:
        selected_file = output/'initials'/f'{initial}.json'
        records = by_initial[initial]
        if selected_file.exists():
            saved = read(selected_file)
            if saved['binding_sha256'] != sha(output/'binding.json'):
                raise ValueError('Existing choice binding changed')
            for source in saved['sources']:
                if sha(source['path']) != source['sha256']:
                    raise ValueError('Frozen scoring evidence changed after selection')
        else:
            if not all((Path(r['cell'])/'complete.json').exists() for r in records):
                pending.append(initial)
                continue
            scores, _, _, audited = read_cells(records, [initial], views)
            anchors, sources = {}, []
            for row in audited:
                cell = Path(row['cell'])
                identity_file = cell.parents[2]/'identity.json'
                identity = read(identity_file)
                if (identity['model'] != protocol['model'] or initial not in identity['initials']
                        or row['view'] not in identity['views']
                        or identity['config_sha256'] != protocol['config_sha256']
                        or identity['population_mode'] != 'fresh_confirmation'
                        or identity['source_sha256'] != sha(Path(protocol['source'])/'benchmarks/Robocasa365/short_view_probe.py')):
                    raise ValueError('Scoring worker identity mismatch')
                anchor = read(cell/'initial-anchor.json')
                if (anchor['initial'] != initial or anchor['view'] != row['view']
                        or anchor['model'] != protocol['model'] or anchor['action_steps'] != 0
                        or anchor['score_repeats'] != 8
                        or sha(cell/'initial-anchor.npz') != anchor['arrays_sha256']):
                    raise ValueError('Initial-only anchor changed')
                with np.load(cell/'initial-anchor.npz', allow_pickle=False) as data:
                    first = data['scores'].copy()
                j = views.index(row['view'])
                if not np.array_equal(first, scores[0, j, :, 0]):
                    raise ValueError('Window1 does not reproduce independent initial-only score')
                anchors[row['view']] = dict(path=str(cell/'initial-anchor.json'),
                                             sha256=sha(cell/'initial-anchor.json'))
                for file in (cell/'complete.json', cell/'probes.npz', cell/'initial-anchor.json',
                             cell/'initial-anchor.npz', identity_file):
                    sources.append(dict(path=str(file), sha256=sha(file)))
            saved = selected_conditions(views, extra, scores[0])
            job = jobs[initial]
            saved.update(initial=initial, task=job['task'], split=job['split'],
                         scene_seed=job['scene_seed'], model=protocol['model'],
                         binding_sha256=sha(output/'binding.json'),
                         sources=list({s['path']: s for s in sources}.values()), anchors=anchors,
                         all_97_candidates_verified=True, window1_exact=True)
            write(selected_file, saved)
        frozen.append(dict(initial=initial, path=str(selected_file), sha256=sha(selected_file),
                           new_rollouts=saved['new_rollouts']))
    summary = dict(status='COMPLETE' if not pending else 'PARTIAL_SELECTIONS_FROZEN',
                   model=protocol['model'], completed_initials=len(frozen), total_initials=80,
                   records=frozen, pending_initials=pending,
                   planned_rollouts_for_ready_initials=sum(r['new_rollouts'] for r in frozen),
                   full_task_evaluations_started_by_this_module=0)
    write(output/'status.json', summary)
    return summary
