"""Deterministic, label-independent camera catalog and matched render planning."""

from collections import defaultdict
import hashlib
import itertools
import json

import numpy as np


def candidate_catalog(seed=41):
    """64 regular training sites, 32 interstitial test sites and a reference.

    Extra evaluation sites are view-coordinate holdouts, not unseen tasks or
    scenes. No score or rollout outcome is used to define this support.
    """
    from scipy.stats import qmc

    train = list(itertools.product(np.linspace(-30, 30, 4),
                                  np.linspace(-15, 15, 4), np.linspace(.8, 1.2, 4)))
    extra = qmc.scale(qmc.Sobol(3, scramble=True, seed=seed).random_base2(5),
                      [-30, -15, .8], [30, 15, 1.2])
    rows = [dict(view_id='canonical', role='reference', parameters=[0., 0., 1.])]
    rows += [dict(view_id=f'train_{i:03d}', role='train', parameters=list(map(float, p)))
             for i, p in enumerate(train)]
    rows += [dict(view_id=f'extra_{i:03d}', role='extra_eval', parameters=list(map(float, p)))
             for i, p in enumerate(extra)]
    return dict(seed=seed, focus_distance_m=1., anchor='episode_frame_zero',
                family='linked_rigid_orbit_3d_not_full_se3', views=rows)


def render_jobs(ledger, catalog, seed=41):
    """Two distinct fixed installations per episode, matched canonical repeats."""
    by_episode = defaultdict(list)
    ids = set()
    for i, row in enumerate(ledger):
        if row['training_pair_index'] != i or row['exposure_indices'] != [2*i, 2*i+1]:
            raise ValueError('Ledger order/exposure identity mismatch')
        if row['source_window_id'] in ids or row['action_horizon'] != 50:
            raise ValueError('Duplicate source identity or changed action horizon')
        ids.add(row['source_window_id'])
        by_episode[(row['dataset_identity'], row['episode_index'])].append(row)
    sites = [r['view_id'] for r in catalog['views'] if r['role'] == 'train']
    if len(sites) < 2:
        raise ValueError('At least two training sites required')
    jobs = []
    for (dataset, episode), rows in sorted(by_episode.items()):
        identity = json.dumps([seed, dataset, episode], separators=(',', ':'))
        digest = hashlib.sha256(identity.encode()).hexdigest()
        # Episode-specific sampling, independent of actions, language and scores.
        rng = np.random.default_rng(int(digest[:16], 16))
        chosen = rng.choice(sites, size=2, replace=False).tolist()
        jobs.append(dict(job_id=digest[:24], dataset_identity=dataset, episode_index=episode,
                         view_ids=chosen, rows=sorted(rows, key=lambda r: r['frame_index'])))
    return jobs
