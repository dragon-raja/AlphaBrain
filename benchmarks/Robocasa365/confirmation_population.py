"""Generate the approved fresh physical initials, without policy/outcome selection.

One serial preparation worker, not a second multi-worker scheduler. Persist the
first generated state before renderer admission. Never retry an unsuccessful
gate, regenerate a committed state, or silently replace a seed.
"""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import random
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.recovery_confirmation import validate_initial_plan
from AlphaBrain.research.dsol.data.render_reference import CAMERAS, LIMITS
from .snapshot import capture_snapshot, inner_env, restore_snapshot, refresh_observation
from .snapshot_bundle import (save_snapshot_bundle, load_snapshot_bundle,
    reconstruct_and_restore, refresh_bundle_observation, env_numeric_fingerprint,
    snapshot_numeric_fingerprint)
from .evaluation_render import pixel_check


def physical_identity(snapshot):
    """Exclude RNG/serialization identity; retain all registered physical fields."""
    return snapshot_numeric_fingerprint(snapshot)


def capture_one(row, output, prior_physical):
    import gymnasium as gym
    import robocasa  # noqa: F401

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(status='STARTING', requested=row, pixel_gate=LIMITS, checks=[],
                  source_sha256=sha(__file__), pid=os.getpid(), host=platform.node())
    write(output/'status.json', result)
    env = None
    try:
        random.seed(row['scene_seed']); np.random.seed(row['scene_seed'])
        env = gym.make('robocasa/'+row['task'], split=row['split'], seed=row['scene_seed'],
                       randomize_cameras=False)
        env.reset()
        base = inner_env(env)
        ep_meta = base.get_ep_meta()
        snapshot = capture_snapshot(env)
        restore_snapshot(env, snapshot)
        obs = refresh_observation(env)
        images = {'video.'+key: np.asarray(obs['video.'+key]).copy() for key in CAMERAS}
        prompt = str(obs['annotation.human.task_description'])
        restore_snapshot(env, snapshot)
        manifest = save_snapshot_bundle(output/'snapshot', env=env, snapshot=snapshot,
            canonical_images=images, metadata=dict(task=row['task'], split=row['split'],
                scene_seed=row['scene_seed'], index=row['index'], prompt=prompt,
                layout_id=ep_meta.get('layout_id'), style_id=ep_meta.get('style_id'),
                capture_semantics='Native reset then full capture; canonical RGB from restore+force refresh; no action steps',
                camera_semantics='Native linked exterior installation; wrist installation unchanged',
                initial_observation_sha256={key: hashlib.sha256(value.tobytes()).hexdigest()
                                            for key,value in images.items()}))
        result['snapshot_numeric_sha256'] = manifest['snapshot_numeric_sha256']
        if physical_identity(snapshot) in prior_physical:
            raise ValueError('Physical snapshot duplicates registered prior/new state; no automatic replacement')
        allowed = range(1,11) if row['split']=='target' else range(11,61)
        if ep_meta.get('layout_id') not in allowed or ep_meta.get('style_id') not in allowed:
            raise ValueError('Generated scene outside approved official split')
        env.close(); env = None
        bundle = load_snapshot_bundle(output/'snapshot')
        # A new renderer/environment, not just a repeated cached observation.
        env = gym.make('robocasa/'+row['task'], split=row['split'], seed=row['scene_seed'],
                       randomize_cameras=False)
        env.reset(); reconstruct_and_restore(env,bundle)
        for repeat in range(4):
            restore_snapshot(env,bundle.snapshot)
            if env_numeric_fingerprint(env)!=manifest['snapshot_numeric_sha256']:
                raise ValueError('Physical restoration mismatch')
            restored,_ = refresh_bundle_observation(env,bundle)
            result['checks'].append({key: pixel_check(restored[key], value, LIMITS['mean_abs'])
                                     for key,value in images.items()})
            if str(restored['annotation.human.task_description']) != prompt:
                raise ValueError('Persisted task prompt mismatch')
        restore_snapshot(env,bundle.snapshot)
        if env_numeric_fingerprint(env)!=manifest['snapshot_numeric_sha256']:
            raise ValueError('Post-render restoration mismatch')
        result['status'] = 'PASS' if all(c['pass'] for check in result['checks']
                                       for c in check.values()) else 'HOLD_RENDER'
        digest = sha(output/'snapshot/manifest.json')
        identity = '|'.join(map(str,(row['split'],row['task'],row['scene_seed'])))+'|'+digest
        result['job'] = dict(task=row['task'], split=row['split'], scene_seed=row['scene_seed'],
            layout_id=ep_meta['layout_id'],style_id=ep_meta['style_id'],
            initial_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
            path=str((output/'snapshot').resolve()), snapshot_manifest_sha256=digest,
            navigation=row['task']=='NavigateKitchen')
    except Exception as exc:
        result.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if env is not None: env.close()
        result['seconds']=time.monotonic()-started
        write(output/'status.json',result)
    return result


def run(release, output, limit=None):
    release, output = Path(release), Path(output)
    approved=read(release/'authorization.json')
    for file,key in [('proposal.json','proposal_sha256'),('planned-initials.json','initials_sha256'),
                     ('development-strata.json','development_strata_sha256'),
                     ('prior-snapshots.json','prior_snapshots_sha256')]:
        if sha(release/file)!=approved[key]:raise ValueError('Frozen authorization asset changed: '+file)
    rows=validate_initial_plan(read(release/'proposal.json'),read(release/'planned-initials.json'))
    if limit is not None:
        if not 1<=limit<=len(rows):raise ValueError('Invalid preparation limit')
        rows=rows[:limit]
    output.mkdir(parents=True,exist_ok=False)
    status=dict(status='RUNNING',pid=os.getpid(),host=platform.node(),started_unix=time.time(),
                authorization_sha256=sha(release/'authorization.json'),expected=len(rows),
                completed=0,full_task_rollouts=0,policy_inferences=0,records=[])
    write(output/'status.json',status)
    prior={r['snapshot_numeric_sha256'] for r in read(release/'prior-snapshots.json')}
    try:
        for row in rows:
            destination=output/'initials'/row['task']/row['split']/str(row['scene_seed'])
            result=capture_one(row,destination,prior)
            status['records'].append(dict(status_file=str(destination/'status.json'),
                status_sha256=sha(destination/'status.json'),status=result['status'],job=result.get('job')))
            if result['status']!='PASS':raise ValueError('Snapshot admission held; retained first capture')
            prior.add(result['snapshot_numeric_sha256'])
            status['completed']+=1
            write(output/'status.json',status)
            print(f"Snapshot admitted {status['completed']}/{len(rows)}: {row['task']} {row['split']}",flush=True)
        status['status']='COMPLETE'
        write(output/'population.json',[r['job'] for r in status['records']])
    except Exception as exc:
        status.update(status='HOLD',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        status['finished_unix']=time.time()
        write(output/'status.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    run(args.release,args.output,args.limit)


if __name__=='__main__':main()
