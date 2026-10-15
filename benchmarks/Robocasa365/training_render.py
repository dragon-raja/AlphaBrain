"""Recorded-trajectory rendering adapter for matched exterior-only training.

No optimizer, rollout success labels, online camera moves or dataset mutation.
Heavy simulation dependencies are imported only inside the explicit runner.
"""

import gzip
import json
from pathlib import Path
import random
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import sha, write


def render_episode(job, catalog, dataset_root, destination, plan_sha, *, initialization_attempts=1):
    import gymnasium as gym
    import pyarrow.parquet as pq
    import robocasa  # noqa: F401; registers Gym environments
    from PIL import Image
    from benchmarks.Robocasa365.camera_rig import ReferenceRig, CAMERA_NAMES
    from benchmarks.Robocasa365.recorded_initialization import reset_for_recorded_replay

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Never overwrite an existing episode render: ' + str(destination))
    destination.mkdir(parents=True)
    started = time.monotonic()
    root = Path(dataset_root) / job['dataset_identity']
    episode = int(job['episode_index'])
    extras = root / 'extras' / f'episode_{episode:06d}'
    parquet, = root.glob(f'data/*/episode_{episode:06d}.parquet')
    paths = [extras / 'ep_meta.json', extras / 'model.xml.gz', extras / 'states.npz',
             root / 'extras/dataset_meta.json', parquet]
    hashes = {str(p): sha(p) for p in paths}
    metadata = json.loads(paths[0].read_text())
    task = json.loads(paths[3].read_text())['env_args']['env_name']
    with gzip.open(paths[1], 'rt') as stream:
        xml = stream.read()
    with np.load(paths[2], allow_pickle=False) as payload:
        states = payload['states'].copy()
    actions = np.asarray(pq.read_table(parquet, columns=['action']).column('action').to_pylist())
    if len(states) != len(actions) or actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError('Recorded state/action identity is invalid')
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError('Nonfinite source states/actions')
    for row in job['rows']:
        if not 0 <= row['frame_index'] <= len(states) - row['action_horizon']:
            raise ValueError('Source action chunk extends beyond episode')
    # Pretraining scenes must not overlap the existing 1..10 held-out layout set.
    if not 11 <= int(metadata['layout_id']) <= 60:
        raise ValueError('Unexpected training layout: held-out split requires review')
    random.seed(episode)
    np.random.seed(episode)
    env = gym.make(f'robocasa/{task}', split='pretrain', seed=episode, randomize_cameras=False)
    records, files = [], {}
    try:
        base = env.unwrapped.env
        # Do not invoke Gym's observation mapper during recorded-state rendering.
        if hasattr(base, 'set_attrs_from_ep_meta'):
            base.set_attrs_from_ep_meta(metadata)
        else:
            base.set_ep_meta(metadata)
        initialization = reset_for_recorded_replay(base, initialization_attempts)
        base.reset_from_xml_string(base.edit_model_xml(xml))
        base.sim.reset()
        base.sim.set_state_from_flattened(states[0])
        base.sim.forward()
        rig = ReferenceRig(base.sim)
        canonical_mount = rig.mounting()
        lookup = {r['view_id']: r for r in catalog['views']}
        mounts = {}
        for view in job['view_ids']:
            if lookup[view]['role'] != 'train':
                raise ValueError('Evaluation-only view cannot enter training')
            rig.apply(*lookup[view]['parameters'])
            mounts[view] = rig.mounting()
        mount_records = {v: dict(positions=p.tolist(), quaternions=q.tolist())
                         for v, (p, q) in mounts.items()}
        for row in job['rows']:
            frame = row['frame_index']
            base.sim.set_state_from_flattened(states[frame])
            rig.install(canonical_mount)
            actual_state = np.asarray(base.sim.get_state().flatten())
            if not np.array_equal(actual_state, states[frame]):
                raise ValueError('Recorded simulator state did not restore exactly')
            image_dir = destination / row['source_window_id']
            image_dir.mkdir()

            def save(name, image):
                path = image_dir / f'{name}.png'
                Image.fromarray(image).save(path, compress_level=3)
                files[str(path.relative_to(destination))] = sha(path)
                return str(path)

            canonical_images = rig.render()
            shared_wrist = save('wrist', canonical_images[2])
            canonical = dict(left=save('canonical_left', canonical_images[0]),
                             right=save('canonical_right', canonical_images[1]), wrist=shared_wrist)
            broad = []
            for view in job['view_ids']:
                rig.install(mounts[view])
                exterior = [np.ascontiguousarray(base.sim.render(height=256, width=256, camera_name=n)[::-1])
                            for n in CAMERA_NAMES[:2]]
                broad.append(dict(view_id=view, image_files=dict(left=save(view+'_left', exterior[0]),
                    right=save(view+'_right', exterior[1]), wrist=shared_wrist)))
                if not np.array_equal(np.asarray(base.sim.get_state().flatten()), actual_state):
                    raise ValueError('Rendering modified physical state')
            records.append(dict(source_window_id=row['source_window_id'],
                training_pair_index=row['training_pair_index'], frame_index=frame,
                canonical_exposures=[dict(view_id='canonical', image_files=canonical)]*2,
                broad_exposures=broad))
        if not all(sha(p) == hashes[str(p)] for p in paths):
            raise ValueError('Source data changed during rendering')
        result = dict(status='PASS', plan_sha256=plan_sha, job_id=job['job_id'], task=task,
            initialization=initialization, renderer_sha256=sha(__file__),
            episode_index=episode, source_hashes=hashes, layout_id=metadata['layout_id'],
            style_id=metadata['style_id'], language=metadata['lang'], state_shape=list(states.shape),
            action_shape=list(actions.shape), records=records, image_sha256=files,
            anchor_frame=0, fixed_local_mounts=mount_records, wrist_shared_exact=True,
            seconds=time.monotonic()-started, codec='lossless_png_rgb256_compress3',
            scope='Training images only; original proprioception/actions/language remain in source loader')
        write(destination / 'summary.json', result)
        return result
    finally:
        env.close()
