"""Opt-in replay of one recorded Human300 frame for new training preparation.

Read original XML, simulator state and action labels without changing sources.
Save a canonical three-camera bundle. This is not a training launch.
"""

import argparse
import gzip
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rig-calibration', action='store_true')
    parser.add_argument('--wrist-repeat-control', type=int, default=0)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    if not 0 <= args.wrist_repeat_control <= 64:
        parser.error('Wrist control repeats must be in 0..64')
    if args.output.exists():
        raise FileExistsError(args.output)
    import gymnasium as gym
    import numpy as np
    import pyarrow.parquet as pq
    import robocasa  # registers environments
    from PIL import Image, ImageDraw
    from AlphaBrain.research.dsol.data.artifacts import sha, write

    episode_name = f'episode_{args.episode:06d}'
    extras = args.dataset / 'extras' / episode_name
    parquet, = args.dataset.glob(f'data/*/{episode_name}.parquet')
    sources = [extras / 'ep_meta.json', extras / 'model.xml.gz', extras / 'states.npz',
               args.dataset / 'extras/dataset_meta.json', parquet]
    hashes = {str(p): sha(p) for p in sources}
    metadata = json.loads(sources[0].read_text())
    dataset_meta = json.loads(sources[3].read_text())
    with gzip.open(sources[1], 'rt') as stream:
        xml = stream.read()
    with np.load(sources[2], allow_pickle=False) as payload:
        states = payload['states'].copy()
    actions = np.asarray(pq.read_table(parquet, columns=['action']).column('action').to_pylist())
    if len(states) != len(actions) or not 0 <= args.frame < len(states):
        raise ValueError('Invalid aligned state/action frame')
    random.seed(args.episode)
    np.random.seed(args.episode)
    task = dataset_meta['env_args']['env_name']
    env = gym.make(f'robocasa/{task}', split='pretrain', seed=args.episode, randomize_cameras=False)
    result = dict(task=task, episode=args.episode, frame=args.frame, source_hashes=hashes,
        scope='Recorded-state canonical rendering readiness, not optimized policy or full replay audit')
    try:
        base = env.unwrapped.env
        # This renderer does not consume Gym observations. DrinkwareConsolidation
        # overrides reset without returning observations, so Gym's mapper fails
        # even though the underlying simulator reset itself succeeds.
        base.reset()
        if hasattr(base, 'set_attrs_from_ep_meta'):
            base.set_attrs_from_ep_meta(metadata)
        else:
            base.set_ep_meta(metadata)
        base.reset()
        base.reset_from_xml_string(base.edit_model_xml(xml))
        base.sim.reset()
        base.sim.set_state_from_flattened(states[args.frame])
        base.sim.forward()
        restored = np.asarray(base.sim.get_state().flatten())
        names = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
        frames = [np.ascontiguousarray(base.sim.render(height=256, width=256, camera_name=n)[::-1]) for n in names]
        after = np.asarray(base.sim.get_state().flatten())
        args.output.mkdir(parents=True, exist_ok=False)
        Image.fromarray(np.concatenate(frames, axis=1)).save(args.output / 'canonical.png')
        result.update(state_max_abs_error=float(np.max(np.abs(restored - states[args.frame]))),
            state_unchanged_by_render=bool(np.array_equal(restored, after)),
            action_shape=list(actions.shape), state_shape=list(states.shape),
            camera_names=names, source_hashes_unchanged=all(sha(p) == hashes[str(p)] for p in sources))
        if args.rig_calibration:
            from benchmarks.Robocasa365.camera_rig import ReferenceRig, CALIBRATION_POSES
            rig = ReferenceRig(base.sim)
            reference = rig.render()
            if args.wrist_repeat_control:
                controls = []
                for _ in range(args.wrist_repeat_control):
                    rig.restore()
                    repeated = rig.render()[2]
                    delta = np.abs(repeated.astype(np.int16) - reference[2].astype(np.int16))
                    controls.append(dict(max_abs=int(delta.max()), mean_abs=float(delta.mean()),
                        fraction_gt1=float(np.mean(delta > 1))))
                result['unchanged_camera_repeat_controls'] = controls
            baseline = float(np.linalg.norm(rig.positions[1] - rig.positions[0]))
            rows = []
            montage = Image.new('RGB', (3 * 768, 5 * 282), 'white')
            for index, (name, az, el, radius) in enumerate(CALIBRATION_POSES):
                target = rig.apply(az, el, radius)
                images = rig.render()
                wrist_delta = np.abs(images[2].astype(np.int16) - reference[2].astype(np.int16))
                actual = np.asarray(base.sim.data.cam_xpos)[rig.ids[:2]]
                row = dict(name=name, parameters=[az, el, radius],
                    position_max_error=float(np.max(np.abs(actual - target['positions']))),
                    baseline_error=float(abs(np.linalg.norm(actual[1] - actual[0]) - baseline)),
                    wrist_mount_exact=bool(np.array_equal(base.sim.model.cam_pos[rig.ids[2]], rig.local_positions[rig.ids[2]]) and
                        np.array_equal(base.sim.model.cam_quat[rig.ids[2]], rig.local_quaternions[rig.ids[2]])),
                    wrist_max_abs=int(wrist_delta.max()), wrist_mean_abs=float(wrist_delta.mean()),
                    wrist_fraction_gt1=float(np.mean(wrist_delta > 1)),
                    state_exact=bool(np.array_equal(restored, np.asarray(base.sim.get_state().flatten()))),
                    image_std=[float(x.std()) for x in images],
                    observed_world_positions=actual.tolist())
                row['geometry_and_pixel_pass'] = (row['position_max_error'] < 1e-8 and row['baseline_error'] < 1e-8
                    and row['wrist_mount_exact'] and row['state_exact'] and row['wrist_max_abs'] <= 2
                    and row['wrist_mean_abs'] <= .002 and row['wrist_fraction_gt1'] <= .0001)
                rows.append(row)
                panel = Image.new('RGB', (768, 282), 'white')
                panel.paste(Image.fromarray(np.concatenate(images, axis=1)), (0, 26))
                ImageDraw.Draw(panel).text((8, 6), f'{name} | left / right / wrist', fill='black')
                montage.paste(panel, ((index % 3) * 768, (index // 3) * 282))
            rig.restore()
            montage.save(args.output / 'rig-calibration.png')
            result['rig_calibration'] = rows
            result['camera_support_release'] = 'NOT_RELEASED: geometric consistency is not visibility/reachability evidence'
        result['status'] = 'PASS' if (result['state_max_abs_error'] == 0. and result['state_unchanged_by_render'] and result['source_hashes_unchanged']) else 'HOLD'
        if args.rig_calibration and not all(r['geometry_and_pixel_pass'] for r in rows):
            result['status'] = 'HOLD'
    except Exception as exc:
        result.update(status='ERROR', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        env.close()
        write(args.output / 'summary.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
