"""Opt-in fixed-state linked-exterior render calibration (not task evaluation)."""

import argparse
import hashlib
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='CloseBlenderLid')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    if args.output.exists():
        raise FileExistsError(args.output)
    import gymnasium as gym
    import numpy as np
    import robocasa  # registers gym environments
    from scipy.spatial.transform import Rotation
    from PIL import Image, ImageDraw
    from AlphaBrain.research.dsol.acquisition.camera_rig import orbit_rig
    from AlphaBrain.research.dsol.data.artifacts import write

    random.seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(f'robocasa/{args.task}', split='pretrain', seed=args.seed, randomize_cameras=False)
    try:
        env.reset()
        base = env.unwrapped.env
        sim = base.sim
        names = ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']
        ids = [sim.model.camera_name2id(n) for n in names]
        local_positions = np.asarray(sim.model.cam_pos).copy()
        local_quaternions = np.asarray(sim.model.cam_quat).copy()
        world_positions = np.asarray(sim.data.cam_xpos)[ids[:2]].copy()
        world_rotations = np.asarray(sim.data.cam_xmat)[ids[:2]].reshape(2, 3, 3).copy()
        baseline = float(np.linalg.norm(world_positions[1] - world_positions[0]))
        before = np.asarray(sim.get_state().flatten()).copy()

        def render():
            return [np.ascontiguousarray(sim.render(height=256, width=256, camera_name=n)[::-1]) for n in names]

        sim.forward()
        canonical = render()
        candidates = [('canonical', 0, 0, 1.), ('yaw_m30', -30, 0, 1.), ('yaw_p30', 30, 0, 1.),
            ('pitch_m15', 0, -15, 1.), ('pitch_p15', 0, 15, 1.), ('near_080', 0, 0, .8), ('far_120', 0, 0, 1.2),
            ('corner_nn_near', -30, -15, .8), ('corner_pp_far', 30, 15, 1.2)]
        rows, panels = [], []
        args.output.mkdir(parents=True, exist_ok=False)
        for name, az, el, rad in candidates:
            sim.model.cam_pos[:] = local_positions
            sim.model.cam_quat[:] = local_quaternions
            target = orbit_rig(world_positions, world_rotations, azimuth_deg=az, elevation_deg=el, radius_scale=rad)
            for i, camera_id in enumerate(ids[:2]):
                body = sim.model.cam_bodyid[camera_id]
                body_rotation = np.asarray(sim.data.body_xmat[body]).reshape(3, 3)
                body_position = np.asarray(sim.data.body_xpos[body])
                sim.model.cam_pos[camera_id] = body_rotation.T @ (target['positions'][i] - body_position)
                local_rotation = body_rotation.T @ target['rotations'][i]
                xyzw = Rotation.from_matrix(local_rotation).as_quat()
                sim.model.cam_quat[camera_id] = xyzw[[3, 0, 1, 2]]
            sim.forward()
            frames = render()
            after = np.asarray(sim.get_state().flatten())
            measured = np.asarray(sim.data.cam_xpos)[ids[:2]]
            wrist_exact = bool(np.array_equal(canonical[2], frames[2]))
            wrist_delta = np.abs(canonical[2].astype(np.int16) - frames[2].astype(np.int16))
            # Same tolerance as the pre-existing restore/render acceptance;
            # retain exact-pixel result and measured errors rather than hiding it.
            wrist_pixel = dict(max_abs=int(wrist_delta.max()), mean_abs=float(wrist_delta.mean()),
                fraction_gt1=float(np.mean(wrist_delta > 1)), fraction_gt2=float(np.mean(wrist_delta > 2)))
            wrist_pixel['within_tolerance'] = (wrist_pixel['max_abs'] <= 2 and wrist_pixel['mean_abs'] <= .002
                and wrist_pixel['fraction_gt1'] <= .0001 and wrist_pixel['fraction_gt2'] == 0.)
            row = dict(name=name, parameters=[az, el, rad], reference_focus_distance_m=1.,
                achieved_world_position_max_error=float(np.max(np.abs(measured - target['positions']))),
                baseline_m=float(np.linalg.norm(measured[1] - measured[0])),
                state_exact=bool(np.array_equal(before, after)), wrist_rgb_exact=wrist_exact,
                wrist_pixel_error=wrist_pixel,
                wrist_mount_exact=bool(np.array_equal(sim.model.cam_pos[ids[2]], local_positions[ids[2]]) and
                    np.array_equal(sim.model.cam_quat[ids[2]], local_quaternions[ids[2]])),
                rgb_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in frames],
                rgb_std=[float(x.std()) for x in frames])
            rows.append(row)
            panel = Image.new('RGB', (768, 282), 'white')
            for i, frame in enumerate(frames):
                panel.paste(Image.fromarray(frame), (i * 256, 26))
            ImageDraw.Draw(panel).text((8, 6), f'{name} | left / right / wrist', fill='black')
            panel.save(args.output / f'{name}.png')
            panels.append(panel)
        montage = Image.new('RGB', (768 * 3, 282 * 3), 'white')
        for i, panel in enumerate(panels):
            montage.paste(panel, ((i % 3) * 768, (i // 3) * 282))
        montage.save(args.output / 'montage.png')
        passed = all(r['state_exact'] and r['wrist_mount_exact'] and r['wrist_pixel_error']['within_tolerance']
            and r['achieved_world_position_max_error'] < 1e-8 and abs(r['baseline_m'] - baseline) < 1e-8 for r in rows)
        write(args.output / 'summary.json', dict(status='PASS' if passed else 'HOLD', task=args.task,
            seed=args.seed, scope='Calibration positions only, not frozen train/eval support or physically reachable camera validation',
            initial_baseline_m=baseline, camera_parent_ids=[int(sim.model.cam_bodyid[x]) for x in ids], rows=rows))
        print(args.output, flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
