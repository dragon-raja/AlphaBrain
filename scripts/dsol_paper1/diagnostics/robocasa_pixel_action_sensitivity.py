"""Explicit VLA input sensitivity on saved observations; no simulator required."""

import argparse
import dataclasses
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.shared import normalize
    from AlphaBrain.research.dsol.data.artifacts import write, sha
    from AlphaBrain.research.dsol.analysis.action_sensitivity import action_difference

    cfg = config.get_config('pi05_pretrain_human300_all_lora_fast25')
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, data_dirs=[]))
    policy = policy_config.create_trained_policy(cfg, args.checkpoint,
        norm_stats=normalize.load(args.checkpoint / 'assets'), sample_kwargs={'num_steps': 10})
    args.output.mkdir(parents=True)
    results, outputs = [], {}
    for index, path in enumerate(args.inputs):
        with np.load(path, allow_pickle=False) as data:
            common = {'prompt': str(data['prompt'].item()), 'observation/state': data['state'].copy()}
            requests = {label: dict(common, **{'observation/' + key: data[label + '_' + key].copy()
                        for key in ('image', 'right_image', 'wrist_image')}) for label in ('historical', 'current')}
        for repeat in range(8):
            seed = 918100 + repeat
            noise = np.random.default_rng(seed).standard_normal((50, 32), dtype=np.float32)
            reference = np.asarray(policy.infer(requests['historical'], noise=noise.copy())['actions'])
            actual = np.asarray(policy.infer(requests['current'], noise=noise.copy())['actions'])
            control = np.asarray(policy.infer(requests['historical'], noise=noise.copy())['actions'])
            results.append(dict(input=str(path), input_sha256=sha(path), seed=seed,
                historical_vs_current=action_difference(reference, actual),
                same_input_repeat=action_difference(reference, control)))
            outputs[f'input_{index}_noise_{repeat}'] = np.stack([reference, actual, control])
            write(args.output / 'progress.json', dict(checkpoint=str(args.checkpoint), records=results))
    np.savez_compressed(args.output / 'actions.npz', **outputs)
    write(args.output / 'summary.json', dict(status='DIAGNOSTIC_COMPLETE_NOT_ADMITTED', records=results,
        checkpoint=str(args.checkpoint), action_sha256=sha(args.output / 'actions.npz'),
        scope='Paired initial action chunks only; no task success or automatic pixel-gate release'))


if __name__ == '__main__':
    main()
