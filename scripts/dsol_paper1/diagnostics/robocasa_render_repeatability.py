"""Explicit RGB repeatability probe; flags are changed only inside the diagnostic."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--calibration', action='store_true', help='Unmodified renderer repeats only; no ablations')
    parser.add_argument('--pipeline-ablation', action='store_true', help='Diagnostic only: rebuild framebuffer at explicit MSAA sample counts')
    parser.add_argument('--sample-readback', action='store_true', help='Diagnostic only: isolate individual MSAA samples and resolve')
    args = parser.parse_args()
    if not args.execute or not 2 <= args.repeats <= 32:
        parser.error('Explicit --execute and 2..32 repeats required')
    if sum((args.calibration, args.pipeline_ablation, args.sample_readback)) > 1:
        parser.error('Choose only one diagnostic mode')
    from benchmarks.Robocasa365.render_diagnostics import probe_snapshot
    result = probe_snapshot(args.snapshot, args.output, args.repeats, args.calibration,
                            args.pipeline_ablation, args.sample_readback)
    for name, stage in result['stages'].items():
        print(name, {key: dict(unique=value['unique_images'],
              max_repeat=max(x['max_abs'] for x in value['repeat']),
              max_mean_repeat=max(x['mean_abs'] for x in value['repeat']))
              for key, value in stage['cameras'].items()}, flush=True)


if __name__ == '__main__':
    main()
