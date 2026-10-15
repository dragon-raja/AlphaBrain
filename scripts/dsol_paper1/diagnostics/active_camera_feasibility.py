"""Explicit no-render CPU audit of native camera joints and installation flags."""
import argparse


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--execute',action='store_true',help='Compile CPU-only MuJoCo models, no renderer/policy')
    parser.add_argument('--kitchen-integration',action='store_true',help='Run bounded no-render kitchen/control probes instead of geometry audit')
    args=parser.parse_args()
    if not args.execute: parser.error('Explicit --execute required for CPU model compilation')
    if args.kitchen_integration:
        from benchmarks.Robocasa365.tiago_integration import run
    else:
        from benchmarks.Robocasa365.active_camera_audit import run
    run(args.config)


if __name__=='__main__': main()
