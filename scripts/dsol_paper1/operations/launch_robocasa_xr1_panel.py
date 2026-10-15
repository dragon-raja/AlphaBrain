"""Freeze/launch the bounded 800-rollout XR-1 diagnostic after a full-task pilot."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','checkpoint','upstream','model-python','sim-python','source','output','pilot'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--phase',choices=('canonical','perturbed','remaining'),default='canonical')
    args=p.parse_args()
    if not args.execute:
        p.error('Explicit --execute required')
    from AlphaBrain.research.dsol.data.xr1_program import prepare_and_launch
    fields=vars(args);fields.pop('execute')
    print(prepare_and_launch(**fields),flush=True)


if __name__=='__main__':
    main()
