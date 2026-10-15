"""Run a bounded XR-1 worker; default is one full-task engineering pilot."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','checkpoint','upstream','model-python','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--formal',action='store_true')
    p.add_argument('--phase',choices=('canonical','perturbed','remaining'),default='canonical')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--work-order',type=Path)
    a=p.parse_args()
    if not a.execute:
        p.error('Explicit --execute required')
    from benchmarks.Robocasa365.xr1_evaluation import run
    print(run(a.panel,a.checkpoint,a.upstream,a.model_python,a.output,
              pilot=not a.formal,shard=a.shard,shards=a.shards,phase=a.phase,work_order=a.work_order),flush=True)


if __name__=='__main__':
    main()
