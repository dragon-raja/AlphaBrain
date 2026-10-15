"""Explicit forward-only adapter for archived render variants; no environment."""
from pathlib import Path
import argparse


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--worker',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('Explicit --execute is required')
    from AlphaBrain.research.dsol.data.artifacts import read
    from AlphaBrain.research.dsol.analysis.render_score_replay import run
    from .evaluation_policy import ResidentPolicy
    plan=read(a.plan);worker=next(w for w in plan['workers'] if w['worker_id']==a.worker);model=plan['models'][worker['arm']]
    policy=ResidentPolicy(Path(model['checkpoint']),model['config'])
    run(a.plan,a.worker,a.output,policy.trace)


if __name__=='__main__':main()
