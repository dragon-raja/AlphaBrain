"""CPU-only audit of settled native PPO reward and exploration, without replay."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from AlphaBrain.research.dsol.analysis.target_search_diagnostic import training_diagnostic


def load(path):
    raw = path.read_bytes()
    return path, json.loads(raw), hashlib.sha256(raw).hexdigest()


def build(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('New diagnostic output required')
    sources, episode_rows, update_rows = {}, [], []
    for model in ('fastwam', 'groot_n15'):
        for seed in range(3):
            run = f'G-seed-{seed}'
            folder = root / model / 'training-v2/runs' / run
            paths = [folder / 'episodes' / f'{i:04d}.json' for i in range(2048)]
            paths += [folder / 'updates' / f'{i:03d}.json' for i in range(1, 65)]
            with ThreadPoolExecutor(max_workers=6) as pool:
                loaded = list(pool.map(load, paths))
            sources.update({str(p): sha for p, _, sha in loaded})
            es = [dict(j['metrics'], run=run) for _, j, _ in loaded[:2048]]
            us = [dict(j, run=run) for _, j, _ in loaded[2048:]]
            for i, e in enumerate(es):
                if e['episode'] != i or e['policy_update'] != i // 32:
                    raise ValueError('Training record identity differs')
            audit = training_diagnostic(es, us, episodes_per_update=32)
            cumulative = []
            for u, a in zip(us, audit):
                idx = u['update']
                batch = es[(idx-1)*32:idx*32]
                cumulative.extend(e['initial_accel'] for e in batch)
                scale = u['reward_settlement']['denominator']
                if u['reward_settlement']['kind'] != 'G' or abs(median(cumulative)-scale) > 1e-12:
                    raise ValueError('Online initial-score median not reconciled')
                update_rows.append(dict(model=model, **a,
                                        disagreement_q75=u['reward_settlement']['disagreement_q75']))
                for e, reward in zip(batch, u['reward_settlement']['rewards']):
                    raw_gain=e['initial_accel']-e['best_accel']
                    cost=.01*(e['queries']-1)
                    if abs(raw_gain/scale-cost-reward)>1e-9:
                        raise ValueError('Episode reward does not reconcile')
                    episode_rows.append(dict(model=model, run=run, update=idx, episode=e['episode'],
                        state=e['initial_identity']['state_key'], queries=e['queries'],
                        scale=scale, initial=e['initial_accel'], best=e['best_accel'],
                        normalized_gain=raw_gain/scale, cost=cost, reward=reward,
                        moved=e['queries']>1, raw_gain=raw_gain))
    summary={}
    for model in ('fastwam','groot_n15'):
        summary[model]={}
        for label, lo, hi in [('updates1-8',1,8),('updates9-16',9,16),('updates17-32',17,32),('updates33-64',33,64),('last6',59,64)]:
            rows=[e for e in episode_rows if e['model']==model and lo<=e['update']<=hi]
            moving=[e for e in rows if e['moved']]
            updates=[u for u in update_rows if u['model']==model and lo<=u['update']<=hi]
            summary[model][label]=dict(episodes=len(rows), moving_episodes=len(moving),
                mean_queries=mean(e['queries'] for e in rows), immediate_stop_fraction=mean(not e['moved'] for e in rows),
                mean_normalized_gain=mean(e['normalized_gain'] for e in rows),mean_cost=mean(e['cost'] for e in rows),
                mean_reward=mean(e['reward'] for e in rows),
                moving_positive_fraction=mean(e['reward']>0 for e in moving) if moving else None,
                moving_mean_reward=mean(e['reward'] for e in moving) if moving else None,
                moving_mean_normalized_gain=mean(e['normalized_gain'] for e in moving) if moving else None,
                moving_mean_cost=mean(e['cost'] for e in moving) if moving else None,
                moving_zero_gain_fraction=mean(e['raw_gain']==0 for e in moving) if moving else None,
                mean_scale=mean(u['scale'] for u in updates),
                mean_stop_probability=mean(u['stop_probability'] for u in updates),
                max_reconciliation_error=max(u['max_reward_reconciliation_error'] for u in updates))
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name, rows in [('episodes.csv',episode_rows),('updates.csv',update_rows)]:
        with (output/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sources[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'input-hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(build(args.root,args.output),indent=2))
