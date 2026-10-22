"""Audit complete training receipts and export the predetermined final policy."""
import math
from pathlib import Path
from .artifacts import read, sha
from .continuous_training_store import write_json


def freeze(training, output):
    import torch
    from ..selectors.continuous_search_policy import ContinuousActorCritic
    root, output = Path(training), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    status, latest = read(root/'status.json'), read(root/'latest.json')
    if (status['status']!='COMPLETE' or status['episodes']!=512 or status['updates']!=16
            or status['manipulation_actions']!=0 or latest['update']!=16
            or (root/'failure.json').exists() or (root/'inflight.json').exists()):
        raise ValueError('Training is not cleanly complete')
    if sha(root/latest['path'])!=latest['sha256']:
        raise ValueError('Final checkpoint digest mismatch')
    calls = [__import__('json').loads(x) for x in (root/'model-attempts.jsonl').read_text().splitlines()]
    if [c['index'] for c in calls]!=list(range(1,status['calls']+1)):
        raise ValueError('Attempt ledger incomplete')
    receipts = sorted((root/'episodes').glob('*.json'))
    if len(receipts)!=512:raise ValueError('Incomplete episode receipts')
    queries = 0; controls = []
    torch.set_num_threads(2)
    state = torch.load(root/latest['path'],map_location='cpu',weights_only=False)
    if (state['update'],state['cursor'],state['queries'])!=(16,512,status['queries']):
        raise ValueError('Final checkpoint counters inconsistent')
    if any(not torch.isfinite(v).all() for v in state['policy'].values()):
        raise ValueError('Nonfinite final parameters')
    actor = ContinuousActorCritic(4608).eval()
    actor.load_state_dict(state['policy'],strict=True);actor.requires_grad_(False)
    for i, path in enumerate(receipts):
        receipt=read(path); metrics=receipt['metrics']
        if (receipt['index']!=i or metrics['episode']!=i or metrics['policy_update']!=i//32
                or metrics['scoring_trajectories']!=2*metrics['queries']
                or not 1<=metrics['queries']<=8 or sha(root/receipt['path'])!=receipt['sha256']):
            raise ValueError('Episode identity/accounting/hash mismatch')
        search_root=root/'episodes'/f'{i:04d}-queries'
        search=read(search_root/'search.json')
        if len(search['query_files'])!=metrics['queries'] or search['manipulation_actions']!=0:
            raise ValueError('Raw search coverage mismatch')
        for query in search['query_files']:
            if sha(search_root/query['path'])!=query['sha256']:
                raise ValueError('Raw query digest mismatch')
        queries+=metrics['queries']
        if i%32==0:
            episode=torch.load(root/receipt['path'],map_location='cpu',weights_only=False)['episode']
            obs={k:v[None] for k,v in episode['transitions'][0]['observation'].items()}
            with torch.inference_mode():action=actor.distribution(obs).sample(deterministic=True)
            controls.append(dict(episode=i,observation=obs,action=action))
    if queries!=status['queries'] or status['calls']!=2*queries or status['calls']>8192:
        raise ValueError('Training total budget/ledger mismatch')
    for i in range(1,17):
        update=read(root/'updates'/f'{i:03d}.json')
        if update['completed_episodes']!=32*i or any(not math.isfinite(v) for v in update['means'].values()):
            raise ValueError('Incomplete/nonfinite optimizer updates')
    torch.save(dict(state_dict=state['policy'],visual_dim=4608,update=16,episodes=512),output/'actor.pt')
    torch.save(controls,output/'deployment-controls.pt')
    receipt=dict(status='PASS_COMPLETE_TRAINING',episodes=512,updates=16,queries=queries,calls=len(calls),
        raw_query_hashes_verified=queries,episode_hashes_verified=512,controls=len(controls),
        source_checkpoint_sha256=latest['sha256'],selection='predetermined_final_update16_only',
        actor_sha256=sha(output/'actor.pt'),controls_sha256=sha(output/'deployment-controls.pt'))
    write_json(output/'receipt.json',receipt)
    return receipt
