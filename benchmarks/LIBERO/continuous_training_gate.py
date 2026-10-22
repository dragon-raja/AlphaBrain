"""Read committed first-batch evidence before allowing the next real query."""
import json
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.execution.manifest import file_digest


def audit_first_batch(trainer):
    import torch
    root=trainer.store.root
    if trainer.cursor!=32 or trainer.update!=1 or trainer.pending:
        raise ValueError('First gate requires exactly 32 completed episodes and one update')
    records=trainer.store.episodes()
    if len(records)!=32:raise ValueError('Incorrect first-batch coverage')
    attempts=[json.loads(line) for line in (root/'model-attempts.jsonl').read_text().splitlines()]
    if (len(attempts)!=2*trainer.queries or len(attempts)>512
            or [r['index'] for r in attempts]!=list(range(1,len(attempts)+1))):
        raise ValueError('First-batch model ledger mismatch')
    expected=[];files=0
    for record in records:
        i=record['index'];m=record['metrics'];directory=root/'episodes'/f'{i:04d}-queries'
        control=json.loads((directory/'search.json').read_text())
        if (m['policy_update']!=0 or m['initial_identity']!=trainer.schedule[i]
                or control['manipulation_actions']!=0 or control['info']['status']!='COMPLETE'
                or control['info']['queries']!=m['queries'] or len(control['query_files'])!=m['queries']):
            raise ValueError('Search coverage/identity mismatch')
        for q,ref in enumerate(control['query_files'],1):
            path=directory/ref['path']
            if path.name!=f'query-{q:02d}.npz' or file_digest(path)!=ref['sha256']:
                raise ValueError('Query evidence changed')
            with np.load(path,allow_pickle=False) as z:
                for member in (0,1):
                    if z[f'actions_{member}'].shape!=(10,7) or z[f'external_{member}'].shape!=(4608,):
                        raise ValueError('Full action/feature recording missing')
                    for key in ('actions','external','wrist','trace','score'):
                        if not np.isfinite(z[f'{key}_{member}']).all():raise ValueError('Nonfinite model evidence')
                    expected.append((i,q,member))
            files+=1
    actual=[(r['identity']['episode'],r['identity']['query'],r['identity']['member']) for r in attempts]
    if actual!=expected:raise ValueError('Query/member/episode order mismatch')
    old=torch.load(root/'checkpoints/update-000.pt',map_location='cpu',weights_only=True)['policy']
    new=trainer.policy.state_dict()
    actor_changed=any(not torch.equal(v,old[k]) for k,v in new.items() if k.startswith(('actor_history.','mean_head.','stop_head.')))
    critic_changed=any(not torch.equal(v,old[k]) for k,v in new.items() if k.startswith(('critic_history.','value_head.')))
    if not actor_changed or not critic_changed:raise ValueError('Actor/critic did not both update')
    if any(not torch.isfinite(v).all() for v in new.values()):raise ValueError('Nonfinite checkpoint')
    return dict(status='PASS_FIRST_REAL_BATCH',episodes=32,updates=1,queries=trainer.queries,
                calls=len(attempts),query_evidence_files=files,actor_changed=True,critic_changed=True,
                manipulation_actions=0,success_labels_used=False,extra_admission_calls=0)
