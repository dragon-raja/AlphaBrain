"""Host-independent rollout RNG and verified, ordered PPO batch ingestion."""
import random
import numpy as np
import torch
from ..execution.manifest import digest, file_digest
from ..data.continuous_training_store import numpy_tree, rng_state, restore_rng
from ..data.local_search_rollouts import complete_update
from .continuous_ppo import collect_episode, batch_observations

RNG_RULE='per_source_episode_sha256_v1'


def episode_seed(seed, row):
    return int(digest([RNG_RULE,seed,row.get('source_episode',row['episode']),row['proposal_seed']])[:8],16)


def seeded_episode(policy, backend, row, output, *, version, seed, source_kind):
    old=rng_state();n=episode_seed(seed,row)
    try:
        random.seed(n);np.random.seed(n);torch.manual_seed(n)
        return collect_episode(policy,backend,row,output,policy_update=version,source_kind=source_kind)
    finally:restore_rng(old)


def validate_rollout(policy, payload, row, version):
    episode=numpy_tree(payload['episode']);m=payload['metrics']
    if (episode['policy_update']!=version or m['policy_update']!=version
            or m['initial_identity']!=row or m['episode']!=row['episode']
            or not 1<=m['queries']<=8 or m['scoring_trajectories']!=m['queries']*2
            or m['decisions']!=len(episode['transitions'])):
        raise ValueError('Wrong episode/actor version/identity/accounting')
    complete_update([episode],episodes_per_update=1)
    policy.eval()
    for t in episode['transitions']:
        obs=batch_observations([t['observation']]);z=torch.as_tensor(t['action']['latent'][None])
        stop=torch.tensor([t['action']['stop']],dtype=torch.bool)
        with torch.no_grad():
            d=policy.distribution(obs);prob=float(d.log_prob(stop,z)[0]);value=float(policy.value(obs)[0])
            delta=np.zeros(2) if stop[0] else d.transform(z)[0].numpy()
        if (prob!=t['log_prob'] or value!=t['value'] or not np.array_equal(delta,t['action']['delta_deg'])):
            raise ValueError('Rollout probability/value/action differs from frozen policy')
    return episode,m


def ingest(trainer, records):
    """All expected versions verified BEFORE any ordered commit or optimizer step."""
    if len(records)!=trainer.config.episodes_per_update:raise ValueError('Incomplete rollout barrier')
    start=trainer.update*trainer.config.episodes_per_update
    expected=trainer.schedule[start:start+trainer.config.episodes_per_update]
    if set(records)!={r['episode'] for r in expected}:raise ValueError('Wrong rollout coverage')
    verified=[]
    for row in expected:
        ref=records[row['episode']]
        if file_digest(ref['path'])!=ref['sha256']:raise ValueError('Worker rollout artifact changed')
        payload=torch.load(ref['path'],map_location='cpu',weights_only=True)
        verified.append(validate_rollout(trainer.policy,payload,row,trainer.update))
    # Committed episodes after a crash are loaded by ContinuousPPO._restore;
    # skip only that verified prefix, never resample or count it twice.
    for episode,m in verified[trainer.cursor-start:]:
        trainer.store.begin('episode',trainer.cursor)
        try:
            trainer.store.save_episode(trainer.cursor,episode,m)
            trainer.pending.append(episode);trainer.cursor+=1;trainer.queries+=m['queries']
            trainer.store.finish()
        except BaseException as error:
            trainer.store.fail(error);raise
