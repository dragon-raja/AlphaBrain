"""Final-only CPU export with immutable provenance and deployment controls."""
import json
from contextlib import closing
from pathlib import Path
import torch

from ..data.continuous_training_store import ContinuousTrainingStore,atomic_file,write_json,digest
from .continuous_search_policy import ContinuousActorCritic
from .continuous_ppo import batch_observations


def export_final(training,output,*,visual_dim=4608,episodes=512,updates=16):
    root=Path(training);target=Path(output)
    read=lambda name:json.loads((root/name).read_text())
    status=read('status.json');latest=read('latest.json')
    if (status['status']!='COMPLETE' or status['episodes']!=episodes or status['updates']!=updates
            or latest['update']!=updates or target.exists()):
        raise ValueError('Only a completed final update may be exported to a new directory')
    # Acquire the existing writer lock and reject incomplete/failed transactions.
    with closing(ContinuousTrainingStore(root,read('binding.json'),resume=True)) as store:
        state=store.load(latest);records=store.episodes()
        if (state['update']!=updates or state['cursor']!=episodes or len(records)!=episodes
                or state['queries']!=status['queries']):
            raise ValueError('Final checkpoint and completed coverage differ')
        if sum(r['metrics']['queries'] for r in records)!=status['queries']:
            raise ValueError('Final query accounting mismatch')
        if (root/'budget/state.json').exists():
            budget=read('budget/state.json')
            if budget['failed'] or budget['reserved'] or budget['spent']!=2*status['queries']:
                raise ValueError('Final global budget is not clean')
        policy=ContinuousActorCritic(visual_dim).eval();policy.load_state_dict(state['policy'],strict=True)
        controls=[]
        for r in records:
            value=store.load(r)
            if len(controls)<16:
                obs=batch_observations([value['episode']['transitions'][0]['observation']])
                with torch.inference_mode():action=policy.distribution(obs).sample(deterministic=True)
                controls.append(dict(observation=obs,action={k:action[k] for k in ('stop','delta_deg','log_prob')}))
        target.mkdir(parents=True,exist_ok=False)
        atomic_file(target/'actor.pt',lambda f:torch.save(dict(state_dict=policy.state_dict(),
            update=updates,episodes=episodes,visual_dim=visual_dim),f))
        atomic_file(target/'controls.pt',lambda f:torch.save(controls,f))
        receipt=dict(status='FROZEN_FINAL_ONLY',checkpoint_sha256=latest['sha256'],
            exporter_sha256=digest(__file__),torch_version=str(torch.__version__),
            status_sha256=digest(root/'status.json'),binding_sha256=digest(root/'binding.json'),
            episodes=episodes,updates=updates,queries=status['queries'],model_calls=0,
            control_rule='first_16_committed_episode_first_observations',
            actor_sha256=digest(target/'actor.pt'),controls_sha256=digest(target/'controls.pt'))
        write_json(target/'receipt.json',receipt)
        return receipt
