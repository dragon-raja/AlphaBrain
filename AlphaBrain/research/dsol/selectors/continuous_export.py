"""Final-only CPU export with immutable provenance and deployment controls."""
import json
import shutil
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
    return _export(root,target,status,latest,visual_dim,episodes,updates,
                   status_path=root/'status.json',receipt_status='FROZEN_FINAL_ONLY')


def export_stage(training,output,*,stage_path,stage_sha256,visual_dim=4608):
    """Freeze an authorized completed prefix without changing learner binding.

    Preserve the full optimizer/RNG checkpoint separately from the deployable
    actor. The original 256-update schedule remains explicitly resumable.
    """
    from ..data.training_stage import load_stage
    stage=load_stage(stage_path,stage_sha256)
    root=Path(training);target=Path(output)
    cfg_spec=stage['runs'].get(root.name)
    if cfg_spec is None:raise ValueError('Unknown stage run')
    cfg=json.loads(Path(cfg_spec['path']).read_text())
    binding=json.loads((root/'binding.json').read_text())
    if Path(cfg['output']).resolve()!=root.resolve() or binding.get('config_sha256')!=cfg_spec['sha256']:
        raise ValueError('Stage run binding mismatch')
    publication=json.loads((root/'publication.json').read_text())
    latest=json.loads((root/'latest.json').read_text())
    updates,episodes=stage['updates'],stage['episodes_per_run']
    if (publication['status'] not in ('PAUSED_AT_UPDATE_BOUNDARY','COMPLETE')
            or publication['version']!=updates or publication['episodes']!=episodes
            or publication['run']!=root.name or latest['update']!=updates or target.exists()
            or digest(publication['policy'])!=publication['policy_sha256']):
        raise ValueError('Stage publication is not the declared complete boundary')
    return _export(root,target,publication,latest,visual_dim,episodes,updates,
                   status_path=root/'publication.json',receipt_status='FROZEN_STAGE_ENDPOINT',
                   stage_proof=dict(path=str(stage_path),sha256=stage_sha256))


def _export(root,target,status,latest,visual_dim,episodes,updates,*,status_path,receipt_status,stage_proof=None):
    read=lambda name:json.loads((root/name).read_text())
    # Acquire the existing writer lock and reject incomplete/failed transactions.
    with closing(ContinuousTrainingStore(root,read('binding.json'),resume=True)) as store:
        state=store.load(latest);records=store.episodes()
        if stage_proof is not None and not {'optimizer','rng'}.issubset(state):
            raise ValueError('Resumable stage requires full optimizer and RNG state')
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
        policy.requires_grad_(False)
        controls=[]
        previous_threads=torch.get_num_threads()
        try:
            # Match the admitted evaluation loader; otherwise controls can
            # depend on the export host's default CPU reduction/thread setup.
            torch.set_num_threads(2)
            for r in records:
                value=store.load(r)
                if len(controls)<16:
                    obs=batch_observations([value['episode']['transitions'][0]['observation']])
                    with torch.inference_mode():action=policy.distribution(obs).sample(deterministic=True)
                    controls.append(dict(observation=obs,action={k:action[k] for k in ('stop','delta_deg','log_prob')}))
        finally:torch.set_num_threads(previous_threads)
        target.mkdir(parents=True,exist_ok=False)
        atomic_file(target/'actor.pt',lambda f:torch.save(dict(state_dict=policy.state_dict(),
            update=updates,episodes=episodes,visual_dim=visual_dim),f))
        atomic_file(target/'controls.pt',lambda f:torch.save(controls,f))
        receipt=dict(status=receipt_status,checkpoint_sha256=latest['sha256'],
            exporter_sha256=digest(__file__),torch_version=str(torch.__version__),
            status_sha256=digest(status_path),binding_sha256=digest(root/'binding.json'),
            episodes=episodes,updates=updates,queries=status['queries'],model_calls=0,
            control_rule='first_16_committed_episode_first_observations',
            control_cpu_threads=2,control_parameters_require_grad=False,
            actor_sha256=digest(target/'actor.pt'),controls_sha256=digest(target/'controls.pt'))
        if stage_proof is not None:
            def copy_checkpoint(stream):
                with (root/latest['path']).open('rb') as source:shutil.copyfileobj(source,stream)
            atomic_file(target/'resume-checkpoint.pt',copy_checkpoint)
            if digest(target/'resume-checkpoint.pt')!=latest['sha256']:
                raise ValueError('Pinned resume checkpoint hash mismatch')
            receipt.update(stage=stage_proof,resume_checkpoint_sha256=latest['sha256'],
                           original_training_root=str(root),original_latest=latest,
                           continuation_requires_explicit_authorization=True)
        write_json(target/'receipt.json',receipt)
        return receipt
