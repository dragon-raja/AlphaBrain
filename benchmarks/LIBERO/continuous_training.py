"""Online continuous-PPO training adapter. Default CLI only validates inputs.

The optimizer lives in research/dsol; this module owns real rendering/scoring.
There is no cache-backed scoring or task-success reward in this adapter.
"""
import argparse
from contextlib import contextmanager
import json
import os
import time
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.execution.manifest import file_digest as sha
from AlphaBrain.research.dsol.data.local_search_plan import validate_schedules


def validate_inputs(path):
    """Read-only identity/split check. Does not import a model or simulator."""
    cfg=read(path)
    if cfg['schema']!='continuous_ppo_launch_v1' or cfg['source_kind']!='LIVE_RENDER_BATCH1_TWO_NOISES':
        raise ValueError('Continuous live training manifest required; no catalog replay')
    for filename,digest in cfg['input_hashes'].items():
        if sha(filename)!=digest:raise ValueError('Bound training input changed: '+filename)
    required=[cfg[k] for k in ('design','training_schedule','evaluation_schedule','initial_checkpoint','sim_config')]
    if any(p not in cfg['input_hashes'] for p in required):raise ValueError('Unbound training input')
    plan=read(cfg['design']); sim=read(cfg['sim_config'])
    if (cfg['trajectory_cap']!=8192 or cfg['cpu_threads']!=2 or cfg['actor_device']!='cpu'
            or not sim['continuous_camera'] or 'outcome_matrix' in sim or 'native' in sim):
        raise ValueError('Wrong resource budget/backend or cached labels/scores supplied')
    manifests={k:read(v)['records'] for k,v in cfg['identity_inputs'].items()}
    if any(v not in cfg['input_hashes'] for v in cfg['identity_inputs'].values()):
        raise ValueError('Unbound split identities')
    for p in (sim['inputs'],sim['anchors'],str(Path(sim['models']['canonical']['path'])/'model.safetensors')):
        if p not in cfg['input_hashes']:raise ValueError('Unbound model/state/noise asset')
    budget=validate_schedules(plan,read(cfg['training_schedule']),read(cfg['evaluation_schedule']),manifests)
    if (plan['schema']!='continuous_search_plan_v1' or plan['training']['episodes']!=512
            or plan['training']['updates']!=16 or plan['training']['episodes_per_update']!=32
            or plan['training']['seed']!=0 or plan['training']['maximum_decisions_per_episode']!=6
            or plan['training']['task_execution'] or plan['training']['success_labels']
            or plan['search']['query_budget']!=8 or plan['search']['score_noise_members']!=[0,1]
            or plan['actor']['score_scale']!=0.16535300100378683):
        raise ValueError('Frozen continuous training contract changed')
    ppo=plan['training']['ppo']
    for key,value in dict(gamma=1.,gae_lambda=.95,clip_range=.2,ent_coef=.01,vf_coef=.5).items():
        if ppo[key]!=value:raise ValueError('Unsupported PPO coefficient: '+key)
    return dict(status='PASS_INPUTS_ONLY',budget=budget,training_started=False),cfg,plan


class DurableScoring:
    """Count attempts before inference, including failures and resumed processes."""
    def __init__(self, runtime, root, expected_calls, limit):
        self.runtime,self.path,self.limit=runtime,Path(root)/'model-attempts.jsonl',limit
        rows=[json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        if [r['index'] for r in rows]!=list(range(1,len(rows)+1)) or len(rows)!=expected_calls:
            raise ValueError('Model ledger and completed episodes differ; no replay')
        self.calls=expected_calls
        self.runtime.calls=expected_calls
        if self.runtime.limit!=limit:raise ValueError('Runtime and durable budget differ')
        self.episode=None

    def infer(self,example,noise,identity,capture=True):
        from AlphaBrain.research.dsol.data.flow_noise import tensor_sha256
        from .local_search_rollout import input_identity
        if self.calls>=self.limit:raise ValueError('Global scoring budget exhausted')
        self.calls+=1
        address=dict(identity,stage='continuous_ppo_training',episode=self.episode)
        row=dict(index=self.calls,unix_time=time.time(),identity=address,input_sha256=input_identity(example),noise_sha256=tensor_sha256(noise))
        with self.path.open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        value=self.runtime.infer(example,noise,address,capture=capture)
        if self.runtime.calls!=self.calls:raise ValueError('Runtime/model ledger divergence')
        return value


class LiveEpisodes:
    """Lazy persistent VLA; fresh simulator per search; explicit fixed state/noise."""
    def __init__(self,cfg,plan,output,*,expected_calls=0):
        self.cfg,self.plan,self.output=cfg,plan,Path(output)
        self.expected_calls=expected_calls;self.scoring=None

    @contextmanager
    def __call__(self,row,output):
        from .search_runtime import PolicyRuntime
        from .search_live import simulator
        from .local_view_runtime import RenderedViewSource
        from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
        from AlphaBrain.research.dsol.acquisition.continuous_search_gym import ContinuousSearchGym
        from AlphaBrain.research.dsol.data.continuous_training_store import write_json
        output.mkdir(exist_ok=False)
        sim=read(self.cfg['sim_config'])
        if self.scoring is None:
            runtime=PolicyRuntime(sim,self.output,self.cfg['trajectory_cap'])
            self.scoring=DurableScoring(runtime,self.output,self.expected_calls,self.cfg['trajectory_cap'])
        self.scoring.episode=row['episode']
        with np.load(sim['anchors'],allow_pickle=False) as z:
            noises=z['first_score_noise'][row['initial'],:2].copy()
        with simulator(self.cfg['sim_config'],sim,row['initial'],output) as request:
            source=RenderedViewSource(request,self.scoring,noises,output)
            source.move_angles(row['angles_deg']);first=source.query()
            original=source.query;pending=[first]
            source.query=lambda: pending.pop() if pending else original()
            core=ContinuousSearch(source.mount,source,initial_angles=row['angles_deg'],
                physical_state_id=source.physical,wrist=source.last_values[0]['wrist'],
                score_scale=self.plan['actor']['score_scale'],seed=row['proposal_seed'])
            yield ContinuousSearchGym(4608),{'session':core}
            if core.info()['queries']!=source.queries:raise ValueError('Core/source accounting mismatch')
            write_json(output/'search.json',dict(info=core.info(),events=core.events,
                physical_state_id=source.physical,noise_members=2,manipulation_actions=0,
                query_files=[dict(path=p.name,sha256=sha(p)) for p in sorted(output.glob('query-*.npz'))]))


def run(config_path,output,*,execute=False,resume=False):
    # Approval guard precedes input/model loading. Draft manifests remain inert.
    cfg=read(config_path)
    if not execute or cfg.get('authorization')!='EXPLICIT_TRAINING_ONLY':
        raise ValueError('Prepared launch is not authorization to train')
    _,cfg,plan=validate_inputs(config_path)
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO,ContinuousPPOConfig
    from AlphaBrain.research.dsol.data.continuous_training_store import write_json
    torch.set_num_threads(cfg['cpu_threads'])
    payload=torch.load(cfg['initial_checkpoint'],map_location='cpu',weights_only=True)
    if payload['visual_dim']!=4608 or payload['optimizer_updates']!=0:raise ValueError('Admitted initialization required')
    policy=ContinuousActorCritic(4608)
    policy.load_state_dict(payload['state_dict'],strict=True)
    settings=plan['training'];p=settings['ppo']
    config=ContinuousPPOConfig(episodes_per_update=settings['episodes_per_update'],updates=settings['updates'],
        seed=settings['seed'],learning_rate=p['learning_rate'],n_epochs=p['n_epochs'],
        batch_size=p['batch_size'],max_grad_norm=p['max_grad_norm'])
    backend=LiveEpisodes(cfg,plan,output)
    with ContinuousPPO(policy,config,read(cfg['training_schedule']),backend,output,
            binding=dict(source_kind=cfg['source_kind'],launch_config_sha256=sha(config_path),
                         source_commit=cfg['source_commit'],cpu_threads=cfg['cpu_threads']),resume=resume) as trainer:
        backend.expected_calls=2*trainer.queries
        if trainer.update == 0:
            trainer.run(through_update=1)
            trainer.store.begin('first_batch_gate',1)
            try:
                from .continuous_training_gate import audit_first_batch
                gate=audit_first_batch(trainer)
                write_json(Path(output)/'first-batch-gate.json',gate)
                trainer.store.finish()
            except BaseException as error:
                trainer.store.fail(error);raise
        elif not (Path(output)/'first-batch-gate.json').exists():
            raise ValueError('Cannot resume beyond first update without its gate')
        result=trainer.run()
        result.update(config_sha256=sha(config_path),calls=2*trainer.queries,manipulation_actions=0)
        write_json(Path(output)/'status.json',result)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--output');p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not a.execute:print(json.dumps(validate_inputs(a.config)[0],indent=2))
    else:
        if not a.output:p.error('--output required for execution')
        print(json.dumps(run(a.config,a.output,execute=True,resume=a.resume),indent=2))
