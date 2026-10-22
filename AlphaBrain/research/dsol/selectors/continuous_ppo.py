"""Complete-episode on-policy PPO with an injected, context-managed query backend.

No simulator imports. The frozen VLA is owned by the backend, never this optimizer.
One fixed actor collects 32 complete searches, then four PPO epochs update it.
"""
from dataclasses import asdict, dataclass
from contextlib import contextmanager, ExitStack
import json
import random
import sys
import time
import numpy as np
import torch
from ..data.local_search_rollouts import complete_update, minibatch_indices
from ..data.continuous_training_store import ContinuousTrainingStore, fingerprint, numpy_tree, rng_state, restore_rng, write_json


@contextmanager
def backend_rng():
    """Model construction/rendering must not consume the search policy's RNG."""
    state=rng_state()
    try:yield
    finally:restore_rng(state)


@dataclass(frozen=True)
class ContinuousPPOConfig:
    episodes_per_update: int = 32
    updates: int = 16
    seed: int = 0
    learning_rate: float = 3e-4
    n_epochs: int = 4
    batch_size: int = 64
    max_grad_norm: float = .5

    def __post_init__(self):
        if (any(type(v) is not int or v < 1 for v in (self.episodes_per_update,self.updates,self.n_epochs,self.batch_size))
                or type(self.seed) is not int or self.seed < 0
                or not np.isfinite([self.learning_rate,self.max_grad_norm]).all()
                or min(self.learning_rate,self.max_grad_norm)<=0):
            raise ValueError('Invalid PPO settings')


def batch_observations(observations):
    return {key:torch.as_tensor(np.stack([o[key] for o in observations])) for key in observations[0]}


class ContinuousPPO:
    def __init__(self, policy, config, schedule, backend, output, *, binding, resume=False):
        if len(schedule) != config.episodes_per_update*config.updates:
            raise ValueError('Exact predeclared episode schedule required')
        if [r['episode'] for r in schedule] != list(range(len(schedule))):
            raise ValueError('Schedule order/identity invalid')
        if any(p.device.type != 'cpu' for p in policy.parameters()):
            raise ValueError('This admitted trainer keeps the search actor/critic on CPU')
        self.policy,self.config,self.schedule,self.backend=policy,config,schedule,backend
        self.optimizer=torch.optim.Adam(policy.parameters(),lr=config.learning_rate,eps=1e-5)
        binding=dict(binding,engine='continuous_ppo_complete_episode_v2',
                     exploration_rule=policy.exploration_rule,config=asdict(config),
                     schedule_sha256=fingerprint(schedule))
        self.store=ContinuousTrainingStore(output,binding,resume=resume)
        self.update=self.cursor=self.queries=0
        self.pending=[]
        if resume:
            try:self._restore()
            except BaseException:
                self.store.close();raise
        else:
            random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
            try:self._checkpoint()
            except BaseException:
                self.store.close();raise

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _checkpoint(self, metrics=None):
        self.store.save_checkpoint(self.update, dict(policy=self.policy.state_dict(),optimizer=self.optimizer.state_dict(),
            update=self.update,cursor=self.cursor,queries=self.queries),metrics)

    def _restore(self):
        latest=json.loads((self.store.root/'latest.json').read_text());state=self.store.load(latest)
        self.policy.load_state_dict(state['policy'],strict=True);self.optimizer.load_state_dict(state['optimizer'])
        self.update,self.cursor,self.queries=state['update'],state['cursor'],state['queries']
        if self.cursor != self.update*self.config.episodes_per_update or latest['update'] != self.update:
            raise ValueError('Checkpoint cursor is not an update boundary')
        records=self.store.episodes()
        if not self.cursor <= len(records) <= min(len(self.schedule),self.cursor+self.config.episodes_per_update):
            raise ValueError('Episode/checkpoint cursor inconsistent')
        # Check all saved observations, including already-consumed ones, before resume.
        rng=state['rng']
        counted_queries=0
        for record in records:
            payload=self.store.load(record)
            metrics=record['metrics'];index=record['index']
            if (metrics['initial_identity']!=self.schedule[index] or metrics['episode']!=index
                    or metrics['policy_update']!=index//self.config.episodes_per_update
                    or payload['episode']['policy_update']!=metrics['policy_update']
                    or not 1<=metrics['queries']<=8 or metrics['scoring_trajectories']!=2*metrics['queries']):
                raise ValueError('Episode identity/accounting mismatch')
            complete_update([numpy_tree(payload['episode'])],episodes_per_update=1)
            if index<self.cursor:counted_queries+=metrics['queries']
            if record['index']>=self.cursor:
                self.pending.append(numpy_tree(payload['episode']));self.queries+=record['metrics']['queries'];rng=payload['rng']
        if counted_queries!=state['queries']:raise ValueError('Checkpoint query count mismatch')
        self.cursor=len(records);restore_rng(rng)

    def collect_one(self):
        if self.cursor>=len(self.schedule) or len(self.pending)>=self.config.episodes_per_update:
            raise ValueError('Schedule exhausted or an update is due')
        self.store.begin('episode',self.cursor)
        started=time.perf_counter()
        try:
            self.policy.eval();transitions=[];row=self.schedule[self.cursor]
            # Backend owns source lifetime; cannot auto-reset or silently reuse failed queries.
            stack=ExitStack()
            try:
                with backend_rng():
                    env,options=stack.enter_context(self.backend(row,self.store.root/'episodes'/f'{self.cursor:04d}-queries'))
                    obs,info=env.reset(options=options)
                if info['queries']!=1:raise ValueError('Initial acquisition must be charged once')
                while True:
                    before={k:np.array(v,copy=True) for k,v in obs.items()}
                    tensor=batch_observations([before])
                    with torch.no_grad():
                        action=self.policy.distribution(tensor).sample()
                        value=float(self.policy.value(tensor)[0])
                    stored=dict(stop=bool(action['stop'][0]),latent=action['latent'][0].numpy().copy(),
                                delta_deg=action['delta_deg'][0].numpy().copy())
                    with backend_rng():
                        obs,reward,done,truncated,info=env.step(dict(stop=int(stored['stop']),delta_deg=stored['delta_deg']))
                    transitions.append(dict(observation=before,action=stored,reward=float(reward),
                                            value=value,log_prob=float(action['log_prob'][0])))
                    if truncated or len(transitions)>6:raise ValueError('Unexpected truncation/decision overflow')
                    if done:break
                if info['status']!='COMPLETE' or not 1<=info['queries']<=8:
                    raise ValueError('Search did not finish within its declared query cap')
                episode=dict(status='COMPLETE',transitions=transitions,policy_update=self.update)
                complete_update([episode],episodes_per_update=1)
                metrics=dict(episode=self.cursor,policy_update=self.update,queries=info['queries'],
                    scoring_trajectories=info['queries']*2,decisions=len(transitions),reason=info['reason'],
                    total_reward=sum(t['reward'] for t in transitions),initial_query_cost=info['initial_query_cost'],
                    total_reward_including_initial=sum(t['reward'] for t in transitions)-info['initial_query_cost'],
                    best_accel=env.core.records[env.core.best][1].accel_mean,
                    initial_accel=env.core.records[0][1].accel_mean,selected_angles=info['selected_angles'],
                    initial_identity=row,source_kind=self.store.binding['source_kind'])
            finally:
                with backend_rng():stack.__exit__(*sys.exc_info())
            metrics['elapsed_seconds']=time.perf_counter()-started
            self.store.save_episode(self.cursor,episode,metrics)
            self.pending.append(episode);self.cursor+=1;self.queries+=info['queries'];self.store.finish()
            return metrics
        except BaseException as error:
            self.store.fail(error);raise

    def optimize(self):
        if len(self.pending)!=self.config.episodes_per_update:raise ValueError('Collect complete episode batch first')
        self.store.begin('update',self.update+1)
        started=time.perf_counter()
        try:
            if any(e['policy_update']!=self.update for e in self.pending):raise ValueError('Mixed on-policy versions')
            rows=complete_update(self.pending,episodes_per_update=self.config.episodes_per_update)
            self.policy.train();metrics=[];batch_sizes=[]
            for epoch in minibatch_indices(len(rows),seed=self.config.seed+self.update,
                    epochs=self.config.n_epochs,batch_size=self.config.batch_size):
                for indices in epoch:
                    selected=[rows[int(i)] for i in indices];obs=batch_observations([r['observation'] for r in selected])
                    stop=torch.tensor([r['action']['stop'] for r in selected],dtype=torch.bool)
                    latent=torch.tensor(np.stack([r['action']['latent'] for r in selected]))
                    args=[torch.tensor([r[key] for r in selected],dtype=torch.float64) for key in ('old_log_prob','advantage','returns')]
                    self.optimizer.zero_grad(set_to_none=True)
                    loss,stats=self.policy.objective(obs,stop,latent,*args,with_metrics=True)
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.config.max_grad_norm,error_if_nonfinite=True)
                    self.optimizer.step()
                    if any(not torch.isfinite(p).all() for p in self.policy.parameters()):raise ValueError('Nonfinite updated parameter')
                    metrics.append(dict(loss=float(loss.detach()),gradient_norm=float(norm),**stats));batch_sizes.append(len(selected))
            self.update+=1
            record=dict(update=self.update,completed_episodes=self.cursor,queries=self.queries,
                elapsed_seconds=time.perf_counter()-started,
                scoring_trajectories=self.queries*2,transitions=len(rows),minibatch_sizes=batch_sizes,
                means={k:float(np.average([m[k] for m in metrics],weights=batch_sizes)) for k in metrics[0]},
                source_kind=self.store.binding['source_kind'])
            self._checkpoint(record);self.pending=[];self.store.finish()
            return record
        except BaseException as error:
            self.store.fail(error);raise

    def run(self, *, through_update=None):
        target=self.config.updates if through_update is None else through_update
        if not self.update<=target<=self.config.updates:raise ValueError('Update limit outside declared schedule')
        while self.update<target:
            while len(self.pending)<self.config.episodes_per_update:self.collect_one()
            self.optimize()
        # Pause only at clean boundaries. Never select a checkpoint on evaluation outcomes.
        status='COMPLETE' if self.update==self.config.updates else 'PAUSED_AT_UPDATE_BOUNDARY'
        record=dict(status=status,updates=self.update,episodes=self.cursor,queries=self.queries,
                    scoring_trajectories=self.queries*2,source_kind=self.store.binding['source_kind'])
        write_json(self.store.root/'status.json',record)
        return record
