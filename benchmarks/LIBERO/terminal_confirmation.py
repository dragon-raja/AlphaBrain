"""Reserve two queries and hide independent terminal scores from the actor."""
from copy import deepcopy
import numpy as np
from AlphaBrain.research.dsol.data.confirmed_reward import RECORD
from AlphaBrain.research.dsol.data.flow_noise import tensor_sha256
from .local_search_rollout import input_identity


def finish_and_confirm(core,noises):
    if core.status=='FAILED':raise ValueError('Confirmation previously failed; explicit audit required')
    try:return _finish_and_confirm(core,noises)
    except BaseException:
        core.status='FAILED';core.source.failed=True
        raise


def _finish_and_confirm(core,noises):
    if hasattr(core,'terminal_confirmation'):return core.terminal_confirmation
    if core.status=='ACTIVE' and core.queries>=5:core._finish('confirmation_reserved')
    if core.status!='COMPLETE':return None
    source=core.source;held=np.asarray(noises)
    if held.shape!=(2,10,7) or not np.isfinite(held).all():raise ValueError('Two explicit held-out noise tensors required')
    search_hashes=[tensor_sha256(v) for v in source.noises];held_hashes=[tensor_sha256(v) for v in held]
    if len(set(search_hashes+held_hashes))!=4:raise ValueError('Search/confirmation noise overlap')
    selected=source.last_example;selected_hash=input_identity(selected)
    # First query is necessarily the perturbed starting observation.
    initial=next(iter(source.references.values()))[0];initial_hash=input_identity(initial)
    base=dict(schema=RECORD,hidden_from_actor=True,search_noise_hashes=search_hashes,
        confirmation_noise_hashes=held_hashes,initial_input_sha256=initial_hash,selected_input_sha256=selected_hash)
    if initial_hash==selected_hash:
        result=dict(base,skipped_same_input=True,query_numbers=[],initial_scores=None,selected_scores=None,queries=core.queries)
    else:
        if core.queries>6:raise ValueError('Confirmation reservation lost before termination')
        old_noises,old_references=source.noises,source.references
        source.noises=held.copy();source.references={}
        selected_angles=core.current.copy();first=core.queries+1
        try:
            core._move(core.initial);core._query('confirmation_initial')
            if input_identity(source.last_example)!=initial_hash:raise ValueError('Confirmation initial input changed')
            a=[float(v['score']) for v in source.last_values]
            core._move(selected_angles);core._query('confirmation_selected')
            if input_identity(source.last_example)!=selected_hash:raise ValueError('Confirmation selected input changed')
            b=[float(v['score']) for v in source.last_values]
        finally:source.noises,source.references=old_noises,old_references
        result=dict(base,skipped_same_input=False,query_numbers=[first,first+1],initial_scores=a,selected_scores=b,queries=core.queries)
    if core.queries>8 or core.queries!=source.queries:raise ValueError('Confirmed query budget diverged')
    core.terminal_confirmation=deepcopy(result)
    return result


class ConfirmedEnv:
    def __init__(self,env,noises):self.env,self.noises=env,np.array(noises,copy=True)
    @property
    def core(self):return self.env.core
    def reset(self,**kwargs):return self.env.reset(**kwargs)
    def step(self,action):
        before_queries=self.core.queries;before_best=self.core.records[self.core.best][1].accel_mean
        obs,reward,done,truncated,info=self.env.step(action)
        result=finish_and_confirm(self.core,self.noises)
        reward=(before_best-self.core.records[self.core.best][1].accel_mean)/self.core.reward_scale-self.core.config.query_cost*(self.core.queries-before_queries)
        if result is not None:
            self.terminal_confirmation=result
            obs=self.env._observation(self.core.observation());done=True;info=self.core.info()
        return obs,reward,done,truncated,info
