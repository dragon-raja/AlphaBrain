"""Actual-entry same-input and full-operation admission for prefix-only changes.

All calls are charged by the existing EvaluationCalls authority boundary. RLDX
has an independently maintained sensor history and direct native reference.
"""
from pathlib import Path
import json
import random
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.data.score_prefix import contract
from AlphaBrain.research.dsol.metrics.score_prefix import score
from AlphaBrain.research.dsol.execution.manifest import digest


def rng_state():
    import torch
    return (torch.get_rng_state(),torch.cuda.get_rng_state_all(),random.getstate(),np.random.get_state())


def restore(state):
    import torch
    torch.set_rng_state(state[0]);torch.cuda.set_rng_state_all(state[1]);random.setstate(state[2]);np.random.set_state(state[3])


def equal_rng(a,b):
    import torch
    return (torch.equal(a[0],b[0]) and len(a[1])==len(b[1]) and all(torch.equal(x,y) for x,y in zip(a[1],b[1]))
            and a[2]==b[2] and a[3][0]==b[3][0] and np.array_equal(a[3][1],b[3][1]) and a[3][2:]==b[3][2:])


def compare(a,b):
    for k in ('actions','normalized_actions','trace','full_actions','initial_noise','native_input_sha256'):
        if k in a or k in b:
            if k not in a or k not in b or not np.array_equal(a[k],b[k]):raise ValueError('Prefix changed native '+k)
    if 'native_features' in a or 'native_features' in b:
        from AlphaBrain.research.dsol.data.native_features import from_wire
        aa=from_wire(a['native_features']);bb=from_wire(b['native_features'])
        if aa.layout!=bb.layout or any(not np.array_equal(x,y) for x,y in zip(aa.values,bb.values)):
            raise ValueError('Prefix changed native cognition')


class RLDXReference:
    """Unmodified native model called with independent per-action history."""
    def __init__(self,wrapped,history):self.wrapped=wrapped;self.history=history
    @property
    def calls(self):return self.wrapped.calls
    def infer(self,example,noise,identity,*,capture=False):
        from .rldx_search_runtime import noise_seed
        from .controlled_rldx_operation import input_stamp
        raw=self.wrapped
        if capture or raw.failed or raw.calls>=raw.limit:raise ValueError('Invalid independent native reference')
        stamp=input_stamp(self.history);raw.calls+=1;raw.failed=True;seed=noise_seed(noise)
        value=raw.native.infer(self.history,seed,identity,instrument=True,features=False)
        with (raw.output/'calls.jsonl').open('a') as f:
            f.write(json.dumps(dict(index=raw.calls,**identity,seed=str(seed)))+'\n');f.flush()
        raw.failed=False
        return dict(actions=value['actions'],normalized_actions=value['normalized_actions'],seed=seed,native_input_sha256=stamp,trace=value['velocities'][:,0,:8,:7].copy())


def run(c,cell,b,output,guard,pool):
    from .search_runtime import create_policy_runtime
    from .score_prefix_runtime import PrefixRuntime
    from .search_live import simulator
    from .continuous_evaluation import EvaluationCalls,append_event
    from .local_search_rollout import input_identity,rollout_local_episode,rollout_groot_episode
    from .controlled_rldx_operation import move_to
    from .rldx_search_runtime import rollout_rldx_episode
    from .rldx_runtime import LiberoHistory
    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
    from AlphaBrain.research.dsol.data.native_features import from_wire
    sim=read(b['sim_config']);family=cell['family'];runtime=create_policy_runtime(sim,output,b['limits']['score'])
    raw=runtime.runtime;p3=PrefixRuntime(raw,contract(family,3));full=PrefixRuntime(raw,contract(family,10 if family=='pi05' else 4))
    calls=EvaluationCalls(raw,output,dict(score=b['limits']['score']),guard=guard);calls.stage='score'
    open_sim=simulator if pool is None else pool.session
    row=b['rows'][0];folder=output/'sensors';folder.mkdir()
    with open_sim(b['sim_config'],sim,row['initial'],folder) as query:example=move_to(query,row.get('angles_deg',[0.,0.]))
    with np.load(sim['anchors'],allow_pickle=False) as z:noise=z['first_score_noise'][row['initial'],0].copy()
    if getattr(raw,'native_needs_warmup',False):calls.infer(example,noise,dict(phase='warmup'),capture=False,warmup=True)
    request=dict(query_id='same-native-input',input_sha256=input_identity(example),history_sha256=digest(dict(admission='reset')))
    state=rng_state();values=[];states=[]
    for label,target in [('legacy',raw),('p3',p3),('full',full)]:
        restore(state);calls.runtime=target
        v=calls.infer(example,noise,dict(phase='numerical',variant=label),capture=True,**({'native_request':request} if family!='pi05' else {}))
        values.append(v);states.append(rng_state())
    compare(values[0],values[1]);compare(values[1],values[2])
    if not equal_rng(states[0],states[1]) or not equal_rng(states[1],states[2]):raise ValueError('Prefix changed RNG')
    if values[0]['score']!=values[1]['score'] or values[2]['score']!=score(values[0]['trace'],full.score_contract):raise ValueError('Prefix score mismatch')
    if family!='pi05':from_wire(values[2]['native_features'])
    np.savez_compressed(output/'numerical.npz',trace=values[0]['trace'],actions=values[0]['actions'])
    write(output/'numerical.json',dict(p3=values[1]['score'],full=values[2]['score'],identical_actions=True,identical_rng=True))
    bank=ExplicitFlowNoiseBank(Path(c['execution_noise']));ep=output/'trajectory';ep.mkdir();comparisons=[]
    with open_sim(b['sim_config'],sim,row['initial'],ep) as query:
        move_to(query,row.get('angles_deg',[0.,0.]));example=query(dict(kind='begin_execution'))['example']
        shadow=LiberoHistory(example) if family=='rldx' else None
        reference=RLDXReference(raw,shadow) if shadow is not None else raw
        def checked_infer(ex,n,ident,capture=False):
            before=rng_state();calls.runtime=reference
            a=calls.infer(ex,n,dict(ident,variant='native-reference'),capture=False);after=rng_state()
            restore(before);calls.runtime=full
            v=calls.infer(ex,n,dict(ident,variant='prefix-entry'),capture=False);end=rng_state()
            compare(a,v)
            if not equal_rng(after,end):raise ValueError('Operation prefix changed RNG')
            comparisons.append(input_identity(ex));return v
        def step(payload):
            reply=query(payload)
            if payload['kind']=='step' and shadow is not None:shadow.append(reply['example'])
            return reply
        record=lambda e:append_event(ep/'events.jsonl',e)
        if family=='rldx':result,arrays=rollout_rldx_episode(step,checked_infer,example,bank,row['state_key'],0,row,runtime=raw,record=record)
        else:
            fn=rollout_local_episode if family=='pi05' else rollout_groot_episode
            extra=dict(plan=read(sim['pilot_execution_plan'])) if family=='pi05' else dict(contract=sim['operation_contract'])
            result,arrays=fn(step,checked_infer,example,bank,row['state_key'],0,row,suite=row['suite'],record=record,**extra)
    if result['replans']!=len(comparisons):raise ValueError('Incomplete full trajectory comparison')
    np.savez_compressed(ep/'trajectory.npz',**arrays);write(ep/'result.json',result)
    if c['schema']=='CONTROLLED_PREFIX_MAINLINE_V1' or family=='rldx':
        import torch
        from .prefix_pilot import policy_for
        from .continuous_training import LiveEpisodes
        from AlphaBrain.research.dsol.selectors.continuous_parallel import seeded_episode
        actor_path=c['engineering_actor']
        actor=policy_for(c,cell,torch.load(actor_path,map_location='cpu',weights_only=True),version=0)
        if cell.get('evaluation_only'):
            from .continuous_evaluation import search
            calls.runtime=full if cell['prefix']!=3 else p3
            with np.load(sim['anchors'],allow_pickle=False) as z:query_noise=z['first_score_noise'][row['initial'],:2].copy()
            folder=output/'deployment-smoke';folder.mkdir()
            testrow=dict(row,method='continuous_ppo',proposal_seed=0,query_cap=cell['mechanism']['query_cap'])
            with open_sim(b['sim_config'],sim,row['initial'],folder) as query:
                info,_=search(testrow,query,calls,query_noise,folder,actor,cell['observation_scale'])
            if info['status']!='COMPLETE' or info['queries']>testrow['query_cap']:raise ValueError('Deployment query-cap check failed')
            write(output/'collector-smoke.json',dict(status='PASS_DEPLOYMENT_ENTRY',queries=info['queries'],optimizer_updates=0,training_episode=False))
        else:
            live=LiveEpisodes(dict(sim_config=b['sim_config'],trajectory_cap=b['limits']['score'],**({'native_policy':cell['native_policy']} if cell['native_policy'] is not None else {}),
                **({'task_condition':cell['task_condition']} if cell.get('task_condition') else {})),
                dict(actor=dict(score_scale=cell['observation_scale'])),output,runtime=full if cell['prefix']!=3 else p3,expected_calls=raw.calls,
                reward_scale=.01214765438717935,simulator_pool=pool)
            engineered=dict(row,episode=0,proposal_seed=0,policy_seed=0)
            episode,metrics=seeded_episode(actor,live,engineered,output/'collector-smoke',version=0,seed=0,source_kind='ENGINEERING_NATIVE_COLLECTOR_ONLY')
            if episode['status']!='COMPLETE' or not 1<=metrics['queries']<=8:raise ValueError('Actual native collector smoke failed')
            write(output/'collector-smoke.json',dict(status='PASS',queries=metrics['queries'],optimizer_updates=0,training_episode=False))
        calls.counts['score']=raw.calls
    from .continuous_authority import source_identity
    write(output/'admission.json',dict(status='PASS_PREFIX_NUMERICAL_AND_TRAJECTORY',family=family,model_hashes=cell['model_hashes'],source_identity=source_identity(),
        **({'search_core_sha256':digest(cell['core_spec'])} if cell.get('core_spec') else {}),
        **({'mechanism_sha256':digest(cell['mechanism'])} if cell.get('mechanism') else {}),
        **({'task_condition_sha256':digest(cell['task_condition'])} if cell.get('task_condition') else {}),
        admitted_host=b['host'],admitted_suite=row['suite'],complete_trajectory=True,replans=result['replans'],steps=result['steps'],calls=calls.counts,all_replans_native_equal=True,
        rldx_independent_per_action_history=family=='rldx',score_prefixes=[3,full.score_contract['prefix']]))


def peer_numerical_proof(proofs,references=None):
    """Strict shared-input cross-profile golden check before scientific dispatch.

    Each host already passed its local native/decorator/collector trajectory.
    Equal shapes or equal package version labels cannot replace numeric equality.
    """
    from AlphaBrain.research.dsol.execution.manifest import file_digest
    references={} if references is None else references;comparisons=[]
    for family,entries in proofs.items():
        suites=sorted({key.split('/')[1] for key in entries})
        for suite in suites:
            paths=[Path(v) for k,v in sorted(entries.items()) if k.endswith('/'+suite)]
            reference=Path(references.get(family+'/'+suite,paths[0]));ref=read(reference)
            for path in paths:
                value=read(path)
                if value['family']!=ref['family'] or value['model_hashes']!=ref['model_hashes'] or value['admitted_suite']!=suite:
                    raise ValueError('Different model or suite in peer numerical golden')
                a=reference.parent/'numerical.npz';b=path.parent/'numerical.npz'
                with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
                    if set(x.files)!=set(y.files) or any(not np.array_equal(x[k],y[k]) for k in x.files):
                        raise ValueError('Peer native actions/trace differ: '+family+'/'+suite+' '+str(path))
                comparisons.append(dict(reference=str(reference),candidate=str(path),reference_sha256=file_digest(a),candidate_sha256=file_digest(b),equal=True))
    return dict(status='PASS_PEER_NATIVE_ACTION_TRACE_EQUALITY',comparisons=comparisons)
