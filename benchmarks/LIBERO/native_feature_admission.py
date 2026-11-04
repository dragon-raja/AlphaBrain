"""Bounded same-input triplets and complete engineering operation trajectories.

Each actual replan performs raw / trace-only / native-feature inference and
requires bitwise equal actions, equal trace and equal post-call RNG. Only the
observed action is stepped. All three calls are charged before execution.
"""
from pathlib import Path
import json
import random
import time
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.data.native_features import from_wire
from AlphaBrain.research.dsol.data.native_training import admission_runtime_sha256
from AlphaBrain.research.dsol.execution.manifest import file_digest,digest

SCHEMA='CONTROLLED_NATIVE_FEATURE_ADMISSION_V1'


def limits(rows,sim,family):
    from .model_adapters import model_adapter
    step=sim['operation_contract']['executed_actions_per_replan']
    return dict(episodes=len(rows),score=18*len(rows)+model_adapter(family).warmup_calls,
        paired_execution=sum(3*((sim['operation_contract']['horizons'][r['suite']]+step-1)//step) for r in rows),nominal_execution=0)


def validate_execution_noise(c,sim,rows):
    """Reject missing addresses or short banks before any GPU/model call."""
    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
    bank=ExplicitFlowNoiseBank(Path(c['execution_noise']))
    if c['input_hashes'].get(str(bank.noise_path))!=bank.manifest['noise_file_sha256']:
        raise ValueError('Unbound native execution noise tensor')
    step=sim['operation_contract']['executed_actions_per_replan']
    for row in rows:
        last=(sim['operation_contract']['horizons'][row['suite']]+step-1)//step-1
        for index in (0,last):
            noise=bank.get(row['state_key'],0,index)['noise']
            if noise.shape!=(10,7) or not np.isfinite(noise).all():raise ValueError('Native execution noise shape/values differ')


def validate_campaign(path):
    c=read(path);sim=read(c['sim_config']);states=read(sim['inputs'])['records']
    from .model_adapters import model_adapter
    adapter=model_adapter(c['family'])
    if (c['schema']!=SCHEMA or not 1<=len(c['blocks'])<=2
            or not 1<=sum(len(b['rows']) for b in c['blocks'])<=2 or not sim.get('continuous_camera')
            or sim.get('continuous_execution')!='PAIRED_EVALUATION_ONLY'
            or sim['native_features']['layout']!=c['native_policy']['layout']
            or digest(sim['native_features'])!=c['native_policy']['producer_sha256']):
        raise ValueError('Bounded native engineering admission contract required')
    from .search_runtime import selected_checkpoint
    model=selected_checkpoint(sim,native_family=c['family'])
    if model['implementation']!=adapter.implementation:raise ValueError('Native admission family/runtime differs')
    contract=adapter.validate_runtime(model['runtime_contract'],model['runtime_contract_sha256'])
    required={c['sim_config'],sim['inputs'],sim['anchors'],c['execution_noise'],model['runtime_contract'],*contract['input_hashes']}
    if not required.issubset(c['input_hashes']):raise ValueError('Unbound native engineering assets')
    # The adapter just verified these bytes in this invocation. Reuse only that
    # result, never an mtime/NFS cache across validations or launch boundaries.
    verified={model['runtime_contract']:model['runtime_contract_sha256'],**contract['input_hashes']}
    for name,stamp in c['input_hashes'].items():
        if verified.get(name)!=stamp and file_digest(name)!=stamp:raise ValueError('Changed native admission input: '+name)
    rows=[r for b in c['blocks'] for r in b['rows']]
    expected={k:0 for k in limits([],sim,c['family'])}
    for index,b in enumerate(c['blocks']):
        block_limits=limits(b['rows'],sim,c['family'])
        if not b['rows'] or b['index']!=index or b['limits']!=block_limits or b['host'] not in c['execution_hosts']:
            raise ValueError('Native engineering scope/budget differs')
        for k,v in block_limits.items():expected[k]+=v
    if c['limits']!=expected or len({r['initial'] for r in rows})!=len(rows):
        raise ValueError('Native engineering total/unique states differ')
    for row in rows:
        state=states[row['initial']]['state']
        if state['split']!='train' or state['pair_key']!=row['state_key'] or state['suite']!=row['suite']:
            raise ValueError('Engineering admission uses declared training states only')
        if adapter.suite_specific and state['suite']!=contract['suite']:raise ValueError('Engineering suite-specific weights differ')
    validate_execution_noise(c,sim,rows)
    return c


def request_for(path,index):
    c=read(path)
    if type(index) is not int or not 0<=index<len(c['blocks']):raise ValueError('Unknown bounded admission block')
    b=dict(c['blocks'][index]);b['addresses']=[dict(model=c['family'],task=r['task'],initial=r['state_key'],view='native_feature_admission',stream='execution',repeat=0,noise='paired-0') for r in b['rows']]
    return dict(campaign=str(path),campaign_sha256=file_digest(path),block=b,host=b['host'],stage='formal')


class Triplet:
    def __init__(self,calls,model_sha,*,producer_sha256,evidence=None):
        self.calls=calls;self.model_sha=model_sha;self.sequence=0
        self.producer_sha256=producer_sha256
        self.evidence=Path(evidence) if evidence is not None else None
        if self.evidence is not None:self.evidence.mkdir(exist_ok=False)
    def infer(self,example,noise,identity,capture=False):
        import torch
        from .local_search_rollout import input_identity
        self.sequence+=1;cuda=torch.cuda.is_available()
        before=torch.get_rng_state();device_before=torch.cuda.get_rng_state_all() if cuda else []
        python_before=random.getstate();numpy_before=np.random.get_state()
        def restore():
            torch.set_rng_state(before)
            if cuda:torch.cuda.set_rng_state_all(device_before)
            random.setstate(python_before);np.random.set_state(numpy_before)
        def rng():
            state=np.random.get_state()
            return ([torch.get_rng_state(),*(torch.cuda.get_rng_state_all() if cuda else [])],
                    (random.getstate(),state[0],state[1].tobytes(),state[2:]))
        def equal_rng(a,b):return a[1]==b[1] and len(a[0])==len(b[0]) and all(torch.equal(x,y) for x,y in zip(a[0],b[0]))
        started=time.perf_counter()
        raw=self.calls.infer(example,noise,dict(identity,variant='raw'),capture=False);raw_rng=rng();raw_end=time.perf_counter()
        restore();control=self.calls.infer(example,noise,dict(identity,variant='trace_control'),capture=True,feature_capture=False);control_rng=rng();control_end=time.perf_counter()
        restore();request=dict(query_id=f'native-admission-{self.sequence}',input_sha256=input_identity(example),history_sha256=digest(dict(identity)))
        observed=self.calls.infer(example,noise,dict(identity,variant='native'),capture=True,native_request=request);observed_rng=rng();native_end=time.perf_counter()
        if (not np.array_equal(raw['actions'],control['actions']) or not np.array_equal(control['actions'],observed['actions'])
                or not np.array_equal(control['trace'],observed['trace']) or control['score']!=observed['score']
                or not equal_rng(raw_rng,control_rng) or not equal_rng(control_rng,observed_rng)):
            raise ValueError('Native features changed actions, flow trace, score or RNG')
        full_chunks={}
        if any('full_actions' in result for result in (raw,control,observed)):
            if (not all('full_actions' in result for result in (raw,control,observed))
                    or not np.array_equal(raw['full_actions'],control['full_actions'])
                    or not np.array_equal(control['full_actions'],observed['full_actions'])):
                raise ValueError('Instrumentation changed the full native action chunk')
            full_chunks=dict(raw_full_actions=raw['full_actions'],
                             control_full_actions=control['full_actions'],native_full_actions=observed['full_actions'])
        packet=from_wire(observed['native_features'])
        if (packet.model_sha256!=self.model_sha or packet.producer_sha256!=self.producer_sha256 or packet.input_sha256!=request['input_sha256']
                or packet.history_sha256!=request['history_sha256'] or packet.query_id!=request['query_id']):
            raise ValueError('Native admission feature provenance mismatch')
        if self.evidence is not None:
            folder=self.evidence/f'{self.sequence:05d}';folder.mkdir()
            np.savez_compressed(folder/'values.npz',raw_actions=raw['actions'],control_actions=control['actions'],
                native_actions=observed['actions'],control_trace=control['trace'],native_trace=observed['trace'],
                native_pooled=packet.packed(),noise=noise,**full_chunks)
            write(folder/'identity.json',dict(identity=identity,input_sha256=request['input_sha256'],features=packet.metadata(),
                actions_exact=True,trace_exact=True,rng_exact=True,
                full_actions_exact=True if full_chunks else None,
                seconds=dict(raw=raw_end-started,trace=control_end-raw_end,native=native_end-control_end),
                memory={k:observed[k] for k in ('peak_allocated_mib','peak_reserved_mib') if k in observed}))
        return observed


def run(request,output,*,guard,runtime_cache=None,simulator_pool=None):
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    reject_production('benchmarks/LIBERO/controlled_admission.py:run')
    c=validate_campaign(request['campaign'])
    if request!=request_for(request['campaign'],request['block']['index']):raise ValueError('Native admission request changed')
    from .search_runtime import create_policy_runtime
    from .search_live import simulator
    from .local_view_runtime import RenderedViewSource
    from .continuous_evaluation import EvaluationCalls,append_event
    from .local_search_rollout import input_identity
    from .model_adapters import model_adapter
    from .continuous_authority import source_identity
    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
    output=Path(output);output.mkdir(parents=True,exist_ok=False);sim=read(c['sim_config']);b=request['block']
    runtime=create_policy_runtime(sim,output,b['limits']['score']+b['limits']['paired_execution'])
    calls=EvaluationCalls(runtime,output,dict(score=b['limits']['score'],execution=b['limits']['paired_execution']),guard=guard)
    expected=sim['models'][sim['operation_model']]['weights_sha256']
    paired=Triplet(calls,expected,producer_sha256=c['native_policy']['producer_sha256'],evidence=output/'pairs')
    with np.load(sim['anchors'],allow_pickle=False) as z:noises=z['first_score_noise'].copy()
    bank=ExplicitFlowNoiseBank(Path(c['execution_noise']));records=[]
    for row in b['rows']:
        folder=output/f"case-{row['initial']}";folder.mkdir();calls.stage='score'
        with simulator(c['sim_config'],sim,row['initial'],folder) as query:
            source=RenderedViewSource(query,calls,noises[row['initial'],:2],folder)
            views=[]
            for angles in ([0.,0.],[7.125,-4.375],[0.,0.]):
                source.move_angles(angles);reply=query(dict(kind='query_pose'));source._receipt(reply['receipt']);example=reply['example']
                if getattr(runtime,'native_needs_warmup',False):calls.infer(example,noises[row['initial'],0],dict(purpose='native_graph_warmup'),capture=False,warmup=True)
                values=[paired.infer(example,noise,dict(initial=row['initial'],pose=len(views),member=m)) for m,noise in enumerate(noises[row['initial'],:2])]
                views.append(dict(input_sha256=input_identity(example),values=values))
            if views[0]['input_sha256']!=views[2]['input_sha256']:raise ValueError('Native camera return pixels/state differ')
            for first,last in zip(views[0]['values'],views[2]['values']):
                if not np.array_equal(from_wire(first['native_features']).packed(),from_wire(last['native_features']).packed()):raise ValueError('Native camera return features differ')
            if views[0]['input_sha256']==views[1]['input_sha256'] or np.array_equal(from_wire(views[0]['values'][0]['native_features']).packed(),from_wire(views[1]['values'][0]['native_features']).packed()):
                raise ValueError('Native freshness not established by the declared camera intervention')
        calls.stage='execution';ep=folder/'engineering-trajectory';ep.mkdir()
        with simulator(c['sim_config'],sim,row['initial'],ep) as query:
            example=query(dict(kind='begin_execution'))['example']
            result,arrays=model_adapter(c['family']).rollout(query,paired.infer,example,bank,row['state_key'],0,row,contract=sim['operation_contract'],suite=row['suite'],record=lambda event:append_event(ep/'events.jsonl',event))
        np.savez_compressed(ep/'trajectory.npz',**arrays);write(ep/'result.json',result)
        records.append(dict(initial=row['initial'],replans=result['replans'],success=result['success'],actions_exact=True,trace_exact=True,rng_exact=True,fresh_features=True,return_exact=True))
    write(output/'admission.json',dict(status='PASS_NATIVE_FEATURE_TRAJECTORY',scope='REAL_NATIVE_MODEL',family=c['family'],
        admitted_sim_config=c['sim_config'],sim_config_sha256=file_digest(c['sim_config']),
        runtime_identity_sha256=admission_runtime_sha256(sim),native_policy_sha256=digest(c['native_policy']),source_identity=source_identity(),
        actions_exact=True,trace_exact=True,rng_exact=True,fresh_features=True,return_exact=True,calls=calls.counts,records=records,
        training_updates=0,used_for_primary_results=False))


def validate_block(request,output):
    output=Path(output);receipt=read(output/'complete.json')
    if receipt['request_sha256']!=digest(request):raise ValueError('Changed native admission completion')
    for name,stamp in receipt['artifacts'].items():
        if file_digest(output/name)!=stamp:raise ValueError('Native admission output changed')
    p=read(output/'admission.json');attempts=[json.loads(x) for x in (output/'model-attempts.jsonl').read_text().splitlines()]
    counts={k:sum(r['identity']['stage']==k for r in attempts) for k in ('score','execution')}
    if (p['status']!='PASS_NATIVE_FEATURE_TRAJECTORY' or p['calls']!=counts or len(p['records'])!=len(request['block']['rows'])
            or counts['score']>request['block']['limits']['score'] or counts['execution']>request['block']['limits']['paired_execution']
            or counts['execution']!=3*sum(r['replans'] for r in p['records'])
            or sum(1 for _ in (output/'calls.jsonl').open())!=sum(counts.values())):
        raise ValueError('Native engineering trajectory/call accounting differs')
    return dict(calls=counts,receipt_sha256=file_digest(output/'complete.json'))
