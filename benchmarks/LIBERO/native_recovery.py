"""Native mainline evaluation schema; old few-shot manifests remain strict."""
from pathlib import Path
import json
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.data import native_training
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest

SCHEMA='CONTROLLED_NATIVE_RECOVERY_V1'


def limits(rows,sim,family):
    from .model_adapters import model_adapter
    step=sim['operation_contract']['executed_actions_per_replan']
    return dict(episodes=2*len(rows),score=sum(20 for r in rows if r['method'] not in ('stay','reference'))+model_adapter(family).warmup_calls,
        paired_execution=sum(2*((sim['operation_contract']['horizons'][r['suite']]+step-1)//step) for r in rows),nominal_execution=0)


def validate_campaign(path):
    c=read(path);p=native_training.from_campaign(c)
    if c.get('explicit_recovery') is not None:
        from .target_recovery_resume import validate_native_baseline_proof
        validate_native_baseline_proof(c)
    from .model_adapters import model_adapter
    adapter=model_adapter(p['family'])
    if c['model_hashes']!=p['model_hashes']:raise ValueError('Native evaluation target differs from protocol')
    required={c['protocol'],c['scientific_plan'],c['master_schedule'],c['evaluation_inputs'],c['score_noise'],c['execution_noise'],
              *c['sim_configs'].values(),*c['native_admissions'].values()}
    if not required.issubset(c['input_hashes']):raise ValueError('Unbound native evaluation inputs')
    for name,stamp in c['input_hashes'].items():
        if file_digest(name)!=stamp:raise ValueError('Native evaluation artifact changed: '+name)
    plan=read(c['scientific_plan']);planned=plan['models'][p['family']]['phases'][c['phase']]
    if (c['schema']!=SCHEMA or c['purpose']!='native_mainline_evaluation' or c['query_cap']!=8
            or c['score_scale']!=p['observation_scale'] or c['execution_repeats']!=[0,1]
            or planned['path']!=c['master_schedule'] or planned['sha256']!=file_digest(c['master_schedule'])):
        raise ValueError('Native mainline evaluation scope differs')
    master=read(c['master_schedule']);known={r['trial_id']:r for r in master}
    if len(known)!=len(master):raise ValueError('Duplicate master native trials')
    states=read(c['evaluation_inputs'])['records'];routes={}
    suites={'libero_spatial','libero_object','libero_goal','libero_10'}
    if len(states)!=400 or set(c['sim_configs'])!=suites or set(c['native_admissions'])!=suites:raise ValueError('Complete evaluation population/routes required')
    from .search_runtime import selected_checkpoint
    for suite,name in c['sim_configs'].items():
        sim=read(name);native_training.validate_admission(dict(c,sim_config=name,native_admission=c['native_admissions'][suite]),sim,p)
        model=selected_checkpoint(sim,native_family=p['family'])
        if (sim['inputs']!=c['evaluation_inputs'] or model['weights_sha256']!=c['model_hashes'][suite]
                or model['runtime_contract_sha256']!=p['runtime_contract_hashes'][suite]
                or model['implementation']!=adapter.implementation):
            raise ValueError('Native evaluation operation route differs')
        routes[suite]=sim
    totals=dict(episodes=0,score=0,paired_execution=0,nominal_execution=0);seen=set()
    for index,b in enumerate(c['blocks']):
        rows=b['rows']
        if b['index']!=index or not 1<=len(rows)<=2 or len({r['suite'] for r in rows})!=1 or b['host'] not in c['execution_hosts']:
            raise ValueError('Short suite-affine native evaluation blocks required')
        suite=rows[0]['suite'];sim=routes[suite]
        if b['sim_config']!=c['sim_configs'][suite] or b['limits']!=limits(rows,sim,p['family']):raise ValueError('Native evaluation block budget differs')
        for r in rows:
            state=states[r['initial']]['state']
            if r['trial_id'] in seen or r!=known.get(r['trial_id']) or state['pair_key']!=r['state_key']:
                raise ValueError('Repeated/unplanned/wrong-state native evaluation trial')
            if r['method'].startswith('G-seed-'):
                actor=c['actors'][r['method']]
                if (actor['version']!=64 or type(actor.get('seed')) is not int or actor['seed'] not in (0,1,2)
                        or r['method']!=f"G-seed-{actor['seed']}"
                        or c['input_hashes'].get(actor['path'])!=file_digest(actor['path'])):
                    raise ValueError('Only predeclared final native search policies admitted')
            elif r['method'] not in ('stay','reference','local','random_from_local'):raise ValueError('Unknown native mainline selection')
            seen.add(r['trial_id'])
        for k in totals:totals[k]+=b['limits'][k]
    if totals!=c['limits']:raise ValueError('Native evaluation total differs')
    return c


def request_for(path,index):
    c=read(path);b=dict(c['blocks'][index]);family=read(c['protocol'])['family']
    b['addresses']=[dict(model=family,task=r['task'],initial=r['state_key'],view=f"{r['view_id']}:{r['method']}",
        stream='execution',repeat=repeat,noise=f'paired-{repeat}') for r in b['rows'] for repeat in (0,1)]
    return dict(campaign=str(path),campaign_sha256=file_digest(path),host=b['host'],stage='formal',block=b)


def validate_block(request,output):
    output=Path(output);receipt=read(output/'complete.json')
    if receipt['request_sha256']!=digest(request):raise ValueError('Native recovery completion identity differs')
    for name,stamp in receipt['artifacts'].items():
        if file_digest(output/name)!=stamp:raise ValueError('Native recovery output changed')
    value=read(output/'results.json');status=read(output/'status.json');rows=request['block']['rows']
    expected={(r['trial_id'],repeat) for r in rows for repeat in (0,1)}
    if ({(r['trial_id'],r['repeat']) for r in value['tasks']}!=expected or len(value['tasks'])!=len(expected)
            or len(value['searches'])!=len(rows) or {r['trial_id'] for r in value['searches']}!={r['trial_id'] for r in rows}):
        raise ValueError('Incomplete native paired operation coverage')
    attempts=[json.loads(line) for line in (output/'model-attempts.jsonl').read_text().splitlines()]
    counts={stage:sum(r['identity']['stage']==stage for r in attempts) for stage in ('score','execution')}
    warmups=sum(r['identity'].get('purpose')=='native_graph_warmup' for r in attempts)
    diagnostics=sum(r['identity'].get('purpose')=='native_diagnostic' for r in attempts)
    reused=dict(score=0,execution=0)
    if (output/'reused-calls.jsonl').exists():
        from .target_recovery_resume import cached_baseline_calls
        reused=cached_baseline_calls(read(request['campaign']),rows,value,output)
    if (status['status']!='COMPLETE' or counts!=status['calls']
            or counts['score']!=2*sum(r['queries'] for r in value['searches'])+warmups+diagnostics
            or diagnostics!=4*sum(r['method'] not in ('stay','reference') for r in rows)
            or sum(1 for _ in (output/'calls.jsonl').open())!=sum(counts.values())
            or counts['execution']+reused['execution']!=sum(r['replans'] for r in value['tasks'])
            or counts['score']>request['block']['limits']['score']
            or counts['execution']+reused['execution']>request['block']['limits']['paired_execution']):
        raise ValueError('Native recovery query/warmup/diagnostic/operation accounting differs')
    return dict(calls=counts,receipt_sha256=file_digest(output/'complete.json'))
