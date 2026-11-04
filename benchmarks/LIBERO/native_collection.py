"""New native-feature schema on the existing controlled training adapter."""
from collections import defaultdict
from pathlib import Path
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.data import native_training
from AlphaBrain.research.dsol.execution.manifest import file_digest


def validate_campaign(path):
    c=read(path);p=native_training.from_campaign(c)
    from .model_adapters import model_adapter
    adapter=model_adapter(p['family'])
    if c['model_hashes']!=p['model_hashes']:raise ValueError('Native collection target differs from protocol')
    from AlphaBrain.research.dsol.selectors.search_policy_factory import validate_policy_spec
    validate_policy_spec(c['native_policy'])
    if (c['schema']!=native_training.COLLECTION_SCHEMA or c['kind']!='rollout'
            or c['schedules']!=p['schedules'] or c['query_cap']!=8 or c['query_cost']!=.01
            or c['observation_scale']!=p['observation_scale'] or not c['settle_reward_at_learner']
            or (c['native_policy']['width'],c['native_policy']['heads'],c['native_policy']['layers'])!=(128,4,2)):
        raise ValueError('Native production G collection contract differs')
    from .search_runtime import selected_checkpoint
    suites={'libero_spatial','libero_object','libero_goal','libero_10'}
    if set(c['sim_configs'])!=suites or set(c['native_admissions'])!=suites:raise ValueError('Explicit four-suite native routes required')
    required={c['training_inputs'],c['training_noise'],c['protocol'],*p['schedules'].values(),*c['sim_configs'].values(),*c['native_admissions'].values()}
    routes={};verified={};contracts={}
    for suite,path in c['sim_configs'].items():
        sim=read(path);route=dict(c,sim_config=path,native_admission=c['native_admissions'][suite])
        native_training.validate_admission(route,sim,p)
        model=selected_checkpoint(sim,native_family=p['family'])
        if (model['implementation']!=adapter.implementation
                or sim['operation_model']!=c['operation_model'] or model['weights_sha256']!=c['model_hashes'][suite]
                or model['runtime_contract_sha256']!=p['runtime_contract_hashes'][suite]):
            raise ValueError('Native target model identity differs')
        if sim['inputs']!=c['training_inputs'] or sim['anchors']!=c['training_noise']:raise ValueError('Wrong native simulator inputs/noise')
        # Verify each immutable runtime once in this invocation, not once per
        # episode block. Never retain this evidence across calls or processes.
        runtime_path=model['runtime_contract'];stamp=model['runtime_contract_sha256']
        if runtime_path not in contracts:
            contract=adapter.validate_runtime(runtime_path,stamp)
            contracts[runtime_path]=(stamp,contract)
            verified.update(contract['input_hashes']);verified[runtime_path]=stamp
        elif contracts[runtime_path][0]!=stamp:
            raise ValueError('Conflicting native runtime contract identities')
        if adapter.suite_specific and contracts[runtime_path][1]['suite']!=suite:
            raise ValueError('Suite-specific checkpoint differs')
        required.add(model['runtime_contract']);routes[suite]=model
    if not required.issubset(c['input_hashes']):raise ValueError('Unbound native collection inputs')
    for path,stamp in c['input_hashes'].items():
        if verified.get(path)!=stamp and file_digest(path)!=stamp:raise ValueError('Native collection input changed: '+path)
    states=read(c['training_inputs'])['records']
    if len(states)!=400 or any(r['state']['split']!='train' for r in states):raise ValueError('Only 400 fixed training states admitted')
    schedules={s:read(path) for s,path in p['schedules'].items()}
    groups=defaultdict(list);seen=set();total=dict(episodes=0,score=0,paired_execution=0,nominal_execution=0)
    for index,b in enumerate(c['blocks']):
        rows=b['rows'];n=len(rows);seed=b['seed'];version=b['version']
        expected=dict(episodes=n,score=16*n+adapter.warmup_calls,paired_execution=0,nominal_execution=0)
        if (b['index']!=index or not 1<=n<=2 or b['limits']!=expected or b['arm']!='G'
                or seed not in (0,1,2) or type(version) is not int or not 0<=version<64
                or b['run']!=f'G-seed-{seed}' or b['reward_scale']!=native_training.reward_config()['fixed_scale']
                or c['input_hashes'].get(b['policy'])!=b['policy_sha256']):
            raise ValueError('Native collection budget/version differs')
        if len({r['suite'] for r in rows})!=1:raise ValueError('Each resident native block has one suite route')
        suite=rows[0]['suite'];model=routes[suite]
        if b['sim_config']!=c['sim_configs'][suite]:raise ValueError('Native block route differs')
        if adapter.suite_specific:
            contract=contracts[model['runtime_contract']][1]
            if any(r['suite']!=contract['suite'] for r in rows):raise ValueError('Suite-specific checkpoint differs')
        for row in rows:
            key=(seed,row['episode']);state=states[row['initial']]['state']
            if (key in seen or row!=schedules[str(seed)][row['episode']] or row['episode']//32!=version
                    or state['pair_key']!=row['state_key'] or state['task_id']!=row['task']):
                raise ValueError('Repeated/held-out/unplanned native episode')
            seen.add(key);groups[(seed,version)].append(row['episode'])
        for key in total:total[key]+=expected[key]
    if not groups or any(sorted(rows)!=list(range(v*32,(v+1)*32)) for (_,v),rows in groups.items()):
        raise ValueError('Complete native 32-episode on-policy barriers required')
    if total!=c['limits']:raise ValueError('Native campaign cap differs')
    validate_recovery(c)
    return c


def validate_recovery(c):
    """Only an explicitly audited stopped phase may supply saved episodes."""
    path=c.get('collection_recovery')
    if not path:return
    if c['input_hashes'].get(path)!=file_digest(path):raise ValueError('Unbound native recovery')
    proof=read(path)
    if (proof.get('schema')!='NATIVE_COLLECTION_RECOVERY_V1'
            or proof.get('status')!='VERIFIED_STOPPED_PARTIAL_COLLECTION'
            or proof.get('all_workers_stopped') is not True or proof.get('automatic_retry') is not False
            or proof.get('protocol_sha256')!=file_digest(c['protocol'])
            or not isinstance(proof.get('held_score_cap'),int) or proof['held_score_cap']<0):
        raise ValueError('Explicit stopped native recovery and retained budget required')
    old=read(proof['source_campaign'])
    for name in (proof['source_campaign'],proof['process_audit']):
        if c['input_hashes'].get(name)!=file_digest(name):raise ValueError('Unbound native recovery audit')
    if (old['protocol']!=c['protocol'] or old['native_policy']!=c['native_policy']
            or read(proof['process_audit']).get('all_workers_stopped') is not True):
        raise ValueError('Native recovery source/protocol/process audit differs')
    previous={(b['run'],r['episode']):(b,r) for b in old['blocks'] for r in b['rows']}
    planned={f"{b['run']}:{r['episode']}":(b,r) for b in c['blocks'] for r in b['rows']}
    if not proof.get('records') or not set(proof['records']).issubset(planned):
        raise ValueError('Recovered episodes outside native campaign')
    for key,ref in proof['records'].items():
        b,row=planned[key];prior,prior_row=previous[(b['run'],row['episode'])]
        if (row!=prior_row or b['version']!=proof['resume_version'] or prior['version']!=b['version']
                or prior['policy_sha256']!=b['policy_sha256']
                or c['input_hashes'].get(ref['path'])!=ref['sha256'] or file_digest(ref['path'])!=ref['sha256']):
            raise ValueError('Recovered native row/actor/artifact differs')
