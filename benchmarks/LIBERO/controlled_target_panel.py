"""Fixed target-model response panel, independent of PPO visitation.

Only missing, predeclared measurements run. Reuse provenance belongs to the
frozen panel manifest; no success-based filtering or automatic retry occurs.
"""
from contextlib import ExitStack
from pathlib import Path
import json
import time
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read, write
from AlphaBrain.research.dsol.execution.manifest import digest, file_digest, task_key
from .controlled_target_evaluation import validate_assets, execution_hosts

SCHEMA = 'CONTROLLED_TARGET_SPATIAL_V1'


def row_key(row):
    return row['state_key'], row['view_id']


def limits(rows, contract):
    return dict(episodes=sum(len(r['execution_repeats']) for r in rows),
                score=2 * len(rows)+int(contract['schema']=='GROOT_N15_LIBERO_OPERATION_V1'), paired_execution=sum(
                    len(r['execution_repeats']) * (contract['horizons'][r['suite']] // 10)
                    for r in rows), nominal_execution=0)


def validate_campaign(path):
    c, sim, states, contract = validate_assets(path)
    native_suite=(read(sim['models'][sim['operation_model']]['runtime_contract'])['suite']
                  if c.get('target_model')=='groot_n15' else None)
    if c['schema'] != SCHEMA or c['purpose'] != 'fixed_spatial_diagnostic':
        raise ValueError('Explicit spatial diagnostic required')
    required = {c['schedule'], c['score_noise'], c['trajectory_proof'], c['reuse_manifest']}
    if not required.issubset(c['input_hashes']) or read(c['trajectory_proof'])['status'] != 'PASS':
        raise ValueError('Missing spatial inputs or trajectory gate')
    schedule = read(c['schedule'])
    planned = {row_key(r): r for r in schedule}
    if not schedule or len(planned) != len(schedule):
        raise ValueError('Duplicate spatial rows')
    actual = []
    for i, block in enumerate(c['blocks']):
        if (block['index'] != i or block['host'] not in execution_hosts(c)
                or not 1 <= len(block['rows']) <= 8
                or block['limits'] != limits(block['rows'], contract)):
            raise ValueError('Spatial block budget/host changed')
        for row in block['rows']:
            if row != planned.get(row_key(row)):
                raise ValueError('Unplanned spatial row')
            if type(row['initial']) is not int or not 0 <= row['initial'] < len(states):
                raise ValueError('Invalid initial state index')
            state = states[row['initial']]['state']
            if native_suite is not None and row['suite']!=native_suite:
                raise ValueError('GR00T suite-specific weights do not match the task')
            if (state['pair_key'] != row['state_key'] or state['suite'] != row['suite']
                    or state['split'] != 'validation' or state['initialization']['settle_steps'] != contract['settle_steps']
                    or row['score_members'] != [0, 1]
                    or row['execution_repeats'] not in ([], [0, 1])):
                raise ValueError('Spatial state/noise/initialization changed')
            from AlphaBrain.research.dsol.acquisition.local_view_search import PanTiltMount
            PanTiltMount(np.eye(4)).angles(row['angles_deg'])
            actual.append(row)
    if (len(actual) != len(schedule) or len({row_key(r) for r in actual}) != len(actual)
            or c['limits'] != {k:sum(b['limits'][k] for b in c['blocks']) for k in c['limits']}
            or c['limits'] != c['authorization']['limits']):
        raise ValueError('Spatial coverage/authorization differs')
    noise = read(c['execution_noise'])
    if any(r['state_key'] not in noise['state_keys'] for r in actual):
        raise ValueError('Missing spatial execution noise')
    return c


def request_for(path, index):
    c = read(path)
    b = dict(c['blocks'][index])
    model=c.get('target_model','fastwam')
    b['addresses'] = [dict(model=model, task=r['state_key'], initial=r['state_key'],
        view=f"fixed-{r['view_id']}", stream=stage, repeat=member, noise=f'{stage}-{member}')
        for r in b['rows'] for stage, members in
        (('score', r['score_members']), ('execution', r['execution_repeats'])) for member in members]
    if model=='groot_n15':
        row=b['rows'][0]
        b['addresses'].append(dict(model=model,task=row['state_key'],initial=row['state_key'],
            view=f"graph-warmup-fixed-{row['view_id']}",stream='score',repeat=0,noise='score-0'))
    return dict(campaign=str(path), campaign_sha256=file_digest(path), host=b['host'],
                stage='formal', block=b)


def validate_spec(binding, key):
    spec = binding['tasks'][key]
    req = spec['request']; block = req['block']
    if (req != request_for(binding['campaign'], block['index'])
            or spec['limits'] != block['limits'] or spec['addresses'] != block['addresses']
            or spec['host'] != req['host'] or spec['stage'] != 'formal'
            or key != task_key(spec['identity_experiment_id'], req['campaign_sha256'], block['addresses'])):
        raise ValueError('Spatial scientific binding changed')
    return block['limits']


def run(request, output, *, guard, runtime_cache=None, simulator_pool=None):
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    reject_production('benchmarks/LIBERO/controlled_target_panel.py:run')
    c = validate_campaign(request['campaign']); b = request['block']
    if request != request_for(request['campaign'], b['index']):
        raise ValueError('Spatial request changed')
    from .search_runtime import create_policy_runtime, bind_policy_task
    from .search_live import simulator
    from .local_view_runtime import RenderedViewSource
    from .continuous_evaluation import EvaluationCalls, append_event
    from .local_search_rollout import rollout_fastwam_episode, rollout_groot_episode, input_identity
    from .search_domain import exact_input
    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    sim = read(c['sim_config']); bank = ExplicitFlowNoiseBank(Path(c['execution_noise']))
    is_groot=c.get('target_model')=='groot_n15'
    rollout=rollout_groot_episode if is_groot else rollout_fastwam_episode
    with np.load(c['score_noise']) as z: noises = z['noise'].copy()
    started = time.monotonic(); tasks = []; scores = []
    open_sim = simulator if simulator_pool is None else simulator_pool.session
    with ExitStack() as stack:
        cap = b['limits']['score'] + b['limits']['paired_execution']
        if runtime_cache is None: runtime = create_policy_runtime(sim, output, cap)
        else:
            runtime, _ = stack.enter_context(runtime_cache.lease(digest(sim),
                lambda: create_policy_runtime(sim, output, cap),
                lambda v: bind_policy_task(v, output, cap)))
        calls = EvaluationCalls(runtime, output,
            dict(score=b['limits']['score'], execution=b['limits']['paired_execution']), guard=guard)
        for i, row in enumerate(b['rows']):
            guard(); folder = output / f'case-{i:04d}'; folder.mkdir()
            values = noises[row['initial'], :2]; calls.identity = dict(row)
            calls.stage = 'score'
            with open_sim(c['sim_config'], sim, row['initial'], folder) as query:
                source = RenderedViewSource(query, calls, values, folder)
                source.move_angles(row['angles_deg'])
                preview = query(dict(kind='query_pose'))
                source._receipt(preview['receipt']); selected = preview['example']
                expected = row.get('expected_input_sha256')
                if expected is not None and input_identity(selected) != expected:
                    raise ValueError('Prior spatial input identity changed')
                if is_groot:
                    # No source-model feature tower, fabricated camera features,
                    # or change to the native four-step sampler is involved.
                    if i==0:
                        calls.infer(selected,values[0],dict(purpose='graph_warmup'),capture=False,warmup=True)
                    measured=[calls.infer(selected,noise,dict(member=m),capture=True)
                              for m,noise in enumerate(values)]
                    saved=dict(camera_intrinsics=selected['camera_intrinsics'],
                               camera_to_world_opencv=selected['camera_to_world_opencv'],
                               robot_state=selected['state'])
                    saved.update({'raw_'+k:v for k,v in selected.get('fastwam_raw',{}).items()
                                  if isinstance(v,np.ndarray)})
                    for m,value in enumerate(measured):
                        for field in ('trace','timesteps','initial_noise','actions','normalized_actions'):
                            saved[f'{field}_{m}']=value[field]
                        saved[f'score_{m}']=value['scores']['3']
                        saved[f'score_p4_{m}']=value['scores']['4']
                    np.savez_compressed(folder/'query-01.npz',**saved)
                    score=dict(row,input_sha256=input_identity(selected),
                        scores=[v['scores']['3'] for v in measured],
                        native_full_scores=[v['scores']['4'] for v in measured],
                        prefixes=[3,4],query_path=str(folder/'query-01.npz'))
                else:
                    source.query(); exact_input(selected, source.last_example)
                    score = dict(row, input_sha256=input_identity(selected),
                        scores=[float(v['score']) for v in source.last_values],
                        query_path=str(folder / 'query-01.npz'))
                write(folder / 'score.json', score); scores.append(score)
            calls.stage = 'execution'
            for repeat in row['execution_repeats']:
                ep = folder / f'episode-{repeat}'; ep.mkdir()
                with open_sim(c['sim_config'], sim, row['initial'], ep) as query:
                    source = RenderedViewSource(query, calls, values, ep)
                    source.move_angles(row['angles_deg'])
                    example = query(dict(kind='begin_execution'))['example']
                    exact_input(selected, example)
                    if is_groot and input_identity(selected)!=input_identity(example):
                        raise ValueError('Native raw pixels/proprioception changed before execution')
                    result, arrays = rollout(query, calls.infer, example, bank,
                        row['state_key'], repeat, row, contract=sim['operation_contract'],
                        suite=row['suite'], record=lambda e: append_event(ep / 'events.jsonl', e))
                asset = ep / 'trajectory.npz'; np.savez_compressed(asset, **arrays)
                result.update(row, repeat=repeat, trajectory=str(asset))
                write(ep / 'result.json', result); tasks.append(result)
                write(output / 'progress.json', dict(tasks=len(tasks), positions=len(scores), calls=calls.counts))
        write(output / 'results.json', dict(tasks=tasks, scores=scores))
        write(output / 'status.json', dict(status='COMPLETE', calls=calls.counts,
            seconds=time.monotonic()-started, optimizer_updates=0))


def validate_block(request, output):
    output = Path(output); receipt = read(output / 'complete.json')
    if receipt['request_sha256'] != digest(request):
        raise ValueError('Spatial receipt changed')
    for path, stamp in receipt['artifacts'].items():
        if file_digest(output / path) != stamp: raise ValueError('Spatial artifact changed')
    result = read(output / 'results.json'); status = read(output / 'status.json')
    rows = request['block']['rows']; cap = request['block']['limits']
    expected = {(row_key(r), s) for r in rows for s in r['execution_repeats']}
    actual = [(row_key(r), r['repeat']) for r in result['tasks']]
    if (len(actual) != len(expected) or set(actual) != expected
            or len(result['scores']) != len(rows)
            or {row_key(r) for r in result['scores']} != {row_key(r) for r in rows}
            or any(len(r['scores']) != 2 or not np.isfinite(r['scores']).all() for r in result['scores'])):
        raise ValueError('Incomplete spatial outcomes')
    attempts = [json.loads(x) for x in (output / 'model-attempts.jsonl').read_text().splitlines()]
    counts = {s: sum(r['identity']['stage'] == s for r in attempts) for s in ('score', 'execution')}
    runtime_calls = sum(1 for _ in (output / 'calls.jsonl').open())
    if (status['status'] != 'COMPLETE' or status['optimizer_updates'] != 0
            or status['calls'] != counts or runtime_calls != sum(counts.values())
            or counts['score'] != cap['score'] or counts['execution'] > cap['paired_execution']
            or counts['execution'] != sum(r['replans'] for r in result['tasks'])
            or any(type(r['success']) is not bool or not 0 < r['steps'] <= r['horizon']
                   or len(r['execution_noise_sha256']) != r['replans'] for r in result['tasks'])):
        raise ValueError('Spatial budget/outcome accounting differs')
    return dict(calls=counts, receipt_sha256=file_digest(output / 'complete.json'))
