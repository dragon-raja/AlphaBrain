"""Ledger-owned final-checkpoint searches and paired manipulation evaluation.

The two independent validation scores are recorded after selection and never
fed to the selector. Nominal-return reuse requires exact physical/input proof.
"""
from contextlib import ExitStack
from pathlib import Path
import json
import time
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read, write
from AlphaBrain.research.dsol.data import native_recovery
from AlphaBrain.research.dsol.data import reward_pilot
from AlphaBrain.research.dsol.data import coordinate_probe
from AlphaBrain.research.dsol.data import coordinate_validation
from AlphaBrain.research.dsol.data import independent_random
from AlphaBrain.research.dsol.data.training_stage import evaluation_endpoint
from AlphaBrain.research.dsol.execution.manifest import digest, file_digest, task_key


def common_asset_hashes(baseline_campaign):
    """Bind exact adapter paths, including aliases absent from baseline hashes."""
    baseline=read(baseline_campaign);cfg=read(baseline['paired_config'])
    spec=cfg['cohorts']['A'];sim=read(spec['sim_config'])
    from .search_runtime import selected_checkpoint
    model=selected_checkpoint(sim,allow_native=True)
    paths=[baseline_campaign,baseline['paired_config'],baseline['score_noise'],
        spec['sim_config'],spec['execution_noise'],sim['inputs'],cfg['design'],
        str(Path(model['path'])/'model.safetensors'),
        str(Path(model['path'])/'assets/physical-intelligence/libero/norm_stats.json')]
    return {str(p):file_digest(p) for p in paths}


def validate_campaign(path, *, block_index=None):
    """Validate the whole contract, then hash all or one block's references.

    Controllers use the full default preflight. Workers repeat global input
    hashes and contract checks, but only need baseline bytes for their own block.
    No hash is trusted from an mtime cache or an earlier process.
    """
    c = read(path)
    mechanism=c['schema']==coordinate_probe.SCHEMA
    full_mechanism=c['schema']==coordinate_validation.SCHEMA
    adaptive=c['schema']==reward_pilot.RECOVERY_SCHEMA
    random_acquisition=c['schema']==independent_random.SCHEMA
    protocol=reward_pilot.from_campaign(c) if adaptive else None
    if not random_acquisition:evaluation_endpoint(c)
    if block_index is not None and (type(block_index) is not int or not 0 <= block_index < len(c['blocks'])):
        raise ValueError('Invalid recovery validation block index')
    if (c['schema'] not in (native_recovery.SCHEMA,reward_pilot.RECOVERY_SCHEMA,coordinate_probe.SCHEMA,coordinate_validation.SCHEMA,independent_random.SCHEMA) or c['model'] != 'native_reference'
            or c['score_scale'] != .16535300100378683 or c['query_cap'] != 8
            or c['selection_members'] != [0, 1] or c['validation_members'] != [2, 3]):
        raise ValueError('Recovery scientific contract changed')
    for p, h in c['input_hashes'].items():
        if file_digest(p) != h:
            raise ValueError('Recovery input changed: ' + p)
    required = {c[k] for k in ('baseline_campaign', 'schedule', 'paired_config', 'score_noise', 'baseline_references')}
    baseline = read(c['baseline_campaign']); cfg = read(c['paired_config'])
    spec = cfg['cohorts']['A']; sim = read(spec['sim_config'])
    required.update([spec['sim_config'], spec['execution_noise'], sim['inputs'], cfg['design']])
    from .search_runtime import selected_checkpoint
    model = selected_checkpoint(sim, allow_native=True)
    required.update([str(Path(model['path']) / 'model.safetensors'),
                     str(Path(model['path']) / 'assets/physical-intelligence/libero/norm_stats.json')])
    if (c['paired_config'] != baseline['paired_config'] or c['score_noise'] != baseline['score_noise']
            or read(cfg['design'])['operation_policy']['checkpoint_sha256'] != model['weights_sha256']):
        raise ValueError('Recovery must share baseline model, simulator and noises')
    planned = (coordinate_validation.planned_rows(baseline,read(sim['inputs'])['records']) if full_mechanism else
               coordinate_probe.planned_rows(baseline,read(sim['inputs'])['records']) if mechanism
               else native_recovery.rows(baseline['states'], read(sim['inputs'])['records']))
    if mechanism:
        coordinate_probe.validate_contract(c,planned)
        required.add(c['mechanism_protocol'])
    if full_mechanism:
        coordinate_validation.validate_contract(c,planned)
        required.add(c['mechanism_protocol'])
    if adaptive:planned=reward_pilot.evaluation_rows(protocol,planned)
    if random_acquisition:
        independent_random.validate_contract(c);planned=independent_random.rows(planned)
        required.add(c['independent_random_protocol'])
    if read(c['schedule']) != planned:
        raise ValueError('Recovery schedule differs from outcome-independent panel')
    planned_by_address = {(r['state_key'], r['view_id'], r['method']): r for r in planned}
    if len(planned_by_address) != len(planned):
        raise ValueError('Duplicate planned recovery address')
    seen = set(); actual = []
    for index, block in enumerate(c['blocks']):
        if (block['index'] != index or block['limits'] != native_recovery.limits(block['rows'],paired_diagnostic=adaptive or random_acquisition)
                or block['addresses'] != native_recovery.addresses(block['rows'])):
            raise ValueError('Recovery block budget or address changed')
        for row in block['rows']:
            address = (row['state_key'], row['view_id'], row['method'])
            if address in seen or row != planned_by_address.get(address):
                raise ValueError('Repeated or unplanned recovery row')
            seen.add(address); actual.append(row)
            if not random_acquisition and (row.get('actor_key') if full_mechanism else row['method'] not in ('local', 'random_from_local')):
                actor = c['actors'][row['method']]; required.update([actor['path'], actor['controls']])
    if c['limits'] != native_recovery.limits(actual,paired_diagnostic=adaptive or random_acquisition):
        raise ValueError('Recovery campaign total budget changed')
    refs = read(c['baseline_references'])
    if refs['status'] != 'PASS_COMPLETE_BASELINE_IDENTITIES':
        raise ValueError('Baseline references are incomplete')
    referenced_rows = actual if block_index is None else c['blocks'][block_index]['rows']
    reference_assets=coordinate_validation.reference_artifacts if full_mechanism else native_recovery.reference_artifacts
    for p, h in reference_assets(refs, referenced_rows).items():
        if file_digest(p) != h:
            raise ValueError('Referenced baseline artifact changed')
    if not required.issubset(c['input_hashes']):
        raise ValueError('Unbound recovery asset')
    return c


def request_for(path, index):
    c = read(path); b = c['blocks'][index]
    if b['index'] != index:
        raise ValueError('Unordered recovery block')
    return dict(campaign=str(path), campaign_sha256=file_digest(path), block=b, host=b['host'], stage='formal')


def validate_spec(binding, key):
    s = binding['tasks'][key]; r = s['request']; b = r['block']
    if (r != request_for(binding['campaign'], b['index']) or s['limits'] != b['limits']
            or s['addresses'] != b['addresses'] or s['host'] != b['host'] or s['stage'] != 'formal'
            or key != task_key(s['identity_experiment_id'], r['campaign_sha256'], b['addresses'])):
        raise ValueError('Unbound recovery request')
    return b['limits']


def random_visited(info, seed):
    candidates = [e['angles'] for e in info['events'] if e['kind'] in ('initial', 'search')]
    if not candidates:
        raise ValueError('No visited candidates')
    index = int(np.random.default_rng(seed).integers(len(candidates)))
    return candidates[index], index, len(candidates)


def load_actors(c):
    if c.get('schema')==independent_random.SCHEMA:
        if c['actors']:raise ValueError('Independent random acquisition cannot load actors')
        return {}
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from .continuous_evaluation import verify_actor
    torch.set_num_threads(2); result = {}
    updates, episodes = evaluation_endpoint(c)
    for name, spec in c['actors'].items():
        payload = torch.load(spec['path'], map_location='cpu', weights_only=True)
        if name == 'old_ppo':
            if payload['episodes'] != 512 or payload['update'] != 16:
                raise ValueError('Wrong historical actor checkpoint')
        elif payload['update'] != updates or payload['episodes'] != episodes:
            raise ValueError('Only the declared final A/B checkpoint may be evaluated')
        actor = ContinuousActorCritic(4608).eval(); actor.load_state_dict(payload['state_dict'], strict=True)
        actor.requires_grad_(False)
        verify_actor(actor, torch.load(spec['controls'], map_location='cpu', weights_only=True))
        result[name] = actor
    return result


def nominal_round_trip(query, calls, noises, directory, row, reference):
    from .local_view_runtime import RenderedViewSource
    from .local_search_rollout import input_identity
    from .search_domain import exact_input
    source = RenderedViewSource(query, calls, noises, directory)
    initial = query(dict(kind='query_pose'))
    source._receipt(initial['receipt'])
    if input_identity(initial['example']) != reference['input_sha256']:
        raise ValueError('Canonical input differs from baseline')
    source.move_angles(row['angles_deg']); source.move_angles([0., 0.])
    returned = query(dict(kind='query_pose')); source._receipt(returned['receipt'])
    exact_input(initial['example'], returned['example'])
    write(directory / 'nominal-return.json', dict(status='PASS_EXACT_INPUT_AND_PHYSICAL_STATE',
          baseline_input_sha256=reference['input_sha256'], physical_state_id=source.physical,
          movement_setpoints=source.moves, model_calls=0))


def run(request, output, *, guard, runtime_cache=None, simulator_pool=None):
    from AlphaBrain.research.dsol.execution.dispatch_policy import reject_production
    reject_production('benchmarks/LIBERO/controlled_recovery.py:run')
    c = validate_campaign(request['campaign'], block_index=request['block']['index'])
    if request != request_for(request['campaign'], request['block']['index']):
        raise ValueError('Changed recovery request')
    from .continuous_evaluation import EvaluationCalls, append_event, search, move_to_search_selection
    from .local_view_runtime import RenderedViewSource
    from .search_live import simulator
    from .search_runtime import create_policy_runtime, bind_policy_task
    from .local_search_rollout import rollout_local_episode, input_identity
    from .search_domain import exact_input
    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    cfg = read(c['paired_config']); spec = cfg['cohorts']['A']; sim = read(spec['sim_config']); plan = read(cfg['design'])
    bank = ExplicitFlowNoiseBank(Path(spec['execution_noise']), verify_file=True)
    with np.load(c['score_noise'], allow_pickle=False) as z: noises = z['noise'].copy()
    refs = read(c['baseline_references'])['views']; actors = load_actors(c)
    limits = request['block']['limits']; cap = dict(score=limits['score'], execution=limits['paired_execution'])
    open_sim = simulator if simulator_pool is None else simulator_pool.session
    tasks = []; searches = []; started = time.monotonic()
    with ExitStack() as stack:
        if runtime_cache is None: runtime = create_policy_runtime(sim, output, sum(cap.values()))
        else:
            runtime, _ = stack.enter_context(runtime_cache.lease(digest(sim),
                lambda: create_policy_runtime(sim, output, sum(cap.values())),
                lambda value: bind_policy_task(value, output, sum(cap.values()))))
        calls = EvaluationCalls(runtime, output, cap, guard=guard)
        for index, row in enumerate(request['block']['rows']):
            folder = output / f'search-{index:03d}'; folder.mkdir()
            identity = {k: row[k] for k in ('state_key', 'task', 'initial', 'view_id', 'method', 'split')}
            calls.identity = identity; calls.stage = 'score'; values = noises[row['initial']]
            with open_sim(spec['sim_config'], sim, row['initial'], folder) as query:
                nominal_round_trip(query, calls, values[:2], folder, row, refs[f"{row['state_key']}::0"])
                method = row.get('selector') or ('local_near_best_accel' if row['method'] in ('local', 'random_from_local') else 'continuous_ppo')
                info, selected = search(dict(row, method=method), query, calls, values[:2], folder,
                                        actors.get(row['method']), c['score_scale'])
                # Historical rows retain their exact baseline proof. The new
                # held-out panel pairs all methods' first arrays in its report.
                with np.load(folder / 'query-01.npz', allow_pickle=False) as z:
                    first = {k: z[k].copy() for k in z.files}
                if c['schema']!=coordinate_validation.SCHEMA:
                    verify_baseline_query(first,refs[f"{row['state_key']}::{row['view_id']}"]['query'])
                info['initial_render_array_sha256']=input_identity(dict(image=[first['external_image'],first['wrist_image']],
                    state=first['robot_state'],camera_intrinsics=first['camera_intrinsics'],
                    camera_to_world_opencv=first['camera_to_world_opencv']))
                check_dir = folder / 'independent-validation'; check_dir.mkdir()
                source = RenderedViewSource(query, calls, values[2:4], check_dir)
                source.current = np.array(info['selected_angles'], dtype=float)
                current = query(dict(kind='query_pose')); source._receipt(current['receipt'])
                exact_input(selected, current['example'])
                if not np.allclose(current['receipt']['camera_to_world'], source.mount.pose(source.current), rtol=0, atol=1e-9):
                    raise ValueError('Selected camera pose changed')
                if row['method'] == 'random_from_local':
                    angles, choice, count = random_visited(info, row['random_seed'])
                    info['local_minimum_angles'] = info['selected_angles']; info['random_candidate_index'] = choice
                    info['random_candidate_count'] = count; info['selected_angles'] = angles
                    source.move_angles(angles)
                source.query(); selected = source.last_example
                info.update(identity, validation_scores=[float(v['score']) for v in source.last_values],
                    validation_queries=1, score_calls=2 * info['queries'] + 2,
                    selected_input_sha256=input_identity(selected), validation_used_for_selection=False)
                if c['schema'] in (reward_pilot.RECOVERY_SCHEMA,independent_random.SCHEMA):
                    source.move_angles(row['angles_deg']);source.query()
                    info.update(reference_validation_scores=[float(v['score']) for v in source.last_values],
                        validation_queries=2,score_calls=2*info['queries']+4,
                        diagnostic_reference_angles=row['angles_deg'])
                write(folder / 'search.json', info); searches.append(info)
            for repeat in (0, 1):
                episode = folder / f'episode-{repeat}'; episode.mkdir(); calls.stage = 'execution'
                with open_sim(spec['sim_config'], sim, row['initial'], episode) as query:
                    source = RenderedViewSource(query, calls, values[:2], episode)
                    move_to_search_selection(source, info); response = query(dict(kind='begin_execution'))
                    exact_input(selected, response['example'])
                    result, arrays = rollout_local_episode(query, calls.infer, response['example'], bank,
                        row['state_key'], repeat, identity, plan=plan, suite=row['suite'],
                        record=lambda event: append_event(episode / 'events.jsonl', event))
                asset = episode / 'trajectory.npz'; np.savez_compressed(asset, **arrays)
                result.update(identity, selected_angles=info['selected_angles'], queries=info['queries'],
                    trajectory=str(asset), trajectory_sha256=file_digest(asset))
                write(episode / 'result.json', result); tasks.append(result)
            write(output / 'progress.json', dict(tasks=len(tasks), searches=len(searches), calls=calls.counts))
        write(output / 'results.json', dict(tasks=tasks, searches=searches))
        write(output / 'status.json', dict(status='COMPLETE', calls=calls.counts, training_updates=0,
              seconds=time.monotonic() - started, phases=calls.timing.rows))


def verify_baseline_query(first,path):
    with np.load(path, allow_pickle=False) as z:
        for key in ('external_image', 'wrist_image', 'robot_state', 'camera_intrinsics', 'camera_to_world_opencv'):
            if not np.array_equal(first[key], z[key]):
                raise ValueError('Search initial input differs from paired baseline: ' + key)
        for member in (0, 1):
            for field in ('actions', 'trace', 'full_trace', 'full_noise'):
                key = f'{field}_{member}'
                if key in z.files and (key not in first or not np.array_equal(first[key], z[key])):
                    raise ValueError('Search initial prediction differs from baseline: ' + key)


def validate_block(request, output):
    output = Path(output); complete = read(output / 'complete.json')
    if complete['status'] != 'COMPLETE' or complete['request_sha256'] != digest(request):
        raise ValueError('Recovery receipt identity changed')
    for name, h in complete['artifacts'].items():
        if file_digest(output / name) != h: raise ValueError('Recovery artifact changed')
    result = read(output / 'results.json'); status = read(output / 'status.json')
    key = lambda r: (r['state_key'], r['view_id'], r['method'])
    expected = {key(r) for r in request['block']['rows']}
    actual = [key(r) for r in result['searches']]
    execution = [(*key(r), r['repeat']) for r in result['tasks']]
    if (set(actual) != expected or len(actual) != len(expected) or len(execution) != 2 * len(expected)
            or set(execution) != {(*k, s) for k in expected for s in (0, 1)}):
        raise ValueError('Missing or repeated recovery outcome')
    attempts = [json.loads(s) for s in (output / 'model-attempts.jsonl').read_text().splitlines()]
    counts = {k: sum(a['identity']['stage'] == k for a in attempts) for k in ('score', 'execution')}
    limits = request['block']['limits']
    if (status['status'] != 'COMPLETE' or status['training_updates'] != 0 or status['calls'] != counts
            or counts['score'] > limits['score'] or counts['execution'] > limits['paired_execution']
            or counts['score'] != sum(s['score_calls'] for s in result['searches'])
            or counts['execution'] != sum(t['replans'] for t in result['tasks'])):
        raise ValueError('Recovery call accounting differs')
    for s in result['searches']:
        paired=read(request['campaign'])['schema'] in (reward_pilot.RECOVERY_SCHEMA,independent_random.SCHEMA)
        if (not 1 <= s['queries'] <= 8 or s['validation_used_for_selection']
                or s['validation_queries'] != (2 if paired else 1)
                or s['score_calls'] != 2*s['queries']+(4 if paired else 2)):
            raise ValueError('Selection and independent validation must remain separate')
        if paired and (len(s['reference_validation_scores'])!=2
                or not np.isfinite(s['reference_validation_scores']).all()):
            raise ValueError('Missing independent reference diagnostics')
    for t in result['tasks']:
        if type(t['success']) is not bool or t['steps'] > t['horizon'] or len(t['execution_noise_sha256']) != t['replans']:
            raise ValueError('Invalid manipulation outcome')
    return dict(calls=counts, receipt_sha256=file_digest(output / 'complete.json'))
