"""CPU preflight for existing LIBERO anchors and Plus camera/initial requests.

No environments or policies are imported. Preparation is not runtime admission
and does not authorize an additional GPU workload.
"""
import hashlib
from pathlib import Path

import numpy as np

from .artifacts import path_mapping, read, sha, write
from ..metrics.accel import shared_flow_noise


def array_signature(array):
    value = np.ascontiguousarray(array)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def score_anchor(ranking, candidate_ids):
    """Restore catalog order from a score-sorted file; never sort by outcomes."""
    rows = ranking['ranking']; ids = [r['candidate_id'] for r in rows]
    if (ranking['status'] != 'PASS' or ranking['ensemble_size'] != 8
            or ranking['candidate_count'] != len(candidate_ids) or len(ids) != len(set(ids))
            or set(ids) != set(candidate_ids) or len(ranking['ensemble_seeds']) != 8
            or len(set(ranking['ensemble_seeds'])) != 8):
        raise ValueError('Incomplete or ambiguous initial score anchors')
    by_id = {r['candidate_id']: r for r in rows}
    members = np.asarray([by_id[v]['member_accel_3'] for v in candidate_ids], dtype=np.float64)
    means = np.asarray([by_id[v]['mean_accel_3'] for v in candidate_ids])
    if members.shape != (len(candidate_ids), 8) or not np.isfinite(members).all():
        raise ValueError('Eight finite member scores per candidate required')
    if not np.array_equal(members.mean(axis=1), means):
        raise ValueError('Original eight-draw score reduction changed')
    noise = np.stack([shared_flow_noise(seed=seed, candidate_count=1,
        action_horizon=ranking['checkpoint_action_horizon'], action_dim=7)[0]
        for seed in ranking['ensemble_seeds']])
    return members, noise


def plus_request(camera, initial, runtime):
    """Preserve the native wrapper request, not just a custom97 pose label."""
    from .libero_camera_inventory import parse_camera_task
    parsed = parse_camera_task(camera['camera_task'])
    if parsed['base_task'] != camera['base_task'] or parsed['robot_initial_variant'] != 0:
        raise ValueError('Unknown base task or robot variant; no automatic substitution')
    parameters = {k: parsed[k] for k in ('horizon', 'vertical', 'distance_scale', 'endpoint_yaw', 'endpoint_pitch')}
    if parameters != camera['camera_parameters']:
        raise ValueError('Camera parameter mapping changed')
    base = Path(runtime)/'libero/libero'
    identity = dict(suite=camera['suite'], camera_task=camera['camera_task'],
                    task_id=initial['task_id'], initial_sha256=initial['array_sha256'])
    import json
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return dict(request_id=key, **identity,
                wrapper_bddl_argument=str(base/'bddl_files'/camera['suite']/(camera['camera_task']+'.bddl')),
                resolved_base_bddl=str(base/'bddl_files'/camera['suite']/(parsed['base_task']+'.bddl')),
                physical_initial=initial, camera_parameters=parameters,
                camera_catalog='official_plus_camera', policy_results_reusable=False,
                status='CPU_REQUEST_ONLY_RUNTIME_NOT_ADMITTED')


def prepare(config_file):
    config = read(config_file); output = Path(config['output'])
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        with path_mapping(config.get('path_mappings', ())):
            root = Path(config['libero_root']); release_file = root/'release.json'
            release = read(release_file); release_sha = sha(release_file)
            if len(release['states']) != 32 or release['candidate_count'] != 97:
                raise ValueError('Original 32-initial / 97-candidate scope required')
            bank_file = release['noise_bank']['path']; bank = read(bank_file)
            if (sha(bank_file) != release['noise_bank']['manifest_sha256']
                    or sha(bank['noise_file']) != release['noise_bank']['file_sha256']
                    or bank['repeat_count'] != 32 or bank['action_horizon'] != 10 or bank['action_dim'] != 7):
                raise ValueError('Historical execution-noise bank changed')
            records = []; score_arrays = []; noise_arrays = []
            for index, state in enumerate(release['states']):
                initialization = state['initialization']
                if sha(initialization['array_path']) != initialization['array_sha256']:
                    raise ValueError('Initial state bytes changed')
                initial = np.load(initialization['array_path'], allow_pickle=False)
                if array_signature(initial) != initialization['array_identity']:
                    raise ValueError('Physical initial shape/dtype/value changed')
                render_file = Path(state['initial_asset_record']); render = read(render_file)
                if (render['status'] != 'PASS' or render['release_sha256'] != release_sha
                        or render['pair_key'] != state['pair_key']
                        or sha(render['artifact']) != render['artifact_sha256']
                        or render['initialization_receipt']['demonstration_state_used']):
                    raise ValueError('Historical observation cache changed')
                with np.load(render['artifact'], allow_pickle=False) as data:
                    views = data['candidate_ids'].tolist()
                    expected_views = [v['selected_candidate_id'] for v in release['protocol']['state_blocks'][index]['candidates']]
                    if views != expected_views:
                        raise ValueError('Candidate order changed')
                    signatures = {r['candidate_id']: r['inputs'] for r in render['input_identities']}
                    for array_name, input_name in [('external_images', 'observation/image'),
                            ('wrist_images', 'observation/wrist_image'), ('camera_intrinsics', 'camera_intrinsics'),
                            ('camera_to_world_opencv', 'camera_to_world_opencv')]:
                        values = data[array_name]
                        for j, view in enumerate(views):
                            if array_signature(values[j]) != signatures[view][input_name]:
                                raise ValueError(f'Cached input identity changed: {index}/{view}/{input_name}')
                    states = data['robot_states']
                    if states.shape != (97, 8) or not np.isfinite(states).all():
                        raise ValueError('Cached observation state malformed')
                anchors = {}; local_scores = []; local_noise = []
                for model in ('canonical', 'broad'):
                    file = root/'scores'/model/'aa0-a'/f'{index:02d}'/'ranking.json'
                    ranking = read(file); completion = read(file.parent/'completion.json')
                    if (completion['ranking_sha256'] != sha(file) or completion['release_sha256'] != release_sha
                            or ranking['pair_key'] != state['pair_key']
                            or ranking['physics_state_sha256'] != render['physics_state_sha256']
                            or ranking['render_artifact_sha256'] != render['artifact_sha256']):
                        raise ValueError('Scoring anchor belongs to another initial/input')
                    scores, noise = score_anchor(ranking, views)
                    anchors[model] = dict(path=str(file), sha256=sha(file),
                        initial_score_seeds=ranking['ensemble_seeds'], initial_noise=array_signature(noise),
                        old_full_velocity_and_action_trace_available=False)
                    local_scores.append(scores); local_noise.append(noise)
                if not np.array_equal(local_noise[0], local_noise[1]):
                    raise ValueError('Historical models used different scoring noise')
                score_arrays.append(local_scores); noise_arrays.append(local_noise[0])
                records.append(dict(initial_index=index, state=state, candidate_ids=views,
                    observation_cache=dict(path=render['artifact'], sha256=render['artifact_sha256'],
                                           receipt=str(render_file), receipt_sha256=sha(render_file)),
                    scoring_anchors=anchors, max_steps=release['max_steps_by_suite'][state['suite']]))
            np.savez_compressed(output/'libero-anchors.npz', member_scores=np.asarray(score_arrays),
                                first_score_noise=np.asarray(noise_arrays))
            write(output/'libero-inputs.json', dict(records=records, models=release['models'],
                  execution_noise_bank=dict(path=bank_file, sha256=sha(bank_file)),
                  native=dict(replan_steps=release['replan_steps'], denoising_steps=release['denoising_steps'],
                              action_horizon=10, action_dim=7, settle_steps=release['settle_steps'],
                              offsamples=release['offsamples'], state_input_enabled=False),
                  arrays_sha256=sha(output/'libero-anchors.npz'), policy_runtime_admitted=False))
        mappings = read(config['plus_mapping']); requests = []
        for camera in mappings:
            for initial in camera['historical_initials']:
                request = plus_request(camera, initial, config['plus_runtime'])
                if (not Path(request['resolved_base_bddl']).is_file()
                        or sha(initial['array_path']) != initial['array_sha256']):
                    raise ValueError('Plus request has missing or changed base/initial')
                requests.append(request)
        if len({r['request_id'] for r in requests}) != len(requests):
            raise ValueError('Duplicate Plus camera/initial address')
        write(output/'plus-requests.json', dict(requests=requests, runtime_admitted=False))
        summary = dict(status='CPU_PREFLIGHT_COMPLETE_GPU_NOT_STARTED',
            sources={str(config_file): sha(config_file), str(release_file): release_sha,
                     config['plus_mapping']: sha(config['plus_mapping']), str(__file__): sha(__file__)},
            libero=dict(initials=32, candidates=97, models=2, image_pose_records_verified=32*97,
                        initial_score_members_verified=32*97*2*8, initial_noise_tensors_reconstructed=32*8,
                        old_execution_repeats=32, replan_steps=release['replan_steps'],
                        five_window_action_steps=[k*release['replan_steps'] for k in range(5)]),
            plus=dict(camera_initial_requests=len(requests), camera_definitions=len({(r['suite'],r['camera_task']) for r in requests}),
                      scope='Only mappings onto the original eight tasks; no new task selection or GPU budget.'),
            gates=['LIBERO independent first-inference replay must match original 8 scores and choices.',
                   'Old initial scores have no full velocity/actions: establish fresh standalone traces before multi-step replay.',
                   'Freeze score-noise addresses for replans1-4; never consume execution-bank noise for selection.',
                   'Plus native runtime pose/RGB and physical-state reset checks remain necessary.',
                   'New multi-worker runs must use the standard queue after actual runtime admission.'],
            caveat='Five replans means native action steps 0/5/10/15/20 here, not RoboCasa pi05 0/25/50/75/100.',
            gpu_jobs_started=0, new_closed_loops=0)
        write(output/'summary.json', summary)
        return summary
    except BaseException as exc:
        write(output/'failure.json', dict(status='FAILED', error=repr(exc), partial_outputs_retained=True))
        raise
