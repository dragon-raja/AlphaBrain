"""Read-only completeness and convention audit of a resident engineering pilot."""
from pathlib import Path
import math

from ..data.artifacts import read, sha
from ..data.evaluation_matrix import validate_plan, flow_noise
from ..data.rollout_replay import array_identity


def audit_pilot(plan_file, root):
    plan_file, root = Path(plan_file), Path(root)
    plan = read(plan_file)
    reference = validate_plan(plan)
    plan_hash = sha(plan_file)
    jobs = {j['initial_id']: j for j in reference['jobs']}
    errors, records, workers = [], [], []
    for worker in plan['workers']:
        directory = root/worker['worker_id']
        progress_file = directory/'progress.json'
        if not progress_file.exists() or read(progress_file)['status'] != 'COMPLETE':
            errors.append(worker['worker_id'] + ': not complete')
            continue
        times, steps, cell_times = [], [], []
        for initial in worker['pilot_initials']:
            for view in plan['pilot_view_ids']:
                cell = directory/'cells'/initial/view
                identity = dict(plan_sha256=plan_hash, worker_id=worker['worker_id'], pilot=True,
                                arm=worker['arm'], initial_id=initial, view_id=view,
                                snapshot_manifest_sha256=jobs[initial]['snapshot_manifest_sha256'])
                try:
                    complete, scores = read(cell/'complete.json'), read(cell/'scores.json')
                    if (complete['provenance'] != identity or scores['provenance'] != identity
                            or not complete['complete'] or len(complete['trials']) != 4
                            or complete['score_metadata_sha256'] != sha(cell/'scores.json')
                            or scores['arrays_sha256'] != sha(cell/'scores.npz') or len(scores['rows']) != 8):
                        raise ValueError('Cell identity, score or payload mismatch')
                    for index, score in enumerate(scores['rows']):
                        if score['repeat'] != index or score['noise'] != array_identity(flow_noise(initial, 'score', index)):
                            raise ValueError('Score noise mismatch')
                    cell_times.append((cell/'complete.json').stat().st_mtime)
                    for index, trial_ref in enumerate(complete['trials']):
                        file = cell/f'execution_{index}.json'
                        trial = read(file)
                        if (trial_ref['file'] != file.name or trial_ref['sha256'] != sha(file)
                                or trial['provenance'] != identity or trial['repeat'] != index
                                or not trial['complete_task_attempt']
                                or trial['horizon'] != plan['task_horizons'][jobs[initial]['task']]
                                or not 0 < trial['steps'] <= trial['horizon']
                                or len(trial['plans']) != math.ceil(trial['steps']/25)
                                or trial['score_arrays_sha256'] != scores['arrays_sha256']
                                or trial['plans'][0]['input_arrays'] != scores['input_arrays']):
                            raise ValueError('Rollout identity, timing or first-input mismatch')
                        for replan, query in enumerate(trial['plans']):
                            if query['step'] != replan*25 or query['noise'] != array_identity(
                                    flow_noise(initial, 'execution', index, replan)):
                                raise ValueError('Execution noise address mismatch')
                        times.append(trial['seconds']); steps.append(trial['steps'])
                        records.append(dict(worker=worker['worker_id'], host=worker['host'],
                            initial=initial, view=view, repeat=index, seconds=trial['seconds'], steps=trial['steps']))
                except (OSError, ValueError, KeyError) as exc:
                    errors.append(f'{worker["worker_id"]}/{initial}/{view}: {exc}')
        if steps:
            per_step = sum(times)/sum(steps)
            workload = sum(plan['task_horizons'][jobs[i]['task']] for i in worker['initials'])*97*4
            workers.append(dict(worker_id=worker['worker_id'], seconds_per_step=per_step,
                full_horizon_steps=workload, full_horizon_execution_days=workload*per_step/86400,
                consecutive_cell_seconds=cell_times[-1]-cell_times[0] if len(cell_times)>1 else None))
    return dict(status='PASS' if not errors and len(records)==len(plan['workers'])*2*4 else 'HOLD',
        errors=errors, full_rollouts=len(records), workers=workers,
        scope='Engineering pilot only; throughput extrapolation is not a success-rate estimate or deadline guarantee',
        plan_sha256=plan_hash, maximum_worker_full_horizon_execution_days=max(
            (w['full_horizon_execution_days'] for w in workers), default=None),
        capacity_caveat='Seen pilot scenes only; full horizon assumes no early success; scoring/IO/startup and scene-dependent cost remain')
