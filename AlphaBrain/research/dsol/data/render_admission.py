"""Load only independently confirmed, provenance-bound numerical gate profiles."""

from pathlib import Path
from .artifacts import read, sha


def confirmed_pixel_gate(result_file):
    result_file = Path(result_file)
    result = read(result_file)
    proposal = result['proposal']
    if proposal['status'] != 'VALIDATED_FIXED_PROPOSAL_NOT_APPLIED':
        raise ValueError('An independently confirmed fixed proposal is required')
    limits = dict(max_abs=2, fraction_gt1=.0001, fraction_gt2=0.)
    if proposal['unchanged_limits'] != limits or proposal['proposed_mean_abs'] not in (.002, .005, .01):
        raise ValueError('Hard limits changed or unapproved mean candidate')
    root = Path(result['source_directory'])
    if sha(root / 'plan.json') != result['plan_sha256']:
        raise ValueError('Confirmation plan changed')
    plan = read(root / 'plan.json')
    status = read(root / 'status.json')
    expected_jobs = {f"{r['role']}/{r['task']}/process-{i}" for r in plan['initials'] for i in range(2)}
    actual_jobs = [r['job'] for r in status['records']]
    if (plan['mode'] != 'confirmation' or plan['fixed_mean_abs'] != proposal['proposed_mean_abs']
            or not status['complete'] or any(r['returncode'] for r in status['records'])
            or len(actual_jobs) != len(expected_jobs) or set(actual_jobs) != expected_jobs
            or proposal['validation_hard_failures'] != 0
            or proposal['validation_max_mean_abs'] > proposal['proposed_mean_abs']):
        raise ValueError('Confirmation is incomplete or inconsistent')
    return dict(mean_abs=proposal['proposed_mean_abs'], **limits,
                confirmation_result=str(result_file.resolve()), confirmation_sha256=sha(result_file),
                interpretation='Numerical image admission, not proof of identical policy actions')
