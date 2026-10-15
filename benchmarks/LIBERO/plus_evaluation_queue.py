"""Native Plus full-rollout production on the standard queue, after pilot.

The first official view at initial0 is retained from admission for both models.
Canonical admission episodes are engineering checks, not extra Plus trials.
"""
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.execution.queue import MissingResult
from benchmarks.Robocasa365.probe_queue import ProbeBackend


def validate_result(config_file, initial, model, output):
    config = read(config_file); output = Path(output)
    if not (output/'status.json').exists() or read(output/'status.json')['status'] != 'COMPLETE':
        raise MissingResult(str(output))
    status = read(output/'status.json'); source = Path(config['source'])
    if (status['initial'] != initial or status['model'] != model or status['config_sha256'] != sha(config_file)
            or status['source_sha256'] != sha(source/'benchmarks/LIBERO/window_probe.py')
            or status['simulator_sha256'] != sha(source/'benchmarks/LIBERO/plus_evaluate.py')
            or status['arrays_sha256'] != sha(output/'probes.npz')):
        raise ValueError('Native Plus result identity changed')
    rendered = Path(config['render_root'])/f'{initial:02d}'
    records = read(rendered/'records.json'); views = ['canonical'] + [x['request_id'] for x in records['requests']]
    indices = config.get('execution_view_indices', list(range(1, len(views))))
    indices = [j for j in indices if j not in config.get('reused_rollout_indices', {}).get(str(initial), [])]
    rows = read(output/'episodes.json')['rows']
    expected = [(views[j], repeat) for j in indices for repeat in range(4)]
    if [(r['view'], r['repeat']) for r in rows] != expected:
        raise ValueError('Missing, duplicated, or extra native Plus episode')
    for row in rows:
        if (row['initial'] != initial or row['model'] != model or row['pair_key'] != records['pair_key']
                or not row['initial_input_exact'] or row['steps'] > row['max_steps']
                or len(row['calls']) != (row['steps']+4)//5):
            raise ValueError('Invalid native Plus episode receipt')
    with np.load(output/'probes.npz', allow_pickle=False) as z:
        if (z['views'].tolist() != views or z['scores'].shape != (len(views), 8)
                or not np.isfinite(z['scores']).all() or z['execution_view_indices'].tolist() != indices
                or not np.array_equal(z['success'], np.asarray([r['success'] for r in rows]).reshape(len(indices), 4))):
            raise ValueError('Native Plus array/episode disagreement')
    return dict(status='PASS', status_sha256=sha(output/'status.json'), episodes_sha256=sha(output/'episodes.json'), episodes=len(rows))


class PlusEvaluationBackend(ProbeBackend):
    def admit(self, slot):
        if slot['host'] != self.runtime['host']:
            raise ValueError('Wrong host coordinator')
        binding = self.plan.get('resource_admission')
        if binding is not None:
            import platform
            if sha(binding['path']) != binding['sha256']:
                raise ValueError('Transferred-host admission changed')
            proof = read(binding['path'])
            if (proof['status'] != 'PASS' or proof['host'] != platform.node()
                    or proof['host_label'] != self.runtime['host']
                    or proof['production_config_sha256'] != sha(self.runtime['config'])
                    or sha(proof['python_wrapper']) != proof['python_wrapper_sha256']
                    or {r['model'] for r in proof['rows']} != {'canonical', 'broad'}
                    or not all(r['initial_scores_exact'] and r['action_and_noise_hashes_exact']
                               and r['completion_steps_exact'] and r['successes_exact'] for r in proof['rows'])):
                raise ValueError('Exact two-model host admission required')
        return self.gpu_available(slot)

    def worker_command(self, task):
        return [self.runtime['python'], '-u', '-m', 'benchmarks.LIBERO.plus_evaluate',
            '--config', self.runtime['config'], '--initial', str(task['payload']['initial']),
            '--model', task['payload']['model'], '--output', task['output'], '--execute']

    def validate(self, task):
        return validate_result(self.runtime['config'], task['payload']['initial'], task['payload']['model'], task['output'])


def factory(plan, root):
    return PlusEvaluationBackend(plan, root)


def prepare(config_file, control):
    from AlphaBrain.research.dsol.execution.manifest import task_key
    from AlphaBrain.research.dsol.execution.queue import prepare as prepare_queue
    from benchmarks.LIBERO.plus_render import requests_for_initial
    config = read(config_file); root = Path(config['root']); source = Path(config['source'])
    admission = read(config['admission_config'])
    for model in ('canonical', 'broad'):
        validate_result(config['admission_config'], 0, model, Path(config['admission_root'])/model/'00')
    records = read(config['inputs'])['records']; requests = read(config['requests'])['requests']
    if len(records) != 32 or len(requests) != 1364:
        raise ValueError('Authorized Plus scope changed')
    fingerprint = sha(config_file); tasks = []
    for initial, record in enumerate(records):
        variants = requests_for_initial(requests, record)
        for model in ('canonical', 'broad'):
            addresses = [dict(model=model, task=record['state']['task_id'], initial=record['state']['pair_key'],
                view=r['request_id'], stream='execution', repeat=repeat, noise=f'INITIAL32::{record["state"]["pair_key"]}::{repeat}')
                for j, r in enumerate(variants, 1) if j not in config['reused_rollout_indices'].get(str(initial), [])
                for repeat in range(4)]
            tasks.append(dict(key=task_key(config['experiment_id'], fingerprint, addresses), addresses=addresses,
                payload=dict(initial=initial, model=model), output=str(root/'plus-evaluation'/model/f'{initial:02d}')))
    if sum(len(t['addresses']) for t in tasks) + 8 != 10912:
        raise ValueError('Pilot and production do not cover exactly10912 Plus episodes')
    paths = [Path(config_file), Path(config['inputs']), Path(config['requests']), Path(config['admission_config'])]
    paths += [Path(config['admission_root'])/model/'00'/'status.json' for model in ('canonical', 'broad')]
    paths += sorted(p for directory in ('AlphaBrain', 'scripts', 'benchmarks') for p in (source/directory).rglob('*.py'))
    plan = dict(format='dsol_work_queue_v1', experiment_id=config['experiment_id'], protocol_sha256=fingerprint,
        tasks=tasks, slots=[dict(host='fresh', id=str(gpu)) for gpu in [4, 5, 6, 7]],
        artifacts=[dict(path=str(p), sha256=sha(p)) for p in paths],
        backend=dict(module='benchmarks.LIBERO.plus_evaluation_queue', factory='factory'),
        runtime=dict(host='fresh', source=str(source), config=str(config_file), python=config['policy_python'],
            workers_per_gpu=1, environment=config['policy_environment']))
    prepare_queue(plan, control)
    write(root/'plus-evaluation-ownership.json', dict(control=str(control), planned_episodes=10912,
        retained_plus_episodes=8, production_episodes=10904, canonical_admission_episodes_excluded=8,
        config_sha256=fingerprint))
    return plan
