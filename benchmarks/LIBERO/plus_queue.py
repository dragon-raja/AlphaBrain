"""Native Plus rendering on the shared durable queue, not a new scheduler.

Compatibility: reuse the process-only Robocasa adapter until it is extracted
into the shared runtime layer; no Robocasa simulator or policy is instantiated.
"""
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read, sha
from AlphaBrain.research.dsol.execution.queue import MissingResult
from benchmarks.Robocasa365.probe_queue import ProbeBackend
from .plus_render import validate_output


class PlusRenderBackend(ProbeBackend):
    def admit(self, slot):
        if slot['host'] != self.runtime['host']:
            raise ValueError('Host-local queue ownership mismatch')
        first = validate_output(self.runtime['config'], self.runtime['admission_initial'], self.runtime['admission'])
        if first['status'] != 'COMPLETE':
            raise ValueError('Real Plus renderer admission missing')
        return self.gpu_available(slot)

    def worker_command(self, task):
        return [self.runtime['python'], '-u', '-m', 'benchmarks.LIBERO.plus_render',
                '--config', self.runtime['config'], '--initial', str(task['payload']['initial']),
                '--output', task['output'], '--execute']

    def validate(self, task):
        if not (Path(task['output'])/'status.json').exists():
            raise MissingResult(task['output'])
        validate_output(self.runtime['config'], task['payload']['initial'], task['output'])
        return dict(status='PASS', receipt_sha256=sha(Path(task['output'])/'status.json'))


def factory(plan, root):
    return PlusRenderBackend(plan, root)


def prepare_render_queue(config_file):
    """Prepare remaining 31 initials; the complete pilot is reused, never rerun."""
    from AlphaBrain.research.dsol.execution.manifest import task_key
    from AlphaBrain.research.dsol.execution.queue import prepare
    from .plus_render import requests_for_initial
    config = read(config_file); root = Path(config['root']); source = Path(config['source'])
    validate_output(config_file, 0, root/'render'/'00')
    inputs = read(config['inputs']); requests = read(config['requests'])['requests']
    if len(inputs['records']) != 32 or len(requests) != 1364:
        raise ValueError('Authorized rendering scope changed')
    fingerprint = sha(config_file); tasks = []
    for initial in range(1, 32):
        record = inputs['records'][initial]
        variants = requests_for_initial(requests, record)
        addresses = [dict(model='observation_cache', task=record['state']['task_id'],
            initial=record['state']['pair_key'], view=r['request_id'], stream='score', repeat=0,
            noise='deterministic_render_no_policy_noise') for r in variants]
        tasks.append(dict(key=task_key(config['experiment_id'], fingerprint, addresses),
            addresses=addresses, payload=dict(initial=initial), output=str(root/'render'/f'{initial:02d}')))
    paths = [Path(config_file), Path(config['inputs']), Path(config['requests'])]
    # Verify the entire frozen implementation at launch, not just its entrypoint.
    paths += sorted(p for directory in ['AlphaBrain', 'scripts', 'benchmarks']
                    for p in (source/directory).rglob('*.py'))
    plan = dict(format='dsol_work_queue_v1', experiment_id=config['experiment_id'],
        protocol_sha256=fingerprint, tasks=tasks,
        slots=[dict(host='fresh', id=str(gpu)) for gpu in range(8)],
        artifacts=[dict(path=str(p), sha256=sha(p)) for p in paths],
        backend=dict(module='benchmarks.LIBERO.plus_queue', factory='factory'),
        runtime=dict(host='fresh', config=str(config_file), python=config['sim_python'],
            source=str(source), admission=str(root/'render'/'00'), admission_initial=0,
            workers_per_gpu=1, environment=dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                PYTHONPATH=':'.join([config['sim_site'], str(source), str(source/'scripts/vla_shared')]
                                    + config['sim_extra_paths']))))
    return prepare(plan, root/'render-queue')
