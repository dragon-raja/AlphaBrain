"""Full-task confirmation adapter, sharing the durable host process lifecycle."""
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.evaluation_matrix import flow_noise
from AlphaBrain.research.dsol.data.rollout_replay import array_identity
from AlphaBrain.research.dsol.execution.queue import MissingResult
from .probe_queue import ProbeBackend
from .xr1_evaluation import noise_seed


class ConfirmationBackend(ProbeBackend):
    def admit(self, slot):
        if slot['host'] != self.runtime['host']:
            raise ValueError('Wrong host coordinator')
        binding = self.plan.get('resource_admission')
        if binding and sha(binding['path']) != binding['sha256']:
            raise ValueError('Resource admission changed')
        receipt = read(binding['path'] if binding else self.runtime['admission'])
        if binding and receipt.get('host') != self.runtime['host']:
            raise ValueError('Resource admission belongs to another host')
        if 'host_partition' in self.runtime and (receipt.get('host') != self.runtime['host']
                or receipt.get('host_partition_sha256') != sha(self.runtime['host_partition'])):
            raise ValueError('Admission belongs to another host partition')
        if (receipt['status'] != 'PASS' or receipt['model'] != self.runtime['model']
                or receipt['plan_sha256'] != sha(self.runtime['plan'])
                or receipt['source_sha256'] != sha(Path(self.runtime['source'])/
                                                    'benchmarks/Robocasa365/confirmation_execution.py')
                or not receipt['complete_task_attempts_verified']):
            raise ValueError('Actual full-task confirmation admission missing')
        return self.gpu_available(slot)

    def worker_command(self, task):
        args = [self.runtime['python'], '-u', '-m', 'benchmarks.Robocasa365.confirmation_execution',
                '--plan', self.runtime['plan'], '--model', self.runtime['model'],
                '--initial', task['payload']['initial'], '--views', *task['payload']['views'],
                '--repeats', *map(str, task['payload']['repeats']), '--output', task['output'], '--execute']
        if self.runtime['model'] == 'xr1':
            args += ['--model-python', self.runtime['model_python'], '--upstream', self.runtime['upstream']]
        return args

    def validate(self, task, *, allow_partial=False):
        root = Path(task['output'])
        if not (root/'status.json').exists():
            raise MissingResult(str(root))
        if not allow_partial and read(root/'status.json')['status'] != 'COMPLETE':
            raise MissingResult(str(root))
        state = read(root/'status.json'); identity = read(root/'identity.json')
        if state['status'] not in ('STARTING', 'RUNNING', 'FAILED', 'COMPLETE'):
            raise ValueError('Unknown endpoint worker state')
        initial, views, repeats = [task['payload'][k] for k in ('initial', 'views', 'repeats')]
        model = self.runtime['model']; plan = read(self.runtime['plan'])
        selection = plan['selections'][model][initial]
        if (identity['model'] != model or identity['initial'] != initial
                or identity['views'] != views or identity['repeats'] != repeats
                or identity['noise_stream'] != 'execution'
                or identity['plan_sha256'] != sha(self.runtime['plan'])
                or identity['selection_sha256'] != selection['sha256']
                or identity['source_sha256'] != sha(Path(self.runtime['source'])/
                                                    'benchmarks/Robocasa365/confirmation_execution.py')
                or (not allow_partial and state['completed_rollouts'] != len(views)*len(repeats))):
            raise ValueError('Full-task result identity or coverage mismatch')
        if sha(selection['path']) != selection['sha256']:
            raise ValueError('Selected-view file changed after execution')
        native_source = 'evaluation_matrix.py' if model == 'official' else 'xr1_evaluation.py'
        if identity['rollout_source_sha256'] != sha(Path(self.runtime['source'])/'benchmarks/Robocasa365'/native_source):
            raise ValueError('Native rollout implementation changed')
        selected = read(selection['path']); horizon = plan['task_horizons'][selected['task']]
        expected = {root/'cells'/initial/v/f'execution_{r}.json' for v in views for r in repeats}
        actual = set((root/'cells').glob('*/*/execution_*.json'))
        if not actual <= expected or (not allow_partial and actual != expected):
            raise ValueError('Unexpected or missing complete-task trial')
        if (not 0 <= state['completed_rollouts'] <= len(actual)
                or (state['status'] == 'COMPLETE' and (actual != expected or state['completed_rollouts'] != len(expected)))):
            raise ValueError('Worker completion counter exceeds committed trials')
        receipts = []
        for path in sorted(actual):
            trial = read(path)
            if (trial['model'] != model or trial['initial'] != initial or trial['view'] != path.parent.name
                    or path.name != f'execution_{trial["repeat"]}.json'
                    or trial['noise_stream'] != 'execution'
                    or trial['selection_sha256'] != selection['sha256']
                    or trial['horizon'] != horizon or not 1 <= trial['steps'] <= horizon
                    or type(trial['success']) is not bool or not trial['complete_task_attempt']
                    or not trial['plans']):
                raise ValueError('Invalid complete-task trial identity')
            if not (trial['success'] or trial['terminated'] or trial['truncated'] or trial['steps'] == horizon):
                raise ValueError('Incomplete task reported as a complete attempt')
            stride = 25 if model == 'official' else 16
            if [p['step'] for p in trial['plans']] != list(range(0, trial['steps'], stride)):
                raise ValueError('Native replan cadence changed')
            for k, replan in enumerate(trial['plans']):
                if model == 'official':
                    valid = replan['noise'] == array_identity(flow_noise(initial, 'execution', trial['repeat'], k))
                else:
                    valid = replan['seed'] == noise_seed(initial, 'execution', trial['repeat'], k)
                if not valid:
                    raise ValueError('Execution noise is not the paired independent address')
            if model == 'xr1' and sha(path.with_name(f'trace_{trial["repeat"]}.npz')) != trial['trace_sha256']:
                raise ValueError('XR-1 trace changed')
            receipts.append(dict(path=str(path), sha256=sha(path)))
        return dict(status='PASS' if actual == expected else 'PASS_PARTIAL',
                    identity_sha256=sha(root/'identity.json'), trials=receipts,
                    scientific_failure_is_valid=True)


def factory(plan, root):
    if 'host_bootstrap' in plan:
        from .host_bootstrap import TransferredHostBackend
        return TransferredHostBackend(plan, root)
    return ConfirmationBackend(plan, root)


class PilotBackend(ConfirmationBackend):
    """Bounded actual admission using the same worker, noise, and result validator."""
    def admit(self, slot):
        from AlphaBrain.research.dsol.data.confirmation_execution import assigned_initials
        endpoint = read(self.runtime['plan'])
        tasks = self.plan['tasks']
        if (slot['host'] != self.runtime['host'] or len(tasks) != 1
                or tasks[0]['payload'] != dict(initial=assigned_initials(self.runtime, endpoint)[0],
                                               views=['canonical', 'extra_002'], repeats=list(range(8)))
                or len(tasks[0]['addresses']) != 16 or len(self.plan['slots']) != 1):
            raise ValueError('Pilot may only admit the predetermined 16 endpoint trials')
        return self.gpu_available(slot)


def pilot_factory(plan, root):
    return PilotBackend(plan, root)


def run_program(runtime_file, control_root):
    """Durable pilot -> audited receipt -> full queue; no retry or score-based gate."""
    import time
    from AlphaBrain.research.dsol.data.confirmation_execution import build_queue
    from AlphaBrain.research.dsol.execution.cli import main
    runtime = read(runtime_file); control = Path(control_root)
    control.mkdir(parents=True, exist_ok=False)
    status_file = control/'program-status.json'
    status = dict(status='PILOT', started_unix=time.time(), runtime_sha256=sha(runtime_file))
    write(status_file, status)
    try:
        pilot = build_queue(runtime_file, control/'pilot', pilot=True)
        main(['run', '--root', str(control/'pilot'), '--execute', '--timeout-seconds', '7200'])
        receipt = ConfirmationBackend(pilot, control/'pilot').validate(pilot['tasks'][0])
        write(runtime['admission'], dict(status='PASS', model=runtime['model'], host=runtime['host'],
              host_partition_sha256=sha(runtime['host_partition']) if 'host_partition' in runtime else None,
              plan_sha256=sha(runtime['plan']),
              source_sha256=sha(Path(runtime['source'])/'benchmarks/Robocasa365/confirmation_execution.py'),
              complete_task_attempts_verified=True, pilot_initial=pilot['tasks'][0]['payload']['initial'],
              pilot_views=['canonical', 'extra_002'], pilot_repeats=list(range(8)),
              pilot_manifest_sha256=sha(control/'pilot/manifest.json'), validation=receipt,
              success_rate_not_used_for_admission=True, admitted_unix=time.time()))
        full = build_queue(runtime_file, control/'full')
        status.update(status='FULL_RUNNING', admitted_unix=time.time(), blocks=len(full['tasks']))
        write(status_file, status)
        main(['run', '--root', str(control/'full'), '--execute', '--timeout-seconds', '432000'])
        status.update(status='COMPLETE', finished_unix=time.time())
        write(status_file, status)
    except BaseException as exc:
        status.update(status='FAILED', finished_unix=time.time(), error=repr(exc))
        write(status_file, status)
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--control-root', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    run_program(args.runtime, args.control_root)
