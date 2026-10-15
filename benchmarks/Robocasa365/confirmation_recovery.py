"""Explicit missing-only recovery without changing the admitted worker source.

Prepare on the owning host after the coordinator and every owned worker have
stopped. Existing outputs remain immutable; new outputs have separate owners.
This module never starts workers or signals a process.
"""
from copy import deepcopy
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.execution.manifest import task_key, verify_artifacts
from AlphaBrain.research.dsol.execution.queue import exclusive, load, prepare
from .confirmation_queue import ConfirmationBackend
from .probe_queue import ProbeBackend


def prepare_missing(old_control, new_control, new_output):
    old_control, new_control, new_output = map(Path, (old_control, new_control, new_output))
    if new_control.exists() or new_output.exists():
        raise FileExistsError('Recovery requires new control and output directories')
    with exclusive(old_control):
        manifest, state = load(old_control)
        verify_artifacts(manifest)
        module=manifest['backend']['module']
        is_probe=module=='benchmarks.Robocasa365.probe_queue'
        if module not in ('benchmarks.Robocasa365.confirmation_queue','benchmarks.Robocasa365.probe_queue') or manifest['backend']['factory']!='factory':
            raise ValueError('Only admitted full endpoint/probe queues may use missing-only recovery')
        backend = (ProbeBackend if is_probe else ConfirmationBackend)(manifest, old_control)
        # Audit all process identities before reading or assigning any outputs.
        for task in manifest['tasks']:
            row = state['tasks'][task['key']]
            if row['status'] == 'COMPLETE':
                continue  # Verified below; a recycled completed PID is not our live worker.
            handle = row.get('handle')
            if row['status'] == 'LAUNCHING':
                handle = backend.locate(task, row['slot'], row['attempt'])
            previous_boot = handle.get('boot') if handle is not None else None
            current_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            if previous_boot is not None and previous_boot != current_boot:
                continue  # Reboot proves the old process cannot still exist; never signal the new PID.
            if handle is not None and backend.alive(handle):
                raise RuntimeError('Owned worker still alive; do not duplicate trials')
        kept, tasks, assets = [], [], {}
        experiment = manifest['experiment_id']+'-missing-only'
        for task in manifest['tasks']:
            output = Path(task['output']); existing = {}
            if (output/'identity.json').exists():
                receipt = backend.validate(task, allow_partial=True)
                if is_probe:
                    # A committed short cell contains all eight five-window traces.
                    # In-flight repeats without a cell receipt are not durable scores.
                    for view in receipt['cells']:
                        cell=output/'cells'/task['payload']['initial']/view
                        for path in cell.iterdir():
                            if path.is_file():assets[str(path)]=sha(path)
                        for address in task['addresses']:
                            if address['view']==view:
                                existing[(view,address['repeat'])]=dict(path=str(cell/'complete.json'),sha256=sha(cell/'complete.json'))
                for row in ([] if is_probe else receipt['trials']):
                    path = Path(row['path']); trial = read(path)
                    existing[(trial['view'], trial['repeat'])] = row
                    assets[str(path)] = row['sha256']
                    if manifest['runtime']['model'] == 'xr1':
                        trace = path.with_name(f'trace_{trial["repeat"]}.npz')
                        assets[str(trace)] = sha(trace)
                for name in ('identity.json', 'status.json'):
                    assets[str(output/name)] = sha(output/name)
            elif list(output.glob('cells/*/*/execution_*.json')) or list(output.glob('cells/*/*/complete.json')):
                raise ValueError('Committed trials without worker identity')
            if state['tasks'][task['key']]['status'] == 'COMPLETE':
                if backend.validate(task) != state['tasks'][task['key']]['receipt']:
                    raise ValueError('Previously complete receipt changed')
            by_view = {}
            for address in task['addresses']:
                identity = (address['view'], address['repeat'])
                if identity in existing:
                    kept.append(dict(address=address, **existing[identity]))
                else:
                    by_view.setdefault(address['view'], []).append(address)
            groups={}
            for view,addresses in by_view.items():
                repeats=tuple(a['repeat'] for a in addresses)
                if is_probe and repeats!=tuple(range(8)):raise ValueError('Probe recovery must preserve whole eight-repeat cells')
                groups.setdefault(repeats,[]).append(view)
            for repeats,views in groups.items():
                # Preserve bounded batching for entirely unstarted work.
                width=32 if is_probe else 4
                for start in range(0,len(views),width):
                    chunk=views[start:start+width];addresses=[a for v in chunk for a in by_view[v]]
                    payload=dict(initial=task['payload']['initial'],views=chunk)
                    if not is_probe:payload['repeats']=list(repeats)
                    key=task_key(experiment,manifest['protocol_sha256'],addresses)
                    tasks.append(dict(key=key,addresses=addresses,payload=payload,output=str(new_output/'workers'/key)))
        original = [a for t in manifest['tasks'] for a in t['addresses']]
        if len(kept)+sum(len(t['addresses']) for t in tasks) != len(original):
            raise ValueError('Recovery lost or added trials')
        new_output.mkdir(parents=True)
        receipt_file = new_output/'recovery-ownership.json'
        write(receipt_file, dict(status='PREPARED' if tasks else 'NO_MISSING_TRIALS',
              old_manifest=str(old_control/'manifest.json'), old_manifest_sha256=sha(old_control/'manifest.json'),
              old_state_sha256=sha(old_control/'state.json'), retained=kept,
              missing_trials=sum(len(t['addresses']) for t in tasks),
              rule='Retain every validated outcome including failure; rerun only missing complete attempts or uncommitted whole probe cells.',
              original_source=manifest['runtime']['source'], original_trial_count=len(original)))
        if not tasks:
            return read(receipt_file)
        new = deepcopy(manifest); new.update(experiment_id=experiment, tasks=tasks)
        assets.update({a['path']: a['sha256'] for a in manifest['artifacts']})
        for path in (old_control/'manifest.json', old_control/'state.json', receipt_file, Path(__file__)):
            assets[str(path.resolve())] = sha(path)
        new['artifacts'] = [dict(path=p, sha256=h) for p,h in sorted(assets.items())]
        write(new_output/'manifest.json', new); prepare(new, new_control)
        return read(receipt_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-control', required=True)
    parser.add_argument('--new-control', required=True)
    parser.add_argument('--new-output', required=True)
    args = parser.parse_args()
    prepare_missing(args.old_control, args.new_control, args.new_output)
