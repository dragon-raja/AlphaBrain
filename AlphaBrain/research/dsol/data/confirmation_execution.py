"""Immutable endpoint publication and bounded standard-queue ownership; CPU only."""
from pathlib import Path
import shutil

from .artifacts import read, sha, write
from ..execution.manifest import task_key
from ..execution.queue import prepare


def publish_host_partition(plan_file, destination, capacities):
    """Disjoint initial ownership, balanced by frozen horizon, never outcomes."""
    if Path(destination).exists():
        raise FileExistsError(destination)
    plan = read(plan_file); model = plan['model_partition']
    if not capacities or any(not isinstance(v, (int, float)) or v <= 0 for v in capacities.values()):
        raise ValueError('Positive host capacities required')
    costs = {}
    for initial in plan['initial_ids']:
        entry = plan['selections'][model][initial]
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Frozen selection changed')
        choice = read(entry['path'])
        costs[initial] = len(choice['evaluation_views']) * 8 * plan['task_horizons'][choice['task']]
    hosts = {h: [] for h in sorted(capacities)}; loads = dict.fromkeys(hosts, 0)
    for initial in sorted(costs, key=lambda i: (-costs[i], i)):
        host = min(hosts, key=lambda h: (loads[h] / capacities[h], h))
        hosts[host].append(initial); loads[host] += costs[initial]
    if any(not ids for ids in hosts.values()):
        raise ValueError('Every host must own work')
    order = {i: n for n, i in enumerate(plan['initial_ids'])}
    for ids in hosts.values(): ids.sort(key=order.__getitem__)
    receipt = dict(status='FROZEN', model=model, plan_sha256=sha(plan_file), hosts=hosts,
                   capacities=capacities, nominal_step_loads=loads,
                   allocation_rule='LPT frozen views times repeats times native horizon; no success labels')
    write(destination, receipt)
    return receipt


def assigned_initials(runtime, plan):
    """Validate global complete/disjoint partition before using this host's slice."""
    if 'host_partition' not in runtime:
        return plan['initial_ids']
    partition = read(runtime['host_partition'])
    hosts = partition['hosts']; flattened = [i for ids in hosts.values() for i in ids]
    if (partition['status'] != 'FROZEN' or partition['model'] != runtime['model']
            or partition['plan_sha256'] != sha(runtime['plan'])
            or len(flattened) != len(set(flattened)) or set(flattened) != set(plan['initial_ids'])
            or runtime['host'] not in hosts or any(not ids for ids in hosts.values())):
        raise ValueError('Host partition is incomplete, overlapping, or bound to another plan')
    return hosts[runtime['host']]


def publish_selections(source, destination):
    """Copy frozen choice bytes; relocate only the index, never the selected views."""
    source, destination = Path(source), Path(destination)
    status = read(source/'status.json')
    if status['status'] != 'COMPLETE' or status['completed_initials'] != 80:
        raise ValueError('All 80 selections must be complete')
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    shutil.copy2(source/'binding.json', destination/'binding.json')
    records = []
    for entry in status['records']:
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Frozen selection changed')
        target = destination/'initials'/Path(entry['path']).name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(entry['path'], target)
        records.append(dict(entry, path=str(target)))
    write(destination/'status.json', dict(status, records=records,
          publication_source=str(source), publication_source_sha256=sha(source/'status.json')))
    return destination


def build_queue(runtime_file, queue_root, *, pilot=False):
    """Pilot is the first initial's canonical/extra_002, and counts toward the budget."""
    runtime_file = Path(runtime_file).resolve(); runtime = read(runtime_file)
    plan_file = Path(runtime['plan']); plan = read(plan_file); model = runtime['model']
    if (plan.get('model_partition') != model or set(plan['selections']) != {model}
            or len(plan['initial_ids']) != 80 or len(set(plan['initial_ids'])) != 80
            or plan['execution_repeats'] != list(range(8)) or plan['max_rollouts'] != 24320):
        raise ValueError('Explicit complete single-model partition required')
    result = Path(runtime['result_root']); result.mkdir(parents=True, exist_ok=True)
    manifest_file = result/('pilot-manifest.json' if pilot else 'manifest.json')
    if manifest_file.exists():
        raise FileExistsError(manifest_file)
    owned = assigned_initials(runtime, plan)
    first = owned[0]; pilot_views = ['canonical', 'extra_002']
    reused = {(first, v) for v in pilot_views}
    inputs = {runtime_file, plan_file, Path(plan['authorization'])}
    if 'host_partition' in runtime:
        inputs.add(Path(runtime['host_partition']))
    for path, expected in plan['selection_sources'].items():
        if sha(path) != expected:
            raise ValueError('Selection publication changed')
        inputs.add(Path(path))
    if not pilot:
        admission = read(runtime['admission'])
        if (admission['status'] != 'PASS' or admission['model'] != model
                or admission['plan_sha256'] != sha(plan_file)
                or not admission['complete_task_attempts_verified']
                or admission['pilot_initial'] != first or admission['pilot_views'] != pilot_views
                or admission['pilot_repeats'] != list(range(8))):
            raise ValueError('Exact pilot admission required')
        if 'host_partition' in runtime and (admission.get('host') != runtime['host']
                or admission.get('host_partition_sha256') != sha(runtime['host_partition'])):
            raise ValueError('Host-specific pilot admission required')
        inputs.add(Path(runtime['admission']))
    stage = 'pilot' if pilot else 'full'
    name = 'recovery-confirmation-execution-'+model+'-'+stage
    fingerprint = sha(plan_file); tasks = []; ownership = []; expected_count = 0; owned_count = 0
    for initial in plan['initial_ids']:
        entry = plan['selections'][model][initial]
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Choice file changed')
        selection = read(entry['path']); inputs.add(Path(entry['path']))
        views = selection['evaluation_views']; expected_count += len(views)*8
        if len(set(views)) != len(views) or selection['execution_repeats'] != list(range(8)):
            raise ValueError('Duplicate choices or changed noise budget')
        if initial not in owned:
            continue
        owned_count += len(views)*8
        if initial == first and not set(pilot_views) <= set(views):
            raise ValueError('Pilot views missing from frozen evaluation')
        for view in views:
            base = result/'pilot' if (initial, view) in reused else None
            if base is not None:
                ownership.append(dict(initial=initial, view=view, cell=str(base/'cells'/initial/view)))
        selected = (pilot_views if initial == first else []) if pilot else [v for v in views if (initial, v) not in reused]
        for start in range(0, len(selected), 4):
            chunk = selected[start:start+4]
            output = result/'pilot' if pilot else result/'workers'/f'{initial}-{start//4:02d}'
            addresses = [dict(model=model, task=selection['task'], initial=initial, view=v,
                         stream='execution', repeat=r, noise=f'execution/{initial}/{r}/all-native-replans')
                         for v in chunk for r in range(8)]
            tasks.append(dict(key=task_key(name, fingerprint, addresses), addresses=addresses,
                              output=str(output), payload=dict(initial=initial, views=chunk, repeats=list(range(8)))))
            if not pilot:
                ownership.extend(dict(initial=initial, view=v, cell=str(output/'cells'/initial/v)) for v in chunk)
    if expected_count != plan['rollout_count'] or expected_count > 24320:
        raise ValueError('Frozen endpoint budget mismatch')
    if not pilot:
        if len(ownership)*8 != owned_count or len({(r['initial'], r['view']) for r in ownership}) != len(ownership):
            raise ValueError('Endpoint ownership incomplete or duplicated')
        write(result/'ownership.json', dict(records=ownership, repeats=list(range(8)),
              total_rollouts=owned_count, global_rollouts=expected_count, host=runtime['host'],
              reused_pilot_rollouts=16, model=model, plan_sha256=fingerprint))
        inputs.add(result/'ownership.json')
    inputs.update(p for p in Path(runtime['source']).rglob('*.py') if '__pycache__' not in p.parts)
    workers_per_gpu=runtime.get('workers_per_gpu',1)
    if workers_per_gpu not in (1,2):raise ValueError('Only admitted one/two-worker GPU layouts supported')
    slots=[dict(host=runtime['host'],id=str(i)) for i in (runtime['gpu_ids'][:1] if pilot else runtime['gpu_ids'])]
    if not pilot and workers_per_gpu==2:
        slots += [dict(host=runtime['host'],id=f'{i}:1',gpu_id=str(i)) for i in runtime['gpu_ids']]
    manifest = dict(format='dsol_work_queue_v1', experiment_id=name, protocol_sha256=fingerprint,
        slots=slots,
        tasks=tasks, artifacts=[dict(path=str(p), sha256=sha(p)) for p in sorted(inputs)], runtime=runtime,
        backend=dict(module='benchmarks.Robocasa365.confirmation_queue', factory='pilot_factory' if pilot else 'factory'))
    write(manifest_file, manifest); prepare(manifest, queue_root)
    return manifest
