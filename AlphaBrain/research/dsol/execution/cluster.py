"""Read-only multi-host queue inventory and ownership audit.

Host connection details are explicit configuration, not repository constants.
No remote service is stopped and no experiment is launched by this module.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import time

from .manifest import digest


SNAPSHOT = r'''
import json,pathlib,subprocess,sys,time
roots=json.loads(sys.argv[1])
root=next((pathlib.Path(r) for r in roots if (pathlib.Path(r)/'manifest.json').exists() and (pathlib.Path(r)/'state.json').exists()),None)
if root is None:raise FileNotFoundError('No prepared queue in explicit control candidates')
plan=json.loads((root/'manifest.json').read_text())
state=json.loads((root/'state.json').read_text())
control=json.loads((root/'resource-control.json').read_text()) if (root/'resource-control.json').exists() else {}
progress=[]
for task in plan['tasks']:
    row=state['tasks'][task['key']]
    path=pathlib.Path(task['output'])/'status.json'
    status=json.loads(path.read_text()) if path.exists() else {}
    completed=len(task['addresses']) if row['status']=='COMPLETE' else status.get('completed_probes',status.get('completed_rollouts',0))
    progress.append(dict(key=task['key'],output=task['output'],addresses=task['addresses'],completed=completed,status=row['status'],slot=row.get('slot')))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(dict(control=str(root),status=state['status'],updated_unix=state['updated_unix'],sample_unix=time.time(),manifest=plan,manifest_sha256=state['manifest_sha256'],retired=(root/'retired.json').exists(),resource_control=control,tasks=progress,gpus=gpu)))
'''


def inspect_host(spec):
    roots=spec.get('controls') or [spec['control']]
    command = [spec.get('python', 'python3'), '-c', SNAPSHOT, json.dumps(roots)]
    if spec.get('ssh'):
        command = [*spec['ssh'], shlex.join(command)]
    process = subprocess.run(command, capture_output=True, text=True,
                             timeout=spec.get('timeout_seconds', 45), check=True)
    value = json.loads(process.stdout)
    if digest(value['manifest']) != value['manifest_sha256']:
        raise ValueError('Queue manifest digest mismatch')
    value['host'] = spec['name']
    return value


def audit(snapshots):
    owners = {}; outputs = {}; duplicate = []; hosts = []; alerts = []; devices = {}
    for snap in snapshots:
        host = snap['host']; plan = snap['manifest']; progress = snap['tasks']
        if snap['retired']:
            alerts.append(dict(host=host, kind='RETIRED_CONTROL', control=snap['control']))
        if snap['status'] in ('FAILED', 'INTERRUPTED'):
            alerts.append(dict(host=host, kind='QUEUE_STOPPED', status=snap['status']))
        drained = {(r['host'], r['gpu']) for r in snap['resource_control'].get('drained', [])}
        # Completed queues retain scientific ownership, but release GPU leases.
        reservations = plan['slots'] if snap['status'] != 'COMPLETE' else []
        for identity in {(s['host'],str(s.get('gpu_id',s['id']))) for s in reservations}:
            if identity in devices:
                duplicate.append(dict(kind='gpu_reservation',host=identity[0],gpu=identity[1],
                    controls=[devices[identity],snap['control']]))
            devices[identity]=snap['control']
        pending = sum(r['status']=='PENDING' for r in progress)
        active = [r for r in progress if r['status'] in ('RUNNING','LAUNCHING')]
        occupied = {(r['slot']['host'],r['slot']['id']) for r in active}
        unused = [s for s in plan['slots'] if (s['host'],s['id']) not in occupied
                  and (s['host'],str(s.get('gpu_id',s['id']))) not in drained]
        if pending and unused:
            alerts.append(dict(host=host,kind='IDLE_SLOTS_WITH_PENDING_WORK',slots=unused,
                               note='Inspect GPU admission/foreign processes before rebalancing; never force dispatch'))
        for row in progress:
            if row['output'] in outputs:
                duplicate.append(dict(kind='output',hosts=[outputs[row['output']],host],output=row['output']))
            outputs[row['output']] = host
            for address in row['addresses']:
                key = digest(dict(protocol=plan['protocol_sha256'],address=address))
                if key in owners:
                    duplicate.append(dict(kind='trial',hosts=[owners[key],host],trial_sha256=key))
                owners[key] = host
        hosts.append(dict(host=host,status=snap['status'],control=snap['control'],
            completed=sum(r['completed'] for r in progress),total=sum(len(r['addresses']) for r in progress),
            active_workers=len(active),configured_slots=len(plan['slots']),pending_blocks=pending,
            drained_gpus=sorted(g for h,g in drained if h==host),gpu_inventory=snap['gpus']))
    return dict(status='PASS' if not duplicate and not any(s['retired'] for s in snapshots) else 'HOLD',
                hosts=hosts,duplicate_ownership=duplicate,alerts=alerts,
                scope='Non-atomic cross-host read-only snapshot; revalidate under coordinator locks before handoff')


def inspect_cluster(config):
    cfg=json.loads(Path(config).read_text())
    with ThreadPoolExecutor(max_workers=min(8,len(cfg['hosts']))) as pool:
        snapshots=list(pool.map(inspect_host,cfg['hosts']))
    result=audit(snapshots);result['sample_unix']=time.time()
    return result
