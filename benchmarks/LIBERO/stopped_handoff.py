"""Finish an interrupted coordinator handoff without signalling live workers.

Only accepts the recorded runtime-equality rejection before any destinations
were published. Scientific worker runtime remains byte-for-byte identical.
"""
from copy import deepcopy
from pathlib import Path
from AlphaBrain.research.dsol.data.artifacts import read,sha,write
from AlphaBrain.research.dsol.execution.queue import load
from AlphaBrain.research.dsol.execution.repartition import repartition
from benchmarks.Robocasa365.probe_queue import process_identity


def resume(config_file):
    cfg=read(config_file); failure=read(cfg['failed_receipt']); original=read(cfg['failed_config'])
    if not failure.get('coordinator_stopped') or not failure.get('no_worker_signals_sent'):
        raise ValueError('No proven coordinator-only stopped transfer')
    if process_identity(original['coordinator_pid']) is not None:raise ValueError('Old PID still exists')
    root=Path(original['old_control']); plan,state=load(root)
    if (root/'retired.json').exists():raise ValueError('Transfer already published')
    for name in ('manifest.json','state.json'):
        if sha(root/name)!=sha(Path(original['evidence_root'])/name):raise ValueError('Stopped ledger changed')
    assignments={h:[] for h in original['destinations']};load_cost={h:0 for h in assignments};pending=[]
    oldhost=plan['runtime']['host']
    for t in plan['tasks']:
        row=state['tasks'][t['key']]
        if row['status']=='PENDING' and row['attempt']==0 and not Path(t['output']).exists():pending.append(t)
        else:
            assignments[oldhost].append(t['key'])
            if row['status']=='RUNNING':load_cost[oldhost]+=len(t['addresses'])
    for t in sorted(pending,key=lambda x:-len(x['addresses'])):
        h=min(assignments,key=lambda h:load_cost[h]/original['destinations'][h]['capacity'])
        assignments[h].append(t['key']);load_cost[h]+=len(t['addresses'])
    assets=[dict(path=str(p),sha256=sha(p)) for p in (Path(config_file),Path(cfg['failed_config']),Path(cfg['failed_receipt']),Path(__file__))]
    destinations=[]
    for host,spec in original['destinations'].items():
        runtime=deepcopy(plan['runtime']);runtime.update(host=host,workers_per_gpu=spec['workers_per_gpu'])
        proof=Path(spec['resource_admission']);binding=dict(path=str(proof),sha256=sha(proof))
        destinations.append(dict(root=spec['control'],runtime=runtime,slots=spec['slots'],
            task_keys=assignments[host],resource_admission=binding,controller_artifacts=assets+[binding]))
    result=repartition(root,destinations,cfg['evidence_root'])
    result.update(status='PREPARED',no_worker_signals_sent=True,config_sha256=sha(config_file),
        note='No runtime environment changes; GNU controller inherits explicitly recorded loader path')
    write(cfg['receipt'],result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('Explicit --execute required')
    print(resume(a.config),flush=True)
