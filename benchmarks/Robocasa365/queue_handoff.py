"""Explicit coordinator-only replacement; live GPU worker groups are preserved."""
from copy import deepcopy
import os
from pathlib import Path
import signal
import time

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.execution.queue import load
from AlphaBrain.research.dsol.execution.repartition import repartition
from .probe_queue import process_identity


def interleave_tasks(tasks):
    """Outcome-blind task round-robin for representative early timing."""
    from collections import defaultdict, deque
    groups = defaultdict(deque)
    for task in tasks:
        groups[task['addresses'][0]['task']].append(task)
    result = []
    while any(groups.values()):
        for name in sorted(groups):
            if groups[name]: result.append(groups[name].popleft())
    return result


def bound_inline_queue(argv, root):
    """Recognize only the historical literal CLI-main wrapper, never eval it."""
    import ast
    if '-c' not in argv: return False
    try:
        tree=ast.parse(argv[argv.index('-c')+1])
        if len(tree.body)!=2: return False
        first,second=tree.body
        if (not isinstance(first,ast.ImportFrom) or first.module!='AlphaBrain.research.dsol.execution.cli'
                or len(first.names)!=1 or first.names[0].name!='main' or first.names[0].asname): return False
        if not isinstance(second,ast.Expr) or not isinstance(second.value,ast.Call): return False
        call=second.value
        if (not isinstance(call.func,ast.Name) or call.func.id!='main'
                or len(call.args)!=1 or call.keywords): return False
        arguments=ast.literal_eval(call.args[0])
        return (isinstance(arguments,list) and arguments[0]=='run' and
                arguments.count('--root')==1 and arguments[arguments.index('--root')+1]==str(root))
    except (ValueError,TypeError,IndexError,SyntaxError):
        return False


def handoff(config_file):
    cfg=read(config_file); root=Path(cfg['old_control']); pid=cfg['coordinator_pid']
    identity=process_identity(pid)
    argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    command=' '.join(argv)
    argument=cfg.get('coordinator_control_argument',str(root))
    direct = Path(argument)==root or (Path(argument)==root.parent and root.name=='full')
    pipeline = (root.name=='full' and '--config' in argv and
                argv[argv.index('--config')+1]==argument and
                'benchmarks.Robocasa365.confirmation_pipeline' in argv and
                Path(read(argument)['control_root'])==root.parent)
    if not (direct or pipeline):
        raise ValueError('Coordinator argument must identify this control or its bound endpoint pipeline')
    if identity is None or (argument not in argv and not bound_inline_queue(argv, root)) or 'python' not in argv[0]:
        raise ValueError('Exact original coordinator command required')
    result=Path(cfg['receipt'])
    if result.exists():raise FileExistsError(result)
    killed=False; suspended=False
    try:
        for _ in range(30):
            current=process_identity(pid)
            if current is None or any(current[k]!=identity[k] for k in ('start','boot','group')):
                raise ValueError('Coordinator identity changed')
            os.kill(pid,signal.SIGSTOP);suspended=True
            for _ in range(100):
                if process_identity(pid)['state']=='T':break
                time.sleep(.01)
            else:raise RuntimeError('Coordinator did not suspend')
            plan,state=load(root)
            if not any(r['status']=='LAUNCHING' for r in state['tasks'].values()):break
            os.kill(pid,signal.SIGCONT);suspended=False;time.sleep(.1)
        else:raise RuntimeError('Cannot resolve launch transition safely')
        hosts=cfg['destinations']; old_host=plan['runtime']['host']
        if old_host not in hosts:raise ValueError('Original host must retain active work')
        for spec in hosts.values():
            if 'environment' in spec and spec['environment'] != plan['runtime']['environment']:
                raise ValueError('Runtime environment changes require separate admission, not a queue handoff')
        costs=cfg.get('seconds_per_task')
        def task_cost(task):
            if costs is None: return float(len(task['addresses']))
            import math
            values=[float(costs[a['task']]) for a in task['addresses']]
            if any(not math.isfinite(v) or v<=0 for v in values):
                raise ValueError('Positive measured/projected task costs required')
            return sum(values)
        assignments={h:[] for h in hosts}; loads={h:0 for h in hosts}; pending=[]
        for task in plan['tasks']:
            row=state['tasks'][task['key']]
            if row['status']=='PENDING' and row['attempt']==0 and not Path(task['output']).exists():
                pending.append(task)
            else:
                assignments[old_host].append(task['key'])
                if row['status']=='RUNNING':
                    status=Path(task['output'])/'status.json'
                    done=read(status).get('completed_probes',read(status).get('completed_rollouts',0)) if status.exists() else 0
                    loads[old_host]+=task_cost(task)*max(0,len(task['addresses'])-done)/len(task['addresses'])
        for task in sorted(pending,key=lambda t:-task_cost(t)):
            h=min(hosts,key=lambda h:(loads[h]/hosts[h]['capacity'],h))
            assignments[h].append(task['key']);loads[h]+=task_cost(task)
        assets=[dict(path=str(p),sha256=sha(p)) for p in sorted(Path(cfg['controller_source']).rglob('*.py'))]
        assets.append(dict(path=str(Path(config_file).resolve()),sha256=sha(config_file)))
        destinations=[]
        for host,spec in hosts.items():
            runtime=deepcopy(plan['runtime']);runtime.update(host=host,workers_per_gpu=spec['workers_per_gpu'])
            selected=set(assignments[host])
            destination=dict(root=spec['control'],runtime=runtime,slots=spec['slots'],
                controller_artifacts=list(assets),task_keys=[t['key'] for t in plan['tasks'] if t['key'] in selected],
                initial_drained_gpus=spec.get('initial_drained_gpus', []))
            if cfg.get('dispatch_order') == 'task_round_robin':
                destination['task_keys'] = [t['key'] for t in interleave_tasks(
                    [t for t in plan['tasks'] if t['key'] in selected])]
            for field in ('resource_admission', 'host_bootstrap'):
                if field in spec:
                    binding=dict(path=str(Path(spec[field]).resolve()),sha256=sha(spec[field]))
                    destination[field]=binding;destination['controller_artifacts'].append(binding)
            destinations.append(destination)
        # Preflight before removing the old controller. A slot dictionary is an
        # identity, not merely a GPU number (legacy primary slots omit gpu_id).
        for destination in destinations:
            if Path(destination['root']).exists(): raise FileExistsError(destination['root'])
            known = {str(s.get('gpu_id', s['id'])) for s in destination['slots']}
            if not set(map(str, destination['initial_drained_gpus'])) <= known:
                raise ValueError('Unknown destination drain device')
            for key in destination['task_keys']:
                row = state['tasks'][key]
                if row['status']=='RUNNING' and row['slot'] not in destination['slots']:
                    raise ValueError('Running worker cannot change host or slot')
        # Only the verified coordinator PID is stopped, never its process group.
        os.kill(pid,signal.SIGKILL);killed=True;suspended=False
        for _ in range(100):
            current=process_identity(pid)
            if current is None or current['state']=='Z':break
            time.sleep(.05)
        else:raise RuntimeError('Coordinator exit not confirmed')
        receipt=repartition(root,destinations,cfg['evidence_root'])
        receipt.update(coordinator_identity=identity,old_command=command,estimated_remaining_work=loads,
                       no_worker_signals_sent=True,config_sha256=sha(config_file),created_unix=time.time())
        write(result,receipt);return receipt
    except BaseException as exc:
        if suspended and not killed:os.kill(pid,signal.SIGCONT)
        write(result,dict(status='FAILED',error=repr(exc),coordinator_stopped=killed,no_worker_signals_sent=True))
        raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not args.execute:p.error('Explicit --execute required')
    print(handoff(args.config),flush=True)
