"""Read-only research-node monitoring when no external monitor is available.

Reuses gpu_guard's trusted supervisor verification. It never controls the
keeper, modifies a host, starts research or reconnects an interrupted task.
"""
import argparse
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import time
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.execution.queue import exclusive,_durable_write
from AlphaBrain.research.dsol.execution.node_health import NodeHealth,trusted_keeper_snapshot


def probe(config):
    script=('import json,sys\n'+inspect.getsource(trusted_keeper_snapshot)
            +'\nprint(json.dumps(trusted_keeper_snapshot(json.loads(sys.argv[1]))))\n')
    command=['env','CUDA_VISIBLE_DEVICES=','WANDB_MODE=disabled','PYTHONPATH='+config['source'],config['python'],'-c',script,json.dumps({k:config[k] for k in ('gpu_guard','keeper_gpu_uuid','shared_root','owned_root')})]
    result=subprocess.run([*config['ssh'],shlex.join(command)],capture_output=True,text=True,timeout=20)
    if result.returncode:return dict(status='FAILED',error='SSH_OR_TRUSTED_PROBE_FAILED',returncode=result.returncode)
    return json.loads(result.stdout)


def run(path,*,once=False):
    config=read(path);local=Path(config['local_state']);mirror=Path(config['persistent_state'])
    if (type(config['interval_seconds']) not in (int,float) or not 10<=config['interval_seconds']<=60
            or type(config['monitor_seconds']) not in (int,float) or not 60<=config['monitor_seconds']<=604800):
        raise ValueError('Explicit bounded research monitoring window required')
    local.parent.mkdir(parents=True,exist_ok=True);mirror.parent.mkdir(parents=True,exist_ok=True)
    # Each observer ID is a fresh explicit operation. Do not clear a prior HOLD.
    if local.exists() or mirror.exists():raise RuntimeError('Existing monitor state requires explicit audit/new observer binding')
    model=NodeHealth(config['expected'],config['observer_id'])
    with exclusive(local.parent):
        deadline=time.monotonic()+config['monitor_seconds'];previous=None
        while True:
            try:snapshot=probe(config)
            except (OSError,ValueError,subprocess.TimeoutExpired):snapshot=dict(status='FAILED',error='PROBE_UNAVAILABLE')
            state=model.observe(snapshot)
            _durable_write(local,state);_durable_write(mirror,state)
            summary=(state['status'],state['reason'],snapshot.get('supervisor_state'))
            if summary!=previous:
                with (mirror.parent/'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(state,sort_keys=True)+'\n');stream.flush()
                previous=summary
            if once or time.monotonic()>=deadline:break
            time.sleep(config['interval_seconds'])
        if not once:
            state=model.observe(dict(status='FAILED',error='MONITOR_WINDOW_ENDED'))
            _durable_write(local,state);_durable_write(mirror,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--once',action='store_true');args=parser.parse_args();run(args.config,once=args.once)
