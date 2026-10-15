"""Explicit per-model handoff from CPU preparation to admitted endpoint execution."""
from pathlib import Path
import time

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.confirmation_execution import publish_selections, publish_host_partition
from AlphaBrain.research.dsol.data.recovery_confirmation import prepare_execution_plan
from .confirmation_queue import run_program


def after_preparation(config_file):
    cfg=read(config_file); status_path=Path(cfg['status_file'])
    if status_path.exists():raise FileExistsError(status_path)
    started=time.time(); identity=sha(config_file)
    state=dict(status='WAITING_FOR_CPU_PREPARATION',started_unix=started,config_sha256=identity)
    write(status_path,state)
    try:
        while True:
            previous=read(cfg['preparation_status'])
            if previous['status']=='READY_FOR_ENDPOINT_RUNTIME_ADMISSION':break
            if previous['status']=='FAILED':raise RuntimeError('CPU preparation failed; no endpoint launch')
            if time.time()-started>cfg['timeout_seconds']:raise TimeoutError('Preparation wait budget expired')
            state.update(checked_unix=time.time(),preparation_status=previous['status']);write(status_path,state)
            time.sleep(min(60,cfg.get('poll_seconds',60)))
        if sha(config_file)!=identity:raise ValueError('Handoff configuration changed')
        runtime=read(cfg['runtime']); model=runtime['model']
        selected=publish_selections(cfg['selection_source'],cfg['selection_destination'])
        plan=prepare_execution_plan(cfg['release'],cfg['probe_config'],{model:selected},runtime['plan'],
                                    model_partition=model,path_mappings=cfg.get('path_mappings',()))
        if 'host_capacities' in cfg:
            publish_host_partition(runtime['plan'], runtime['host_partition'], cfg['host_capacities'])
            write(cfg['publication_ready'], dict(status='READY', plan_sha256=sha(runtime['plan']),
                  partition_sha256=sha(runtime['host_partition']), config_sha256=identity))
        state.update(status='ENDPOINT_PROGRAM',rollouts=plan['rollout_count'],handoff_unix=time.time())
        write(status_path,state)
        run_program(cfg['runtime'],cfg['control_root'])
        state.update(status='COMPLETE',finished_unix=time.time());write(status_path,state)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc),finished_unix=time.time());write(status_path,state);raise


def after_publication(config_file):
    """A second host waits for the single publisher; no re-selection or duplicate plan."""
    cfg = read(config_file); identity = sha(config_file); started = time.time()
    status_file = Path(cfg['status_file'])
    if status_file.exists(): raise FileExistsError(status_file)
    state = dict(status='WAITING_FOR_FROZEN_PARTITION', config_sha256=identity, started_unix=started)
    write(status_file, state)
    try:
        while not Path(cfg['publication_ready']).exists():
            if Path(cfg['publisher_status']).exists() and read(cfg['publisher_status'])['status'] == 'FAILED':
                raise RuntimeError('Partition publisher failed')
            if time.time()-started > cfg['timeout_seconds']: raise TimeoutError('Publication timeout')
            state['checked_unix'] = time.time(); write(status_file, state)
            time.sleep(min(60, cfg.get('poll_seconds', 60)))
        receipt = read(cfg['publication_ready']); runtime = read(cfg['runtime'])
        if (sha(config_file) != identity or receipt['status'] != 'READY'
                or receipt['plan_sha256'] != sha(runtime['plan'])
                or receipt['partition_sha256'] != sha(runtime['host_partition'])):
            raise ValueError('Partition publication binding mismatch')
        state['status'] = 'ENDPOINT_PROGRAM'; write(status_file, state)
        run_program(cfg['runtime'], cfg['control_root'])
        state.update(status='COMPLETE', finished_unix=time.time()); write(status_file, state)
    except BaseException as exc:
        state.update(status='FAILED', error=repr(exc), finished_unix=time.time()); write(status_file, state); raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--wait-publication', action='store_true')
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required')
    (after_publication if args.wait_publication else after_preparation)(args.config)
