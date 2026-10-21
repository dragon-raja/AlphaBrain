"""Coordinate admission phases through existing host-local durable queues.

All hosts share an explicit fail-stop file. Scientific task scheduling and
process ownership remain in execution.queue and its process backend.
"""
import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time

from AlphaBrain.research.dsol.data.artifacts import read, write
from AlphaBrain.research.dsol.execution.queue import prepare, run
from benchmarks.Robocasa365.probe_queue import process_identity
from .search_queue import SearchBackend


def first_stop(path, reason):
    """Keep the originating failure, not a later cancellation exception."""
    import json
    try:
        with Path(path).open('x') as stream:
            json.dump(reason,stream,ensure_ascii=False,indent=2)
    except FileExistsError:
        pass


def local(manifest, control, shared):
    shared=Path(shared); plan=read(manifest)
    prepare(plan,control)
    write(shared/'coordinator.json',process_identity(os.getpid()))
    stopping=threading.Event()
    def interrupted(signum, frame):
        raise InterruptedError('Fleet stop or coordinator interruption')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGHUP,interrupted)
    def watch():
        while not stopping.wait(.5):
            if Path(str(control)+'.stop').exists() or (shared.parent/'STOP.json').exists():
                os.kill(os.getpid(),signal.SIGTERM)
                return
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    try:
        if Path(str(control)+'.stop').exists():raise RuntimeError('Fleet stopped before dispatch')
        backend=SearchBackend(plan,control)
        # Preflight every physical card before launching the first model. In a
        # synchronized capacity phase an unavailable slot cannot be skipped.
        for slot in plan['slots']:
            if not backend.admit(slot):
                raise RuntimeError('Capacity resource preflight refused slot: '+str(slot))
        run(control,backend,timeout_seconds=plan['runtime'].get('queue_timeout_seconds',1800))
        write(shared/'queue-result.json',dict(status='COMPLETE'))
    except BaseException as error:
        write(shared/'queue-result.json',dict(status='FAILED_STOPPED',error=repr(error)))
        first_stop(shared.parent/'STOP.json',dict(error=repr(error),host=plan['runtime']['host']))
        raise
    finally:
        stopping.set()
        for name in ('state.json','manifest.json'):
            target=Path(control)/name
            if target.exists():
                write(shared/name,read(target))


def launch_command(host, phase):
    command=['env', *[k+'='+v for k,v in host['environment'].items()],host['python'],
        '-u','-m','benchmarks.LIBERO.search_fleet','--local',host['manifest'],
        '--control',host['control'],'--shared',str(Path(phase)/host['name']),'--execute']
    if host.get('ssh'):
        return host['ssh']+[shlex.join(command)]
    return command


def stop_hosts(config):
    """Notify only this fleet's local controls; no signals to external processes.

Shared-filesystem negative lookup caching can delay STOP visibility. The SSH
channel already used for launch also writes the unique local stop marker.
"""
    errors=[]
    for host in config['hosts']:
        code='from pathlib import Path; import sys; Path(sys.argv[1]).write_text("fleet stopped\\n")'
        command=[host['python'],'-c',code,host['control']+'.stop']
        if host.get('ssh'):command=host['ssh']+[shlex.join(command)]
        try:subprocess.run(command,capture_output=True,text=True,timeout=15,check=True)
        except (OSError,subprocess.SubprocessError) as error:errors.append(dict(host=host['name'],error=repr(error)))
    return errors


def capacity(config_path, *, execute=False):
    if not execute:
        raise ValueError('Explicit --execute required')
    config=read(config_path);phase=Path(config['phase'])
    if (phase/'started.json').exists():
        raise FileExistsError('No automatic capacity retries')
    write(phase/'started.json',dict(time=time.time()))
    processes=[];logs=[];begin=time.monotonic();timed=False
    try:
        for host in config['hosts']:
            shared=phase/host['name'];shared.mkdir(exist_ok=True)
            log=(shared/'coordinator.log').open('xb');logs.append(log)
            processes.append(subprocess.Popen(launch_command(host,phase),stdout=log,stderr=subprocess.STDOUT))
        outputs=[Path(t['output']) for h in config['hosts'] for t in read(h['manifest'])['tasks']]
        while True:
            if (phase/'STOP.json').exists():
                raise RuntimeError('A host stopped; see STOP.json and worker logs')
            if time.monotonic()-begin>1200:
                raise TimeoutError('Fleet capacity deadline exceeded')
            for process in processes:
                if process.poll() not in (None,0):
                    raise RuntimeError('Capacity coordinator exited unsuccessfully')
            # Read memory on every participating physical card, including keepalive.
            samples=[]
            for host in config['hosts']:
                args=['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']
                command=host.get('ssh',[])+([shlex.join(args)] if host.get('ssh') else args)
                result=subprocess.run(command,capture_output=True,text=True,timeout=15,check=True)
                rows=[[int(v.strip()) for v in line.split(',')] for line in result.stdout.splitlines()]
                if len(rows)!=8 or any(row[2]<2048 for row in rows):
                    raise RuntimeError('Capacity GPU memory reserve below 2 GiB or missing device')
                samples.append(dict(host=host['name'],gpus=rows))
            with (phase/'memory.jsonl').open('a') as stream:
                import json
                stream.write(json.dumps(dict(time=time.time(),hosts=samples))+'\n')
            if not timed and all((p/'ready.json').exists() for p in outputs):
                write(phase/'GO.json',dict(time=time.time()));timed=True
            if all(p.poll()==0 for p in processes):
                break
            time.sleep(1)
        statuses=[read(p/'status.json') for p in outputs]
        if not timed or not all(r['status']=='COMPLETE' and r['exact'] for r in statuses):
            raise ValueError('Capacity coverage or exactness failed')
        times=[read(p/'timing.json') for p in outputs]
        span=max(r['finished'] for r in times)-min(r['started'] for r in times)
        result=dict(status='PASS',workers=len(outputs),calls=sum(r['calls'] for r in statuses),
            query_throughput=len(times)/span,timed_seconds=span,elapsed_seconds=time.monotonic()-begin,
            scope='one synchronized real-render query per warmed worker; not full-task throughput')
        write(phase/'receipt.json',result)
        return result
    except BaseException as error:
        first_stop(phase/'STOP.json',dict(error=repr(error),time=time.time()))
        write(phase/'stop-delivery.json',dict(errors=stop_hosts(config)))
        # Host watchers interrupt only their owned queues and reap their workers.
        for process in processes:
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                pass  # Preserve unresolved process identity; never silently reissue.
        write(phase/'failure.json',dict(status='FAILED_STOPPED',error=repr(error)))
        raise
    finally:
        for log in logs:
            log.close()


def evaluation(config_path, *, execute=False):
    """Supervise independent standard queues; fail-stop without automatic retry."""
    if not execute:raise ValueError('Explicit --execute required')
    config=read(config_path);phase=Path(config['phase'])
    if config.get('mode')!='full_tasks':raise ValueError('Explicit full-task fleet required')
    if (phase/'started.json').exists():raise FileExistsError('No automatic evaluation retries')
    write(phase/'started.json',dict(time=time.time()))
    processes=[];logs=[];begin=time.monotonic()
    try:
        for host in config['hosts']:
            shared=phase/host['name'];shared.mkdir(exist_ok=True)
            log=(shared/'coordinator.log').open('xb');logs.append(log)
            processes.append(subprocess.Popen(launch_command(host,phase),stdout=log,stderr=subprocess.STDOUT))
        outputs=[Path(t['output']) for h in config['hosts'] for t in read(h['manifest'])['tasks']]
        while True:
            if (phase/'STOP.json').exists() or any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError('An evaluation host stopped; see original STOP/worker evidence')
            if time.monotonic()-begin>config['timeout_seconds']:
                raise TimeoutError('Evaluation fleet deadline exceeded')
            complete=[read(p/'status.json') for p in outputs if (p/'status.json').exists()]
            if any(v['status']!='COMPLETE' for v in complete):raise ValueError('Invalid block receipt')
            write(phase/'progress.json',dict(completed_blocks=len(complete),total_blocks=len(outputs),
                completed_tasks=sum(len(v['tasks']) for v in complete),elapsed_seconds=time.monotonic()-begin))
            if all(p.poll()==0 for p in processes):break
            time.sleep(5)
        if len(complete)!=len(outputs):raise ValueError('Missing evaluation block')
        write(phase/'receipt.json',dict(status='COMPLETE',blocks=len(complete),
            tasks=sum(len(v['tasks']) for v in complete),elapsed_seconds=time.monotonic()-begin))
    except BaseException as error:
        first_stop(phase/'STOP.json',dict(error=repr(error),time=time.time()))
        write(phase/'stop-delivery.json',dict(errors=stop_hosts(config)))
        for process in processes:
            try:process.wait(timeout=45)
            except subprocess.TimeoutExpired:pass
        write(phase/'failure.json',dict(status='FAILED_STOPPED',error=repr(error)))
        raise
    finally:
        for log in logs:log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config');parser.add_argument('--local');parser.add_argument('--control');parser.add_argument('--shared')
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    if args.local:
        local(args.local,args.control,args.shared)
    else:
        if read(args.config).get('mode')=='full_tasks':evaluation(args.config,execute=True)
        else:capacity(args.config,execute=True)
